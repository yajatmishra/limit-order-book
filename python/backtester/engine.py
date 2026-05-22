"""
Backtester Engine
=================
Event-driven backtester with three interchangeable data sources:

  1. ShmDataSource    – reads live LOB snapshots from the POSIX shared-memory
                        region written by the C++ ShmWriter (seqlock-safe).
  2. ItchReplayer     – pure-Python ITCH 5.0 parser that reconstructs a LOB
                        and emits LOBSnapshot events; enables full end-to-end
                        replay of raw NASDAQ pcap/ITCH files with no C++ process.
  3. SnapshotSource   – plays back a pre-recorded list of LOBSnapshot objects
                        (used for unit tests and parameter sweeps).

Architecture
------------
  DataSource ──► LOBSnapshot stream
                     │
                     ▼
             BacktestEngine.run()
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
  strategy.on_snapshot()    portfolio.mark()
        │
        ▼
  list[Order] ──► FillSimulator ──► portfolio.fill()

SHM layout (from shm_writer.hpp)
---------------------------------
  ShmHeader (64 B, cache-line aligned):
    magic        uint64   8 B
    version      uint32   4 B
    _pad0        uint32   4 B
    seqlock      uint64   8 B  (atomic, even=stable, odd=writing)
    _pad1               40 B
  timestamp_ns   uint64   8 B
  bid_count      uint32   4 B
  ask_count      uint32   4 B
  bids[10]       DepthLevel×10  (price uint64, qty uint64, n_orders uint64) × 10 = 240 B
  asks[10]       DepthLevel×10                                                    = 240 B
  ── total ──────────────────────────────────────────── 560 B

ITCH 5.0 framing (MoldUDP64)
-----------------------------
  Each message:  2-byte big-endian length | 1-byte type | body

References
----------
  NASDAQ TotalView-ITCH 5.0 specification (Feb 2014).
  Pole, West & Harrison (1994). "Applied Bayesian Forecasting."
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DepthLevel:
    """One price level in the order book."""
    price:       float   # dollars (converted from integer ticks / 10000)
    quantity:    int
    order_count: int

    def __repr__(self) -> str:
        return f"DepthLevel(px={self.price:.4f}, qty={self.quantity}, n={self.order_count})"


@dataclass
class LOBSnapshot:
    """Point-in-time order-book state snapshot emitted by a DataSource."""
    symbol:       str
    timestamp_ns: int
    bids:         List[DepthLevel]   # depth levels, best first
    asks:         List[DepthLevel]   # depth levels, best first
    seq:          int = 0            # sequence number within the replay

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return (b + a) / 2.0 if (b is not None and a is not None) else None

    @property
    def spread(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return (a - b) if (b is not None and a is not None) else None

    @property
    def spread_bps(self) -> Optional[float]:
        m = self.mid
        s = self.spread
        return s / m * 1e4 if (m and s is not None and m > 0) else None

    @property
    def bid_depth_qty(self) -> int:
        return sum(d.quantity for d in self.bids)

    @property
    def ask_depth_qty(self) -> int:
        return sum(d.quantity for d in self.asks)

    def __repr__(self) -> str:
        return (f"LOBSnapshot({self.symbol} @{self.timestamp_ns} "
                f"bid={self.best_bid} ask={self.best_ask})")


@dataclass
class Order:
    """Order submitted by a strategy."""
    symbol:    str
    quantity:  float    # signed: + buy, − sell
    order_type: str     = "market"   # "market" or "limit"
    limit_price: Optional[float] = None

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0 else "sell"


@dataclass
class EngineResult:
    """Aggregate output of BacktestEngine.run()."""
    snapshots_processed: int
    fills:               list      # list of Fill from portfolio
    equity_curve:        np.ndarray
    returns:             np.ndarray
    timestamps:          List[int]  # timestamp_ns per snapshot
    symbol:              str


# ═══════════════════════════════════════════════════════════════════════════════
# Data sources
# ═══════════════════════════════════════════════════════════════════════════════

class DataSource(ABC):
    """Abstract base for all snapshot providers."""

    @abstractmethod
    def snapshots(self) -> Iterator[LOBSnapshot]:
        """Yield LOBSnapshot objects in chronological order."""
        ...

    def close(self) -> None:
        pass


# ── SHM constants from shm_writer.hpp ────────────────────────────────────────

_SHM_MAGIC        = 0x5349474D45444745   # b"SIGMEDGE"
_SHM_HEADER_FMT   = "<QIIq40x"           # magic, version, pad0, seqlock(int64), pad1[40]
_SHM_HEADER_SIZE  = struct.calcsize(_SHM_HEADER_FMT)   # should be 64
_DEPTH_LEVEL_FMT  = "<QQQ"              # price, quantity, order_count (uint64 each)
_DEPTH_LEVEL_SIZE = struct.calcsize(_DEPTH_LEVEL_FMT)   # 24
_MAX_DEPTH        = 10
_SNAP_BODY_FMT    = "<QII"              # timestamp_ns, bid_count, ask_count
_SNAP_BODY_SIZE   = struct.calcsize(_SNAP_BODY_FMT)     # 16

assert _SHM_HEADER_SIZE == 64, f"ShmHeader size mismatch: {_SHM_HEADER_SIZE}"


class ShmDataSource(DataSource):
    """
    Reads LOB snapshots from the POSIX shared-memory region written by the
    C++ ShmWriter.  Uses the seqlock protocol to guarantee consistency.

    Parameters
    ----------
    shm_name : POSIX shm name (e.g. "/lob").
    symbol   : instrument ticker label for the snapshot stream.
    max_snapshots : stop after this many consistent reads (0 = run forever).
    poll_interval_us : microseconds to sleep between polls (default 1000 = 1 ms).
    """

    def __init__(
        self,
        shm_name:          str,
        symbol:            str   = "UNKNOWN",
        max_snapshots:     int   = 0,
        poll_interval_us:  int   = 1_000,
    ) -> None:
        self.shm_name         = shm_name
        self.symbol           = symbol
        self.max_snapshots    = max_snapshots
        self.poll_interval_us = poll_interval_us
        self._mm: Optional[mmap.mmap] = None
        self._fd: int = -1

    def _open(self) -> None:
        try:
            self._fd = os.open(f"/dev/shm{self.shm_name}", os.O_RDONLY)
        except FileNotFoundError:
            # Try the standard POSIX path via shm_open equivalent
            shm_path = f"/dev/shm/{self.shm_name.lstrip('/')}"
            self._fd = os.open(shm_path, os.O_RDONLY)
        region_size = 64 + 16 + _MAX_DEPTH * _DEPTH_LEVEL_SIZE * 2
        self._mm = mmap.mmap(self._fd, region_size, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def snapshots(self) -> Iterator[LOBSnapshot]:
        self._open()
        count = 0
        seq_num = 0
        last_seq = -1

        while self.max_snapshots == 0 or count < self.max_snapshots:
            snap = self._read_once()
            if snap is None:
                time.sleep(self.poll_interval_us / 1e6)
                continue
            ts = snap["timestamp_ns"]
            if ts == last_seq:   # no new data
                time.sleep(self.poll_interval_us / 1e6)
                continue
            last_seq = ts
            yield LOBSnapshot(
                symbol       = self.symbol,
                timestamp_ns = ts,
                bids         = snap["bids"],
                asks         = snap["asks"],
                seq          = seq_num,
            )
            seq_num += 1
            count   += 1

    def _read_once(self) -> Optional[dict]:
        """Seqlock-safe read of the ShmLayout region."""
        mm = self._mm
        for _ in range(1000):   # spin limit
            mm.seek(0)
            raw_hdr = mm.read(_SHM_HEADER_SIZE)
            magic, version, _pad0, seqlock0 = struct.unpack_from(_SHM_HEADER_FMT, raw_hdr)

            if magic != _SHM_MAGIC:
                return None
            if seqlock0 & 1:      # write in progress
                continue

            # Read body
            mm.seek(_SHM_HEADER_SIZE)
            raw_body = mm.read(_SNAP_BODY_SIZE + _MAX_DEPTH * _DEPTH_LEVEL_SIZE * 2)

            # Re-read seqlock
            mm.seek(16)           # offset of seqlock in header
            seqlock1 = struct.unpack_from("<q", mm.read(8))[0]

            if seqlock0 != seqlock1:
                continue          # torn read; retry

            ts_ns, bid_cnt, ask_cnt = struct.unpack_from(_SNAP_BODY_FMT, raw_body, 0)
            offset = _SNAP_BODY_SIZE

            bids = []
            for _ in range(min(int(bid_cnt), _MAX_DEPTH)):
                px, qty, n = struct.unpack_from(_DEPTH_LEVEL_FMT, raw_body, offset)
                bids.append(DepthLevel(price=px / 10_000.0, quantity=qty,
                                        order_count=n))
                offset += _DEPTH_LEVEL_SIZE

            offset = _SNAP_BODY_SIZE + _MAX_DEPTH * _DEPTH_LEVEL_SIZE
            asks = []
            for _ in range(min(int(ask_cnt), _MAX_DEPTH)):
                px, qty, n = struct.unpack_from(_DEPTH_LEVEL_FMT, raw_body, offset)
                asks.append(DepthLevel(price=px / 10_000.0, quantity=qty,
                                        order_count=n))
                offset += _DEPTH_LEVEL_SIZE

            return {"timestamp_ns": ts_ns, "bids": bids, "asks": asks}

        return None   # too many retries


# ── Snapshot list source (testing / parameter sweeps) ─────────────────────────

class SnapshotSource(DataSource):
    """Replay a pre-built list of LOBSnapshot objects."""

    def __init__(self, snapshots: List[LOBSnapshot]) -> None:
        self._snaps = snapshots

    def snapshots(self) -> Iterator[LOBSnapshot]:
        yield from self._snaps


# ── Python ITCH 5.0 LOB + replayer ────────────────────────────────────────────

class _PythonLOB:
    """
    Minimal Python limit order book for ITCH replay.
    Prices stored as integer ticks (ITCH uint32 directly — increments of $0.0001).
    """

    def __init__(self) -> None:
        # price → total_qty, price → order_count
        self._bids: Dict[int, int] = {}   # price → qty
        self._asks: Dict[int, int] = {}
        self._bid_oc: Dict[int, int] = defaultdict(int)
        self._ask_oc: Dict[int, int] = defaultdict(int)
        self._orders: Dict[int, Tuple[int, int, str]] = {}  # ref → (price, qty, side)

    def add(self, ref: int, price: int, qty: int, side: str) -> None:
        book, oc = (self._bids, self._bid_oc) if side == 'B' else (self._asks, self._ask_oc)
        self._orders[ref] = (price, qty, side)
        book[price] = book.get(price, 0) + qty
        oc[price]  += 1

    def _remove(self, ref: int, qty: Optional[int] = None) -> None:
        if ref not in self._orders:
            return
        price, orig_qty, side = self._orders[ref]
        remove_qty = qty if qty is not None else orig_qty
        book, oc = (self._bids, self._bid_oc) if side == 'B' else (self._asks, self._ask_oc)
        if price in book:
            book[price] = max(0, book[price] - remove_qty)
            if book[price] == 0:
                del book[price]
                oc.pop(price, None)
            else:
                oc[price] = max(0, oc.get(price, 1) - (1 if remove_qty >= orig_qty else 0))
        if qty is None or remove_qty >= orig_qty:
            del self._orders[ref]
        else:
            self._orders[ref] = (price, orig_qty - remove_qty, side)

    def delete(self, ref: int) -> None:
        self._remove(ref)

    def execute(self, ref: int, qty: int) -> None:
        self._remove(ref, qty)

    def cancel(self, ref: int, cancel_qty: int) -> None:
        self._remove(ref, cancel_qty)

    def replace(self, old_ref: int, new_ref: int, new_price: int, new_qty: int) -> None:
        if old_ref in self._orders:
            _, _, side = self._orders[old_ref]
            self.delete(old_ref)
            self.add(new_ref, new_price, new_qty, side)

    def depth(self, n: int = 10) -> Tuple[List[DepthLevel], List[DepthLevel]]:
        bids = sorted(self._bids.items(), reverse=True)[:n]
        asks = sorted(self._asks.items())[:n]
        return (
            [DepthLevel(price=p/10_000.0, quantity=q, order_count=self._bid_oc.get(p,0)) for p,q in bids],
            [DepthLevel(price=p/10_000.0, quantity=q, order_count=self._ask_oc.get(p,0)) for p,q in asks],
        )


class ItchReplayer(DataSource):
    """
    Pure-Python ITCH 5.0 replayer.

    Parses a raw ITCH binary stream (no MoldUDP header — just the message-length
    framed messages as written by NASDAQ TotalView), reconstructs the LOB for
    the requested symbols, and emits LOBSnapshot events after every book-changing
    message.

    Parameters
    ----------
    data      : bytes-like ITCH binary data.
    symbols   : set of 8-char ticker strings to track (uppercase, space-padded).
                If empty, tracks all symbols (may be slow for full-day files).
    snap_every : emit a snapshot every N book-changing events (default 1 = every event).
    max_events : stop after this many book-changing events (0 = no limit).

    ITCH 5.0 message format (framed)
    ---------------------------------
      2 bytes big-endian : message length (does NOT include the length field itself)
      1 byte             : message type
      N-3 bytes          : body (length - 1 bytes)

    Key message types processed
    ---------------------------
      'A' (0x41) Add Order No MPID  — 35 bytes body
      'F' (0x46) Add Order with MPID — 39 bytes body
      'E' (0x45) Order Executed      — 30 bytes body
      'C' (0x43) Order Executed w/ Price — 35 bytes body
      'X' (0x58) Order Cancel        — 22 bytes body
      'D' (0x44) Order Delete        — 18 bytes body
      'U' (0x55) Order Replace       — 34 bytes body
    """

    # Body formats (big-endian, after stripping 1-byte type)
    # Fields: stock_locate(H), tracking(H), ts_hi(H), ts_lo(I), ...
    # Combined timestamp = (ts_hi << 32) | ts_lo  → nanoseconds since midnight
    _HDR = ">HHHl"          # stock_locate(2), tracking(2), ts_hi(2), ts_lo(4) [signed for simplicity]
    _HDR_SZ = struct.calcsize(_HDR)  # 10 bytes shared prefix

    def __init__(
        self,
        data:       bytes,
        symbols:    Optional[set]  = None,
        snap_every: int = 1,
        max_events: int = 0,
    ) -> None:
        self._data       = memoryview(data)
        self._symbols    = {s.upper().ljust(8)[:8] for s in symbols} if symbols else set()
        self._snap_every = max(1, snap_every)
        self._max_events = max_events

    def snapshots(self) -> Iterator[LOBSnapshot]:
        data  = self._data
        n     = len(data)
        pos   = 0
        lobs: Dict[str, _PythonLOB] = {}
        # ref → symbol mapping (needed because Executed/Delete only carry ref)
        ref_sym: Dict[int, str] = {}
        event_count = 0
        snap_seq    = 0

        while pos + 2 <= n:
            msg_len = struct.unpack_from(">H", data, pos)[0]
            pos += 2
            if pos + msg_len > n:
                break
            mtype = data[pos:pos+1].tobytes()
            body  = data[pos+1 : pos+msg_len]
            pos  += msg_len

            if len(body) < self._HDR_SZ:
                continue  # undersized (need at least the 10-byte common header)

            try:
                changed_sym = self._process(mtype, body, lobs, ref_sym)
            except Exception:
                continue

            if changed_sym is None:
                continue

            event_count += 1
            if event_count % self._snap_every != 0:
                continue

            lob = lobs[changed_sym]
            bids, asks = lob.depth(10)
            # Timestamp starts at offset 4 (after stock_locate[0:2] + tracking[2:4])
            ts_hi, ts_lo = struct.unpack_from(">HI", body, 4)
            ts_ns = (ts_hi << 32) | ts_lo

            yield LOBSnapshot(
                symbol       = changed_sym.rstrip(),
                timestamp_ns = ts_ns,
                bids         = bids,
                asks         = asks,
                seq          = snap_seq,
            )
            snap_seq += 1

            if self._max_events > 0 and event_count >= self._max_events:
                break

    def _process(
        self,
        mtype:    bytes,
        body:     memoryview,
        lobs:     Dict[str, _PythonLOB],
        ref_sym:  Dict[int, str],
    ) -> Optional[str]:
        """Dispatch one ITCH message.  Returns symbol name if LOB changed, else None."""

        # ITCH body layout (after stripping the 1-byte type):
        #   [0:2]  stock_locate  (H)
        #   [2:4]  tracking      (H)
        #   [4:6]  ts_hi         (H)  — upper 16 bits of 48-bit ns timestamp
        #   [6:10] ts_lo         (I)  — lower 32 bits
        #   [10:]  message-specific fields
        _B = 10   # base offset for all message-specific fields

        if mtype == b'A':   # Add Order No MPID
            # order_ref(Q=8), side(c=1), shares(I=4), stock(8s=8), price(I=4)
            if len(body) < _B + 25:
                return None
            ref,   = struct.unpack_from(">Q", body, _B)
            side   = chr(body[_B + 8])
            shares, = struct.unpack_from(">I", body, _B + 9)
            stock  = body[_B + 13 : _B + 21].tobytes().decode("ascii")
            price, = struct.unpack_from(">I", body, _B + 21)
            sym = stock
            if self._symbols and sym not in self._symbols:
                return None
            if sym not in lobs:
                lobs[sym] = _PythonLOB()
            lobs[sym].add(ref, price, shares, side)
            ref_sym[ref] = sym
            return sym

        elif mtype == b'F':   # Add Order with MPID (4 extra bytes at end)
            # order_ref(Q=8), side(c=1), shares(I=4), stock(8s=8), price(I=4), mpid(4s=4)
            if len(body) < _B + 29:
                return None
            ref,   = struct.unpack_from(">Q", body, _B)
            side   = chr(body[_B + 8])
            shares, = struct.unpack_from(">I", body, _B + 9)
            stock  = body[_B + 13 : _B + 21].tobytes().decode("ascii")
            price, = struct.unpack_from(">I", body, _B + 21)
            sym = stock
            if self._symbols and sym not in self._symbols:
                return None
            if sym not in lobs:
                lobs[sym] = _PythonLOB()
            lobs[sym].add(ref, price, shares, side)
            ref_sym[ref] = sym
            return sym

        elif mtype == b'E':   # Order Executed
            # order_ref(Q=8), shares(I=4), match_number(Q=8)
            if len(body) < _B + 20:
                return None
            ref,    = struct.unpack_from(">Q", body, _B)
            shares, = struct.unpack_from(">I", body, _B + 8)
            sym = ref_sym.get(ref)
            if sym and sym in lobs:
                lobs[sym].execute(ref, shares)
                return sym
            return None

        elif mtype == b'C':   # Order Executed with Price
            # order_ref(Q=8), shares(I=4), match_number(Q=8), printable(c=1), price(I=4)
            if len(body) < _B + 25:
                return None
            ref,    = struct.unpack_from(">Q", body, _B)
            shares, = struct.unpack_from(">I", body, _B + 8)
            sym = ref_sym.get(ref)
            if sym and sym in lobs:
                lobs[sym].execute(ref, shares)
                return sym
            return None

        elif mtype == b'X':   # Order Cancel
            # order_ref(Q=8), canceled_shares(I=4)
            if len(body) < _B + 12:
                return None
            ref,    = struct.unpack_from(">Q", body, _B)
            shares, = struct.unpack_from(">I", body, _B + 8)
            sym = ref_sym.get(ref)
            if sym and sym in lobs:
                lobs[sym].cancel(ref, shares)
                return sym
            return None

        elif mtype == b'D':   # Order Delete
            # order_ref(Q=8)
            if len(body) < _B + 8:
                return None
            ref, = struct.unpack_from(">Q", body, _B)
            sym = ref_sym.get(ref)
            if sym and sym in lobs:
                lobs[sym].delete(ref)
                ref_sym.pop(ref, None)
                return sym
            return None

        elif mtype == b'U':   # Order Replace
            # original_order_ref(Q=8), new_order_ref(Q=8), shares(I=4), price(I=4)
            if len(body) < _B + 24:
                return None
            old_ref, = struct.unpack_from(">Q", body, _B)
            new_ref, = struct.unpack_from(">Q", body, _B + 8)
            shares,  = struct.unpack_from(">I", body, _B + 16)
            price,   = struct.unpack_from(">I", body, _B + 20)
            sym = ref_sym.get(old_ref)
            if sym and sym in lobs:
                lobs[sym].replace(old_ref, new_ref, price, shares)
                ref_sym.pop(old_ref, None)
                ref_sym[new_ref] = sym
                return sym
            return None

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy protocol
# ═══════════════════════════════════════════════════════════════════════════════

class Strategy(ABC):
    """
    Base class for backtested strategies.

    Implement `on_snapshot` to generate orders given the current LOB snapshot
    and portfolio state.
    """

    @abstractmethod
    def on_snapshot(
        self,
        snapshot:  LOBSnapshot,
        portfolio: "Portfolio",   # forward-ref resolved at runtime
    ) -> List[Order]:
        ...

    def on_fill(self, fill: object, portfolio: "Portfolio") -> None:
        """Called after each fill (optional override)."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Parameters
    ----------
    source    : DataSource that yields LOBSnapshot objects.
    strategy  : Strategy subclass implementing on_snapshot().
    portfolio : Portfolio object (imported at runtime from portfolio module).
    spread_bps : assumed half-spread for market order fills (default 5 bps).
    impact_eta : market impact coefficient η (default 0.05); 0 = no impact.
    adv        : assumed average daily volume for impact scaling.

    Usage
    -----
    >>> source = ItchReplayer(raw_bytes, symbols={"AAPL    "})
    >>> engine = BacktestEngine(source, MyStrategy(), portfolio)
    >>> result = engine.run()
    """

    def __init__(
        self,
        source:     DataSource,
        strategy:   Strategy,
        portfolio:  object,
        spread_bps: float = 5.0,
        impact_eta: float = 0.05,
        adv:        float = 1e6,
    ) -> None:
        self.source     = source
        self.strategy   = strategy
        self.portfolio  = portfolio
        self.spread_bps = spread_bps
        self.impact_eta = impact_eta
        self.adv        = adv

    def run(self) -> EngineResult:
        """
        Run the backtest end-to-end.

        Returns EngineResult with equity curve, returns, and fill history.
        """
        from backtester.portfolio import Portfolio  # avoid circular import
        portfolio: Portfolio = self.portfolio

        equity_curve: List[float] = []
        timestamps:   List[int]   = []
        symbol_seen   = "UNKNOWN"

        for snap in self.source.snapshots():
            symbol_seen = snap.symbol

            # Mark portfolio to current mid-price
            mid = snap.mid
            if mid is not None:
                portfolio.mark(snap.symbol, mid)

            # Ask strategy for orders
            orders = self.strategy.on_snapshot(snap, portfolio)

            # Simulate fills
            for order in (orders or []):
                fill = self._fill(order, snap)
                if fill is not None:
                    portfolio.fill(fill)
                    self.strategy.on_fill(fill, portfolio)

            equity_curve.append(portfolio.equity())
            timestamps.append(snap.timestamp_ns)

        equity_arr = np.array(equity_curve, dtype=float)
        if len(equity_arr) > 1:
            returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-12)
        else:
            returns = np.array([], dtype=float)

        return EngineResult(
            snapshots_processed = len(equity_curve),
            fills               = list(portfolio.tracker.blotter()),
            equity_curve        = equity_arr,
            returns             = returns,
            timestamps          = timestamps,
            symbol              = symbol_seen,
        )

    def _fill(self, order: Order, snap: LOBSnapshot) -> Optional[object]:
        """Simple market-order fill simulator."""
        from backtester.portfolio import Fill  # avoid circular import

        if order.order_type == "market":
            ref_price = snap.best_ask if order.quantity > 0 else snap.best_bid
            if ref_price is None:
                return None
            half_spread = ref_price * self.spread_bps / 2.0 / 1e4
            direction   = 1.0 if order.quantity > 0 else -1.0

            # Market impact (square-root model)
            if self.adv > 0 and self.impact_eta > 0:
                vol = 0.02   # assumed daily vol for impact
                impact = self.impact_eta * vol * (abs(order.quantity) / self.adv) ** 0.5 * ref_price
            else:
                impact = 0.0

            fill_price = ref_price + direction * (half_spread + impact)
            return Fill(
                symbol     = order.symbol,
                quantity   = order.quantity,
                price      = fill_price,
                timestamp  = snap.timestamp_ns,
            )

        elif order.order_type == "limit" and order.limit_price is not None:
            # Passive limit: fill only if limit is through the market
            if order.quantity > 0 and snap.best_ask and order.limit_price >= snap.best_ask:
                return Fill(order.symbol, order.quantity,
                            snap.best_ask, snap.timestamp_ns)
            elif order.quantity < 0 and snap.best_bid and order.limit_price <= snap.best_bid:
                return Fill(order.symbol, order.quantity,
                            snap.best_bid, snap.timestamp_ns)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — synthetic ITCH data generator (for tests)
# ═══════════════════════════════════════════════════════════════════════════════

def build_itch_add(ref: int, price_ticks: int, qty: int, side: str,
                   stock: str = "AAPL    ", ts_ns: int = 0) -> bytes:
    """
    Build a single ITCH 5.0 'A' (Add Order No MPID) message including
    the 2-byte length prefix.

    price_ticks : ITCH integer price ($0.0001 increments).
    """
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo =  ts_ns        & 0xFFFFFFFF
    body  = struct.pack(">HHHIQcI8sI",
                        0,           # stock_locate
                        0,           # tracking
                        ts_hi, ts_lo,
                        ref,         # order_ref_num (uint64)
                        side.encode(),
                        qty,
                        stock[:8].ljust(8).encode(),
                        price_ticks)
    msg   = b'A' + body
    return struct.pack(">H", len(msg)) + msg


def build_itch_delete(ref: int, ts_ns: int = 0) -> bytes:
    """Build a ITCH 5.0 'D' (Order Delete) message."""
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo =  ts_ns        & 0xFFFFFFFF
    body  = struct.pack(">HHHIQ",
                        0, 0, ts_hi, ts_lo, ref)
    msg   = b'D' + body
    return struct.pack(">H", len(msg)) + msg


def build_itch_execute(ref: int, qty: int, ts_ns: int = 0) -> bytes:
    """Build an ITCH 5.0 'E' (Order Executed) message."""
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo =  ts_ns        & 0xFFFFFFFF
    body  = struct.pack(">HHHIQIQ",
                        0, 0, ts_hi, ts_lo,
                        ref, qty,
                        0)          # match_number
    msg   = b'E' + body
    return struct.pack(">H", len(msg)) + msg
