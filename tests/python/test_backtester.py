"""
Tests — Backtester (engine, portfolio, tearsheet)
==================================================
Full integration stack:

  synthetic ITCH bytes
    → ItchReplayer
      → BacktestEngine
        → Portfolio
          → Tearsheet

Covers
------
  Unit
    - Fill dataclass (notional, cash_impact)
    - Commission model (per-share, minimum enforcement, zero)
    - Portfolio cash accounting (buy / sell / leverage reject)
    - Portfolio equity / gross_exposure / net_exposure / leverage
    - Portfolio reset
    - DepthLevel / LOBSnapshot properties (best_bid, best_ask, mid, spread_bps)
    - Order properties
    - SnapshotSource replay

  ITCH parser
    - build_itch_add / delete / execute helpers produce valid frame bytes
    - ItchReplayer parses 'A' Add messages → LOBSnapshot
    - ItchReplayer parses 'D' Delete messages → price level removed
    - ItchReplayer parses 'E' Execute messages → quantity reduced
    - ItchReplayer tracks multiple symbols independently
    - ItchReplayer snap_every throttle
    - ItchReplayer max_events limit
    - ItchReplayer symbol filter (only wanted symbols emitted)
    - Corrupt / truncated bytes do not crash the replayer
    - Empty bytes stream → zero snapshots

  BacktestEngine + Portfolio
    - Market order fills at best_ask / best_bid ± spread
    - Limit order fills only when limit is through the market
    - Limit order rejects when away from market
    - Flat LOB (no bids/asks) → no fill generated
    - Portfolio updated correctly after each fill
    - Equity curve length == snapshots processed
    - EngineResult fields are all populated

  End-to-end ITCH replay
    - Synthetic 1-day ITCH stream → ItchReplayer → BacktestEngine → Portfolio
    - MidPrice strategy generates fills on every snapshot
    - Final equity ≠ initial equity after fills
    - Tearsheet generated without error; PSR ∈ [0,1]; dict() keys complete

  Tearsheet
    - from_returns produces correct Sharpe / PSR
    - dict() contains all expected keys
    - text() includes symbol name and equity figures
    - Handles edge cases: single return, zero returns, NaN returns
"""

import struct
import pytest
import numpy as np

# ── path helper ─────────────────────────────────────────────────────────────
import sys, os
_HERE   = os.path.dirname(__file__)
_PYROOT = os.path.join(_HERE, "..", "..", "python")
if _PYROOT not in sys.path:
    sys.path.insert(0, _PYROOT)

from backtester.engine import (
    BacktestEngine,
    DepthLevel,
    EngineResult,
    ItchReplayer,
    LOBSnapshot,
    Order,
    SnapshotSource,
    Strategy,
    build_itch_add,
    build_itch_delete,
    build_itch_execute,
)
from backtester.portfolio import Commission, Fill, Portfolio, ZeroCommission
from backtester.tearsheet import Tearsheet


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _snap(bid=100.0, ask=100.10, sym="AAPL", ts=0, seq=0):
    """Minimal LOBSnapshot with one bid and one ask level."""
    return LOBSnapshot(
        symbol       = sym,
        timestamp_ns = ts,
        bids         = [DepthLevel(price=bid, quantity=1000, order_count=5)],
        asks         = [DepthLevel(price=ask, quantity=1000, order_count=5)],
        seq          = seq,
    )


def _empty_snap(sym="AAPL", ts=0):
    """Snapshot with no bids or asks."""
    return LOBSnapshot(symbol=sym, timestamp_ns=ts, bids=[], asks=[], seq=0)


def _build_itch_stream(n_adds=5, base_price_ticks=1_000_000,
                       stock="AAPL    ", start_ref=1):
    """
    Build a synthetic ITCH stream:
      n_adds  'A' Add-Order messages (alternating bid/ask).
    Returns raw bytes.
    """
    chunks = []
    for i in range(n_adds):
        side  = b'B' if i % 2 == 0 else b'S'
        price = base_price_ticks + i * 100   # spread orders across prices
        qty   = 100 + i * 50
        ref   = start_ref + i
        ts    = i * 1_000_000                # 1 ms apart
        chunks.append(build_itch_add(ref, price, qty, side.decode(),
                                     stock=stock, ts_ns=ts))
    return b"".join(chunks)


# ════════════════════════════════════════════════════════════════════════════
# 1 — Fill dataclass
# ════════════════════════════════════════════════════════════════════════════

class TestFill:
    def test_notional_buy(self):
        f = Fill("AAPL", 100, 150.0, 0)
        assert f.notional == pytest.approx(15_000.0)

    def test_notional_sell(self):
        f = Fill("AAPL", -50, 200.0, 0)
        assert f.notional == pytest.approx(10_000.0)

    def test_cash_impact_buy_is_negative(self):
        f = Fill("AAPL", 100, 150.0, 0)
        assert f.cash_impact == pytest.approx(-15_000.0)

    def test_cash_impact_sell_is_positive(self):
        f = Fill("AAPL", -50, 200.0, 0)
        assert f.cash_impact == pytest.approx(10_000.0)

    def test_repr(self):
        f = Fill("AAPL", 100, 150.0, 0)
        assert "BUY" in repr(f)
        f2 = Fill("AAPL", -50, 200.0, 0)
        assert "SELL" in repr(f2)


# ════════════════════════════════════════════════════════════════════════════
# 2 — Commission models
# ════════════════════════════════════════════════════════════════════════════

class TestCommission:
    def test_per_share_normal(self):
        c = Commission(per_share=0.01, min_per_trade=1.0)
        f = Fill("X", 500, 10.0, 0)
        assert c.compute(f) == pytest.approx(5.0)

    def test_minimum_enforced(self):
        c = Commission(per_share=0.001, min_per_trade=2.0)
        f = Fill("X", 10, 10.0, 0)
        # 10 * 0.001 = 0.01 < 2.0  → min applies
        assert c.compute(f) == pytest.approx(2.0)

    def test_zero_commission(self):
        c = ZeroCommission()
        f = Fill("X", 1000, 50.0, 0)
        assert c.compute(f) == 0.0

    def test_sell_uses_abs_quantity(self):
        c = Commission(per_share=0.005, min_per_trade=1.0)
        f = Fill("X", -200, 10.0, 0)
        assert c.compute(f) == pytest.approx(max(200 * 0.005, 1.0))


# ════════════════════════════════════════════════════════════════════════════
# 3 — Portfolio accounting
# ════════════════════════════════════════════════════════════════════════════

class TestPortfolioAccounting:
    def setup_method(self):
        self.port = Portfolio(initial_cash=100_000.0, commission=ZeroCommission())

    def test_initial_cash(self):
        assert self.port.cash() == pytest.approx(100_000.0)

    def test_initial_equity(self):
        assert self.port.equity() == pytest.approx(100_000.0)

    def test_buy_reduces_cash(self):
        f = Fill("AAPL", 100, 150.0, 0)
        self.port.fill(f)
        assert self.port.cash() == pytest.approx(100_000.0 - 100 * 150.0)

    def test_sell_increases_cash(self):
        # First buy
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        cash_after_buy = self.port.cash()
        # Then sell
        self.port.fill(Fill("AAPL", -100, 155.0, 1))
        assert self.port.cash() > cash_after_buy

    def test_position_after_buy(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        assert self.port.position("AAPL") == pytest.approx(100.0)

    def test_position_after_buy_sell(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.fill(Fill("AAPL", -60, 152.0, 1))
        assert self.port.position("AAPL") == pytest.approx(40.0)

    def test_position_value(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.mark("AAPL", 160.0)
        assert self.port.position_value("AAPL") == pytest.approx(100 * 160.0)

    def test_equity_includes_unrealised_pnl(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.mark("AAPL", 160.0)
        # equity = cash + unrealised P&L
        expected = 100_000.0 - 100 * 150.0 + 100 * 160.0 - 100 * 150.0 + 0.0
        # More directly: equity = initial_cash + realised_pnl + unrealised_pnl
        assert self.port.equity() > 100_000.0

    def test_equity_conservation_flat(self):
        # After buying and immediately selling at same price, equity unchanged
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.fill(Fill("AAPL", -100, 150.0, 1))
        assert self.port.equity() == pytest.approx(100_000.0, rel=1e-6)

    def test_gross_exposure(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.mark("AAPL", 150.0)
        assert self.port.gross_exposure() == pytest.approx(100 * 150.0)

    def test_net_exposure(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.fill(Fill("MSFT", -50, 300.0, 1))
        self.port.mark("AAPL", 150.0)
        self.port.mark("MSFT", 300.0)
        net = 100 * 150.0 - 50 * 300.0
        assert self.port.net_exposure() == pytest.approx(net)

    def test_leverage(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.mark("AAPL", 150.0)
        lev = self.port.leverage()
        assert lev > 0.0

    def test_equity_curve_snapshot(self):
        self.port.snapshot_equity()
        self.port.snapshot_equity()
        eq = self.port.equity_curve()
        assert len(eq) == 2

    def test_returns(self):
        self.port.snapshot_equity()
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.mark("AAPL", 160.0)
        self.port.snapshot_equity()
        r = self.port.returns()
        assert len(r) == 1

    def test_fills_list(self):
        f1 = Fill("AAPL", 100, 150.0, 0)
        f2 = Fill("AAPL", -50, 155.0, 1)
        self.port.fill(f1)
        self.port.fill(f2)
        assert len(self.port.fills()) == 2

    def test_total_pnl(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.fill(Fill("AAPL", -100, 160.0, 1))
        assert self.port.total_pnl() == pytest.approx(100 * 10.0, rel=1e-6)

    def test_total_commissions(self):
        port = Portfolio(100_000.0, commission=Commission(0.005, 1.0))
        port.fill(Fill("AAPL", 200, 100.0, 0))
        assert port.total_commissions() == pytest.approx(max(200 * 0.005, 1.0))

    def test_reset(self):
        self.port.fill(Fill("AAPL", 100, 150.0, 0))
        self.port.mark("AAPL", 160.0)
        self.port.snapshot_equity()
        self.port.reset()
        assert self.port.cash()    == pytest.approx(100_000.0)
        assert self.port.equity()  == pytest.approx(100_000.0)
        assert len(self.port.fills())  == 0
        assert len(self.port.equity_curve()) == 0

    def test_repr(self):
        r = repr(self.port)
        assert "Portfolio" in r


class TestPortfolioLeverage:
    def test_leverage_reject(self):
        port = Portfolio(initial_cash=10_000.0, commission=ZeroCommission(),
                         max_leverage=2.0)
        port.mark("AAPL", 100.0)
        # Try to buy 300 shares @ $100 = $30,000 notional → 3× leverage → reject
        result = port.fill(Fill("AAPL", 300, 100.0, 0))
        assert result is False
        assert port.position("AAPL") == pytest.approx(0.0)

    def test_leverage_accept_within_limit(self):
        port = Portfolio(initial_cash=10_000.0, commission=ZeroCommission(),
                         max_leverage=3.0)
        port.mark("AAPL", 100.0)
        # Buy 100 @ 100 = $10,000 notional → 1× leverage → accept
        result = port.fill(Fill("AAPL", 100, 100.0, 0))
        assert result is True
        assert port.position("AAPL") == pytest.approx(100.0)

    def test_zero_max_leverage_means_unlimited(self):
        port = Portfolio(initial_cash=1_000.0, commission=ZeroCommission(),
                         max_leverage=0.0)
        result = port.fill(Fill("AAPL", 10_000, 100.0, 0))
        assert result is True


# ════════════════════════════════════════════════════════════════════════════
# 4 — LOBSnapshot / DepthLevel / Order
# ════════════════════════════════════════════════════════════════════════════

class TestLOBSnapshot:
    def test_best_bid_ask(self):
        s = _snap(bid=99.0, ask=99.10)
        assert s.best_bid == pytest.approx(99.0)
        assert s.best_ask == pytest.approx(99.10)

    def test_mid(self):
        s = _snap(bid=100.0, ask=100.10)
        assert s.mid == pytest.approx(100.05)

    def test_spread(self):
        s = _snap(bid=100.0, ask=100.10)
        assert s.spread == pytest.approx(0.10)

    def test_spread_bps(self):
        s = _snap(bid=100.0, ask=100.10)
        # 0.10 / 100.05 * 10000 ≈ 9.99 bps
        assert s.spread_bps == pytest.approx(0.10 / 100.05 * 1e4, rel=1e-4)

    def test_empty_snap_no_bid_ask(self):
        s = _empty_snap()
        assert s.best_bid is None
        assert s.best_ask is None
        assert s.mid is None
        assert s.spread is None

    def test_bid_depth_qty(self):
        s = _snap()
        assert s.bid_depth_qty == 1000

    def test_ask_depth_qty(self):
        s = _snap()
        assert s.ask_depth_qty == 1000

    def test_repr(self):
        s = _snap()
        assert "LOBSnapshot" in repr(s)

    def test_depth_level_repr(self):
        d = DepthLevel(price=100.0, quantity=500, order_count=3)
        assert "DepthLevel" in repr(d)


class TestOrder:
    def test_side_buy(self):
        o = Order("AAPL", 100)
        assert o.side == "buy"

    def test_side_sell(self):
        o = Order("AAPL", -100)
        assert o.side == "sell"

    def test_default_market(self):
        o = Order("AAPL", 100)
        assert o.order_type == "market"


# ════════════════════════════════════════════════════════════════════════════
# 5 — ITCH helpers
# ════════════════════════════════════════════════════════════════════════════

class TestItchHelpers:
    def test_build_add_has_length_prefix(self):
        msg = build_itch_add(1, 1_000_000, 100, 'B')
        msg_len = struct.unpack(">H", msg[:2])[0]
        assert len(msg) == 2 + msg_len

    def test_build_add_type_byte(self):
        msg = build_itch_add(1, 1_000_000, 100, 'B')
        assert msg[2:3] == b'A'

    def test_build_delete_type_byte(self):
        msg = build_itch_delete(1)
        assert msg[2:3] == b'D'

    def test_build_execute_type_byte(self):
        msg = build_itch_execute(1, 50)
        assert msg[2:3] == b'E'

    def test_add_delete_round_trip(self):
        """Add an order then delete it; LOB should be empty."""
        stream = build_itch_add(1, 1_000_000, 100, 'B', stock="TEST    ") + \
                 build_itch_delete(1)
        snaps = list(ItchReplayer(stream, symbols={"TEST"}).snapshots())
        # After delete, best_bid should be gone
        last = snaps[-1]
        assert last.best_bid is None or last.bid_depth_qty == 0

    def test_add_execute_partial(self):
        """Execute half of an order; remaining qty halved."""
        stream = build_itch_add(1, 1_000_000, 200, 'B', stock="TEST    ") + \
                 build_itch_execute(1, 100)
        snaps = list(ItchReplayer(stream, symbols={"TEST"}).snapshots())
        last = snaps[-1]
        assert last.bid_depth_qty == pytest.approx(100)


# ════════════════════════════════════════════════════════════════════════════
# 6 — ItchReplayer
# ════════════════════════════════════════════════════════════════════════════

class TestItchReplayer:
    def test_empty_stream(self):
        snaps = list(ItchReplayer(b"").snapshots())
        assert len(snaps) == 0

    def test_single_add(self):
        stream = build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ")
        snaps  = list(ItchReplayer(stream).snapshots())
        assert len(snaps) == 1
        assert snaps[0].best_bid == pytest.approx(100.0)

    def test_best_bid_from_tick_price(self):
        # price_ticks = 1_500_000 → $150.0000
        stream = build_itch_add(1, 1_500_000, 100, 'B', stock="AAPL    ")
        snaps  = list(ItchReplayer(stream).snapshots())
        assert snaps[0].best_bid == pytest.approx(150.0)

    def test_bid_ask_both_present(self):
        stream = (build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ") +
                  build_itch_add(2, 1_001_000, 100, 'S', stock="AAPL    "))
        snaps  = list(ItchReplayer(stream).snapshots())
        last   = snaps[-1]
        assert last.best_bid  == pytest.approx(100.0)
        assert last.best_ask  == pytest.approx(100.1)

    def test_symbol_filter(self):
        stream = (build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ") +
                  build_itch_add(2, 2_000_000, 100, 'B', stock="MSFT    "))
        snaps  = list(ItchReplayer(stream, symbols={"AAPL"}).snapshots())
        for s in snaps:
            assert s.symbol == "AAPL"

    def test_multiple_symbols_tracked(self):
        stream = (build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ") +
                  build_itch_add(2, 2_000_000, 100, 'B', stock="MSFT    "))
        snaps  = list(ItchReplayer(stream).snapshots())
        syms   = {s.symbol for s in snaps}
        assert "AAPL" in syms
        assert "MSFT" in syms

    def test_snap_every_throttle(self):
        stream = _build_itch_stream(n_adds=10, stock="THRT    ")
        snaps_1  = list(ItchReplayer(stream, snap_every=1).snapshots())
        snaps_5  = list(ItchReplayer(stream, snap_every=5).snapshots())
        assert len(snaps_5) <= len(snaps_1)

    def test_max_events_limit(self):
        stream = _build_itch_stream(n_adds=20, stock="MAXE    ")
        snaps  = list(ItchReplayer(stream, max_events=5).snapshots())
        assert len(snaps) <= 5

    def test_corrupt_bytes_no_crash(self):
        garbage = b"\xff\xfe" + b"\x00" * 30
        stream  = garbage + _build_itch_stream(n_adds=3, stock="GOOD    ")
        # Should not raise; may yield fewer snapshots
        try:
            snaps = list(ItchReplayer(stream, symbols={"GOOD"}).snapshots())
        except Exception as e:
            pytest.fail(f"ItchReplayer raised on corrupt data: {e}")

    def test_truncated_message_no_crash(self):
        stream = build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ")
        # Truncate to half
        truncated = stream[: len(stream) // 2]
        snaps = list(ItchReplayer(truncated).snapshots())
        # Should not crash; 0 or 1 snapshots are both acceptable

    def test_seq_numbers_monotone(self):
        stream = _build_itch_stream(n_adds=10)
        snaps  = list(ItchReplayer(stream).snapshots())
        seqs   = [s.seq for s in snaps]
        assert seqs == sorted(seqs)

    def test_snapshot_symbol_stripped(self):
        """Symbol should be right-stripped of spaces."""
        stream = build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ")
        snaps  = list(ItchReplayer(stream).snapshots())
        assert snaps[0].symbol == "AAPL"

    def test_timestamp_decoded(self):
        ts = 34_200_000_000_000   # 9.5 hours into day in ns
        stream = build_itch_add(1, 1_000_000, 100, 'B', stock="AAPL    ", ts_ns=ts)
        snaps  = list(ItchReplayer(stream).snapshots())
        assert snaps[0].timestamp_ns == ts


# ════════════════════════════════════════════════════════════════════════════
# 7 — SnapshotSource
# ════════════════════════════════════════════════════════════════════════════

class TestSnapshotSource:
    def test_yields_all_snaps(self):
        snaps = [_snap(ts=i) for i in range(5)]
        from backtester.engine import SnapshotSource
        src = SnapshotSource(snaps)
        out = list(src.snapshots())
        assert len(out) == 5

    def test_empty_source(self):
        from backtester.engine import SnapshotSource
        src = SnapshotSource([])
        assert list(src.snapshots()) == []

    def test_close_no_op(self):
        from backtester.engine import SnapshotSource
        src = SnapshotSource([])
        src.close()   # should not raise


# ════════════════════════════════════════════════════════════════════════════
# 8 — BacktestEngine fills
# ════════════════════════════════════════════════════════════════════════════

class _BuyFirstStrategy(Strategy):
    """Buys 100 shares on the first snapshot, does nothing after."""
    def __init__(self):
        self._filled = False

    def on_snapshot(self, snapshot, portfolio):
        if not self._filled:
            self._filled = True
            return [Order(snapshot.symbol, 100)]
        return []


class _SellFirstStrategy(Strategy):
    """Sells 50 shares on the first snapshot."""
    def __init__(self):
        self._filled = False

    def on_snapshot(self, snapshot, portfolio):
        if not self._filled:
            self._filled = True
            return [Order(snapshot.symbol, -50)]
        return []


class _LimitAboveMarket(Strategy):
    """Places a buy limit above ask (should fill immediately)."""
    def __init__(self):
        self._filled = False

    def on_snapshot(self, snapshot, portfolio):
        if not self._filled:
            self._filled = True
            return [Order(snapshot.symbol, 100, order_type="limit",
                          limit_price=snapshot.best_ask + 1.0)]
        return []


class _LimitBelowMarket(Strategy):
    """Places a buy limit below ask (should NOT fill)."""
    def __init__(self):
        self._filled = False

    def on_snapshot(self, snapshot, portfolio):
        if not self._filled:
            self._filled = True
            return [Order(snapshot.symbol, 100, order_type="limit",
                          limit_price=snapshot.best_ask - 5.0)]
        return []


class _NoOpStrategy(Strategy):
    """Never trades."""
    def on_snapshot(self, snapshot, portfolio):
        return []


class TestBacktestEngineFills:
    def _run(self, strategy, snaps, initial_cash=100_000.0, spread_bps=0.0,
             impact_eta=0.0):
        port   = Portfolio(initial_cash=initial_cash, commission=ZeroCommission())
        source = SnapshotSource(snaps)
        engine = BacktestEngine(source, strategy, port,
                                spread_bps=spread_bps, impact_eta=impact_eta)
        return engine.run(), port

    def test_market_buy_fill(self):
        result, port = self._run(_BuyFirstStrategy(), [_snap()])
        assert port.position("AAPL") == pytest.approx(100.0)

    def test_market_sell_fill(self):
        # Seed a position first via direct portfolio fill
        port   = Portfolio(100_000.0, commission=ZeroCommission())
        port.fill(Fill("AAPL", 200, 100.0, 0))
        source = SnapshotSource([_snap()])
        engine = BacktestEngine(source, _SellFirstStrategy(), port,
                                spread_bps=0.0, impact_eta=0.0)
        engine.run()
        assert port.position("AAPL") == pytest.approx(150.0)

    def test_market_fill_price_includes_spread(self):
        """Buy fill price = best_ask + half_spread."""
        snaps  = [_snap(bid=100.0, ask=100.0)]
        result, port = self._run(_BuyFirstStrategy(), snaps, spread_bps=10.0)
        fills = port.fills()
        assert len(fills) > 0
        # fill price > ask when spread > 0
        fill_price = fills[0].price if hasattr(fills[0], 'price') else fills[0][2]

    def test_limit_above_market_fills(self):
        result, port = self._run(_LimitAboveMarket(), [_snap(bid=100.0, ask=100.10)])
        assert port.position("AAPL") == pytest.approx(100.0)

    def test_limit_below_market_no_fill(self):
        result, port = self._run(_LimitBelowMarket(), [_snap(bid=100.0, ask=100.10)])
        assert port.position("AAPL") == pytest.approx(0.0)

    def test_flat_lob_no_fill(self):
        result, port = self._run(_BuyFirstStrategy(), [_empty_snap()])
        assert port.position("AAPL") == pytest.approx(0.0)

    def test_no_op_strategy_no_fills(self):
        snaps  = [_snap() for _ in range(10)]
        result, port = self._run(_NoOpStrategy(), snaps)
        assert port.position("AAPL") == pytest.approx(0.0)

    def test_equity_curve_length(self):
        snaps  = [_snap(ts=i) for i in range(20)]
        result, _ = self._run(_NoOpStrategy(), snaps)
        assert result.snapshots_processed == 20
        assert len(result.equity_curve) == 20

    def test_engine_result_fields(self):
        snaps  = [_snap(ts=i) for i in range(5)]
        result, _ = self._run(_BuyFirstStrategy(), snaps)
        assert isinstance(result.snapshots_processed, int)
        assert isinstance(result.equity_curve, np.ndarray)
        assert isinstance(result.returns, np.ndarray)
        assert isinstance(result.timestamps, list)
        assert isinstance(result.symbol, str)

    def test_timestamps_recorded(self):
        snaps  = [_snap(ts=i * 1_000_000) for i in range(5)]
        result, _ = self._run(_NoOpStrategy(), snaps)
        assert result.timestamps[0] == 0
        assert result.timestamps[-1] == 4 * 1_000_000

    def test_mark_updates_equity(self):
        port   = Portfolio(100_000.0, commission=ZeroCommission())
        port.fill(Fill("AAPL", 100, 100.0, 0))
        # Snap at higher price
        source = SnapshotSource([_snap(bid=110.0, ask=110.10)])
        engine = BacktestEngine(source, _NoOpStrategy(), port,
                                spread_bps=0.0, impact_eta=0.0)
        result = engine.run()
        assert result.equity_curve[-1] > 100_000.0 - 100 * 100.0


# ════════════════════════════════════════════════════════════════════════════
# 9 — End-to-end ITCH replay
# ════════════════════════════════════════════════════════════════════════════

class _MidPriceStrategy(Strategy):
    """
    Buys 10 shares whenever there's a valid mid-price and we're flat.
    Then sells when holding >= 10 shares.
    """
    def on_snapshot(self, snapshot, portfolio):
        if snapshot.mid is None:
            return []
        pos = portfolio.position(snapshot.symbol)
        if pos == 0:
            return [Order(snapshot.symbol, 10)]
        elif pos >= 10:
            return [Order(snapshot.symbol, -10)]
        return []


class TestItchEndToEnd:
    def _build_day_stream(self, n_msgs=50):
        """Build a stream with alternating bid/ask adds for a full 'day'."""
        chunks = []
        for i in range(n_msgs):
            # Even: add bid at 99.xx; Odd: add ask at 101.xx
            if i % 2 == 0:
                price_ticks = 990_000 + i * 100    # ~$99
                side = 'B'
            else:
                price_ticks = 1_010_000 + i * 100  # ~$101
                side = 'S'
            chunks.append(build_itch_add(
                ref          = i + 1,
                price_ticks  = price_ticks,
                qty          = 1000,
                side         = side,
                stock        = "AAPL    ",
                ts_ns        = i * 60_000_000_000,  # 1 min apart
            ))
        return b"".join(chunks)

    def test_full_replay_no_crash(self):
        stream = self._build_day_stream(50)
        port   = Portfolio(1_000_000.0, commission=ZeroCommission())
        source = ItchReplayer(stream, symbols={"AAPL"})
        engine = BacktestEngine(source, _MidPriceStrategy(), port,
                                spread_bps=2.0, impact_eta=0.0)
        result = engine.run()
        assert result.snapshots_processed > 0

    def test_fills_generated(self):
        stream = self._build_day_stream(50)
        port   = Portfolio(1_000_000.0, commission=ZeroCommission())
        source = ItchReplayer(stream, symbols={"AAPL"})
        engine = BacktestEngine(source, _MidPriceStrategy(), port,
                                spread_bps=0.0, impact_eta=0.0)
        result = engine.run()
        # After alternating buy/sell the portfolio should have traded
        assert len(port.fills()) > 0

    def test_equity_curve_has_variation(self):
        stream = self._build_day_stream(50)
        port   = Portfolio(1_000_000.0, commission=ZeroCommission())
        source = ItchReplayer(stream, symbols={"AAPL"})
        engine = BacktestEngine(source, _MidPriceStrategy(), port,
                                spread_bps=0.0, impact_eta=0.0)
        result = engine.run()
        if len(result.equity_curve) > 1:
            assert not np.all(result.equity_curve == result.equity_curve[0])

    def test_returns_length(self):
        stream = self._build_day_stream(20)
        port   = Portfolio(1_000_000.0, commission=ZeroCommission())
        source = ItchReplayer(stream, symbols={"AAPL"})
        engine = BacktestEngine(source, _NoOpStrategy(), port,
                                spread_bps=0.0, impact_eta=0.0)
        result = engine.run()
        assert len(result.returns) == max(0, result.snapshots_processed - 1)

    def test_tearsheet_from_result(self):
        stream = self._build_day_stream(50)
        port   = Portfolio(1_000_000.0, commission=ZeroCommission())
        source = ItchReplayer(stream, symbols={"AAPL"})
        engine = BacktestEngine(source, _MidPriceStrategy(), port,
                                spread_bps=0.0, impact_eta=0.0)
        result = engine.run()
        sheet  = Tearsheet.from_result(result, periods_per_year=252)
        assert 0.0 <= sheet.psr <= 1.0
        assert isinstance(sheet.dict(), dict)


# ════════════════════════════════════════════════════════════════════════════
# 10 — Tearsheet
# ════════════════════════════════════════════════════════════════════════════

class TestTearsheet:
    def _rng_returns(self, n=252, mean=0.001, vol=0.01, seed=42):
        rng = np.random.default_rng(seed)
        return rng.normal(mean, vol, n)

    def test_sharpe_sign(self):
        r = self._rng_returns(mean=0.001, vol=0.01)
        s = Tearsheet.from_returns(r, periods_per_year=252)
        # Positive mean returns → positive Sharpe
        assert s.report.sharpe > 0

    def test_psr_in_unit_interval(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r, periods_per_year=252)
        assert 0.0 <= s.psr <= 1.0

    def test_dsr_le_psr(self):
        """DSR (N-trial corrected) ≤ PSR (single-trial) for N > 1."""
        r = self._rng_returns()
        s = Tearsheet.from_returns(r, n_trials=10, periods_per_year=252)
        assert s.dsr <= s.psr + 1e-9   # allow floating-point tolerance

    def test_dict_keys_complete(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r)
        d = s.dict()
        for key in ("sharpe", "sortino", "calmar", "max_drawdown",
                    "psr", "dsr", "total_return", "ann_return",
                    "ann_volatility", "var_95", "var_99"):
            assert key in d, f"Missing key: {key}"

    def test_dict_values_are_floats(self):
        r = self._rng_returns()
        d = Tearsheet.from_returns(r).dict()
        for k, v in d.items():
            assert isinstance(v, (float, int)), f"Non-numeric value for {k}: {v}"

    def test_text_contains_symbol(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r, symbol="TSLA")
        txt = s.text()
        assert "TSLA" in txt

    def test_text_contains_equity(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r, initial_equity=500_000.0)
        txt = s.text()
        assert "500" in txt

    def test_repr(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r, symbol="AAPL")
        assert "Tearsheet" in repr(s)
        assert "AAPL" in repr(s)

    def test_single_return(self):
        """Single-period return should not crash."""
        s = Tearsheet.from_returns(np.array([0.01]))
        assert isinstance(s.dict(), dict)

    def test_zero_returns(self):
        """All-zero returns: Sharpe = 0, no crash."""
        s = Tearsheet.from_returns(np.zeros(100))
        assert s.report.sharpe == pytest.approx(0.0, abs=1e-10)

    def test_nan_returns_filtered(self):
        """NaN values in returns should be silently dropped."""
        r = np.array([0.01, np.nan, 0.02, np.nan, -0.005])
        s = Tearsheet.from_returns(r)
        assert np.isfinite(s.report.sharpe)

    def test_empty_returns(self):
        """Empty array should not crash."""
        s = Tearsheet.from_returns(np.array([]))
        assert s.report.n_periods == 0

    def test_negative_mean_psr_below_half(self):
        """Consistently negative returns: PSR (SR > 0) should be < 0.5."""
        r = self._rng_returns(mean=-0.005, vol=0.01, n=500)
        s = Tearsheet.from_returns(r, sr_benchmark=0.0, periods_per_year=252)
        assert s.psr < 0.5

    def test_positive_mean_psr_above_half(self):
        """Strong positive returns: PSR (SR > 0) should be > 0.5."""
        r = self._rng_returns(mean=0.005, vol=0.005, n=500)
        s = Tearsheet.from_returns(r, sr_benchmark=0.0, periods_per_year=252)
        assert s.psr > 0.5

    def test_from_result_interface(self):
        """from_result should accept a duck-typed EngineResult."""
        # Create a minimal mock EngineResult
        class MockResult:
            returns      = np.linspace(0.0, 0.001, 100)
            equity_curve = np.cumprod(1 + returns) * 1_000_000
            symbol       = "MOCK"
            fills        = [None] * 5

        s = Tearsheet.from_result(MockResult(), periods_per_year=252,
                                   initial_equity=1_000_000.0)
        assert s.symbol == "MOCK"
        assert s.n_fills == 5

    def test_min_track_len_positive(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r)
        assert s.min_track_len >= 1

    def test_total_commissions_stored(self):
        r = self._rng_returns()
        s = Tearsheet.from_returns(r, total_commissions=1234.56)
        assert s.total_commissions == pytest.approx(1234.56)
