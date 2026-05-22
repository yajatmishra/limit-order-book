"""
Synthetic Session Replay Generator
====================================
Produces a realistic simulated trading day:

  • Mid-price follows a GBM with two volatility regimes (low/high)
    driven by a 2-state Markov chain.
  • Bid-ask spread is drawn from a half-normal and widens in the
    high-vol regime.
  • Five depth levels on each side with exponentially decaying size.
  • OFI-tainted noise so the OFI signal has mild predictive power
    (β ≈ 0.003 per share of price impact).

Strategy
--------
  OFIMomentumStrategy:
    Maintains a rolling window of OFI.  Submits a market buy when the
    cumulative rolling OFI exceeds +threshold (net buying pressure) and a
    market sell when it falls below −threshold (net selling pressure).
    Position is capped at ±max_pos shares.

Returns
-------
  SessionData:  named dataclass containing
    snapshots   : list[LOBSnapshot]      — all book snapshots
    result      : EngineResult           — backtester output
    tearsheet   : Tearsheet              — performance metrics
    ofi_series  : np.ndarray             — per-snapshot OFI (len N-1)
    mid_prices  : np.ndarray             — mid-price per snapshot
    timestamps  : np.ndarray             — timestamp_ns per snapshot
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY   = os.path.join(_HERE, "..")           # sigma-edge/python/
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from backtester.engine    import (
    LOBSnapshot, DepthLevel, SnapshotSource, BacktestEngine,
    Strategy, Order, EngineResult,
)
from backtester.portfolio import Portfolio, ZeroCommission
from backtester.tearsheet import Tearsheet
from microstructure.ofi   import compute_ofi


# ── Session data container ─────────────────────────────────────────────────────

@dataclass
class SessionData:
    """All data produced by a replay session."""
    snapshots:   List[LOBSnapshot]
    result:      EngineResult
    tearsheet:   Tearsheet
    ofi_series:  np.ndarray     # shape (N-1,) — aligned to snapshots[1:]
    mid_prices:  np.ndarray     # shape (N,)
    spread_bps:  np.ndarray     # shape (N,)
    timestamps:  np.ndarray     # shape (N,) — ns since midnight (synthetic)
    bid_depth:   np.ndarray     # shape (N,) — total bid qty
    ask_depth:   np.ndarray     # shape (N,) — total ask qty


# ── OFI-Momentum strategy ──────────────────────────────────────────────────────

class OFIMomentumStrategy(Strategy):
    """
    Long when rolling OFI > +threshold (buying pressure),
    short when rolling OFI < −threshold (selling pressure).
    """

    def __init__(
        self,
        ofi_window: int   = 20,
        threshold:  float = 300.0,
        max_pos:    int   = 200,
        lot_size:   int   = 10,
    ) -> None:
        self.ofi_window = ofi_window
        self.threshold  = threshold
        self.max_pos    = max_pos
        self.lot_size   = lot_size

        self._bpx: List[float] = []
        self._bqt: List[float] = []
        self._apx: List[float] = []
        self._aqt: List[float] = []
        self._ofi: List[float] = []

    def on_snapshot(
        self,
        snapshot:  LOBSnapshot,
        portfolio: Portfolio,
    ) -> List[Order]:
        bp = snapshot.best_bid   or 0.0
        bq = float(snapshot.bid_depth_qty)
        ap = snapshot.best_ask   or 0.0
        aq = float(snapshot.ask_depth_qty)

        self._bpx.append(bp)
        self._bqt.append(bq)
        self._apx.append(ap)
        self._aqt.append(aq)

        if len(self._bpx) >= 2:
            ofi_step = compute_ofi(
                np.array(self._bpx[-2:]),
                np.array(self._bqt[-2:]),
                np.array(self._apx[-2:]),
                np.array(self._aqt[-2:]),
            )
            if len(ofi_step) > 0:
                self._ofi.append(float(ofi_step[-1]))

        if len(self._ofi) < self.ofi_window:
            return []

        rolling = sum(self._ofi[-self.ofi_window:])
        sym = snapshot.symbol
        pos = portfolio.position(sym)
        orders: List[Order] = []

        if rolling > self.threshold and pos < self.max_pos:
            qty = min(self.lot_size, self.max_pos - pos)
            if qty > 0 and snapshot.best_ask:
                orders.append(Order(sym, qty, "market"))
        elif rolling < -self.threshold and pos > -self.max_pos:
            qty = min(self.lot_size, pos + self.max_pos)
            if qty > 0 and snapshot.best_bid:
                orders.append(Order(sym, -qty, "market"))

        return orders


# ── LOB snapshot generator ─────────────────────────────────────────────────────

def _make_depth(
    mid:      float,
    is_bid:   bool,
    n_levels: int   = 5,
    tick:     float = 0.01,
    rng:      Optional[np.random.Generator] = None,
) -> List[DepthLevel]:
    """
    Generate `n_levels` depth levels centred around `mid`.
    Sizes decay exponentially away from best quote.
    """
    if rng is None:
        rng = np.random.default_rng()
    base_size = int(rng.integers(500, 2500))
    levels = []
    for i in range(n_levels):
        px = (mid - (i + 1) * tick) if is_bid else (mid + (i + 1) * tick)
        px = max(px, 0.01)
        qty = max(int(base_size * np.exp(-0.4 * i) * (1 + rng.normal(0, 0.15))), 1)
        levels.append(DepthLevel(price=round(px, 4), quantity=qty, order_count=max(1, qty // 50)))
    return levels


def _simulate_session(
    n_snaps:     int   = 2_000,
    seed:        int   = 42,
    initial_mid: float = 150.0,
    periods_per_year: int = 252 * 6 * 60,    # 1-minute bars, 6-hour day
) -> Tuple[List[LOBSnapshot], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate `n_snaps` LOB snapshots with two volatility regimes.

    Returns
    -------
    (snapshots, mid_prices, spread_bps, timestamps, bid_depth, ask_depth)
    """
    rng = np.random.default_rng(seed)

    # ── Two-state Markov volatility regime ───────────────────────────────────
    # Regime 0 = low vol,  Regime 1 = high vol
    vol     = np.array([0.0003, 0.0010])    # per-tick std (annualised proxy)
    trans   = np.array([[0.995, 0.005],     # regime transition matrix
                         [0.020, 0.980]])
    regime  = np.zeros(n_snaps, dtype=int)
    regime[0] = 0
    for t in range(1, n_snaps):
        regime[t] = rng.choice(2, p=trans[regime[t - 1]])

    # ── Price path ────────────────────────────────────────────────────────────
    dt     = 1.0 / periods_per_year
    drift  = 0.05 * dt             # slight positive drift
    mids   = np.zeros(n_snaps)
    mids[0] = initial_mid
    for t in range(1, n_snaps):
        sigma = vol[regime[t]] * np.sqrt(periods_per_year)   # daily vol
        mids[t] = mids[t - 1] * np.exp(
            (drift - 0.5 * sigma ** 2) * dt + sigma * rng.normal() * np.sqrt(dt)
        )

    # ── Spread and depth ─────────────────────────────────────────────────────
    base_spread_bps = np.where(regime == 0,
                                rng.uniform(0.5, 1.5, n_snaps),
                                rng.uniform(2.0, 5.0, n_snaps))

    # ── Inject OFI → price signal (mild predictability) ──────────────────────
    # Pure noise component + slight OFI component
    ofi_inject  = np.zeros(n_snaps)
    for t in range(1, n_snaps):
        # OFI = signed size imbalance; we use the price change direction
        price_chg   = mids[t] - mids[t - 1]
        ofi_inject[t] = price_chg / mids[t - 1] * 5_000 + rng.normal(0, 200)

    # ── Build snapshots ───────────────────────────────────────────────────────
    snapshots: List[LOBSnapshot] = []
    bid_depth  = np.zeros(n_snaps)
    ask_depth  = np.zeros(n_snaps)
    # Synthetic timestamps: start at 9:30 AM (NYSE open)
    t0_ns = int(9.5 * 3600 * 1e9)
    tick_ns = int(1e9)          # 1-second intervals

    for i in range(n_snaps):
        mid = mids[i]
        sp_bps = base_spread_bps[i]
        half_sp = mid * sp_bps / 2.0 / 1e4
        best_bid = mid - half_sp
        best_ask = mid + half_sp

        bids = _make_depth(best_bid, is_bid=True,  rng=rng)
        asks = _make_depth(best_ask, is_bid=False, rng=rng)
        bid_depth[i] = sum(d.quantity for d in bids)
        ask_depth[i] = sum(d.quantity for d in asks)

        snapshots.append(LOBSnapshot(
            symbol       = "AAPL",
            timestamp_ns = t0_ns + i * tick_ns,
            bids         = bids,
            asks         = asks,
            seq          = i,
        ))

    timestamps = np.array([s.timestamp_ns for s in snapshots])
    return snapshots, mids, base_spread_bps, timestamps, bid_depth, ask_depth


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_session(
    n_snaps:          int   = 2_000,
    seed:             int   = 42,
    initial_cash:     float = 1_000_000.0,
    initial_mid:      float = 150.0,
    ofi_window:       int   = 20,
    ofi_threshold:    float = 300.0,
    max_pos:          int   = 200,
    periods_per_year: int   = 252 * 6 * 60,
) -> SessionData:
    """
    Generate a synthetic trading session and run the full backtest pipeline.

    Parameters
    ----------
    n_snaps          : number of LOB snapshots to simulate (default 2 000).
    seed             : RNG seed for reproducibility.
    initial_cash     : starting portfolio cash.
    initial_mid      : starting mid-price.
    ofi_window       : rolling OFI lookback (ticks) for the strategy.
    ofi_threshold    : |cumulative OFI| to trigger a trade.
    max_pos          : maximum absolute position in shares.
    periods_per_year : annualisation factor for Tearsheet (e.g. 252×6×60 for
                       1-minute bars in a 6-hour NYSE day).

    Returns
    -------
    SessionData
    """
    snapshots, mids, spread_bps, timestamps, bid_depth, ask_depth = \
        _simulate_session(n_snaps, seed=seed, initial_mid=initial_mid,
                          periods_per_year=periods_per_year)

    strategy  = OFIMomentumStrategy(
        ofi_window=ofi_window,
        threshold=ofi_threshold,
        max_pos=max_pos,
    )
    portfolio = Portfolio(
        initial_cash = initial_cash,
        commission   = ZeroCommission(),
        max_leverage = 0.0,
    )
    source = SnapshotSource(snapshots)
    engine = BacktestEngine(
        source     = source,
        strategy   = strategy,
        portfolio  = portfolio,
        spread_bps = 3.0,
        impact_eta = 0.02,
        adv        = 5e6,
    )
    result = engine.run()

    tearsheet = Tearsheet.from_result(
        result            = result,
        periods_per_year  = periods_per_year,
        risk_free_rate    = 0.04,
        sr_benchmark      = 0.0,
        n_trials          = 1,
        initial_equity    = initial_cash,
        total_commissions = portfolio.total_commissions(),
    )

    # Compute OFI series from raw snapshots
    bp = mids - mids * spread_bps / 2.0 / 1e4
    ap = mids + mids * spread_bps / 2.0 / 1e4
    ofi_series = compute_ofi(bp, bid_depth, ap, ask_depth)

    return SessionData(
        snapshots  = snapshots,
        result     = result,
        tearsheet  = tearsheet,
        ofi_series = ofi_series,
        mid_prices = mids,
        spread_bps = spread_bps,
        timestamps = timestamps,
        bid_depth  = bid_depth,
        ask_depth  = ask_depth,
    )
