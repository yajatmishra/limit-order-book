"""
Strategy Lab
============
A small comparison harness that runs several trading strategies on the same
synthetic market and reports their performance side by side.

The market here is richer than the dashboard's replay generator. It embeds three
pieces of structure that the strategies are designed to exploit:

  1. A slow, persistent trend factor (an AR(1) drift). This makes returns mildly
     autocorrelated, so a moving-average crossover can profit.
  2. A fast, mean-reverting microstructure component added on top of the efficient
     price. A short-window z-score strategy can fade it.
  3. An order-book imbalance that leads the trend, so a smoothed-imbalance
     strategy can position ahead of the move.

This is a synthetic market with known, injected signals. Good results here
demonstrate that the backtest and reporting pipeline works end to end. They are
not evidence of real-world alpha.

Public API
----------
run_comparison(...) -> ComparisonResult
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(_HERE, "..")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from backtester.engine import (
    BacktestEngine,
    DepthLevel,
    EngineResult,
    LOBSnapshot,
    Order,
    SnapshotSource,
    Strategy,
)
from backtester.portfolio import Portfolio, ZeroCommission
from backtester.tearsheet import Tearsheet


# ════════════════════════════════════════════════════════════════════════════════
# Synthetic market with embedded structure
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Market:
    """A generated market: snapshots plus the underlying series."""
    snapshots:  List[LOBSnapshot]
    mids:       np.ndarray   # observed mid price per snapshot
    imbalance:  np.ndarray   # book imbalance in [-1, 1] per snapshot
    periods_per_year: int


def generate_market(
    n_snaps:          int   = 2_000,
    seed:             int   = 7,
    base_price:       float = 150.0,
    mu0:              float = 3.0e-5,  # baseline upward drift per tick
    phi_g:            float = 0.998,   # trend persistence (slow factor)
    sigma_g:          float = 8.0e-6,  # trend innovation
    sigma_e:          float = 8.0e-5,  # efficient-price noise per tick
    phi_u:            float = 0.70,    # microstructure reversion (fast factor)
    sigma_u:          float = 1.8e-3,  # microstructure innovation
    imb_gain:         float = 1.8,     # how strongly imbalance tracks the trend
    imb_noise:        float = 0.18,    # noise on the imbalance signal
    periods_per_year: int   = 252 * 6 * 60,
) -> Market:
    """Generate a synthetic order-book session with exploitable structure."""
    rng = np.random.default_rng(seed)

    # 1. Persistent trend factor g_t (AR(1)).
    g = np.zeros(n_snaps)
    for t in range(1, n_snaps):
        g[t] = phi_g * g[t - 1] + sigma_g * rng.standard_normal()

    # 2. Efficient log price = cumulative (drift + trend + noise).
    eff_ret = mu0 + g + sigma_e * rng.standard_normal(n_snaps)
    eff_logp = np.cumsum(eff_ret)

    # 3. Transient, fast-reverting microstructure deviation u_t (AR(1)).
    u = np.zeros(n_snaps)
    for t in range(1, n_snaps):
        u[t] = phi_u * u[t - 1] + sigma_u * rng.standard_normal()

    obs_logp = eff_logp + u
    mids = base_price * np.exp(obs_logp)

    # 4. Book imbalance that leads the trend. g is standardized so the gain has a
    #    stable meaning regardless of sigma_g.
    g_std = sigma_g / np.sqrt(max(1.0 - phi_g ** 2, 1e-9))
    g_norm = g / max(g_std, 1e-12)
    imbalance = np.tanh(imb_gain * g_norm) + imb_noise * rng.standard_normal(n_snaps)
    imbalance = np.clip(imbalance, -0.95, 0.95)

    # 5. Build five-level depth on each side, split by the imbalance.
    t0_ns = int(9.5 * 3600 * 1e9)
    tick_ns = int(1e9)
    spread_bps = 1.0
    snapshots: List[LOBSnapshot] = []
    for i in range(n_snaps):
        mid = float(mids[i])
        half_sp = mid * spread_bps / 2.0 / 1e4
        best_bid = mid - half_sp
        best_ask = mid + half_sp

        total = int(rng.integers(4_000, 8_000))
        frac = (imbalance[i] + 1.0) / 2.0          # 0..1, share on the bid
        bid_total = max(int(total * frac), 1)
        ask_total = max(total - bid_total, 1)

        bids = _depth_levels(best_bid, bid_total, is_bid=True, rng=rng)
        asks = _depth_levels(best_ask, ask_total, is_bid=False, rng=rng)

        snapshots.append(LOBSnapshot(
            symbol       = "SYNTH",
            timestamp_ns = t0_ns + i * tick_ns,
            bids         = bids,
            asks         = asks,
            seq          = i,
        ))

    return Market(snapshots=snapshots, mids=mids, imbalance=imbalance,
                  periods_per_year=periods_per_year)


def _depth_levels(
    best:     float,
    total:    int,
    is_bid:   bool,
    n_levels: int = 5,
    tick:     float = 0.01,
    rng:      np.random.Generator | None = None,
) -> List[DepthLevel]:
    """Spread `total` quantity over `n_levels`, decaying away from the best."""
    if rng is None:
        rng = np.random.default_rng()
    weights = np.exp(-0.4 * np.arange(n_levels))
    weights = weights / weights.sum()
    levels = []
    for i in range(n_levels):
        px = (best - i * tick) if is_bid else (best + i * tick)
        px = max(px, 0.01)
        qty = max(int(total * weights[i]), 1)
        levels.append(DepthLevel(price=round(px, 4), quantity=qty,
                                 order_count=max(1, qty // 50)))
    return levels


# ════════════════════════════════════════════════════════════════════════════════
# Strategies
# ════════════════════════════════════════════════════════════════════════════════

def _rebalance(symbol: str, target: int, pos: int, lot: int) -> List[Order]:
    """Emit at most one market order moving `pos` toward `target`."""
    delta = target - pos
    if abs(delta) < lot:
        return []
    step = int(np.clip(delta, -abs(delta), abs(delta)))
    return [Order(symbol, step, "market")]


class BuyHoldStrategy(Strategy):
    """Buy a fixed long position on the first snapshot and hold it."""

    def __init__(self, notional_fraction: float = 1.0, initial_cash: float = 1_000_000.0):
        self.target_notional = notional_fraction * initial_cash
        self._done = False

    def on_snapshot(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> List[Order]:
        if self._done or snapshot.best_ask is None:
            return []
        self._done = True
        shares = int(self.target_notional / snapshot.best_ask)
        return [Order(snapshot.symbol, shares, "market")] if shares > 0 else []


class MACrossoverStrategy(Strategy):
    """Go long when a fast moving average is above a slow one, short otherwise."""

    def __init__(self, fast: int = 40, slow: int = 200, max_notional: float = 1_000_000.0,
                 lot_notional: float = 100_000.0):
        self.fast = fast
        self.slow = slow
        self.max_notional = max_notional
        self.lot_notional = lot_notional
        self._mids: List[float] = []

    def on_snapshot(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> List[Order]:
        mid = snapshot.mid
        if mid is None:
            return []
        self._mids.append(mid)
        if len(self._mids) < self.slow:
            return []
        fast_ma = np.mean(self._mids[-self.fast:])
        slow_ma = np.mean(self._mids[-self.slow:])
        direction = 1 if fast_ma > slow_ma else -1
        target = int(direction * self.max_notional / mid)
        pos = portfolio.position(snapshot.symbol)
        lot = max(1, int(self.lot_notional / mid))
        return _rebalance(snapshot.symbol, target, pos, lot)


class MeanReversionStrategy(Strategy):
    """Fade short-term deviations of the mid from its rolling mean."""

    def __init__(self, window: int = 15, entry_z: float = 1.0, exit_z: float = 0.3,
                 max_notional: float = 1_000_000.0):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_notional = max_notional
        self._mids: List[float] = []

    def on_snapshot(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> List[Order]:
        mid = snapshot.mid
        if mid is None:
            return []
        self._mids.append(mid)
        if len(self._mids) < self.window:
            return []
        window = np.asarray(self._mids[-self.window:])
        mu = window.mean()
        sd = window.std()
        if sd <= 0:
            return []
        z = (mid - mu) / sd
        pos = portfolio.position(snapshot.symbol)
        full = int(self.max_notional / mid)

        target = pos
        if z >= self.entry_z:
            target = -full          # price rich, short it
        elif z <= -self.entry_z:
            target = +full          # price cheap, buy it
        elif abs(z) <= self.exit_z:
            target = 0              # reverted, flatten
        lot = max(1, full // 10)
        return _rebalance(snapshot.symbol, target, pos, lot)


class OFIMomentumStrategy(Strategy):
    """Trade in the direction of smoothed order-book imbalance."""

    def __init__(self, window: int = 15, entry: float = 0.15,
                 max_notional: float = 1_000_000.0, lot_notional: float = 100_000.0):
        self.window = window
        self.entry = entry
        self.max_notional = max_notional
        self.lot_notional = lot_notional
        self._imb: List[float] = []

    def on_snapshot(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> List[Order]:
        mid = snapshot.mid
        if mid is None:
            return []
        bd = float(snapshot.bid_depth_qty)
        ad = float(snapshot.ask_depth_qty)
        denom = bd + ad
        imb = (bd - ad) / denom if denom > 0 else 0.0
        self._imb.append(imb)
        if len(self._imb) < self.window:
            return []
        smooth = float(np.mean(self._imb[-self.window:]))
        pos = portfolio.position(snapshot.symbol)
        full = int(self.max_notional / mid)

        target = pos
        if smooth > self.entry:
            target = +full
        elif smooth < -self.entry:
            target = -full
        lot = max(1, int(self.lot_notional / mid))
        return _rebalance(snapshot.symbol, target, pos, lot)


# ════════════════════════════════════════════════════════════════════════════════
# Comparison runner
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyRun:
    name:      str
    color:     str
    result:    EngineResult
    tearsheet: Tearsheet


@dataclass
class ComparisonResult:
    market: Market
    runs:   List[StrategyRun]


# Palette aligned with the dashboard theme.
_COLORS = {
    "Buy & Hold":     "#94a3b8",   # slate-400 (benchmark)
    "MA Crossover":   "#fbbf24",   # amber-400
    "Mean Reversion": "#22c55e",   # green-500
    "OFI Momentum":   "#93c5fd",   # blue-300
}


def _run_one(name: str, strategy: Strategy, market: Market,
             initial_cash: float) -> StrategyRun:
    portfolio = Portfolio(initial_cash=initial_cash, commission=ZeroCommission(),
                          max_leverage=0.0)
    engine = BacktestEngine(
        source     = SnapshotSource(market.snapshots),
        strategy   = strategy,
        portfolio  = portfolio,
        spread_bps = 1.0,
        impact_eta = 0.02,
        adv        = 5e6,
    )
    result = engine.run()
    tearsheet = Tearsheet.from_result(
        result            = result,
        periods_per_year  = market.periods_per_year,
        risk_free_rate    = 0.04,
        sr_benchmark      = 0.0,
        n_trials          = 1,
        initial_equity    = initial_cash,
        total_commissions = portfolio.total_commissions(),
    )
    return StrategyRun(name=name, color=_COLORS.get(name, "#e2e8f0"),
                       result=result, tearsheet=tearsheet)


def run_comparison(
    n_snaps:      int   = 2_000,
    seed:         int   = 28,
    initial_cash: float = 1_000_000.0,
) -> ComparisonResult:
    """Generate one market and run every strategy on it."""
    market = generate_market(n_snaps=n_snaps, seed=seed)
    cash = initial_cash

    strategies = [
        ("Buy & Hold",     BuyHoldStrategy(notional_fraction=1.0, initial_cash=cash)),
        ("MA Crossover",   MACrossoverStrategy(max_notional=cash)),
        ("Mean Reversion", MeanReversionStrategy(max_notional=cash)),
        ("OFI Momentum",   OFIMomentumStrategy(max_notional=cash)),
    ]
    runs = [_run_one(name, strat, market, cash) for name, strat in strategies]
    return ComparisonResult(market=market, runs=runs)


if __name__ == "__main__":
    comp = run_comparison()
    header = f"{'Strategy':<16}{'Return':>10}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'Fills':>8}"
    print(header)
    print("-" * len(header))
    for run in comp.runs:
        r = run.tearsheet.report
        print(f"{run.name:<16}"
              f"{r.total_return:>9.2%} "
              f"{r.sharpe:>8.2f} "
              f"{r.sortino:>8.2f} "
              f"{r.max_drawdown:>8.2%} "
              f"{run.tearsheet.n_fills:>7d}")
