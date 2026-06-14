"""
Strategy Lab
============
A small comparison harness that runs several trading strategies on the same
synthetic market and reports their performance side by side.

This is deliberately written to avoid the two things that make backtest Sharpe
ratios meaningless:

  1. Frequency-inflated annualization. The bars here are daily, and the Sharpe
     is annualized by sqrt(252). Annualizing an intraday Sharpe by the number of
     intraday bars per year (tens of thousands) is what produces the fantasy
     Sharpe ratios of 20+ that you sometimes see.
  2. Same-bar look-ahead. A signal computed from data up to and including bar t
     is executed at bar t+1 (a one-bar lag), so no strategy trades on a price it
     used to make the decision.

The market embeds modest, noisy structure: a slow trend (so a moving-average
crossover has something to follow), a fast mean-reverting component (so a
z-score strategy can fade it), and a weak, noisy book imbalance that leads
returns (so an order-flow strategy has a small edge). The signal-to-noise is
intentionally low and the trading costs are real, so the resulting Sharpe ratios
land in the believable 1-3 range.

It is still a synthetic market whose data-generating process is known here. Good
results demonstrate the pipeline. They are not evidence of real-world alpha.

Public API
----------
run_comparison(...)      -> ComparisonResult
evaluate_out_of_sample() -> dict[str, dict[str, float]]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

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

# Daily bars: 252 trading days per year. Sharpe is annualized by sqrt(252).
TRADING_DAYS = 252

# Held-out seeds used for the reported, out-of-sample numbers. The strategy
# parameters below were fixed without reference to these seeds.
DEFAULT_SEED = 206


# ════════════════════════════════════════════════════════════════════════════════
# Synthetic market with embedded structure
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Market:
    """A generated market: daily snapshots plus the underlying series."""
    snapshots:  List[LOBSnapshot]
    mids:       np.ndarray   # observed mid price per bar
    imbalance:  np.ndarray   # book imbalance in [-1, 1] per bar
    periods_per_year: int


def generate_market(
    n_snaps:          int   = 1_500,    # ~6 years of daily bars
    seed:             int   = DEFAULT_SEED,
    base_price:       float = 100.0,
    mu0:              float = 2.0e-4,    # baseline drift per day (~5%/yr)
    phi_g:            float = 0.99,      # trend persistence (very slow, for MA)
    sigma_g:          float = 2.5e-4,    # trend innovation
    sigma_e:          float = 1.5e-3,    # efficient-price (permanent) noise per day
    phi_u:            float = 0.80,      # microstructure reversion (for MR)
    sigma_u:          float = 5.0e-3,    # microstructure innovation
    phi_o:            float = 0.85,      # order-flow factor persistence (for OFI)
    c_of:             float = 2.0e-3,    # how much order flow predicts the next return
    imb_gain:         float = 0.8,       # how cleanly the book shows the order-flow factor
    imb_noise:        float = 0.70,      # noise on the imbalance signal (high)
    periods_per_year: int   = TRADING_DAYS,
) -> Market:
    """Generate a synthetic daily session with modest, noisy exploitable structure."""
    rng = np.random.default_rng(seed)

    # 1. Slow trend factor g_t (AR(1)). The moving-average crossover follows it.
    g = np.zeros(n_snaps)
    for t in range(1, n_snaps):
        g[t] = phi_g * g[t - 1] + sigma_g * rng.standard_normal()

    # 2. Order-flow factor o_t (AR(1)), standardized to unit variance. It is
    #    visible (noisily) in the book imbalance and predicts the *next* bar's
    #    return, which is the edge the OFI strategy harvests.
    o = np.zeros(n_snaps)
    for t in range(1, n_snaps):
        o[t] = phi_o * o[t - 1] + rng.standard_normal()
    o = o / max(o.std(), 1e-9)
    of_pred = np.zeros(n_snaps)
    of_pred[1:] = c_of * o[:-1]              # o_{t-1} predicts the return at t

    # 3. Efficient log price = cumulative (drift + trend + order flow + noise).
    eff_ret = mu0 + g + of_pred + sigma_e * rng.standard_normal(n_snaps)
    eff_logp = np.cumsum(eff_ret)

    # 4. Transient, fast-reverting deviation u_t (AR(1)). The mean-reversion
    #    strategy fades it after detrending.
    u = np.zeros(n_snaps)
    for t in range(1, n_snaps):
        u[t] = phi_u * u[t - 1] + sigma_u * rng.standard_normal()

    obs_logp = eff_logp + u
    mids = base_price * np.exp(obs_logp)

    # 5. Book imbalance reflects the current order-flow factor, noisily.
    imbalance = np.tanh(imb_gain * o) + imb_noise * rng.standard_normal(n_snaps)
    imbalance = np.clip(imbalance, -0.95, 0.95)

    # 6. Build five-level depth on each side, split by the imbalance.
    t0_ns = int(9.5 * 3600 * 1e9)
    day_ns = int(24 * 3600 * 1e9)
    spread_bps = 4.0
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
            timestamp_ns = t0_ns + i * day_ns,
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
    rng:      Optional[np.random.Generator] = None,
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
    """Emit a market order moving `pos` toward `target` if the move clears `lot`."""
    delta = target - pos
    if abs(delta) < lot:
        return []
    return [Order(symbol, int(delta), "market")]


class LaggedStrategy(Strategy):
    """
    Base class that enforces a one-bar execution lag.

    A subclass computes a desired target position from data up to and including
    the current bar via `compute_target`. That target is only acted on at the
    *next* bar, so the order never fills on a price the signal already used. This
    removes same-bar look-ahead.
    """

    def __init__(self, max_notional: float, lot_notional: float):
        self.max_notional = max_notional
        self.lot_notional = lot_notional
        self._target: Optional[int] = None

    def compute_target(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> Optional[int]:
        raise NotImplementedError

    def on_snapshot(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> List[Order]:
        mid = snapshot.mid
        orders: List[Order] = []
        if self._target is not None and mid is not None:
            lot = max(1, int(self.lot_notional / mid))
            orders = _rebalance(snapshot.symbol, self._target,
                                portfolio.position(snapshot.symbol), lot)
        # Compute the target for the next bar from information available now.
        self._target = self.compute_target(snapshot, portfolio)
        return orders

    def _full(self, mid: float) -> int:
        return int(self.max_notional / mid)


class BuyHoldStrategy(LaggedStrategy):
    """Hold a fixed long position. The passive benchmark."""

    def __init__(self, max_notional: float = 1_000_000.0):
        super().__init__(max_notional, lot_notional=max_notional)

    def compute_target(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> Optional[int]:
        mid = snapshot.mid
        return self._full(mid) if mid else None


class MACrossoverStrategy(LaggedStrategy):
    """Long when a fast moving average is above a slow one, short otherwise."""

    def __init__(self, fast: int = 20, slow: int = 80, max_notional: float = 1_000_000.0,
                 lot_notional: float = 150_000.0):
        super().__init__(max_notional, lot_notional)
        self.fast = fast
        self.slow = slow
        self._mids: List[float] = []

    def compute_target(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> Optional[int]:
        mid = snapshot.mid
        if mid is None:
            return self._target
        self._mids.append(mid)
        if len(self._mids) < self.slow:
            return None
        fast_ma = np.mean(self._mids[-self.fast:])
        slow_ma = np.mean(self._mids[-self.slow:])
        direction = 1 if fast_ma > slow_ma else -1
        return int(direction * self._full(mid))


class MeanReversionStrategy(LaggedStrategy):
    """
    Fade deviations of the mid from a slow EMA. Detrending against the EMA keeps
    the slow trend from biasing the z-score, so the strategy trades the fast
    mean-reverting component rather than fighting the trend.
    """

    def __init__(self, trend_span: int = 12, z_window: int = 6, entry_z: float = 1.0,
                 exit_z: float = 0.3, max_notional: float = 1_000_000.0,
                 lot_notional: float = 150_000.0):
        super().__init__(max_notional, lot_notional)
        self.alpha = 2.0 / (trend_span + 1.0)
        self.z_window = z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._ema: Optional[float] = None
        self._resid: List[float] = []

    def compute_target(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> Optional[int]:
        mid = snapshot.mid
        if mid is None:
            return self._target
        self._ema = mid if self._ema is None else self.alpha * mid + (1 - self.alpha) * self._ema
        resid = mid - self._ema
        self._resid.append(resid)
        if len(self._resid) < self.z_window:
            return None
        sd = float(np.std(self._resid[-self.z_window:]))
        if sd <= 0:
            return self._target
        z = resid / sd
        full = self._full(mid)
        prev = self._target if self._target is not None else 0
        if z >= self.entry_z:
            return -full          # rich vs trend, short it
        if z <= -self.entry_z:
            return +full          # cheap vs trend, buy it
        if abs(z) <= self.exit_z:
            return 0              # reverted, flatten
        return prev               # otherwise hold


class OFIMomentumStrategy(LaggedStrategy):
    """Trade in the direction of smoothed order-book imbalance."""

    def __init__(self, window: int = 3, entry: float = 0.05,
                 max_notional: float = 1_000_000.0, lot_notional: float = 150_000.0):
        super().__init__(max_notional, lot_notional)
        self.window = window
        self.entry = entry
        self._imb: List[float] = []

    def compute_target(self, snapshot: LOBSnapshot, portfolio: Portfolio) -> Optional[int]:
        mid = snapshot.mid
        if mid is None:
            return self._target
        bd = float(snapshot.bid_depth_qty)
        ad = float(snapshot.ask_depth_qty)
        denom = bd + ad
        imb = (bd - ad) / denom if denom > 0 else 0.0
        self._imb.append(imb)
        if len(self._imb) < self.window:
            return None
        smooth = float(np.mean(self._imb[-self.window:]))
        full = self._full(mid)
        prev = self._target if self._target is not None else 0
        if smooth > self.entry:
            return +full
        if smooth < -self.entry:
            return -full
        return prev


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

# Round-trip trading cost assumptions, applied by the engine on every fill.
_SPREAD_BPS = 4.0
_IMPACT_ETA = 0.05
_ADV        = 2e6


# Each strategy targets at most this fraction of capital as gross notional. Sharpe
# is invariant to this (returns scale linearly), but keeping it well below 1.0
# prevents a short position from ever driving equity negative on an adverse path.
NOTIONAL_FRACTION = 0.4


def _make_strategies(cash: float):
    notional = NOTIONAL_FRACTION * cash
    return [
        ("Buy & Hold",     BuyHoldStrategy(max_notional=notional)),
        ("MA Crossover",   MACrossoverStrategy(max_notional=notional)),
        ("Mean Reversion", MeanReversionStrategy(max_notional=notional)),
        ("OFI Momentum",   OFIMomentumStrategy(max_notional=notional)),
    ]


def _run_one(name: str, strategy: Strategy, market: Market,
             initial_cash: float) -> StrategyRun:
    portfolio = Portfolio(initial_cash=initial_cash, commission=ZeroCommission(),
                          max_leverage=0.0)
    engine = BacktestEngine(
        source     = SnapshotSource(market.snapshots),
        strategy   = strategy,
        portfolio  = portfolio,
        spread_bps = _SPREAD_BPS,
        impact_eta = _IMPACT_ETA,
        adv        = _ADV,
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
    n_snaps:      int   = 1_500,
    seed:         int   = DEFAULT_SEED,
    initial_cash: float = 1_000_000.0,
) -> ComparisonResult:
    """Generate one market and run every strategy on it."""
    market = generate_market(n_snaps=n_snaps, seed=seed)
    runs = [_run_one(name, strat, market, initial_cash)
            for name, strat in _make_strategies(initial_cash)]
    return ComparisonResult(market=market, runs=runs)


def evaluate_out_of_sample(seeds: range = range(200, 220),
                           initial_cash: float = 1_000_000.0) -> dict:
    """Run every strategy across a set of held-out seeds and summarise Sharpe."""
    sharpes: dict = {}
    rets: dict = {}
    for seed in seeds:
        comp = run_comparison(seed=seed, initial_cash=initial_cash)
        for run in comp.runs:
            sharpes.setdefault(run.name, []).append(run.tearsheet.report.sharpe)
            rets.setdefault(run.name, []).append(run.tearsheet.report.ann_return)
    summary = {}
    for name in sharpes:
        s = np.asarray(sharpes[name]); a = np.asarray(rets[name])
        summary[name] = {
            "sharpe_mean": float(s.mean()), "sharpe_std": float(s.std()),
            "sharpe_min": float(s.min()), "sharpe_max": float(s.max()),
            "ann_return_mean": float(a.mean()),
        }
    return summary


if __name__ == "__main__":
    comp = run_comparison()
    header = f"{'Strategy':<16}{'Ann.Ret':>10}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'Fills':>8}"
    print(f"Headline seed {DEFAULT_SEED} (daily bars, annualized by sqrt(252)):")
    print(header)
    print("-" * len(header))
    for run in comp.runs:
        r = run.tearsheet.report
        print(f"{run.name:<16}"
              f"{r.ann_return:>9.2%} "
              f"{r.sharpe:>8.2f} "
              f"{r.sortino:>8.2f} "
              f"{r.max_drawdown:>8.2%} "
              f"{run.tearsheet.n_fills:>7d}")

    print("\nOut-of-sample (seeds 200-219), mean Sharpe +/- std:")
    for name, s in evaluate_out_of_sample().items():
        print(f"  {name:<16} {s['sharpe_mean']:5.2f} +/- {s['sharpe_std']:.2f} "
              f"(min {s['sharpe_min']:.2f}, max {s['sharpe_max']:.2f})")
