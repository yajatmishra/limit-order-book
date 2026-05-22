"""
Time-Weighted Average Price (TWAP) Execution Algorithm
=======================================================
Splits a parent order into N equal child orders spaced uniformly over a
target execution window [0, T].  The goal is to minimise the deviation
between the strategy's average fill price and the time-weighted mid-price.

Model
-----
  Schedule:  q_k = Q / N  for k = 0, 1, …, N-1
  Slice times: t_k = start + k * (duration / N)

Optionally, the scheduler can randomise slice timing within each bucket
(uniform jitter ±jitter_fraction · slice_duration) to reduce information
leakage — a common anti-gaming measure.

Tracking error  (shortfall vs. TWAP benchmark)
----------------------------------------------
  TWAP_price   = (1/T) ∫ mid(t) dt  ≈  mean of mid prices at slice times
  IS           = sign(Q) · (avg_fill − TWAP_price)   (bps)

References
----------
  Almgren & Chriss (2001). "Optimal execution of portfolio transactions."
  Journal of Risk 3(2), 5-39.

  Kissell (2014). "The Science of Algorithmic Trading." Academic Press, Ch. 3.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Schedule & result containers ──────────────────────────────────────────────

@dataclass
class TWAPSlice:
    """One child order in a TWAP schedule."""
    index:     int     # slice number 0..N-1
    time:      float   # scheduled execution time (e.g. seconds from start)
    quantity:  float   # signed quantity (+= buy, -= sell)
    filled:    float   # quantity actually filled (updated online)
    fill_price: float  # average fill price (updated online)

    @property
    def is_filled(self) -> bool:
        return abs(self.filled) >= abs(self.quantity) * (1 - 1e-9)

    def __repr__(self) -> str:
        return (f"TWAPSlice(t={self.time:.1f}, q={self.quantity:.2f}, "
                f"filled={self.filled:.2f}@{self.fill_price:.4f})")


@dataclass
class TWAPResult:
    """Completed TWAP execution summary."""
    total_quantity:   float
    avg_fill_price:   float
    twap_benchmark:   float   # mean of mid prices at slice times (if provided)
    implementation_shortfall_bps: float
    n_slices:         int
    slices:           List[TWAPSlice]
    total_notional:   float

    def __repr__(self) -> str:
        return (f"TWAPResult(Q={self.total_quantity:.2f}, "
                f"avg_fill={self.avg_fill_price:.4f}, "
                f"IS={self.implementation_shortfall_bps:.2f}bps)")


# ── TWAP scheduler ────────────────────────────────────────────────────────────

class TWAPScheduler:
    """
    Generates and tracks a TWAP execution schedule.

    Parameters
    ----------
    n_slices       : number of child orders N (default 10).
    duration       : total execution window in seconds (default 3600).
    start_time     : reference start time; slices are at start + k*interval.
    jitter         : fraction of slice interval to randomise (0 = deterministic).
                     E.g. jitter=0.2 means ±10% of interval around each slice time.
    seed           : RNG seed for jitter.

    Usage
    -----
    >>> sched = TWAPScheduler(n_slices=12, duration=3600)
    >>> schedule = sched.build(quantity=10_000)
    >>> # On each time event:
    >>> sched.fill(slice_idx=0, filled=833.3, fill_price=100.5)
    >>> result = sched.summary(mid_prices=[100.0, 100.2, ...])
    """

    def __init__(
        self,
        n_slices:   int   = 10,
        duration:   float = 3600.0,
        start_time: float = 0.0,
        jitter:     float = 0.0,
        seed:       int   = 42,
    ) -> None:
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        if duration <= 0:
            raise ValueError("duration must be > 0")
        if not (0.0 <= jitter < 1.0):
            raise ValueError("jitter must be in [0, 1)")

        self.n_slices   = n_slices
        self.duration   = duration
        self.start_time = start_time
        self.jitter     = jitter
        self.seed       = seed
        self._slices: List[TWAPSlice] = []

    # ── Schedule construction ─────────────────────────────────────────────────

    def build(self, quantity: float) -> List[TWAPSlice]:
        """
        Build the TWAP schedule for a total signed quantity.

        Returns list of TWAPSlice, one per time bucket.
        Last slice absorbs rounding residual to ensure Σq = quantity.
        """
        rng      = np.random.default_rng(self.seed)
        interval = self.duration / self.n_slices
        base_q   = quantity / self.n_slices

        slices = []
        for k in range(self.n_slices):
            t = self.start_time + k * interval
            if self.jitter > 0:
                half = interval * self.jitter * 0.5
                t   += float(rng.uniform(-half, half))

            q = base_q
            if k == self.n_slices - 1:
                # Last slice: absorb rounding
                filled_so_far = sum(s.quantity for s in slices)
                q = quantity - filled_so_far

            slices.append(TWAPSlice(
                index      = k,
                time       = t,
                quantity   = q,
                filled     = 0.0,
                fill_price = 0.0,
            ))

        self._slices = slices
        return slices

    # ── Online fill tracking ──────────────────────────────────────────────────

    def fill(self, slice_idx: int, filled: float, fill_price: float) -> None:
        """
        Record a fill for slice `slice_idx`.

        Partial fills are allowed; call fill() multiple times for the same slice.
        fill_price should be the price for the incremental fill.
        """
        s = self._slices[slice_idx]
        prev_filled  = s.filled
        new_filled   = prev_filled + filled
        # Volume-weighted average fill price
        if abs(new_filled) > 1e-12:
            s.fill_price = (prev_filled * s.fill_price + filled * fill_price) / new_filled
        s.filled = new_filled

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(
        self,
        mid_prices: Optional[List[float]] = None,
    ) -> TWAPResult:
        """
        Compute execution summary.

        Parameters
        ----------
        mid_prices : mid prices at each slice time (for IS benchmark).
                     If not provided, IS is reported as 0.
        """
        slices = self._slices
        total_q      = sum(s.quantity for s in slices)
        total_filled = sum(s.filled   for s in slices)
        total_notional = sum(abs(s.filled) * s.fill_price for s in slices)

        avg_fill = (total_notional / abs(total_filled)
                    if abs(total_filled) > 1e-12 else 0.0)

        if mid_prices and len(mid_prices) == len(slices):
            twap_bench = float(np.mean(mid_prices))
            direction  = 1.0 if total_q >= 0 else -1.0
            is_bps     = direction * (avg_fill - twap_bench) / twap_bench * 1e4
        else:
            twap_bench = avg_fill
            is_bps     = 0.0

        return TWAPResult(
            total_quantity   = total_q,
            avg_fill_price   = avg_fill,
            twap_benchmark   = twap_bench,
            implementation_shortfall_bps = is_bps,
            n_slices         = len(slices),
            slices           = list(slices),
            total_notional   = total_notional,
        )

    @property
    def slices(self) -> List[TWAPSlice]:
        return self._slices

    def remaining_quantity(self) -> float:
        """Total unfilled quantity remaining."""
        return sum(s.quantity - s.filled for s in self._slices)

    def __repr__(self) -> str:
        return (f"TWAPScheduler(n_slices={self.n_slices}, "
                f"duration={self.duration:.0f}s, jitter={self.jitter})")


# ── Batch simulator ───────────────────────────────────────────────────────────

def simulate_twap(
    quantity:    float,
    mid_prices:  np.ndarray,
    n_slices:    Optional[int] = None,
    spread_bps:  float = 5.0,
    seed:        int   = 42,
) -> TWAPResult:
    """
    Simulate TWAP execution against a mid-price series.

    Each slice executes at mid ± half-spread (buy crosses ask, sell crosses bid).

    Parameters
    ----------
    quantity    : total signed order quantity.
    mid_prices  : time series of mid prices, shape (T,).
    n_slices    : slices; defaults to len(mid_prices).
    spread_bps  : bid-ask spread in basis points.
    """
    mids = np.asarray(mid_prices, dtype=float)
    T    = len(mids)
    N    = n_slices or T

    sched = TWAPScheduler(n_slices=N, duration=float(T))
    slices = sched.build(quantity)

    half_spread_frac = spread_bps / 2.0 / 1e4
    direction        = 1.0 if quantity >= 0 else -1.0

    # Map each slice to the closest mid index
    for s in slices:
        idx  = min(int(s.time), T - 1)
        mid  = mids[idx]
        fill = mid * (1.0 + direction * half_spread_frac)
        sched.fill(s.index, s.quantity, fill)

    return sched.summary(mid_prices=mids[[
        min(int(s.time), T - 1) for s in slices
    ]].tolist())
