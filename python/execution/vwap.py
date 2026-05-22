"""
Volume-Weighted Average Price (VWAP) Execution Algorithm
=========================================================
Schedules child orders proportionally to the expected intraday volume profile
so that the strategy's average fill price tracks the session VWAP.

Volume profile
--------------
Two options:
  1. Historical profile: pass a vector of expected volume fractions v_k ≥ 0
     with Σ v_k = 1.  Slice k receives q_k = Q * v_k.
  2. U-shaped default: empirical intraday equity volume follows a U-curve
     (high at open and close, low at midday).  The default uses a simple
     parametric U:  v(t) = a + b·(2t/T − 1)² , normalised to sum to 1.

Performance benchmark
---------------------
VWAP = Σ_t (price_t * volume_t) / Σ_t volume_t

Implementation shortfall:
  IS (bps) = sign(Q) · (avg_fill − VWAP) / VWAP · 1e4

A negative IS means the algorithm filled *better* than the VWAP benchmark
(favourable for buys if avg_fill < VWAP).

References
----------
  Madhavan (2002). "VWAP strategies." Trading, Spring 2002.
  Berkowitz, Logue & Noser (1988). "The total cost of transactions on the NYSE."
  Journal of Finance 43(1), 97-112.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class VWAPSlice:
    """One child order in a VWAP schedule."""
    index:        int
    time:         float
    quantity:     float    # target quantity for this bucket
    vol_fraction: float    # expected volume fraction v_k
    filled:       float
    fill_price:   float

    def __repr__(self) -> str:
        return (f"VWAPSlice(t={self.time:.1f}, q={self.quantity:.2f}, "
                f"v={self.vol_fraction:.3f}, fill={self.fill_price:.4f})")


@dataclass
class VWAPResult:
    """Completed VWAP execution summary."""
    total_quantity:   float
    avg_fill_price:   float
    vwap_benchmark:   float
    implementation_shortfall_bps: float
    n_slices:         int
    slices:           List[VWAPSlice]
    total_notional:   float

    def __repr__(self) -> str:
        return (f"VWAPResult(Q={self.total_quantity:.2f}, "
                f"avg_fill={self.avg_fill_price:.4f}, "
                f"IS={self.implementation_shortfall_bps:.2f}bps)")


# ── Volume profile helpers ────────────────────────────────────────────────────

def u_shaped_profile(n: int, alpha: float = 2.0) -> np.ndarray:
    """
    Generate a U-shaped intraday volume profile with `n` buckets.

    v(k) ∝ 1 + alpha · (2k/(n-1) − 1)²   for k = 0..n-1.

    Parameters
    ----------
    n     : number of time buckets.
    alpha : U-shape steepness (0 = flat, higher = more pronounced U).

    Returns normalised array summing to 1.
    """
    if n < 2:
        return np.array([1.0])
    k   = np.linspace(0.0, 1.0, n)
    v   = 1.0 + alpha * (2.0 * k - 1.0) ** 2
    return v / v.sum()


def flat_profile(n: int) -> np.ndarray:
    """Uniform (TWAP-like) volume profile."""
    return np.full(n, 1.0 / n)


# ── VWAP scheduler ────────────────────────────────────────────────────────────

class VWAPScheduler:
    """
    Generates and tracks a VWAP execution schedule.

    Parameters
    ----------
    n_slices   : number of time buckets (default 12).
    duration   : total execution window in seconds (default 3600).
    start_time : reference start time.
    profile    : volume fraction array of shape (n_slices,), summing to 1.
                 If None, a U-shaped default profile is used.

    Usage
    -----
    >>> sched = VWAPScheduler(n_slices=12, duration=23400)  # 6.5-hour session
    >>> schedule = sched.build(quantity=-5000)   # sell 5000 shares
    >>> # At each bucket:
    >>> sched.fill(k, filled=418.0, fill_price=99.8)
    >>> result = sched.summary(prices=mids, volumes=vol_series)
    """

    def __init__(
        self,
        n_slices:   int   = 12,
        duration:   float = 3600.0,
        start_time: float = 0.0,
        profile:    Optional[np.ndarray] = None,
    ) -> None:
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        if duration <= 0:
            raise ValueError("duration must be > 0")

        self.n_slices   = n_slices
        self.duration   = duration
        self.start_time = start_time

        if profile is not None:
            p = np.asarray(profile, dtype=float)
            if len(p) != n_slices:
                raise ValueError(f"profile length {len(p)} != n_slices {n_slices}")
            if np.any(p < 0):
                raise ValueError("profile must be non-negative")
            total = p.sum()
            if total < 1e-10:
                raise ValueError("profile must not be all zeros")
            self._profile = p / total
        else:
            self._profile = u_shaped_profile(n_slices)

        self._slices: List[VWAPSlice] = []

    def build(self, quantity: float) -> List[VWAPSlice]:
        """
        Build the VWAP schedule for total signed quantity.

        Last slice absorbs rounding residual.
        """
        interval = self.duration / self.n_slices
        slices   = []

        for k in range(self.n_slices):
            t = self.start_time + k * interval
            q = quantity * self._profile[k]
            if k == self.n_slices - 1:
                q = quantity - sum(s.quantity for s in slices)

            slices.append(VWAPSlice(
                index        = k,
                time         = t,
                quantity     = q,
                vol_fraction = float(self._profile[k]),
                filled       = 0.0,
                fill_price   = 0.0,
            ))

        self._slices = slices
        return slices

    def fill(self, slice_idx: int, filled: float, fill_price: float) -> None:
        """Record a fill (supports partial fills via VWAP of fills)."""
        s            = self._slices[slice_idx]
        prev_filled  = s.filled
        new_filled   = prev_filled + filled
        if abs(new_filled) > 1e-12:
            s.fill_price = (prev_filled * s.fill_price + filled * fill_price) / new_filled
        s.filled = new_filled

    def summary(
        self,
        prices:  Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
    ) -> VWAPResult:
        """
        Compute VWAP execution summary.

        Parameters
        ----------
        prices  : mid prices at each bucket (shape n_slices) for benchmark.
        volumes : realised volumes at each bucket (shape n_slices) for VWAP.
        """
        slices = self._slices
        total_q  = sum(s.quantity for s in slices)
        total_notional = sum(abs(s.filled) * s.fill_price for s in slices
                             if abs(s.filled) > 1e-12)
        total_filled   = sum(s.filled for s in slices)
        avg_fill = (total_notional / abs(total_filled)
                    if abs(total_filled) > 1e-12 else 0.0)

        # VWAP benchmark
        if prices is not None and volumes is not None:
            p = np.asarray(prices,  dtype=float)[:len(slices)]
            v = np.asarray(volumes, dtype=float)[:len(slices)]
            vwap_bench = float(np.dot(p, v) / (v.sum() + 1e-30))
        elif prices is not None:
            p = np.asarray(prices, dtype=float)[:len(slices)]
            # No volume: weight by expected profile
            vwap_bench = float(np.dot(p, self._profile))
        else:
            vwap_bench = avg_fill

        direction = 1.0 if total_q >= 0 else -1.0
        is_bps = (direction * (avg_fill - vwap_bench) / max(vwap_bench, 1e-12) * 1e4
                  if abs(avg_fill) > 1e-12 else 0.0)

        return VWAPResult(
            total_quantity   = total_q,
            avg_fill_price   = avg_fill,
            vwap_benchmark   = vwap_bench,
            implementation_shortfall_bps = is_bps,
            n_slices         = len(slices),
            slices           = list(slices),
            total_notional   = total_notional,
        )

    @property
    def profile(self) -> np.ndarray:
        """Volume fraction profile."""
        return self._profile.copy()

    @property
    def slices(self) -> List[VWAPSlice]:
        """The current slice list (populated after build())."""
        return self._slices

    def remaining_quantity(self) -> float:
        return sum(s.quantity - s.filled for s in self._slices)

    def __repr__(self) -> str:
        return f"VWAPScheduler(n_slices={self.n_slices}, duration={self.duration:.0f}s)"


# ── Batch simulator ───────────────────────────────────────────────────────────

def simulate_vwap(
    quantity:    float,
    mid_prices:  np.ndarray,
    volumes:     Optional[np.ndarray] = None,
    spread_bps:  float = 5.0,
    profile:     Optional[np.ndarray] = None,
) -> VWAPResult:
    """
    Simulate VWAP execution against a mid-price / volume series.

    Each slice executes at mid ± half-spread.
    """
    mids = np.asarray(mid_prices, dtype=float)
    T    = len(mids)
    vols = (np.asarray(volumes, dtype=float) if volumes is not None
            else np.ones(T))

    sched = VWAPScheduler(n_slices=T, duration=float(T),
                          profile=(vols / vols.sum()) if profile is None else profile)
    sched.build(quantity)

    half_spread_frac = spread_bps / 2.0 / 1e4
    direction        = 1.0 if quantity >= 0 else -1.0

    for s in sched.slices:
        idx  = min(int(s.time), T - 1)
        fill = mids[idx] * (1.0 + direction * half_spread_frac)
        sched.fill(s.index, s.quantity, fill)

    return sched.summary(prices=mids, volumes=vols)
