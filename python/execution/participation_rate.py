"""
Participation Rate (Percent-of-Volume / POV) Execution Algorithm
=================================================================
Trades a fixed fraction `rate` of observed market volume each period until
the parent order is fully filled.  The algorithm adapts naturally to
variations in liquidity — trading more when markets are active and less
when they are thin.

Model
-----
  At time t:  q_t = min(rate · V_t,  remaining_quantity)

  where V_t is the realised (or forecast) market volume in period t.

Overshoot guard: if the cumulative fill would exceed the parent order, only
the residual is executed.

Slippage model (optional)
-------------------------
If market volume and intraday impact are relevant:
  fill_price_t = mid_t · (1 ± half_spread ± impact)
  impact = eta · σ · sqrt(q_t / V_t)

The participation rate is the primary lever for controlling market impact:
  rate ↑  →  faster execution, higher impact
  rate ↓  →  slower execution, lower impact, more timing risk

References
----------
  Almgren & Chriss (2001). "Optimal execution of portfolio transactions."
  Bertsimas & Lo (1998). "Optimal control of execution costs."
  Journal of Financial Markets 1(1), 1-50.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class POVFill:
    """Execution record for one period in a POV run."""
    period:      int
    mkt_volume:  float   # observed market volume this period
    quantity:    float   # quantity traded
    fill_price:  float   # average fill price
    remaining:   float   # quantity still to fill after this period

    def __repr__(self) -> str:
        return (f"POVFill(t={self.period}, V={self.mkt_volume:.0f}, "
                f"q={self.quantity:.2f}@{self.fill_price:.4f}, "
                f"rem={self.remaining:.2f})")


@dataclass
class POVResult:
    """Completed POV execution summary."""
    total_quantity:    float
    avg_fill_price:    float
    n_periods:         int       # periods needed to fill
    actual_rate:       float     # realised avg participation rate
    fills:             List[POVFill]
    total_notional:    float
    complete:          bool      # True if fully filled

    def __repr__(self) -> str:
        return (f"POVResult(Q={self.total_quantity:.2f}, "
                f"avg_fill={self.avg_fill_price:.4f}, "
                f"periods={self.n_periods}, "
                f"rate={self.actual_rate:.2%}, "
                f"complete={self.complete})")


# ── POV executor ─────────────────────────────────────────────────────────────

class ParticipationRateExecutor:
    """
    Percent-of-Volume (POV) execution engine.

    Parameters
    ----------
    rate       : target participation rate ∈ (0, 1].  E.g. 0.10 = 10% of
                 market volume each period.
    max_rate   : hard cap on participation (e.g. 0.30 avoids moving the market).
    spread_bps : half-spread cost in basis points (applied to each fill).
    eta        : market impact coefficient for square-root model (0 = no impact).
    daily_vol  : daily return vol for impact scaling (only used if eta > 0).

    Usage — online
    --------------
    >>> pov = ParticipationRateExecutor(rate=0.10)
    >>> pov.start(quantity=50_000)
    >>> while not pov.is_complete():
    ...     vol_t   = get_market_volume()
    ...     mid_t   = get_mid_price()
    ...     fill    = pov.step(mkt_volume=vol_t, mid_price=mid_t)

    Usage — batch simulation
    ------------------------
    >>> result = pov.simulate(quantity=50_000, mkt_volumes=vols, mid_prices=mids)
    """

    def __init__(
        self,
        rate:       float = 0.10,
        max_rate:   float = 0.30,
        spread_bps: float = 5.0,
        eta:        float = 0.0,
        daily_vol:  float = 0.02,
    ) -> None:
        if not (0.0 < rate <= 1.0):
            raise ValueError("rate must be in (0, 1]")
        if not (0.0 < max_rate <= 1.0):
            raise ValueError("max_rate must be in (0, 1]")
        self.rate       = rate
        self.max_rate   = max_rate
        self.spread_bps = spread_bps
        self.eta        = eta
        self.daily_vol  = daily_vol

        self._quantity:  float = 0.0
        self._remaining: float = 0.0
        self._direction: float = 1.0
        self._fills: List[POVFill] = []
        self._period: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, quantity: float) -> None:
        """Initialise a new parent order."""
        self._quantity  = float(quantity)
        self._remaining = float(quantity)
        self._direction = 1.0 if quantity >= 0 else -1.0
        self._fills     = []
        self._period    = 0

    def is_complete(self) -> bool:
        return abs(self._remaining) < 1e-6

    def remaining(self) -> float:
        return self._remaining

    # ── Single-period step ────────────────────────────────────────────────────

    def step(
        self,
        mkt_volume: float,
        mid_price:  float,
    ) -> Optional[POVFill]:
        """
        Process one market period.

        Parameters
        ----------
        mkt_volume : observed market volume this period (shares / contracts).
        mid_price  : current mid-price.

        Returns
        -------
        POVFill if a trade was executed this period, None if already complete.
        """
        if self.is_complete():
            return None

        v   = max(float(mkt_volume), 0.0)
        eff_rate = min(self.rate, self.max_rate)
        q   = min(eff_rate * v, abs(self._remaining))

        if q < 1e-9:
            self._period += 1
            return None

        signed_q = self._direction * q

        # Fill price: mid ± half-spread ± market impact
        half_spread = mid_price * self.spread_bps / 2.0 / 1e4
        if self.eta > 0 and v > 0:
            impact = self.eta * self.daily_vol * np.sqrt(q / v) * mid_price
        else:
            impact = 0.0

        fill_price = mid_price + self._direction * (half_spread + impact)

        self._remaining -= signed_q
        fill = POVFill(
            period     = self._period,
            mkt_volume = v,
            quantity   = signed_q,
            fill_price = fill_price,
            remaining  = self._remaining,
        )
        self._fills.append(fill)
        self._period += 1
        return fill

    # ── Batch simulation ──────────────────────────────────────────────────────

    def simulate(
        self,
        quantity:    float,
        mkt_volumes: np.ndarray,
        mid_prices:  np.ndarray,
    ) -> POVResult:
        """
        Simulate POV execution over a sequence of market periods.

        Parameters
        ----------
        quantity    : total signed order quantity.
        mkt_volumes : market volume per period, shape (T,).
        mid_prices  : mid price per period, shape (T,).

        Returns
        -------
        POVResult with full fill history and summary statistics.
        """
        vols  = np.asarray(mkt_volumes, dtype=float)
        mids  = np.asarray(mid_prices,  dtype=float)
        if len(vols) != len(mids):
            raise ValueError("mkt_volumes and mid_prices must have equal length")

        self.start(quantity)

        for v, m in zip(vols, mids):
            if self.is_complete():
                break
            self.step(mkt_volume=v, mid_price=m)

        return self._build_result()

    def result(self) -> POVResult:
        """Build result from current fill history."""
        return self._build_result()

    def _build_result(self) -> POVResult:
        fills = self._fills
        if not fills:
            return POVResult(
                total_quantity  = self._quantity,
                avg_fill_price  = 0.0,
                n_periods       = 0,
                actual_rate     = 0.0,
                fills           = [],
                total_notional  = 0.0,
                complete        = False,
            )

        total_filled   = sum(abs(f.quantity) for f in fills)
        total_notional = sum(abs(f.quantity) * f.fill_price for f in fills)
        avg_fill       = total_notional / max(total_filled, 1e-12)
        total_mkt_vol  = sum(f.mkt_volume for f in fills)
        actual_rate    = total_filled / max(total_mkt_vol, 1e-12)

        return POVResult(
            total_quantity  = self._quantity,
            avg_fill_price  = avg_fill,
            n_periods       = len(fills),
            actual_rate     = actual_rate,
            fills           = list(fills),
            total_notional  = total_notional,
            complete        = self.is_complete(),
        )

    def __repr__(self) -> str:
        return (f"ParticipationRateExecutor(rate={self.rate:.0%}, "
                f"max_rate={self.max_rate:.0%}, spread={self.spread_bps}bps)")
