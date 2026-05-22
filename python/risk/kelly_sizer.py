"""
Kelly Position Sizer
====================
Computes optimal bet sizes using the Kelly criterion and its variants.

Kelly criterion
---------------
For a strategy with expected return μ and variance σ² per period, the
Kelly fraction is:

  f* = μ / σ²       (continuous, Gaussian returns)

For a binary bet with win probability p, win payoff b (per unit risked),
and loss of 1:

  f* = (p·b − q) / b   where q = 1 − p  (classical discrete Kelly)

Multi-asset Kelly (log-optimal portfolio):

  f* = Σ⁻¹ · μ       (covariance-matrix form)

Fractional Kelly
----------------
Full Kelly can be very aggressive and has poor out-of-sample properties
when parameters are estimated with uncertainty.  Practitioners typically use:

  f = c · f*    where c ∈ (0, 1]  (half-Kelly: c=0.5)

Constraints
-----------
  max_position : hard cap on position size (absolute units).
  max_leverage : cap on gross exposure / equity.
  min_edge_bps : minimum required edge in basis points; returns 0 if below.

References
----------
  Kelly (1956). "A new interpretation of information rate."
  Bell System Technical Journal 35(4), 917-926.

  Thorp (1997). "The Kelly criterion in blackjack, sports betting, and the
  stock market." 10th International Conference on Gambling and Risk Taking.

  MacLean, Thorp & Ziemba (2011). "The Kelly Capital Growth Investment
  Criterion." World Scientific.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    """Output of a Kelly sizer computation."""
    full_kelly:       float   # unconstrained f*
    fractional_kelly: float   # c · f*
    final_size:       float   # after constraints
    fraction_used:    float   # c (fractional Kelly multiplier)
    edge_bps:         float   # μ / price in bps (if price provided)
    constrained:      bool    # True if a constraint was binding

    def __repr__(self) -> str:
        c_str = " [constrained]" if self.constrained else ""
        return (f"KellyResult(f*={self.full_kelly:.4f}, "
                f"f_frac={self.fractional_kelly:.4f}, "
                f"size={self.final_size:.4f}{c_str})")


# ── Single-asset Kelly sizer ──────────────────────────────────────────────────

class KellySizer:
    """
    Single-asset Kelly position sizer.

    Parameters
    ----------
    fraction     : fractional Kelly multiplier c ∈ (0, 1].  Default 0.5.
    max_position : maximum absolute position (in units, e.g. shares).
    max_leverage : maximum |position| / equity.  Applied if equity is passed.
    min_edge_bps : minimum edge in bps to take a position.  Default 0.

    Usage
    -----
    >>> sizer = KellySizer(fraction=0.5, max_position=10_000)
    >>> result = sizer.size_from_moments(mu=0.001, sigma=0.02, equity=1e6)
    >>> result = sizer.size_binary(p_win=0.55, payoff=1.0, equity=1e6)
    """

    def __init__(
        self,
        fraction:     float = 0.5,
        max_position: float = np.inf,
        max_leverage: float = np.inf,
        min_edge_bps: float = 0.0,
    ) -> None:
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1]")
        self.fraction     = fraction
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.min_edge_bps = min_edge_bps

    # ── Gaussian returns ──────────────────────────────────────────────────────

    def size_from_moments(
        self,
        mu:     float,
        sigma:  float,
        equity: float = 1.0,
        price:  float = 1.0,
    ) -> KellyResult:
        """
        Kelly size for a strategy with known μ and σ (per period).

        f* = μ / σ²  →  position = f* · equity / price

        Parameters
        ----------
        mu     : per-period expected return (fractional).
        sigma  : per-period return std-dev.
        equity : account equity in dollars.
        price  : current asset price (for converting f to shares).

        Returns
        -------
        KellyResult with fraction of equity to allocate.
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        var          = sigma ** 2
        full_kelly_f = mu / var             # fraction of equity
        frac_kelly_f = self.fraction * full_kelly_f

        edge_bps = mu / max(price, 1e-12) * 1e4
        if abs(edge_bps) < self.min_edge_bps:
            return KellyResult(full_kelly=full_kelly_f, fractional_kelly=0.0,
                               final_size=0.0, fraction_used=self.fraction,
                               edge_bps=edge_bps, constrained=True)

        # Convert to shares/units
        raw_size = frac_kelly_f * equity / max(price, 1e-12)
        return self._apply_constraints(full_kelly_f, frac_kelly_f, raw_size,
                                       equity, price, edge_bps)

    # ── Binary Kelly (discrete bet) ───────────────────────────────────────────

    def size_binary(
        self,
        p_win:  float,
        payoff: float,
        equity: float = 1.0,
    ) -> KellyResult:
        """
        Kelly fraction for a binary bet.

        f* = (p·b − q) / b   where q = 1−p, b = payoff per unit risked.

        Returns fraction of equity to wager; negative means bet the other side.

        Parameters
        ----------
        p_win  : probability of winning.
        payoff : net gain per unit wagered on a win (e.g. 1.0 for even odds).
        equity : account equity.
        """
        if not (0.0 < p_win < 1.0):
            raise ValueError("p_win must be in (0, 1)")
        if payoff <= 0:
            raise ValueError("payoff must be > 0")

        q = 1.0 - p_win
        full_kelly_f = (p_win * payoff - q) / payoff
        frac_kelly_f = self.fraction * full_kelly_f

        edge_bps = full_kelly_f * 1e4
        if abs(edge_bps) < self.min_edge_bps:
            return KellyResult(full_kelly=full_kelly_f, fractional_kelly=0.0,
                               final_size=0.0, fraction_used=self.fraction,
                               edge_bps=edge_bps, constrained=True)

        raw_size = frac_kelly_f * equity
        return self._apply_constraints(full_kelly_f, frac_kelly_f, raw_size,
                                       equity, 1.0, edge_bps)

    # ── Sharpe-based sizing ───────────────────────────────────────────────────

    def size_from_sharpe(
        self,
        sharpe_ratio: float,
        sigma:        float,
        equity:       float = 1.0,
        price:        float = 1.0,
    ) -> KellyResult:
        """
        Convenience wrapper: derive μ = SR · σ, then call size_from_moments.

        Useful when you have an estimated Sharpe (e.g. from validation).
        """
        mu = sharpe_ratio * sigma
        return self.size_from_moments(mu=mu, sigma=sigma, equity=equity, price=price)

    # ── Constraints ───────────────────────────────────────────────────────────

    def _apply_constraints(
        self,
        full_kelly_f:  float,
        frac_kelly_f:  float,
        raw_size:      float,
        equity:        float,
        price:         float,
        edge_bps:      float,
    ) -> KellyResult:
        size       = raw_size
        constrained = False

        # Max position (absolute units)
        if abs(size) > self.max_position:
            size = np.sign(size) * self.max_position
            constrained = True

        # Max leverage: |size| * price / equity ≤ max_leverage
        if equity > 0 and price > 0:
            lev = abs(size) * price / equity
            if lev > self.max_leverage:
                size = np.sign(size) * self.max_leverage * equity / price
                constrained = True

        return KellyResult(
            full_kelly       = full_kelly_f,
            fractional_kelly = frac_kelly_f,
            final_size       = size,
            fraction_used    = self.fraction,
            edge_bps         = edge_bps,
            constrained      = constrained,
        )


# ── Multi-asset Kelly ─────────────────────────────────────────────────────────

def multi_asset_kelly(
    mu:       np.ndarray,
    cov:      np.ndarray,
    fraction: float = 0.5,
) -> np.ndarray:
    """
    Log-optimal (full Kelly) portfolio weights for N assets.

    f* = Σ⁻¹ · μ    (unnormalised fractions of equity per asset)

    Parameters
    ----------
    mu       : expected per-period returns, shape (N,).
    cov      : covariance matrix, shape (N, N).
    fraction : fractional Kelly multiplier.

    Returns
    -------
    weights : shape (N,) — fraction of equity in each asset.
    """
    mu  = np.asarray(mu,  dtype=float)
    cov = np.asarray(cov, dtype=float)
    N   = len(mu)

    if cov.shape != (N, N):
        raise ValueError(f"cov must be ({N},{N}); got {cov.shape}")

    try:
        cov_inv = np.linalg.inv(cov + np.eye(N) * 1e-10)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    return fraction * (cov_inv @ mu)
