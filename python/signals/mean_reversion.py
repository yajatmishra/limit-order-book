"""
Mean-Reversion Signals
======================
Implements Ornstein-Uhlenbeck parameter estimation and z-score–based
mean-reversion signals for single-asset and spread time series.

Ornstein-Uhlenbeck (OU) process:
    dX = κ(μ − X) dt + σ dW

Discrete AR(1) approximation used for estimation (Chan 2013):
    ΔX_t = a + b · X_{t-1} + ε_t
where
    b     = −κ · Δt          → κ = −b / Δt
    a     = κ · μ · Δt       → μ = −a / b
    Var(ε)= σ² · Δt           → σ_eq = √(σ²_resid / (1 − exp(−2κΔt))) / √(2κ)
    half-life = ln(2) / κ
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── OU Estimation ─────────────────────────────────────────────────────────────

@dataclass
class OUResult:
    """Estimated OU process parameters."""
    kappa:     float   # mean-reversion speed  (>0 for stationarity)
    mu:        float   # long-run mean
    sigma:     float   # equilibrium volatility (annualised if Δt=1/252)
    half_life: float   # ln(2)/κ — time to halve a deviation
    residual_std: float  # standard deviation of AR(1) residuals
    r_squared: float   # in-sample R² of the AR(1) regression

    @property
    def is_mean_reverting(self) -> bool:
        """Returns True when κ > 0 and half-life is finite."""
        return self.kappa > 0 and np.isfinite(self.half_life)

    def __repr__(self) -> str:
        return (
            f"OUResult(κ={self.kappa:.4f}, μ={self.mu:.4f}, "
            f"σ={self.sigma:.4f}, HL={self.half_life:.2f})"
        )


def estimate_ou(
    series: np.ndarray,
    dt:     float = 1.0,
) -> OUResult:
    """
    Fit an OU process to `series` via OLS on the AR(1) representation:

        ΔX_t = a + b · X_{t-1} + ε_t

    Parameters
    ----------
    series : stationary or near-stationary price / spread series.
    dt     : observation interval (e.g. 1/252 for daily data in years).

    Returns
    -------
    OUResult with κ, μ, σ, half-life, and diagnostic R².
    """
    x  = np.asarray(series, dtype=float)
    dx = np.diff(x)
    xl = x[:-1]

    # OLS: [a, b] via normal equations with constant
    X   = np.column_stack([np.ones_like(xl), xl])
    sol = np.linalg.lstsq(X, dx, rcond=None)
    a, b = sol[0]

    # Residuals and R²
    fitted     = X @ [a, b]
    resid      = dx - fitted
    ss_res     = np.dot(resid, resid)
    ss_tot     = np.dot(dx - dx.mean(), dx - dx.mean())
    r2         = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_std  = float(np.sqrt(ss_res / max(len(dx) - 2, 1)))

    # OU parameters from AR(1) coefficients
    kappa = -b / dt  if b != 0 else 1e-10
    mu    = -a / b   if b != 0 else float(x.mean())
    kappa = max(kappa, 1e-10)   # enforce positive for numerical safety

    # Equilibrium vol: σ_eq from σ²(X) = σ² / (2κ) in continuous time
    # From discrete residuals: σ_eq² ≈ Var(ε) / (1 − exp(−2κΔt))
    factor   = 1.0 - np.exp(-2.0 * kappa * dt)
    sigma    = float(resid_std / np.sqrt(factor)) if factor > 0 else resid_std

    half_life = float(np.log(2.0) / kappa)

    return OUResult(
        kappa      = float(kappa),
        mu         = float(mu),
        sigma      = float(sigma),
        half_life  = half_life,
        residual_std = resid_std,
        r_squared  = float(r2),
    )


# ── Z-score signal ─────────────────────────────────────────────────────────────

def mean_reversion_zscore(
    series:     np.ndarray,
    window:     int,
    entry_z:    float = 2.0,
    exit_z:     float = 0.5,
) -> np.ndarray:
    """
    Rolling z-score signal for mean reversion.

        z_t = (X_t − μ_t) / σ_t

    where μ_t and σ_t are rolling window statistics.

    First (window-1) values are NaN.

    Parameters
    ----------
    series  : price / spread to trade.
    window  : rolling lookback for mean and std estimation.
    entry_z : z-score magnitude at which to enter  (informational only;
              returned z-score is raw — caller applies thresholds).
    exit_z  : z-score magnitude at which to exit.

    Returns
    -------
    z : rolling z-score array, same length as series.
    """
    x   = np.asarray(series, dtype=float)
    n   = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = x[i - window + 1 : i + 1]
        mu    = chunk.mean()
        sig   = chunk.std(ddof=1)
        out[i] = (x[i] - mu) / sig if sig > 0 else 0.0
    return out


def ou_zscore(
    series: np.ndarray,
    ou:     OUResult,
) -> np.ndarray:
    """
    Z-score standardised by the OU equilibrium mean and volatility:

        z_t = (X_t − μ) / σ_eq

    Uses the long-run OU parameters rather than a rolling window — works
    best when the OU parameters are stable over the estimation horizon.
    """
    x = np.asarray(series, dtype=float)
    return (x - ou.mu) / ou.sigma if ou.sigma > 0 else np.zeros_like(x)


# ── Mean-reversion strength ────────────────────────────────────────────────────

def mean_reversion_strength(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling mean-reversion strength: the AR(1) coefficient b̂ from

        ΔX_t = a + b · X_{t-1} + ε_t

    b̂ ∈ (−1, 0) indicates mean reversion; the more negative, the faster.
    Returns a rolling estimate over each `window`-length sub-series.

    First (window-1) values are NaN.
    """
    x   = np.asarray(series, dtype=float)
    n   = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = x[i - window + 1 : i + 1]
        dx    = np.diff(chunk)
        xl    = chunk[:-1]
        X     = np.column_stack([np.ones_like(xl), xl])
        sol   = np.linalg.lstsq(X, dx, rcond=None)
        out[i] = float(sol[0][1])   # AR(1) slope
    return out


# ── Bollinger Band signal ──────────────────────────────────────────────────────

@dataclass
class BollingerSignal:
    """Signal derived from Bollinger Band position."""
    position:    np.ndarray   # z-score (price−mid) / (n_std·σ)
    upper:       np.ndarray   # upper band  = mid + n_std·σ
    lower:       np.ndarray   # lower band  = mid − n_std·σ
    mid:         np.ndarray   # rolling mean
    pct_b:       np.ndarray   # %B = (price−lower) / (upper−lower)


def bollinger_signal(
    prices: np.ndarray,
    window: int   = 20,
    n_std:  float = 2.0,
) -> BollingerSignal:
    """
    Compute Bollinger Band components.

    %B = (P − lower) / (upper − lower)
        %B > 1 → above upper band (overbought)
        %B < 0 → below lower band (oversold)
        %B = 0.5 → at midband

    All arrays are NaN for the first (window-1) observations.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)

    pos    = np.full(n, np.nan)
    upper  = np.full(n, np.nan)
    lower  = np.full(n, np.nan)
    mid    = np.full(n, np.nan)
    pct_b  = np.full(n, np.nan)

    for i in range(window - 1, n):
        chunk = prices[i - window + 1 : i + 1]
        mu    = chunk.mean()
        sig   = chunk.std(ddof=1)
        bw    = n_std * sig
        mid[i]   = mu
        upper[i] = mu + bw
        lower[i] = mu - bw
        if bw > 0:
            pos[i]   = (prices[i] - mu) / bw
            pct_b[i] = (prices[i] - lower[i]) / (upper[i] - lower[i])

    return BollingerSignal(
        position=pos, upper=upper, lower=lower, mid=mid, pct_b=pct_b
    )


# ── Half-life from autocorrelation ─────────────────────────────────────────────

def half_life_from_autocorr(
    series: np.ndarray,
    max_lag: int = 1,
) -> float:
    """
    Estimate mean-reversion half-life from the lag-1 autocorrelation ρ:

        κ ≈ −ln(ρ) / Δt   →   HL = ln(2) / κ = −ln(2) / ln(ρ)

    This is a quick alternative to fitting the full AR(1) model.
    Returns inf when ρ ≥ 1 (random walk or trending).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 2:
        return np.inf
    xm   = x - x.mean()
    rho  = float(np.dot(xm[max_lag:], xm[:-max_lag]) /
                 np.dot(xm, xm))
    if rho >= 1.0 or rho <= 0.0:
        return np.inf
    return float(-np.log(2.0) / np.log(rho))
