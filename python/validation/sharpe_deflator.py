"""
Sharpe Ratio Deflation
======================
Corrects the observed Sharpe ratio for multiple testing and the non-normality
of strategy returns, yielding more conservative and realistic estimates of
out-of-sample performance.

Three statistics are implemented:

1. Probabilistic Sharpe Ratio (PSR)
   P(SR* > SR_0) — probability that the true SR exceeds a benchmark SR_0.
   Accounts for skewness and excess kurtosis of the return distribution.

   PSR(SR_0) = Φ( (SR_hat − SR_0) · √(T−1) /
                  √(1 − γ̂₃·SR_hat + (γ̂₄−1)/4 · SR_hat²) )

   Reference: Bailey & Lopez de Prado (2012). "The Sharpe Ratio Efficient
   Frontier." Journal of Risk 15(2).

2. Deflated Sharpe Ratio (DSR)
   PSR evaluated at the expected maximum SR from a set of N independent
   trials / backtest configurations.

   SR_0(N) ≈ (1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))
   where γ = Euler-Mascheroni constant ≈ 0.5772.

   Reference: Bailey & Lopez de Prado (2014). "The Deflated Sharpe Ratio."
   Financial Analysts Journal 70(5).

3. Minimum Track Record Length (MTRL)
   Minimum number of observations needed for PSR(SR_0) ≥ 1 − α at a
   given significance level α.

   MTRL = 1 + (1 − γ̂₃·SR_hat + (γ̂₄−1)/4 · SR_hat²)
              · (Φ⁻¹(1 − α) / (SR_hat − SR_0))²
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.stats import norm as _norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Gaussian CDF / quantile (pure-numpy fallback) ─────────────────────────────

def _phi(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    if _HAS_SCIPY:
        return float(_norm.cdf(x))
    # Abramowitz & Stegun approximation (max error 7.5e-8)
    sign = 1.0 if x >= 0 else -1.0
    z    = abs(x) / np.sqrt(2.0)
    t    = 1.0 / (1.0 + 0.3275911 * z)
    p    = (t * (0.254829592
                 + t * (-0.284496736
                        + t * (1.421413741
                               + t * (-1.453152027
                                      + t * 1.061405429)))))
    return float(0.5 * (1.0 + sign * (1.0 - p * np.exp(-z * z))))


def _phi_inv(p: float) -> float:
    """Inverse normal CDF Φ⁻¹(p) using scipy or rational approximation."""
    if _HAS_SCIPY:
        return float(_norm.ppf(p))
    # Rational approximation (Beasley-Springer-Moro)
    p = float(np.clip(p, 1e-15, 1 - 1e-15))
    if p < 0.5:
        sign = -1.0
        q    = p
    else:
        sign = 1.0
        q    = 1.0 - p
    r = np.sqrt(-2.0 * np.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * r + c2 * r * r
    den = 1.0 + d1 * r + d2 * r * r + d3 * r * r * r
    return float(sign * (r - num / den))


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SharpeDeflatorResult:
    """Output of SharpeDeflator.compute()."""
    sharpe_hat:     float   # observed annualised SR
    psr:            float   # Probabilistic SR (probability)
    dsr:            float   # Deflated SR (probability with N-trial correction)
    mtrl:           float   # Minimum Track Record Length (observations)
    sr_benchmark:   float   # SR_0 used for PSR
    skew:           float   # estimated skewness γ̂₃
    kurt:           float   # estimated excess kurtosis γ̂₄
    n_obs:          int     # number of observations
    n_trials:       int     # number of trials for DSR

    def __repr__(self) -> str:
        return (f"SharpeDeflatorResult("
                f"SR_hat={self.sharpe_hat:.3f}, "
                f"PSR={self.psr:.4f}, "
                f"DSR={self.dsr:.4f}, "
                f"MTRL={self.mtrl:.1f})")


# ── SharpeDeflator ────────────────────────────────────────────────────────────

class SharpeDeflator:
    """
    Sharpe ratio statistical testing and deflation.

    Parameters
    ----------
    sr_benchmark : SR_0, the benchmark Sharpe ratio to test against.
                   Default 0.0 (is the strategy better than random?).
    n_trials     : number of independent strategy configs tried during
                   development (used for DSR).  Default 1.
    periods_per_year : for annualising the Sharpe.  Default 252 (daily).
    alpha        : significance level for MTRL.  Default 0.05.

    Usage
    -----
    >>> deflator = SharpeDeflator(sr_benchmark=0.5, n_trials=20)
    >>> result = deflator.compute(daily_returns)
    >>> print(result.dsr)   # probability SR > 0.5 after 20-trial correction
    """

    EULER_MASCHERONI = 0.5772156649015328

    def __init__(
        self,
        sr_benchmark:      float = 0.0,
        n_trials:          int   = 1,
        periods_per_year:  int   = 252,
        alpha:             float = 0.05,
    ) -> None:
        self.sr_benchmark     = sr_benchmark
        self.n_trials         = max(1, n_trials)
        self.periods_per_year = periods_per_year
        self.alpha            = alpha

    def compute(self, returns: np.ndarray) -> SharpeDeflatorResult:
        """
        Compute PSR, DSR, and MTRL for a return series.

        Parameters
        ----------
        returns : per-period return series, shape (T,).
                  Assumed to be at `periods_per_year` frequency.

        Returns
        -------
        SharpeDeflatorResult.
        """
        r  = np.asarray(returns, dtype=float)
        r  = r[np.isfinite(r)]
        T  = len(r)

        if T < 4:
            raise ValueError("Need at least 4 observations")

        mu   = r.mean()
        sig  = r.std(ddof=1)
        if sig < 1e-12:
            raise ValueError("Return series has zero variance")

        # Observed (annualised) Sharpe
        sr_hat = float(mu / sig * np.sqrt(self.periods_per_year))

        # Per-period SR (not annualised) for PSR formula
        sr_pp = float(mu / sig)

        # Moments
        skew = float(((r - mu) ** 3).mean() / sig ** 3)
        kurt = float(((r - mu) ** 4).mean() / sig ** 4) - 3.0   # excess

        # PSR(SR_0): probability that true SR > sr_benchmark (annualised)
        sr0_pp = self.sr_benchmark / np.sqrt(self.periods_per_year)
        psr    = self._psr(sr_pp, sr0_pp, T, skew, kurt)

        # DSR: adjusted benchmark using expected max SR from n_trials
        sr0_dsr_pp = self._sr_expected_max(T, self.n_trials) / np.sqrt(self.periods_per_year)
        dsr        = self._psr(sr_pp, sr0_dsr_pp, T, skew, kurt)

        # MTRL
        mtrl = self._mtrl(sr_pp, sr0_pp, skew, kurt, self.alpha)

        return SharpeDeflatorResult(
            sharpe_hat   = sr_hat,
            psr          = psr,
            dsr          = dsr,
            mtrl         = mtrl,
            sr_benchmark = self.sr_benchmark,
            skew         = skew,
            kurt         = kurt,
            n_obs        = T,
            n_trials     = self.n_trials,
        )

    # ── Statistical formulas ──────────────────────────────────────────────────

    @staticmethod
    def _psr(
        sr_hat: float,
        sr0:    float,
        T:      int,
        skew:   float,
        kurt:   float,
    ) -> float:
        """
        Probabilistic SR: PSR(SR_0) = Φ(z).

        All SRs are per-period (not annualised).
        """
        variance_adj = max(1.0 - skew * sr_hat + (kurt / 4.0) * sr_hat ** 2, 1e-12)
        z = (sr_hat - sr0) * np.sqrt(T - 1) / np.sqrt(variance_adj)
        return _phi(float(z))

    @staticmethod
    def _sr_expected_max(T: int, N: int) -> float:
        """
        Expected maximum SR from N independent trials, each of length T
        (per-period, annualised by caller).

        SR_0^*(N) ≈ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))
        """
        gamma = SharpeDeflator.EULER_MASCHERONI
        if N <= 1:
            return 0.0
        term1 = (1.0 - gamma) * _phi_inv(1.0 - 1.0 / N)
        term2 = gamma          * _phi_inv(1.0 - 1.0 / (N * np.e))
        # Scale from standard-normal units to per-period SR
        # (the formula above produces Z-scores; we scale by 1/√T)
        return float((term1 + term2) / np.sqrt(T))

    @staticmethod
    def _mtrl(
        sr_hat: float,
        sr0:    float,
        skew:   float,
        kurt:   float,
        alpha:  float = 0.05,
    ) -> float:
        """
        Minimum Track Record Length for PSR(SR_0) ≥ 1 − α.

        All SRs are per-period.
        """
        z_alpha = _phi_inv(1.0 - alpha)
        diff    = sr_hat - sr0
        if abs(diff) < 1e-12:
            return np.inf
        variance_adj = max(1.0 - skew * sr_hat + (kurt / 4.0) * sr_hat ** 2, 1e-12)
        mtrl = 1.0 + variance_adj * (z_alpha / diff) ** 2
        return float(max(mtrl, 2.0))


# ── Convenience functions ─────────────────────────────────────────────────────

def probabilistic_sharpe(
    returns:      np.ndarray,
    sr_benchmark: float = 0.0,
    periods:      int   = 252,
) -> float:
    """Return PSR as a single float."""
    return SharpeDeflator(sr_benchmark=sr_benchmark,
                          periods_per_year=periods).compute(returns).psr


def deflated_sharpe(
    returns:  np.ndarray,
    n_trials: int   = 1,
    periods:  int   = 252,
) -> float:
    """Return DSR as a single float."""
    return SharpeDeflator(n_trials=n_trials,
                          periods_per_year=periods).compute(returns).dsr


def min_track_record_length(
    returns:      np.ndarray,
    sr_benchmark: float = 0.0,
    alpha:        float = 0.05,
    periods:      int   = 252,
) -> float:
    """Return MTRL as a single float (in observations)."""
    return SharpeDeflator(sr_benchmark=sr_benchmark, alpha=alpha,
                          periods_per_year=periods).compute(returns).mtrl
