"""
Cointegration Analysis
======================
Tests for cointegration and constructs mean-reverting spreads.

References:
  Engle & Granger (1987). "Co-integration and Error Correction."
  Econometrica 55(2), 251-276.

  Johansen (1988). "Statistical Analysis of Cointegration Vectors."
  Journal of Economic Dynamics and Control 12(2), 231-254.

  MacKinnon (1994/2010). Critical values for cointegration tests.

  Chan (2013). "Algorithmic Trading." Wiley.

Two-step Engle-Granger procedure:
  1.  OLS: y1 = β·y2 + α + ε
  2.  ADF test on residuals ε — reject H0 (unit root) → cointegrated.

Johansen trace test for 2 assets (simplified 2×2 case):
  Tests H0: rank(Π) = 0  vs  H1: rank(Π) ≥ 1
  via the trace statistic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# statsmodels is the gold standard for ADF; we use it with a pure-numpy fallback
try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ── MacKinnon critical values for EG residual-based test ─────────────────────
# Source: MacKinnon (2010), n=2 (one covariate, trend='c')
_EG_CRITICAL = {
    "1%": -3.9001,
    "5%": -3.3377,
    "10%": -3.0462,
}


# ── ADF helpers ───────────────────────────────────────────────────────────────

def _adf_numpy(series: np.ndarray, maxlag: int = 1) -> Tuple[float, float]:
    """
    Pure-numpy ADF test (with constant, no trend, up to maxlag augmentation lags).
    Returns (t_stat, approx_p_value) using a logistic approximation.
    """
    x  = np.asarray(series, dtype=float)
    dx = np.diff(x)
    xl = x[:-1]
    n  = len(dx)

    # Trim for augmentation
    lag = min(maxlag, n // 4)
    if lag == 0:
        X = np.column_stack([xl, np.ones(n)])
        y = dx
    else:
        # Stack lagged differences
        lags = [dx[lag - k - 1 : n - k - 1] for k in range(lag)]
        X = np.column_stack([xl[lag:], np.ones(n - lag)] + lags)
        y = dx[lag:]

    sol   = np.linalg.lstsq(X, y, rcond=None)
    beta  = sol[0]
    resid = y - X @ beta
    s2    = np.dot(resid, resid) / max(len(resid) - X.shape[1], 1)
    cov   = np.linalg.pinv(X.T @ X) * s2
    se    = np.sqrt(max(cov[0, 0], 1e-30))
    t     = beta[0] / se

    # Rough p-value via MacKinnon (2010) polynomial approximation for n=1 covariate
    # Not precise — caller should use statsmodels when available
    p = float(np.clip(1.0 / (1.0 + np.exp(-1.85 * (t + 3.33))), 0.0, 1.0))
    return float(t), p


def _adf(series: np.ndarray, maxlag: int = 1) -> Tuple[float, float]:
    """ADF wrapper: use statsmodels if available, else numpy fallback."""
    if _HAS_STATSMODELS:
        result = _adfuller(series, maxlag=maxlag, regression='c', autolag=None)
        return float(result[0]), float(result[1])
    return _adf_numpy(series, maxlag=maxlag)


# ── Engle-Granger ─────────────────────────────────────────────────────────────

@dataclass
class EngleGrangerResult:
    """Result of the two-step Engle-Granger cointegration test."""
    hedge_ratio:    float   # β: OLS coefficient on y2
    intercept:      float   # α
    adf_stat:       float   # ADF t-statistic on residuals
    adf_pvalue:     float   # approximate p-value
    critical_values: dict   # MacKinnon critical values for EG test
    cointegrated_1pct:  bool
    cointegrated_5pct:  bool
    cointegrated_10pct: bool
    residuals:      np.ndarray   # spread = y1 − β·y2 − α

    @property
    def is_cointegrated(self) -> bool:
        """Reject H0 at 5% significance."""
        return self.cointegrated_5pct

    def __repr__(self) -> str:
        stars = ("***" if self.cointegrated_1pct else
                 "**"  if self.cointegrated_5pct  else
                 "*"   if self.cointegrated_10pct else "")
        return (f"EngleGrangerResult(β={self.hedge_ratio:.4f}, "
                f"ADF={self.adf_stat:.3f}{stars}, p={self.adf_pvalue:.4f})")


def engle_granger(
    y1:     np.ndarray,
    y2:     np.ndarray,
    maxlag: int = 1,
) -> EngleGrangerResult:
    """
    Two-step Engle-Granger cointegration test.

    Step 1: OLS  y1_t = α + β·y2_t + ε_t
    Step 2: ADF test on residuals ε̂  (H0: unit root → not cointegrated).

    Parameters
    ----------
    y1, y2 : price series of equal length, shape (N,).
    maxlag  : max augmentation lags in ADF step.

    Returns
    -------
    EngleGrangerResult with hedge ratio, test stat, and significance flags.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have equal length")

    # OLS
    X  = np.column_stack([np.ones_like(y2), y2])
    sol = np.linalg.lstsq(X, y1, rcond=None)
    alpha, beta = sol[0]
    resid = y1 - (alpha + beta * y2)

    # ADF on residuals
    t_stat, p_val = _adf(resid, maxlag=maxlag)

    return EngleGrangerResult(
        hedge_ratio          = float(beta),
        intercept            = float(alpha),
        adf_stat             = t_stat,
        adf_pvalue           = p_val,
        critical_values      = dict(_EG_CRITICAL),
        cointegrated_1pct    = t_stat < _EG_CRITICAL["1%"],
        cointegrated_5pct    = t_stat < _EG_CRITICAL["5%"],
        cointegrated_10pct   = t_stat < _EG_CRITICAL["10%"],
        residuals            = resid,
    )


# ── Johansen (simplified 2-asset) ─────────────────────────────────────────────

@dataclass
class JohansenResult:
    """Simplified Johansen trace test for 2 assets."""
    trace_stat:     float   # trace statistic
    critical_5pct:  float   # 5% critical value (r=0, n=2)
    cointegrated:   bool
    hedge_vector:   np.ndarray   # normalised cointegrating vector


def johansen_trace(
    y1:    np.ndarray,
    y2:    np.ndarray,
    k:     int = 1,
) -> JohansenResult:
    """
    Simplified Johansen trace test for a 2-variable system.

    Estimates the cointegrating vector via reduced-rank regression on
    the VECM representation with `k` lags.

    Critical value (r=0, n=2, k=1): 15.41 at 5%  (Osterwald-Lenum 1992).
    """
    Y = np.column_stack([np.asarray(y1, dtype=float),
                         np.asarray(y2, dtype=float)])
    T, p = Y.shape
    dY   = np.diff(Y, axis=0)   # (T-1, 2)
    Yl   = Y[:-1, :]            # lagged levels (T-1, 2)

    # Residuals from auxiliary regressions (Johansen 1988, step 1)
    # R0 = residuals of regressing dY on lagged dY's (if k > 1)
    # R1 = residuals of regressing Yl on lagged dY's
    if k == 1:
        R0 = dY
        R1 = Yl
    else:
        lags = min(k - 1, (T - 2) // 2)
        # Build lagged-difference matrix
        ld = np.zeros((T - 1 - lags, p * lags))
        for j in range(lags):
            ld[:, j * p : (j + 1) * p] = dY[lags - j - 1 : T - 1 - j, :]
        R0 = _resid(dY[lags:, :], ld)
        R1 = _resid(Yl[lags:, :], ld)

    n = R0.shape[0]
    # Moment matrices
    S00 = R0.T @ R0 / n
    S11 = R1.T @ R1 / n
    S01 = R0.T @ R1 / n

    # Generalised eigenvalue problem: S01 S11^{-1} S01' β = λ S00 β
    try:
        S11_inv   = np.linalg.inv(S11)
        M         = np.linalg.inv(S00) @ S01 @ S11_inv @ S01.T
        eigenvals = np.sort(np.linalg.eigvals(M))[::-1].real
    except np.linalg.LinAlgError:
        return JohansenResult(trace_stat=0.0, critical_5pct=15.41,
                              cointegrated=False,
                              hedge_vector=np.array([1.0, -1.0]))

    # Trace statistic for H0: rank = 0
    trace = float(-n * np.sum(np.log(1.0 - np.clip(eigenvals, 0.0, 1.0 - 1e-10))))
    cv_5  = 15.41   # 5% critical value for p=2, r=0

    # Extract cointegrating vector from eigenvector of largest eigenvalue
    try:
        S11_inv_sqrt = np.linalg.cholesky(S11_inv).T
        A            = S11_inv_sqrt @ S01.T @ np.linalg.inv(S00) @ S01 @ S11_inv_sqrt.T
        _, vecs      = np.linalg.eigh(A)
        hedge_vec    = S11_inv_sqrt.T @ vecs[:, -1]
        hedge_vec   /= hedge_vec[0] if abs(hedge_vec[0]) > 1e-10 else 1.0
    except (np.linalg.LinAlgError, ZeroDivisionError):
        hedge_vec = np.array([1.0, -1.0])

    return JohansenResult(
        trace_stat   = trace,
        critical_5pct = cv_5,
        cointegrated = trace > cv_5,
        hedge_vector = hedge_vec,
    )


def _resid(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals Y − X(X'X)^{-1}X'Y."""
    sol = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ sol[0]


# ── SpreadModel ────────────────────────────────────────────────────────────────

@dataclass
class SpreadModel:
    """
    Fixed-hedge-ratio spread model for live trading.

    Constructed from Engle-Granger or Johansen estimates; updated online
    one observation at a time.

    spread_t = y1_t − β·y2_t − α

    Usage
    -----
    >>> model = SpreadModel.from_engle_granger(y1_train, y2_train)
    >>> z = model.zscore_window(y1_live, y2_live, window=30)
    """
    hedge_ratio:  float
    intercept:    float
    zscore_window_default: int = 60

    @classmethod
    def from_engle_granger(
        cls,
        y1:     np.ndarray,
        y2:     np.ndarray,
        window: int = 60,
    ) -> "SpreadModel":
        eg = engle_granger(y1, y2)
        return cls(hedge_ratio=eg.hedge_ratio,
                   intercept=eg.intercept,
                   zscore_window_default=window)

    def spread(
        self,
        y1: np.ndarray,
        y2: np.ndarray,
    ) -> np.ndarray:
        """Compute the spread series."""
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        return y1 - self.hedge_ratio * y2 - self.intercept

    def zscore(
        self,
        y1:     np.ndarray,
        y2:     np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Rolling z-score of the spread.
        First (window-1) values are NaN.
        """
        w = window or self.zscore_window_default
        s = self.spread(y1, y2)
        n = len(s)
        z = np.full(n, np.nan)
        for i in range(w - 1, n):
            chunk = s[i - w + 1 : i + 1]
            mu    = chunk.mean()
            sig   = chunk.std(ddof=1)
            z[i]  = (s[i] - mu) / sig if sig > 0 else 0.0
        return z


# ── Convenience: rolling cointegration ────────────────────────────────────────

def rolling_hedge_ratio(
    y1:     np.ndarray,
    y2:     np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling OLS hedge ratio β_t, estimated on a sliding window.

    Returns shape (N,); first (window-1) values are NaN.
    This is the "static rolling" version — for dynamic estimation
    use KalmanPairsTrader from kalman_pairs.py.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    n  = len(y1)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        y1_w = y1[i - window + 1 : i + 1]
        y2_w = y2[i - window + 1 : i + 1]
        X    = np.column_stack([np.ones(window), y2_w])
        sol  = np.linalg.lstsq(X, y1_w, rcond=None)
        out[i] = float(sol[0][1])
    return out
