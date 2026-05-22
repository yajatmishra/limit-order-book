"""
Regime Tester
=============
Evaluates signal performance conditional on market regime labels.

Provides:
  - Information Coefficient (IC) per regime
  - Regime-conditional Sharpe ratio
  - Regime transition statistics
  - Signal stability analysis across regimes

IC is defined as the Spearman rank correlation between the signal
observed at time t and the forward return realised over the next
`horizon` periods.

The regime labels (0, 1, …, K-1) are assumed to be produced by an
HMM or any other classifier; regime 0 is conventionally the lowest-
mean / bearish state.

Reference:
  Grinold & Kahn (2000). "Active Portfolio Management." Ch. 6.
  Qian, Hua & Sorensen (2007). "Quantitative Equity Portfolio
  Management." CRC Press.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class RegimeStats:
    """Performance statistics within a single regime."""
    regime:      int
    n_obs:       int        # observations in this regime
    ic:          float      # Spearman IC(signal, fwd_return)
    ic_tstat:    float      # t-stat of IC (H0: IC=0)
    sharpe:      float      # Sharpe of signal-scaled returns
    mean_return: float      # mean forward return in this regime
    vol_return:  float      # vol of forward returns in this regime
    signal_mean: float      # mean signal value in this regime
    signal_std:  float      # std of signal in this regime

    def __repr__(self) -> str:
        return (f"RegimeStats(regime={self.regime}, n={self.n_obs}, "
                f"IC={self.ic:.4f}(t={self.ic_tstat:.2f}), "
                f"Sharpe={self.sharpe:.2f})")


@dataclass
class RegimeTesterResult:
    """Full result of RegimeTester.analyse()."""
    regime_stats:  Dict[int, RegimeStats]   # keyed by regime label
    overall_ic:    float                    # IC across all regimes
    overall_sharpe: float
    regime_counts: Dict[int, int]           # observations per regime
    n_regimes:     int
    horizon:       int

    def best_regime(self) -> int:
        """Regime with highest IC."""
        return max(self.regime_stats, key=lambda k: self.regime_stats[k].ic)

    def worst_regime(self) -> int:
        """Regime with lowest IC."""
        return min(self.regime_stats, key=lambda k: self.regime_stats[k].ic)

    def __repr__(self) -> str:
        lines = [f"RegimeTesterResult(n_regimes={self.n_regimes}, "
                 f"horizon={self.horizon}, overall_IC={self.overall_ic:.4f})"]
        for k, s in sorted(self.regime_stats.items()):
            lines.append(f"  {s}")
        return "\n".join(lines)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _spearman_ic(signal: np.ndarray, returns: np.ndarray) -> float:
    """Spearman rank-correlation IC, ignoring NaNs."""
    mask = np.isfinite(signal) & np.isfinite(returns)
    if mask.sum() < 3:
        return np.nan
    x = signal[mask]
    y = returns[mask]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    n  = len(rx)
    rx -= rx.mean(); ry -= ry.mean()
    sx = rx.std(); sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.dot(rx, ry) / (n * sx * sy))


def _ic_tstat(ic: float, n: int) -> float:
    """t-statistic for H0: IC = 0 (Fisher approximation)."""
    if n < 3 or not np.isfinite(ic):
        return np.nan
    denom = max(1.0 - ic ** 2, 1e-12) / max(n - 2, 1)
    return float(ic / np.sqrt(denom))


def _sharpe(returns: np.ndarray, scale: float = 1.0) -> float:
    """Annualised Sharpe ratio (scale = √periods_per_year)."""
    mask = np.isfinite(returns)
    if mask.sum() < 2:
        return np.nan
    r   = returns[mask]
    mu  = r.mean()
    sig = r.std(ddof=1)
    if sig < 1e-12:
        return np.nan
    return float(scale * mu / sig)


# ── RegimeTester ──────────────────────────────────────────────────────────────

class RegimeTester:
    """
    Evaluate signal quality across market regimes.

    Parameters
    ----------
    horizon     : forward-return look-ahead horizon (default 1).
                  IC is computed between signal[t] and sum(returns[t:t+horizon]).
    annual_scale : √(periods per year) for Sharpe annualisation (default √252).

    Usage
    -----
    >>> tester = RegimeTester(horizon=1)
    >>> result = tester.analyse(signal, returns, regime_labels)
    >>> print(result)
    """

    def __init__(
        self,
        horizon:      int   = 1,
        annual_scale: float = np.sqrt(252),
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon      = horizon
        self.annual_scale = annual_scale

    def analyse(
        self,
        signal:  np.ndarray,
        returns: np.ndarray,
        regimes: np.ndarray,
    ) -> RegimeTesterResult:
        """
        Compute regime-conditional signal statistics.

        Parameters
        ----------
        signal  : shape (T,) — alpha signal values.
        returns : shape (T,) — single-period return series.
        regimes : shape (T,) — integer regime labels (e.g. from HMM Viterbi).

        Returns
        -------
        RegimeTesterResult.
        """
        s   = np.asarray(signal,  dtype=float).ravel()
        r   = np.asarray(returns, dtype=float).ravel()
        reg = np.asarray(regimes, dtype=int).ravel()
        T   = len(s)

        if not (len(r) == len(reg) == T):
            raise ValueError("signal, returns, regimes must all have equal length")

        # Forward returns over `horizon` steps
        h = self.horizon
        fwd = np.full(T, np.nan)
        for t in range(T - h):
            fwd[t] = r[t : t + h].sum()

        # Overall IC
        overall_ic     = _spearman_ic(s, fwd)
        # Overall signal-scaled returns
        sig_rets       = s * np.where(np.isfinite(fwd), fwd, 0.0)
        overall_sharpe = _sharpe(sig_rets[np.isfinite(fwd)], self.annual_scale)

        # Per-regime statistics
        unique_regimes = sorted(np.unique(reg))
        regime_stats   = {}
        regime_counts  = {}

        for k in unique_regimes:
            mask = reg == k
            sk   = s[mask]
            rk   = fwd[mask]
            ret_k = r[mask]

            n_obs = int(mask.sum())
            regime_counts[k] = n_obs

            ic    = _spearman_ic(sk, rk)
            t_ic  = _ic_tstat(ic, n_obs)

            # Sharpe using signal-scaled returns in this regime
            sr_k = sk * np.where(np.isfinite(rk), rk, 0.0)
            valid = np.isfinite(rk)
            sh   = _sharpe(sr_k[valid], self.annual_scale) if valid.sum() > 1 else np.nan

            regime_stats[k] = RegimeStats(
                regime      = k,
                n_obs       = n_obs,
                ic          = float(ic) if np.isfinite(ic) else 0.0,
                ic_tstat    = float(t_ic) if np.isfinite(t_ic) else 0.0,
                sharpe      = float(sh) if np.isfinite(sh) else 0.0,
                mean_return = float(np.nanmean(rk)) if n_obs > 0 else 0.0,
                vol_return  = float(np.nanstd(rk, ddof=1)) if n_obs > 1 else 0.0,
                signal_mean = float(np.nanmean(sk)) if n_obs > 0 else 0.0,
                signal_std  = float(np.nanstd(sk, ddof=1)) if n_obs > 1 else 0.0,
            )

        return RegimeTesterResult(
            regime_stats   = regime_stats,
            overall_ic     = float(overall_ic) if np.isfinite(overall_ic) else 0.0,
            overall_sharpe = float(overall_sharpe) if np.isfinite(overall_sharpe) else 0.0,
            regime_counts  = regime_counts,
            n_regimes      = len(unique_regimes),
            horizon        = h,
        )

    def ic_decay(
        self,
        signal:  np.ndarray,
        returns: np.ndarray,
        max_horizon: int = 20,
    ) -> np.ndarray:
        """
        Compute IC at horizons 1..max_horizon (IC decay curve).

        Returns array of shape (max_horizon,) — IC[k] = IC at horizon k+1.
        """
        s = np.asarray(signal,  dtype=float).ravel()
        r = np.asarray(returns, dtype=float).ravel()
        T = len(s)
        ics = np.empty(max_horizon)
        for h in range(1, max_horizon + 1):
            fwd = np.full(T, np.nan)
            for t in range(T - h):
                fwd[t] = r[t : t + h].sum()
            ics[h - 1] = _spearman_ic(s, fwd)
        return ics
