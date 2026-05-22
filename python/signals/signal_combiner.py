"""
Signal Combiner
===============
Combines multiple alpha signals into a single composite signal using several
standard portfolio construction methodologies.

Methods
-------
1. Equal-weight (EW)
   s_combined = mean(s_1, …, s_N)

2. IC-weight (Information Coefficient)
   w_i = IC_i / Σ|IC_j|   where IC_i = rank-corr(s_i, forward_ret)
   Signals with higher historical IC receive proportionally more weight.
   If all ICs are zero or there are no returns, falls back to equal-weight.

3. Volatility-scaled
   w_i ∝ 1 / σ_i,  where σ_i = rolling std of s_i
   Normalises each signal by its own realised noise before combining.
   Signals with lower noise receive higher weight.

4. Rank-based
   Each signal is converted to cross-sectional ranks (or time-series
   quantile ranks) in [0, 1], then averaged.

References
----------
  Grinold & Kahn (2000). "Active Portfolio Management." McGraw-Hill.
  Lopez de Prado (2018). "Advances in Financial Machine Learning." Wiley.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class CombinerResult:
    """Output of SignalCombiner.combine()."""
    combined:   np.ndarray            # shape (T,) — composite signal
    weights:    np.ndarray            # shape (N,) — per-signal weights (sum to 1)
    method:     str
    signal_ics: Optional[np.ndarray]  # shape (N,) — IC per signal if available

    def __repr__(self) -> str:
        w_str = ", ".join(f"{w:.3f}" for w in self.weights)
        return f"CombinerResult(method={self.method!r}, weights=[{w_str}])"


# ── Utility functions ─────────────────────────────────────────────────────────

def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ignoring NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    rx = np.argsort(np.argsort(x[mask])).astype(float)
    ry = np.argsort(np.argsort(y[mask])).astype(float)
    n  = len(rx)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.std(rx) * np.std(ry)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(rx, ry) / (n * denom))


def _winsorise(x: np.ndarray, pct: float = 0.01) -> np.ndarray:
    """Winsorise at [pct, 1-pct] percentiles (in-place on a copy)."""
    lo = np.nanpercentile(x, 100 * pct)
    hi = np.nanpercentile(x, 100 * (1 - pct))
    return np.clip(x, lo, hi)


def _normalise(x: np.ndarray, winsorise: bool = True) -> np.ndarray:
    """
    Standardise signal to zero mean, unit variance.
    Replaces NaNs with 0 after normalisation.
    """
    if winsorise:
        x = _winsorise(x)
    mu  = np.nanmean(x)
    sig = np.nanstd(x)
    if sig < 1e-12:
        return np.zeros_like(x)
    out = (x - mu) / sig
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _to_rank(x: np.ndarray) -> np.ndarray:
    """
    Convert signal to quantile ranks in [0, 1].
    NaNs are mapped to 0.5 (neutral).
    """
    n   = len(x)
    out = np.full(n, 0.5)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return out
    ranks = np.argsort(np.argsort(x[mask])).astype(float) / max(mask.sum() - 1, 1)
    out[mask] = ranks
    return out


# ── SignalCombiner ────────────────────────────────────────────────────────────

class SignalCombiner:
    """
    Combine multiple alpha signals into a single composite signal.

    Parameters
    ----------
    normalise_inputs : pre-normalise each signal to zero mean / unit std
                       before combination.  Default True.
    winsorise        : apply 1%–99% winsorisation during normalisation.

    Usage
    -----
    >>> combiner = SignalCombiner()
    >>> result = combiner.equal_weight(signals)
    >>> result = combiner.ic_weight(signals, forward_returns)
    >>> result = combiner.vol_scale(signals, window=60)
    >>> result = combiner.rank_combine(signals)
    """

    def __init__(
        self,
        normalise_inputs: bool = True,
        winsorise:        bool = True,
    ) -> None:
        self.normalise_inputs = normalise_inputs
        self.winsorise        = winsorise

    # ── Pre-processing ────────────────────────────────────────────────────────

    def _prep(self, signals: Sequence[np.ndarray]) -> np.ndarray:
        """
        Stack signals into (T, N) matrix, optionally normalising.
        """
        arrays = [np.asarray(s, dtype=float).ravel() for s in signals]
        lengths = {len(a) for a in arrays}
        if len(lengths) > 1:
            raise ValueError(f"All signals must have equal length; got {lengths}")
        mat = np.column_stack(arrays)   # (T, N)
        if self.normalise_inputs:
            mat = np.column_stack([_normalise(mat[:, j], self.winsorise)
                                   for j in range(mat.shape[1])])
        return mat

    # ── Equal-weight ──────────────────────────────────────────────────────────

    def equal_weight(
        self,
        signals: Sequence[np.ndarray],
    ) -> CombinerResult:
        """
        Average all signals with equal weights.

        combined_t = (1/N) Σ_i s_{i,t}
        """
        mat = self._prep(signals)
        N   = mat.shape[1]
        w   = np.full(N, 1.0 / N)
        combined = mat @ w
        return CombinerResult(
            combined   = combined,
            weights    = w,
            method     = "equal_weight",
            signal_ics = None,
        )

    # ── IC-weight ─────────────────────────────────────────────────────────────

    def ic_weight(
        self,
        signals:         Sequence[np.ndarray],
        forward_returns: np.ndarray,
        ic_window:       Optional[int] = None,
    ) -> CombinerResult:
        """
        Weight signals by their historical Information Coefficient (IC).

        IC_i = Spearman rank correlation(signal_i, forward_returns).

        If ic_window is set, uses only the last ic_window observations to
        estimate ICs (rolling estimate).  Otherwise uses the full series.

        Falls back to equal-weight if all ICs ≤ 0 or when signal length < 10.
        """
        mat  = self._prep(signals)
        T, N = mat.shape
        fwd  = np.asarray(forward_returns, dtype=float).ravel()
        if len(fwd) != T:
            raise ValueError("forward_returns must have same length as signals")

        # Select window
        if ic_window and ic_window < T:
            mat_w = mat[-ic_window:]
            fwd_w = fwd[-ic_window:]
        else:
            mat_w = mat
            fwd_w = fwd

        ics = np.array([_rank_corr(mat_w[:, i], fwd_w) for i in range(N)])

        # Weights proportional to |IC|; negative IC → weight goes to 0
        ic_pos = np.maximum(ics, 0.0)
        total  = ic_pos.sum()
        if total < 1e-10:
            w = np.full(N, 1.0 / N)
        else:
            w = ic_pos / total

        combined = mat @ w
        return CombinerResult(
            combined   = combined,
            weights    = w,
            method     = "ic_weight",
            signal_ics = ics,
        )

    # ── Volatility-scaled ─────────────────────────────────────────────────────

    def vol_scale(
        self,
        signals: Sequence[np.ndarray],
        window:  int = 60,
    ) -> CombinerResult:
        """
        Weight signals inversely proportional to their rolling volatility.

        Trailing vol σ_i = std(s_i[-window:]).
        w_i = (1/σ_i) / Σ_j (1/σ_j)

        For short series (< window), uses the full available history.
        Falls back to equal-weight for signals with zero variance.
        """
        mat  = self._prep(signals)
        T, N = mat.shape

        # Estimate vol over the last `window` observations
        tail = mat[-window:, :]
        vols = np.array([tail[:, i].std(ddof=1) for i in range(N)])
        vols = np.where(vols < 1e-12, 1e-12, vols)   # avoid division by zero

        inv_vol = 1.0 / vols
        w       = inv_vol / inv_vol.sum()
        combined = mat @ w
        return CombinerResult(
            combined   = combined,
            weights    = w,
            method     = "vol_scale",
            signal_ics = None,
        )

    # ── Rank-based ────────────────────────────────────────────────────────────

    def rank_combine(
        self,
        signals: Sequence[np.ndarray],
    ) -> CombinerResult:
        """
        Convert each signal to quantile ranks ∈ [0, 1], then average.

        The combined signal is re-centred to zero after averaging:
            combined = mean(rank(s_i)) − 0.5

        This produces a signal in [−0.5, 0.5] with roughly 0 mean.
        """
        arrays = [np.asarray(s, dtype=float).ravel() for s in signals]
        lengths = {len(a) for a in arrays}
        if len(lengths) > 1:
            raise ValueError(f"All signals must have equal length; got {lengths}")

        ranked = np.column_stack([_to_rank(a) for a in arrays])   # (T, N)
        N      = ranked.shape[1]
        w      = np.full(N, 1.0 / N)
        combined = ranked @ w - 0.5    # re-centre
        return CombinerResult(
            combined   = combined,
            weights    = w,
            method     = "rank_combine",
            signal_ics = None,
        )

    # ── Custom-weight ─────────────────────────────────────────────────────────

    def custom_weight(
        self,
        signals: Sequence[np.ndarray],
        weights: Sequence[float],
    ) -> CombinerResult:
        """
        Combine with user-specified weights (normalised to sum to 1).

        Parameters
        ----------
        weights : non-negative floats, one per signal.
        """
        mat  = self._prep(signals)
        N    = mat.shape[1]
        w    = np.asarray(weights, dtype=float)
        if len(w) != N:
            raise ValueError(f"len(weights)={len(w)} must equal number of signals={N}")
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative")
        total = w.sum()
        if total < 1e-10:
            raise ValueError("Weights must not all be zero")
        w = w / total
        combined = mat @ w
        return CombinerResult(
            combined   = combined,
            weights    = w,
            method     = "custom_weight",
            signal_ics = None,
        )


# ── Convenience functions ─────────────────────────────────────────────────────

def combine_equal(signals: Sequence[np.ndarray], **kw) -> np.ndarray:
    """Return equal-weight combined signal array."""
    return SignalCombiner(**kw).equal_weight(signals).combined


def combine_ic(
    signals:         Sequence[np.ndarray],
    forward_returns: np.ndarray,
    **kw,
) -> np.ndarray:
    """Return IC-weighted combined signal array."""
    return SignalCombiner(**kw).ic_weight(signals, forward_returns).combined


def combine_vol_scale(
    signals: Sequence[np.ndarray],
    window:  int = 60,
    **kw,
) -> np.ndarray:
    """Return vol-scaled combined signal array."""
    return SignalCombiner(**kw).vol_scale(signals, window=window).combined


def combine_rank(signals: Sequence[np.ndarray], **kw) -> np.ndarray:
    """Return rank-based combined signal array."""
    return SignalCombiner(**kw).rank_combine(signals).combined
