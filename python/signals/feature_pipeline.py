"""
Feature Pipeline
================
Transforms raw OHLCV and tick data into a named feature matrix for
downstream signal models.  All functions are pure numpy; no pandas required.

Features produced
-----------------
  log_ret       : log returns log(P_t / P_{t-1})
  ewma_vol      : exponentially-weighted realised volatility
  rolling_vol   : rolling-window realised volatility
  zscore        : (price − rolling_mean) / rolling_std
  rsi           : Wilder RSI in [0, 100]
  bb_zscore     : Bollinger Band position  (price − mid) / (n_std × σ)
  momentum_N    : N-period raw return P_t / P_{t-N} − 1  (one per window)
  vwap_dev      : fractional deviation from rolling VWAP (requires volume)

First `window − 1` values of rolling features are NaN (no look-ahead).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    # Volatility
    vol_window:       int   = 20
    ewma_halflife:    float = 10.0    # EWMA half-life in bars

    # Mean-reversion / Bollinger
    bb_window:        int   = 20
    bb_std:           float = 2.0
    zscore_window:    int   = 20

    # Momentum lookbacks (multiple)
    momentum_windows: Tuple[int, ...] = (5, 20, 60)

    # RSI
    rsi_window:       int   = 14

    # VWAP
    vwap_window:      int   = 20


# ── Scalar feature functions ──────────────────────────────────────────────────

def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Log returns r_t = log(P_t / P_{t-1}).
    Returns array of same length; first element is NaN.
    """
    prices = np.asarray(prices, dtype=float)
    out    = np.full(len(prices), np.nan)
    out[1:] = np.log(prices[1:] / prices[:-1])
    return out


def ewma_volatility(
    returns:  np.ndarray,
    halflife: float = 10.0,
) -> np.ndarray:
    """
    Exponentially-weighted moving-average volatility (annualised by √252 if
    returns are daily; raw otherwise).

    Uses RiskMetrics-style EWMA:  σ²_t = λ·σ²_{t-1} + (1−λ)·r²_{t-1}
    where λ = exp(−ln2 / halflife).

    Returns
    -------
    vol : np.ndarray, same length as returns; first element is NaN.
    """
    r   = np.asarray(returns, dtype=float)
    lam = np.exp(-np.log(2.0) / halflife)
    n   = len(r)
    var = np.full(n, np.nan)
    # Seed with first non-NaN squared return
    start = next((i for i in range(n) if np.isfinite(r[i])), None)
    if start is None:
        return var
    var[start] = r[start] ** 2
    for t in range(start + 1, n):
        if np.isnan(r[t]):
            var[t] = var[t - 1]
        else:
            var[t] = lam * var[t - 1] + (1.0 - lam) * r[t] ** 2
    return np.sqrt(var)


def rolling_volatility(
    returns: np.ndarray,
    window:  int,
) -> np.ndarray:
    """
    Rolling standard deviation of returns over `window` bars.
    First (window-1) values are NaN.
    """
    r   = np.asarray(returns, dtype=float)
    n   = len(r)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = r[i - window + 1 : i + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) >= 2:
            out[i] = valid.std(ddof=1)
    return out


def rolling_zscore(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling z-score:  z_t = (x_t − mean_{t−w:t}) / std_{t−w:t}.
    First (window-1) values are NaN.  Uses ddof=1 for std.
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


def rsi(
    prices: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Wilder RSI (Relative Strength Index) in [0, 100].

    Uses the Wilder smoothing method (exponential with α = 1/window):
        AvgGain_t = ((window−1) · AvgGain_{t-1} + gain_t) / window

    First (window) values are NaN.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)
    out    = np.full(n, np.nan)

    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if n < window + 1:
        return out

    # Seed averages from first `window` bars
    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    for i in range(window, n - 1):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


def bollinger_zscore(
    prices: np.ndarray,
    window: int   = 20,
    n_std:  float = 2.0,
) -> np.ndarray:
    """
    Bollinger Band z-score: position of price within the bands.

        bb_z_t = (P_t − mid_t) / (n_std · σ_t)

    Returns values in [−1, +1] when price is within the bands.
    First (window-1) values are NaN.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)
    out    = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = prices[i - window + 1 : i + 1]
        mu    = chunk.mean()
        sig   = chunk.std(ddof=1)
        if sig > 0:
            out[i] = (prices[i] - mu) / (n_std * sig)
    return out


def momentum_return(
    prices:  np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Simple `lookback`-period raw return P_t / P_{t-n} − 1.
    First `lookback` values are NaN.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)
    out    = np.full(n, np.nan)
    out[lookback:] = prices[lookback:] / prices[:n - lookback] - 1.0
    return out


def vwap_deviation(
    prices:  np.ndarray,
    volumes: np.ndarray,
    window:  int = 20,
) -> np.ndarray:
    """
    Fractional deviation from the rolling VWAP:

        dev_t = (P_t − VWAP_t) / VWAP_t

    VWAP_t = Σ_{i=t−w+1}^{t} P_i · V_i / Σ V_i

    First (window-1) values are NaN.
    """
    prices  = np.asarray(prices,  dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n       = len(prices)
    out     = np.full(n, np.nan)
    for i in range(window - 1, n):
        p_w = prices [i - window + 1 : i + 1]
        v_w = volumes[i - window + 1 : i + 1]
        total_vol = v_w.sum()
        if total_vol > 0:
            vwap   = (p_w * v_w).sum() / total_vol
            out[i] = (prices[i] - vwap) / vwap if vwap != 0 else 0.0
    return out


# ── FeaturePipeline ────────────────────────────────────────────────────────────

class FeaturePipeline:
    """
    Computes a dictionary of named feature arrays from a price (and optional
    volume) series.

    Usage
    -----
    >>> pipe = FeaturePipeline(FeatureConfig(vol_window=20, rsi_window=14))
    >>> feats = pipe.transform(prices, volumes)
    >>> zscore = feats["zscore"]
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()) -> None:
        self.cfg = config

    def transform(
        self,
        prices:  np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for the supplied price series.

        Parameters
        ----------
        prices  : close prices, shape (N,).
        volumes : traded volumes, shape (N,).  Required for VWAP deviation.

        Returns
        -------
        features : dict mapping feature name → np.ndarray of shape (N,).
        """
        cfg = self.cfg
        ret = log_returns(prices)

        features: Dict[str, np.ndarray] = {
            "log_ret":    ret,
            "ewma_vol":   ewma_volatility(ret, halflife=cfg.ewma_halflife),
            "rolling_vol":rolling_volatility(ret, window=cfg.vol_window),
            "zscore":     rolling_zscore(prices, window=cfg.zscore_window),
            "rsi":        rsi(prices, window=cfg.rsi_window),
            "bb_zscore":  bollinger_zscore(prices, window=cfg.bb_window,
                                           n_std=cfg.bb_std),
        }

        for w in cfg.momentum_windows:
            features[f"momentum_{w}"] = momentum_return(prices, lookback=w)

        if volumes is not None:
            features["vwap_dev"] = vwap_deviation(prices, volumes,
                                                   window=cfg.vwap_window)

        return features

    def feature_names(self, with_volume: bool = False) -> List[str]:
        """Return the list of feature names that transform() will produce."""
        names = ["log_ret", "ewma_vol", "rolling_vol", "zscore",
                 "rsi", "bb_zscore"]
        names += [f"momentum_{w}" for w in self.cfg.momentum_windows]
        if with_volume:
            names.append("vwap_dev")
        return names
