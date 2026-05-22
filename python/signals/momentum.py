"""
Momentum Signals
================
Time-series and cross-sectional momentum indicators for equities.

References:
  Jegadeesh & Titman (1993) — cross-sectional momentum (buy recent winners).
  Moskowitz, Ooi & Pedersen (2012) — time-series momentum (TSMOM).
  Wilder (1978) — RSI.
  Appel (1979) — MACD.

All functions operate on numpy arrays (no pandas dependency) and preserve
array length with NaN fill for the warm-up period.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Time-Series Momentum (TSMOM) ──────────────────────────────────────────────

def time_series_momentum(
    returns:  np.ndarray,
    lookback: int,
    scale:    bool = True,
) -> np.ndarray:
    """
    Time-series momentum signal: sign of the cumulative lookback-period return,
    optionally scaled by the recent volatility (Moskowitz et al. 2012):

        signal_t = sign(R_{t-lookback:t}) / σ_{recent}

    When scale=False, returns raw cumulative return (not sign).

    Parameters
    ----------
    returns  : log-return series, shape (N,).
    lookback : number of bars to accumulate.
    scale    : if True, divide by recent (lookback-window) volatility.

    Returns
    -------
    signal : shape (N,); first `lookback` values are NaN.
    """
    r   = np.asarray(returns, dtype=float)
    n   = len(r)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        chunk   = r[i - lookback : i]
        cum_ret = np.nansum(chunk)
        if scale:
            vol = np.nanstd(chunk, ddof=1)
            out[i] = np.sign(cum_ret) / vol if vol > 0 else 0.0
        else:
            out[i] = cum_ret
    return out


def cross_sectional_momentum(
    returns_matrix: np.ndarray,
    lookback:       int,
) -> np.ndarray:
    """
    Cross-sectional momentum signal (Jegadeesh & Titman 1993).

    For each time step, rank assets by their `lookback`-period cumulative
    return and return a z-scored rank (long top, short bottom).

    Parameters
    ----------
    returns_matrix : shape (N, M) — M assets, N time periods.
    lookback       : formation period in bars.

    Returns
    -------
    signals : shape (N, M) — cross-sectional z-score ranks;
              first `lookback` rows are NaN.
    """
    R = np.asarray(returns_matrix, dtype=float)
    N, M = R.shape
    out = np.full((N, M), np.nan)
    for t in range(lookback, N):
        cum_ret = np.nansum(R[t - lookback : t, :], axis=0)  # (M,)
        valid   = np.isfinite(cum_ret)
        if valid.sum() < 2:
            continue
        mu  = cum_ret[valid].mean()
        sig = cum_ret[valid].std(ddof=1)
        if sig > 0:
            out[t, valid] = (cum_ret[valid] - mu) / sig
    return out


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(
    prices: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Wilder Relative Strength Index in [0, 100].

    Formula:
        RS   = AvgGain / AvgLoss
        RSI  = 100 − 100 / (1 + RS)

    Smoothing uses Wilder's method: α = 1/window (equivalent to EMA α=1/period).

    Returns
    -------
    rsi : shape (N,); first `window` values are NaN.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)
    out    = np.full(n, np.nan)
    if n < window + 1:
        return out

    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    for i in range(window, n - 1):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss == 0.0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return out


def rsi_signal(
    prices:     np.ndarray,
    window:     int   = 14,
    oversold:   float = 30.0,
    overbought: float = 70.0,
) -> np.ndarray:
    """
    Directional RSI signal:
        +1 when RSI < oversold   (buy)
        −1 when RSI > overbought (sell)
         0 otherwise

    Returns
    -------
    signal : integer array in {−1, 0, +1}, shape (N,).
    """
    r   = rsi(prices, window)
    out = np.zeros(len(prices), dtype=int)
    out[r < oversold]   =  1
    out[r > overbought] = -1
    return out


# ── MACD ──────────────────────────────────────────────────────────────────────

@dataclass
class MACDResult:
    macd:    np.ndarray   # fast EMA − slow EMA
    signal:  np.ndarray   # EMA of MACD
    hist:    np.ndarray   # MACD − signal (histogram)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with α = 2/(span+1)."""
    alpha = 2.0 / (span + 1)
    out   = np.full(len(series), np.nan)
    start = next((i for i in range(len(series)) if np.isfinite(series[i])), None)
    if start is None:
        return out
    out[start] = series[start]
    for i in range(start + 1, len(series)):
        if np.isfinite(series[i]):
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def macd(
    prices:      np.ndarray,
    fast:        int = 12,
    slow:        int = 26,
    signal_span: int = 9,
) -> MACDResult:
    """
    MACD (Moving Average Convergence/Divergence).

    macd    = EMA(fast) − EMA(slow)
    signal  = EMA(macd, signal_span)
    hist    = macd − signal

    Parameters
    ----------
    prices : close price series, shape (N,).

    Returns
    -------
    MACDResult with .macd, .signal, .hist arrays of shape (N,).
    """
    prices = np.asarray(prices, dtype=float)
    ema_f  = _ema(prices, fast)
    ema_s  = _ema(prices, slow)
    m      = ema_f - ema_s
    sig    = _ema(m, signal_span)
    hist   = m - sig
    return MACDResult(macd=m, signal=sig, hist=hist)


def macd_signal(result: MACDResult) -> np.ndarray:
    """
    Directional MACD histogram signal:
        +1 when histogram > 0 (bullish crossover)
        −1 when histogram < 0 (bearish crossover)
         0 when NaN

    Returns
    -------
    signal : integer array in {−1, 0, +1}.
    """
    out = np.zeros(len(result.hist), dtype=int)
    out[result.hist > 0] =  1
    out[result.hist < 0] = -1
    return out


# ── Rate of Change (ROC) ──────────────────────────────────────────────────────

def rate_of_change(
    prices:  np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Rate of Change:  ROC_t = (P_t − P_{t-n}) / P_{t-n}  × 100.
    First `lookback` values are NaN.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)
    out    = np.full(n, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        out[lookback:] = (prices[lookback:] - prices[:n - lookback]) / prices[:n - lookback] * 100.0
    return out


# ── Return autocorrelation signal ─────────────────────────────────────────────

def autocorrelation_signal(
    returns: np.ndarray,
    lag:     int = 1,
    window:  int = 60,
) -> np.ndarray:
    """
    Rolling lag-1 autocorrelation of returns as a regime signal.

    Positive autocorrelation → trending (use momentum).
    Negative autocorrelation → mean-reverting (fade moves).

    Computed over a rolling `window`; first (window-1) values are NaN.

    Returns
    -------
    autocorr : shape (N,) in [−1, +1].
    """
    r   = np.asarray(returns, dtype=float)
    n   = len(r)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = r[i - window + 1 : i + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) < lag + 2:
            continue
        mu  = valid.mean()
        vm  = valid - mu
        cov = np.dot(vm[lag:], vm[:-lag]) / len(vm[lag:])
        var = np.dot(vm, vm) / len(vm)
        out[i] = cov / var if var > 0 else 0.0
    return out


# ── Volume-weighted momentum ──────────────────────────────────────────────────

def volume_weighted_momentum(
    returns:  np.ndarray,
    volumes:  np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Volume-weighted cumulative return over `lookback` bars.

    Weights each bar's return by its volume fraction:
        signal_t = Σ_{i=t-n}^{t-1} r_i · V_i / Σ V_i

    Upweights bars with high participation, reducing noise from low-volume moves.
    """
    r   = np.asarray(returns, dtype=float)
    v   = np.asarray(volumes, dtype=float)
    n   = len(r)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        r_w = r[i - lookback : i]
        v_w = v[i - lookback : i]
        total_v = v_w.sum()
        if total_v > 0:
            out[i] = np.nansum(r_w * v_w) / total_v
    return out


# ── Combined momentum score ────────────────────────────────────────────────────

def composite_momentum(
    prices:   np.ndarray,
    windows:  Tuple[int, ...] = (20, 60, 120),
    scale:    bool            = True,
) -> np.ndarray:
    """
    Equal-weight combination of multiple lookback TSMOM signals,
    each z-scored across the window to normalise units.

    Returns a composite signal in roughly [−1, +1] range.
    """
    ret = np.concatenate([[np.nan], np.log(np.asarray(prices)[1:] / np.asarray(prices)[:-1])])
    components = []
    for w in windows:
        tsmom = time_series_momentum(ret, lookback=w, scale=scale)
        # Rolling z-score to normalise
        n   = len(tsmom)
        z   = np.full(n, np.nan)
        win = min(w * 2, n)
        for i in range(win - 1, n):
            chunk = tsmom[i - win + 1 : i + 1]
            valid = chunk[np.isfinite(chunk)]
            if len(valid) >= 2:
                mu  = valid.mean()
                sig = valid.std(ddof=1)
                if sig > 0:
                    z[i] = (tsmom[i] - mu) / sig
        components.append(z)

    # Stack and average, ignoring NaN
    mat    = np.stack(components, axis=0)  # (K, N)
    counts = np.sum(np.isfinite(mat), axis=0).astype(float)
    counts[counts == 0] = np.nan
    out    = np.nansum(mat, axis=0) / counts
    return out
