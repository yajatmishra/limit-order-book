"""
Order Flow Imbalance (OFI)
==========================
Reference:
  Cont, Kukanov & Stoikov (2014). "The Price Impact of Order Book Events."
  Journal of Financial Econometrics 12(1), 47-88.

Single-level OFI captures the signed net-pressure at the best quotes:

  e_t^bid = Q_t^b · 1{P_t^b ≥ P_{t-1}^b} − Q_{t-1}^b · 1{P_t^b ≤ P_{t-1}^b}
  e_t^ask = Q_t^a · 1{P_t^a ≤ P_{t-1}^a} − Q_{t-1}^a · 1{P_t^a ≥ P_{t-1}^a}
  OFI_t   = e_t^bid − e_t^ask

Positive OFI → net buying pressure; negative OFI → net selling pressure.

Multi-level OFI weights contributions across the first L levels of the book,
capturing deeper order-book dynamics ignored by single-level measures.
"""

import numpy as np
from typing import Optional


# ── Single-level OFI ──────────────────────────────────────────────────────────

def compute_ofi(
    bid_prices: np.ndarray,
    bid_sizes:  np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes:  np.ndarray,
) -> np.ndarray:
    """
    Single-level Order Flow Imbalance (best bid/ask).

    Parameters
    ----------
    bid_prices, bid_sizes : shape (N,)  — best bid price and size at each tick.
    ask_prices, ask_sizes : shape (N,)  — best ask price and size at each tick.

    Returns
    -------
    ofi : np.ndarray, shape (N-1,)
        OFI indexed to the *later* observation of each consecutive pair.
    """
    bp = np.asarray(bid_prices, dtype=float)
    bq = np.asarray(bid_sizes,  dtype=float)
    ap = np.asarray(ask_prices, dtype=float)
    aq = np.asarray(ask_sizes,  dtype=float)

    if len(bp) < 2:
        return np.array([], dtype=float)

    # Bid-side event: positive when bid improves (price rises or size added)
    e_bid = (bq[1:] * (bp[1:] >= bp[:-1]).astype(float)
           - bq[:-1] * (bp[1:] <= bp[:-1]).astype(float))

    # Ask-side event: positive when ask improves (price falls or size added)
    e_ask = (aq[1:] * (ap[1:] <= ap[:-1]).astype(float)
           - aq[:-1] * (ap[1:] >= ap[:-1]).astype(float))

    return e_bid - e_ask


# ── Multi-level OFI ───────────────────────────────────────────────────────────

def compute_multi_level_ofi(
    bids:    np.ndarray,
    asks:    np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Multi-level Order Flow Imbalance.

    Aggregates OFI across the first L price levels, weighted by `weights`.
    Using uniform weights (1/L) matches the Cont et al. baseline; exponential
    decay weights can be used to up-weight near-touch levels.

    Parameters
    ----------
    bids : np.ndarray, shape (N, L, 2)
        bids[:, lev, 0] = price at level `lev`, bids[:, lev, 1] = size.
    asks : np.ndarray, shape (N, L, 2)
    weights : shape (L,), optional.  Defaults to uniform (1/L).

    Returns
    -------
    ofi : np.ndarray, shape (N-1,)
    """
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    N, L, _ = bids.shape

    if weights is None:
        weights = np.ones(L, dtype=float) / L
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (L,):
        raise ValueError(f"weights must have shape ({L},); got {weights.shape}")

    ofi_levels = np.zeros((N - 1, L), dtype=float)
    for lev in range(L):
        ofi_levels[:, lev] = compute_ofi(
            bids[:, lev, 0], bids[:, lev, 1],
            asks[:, lev, 0], asks[:, lev, 1],
        )

    return ofi_levels @ weights   # (N-1,)


# ── Normalization and rolling aggregation ─────────────────────────────────────

def normalized_ofi(
    ofi:          np.ndarray,
    total_volume: np.ndarray,
    min_vol:      float = 1.0,
) -> np.ndarray:
    """
    OFI scaled by contemporaneous total quoted or traded volume.

    Dividing by volume makes OFI comparable across stocks with different
    tick sizes or average trade sizes.

    Parameters
    ----------
    ofi          : shape (M,)
    total_volume : shape (M,)  — must match ofi length.
    min_vol      : floor to avoid division by zero.

    Returns
    -------
    norm_ofi : shape (M,), values in [−1, +1] when volume = bid+ask size.
    """
    ofi = np.asarray(ofi, dtype=float)
    vol = np.maximum(np.asarray(total_volume, dtype=float), min_vol)
    return ofi / vol


def rolling_ofi(
    ofi:    np.ndarray,
    window: int,
    agg:    str = "mean",
) -> np.ndarray:
    """
    Causal rolling aggregation of OFI — no look-ahead.

    Parameters
    ----------
    ofi    : raw OFI series, shape (M,).
    window : lookback length in ticks.
    agg    : "mean" | "sum".

    Returns
    -------
    result : shape (M,); first (window-1) entries are NaN.
    """
    ofi = np.asarray(ofi, dtype=float)
    n = len(ofi)
    result = np.full(n, np.nan)
    if window < 1:
        raise ValueError("window must be >= 1")
    for i in range(window - 1, n):
        chunk = ofi[i - window + 1 : i + 1]
        result[i] = chunk.mean() if agg == "mean" else chunk.sum()
    return result


# ── Price-impact regression ───────────────────────────────────────────────────

def ofi_price_impact_regression(
    ofi:               np.ndarray,
    mid_price_changes: np.ndarray,
) -> tuple:
    """
    Regress midprice changes on contemporaneous OFI (Cont et al. 2014, Eq. 1):

        ΔS_t = β · OFI_t + ε_t

    β is the price impact per unit of order-flow imbalance.

    Parameters
    ----------
    ofi               : OFI series, shape (M,).
    mid_price_changes : ΔS series, shape (M,).

    Returns
    -------
    (beta, r_squared) : OLS coefficient and in-sample R².
    """
    ofi = np.asarray(ofi, dtype=float)
    y   = np.asarray(mid_price_changes, dtype=float)
    X   = ofi.reshape(-1, 1)

    beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
    y_hat  = ofi * beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta), float(r2)


# ── Convenience: OFI from tick-by-tick DataFrame-like arrays ──────────────────

def ofi_from_book_snapshots(
    snapshots: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper that unpacks a (N, 4) array of
    [bid_price, bid_size, ask_price, ask_size] and returns the OFI series.
    """
    snaps = np.asarray(snapshots, dtype=float)
    if snaps.ndim != 2 or snaps.shape[1] != 4:
        raise ValueError("snapshots must have shape (N, 4): [bp, bq, ap, aq]")
    return compute_ofi(snaps[:, 0], snaps[:, 1], snaps[:, 2], snaps[:, 3])
