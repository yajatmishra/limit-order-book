"""
Bid-Ask Spread Decomposition
=============================
References:
  Roll (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread."
  Journal of Finance 39(4), 1127-1139.

  Kyle (1985). "Continuous Auctions and Insider Trading."
  Econometrica 53(6), 1315-1335.

  Amihud (2002). "Illiquidity and Stock Returns."
  Journal of Financial Markets 5(1), 31-56.

  Glosten & Harris (1988). "Estimating the Components of the Bid-Ask Spread."
  Journal of Financial Economics 21(1), 123-142.

Spread components:
  Effective spread = adverse selection + inventory + order processing costs.

Roll estimator recovers effective spread from the serial covariance of
transaction price changes.  Kyle lambda measures price impact per unit of
signed order flow (the adverse selection component in reduced form).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Roll (1984) effective spread ──────────────────────────────────────────────

def roll_spread(mid_prices: np.ndarray) -> float:
    """
    Roll (1984) implicit effective spread estimator.

    Under the Roll model the transaction price switches between Bid = V − c
    and Ask = V + c (V = fundamental value, c = half-spread).  This induces
    a negative first-order serial covariance in price changes:

        Cov(ΔP_t, ΔP_{t+1}) = −c²

    Hence:  c = √(−Cov)  →  effective spread = 2c.

    Parameters
    ----------
    mid_prices : array of transaction / mid-quote prices.

    Returns
    -------
    effective_spread : 2c in price units.  NaN when covariance is non-negative
                       (violates Roll model assumptions).
    """
    prices = np.asarray(mid_prices, dtype=float)
    dp = np.diff(prices)
    if len(dp) < 2:
        return np.nan
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    if cov >= 0:
        return np.nan          # Roll model requires negative serial covariance
    return 2.0 * np.sqrt(-cov)


def roll_spread_robust(
    mid_prices: np.ndarray,
    quantile:   float = 0.25,
) -> float:
    """
    Robust Roll spread using the lower quantile of the cross-product
    dp_t · dp_{t+1} instead of the mean, which is less sensitive to
    large transient price moves that flip the sign of the covariance.

    Parameters
    ----------
    quantile : lower quantile of negative cross-products to use (default 0.25).

    Returns
    -------
    effective_spread : 2√(−q) where q is the chosen quantile.
    """
    prices = np.asarray(mid_prices, dtype=float)
    dp = np.diff(prices)
    cross = dp[:-1] * dp[1:]
    neg   = cross[cross < 0]
    if len(neg) == 0:
        return np.nan
    c2 = -np.quantile(neg, quantile)
    return 2.0 * np.sqrt(c2)


# ── Kyle (1985) lambda ────────────────────────────────────────────────────────

def kyle_lambda(
    price_changes: np.ndarray,
    order_flow:    np.ndarray,
) -> float:
    """
    Kyle (1985) price-impact coefficient λ.

    The linear-impact model posits:

        ΔP_t = λ · x_t + ε_t

    where x_t is signed order flow (buy volume − sell volume).
    λ is estimated by OLS.  A larger λ indicates a less liquid market
    in which informed order flow moves prices more.

    Parameters
    ----------
    price_changes : array of mid-price changes ΔP, shape (M,).
    order_flow    : signed volume (buy − sell) per bar, shape (M,).

    Returns
    -------
    lambda_ : OLS price-impact coefficient (units: price / share).
    """
    dp = np.asarray(price_changes, dtype=float)
    of = np.asarray(order_flow,    dtype=float)
    X  = of.reshape(-1, 1)
    beta = np.linalg.lstsq(X, dp, rcond=None)[0][0]
    return float(beta)


# ── Amihud (2002) illiquidity ─────────────────────────────────────────────────

def amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray,
    scale:   float = 1e6,
) -> np.ndarray:
    """
    Amihud (2002) daily illiquidity ratio.

        ILLIQ_t = |r_t| / (DollarVolume_t / scale)

    Measures price impact per dollar traded; commonly scaled per million.
    Higher values → less liquid.

    Parameters
    ----------
    returns : daily returns (fractional), shape (N,).
    volumes : daily dollar trading volumes, shape (N,).
    scale   : normalisation constant (default 1e6 = per million dollars).

    Returns
    -------
    illiq : shape (N,), per-day illiquidity ratios.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(volumes, dtype=float)
    return np.abs(r) / (v / scale + 1e-300)


# ── Effective spread and decomposition from trade-level data ──────────────────

def effective_spread(
    trade_prices: np.ndarray,
    mid_prices:   np.ndarray,
    side:         np.ndarray,
) -> float:
    """
    Effective spread (Lee & Ready 1991).

        ES_t = 2 · side_t · (trade_t − mid_t)

    Returns the cross-sectional mean effective spread.

    Parameters
    ----------
    trade_prices : price at which each trade executed.
    mid_prices   : quote midpoint at time of trade.
    side         : +1 buyer-initiated, −1 seller-initiated.
    """
    t = np.asarray(trade_prices, dtype=float)
    m = np.asarray(mid_prices,   dtype=float)
    s = np.asarray(side,         dtype=float)
    return float(np.mean(2.0 * s * (t - m)))


def realized_spread(
    trade_prices: np.ndarray,
    future_mids:  np.ndarray,
    side:         np.ndarray,
) -> float:
    """
    Realized spread: dealer revenue after controlling for price impact.

        RS_t = 2 · side_t · (trade_t − mid_{t+τ})

    where mid_{t+τ} is the midpoint τ periods after the trade.
    Approximates what the market-maker nets after adverse selection.
    """
    t = np.asarray(trade_prices, dtype=float)
    f = np.asarray(future_mids,  dtype=float)
    s = np.asarray(side,         dtype=float)
    return float(np.mean(2.0 * s * (t - f)))


def price_impact_from_trades(
    mid_prices:  np.ndarray,
    future_mids: np.ndarray,
    side:        np.ndarray,
) -> float:
    """
    Price impact (adverse selection cost).

        PI_t = 2 · side_t · (mid_{t+τ} − mid_t)

    Positive means price moved against the passive side (market maker loses).
    """
    m = np.asarray(mid_prices,  dtype=float)
    f = np.asarray(future_mids, dtype=float)
    s = np.asarray(side,        dtype=float)
    return float(np.mean(2.0 * s * (f - m)))


@dataclass
class SpreadDecomposition:
    """Glosten-Harris (1988) style decomposition of the effective spread."""
    effective:        float   # total effective half-spread
    adverse_selection:float   # informed-trading component (price impact)
    realized:         float   # realized spread (dealer revenue proxy)

    @property
    def adverse_selection_fraction(self) -> float:
        """Fraction of effective spread attributable to adverse selection."""
        if abs(self.effective) < 1e-12:
            return 0.0
        return self.adverse_selection / self.effective


def decompose_spread(
    trade_prices: np.ndarray,
    mid_prices:   np.ndarray,
    future_mids:  np.ndarray,
    side:         np.ndarray,
) -> SpreadDecomposition:
    """
    Decompose the effective spread into adverse selection and realized spread.

    Identity:  effective_spread = price_impact + realized_spread.
    """
    es = effective_spread(trade_prices, mid_prices, side)
    pi = price_impact_from_trades(mid_prices, future_mids, side)
    rs = realized_spread(trade_prices, future_mids, side)
    return SpreadDecomposition(
        effective         = es,
        adverse_selection = pi,
        realized          = rs,
    )
