"""
Probability of Informed Trading (PIN) and VPIN
===============================================
References:
  Easley, Kiefer, O'Hara & Paperman (1996).
  "Liquidity, Information, and Infrequently Traded Stocks."
  Journal of Finance 51(4), 1405-1436.

  Easley, Lopez de Prado & O'Hara (2012).
  "Flow Toxicity and Liquidity in a High-frequency World."
  Review of Financial Studies 25(5), 1457-1493.

PIN Model:
----------
Each trading day, a private-information event occurs with probability α.
Given an event, the news is bad with probability δ (good with 1-δ).

Poisson arrival rates:
  Informed traders: μ  (only active on information days, on the informed side)
  Uninformed buys:  ε
  Uninformed sells: ε

Daily buy/sell counts (B_i, S_i) arise from a 3-component mixture:
  [bad news]  α·δ     · Pois(B; ε)   · Pois(S; ε+μ)
  [good news] α·(1-δ) · Pois(B; ε+μ) · Pois(S; ε)
  [no event]  (1-α)   · Pois(B; ε)   · Pois(S; ε)

PIN = αμ / (αμ + 2ε)
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from typing import Optional, Sequence


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PINResult:
    """MLE estimates for the PIN model."""
    alpha:          float   # prob. of information event per day
    delta:          float   # prob. bad news | information event
    mu:             float   # informed trader arrival rate
    epsilon:        float   # uninformed trader arrival rate (each side)
    log_likelihood: float
    converged:      bool

    @property
    def pin(self) -> float:
        """
        PIN = αμ / (αμ + 2ε)

        Fraction of order flow originating from informed traders.
        Values above ~0.20 indicate significant information asymmetry.
        """
        denom = self.alpha * self.mu + 2.0 * self.epsilon
        return (self.alpha * self.mu) / denom if denom > 0 else 0.0

    @property
    def theta(self) -> tuple:
        """Parameter vector (α, δ, μ, ε)."""
        return (self.alpha, self.delta, self.mu, self.epsilon)

    def __repr__(self) -> str:
        return (
            f"PINResult(α={self.alpha:.4f}, δ={self.delta:.4f}, "
            f"μ={self.mu:.2f}, ε={self.epsilon:.2f}, "
            f"PIN={self.pin:.4f}, LL={self.log_likelihood:.2f})"
        )


# ── Core likelihood ───────────────────────────────────────────────────────────

def _log_poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    """Log PMF of Poisson(λ) evaluated element-wise at k."""
    return k * np.log(lam + 1e-300) - lam - gammaln(k + 1.0)


def _log_sum_exp(X: np.ndarray) -> np.ndarray:
    """Numerically stable log-sum-exp over the last axis (rows → scalar)."""
    max_val = X.max(axis=-1, keepdims=True)
    return (max_val.squeeze(-1)
            + np.log(np.exp(X - max_val).sum(axis=-1) + 1e-300))


def pin_log_likelihood(
    params: Sequence[float],
    buys:   np.ndarray,
    sells:  np.ndarray,
) -> float:
    """
    Negative log-likelihood of the PIN model (for minimization).

    Computed in log-sum-exp for numerical stability; avoids overflow
    from raw Poisson PMF products on long daily series.

    Parameters
    ----------
    params : (α, δ, μ, ε)  — must all be ≥ 0, α ≤ 1, δ ≤ 1.
    buys, sells : daily buy / sell counts, shape (N_days,).

    Returns
    -------
    neg_log_lik : float (positive; minimize this).
    """
    alpha, delta, mu, epsilon = params

    # Parameter guard (optimizer may probe outside bounds briefly)
    if (alpha <= 0 or alpha >= 1 or delta <= 0 or delta >= 1
            or mu <= 0 or epsilon <= 0):
        return 1e18

    lam_informed = epsilon + mu   # arrival rate on the informed side

    lp_B_e  = _log_poisson_pmf(buys,  epsilon)       # log P(B | ε)
    lp_S_e  = _log_poisson_pmf(sells, epsilon)        # log P(S | ε)
    lp_B_em = _log_poisson_pmf(buys,  lam_informed)  # log P(B | ε+μ)
    lp_S_em = _log_poisson_pmf(sells, lam_informed)  # log P(S | ε+μ)

    # Three mixture components: bad news, good news, no event
    log_w_bad  = np.log(alpha * delta)
    log_w_good = np.log(alpha * (1.0 - delta))
    log_w_none = np.log(1.0 - alpha)

    # Per-day log-likelihoods for each scenario
    log_bad  = log_w_bad  + lp_B_e  + lp_S_em   # informed on sell side
    log_good = log_w_good + lp_B_em + lp_S_e    # informed on buy side
    log_none = log_w_none + lp_B_e  + lp_S_e    # no informed trading

    # Log-sum-exp over the three scenarios → per-day log-likelihood
    log_lik = _log_sum_exp(np.stack([log_bad, log_good, log_none], axis=1))
    return float(-np.sum(log_lik))


# ── MLE estimation ────────────────────────────────────────────────────────────

def estimate_pin(
    buys:     np.ndarray,
    sells:    np.ndarray,
    n_starts: int = 8,
    seed:     int = 42,
) -> PINResult:
    """
    Maximum-likelihood estimation of PIN model parameters.

    Uses multiple random restarts to avoid local optima; returns the
    global best (lowest negative log-likelihood) solution found.

    Parameters
    ----------
    buys, sells : daily buy/sell trade counts, shape (N_days,).
                  Minimum recommended: 30 days.
    n_starts    : random restarts (more → slower but more robust).
    seed        : RNG seed for reproducibility.

    Returns
    -------
    PINResult with MLE estimates.

    Raises
    ------
    RuntimeError if all starts fail to converge.
    """
    buys  = np.asarray(buys,  dtype=float)
    sells = np.asarray(sells, dtype=float)
    if len(buys) != len(sells):
        raise ValueError("buys and sells must have the same length")
    if len(buys) < 5:
        raise ValueError("Need at least 5 days to estimate PIN")

    rng = np.random.default_rng(seed)

    # Parameter bounds: (α, δ, μ, ε) all strictly positive; α, δ ≤ 1
    bounds = Bounds(
        lb=[1e-4, 1e-4,  0.1,  0.1],
        ub=[1-1e-4, 1-1e-4, 1e6, 1e6],
    )

    # Heuristic seed from method-of-moments (Easley et al.)
    mean_b, mean_s = buys.mean(), sells.mean()
    mu0  = abs(mean_b - mean_s) + 1.0
    eps0 = min(mean_b, mean_s)
    x0_mom = np.array([0.4, 0.5, mu0, eps0])

    starts = [x0_mom]
    for _ in range(n_starts - 1):
        starts.append(np.array([
            rng.uniform(0.1, 0.9),          # α
            rng.uniform(0.1, 0.9),          # δ
            rng.uniform(0.5, mean_b * 2),   # μ
            rng.uniform(0.5, mean_s * 2),   # ε
        ]))

    best_nll    = np.inf
    best_result = None

    for x0 in starts:
        try:
            res = minimize(
                pin_log_likelihood,
                x0,
                args=(buys, sells),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9},
            )
            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("PIN estimation: all optimizer starts failed.")

    alpha, delta, mu, epsilon = best_result.x
    return PINResult(
        alpha          = float(alpha),
        delta          = float(delta),
        mu             = float(mu),
        epsilon        = float(epsilon),
        log_likelihood = float(-best_nll),
        converged      = bool(best_result.success),
    )


# ── VPIN ──────────────────────────────────────────────────────────────────────

def vpin(
    buy_volume:  np.ndarray,
    sell_volume: np.ndarray,
    window:      int = 50,
) -> np.ndarray:
    """
    Volume-Synchronized PIN (VPIN) — Easley, Lopez de Prado & O'Hara (2012).

    VPIN_t = (1/n) Σ_{i=t-n+1}^{t} |V_i^buy − V_i^sell| / V_i^total

    Measures the fraction of volume that is "one-sided" (imbalanced),
    serving as a real-time order-flow toxicity proxy.

    Parameters
    ----------
    buy_volume, sell_volume : volume per volume-bar, shape (M,).
    window                  : rolling window length n.

    Returns
    -------
    vpin_series : shape (M,); first (window-1) values are NaN.
    """
    bv = np.asarray(buy_volume,  dtype=float)
    sv = np.asarray(sell_volume, dtype=float)
    total    = bv + sv
    toxicity = np.abs(bv - sv) / np.maximum(total, 1.0)

    n      = len(toxicity)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = toxicity[i - window + 1 : i + 1].mean()
    return result


def classify_trades_bulk(
    trade_prices: np.ndarray,
    mid_prices:   np.ndarray,
) -> np.ndarray:
    """
    Lee-Ready (1991) trade classification.

    Returns +1 (buyer-initiated) or −1 (seller-initiated) for each trade.
    Used to split trade volume into buy/sell buckets for VPIN.
    """
    trade = np.asarray(trade_prices, dtype=float)
    mid   = np.asarray(mid_prices,   dtype=float)
    side  = np.sign(trade - mid)
    # Tick rule fallback for trades exactly at mid
    for i in np.where(side == 0)[0]:
        if i > 0:
            side[i] = np.sign(trade[i] - trade[i - 1])
    side[side == 0] = 1.0   # default to buy if still ambiguous
    return side
