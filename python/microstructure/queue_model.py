"""
Limit Order Queue Model
========================
References:
  Cont, Stoikov & Talreja (2010). "A Stochastic Model for Order Book Dynamics."
  Operations Research 58(3), 549-563.

  Avellaneda & Stoikov (2008). "High-frequency trading in a limit order book."
  Quantitative Finance 8(3), 217-224.  (Section 3 on queue position.)

Model summary:
  A limit order rests at price level p with Q_0 orders ahead of it in the queue.
  Market orders arrive and consume the front of the queue at Poisson rate λ.
  Each resting order independently cancels at rate θ (per order, per unit time).
  Our order fills when all Q_0 orders ahead have been either executed (by a
  market order) or cancelled.

  Simple closed-form (no cancellations):
    P(fill | Q_0, T) = P(Poisson(λT) ≥ Q_0) = 1 − CDF_{Pois(λT)}(Q_0 − 1)

  Expected time (with cancellations, competitive consumption model):
    E[T_fill] ≈ Q_0 / (λ + θ·Q_0)
    (each order ahead is consumed at rate λ/Q_0 by market orders
     and independently at rate θ by cancellation)
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.stats import poisson as scipy_poisson
from typing import Optional


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class QueueStats:
    """Fill-probability profile for a limit order in the queue."""
    fill_probability:        float         # P(fill) by horizon
    expected_fill_time:      float         # E[T_fill]
    fill_probability_by_time: np.ndarray  # CDF over time_grid
    time_grid:               np.ndarray   # times at which CDF is evaluated


# ── Fill probability ──────────────────────────────────────────────────────────

def fill_probability_poisson(
    queue_ahead:  int,
    arrival_rate: float,
    cancel_rate:  float,
    horizon:      float,
) -> float:
    """
    Probability that a limit order with `queue_ahead` orders ahead fills
    within [0, horizon], ignoring cancellations (pure Poisson model).

        P(fill | Q_0, T) = P(Poisson(λT) ≥ Q_0)
                         = 1 − Σ_{k=0}^{Q_0−1} e^{−λT}(λT)^k / k!

    cancel_rate is accepted for API consistency but ignored here; use
    fill_probability_with_cancels for a model that accounts for it.

    Parameters
    ----------
    queue_ahead  : orders ahead of ours (non-negative integer).
    arrival_rate : λ, market-order arrival rate hitting this side.
    cancel_rate  : θ, per-order cancellation rate (ignored in this fn).
    horizon      : time horizon in the same units as arrival_rate.

    Returns
    -------
    prob : float in [0, 1].
    """
    q = int(queue_ahead)
    if q <= 0:
        return 1.0
    expected = arrival_rate * horizon
    if expected <= 0:
        return 0.0
    return float(1.0 - scipy_poisson.cdf(q - 1, expected))


def fill_probability_with_cancels(
    queue_ahead:  int,
    arrival_rate: float,
    cancel_rate:  float,
    horizon:      float,
    n_sims:       int = 50_000,
    seed:         int = 0,
) -> float:
    """
    Monte Carlo estimate of fill probability accounting for cancellations.

    Each order ahead is independently cancelled at rate θ.  Market orders
    arrive at rate λ and execute the front of the remaining queue.

    Parameters
    ----------
    queue_ahead  : orders ahead of ours.
    arrival_rate : λ.
    cancel_rate  : θ (per order, per unit time).
    horizon      : simulation time horizon.
    n_sims       : Monte Carlo replications.
    seed         : RNG seed.

    Returns
    -------
    prob : estimated fill probability.
    """
    if queue_ahead <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    q   = int(queue_ahead)

    # Sample cancellation times for all orders ahead, all replications
    # Shape: (n_sims, q)  — Exp(θ) inter-event times
    if cancel_rate > 0:
        cancel_times = rng.exponential(1.0 / cancel_rate, size=(n_sims, q))
    else:
        cancel_times = np.full((n_sims, q), np.inf)

    # Number of market orders in [0, horizon] per replication
    n_market = rng.poisson(arrival_rate * horizon, size=n_sims)

    # Remaining queue = orders that have NOT been cancelled by horizon
    still_alive = np.sum(cancel_times > horizon, axis=1)   # (n_sims,)

    # Filled if enough market orders arrive to clear the surviving queue
    filled = n_market >= still_alive
    return float(filled.mean())


# ── Expected fill time ────────────────────────────────────────────────────────

def expected_fill_time(
    queue_ahead:  int,
    arrival_rate: float,
    cancel_rate:  float,
) -> float:
    """
    Approximate expected time to fill for a limit order with Q_0 ahead.

    Each order ahead exits the queue (by execution or cancellation) at
    rate (λ + θ).  The Q_0 orders drain sequentially, so:

        E[T_fill] ≈ Q_0 / (λ + θ)

    This is an upper bound; cancellations thin the queue faster, so the
    actual expected fill time is shorter when θ > 0.

    Parameters
    ----------
    queue_ahead  : orders ahead of ours.
    arrival_rate : λ.
    cancel_rate  : θ.

    Returns
    -------
    E[T_fill] in units of 1/(arrival_rate).
    """
    if queue_ahead <= 0:
        return 0.0
    rate_per_order = arrival_rate + cancel_rate
    if rate_per_order <= 0:
        return np.inf
    return float(queue_ahead / rate_per_order)


# ── Full CDF profile ──────────────────────────────────────────────────────────

def queue_fill_distribution(
    queue_ahead:  int,
    arrival_rate: float,
    cancel_rate:  float,
    horizon:      float,
    n_steps:      int = 200,
) -> QueueStats:
    """
    Compute the cumulative fill-probability profile over [0, horizon].

    Parameters
    ----------
    queue_ahead  : orders ahead of ours.
    arrival_rate : λ.
    cancel_rate  : θ (used for E[T_fill]; the CDF uses the Poisson model).
    horizon      : total time horizon.
    n_steps      : number of evaluation points in [0, horizon].

    Returns
    -------
    QueueStats with CDF and summary statistics.
    """
    time_grid = np.linspace(0.0, horizon, n_steps + 1)
    cdf = np.array([
        fill_probability_poisson(queue_ahead, arrival_rate, cancel_rate, t)
        for t in time_grid
    ])
    return QueueStats(
        fill_probability          = float(cdf[-1]),
        expected_fill_time        = expected_fill_time(queue_ahead, arrival_rate, cancel_rate),
        fill_probability_by_time  = cdf,
        time_grid                 = time_grid,
    )


# ── Optimal limit order depth ─────────────────────────────────────────────────

def optimal_limit_depth(
    fill_probs:        np.ndarray,
    price_improvement: np.ndarray,
) -> int:
    """
    Optimal tick distance from mid that maximises expected value:

        max_i  P(fill_i) × price_improvement_i

    Parameters
    ----------
    fill_probs        : P(fill) at each depth tick, shape (K,).
    price_improvement : price advantage over market order at each tick, shape (K,).

    Returns
    -------
    idx : index of the optimal depth (0 = one tick from mid).
    """
    fill_probs        = np.asarray(fill_probs,        dtype=float)
    price_improvement = np.asarray(price_improvement, dtype=float)
    score = fill_probs * price_improvement
    return int(np.argmax(score))


def queue_sensitivity(
    queue_ahead_range: np.ndarray,
    arrival_rate:      float,
    cancel_rate:       float,
    horizon:           float,
) -> np.ndarray:
    """
    Return fill probabilities for a range of queue positions.

    Useful for plotting how fill probability decays with queue depth.

    Parameters
    ----------
    queue_ahead_range : array of queue positions to evaluate.
    arrival_rate, cancel_rate, horizon : model parameters.

    Returns
    -------
    probs : shape (len(queue_ahead_range),).
    """
    return np.array([
        fill_probability_poisson(int(q), arrival_rate, cancel_rate, horizon)
        for q in queue_ahead_range
    ])
