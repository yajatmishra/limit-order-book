"""
Avellaneda-Stoikov (2008) Optimal Market-Making Model
=====================================================
Reference:
  Avellaneda, M. & Stoikov, S. (2008).
  "High-frequency trading in a limit order book."
  Quantitative Finance 8(3), 217-224.

The AS model solves the market-maker's utility-maximisation problem
under inventory risk.  The closed-form solution gives:

  Reservation price:   r(s, q, t) = s − q · γ · σ² · (T−t)
  Optimal half-spread: δ*(t)       = (γ · σ²· (T−t)) / 2
                                    + (1/γ) · ln(1 + γ/k)
  Bid quote:           b*(t) = r(t) − δ*(t)
  Ask quote:           a*(t) = r(t) + δ*(t)

Order arrival rates follow an exponential thinning model:
  λ(δ) = A · exp(−k · δ)

where δ is the depth of the quote from the reservation price.

Parameters:
  γ  > 0  — absolute risk-aversion coefficient.
  σ  > 0  — midprice volatility (same time units as T).
  k  > 0  — order-book depth / sensitivity of arrivals to depth.
  A  > 0  — baseline order arrival intensity.
  T  > 0  — trading horizon.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ── Model parameters ──────────────────────────────────────────────────────────

@dataclass
class ASParams:
    """
    Avellaneda-Stoikov model parameters.

    Attributes
    ----------
    gamma : risk-aversion coefficient (> 0).  Higher γ → wider spread.
    sigma : midprice volatility per unit time (> 0).
    k     : order arrival sensitivity to depth (> 0).
    A     : base order arrival intensity (> 0).
    T     : trading horizon in the same time units as σ.
    """
    gamma: float
    sigma: float
    k:     float
    A:     float
    T:     float

    def __post_init__(self) -> None:
        for name, val in [("gamma", self.gamma), ("sigma", self.sigma),
                          ("k", self.k), ("A", self.A), ("T", self.T)]:
            if val <= 0:
                raise ValueError(f"ASParams.{name} must be positive; got {val}")


# ── Core analytical formulas ──────────────────────────────────────────────────

def reservation_price(
    mid:            float,
    inventory:      float,
    gamma:          float,
    sigma:          float,
    time_remaining: float,
) -> float:
    """
    Risk-adjusted reservation price.

        r(s, q, t) = s − q · γ · σ² · (T−t)

    The market maker discounts (increases) the midprice when long (short)
    to account for the inventory risk of holding a position to horizon T.

    Parameters
    ----------
    mid            : current midprice s.
    inventory      : signed inventory q (positive = long).
    gamma          : risk-aversion coefficient.
    sigma          : midprice volatility.
    time_remaining : T − t, time left until horizon.

    Returns
    -------
    r : reservation price.
    """
    return mid - inventory * gamma * sigma ** 2 * time_remaining


def optimal_half_spread(
    gamma:          float,
    sigma:          float,
    time_remaining: float,
    k:              float,
) -> float:
    """
    Optimal symmetric half-spread δ*(t).

        δ*(t) = (γ · σ² · (T−t)) / 2  +  (1/γ) · ln(1 + γ/k)

    The first term grows with remaining inventory risk.
    The second is a market-power term that is time-invariant.

    Total spread = 2 · δ*(t); bid = r − δ*, ask = r + δ*.

    Parameters
    ----------
    gamma, sigma    : AS parameters.
    time_remaining  : T − t.
    k               : arrival sensitivity parameter.

    Returns
    -------
    delta_star : optimal half-spread (≥ 0).
    """
    if time_remaining <= 0:
        # At horizon: only the market-power term remains
        return (1.0 / gamma) * np.log(1.0 + gamma / k)
    risk_term        = 0.5 * gamma * sigma ** 2 * time_remaining
    market_power     = (1.0 / gamma) * np.log(1.0 + gamma / k)
    return risk_term + market_power


def optimal_quotes(
    mid:            float,
    inventory:      float,
    params:         ASParams,
    time_remaining: float,
) -> tuple:
    """
    Compute optimal bid and ask quotes.

    Parameters
    ----------
    mid            : current midprice.
    inventory      : signed inventory q.
    params         : ASParams.
    time_remaining : T − t.

    Returns
    -------
    (bid, ask, reservation_price, half_spread)
    """
    r     = reservation_price(mid, inventory,
                              params.gamma, params.sigma, time_remaining)
    delta = optimal_half_spread(params.gamma, params.sigma,
                                time_remaining, params.k)
    return r - delta, r + delta, r, delta


def arrival_rate(depth: float, A: float, k: float) -> float:
    """
    Order arrival rate at depth δ from the reservation price.

        λ(δ) = A · exp(−k · δ)

    Depth δ = 0 → maximum rate A; larger δ → fewer orders.
    """
    return A * np.exp(-k * max(depth, 0.0))


# ── Simulation state and results ──────────────────────────────────────────────

@dataclass
class SimState:
    """Market-maker state snapshot at one timestep."""
    step:        int
    time:        float
    mid:         float
    bid:         float
    ask:         float
    reservation: float
    half_spread: float
    inventory:   int
    cash:        float
    pnl:         float    # mark-to-market: cash + inventory × mid


@dataclass
class SimResult:
    """Full trajectory returned by AvellanedaStoikov.simulate()."""
    history:       List[SimState] = field(default_factory=list)
    final_pnl:     float          = 0.0
    max_inventory: int            = 0
    fill_count:    int            = 0
    sharpe:        float          = 0.0   # annualised step-level Sharpe

    @property
    def pnl_series(self) -> np.ndarray:
        return np.array([s.pnl for s in self.history])

    @property
    def inventory_series(self) -> np.ndarray:
        return np.array([s.inventory for s in self.history], dtype=int)

    @property
    def spread_series(self) -> np.ndarray:
        return np.array([2.0 * s.half_spread for s in self.history])


# ── Main class ────────────────────────────────────────────────────────────────

class AvellanedaStoikov:
    """
    Discrete-time simulation of the AS optimal market-making strategy.

    Each timestep dt the market maker:
      1. Computes reservation price r and optimal half-spread δ*.
      2. Posts bid = r − δ*, ask = r + δ*.
      3. Each side receives a fill with probability ≈ λ(δ*) · dt
         (Poisson thinning; capped at 1).
      4. Inventory and cash are updated; midprice diffuses by σ√dt.

    Usage
    -----
    >>> p = ASParams(gamma=0.1, sigma=2.0, k=1.5, A=140.0, T=1.0)
    >>> model = AvellanedaStoikov(p)
    >>> result = model.simulate(n_steps=500, dt=1/252)
    >>> print(f"Final PnL: {result.final_pnl:.2f}")
    """

    def __init__(self, params: ASParams) -> None:
        self.p = params

    def simulate(
        self,
        n_steps:       int   = 1_000,
        dt:            float = 1.0 / 252,
        s0:            float = 100.0,
        q0:            int   = 0,
        cash0:         float = 0.0,
        seed:          int   = 42,
        max_inventory: int   = 10,
    ) -> SimResult:
        """
        Run the AS market-making strategy for n_steps timesteps.

        Parameters
        ----------
        n_steps       : number of simulation steps.
        dt            : step size in the same time units as params.T and params.sigma.
        s0            : initial midprice.
        q0            : initial inventory.
        cash0         : initial cash.
        seed          : RNG seed for reproducibility.
        max_inventory : hard inventory cap (in absolute value).
                        Orders that would breach this limit are not posted.

        Returns
        -------
        SimResult containing full state history and summary statistics.
        """
        rng    = np.random.default_rng(seed)
        p      = self.p
        s      = float(s0)
        q      = int(q0)
        cash   = float(cash0)
        result = SimResult()

        for step in range(n_steps):
            t_remaining = max(p.T - step * dt, 1e-10)
            bid, ask, r, delta = optimal_quotes(s, q, p, t_remaining)

            # Record state before fill
            pnl = cash + q * s
            result.history.append(SimState(
                step        = step,
                time        = step * dt,
                mid         = s,
                bid         = bid,
                ask         = ask,
                reservation = r,
                half_spread = delta,
                inventory   = q,
                cash        = cash,
                pnl         = pnl,
            ))

            # Poisson thinning: probability of a fill on each side this step
            lam      = arrival_rate(delta, p.A, p.k)
            p_fill   = min(lam * dt, 1.0)

            bid_fill = rng.random() < p_fill and (q + 1) <=  max_inventory
            ask_fill = rng.random() < p_fill and (q - 1) >= -max_inventory

            if bid_fill:
                q    += 1
                cash -= bid
                result.fill_count += 1
            if ask_fill:
                q    -= 1
                cash += ask
                result.fill_count += 1

            # Midprice Brownian motion
            s += p.sigma * np.sqrt(dt) * rng.standard_normal()

        # Final mark-to-market
        result.final_pnl     = cash + q * s
        result.max_inventory = int(max(abs(st.inventory) for st in result.history))

        # Step-level Sharpe ratio
        dpnl = np.diff(result.pnl_series)
        if len(dpnl) > 0 and dpnl.std() > 0:
            result.sharpe = float(dpnl.mean() / dpnl.std() * np.sqrt(1.0 / dt))

        return result

    def quote_grid(
        self,
        inventories:    np.ndarray,
        time_remaining: float,
        mid:            float = 100.0,
    ) -> np.ndarray:
        """
        Return optimal quotes for a range of inventory levels.

        Useful for visualising how quotes shift with inventory.

        Parameters
        ----------
        inventories    : array of inventory values to evaluate.
        time_remaining : T − t.
        mid            : midprice (default 100).

        Returns
        -------
        quotes : shape (len(inventories), 4) — [bid, ask, reservation, half_spread].
        """
        rows = []
        for inv in inventories:
            b, a, r, d = optimal_quotes(float(mid), float(inv), self.p, time_remaining)
            rows.append([b, a, r, d])
        return np.array(rows, dtype=float)
