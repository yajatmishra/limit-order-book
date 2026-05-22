"""
Market Impact Models
====================
Estimates the price impact of executing an order of size Q in a stock
with given daily volatility σ and average daily volume (ADV).

Models implemented
------------------
1. Almgren-Chriss (2001) — linear permanent + temporary impact
   Permanent:  g(v) = γ · σ · v / ADV          (linear in rate v)
   Temporary:  h(v) = η · σ · sign(v) · |v/ADV|  (linear in rate)

2. Square-root (Almgren et al. 2005)
   Temporary:  h(v) = η · σ · sign(v) · √(|v|/ADV)
   This is the industry standard for large-cap equities.

3. Three-fifths (Grinold & Kahn)
   Temporary:  h(v) = η · σ · sign(v) · (|v|/ADV)^(3/5)

4. Optimal Almgren-Chriss trajectory
   Computes the closed-form optimal execution schedule that minimises
   E[cost] + λ · Var[cost]:

     x(t) = x_0 · sinh(κ(T−t)) / sinh(κT)
     κ² = λ·σ² / η̃     (η̃ = η / (2·ADV))

   Returns the time-discretised trade list.

References
----------
  Almgren & Chriss (2001). "Optimal execution of portfolio transactions."
  Journal of Risk 3(2), 5-39.

  Almgren et al. (2005). "Direct estimation of equity market impact."
  Risk Magazine 18(7), 58-62.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Impact estimates ──────────────────────────────────────────────────────────

def square_root_impact(
    quantity:   float,
    adv:        float,
    daily_vol:  float,
    eta:        float = 0.1,
) -> float:
    """
    Almgren et al. (2005) square-root temporary market impact.

    impact = η · σ · sign(Q) · √(|Q| / ADV)

    Returns impact in price units (same units as daily_vol·price, i.e.
    if daily_vol is the daily *fractional* vol, result is a fractional impact).
    """
    if adv <= 0:
        raise ValueError("adv must be > 0")
    sign = 1.0 if quantity >= 0 else -1.0
    return float(sign * eta * daily_vol * np.sqrt(abs(quantity) / adv))


def linear_impact(
    quantity:  float,
    adv:       float,
    daily_vol: float,
    eta:       float = 0.1,
) -> float:
    """Linear temporary impact: η · σ · Q / ADV."""
    if adv <= 0:
        raise ValueError("adv must be > 0")
    return float(eta * daily_vol * quantity / adv)


def three_fifths_impact(
    quantity:  float,
    adv:       float,
    daily_vol: float,
    eta:       float = 0.1,
) -> float:
    """Grinold-Kahn 3/5-power impact: η · σ · sign(Q) · (|Q|/ADV)^0.6."""
    if adv <= 0:
        raise ValueError("adv must be > 0")
    sign = 1.0 if quantity >= 0 else -1.0
    return float(sign * eta * daily_vol * (abs(quantity) / adv) ** 0.6)


def impact_bps(
    quantity:   float,
    adv:        float,
    daily_vol:  float,
    eta:        float = 0.1,
    model:      str   = "sqrt",
) -> float:
    """
    Fractional market impact in basis points.

    Parameters
    ----------
    model : "sqrt" | "linear" | "three_fifths"
    """
    if model == "sqrt":
        frac = square_root_impact(quantity, adv, daily_vol, eta)
    elif model == "linear":
        frac = linear_impact(quantity, adv, daily_vol, eta)
    elif model == "three_fifths":
        frac = three_fifths_impact(quantity, adv, daily_vol, eta)
    else:
        raise ValueError(f"Unknown model: {model!r}")
    return float(abs(frac) * 1e4)


# ── Almgren-Chriss optimal trajectory ─────────────────────────────────────────

@dataclass
class ACParams:
    """Almgren-Chriss model parameters."""
    sigma:     float   # daily return volatility (fractional)
    eta:       float   # temporary impact coefficient (default 0.1)
    gamma:     float   # permanent impact coefficient (default 0.05)
    adv:       float   # average daily volume (shares)
    lam:       float   # risk aversion coefficient λ (default 1e-6)


@dataclass
class ACTrajectory:
    """Optimal Almgren-Chriss execution trajectory."""
    times:       np.ndarray   # time grid  t_0=0, …, t_N=T (length N+1)
    holdings:    np.ndarray   # remaining shares x(t_k) (length N+1)
    trades:      np.ndarray   # shares traded at each step (length N), signed
    exp_cost:    float        # expected total cost (fractional)
    exp_var:     float        # variance of cost (fractional²)
    kappa:       float        # decay rate κ

    @property
    def exp_cost_bps(self) -> float:
        return self.exp_cost * 1e4

    def __repr__(self) -> str:
        return (f"ACTrajectory(N={len(self.trades)}, "
                f"E[cost]={self.exp_cost_bps:.2f}bps, "
                f"κ={self.kappa:.4f})")


class AlmgrenChriss:
    """
    Almgren-Chriss (2001) optimal liquidation model.

    Finds the execution trajectory x(t) that minimises:
        E[cost] + λ · Var[cost]

    under the linear market impact model.

    Parameters
    ----------
    sigma  : daily return volatility (e.g. 0.02 = 2% per day).
    eta    : temporary impact coefficient (default 0.1).
    gamma  : permanent impact coefficient (default 0.05).
    adv    : average daily volume.
    lam    : risk-aversion parameter λ.  Higher λ → faster execution
             (trades sooner to reduce variance at the cost of higher impact).

    Usage
    -----
    >>> ac = AlmgrenChriss(sigma=0.02, eta=0.1, gamma=0.05, adv=1e6, lam=1e-6)
    >>> traj = ac.optimal_trajectory(quantity=50_000, T=1.0, N=10)
    """

    def __init__(
        self,
        sigma:  float = 0.02,
        eta:    float = 0.1,
        gamma:  float = 0.05,
        adv:    float = 1e6,
        lam:    float = 1e-6,
    ) -> None:
        self.sigma = sigma
        self.eta   = eta
        self.gamma = gamma
        self.adv   = adv
        self.lam   = lam

    def optimal_trajectory(
        self,
        quantity: float,
        T:        float,
        N:        int,
    ) -> ACTrajectory:
        """
        Compute the discrete-time optimal liquidation trajectory.

        Parameters
        ----------
        quantity : total shares to liquidate (positive = sell).
        T        : total time horizon (in days).
        N        : number of time periods.

        Returns
        -------
        ACTrajectory with holdings at each time step and trade list.
        """
        if N < 1:
            raise ValueError("N must be >= 1")
        if T <= 0:
            raise ValueError("T must be > 0")

        x0  = float(quantity)
        tau = T / N   # period length in days

        # Effective impact coefficients per unit of ADV
        eta_eff   = self.eta   / self.adv
        gamma_eff = self.gamma / self.adv

        # κ² = λσ² / η̃,  η̃ = η_eff - γ_eff·τ/2  (discrete version)
        eta_tilde = eta_eff - 0.5 * gamma_eff * tau
        if eta_tilde <= 0:
            eta_tilde = 1e-12

        kappa_sq  = max(self.lam * self.sigma ** 2 / eta_tilde, 0.0)
        kappa     = float(np.sqrt(kappa_sq))

        # Discrete holdings: x_j = x_0 · sinh(κ(T − t_j)) / sinh(κT)
        times    = np.linspace(0.0, T, N + 1)
        kT       = kappa * T

        if kT < 1e-10:
            # κ → 0: uniform liquidation (TWAP limit)
            holdings = x0 * (1.0 - times / T)
        else:
            holdings = x0 * np.sinh(kappa * (T - times)) / np.sinh(kT)

        holdings = np.maximum(holdings, 0.0)   # numerical floor
        trades   = -np.diff(holdings)          # negative holdings change = sell

        # Expected cost  E[C] = γ/2 · σ²_perm + Σ η/ADV · n_k²/τ
        # Variance  Var[C] = σ² · Σ τ · x_k²
        exp_cost = (0.5 * gamma_eff * x0 ** 2 * self.sigma ** 2
                    + eta_eff / tau * np.sum(trades ** 2))
        exp_var  = self.sigma ** 2 * tau * np.sum(holdings[:-1] ** 2)

        return ACTrajectory(
            times    = times,
            holdings = holdings,
            trades   = trades,
            exp_cost = float(exp_cost),
            exp_var  = float(exp_var),
            kappa    = kappa,
        )

    def efficient_frontier(
        self,
        quantity: float,
        T:        float,
        N:        int,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Almgren-Chriss efficient frontier.

        Returns
        -------
        (expected_costs, variances) as arrays of shape (n_points,).
        """
        lambdas = np.logspace(-8, -2, n_points)
        costs   = np.empty(n_points)
        varis   = np.empty(n_points)
        orig_lam = self.lam

        for i, lam in enumerate(lambdas):
            self.lam = float(lam)
            traj     = self.optimal_trajectory(quantity, T, N)
            costs[i] = traj.exp_cost
            varis[i] = traj.exp_var

        self.lam = orig_lam
        return costs, varis
