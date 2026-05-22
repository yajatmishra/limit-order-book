"""
GARCH(1,1)-X — Conditional Volatility with Exogenous Variable
==============================================================
Fits a GARCH(1,1) model augmented with one exogenous predictor X:

    ε_t   = r_t − μ
    σ²_t  = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·X_{t-1}

where stationarity requires  α + β < 1  and ω, σ² > 0.

References:
  Engle (1982). "Autoregressive conditional heteroscedasticity."
  Econometrica 50(4), 987-1007.

  Engle (2002). "Dynamic conditional correlation."
  JBES 20(3), 339-350.

Estimation
----------
Maximise the Gaussian log-likelihood:
    ℓ = −½ Σ_t [log(2π) + log(σ²_t) + ε²_t / σ²_t]

Unconstrained reparameterisation (avoids boundary penalties):
    ω = exp(p[0])                                        (ω > 0)
    α = 0.3 · sigmoid(p[1])                             (0 < α < 0.3)
    β = (1 − α − 1e-4) · sigmoid(p[2])                  (β < 1−α)
    γ = p[3]                                             (unrestricted)
    μ = p[4]                                             (mean)

The variance recursion is initialised at the unconditional variance
    σ²_0 = ω / max(1 − α − β, 1e-8)
and X values outside the sample are set to zero.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def _params_from_raw(p: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Map unconstrained vector p → (ω, α, β, γ, μ)."""
    omega = float(np.exp(p[0]))
    alpha = float(0.3 * _sigmoid(p[1]))
    beta  = float((1.0 - alpha - 1e-4) * _sigmoid(p[2]))
    gamma = float(p[3])
    mu    = float(p[4])
    return omega, alpha, beta, gamma, mu


def _garch_variance(
    resid: np.ndarray,
    X:     np.ndarray,
    omega: float,
    alpha: float,
    beta:  float,
    gamma: float,
) -> np.ndarray:
    """
    Compute the conditional variance series σ²_t for t=0..T-1.

    Parameters
    ----------
    resid : ε_t = r_t − μ, shape (T,)
    X     : exogenous variable aligned so X[t] is the *lagged* value
            that enters σ²_{t+1}. Shape (T,); X[0] is treated as zero.
    """
    T    = len(resid)
    var  = np.empty(T)
    denom = max(1.0 - alpha - beta, 1e-8)
    var[0] = omega / denom                     # unconditional variance

    for t in range(1, T):
        x_lag   = float(X[t - 1])             # X_{t-1}
        eps_lag = float(resid[t - 1])
        var[t]  = (omega
                   + alpha * eps_lag ** 2
                   + beta  * var[t - 1]
                   + gamma * x_lag)
        var[t] = max(var[t], 1e-12)           # floor

    return var


def _neg_log_lik(
    p:     np.ndarray,
    ret:   np.ndarray,
    X:     np.ndarray,
) -> float:
    """Negative Gaussian log-likelihood (to minimise)."""
    omega, alpha, beta, gamma, mu = _params_from_raw(p)
    # Stationarity guard: if α+β ≥ 1, return large value
    if alpha + beta >= 1.0 - 1e-8:
        return 1e10
    resid = ret - mu
    var   = _garch_variance(resid, X, omega, alpha, beta, gamma)
    # Positivity guard
    if np.any(var <= 0):
        return 1e10
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * var) + resid ** 2 / var)
    return -float(ll)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class GARCHXResult:
    """Fitted GARCH(1,1)-X parameters and diagnostics."""
    omega:          float   # constant term ω
    alpha:          float   # ARCH coefficient
    beta:           float   # GARCH coefficient
    gamma:          float   # exogenous variable coefficient
    mu:             float   # conditional mean
    log_likelihood: float
    aic:            float
    bic:            float
    conditional_var: np.ndarray   # σ²_t, shape (T,)
    converged:      bool
    n_obs:          int

    @property
    def conditional_vol(self) -> np.ndarray:
        """Conditional standard deviation series."""
        return np.sqrt(self.conditional_var)

    @property
    def persistence(self) -> float:
        """α + β (should be < 1 for stationarity)."""
        return self.alpha + self.beta

    @property
    def unconditional_var(self) -> float:
        """Long-run (unconditional) variance ω / (1 − α − β)."""
        denom = max(1.0 - self.alpha - self.beta, 1e-8)
        return self.omega / denom

    def __repr__(self) -> str:
        return (f"GARCHXResult(ω={self.omega:.2e}, α={self.alpha:.4f}, "
                f"β={self.beta:.4f}, γ={self.gamma:.4f}, "
                f"persist={self.persistence:.4f}, "
                f"LL={self.log_likelihood:.1f})")


# ── Fitting ────────────────────────────────────────────────────────────────────

class GARCHX:
    """
    GARCH(1,1)-X: GARCH model augmented with one exogenous variable.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·X_{t-1}

    Parameters
    ----------
    n_starts : number of random restarts to avoid local optima (default 5).
    seed     : RNG seed for multi-start initialisation.

    Usage
    -----
    >>> model = GARCHX()
    >>> result = model.fit(returns, exog_variable)
    >>> vol_forecast = result.conditional_vol[-1]  # last in-sample vol
    """

    def __init__(
        self,
        n_starts: int = 5,
        seed:     int = 42,
    ) -> None:
        self.n_starts = n_starts
        self.seed     = seed
        self._result: Optional[GARCHXResult] = None

    def fit(
        self,
        returns: np.ndarray,
        X:       Optional[np.ndarray] = None,
    ) -> GARCHXResult:
        """
        Fit GARCH(1,1)-X by maximum likelihood.

        Parameters
        ----------
        returns : return series, shape (T,).
        X       : exogenous variable, shape (T,).  If None, γ is fixed to 0
                  and the model reduces to plain GARCH(1,1).

        Returns
        -------
        GARCHXResult with fitted parameters and conditional variance series.
        """
        ret = np.asarray(returns, dtype=float).ravel()
        T   = len(ret)

        if X is None:
            exog = np.zeros(T)
        else:
            exog = np.asarray(X, dtype=float).ravel()
            if len(exog) != T:
                raise ValueError("X must have the same length as returns")

        if not _HAS_SCIPY:
            return self._fit_numpy_fallback(ret, exog)

        rng  = np.random.default_rng(self.seed)
        best_ll  = np.inf
        best_res = None

        # Starting-point grid
        starts = [
            np.array([-5.0, 0.0, 0.0, 0.0, 0.0]),    # small ω, moderate α, β
            np.array([-6.0, -1.0, 1.5, 0.0, 0.0]),
            np.array([-4.0, 1.0, 2.0, 0.1, ret.mean()]),
        ]
        # Random starts
        for _ in range(max(0, self.n_starts - len(starts))):
            starts.append(rng.standard_normal(5) * np.array([1, 1, 1, 0.5, 0.01]))

        for p0 in starts[:self.n_starts]:
            try:
                res = _scipy_minimize(
                    _neg_log_lik,
                    x0     = p0,
                    args   = (ret, exog),
                    method = "L-BFGS-B",
                    options = {"maxiter": 500, "ftol": 1e-10},
                )
                if res.fun < best_ll:
                    best_ll  = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is None:
            # All starts failed — use default params
            omega, alpha, beta, gamma, mu = 1e-5, 0.05, 0.90, 0.0, float(ret.mean())
            converged = False
        else:
            omega, alpha, beta, gamma, mu = _params_from_raw(best_res.x)
            converged = bool(best_res.success)

        resid = ret - mu
        var   = _garch_variance(resid, exog, omega, alpha, beta, gamma)
        ll    = -float(_neg_log_lik(best_res.x if best_res else
                                    np.array([np.log(omega), 0., 0., gamma, mu]),
                                    ret, exog))
        n_params = 5
        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + np.log(T) * n_params

        self._result = GARCHXResult(
            omega           = omega,
            alpha           = alpha,
            beta            = beta,
            gamma           = gamma,
            mu              = mu,
            log_likelihood  = ll,
            aic             = aic,
            bic             = bic,
            conditional_var = var,
            converged       = converged,
            n_obs           = T,
        )
        return self._result

    def forecast(
        self,
        h:              int = 1,
        last_resid:     Optional[float] = None,
        last_var:       Optional[float] = None,
        last_X:         float = 0.0,
    ) -> np.ndarray:
        """
        Multi-step variance forecast using the fitted parameters.

        h-step forecast: for k > 1 the exogenous term is set to zero
        (no forecast of X available), so the recursion reduces to plain
        GARCH projection.

        Returns array of shape (h,) — variance forecasts σ²_{T+1}, …, σ²_{T+h}.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before forecast()")
        r = self._result
        e2   = (last_resid or 0.0) ** 2
        prev = last_var or r.unconditional_var
        fwd  = np.empty(h)
        for k in range(h):
            x = last_X if k == 0 else 0.0
            v = r.omega + r.alpha * e2 + r.beta * prev + r.gamma * x
            v = max(v, 1e-12)
            fwd[k] = v
            # For k>0: e²_{T+k} = E[ε²] = σ²_{T+k}
            e2   = v
            prev = v
        return fwd

    # ── Fallback (no scipy) ───────────────────────────────────────────────────

    def _fit_numpy_fallback(
        self,
        ret:  np.ndarray,
        exog: np.ndarray,
    ) -> GARCHXResult:
        """Grid-search fallback when scipy is unavailable."""
        T   = len(ret)
        mu  = float(ret.mean())
        resid = ret - mu
        best_ll = -np.inf
        best_p  = (1e-5, 0.05, 0.90, 0.0)

        for alpha in [0.05, 0.10, 0.15]:
            for beta in [0.80, 0.85, 0.90]:
                if alpha + beta >= 1.0:
                    continue
                omega = float(resid.var() * (1.0 - alpha - beta))
                omega = max(omega, 1e-8)
                var   = _garch_variance(resid, exog, omega, alpha, beta, 0.0)
                if np.any(var <= 0):
                    continue
                ll = float(-0.5 * np.sum(np.log(2*np.pi*var) + resid**2/var))
                if ll > best_ll:
                    best_ll = ll
                    best_p  = (omega, alpha, beta, 0.0)

        omega, alpha, beta, gamma = best_p
        var = _garch_variance(resid, exog, omega, alpha, beta, gamma)
        n_params = 5
        aic = -2*best_ll + 2*n_params
        bic = -2*best_ll + np.log(T)*n_params

        self._result = GARCHXResult(
            omega=omega, alpha=alpha, beta=beta, gamma=gamma, mu=mu,
            log_likelihood=best_ll, aic=aic, bic=bic,
            conditional_var=var, converged=False, n_obs=T,
        )
        return self._result


# ── Convenience function ───────────────────────────────────────────────────────

def fit_garch(
    returns:  np.ndarray,
    X:        Optional[np.ndarray] = None,
    n_starts: int = 5,
) -> GARCHXResult:
    """
    Fit GARCH(1,1)-X and return the result in one call.

    If X is None, fits a standard GARCH(1,1) (γ forced to 0 via exog=0).
    """
    model = GARCHX(n_starts=n_starts)
    return model.fit(returns, X)
