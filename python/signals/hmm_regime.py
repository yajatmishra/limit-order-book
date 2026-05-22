"""
Gaussian Hidden Markov Model — Regime Detection
================================================
Implements a K-state Gaussian HMM fitted via the Baum-Welch EM algorithm
in log-space for numerical stability.  Used to label market regimes
(e.g. low-vol/high-vol, bull/bear/sideways) from a univariate return or
volatility series.

Reference:
  Rabiner (1989). "A tutorial on hidden Markov models."
  Proc. IEEE 77(2), 257-286.

  Cont (2001). "Empirical properties of asset returns."
  Quantitative Finance 1(2), 223-236.

Algorithm summary
-----------------
  E-step  : log-space forward (α) and backward (β) passes → posterior
             state probabilities γ_t(k) and transition counts ξ_t(i,j).
  M-step  : update π, A, {μ_k, σ_k} by weighted MLE.
  Viterbi : log-space dynamic programming for the most likely state path.

Regimes are labelled 0 … K-1 ordered by ascending state mean (so regime 0
is always the "lowest-return / highest-stress" state).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ── Numerical helpers ─────────────────────────────────────────────────────────

def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    if axis is None:
        m = np.max(a)
        return m + np.log(np.sum(np.exp(a - m)) + 1e-300)
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(
        np.sum(np.exp(a - m), axis=axis) + 1e-300
    )


def _log_normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log PDF of N(μ, σ²) at each element of x.  σ is clipped to ≥ 1e-6."""
    sigma = max(float(sigma), 1e-6)
    return -0.5 * np.log(2.0 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2.0 * sigma ** 2)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class HMMResult:
    """Fitted Gaussian HMM parameters and diagnostics."""
    n_states:      int
    means:         np.ndarray   # (K,) — state emission means
    stds:          np.ndarray   # (K,) — state emission std-devs
    transmat:      np.ndarray   # (K, K) — row-stochastic transition matrix
    startprob:     np.ndarray   # (K,) — initial-state distribution
    log_likelihood: float
    n_iter:        int           # EM iterations used
    converged:     bool

    @property
    def regime_order(self) -> np.ndarray:
        """Indices that sort states by ascending mean (regime 0 = bearish)."""
        return np.argsort(self.means)

    def __repr__(self) -> str:
        s = ", ".join(f"μ{k}={m:.3f}(σ={s:.3f})"
                      for k, (m, s) in enumerate(zip(self.means, self.stds)))
        return f"HMMResult(K={self.n_states}, {s}, LL={self.log_likelihood:.1f})"


# ── GaussianHMM ───────────────────────────────────────────────────────────────

class GaussianHMM:
    """
    K-state Gaussian HMM fitted by Baum-Welch EM.

    Parameters
    ----------
    n_states : number of hidden states (default 2).
    n_iter   : maximum EM iterations (default 200).
    tol      : convergence threshold on log-likelihood change (default 1e-6).
    min_std  : minimum emission standard deviation (regularisation, default 1e-4).
    seed     : RNG seed for initialisation.

    Usage
    -----
    >>> hmm = GaussianHMM(n_states=2)
    >>> result = hmm.fit(returns)
    >>> states = hmm.predict(returns)
    >>> proba  = hmm.predict_proba(returns)   # shape (T, K)
    """

    def __init__(
        self,
        n_states: int   = 2,
        n_iter:   int   = 200,
        tol:      float = 1e-6,
        min_std:  float = 1e-4,
        seed:     int   = 42,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.min_std  = min_std
        self.seed     = seed
        self._result: Optional[HMMResult] = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> HMMResult:
        """
        Fit the HMM to observation sequence X via Baum-Welch EM.

        Parameters
        ----------
        X : univariate observation series, shape (T,).

        Returns
        -------
        HMMResult with fitted parameters.
        """
        x   = np.asarray(X, dtype=float).ravel()
        T   = len(x)
        K   = self.n_states
        rng = np.random.default_rng(self.seed)

        # ── Initialisation ────────────────────────────────────────────────────
        # Sort and split observations into K equal segments → seed μ, σ
        sorted_x = np.sort(x)
        split    = np.array_split(sorted_x, K)
        mu    = np.array([s.mean() for s in split])
        sigma = np.array([max(s.std(ddof=1) if len(s) > 1 else 1.0, self.min_std)
                          for s in split])
        # Uniform transition matrix and initial distribution
        A    = np.full((K, K), 1.0 / K)
        pi   = np.full(K, 1.0 / K)

        log_lik_prev = -np.inf
        converged    = False

        for it in range(self.n_iter):
            # ── E-step ────────────────────────────────────────────────────────
            log_B = np.column_stack([_log_normal_pdf(x, mu[k], sigma[k])
                                     for k in range(K)])   # (T, K)
            log_A  = np.log(A + 1e-300)
            log_pi = np.log(pi + 1e-300)

            log_alpha = self._forward(log_B, log_A, log_pi, T, K)
            log_beta  = self._backward(log_B, log_A, T, K)

            # Log-likelihood
            log_lik = _logsumexp(log_alpha[-1])

            # Posterior state probabilities: γ_t(k) = α_t(k)β_t(k) / P(X)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)                        # (T, K)

            # Joint posterior ξ_t(i,j) for t = 0..T-2
            log_xi = self._compute_log_xi(log_alpha, log_beta, log_A, log_B, T, K)

            # ── M-step ────────────────────────────────────────────────────────
            pi = gamma[0] / gamma[0].sum()
            # Transition matrix
            sum_xi    = np.exp(_logsumexp(log_xi, axis=0))   # (K,K): Σ_t ξ_t
            A         = sum_xi / (sum_xi.sum(axis=1, keepdims=True) + 1e-300)

            # Emission parameters (weighted MLE)
            gamma_sum = gamma.sum(axis=0) + 1e-300           # (K,)
            mu        = (gamma.T @ x) / gamma_sum
            for k in range(K):
                diff    = x - mu[k]
                var     = np.dot(gamma[:, k], diff ** 2) / gamma_sum[k]
                sigma[k] = max(np.sqrt(var), self.min_std)

            # ── Convergence check ─────────────────────────────────────────────
            delta = log_lik - log_lik_prev
            if abs(delta) < self.tol and it > 0:
                converged = True
                break
            log_lik_prev = log_lik

        # Sort states by ascending mean so regime 0 = lowest mean
        order = np.argsort(mu)
        mu    = mu[order]
        sigma = sigma[order]
        A     = A[np.ix_(order, order)]
        pi    = pi[order]

        self._result = HMMResult(
            n_states       = K,
            means          = mu,
            stds           = sigma,
            transmat       = A,
            startprob      = pi,
            log_likelihood = float(log_lik),
            n_iter         = it + 1,
            converged      = converged,
        )
        # Store log_B with sorted params for prediction
        self._mu    = mu
        self._sigma = sigma
        self._log_A = np.log(A + 1e-300)
        self._log_pi = np.log(pi + 1e-300)
        return self._result

    # ── Decoding ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi most-likely state sequence.

        Returns
        -------
        states : int array, shape (T,), values in 0 … K-1.
        """
        return self._viterbi(np.asarray(X, dtype=float).ravel())[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior state probabilities γ_t(k) from forward-backward.

        Returns
        -------
        proba : shape (T, K), rows sum to 1.
        """
        x   = np.asarray(X, dtype=float).ravel()
        T, K = len(x), self.n_states
        log_B     = np.column_stack([_log_normal_pdf(x, self._mu[k], self._sigma[k])
                                     for k in range(K)])
        log_alpha = self._forward(log_B, self._log_A, self._log_pi, T, K)
        log_beta  = self._backward(log_B, self._log_A, T, K)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        return np.exp(log_gamma)

    def decode(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi decoding, also returns log-likelihood of the path.

        Returns (states, log_likelihood).
        """
        return self._viterbi(np.asarray(X, dtype=float).ravel())

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _forward(
        log_B:  np.ndarray,   # (T, K)
        log_A:  np.ndarray,   # (K, K)
        log_pi: np.ndarray,   # (K,)
        T:      int,
        K:      int,
    ) -> np.ndarray:
        """Log-space forward pass.  Returns log_alpha (T, K)."""
        log_alpha    = np.empty((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            # log_alpha[t, k] = logsumexp_j(log_alpha[t-1,j] + log_A[j,k]) + log_B[t,k]
            mat = log_alpha[t - 1, :, None] + log_A   # (K_from, K_to)
            log_alpha[t] = _logsumexp(mat, axis=0) + log_B[t]
        return log_alpha

    @staticmethod
    def _backward(
        log_B: np.ndarray,   # (T, K)
        log_A: np.ndarray,   # (K, K)
        T:     int,
        K:     int,
    ) -> np.ndarray:
        """Log-space backward pass.  Returns log_beta (T, K)."""
        log_beta       = np.zeros((T, K))   # log(1) = 0 at T-1
        for t in range(T - 2, -1, -1):
            # log_beta[t, k] = logsumexp_j(log_A[k,j] + log_B[t+1,j] + log_beta[t+1,j])
            mat = log_A + log_B[t + 1] + log_beta[t + 1]   # (K, K): [k,j]
            log_beta[t] = _logsumexp(mat, axis=1)
        return log_beta

    @staticmethod
    def _compute_log_xi(
        log_alpha: np.ndarray,   # (T, K)
        log_beta:  np.ndarray,   # (T, K)
        log_A:     np.ndarray,   # (K, K)
        log_B:     np.ndarray,   # (T, K)
        T:         int,
        K:         int,
    ) -> np.ndarray:
        """
        Log joint posterior ξ_t(i,j) for t = 0..T-2.
        Returns log_xi of shape (T-1, K, K).
        """
        log_xi = np.empty((T - 1, K, K))
        for t in range(T - 1):
            # log_xi[t,i,j] = α_t(i) + A[i,j] + B_{t+1}(j) + β_{t+1}(j)
            mat = (log_alpha[t, :, None]
                   + log_A
                   + log_B[t + 1]
                   + log_beta[t + 1])   # (K, K)
            log_xi[t] = mat - _logsumexp(mat)   # normalise
        return log_xi

    def _viterbi(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Log-space Viterbi algorithm."""
        T, K = len(x), self.n_states
        log_B    = np.column_stack([_log_normal_pdf(x, self._mu[k], self._sigma[k])
                                    for k in range(K)])
        log_delta = np.empty((T, K))
        psi       = np.zeros((T, K), dtype=int)

        log_delta[0] = self._log_pi + log_B[0]
        for t in range(1, T):
            trans = log_delta[t - 1, :, None] + self._log_A   # (K, K)
            psi[t]       = np.argmax(trans, axis=0)
            log_delta[t] = np.max(trans, axis=0) + log_B[t]

        # Backtrack
        states       = np.empty(T, dtype=int)
        states[-1]   = np.argmax(log_delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states, float(np.max(log_delta[-1]))


# ── Convenience: regime labelling ─────────────────────────────────────────────

def label_regimes(
    X:        np.ndarray,
    n_states: int   = 2,
    seed:     int   = 42,
) -> Tuple[np.ndarray, HMMResult]:
    """
    Fit a Gaussian HMM and return per-observation regime labels (Viterbi).

    Regime 0 = lowest-mean state (bearish / high-stress).
    Regime K-1 = highest-mean state (bullish / low-stress).

    Returns
    -------
    (labels, HMMResult)
    """
    model  = GaussianHMM(n_states=n_states, seed=seed)
    result = model.fit(X)
    labels = model.predict(X)
    return labels, result
