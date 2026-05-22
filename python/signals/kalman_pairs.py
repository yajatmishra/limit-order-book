"""
Kalman Filter Pairs Trading
============================
Implements a Dynamic Linear Model (DLM) Kalman filter that tracks the
time-varying hedge ratio and intercept of a cointegrated pair online.

Reference:
  Pole, West & Harrison (1994). "Applied Bayesian Forecasting and Time
  Series Analysis." Chapman & Hall.

  Pairs trading application:
  Elliot, Van Der Hoek & Malcolm (2005). "Pairs Trading."
  Quantitative Finance 5(3), 271-276.

State-space model
-----------------
  Measurement:  y1_t = H_t · θ_t + v_t,   v_t ~ N(0, R)
                where H_t = [1,  y2_t]  (row vector)
                      θ_t = [α_t, β_t]  (intercept, hedge ratio)

  Transition:   θ_t = θ_{t-1} + w_t,    w_t ~ N(0, Q)
                Q = δ · I₂  (random-walk drift on parameters)

Kalman recursions
-----------------
  Predict:
    θ_{t|t-1} = θ_{t-1|t-1}
    P_{t|t-1} = P_{t-1|t-1} + Q

  Update:
    e_t      = y1_t − H_t · θ_{t|t-1}       (innovation / spread)
    S_t      = H_t · P_{t|t-1} · H_tᵀ + R   (innovation variance)
    K_t      = P_{t|t-1} · H_tᵀ / S_t       (Kalman gain)
    θ_{t|t}  = θ_{t|t-1} + K_t · e_t
    P_{t|t}  = (I − K_t · H_t) · P_{t|t-1}

The normalised innovation e_t / √S_t is the model's "standardised spread"
and forms the basis of the trading signal.  A rolling window z-score of e_t
provides an alternative with a longer memory for signal smoothing.

Hyperparameters
---------------
  delta : transition variance (process noise).
          Small δ → slow adaptation (stable β).
          Large δ → fast adaptation (jumpy β).
  R     : measurement noise variance.
          Should be set to the empirical variance of the spread.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── KalmanFilter (single-step, online) ───────────────────────────────────────

class KalmanFilter:
    """
    Online 2-state Kalman filter for the DLM pairs model.

    Tracks θ = [α, β] where y1 ≈ α + β·y2.

    Parameters
    ----------
    delta : float
        Process noise variance per step (controls how fast β can drift).
        Typical values: 1e-5 (slow) … 1e-3 (fast).  Default 1e-4.
    R     : float
        Measurement noise variance.  Default 1e-2.
    P0    : float
        Initial state-covariance diagonal.  Default 1.0.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        R:     float = 1e-2,
        P0:    float = 1.0,
    ) -> None:
        if delta <= 0 or R <= 0 or P0 <= 0:
            raise ValueError("delta, R, and P0 must be positive")
        self.delta = delta
        self.R     = R

        # State vector [alpha, beta] — initialised to zero
        self.theta: np.ndarray = np.zeros(2)
        # State covariance — large to express prior uncertainty
        self.P:     np.ndarray = np.eye(2) * P0
        # Process noise matrix (diagonal, constant)
        self.Q:     np.ndarray = np.eye(2) * delta

    def update(self, y1: float, y2: float) -> Tuple[float, float]:
        """
        Process one observation (y1_t, y2_t) and return the innovation.

        Parameters
        ----------
        y1 : observed value of asset 1 (dependent variable).
        y2 : observed value of asset 2 (independent variable / factor).

        Returns
        -------
        (innovation, innovation_variance) : (e_t, S_t)
            innovation   = y1 − H·θ_{t|t-1}
            innov_var    = H·P_{t|t-1}·Hᵀ + R
        """
        H = np.array([1.0, y2])   # measurement row vector

        # Predict
        P_prior = self.P + self.Q        # P_{t|t-1} = P + Q  (θ unchanged)

        # Innovation
        innovation = y1 - H @ self.theta
        S          = float(H @ P_prior @ H) + self.R    # scalar

        # Kalman gain
        K = P_prior @ H / S    # shape (2,)

        # Update
        self.theta = self.theta + K * innovation
        self.P     = (np.eye(2) - np.outer(K, H)) @ P_prior

        return float(innovation), float(S)

    def predict_y1(self, y2: float) -> float:
        """
        Prior prediction of y1 given y2, using current (posterior) state.
        """
        return float(self.theta[0] + self.theta[1] * y2)

    @property
    def alpha(self) -> float:
        """Current intercept estimate."""
        return float(self.theta[0])

    @property
    def beta(self) -> float:
        """Current hedge ratio estimate."""
        return float(self.theta[1])

    def reset(self, P0: float = 1.0) -> None:
        """Reset state to zero with covariance P0 * I."""
        self.theta = np.zeros(2)
        self.P     = np.eye(2) * P0


# ── KalmanPairsTrader (batch / history tracking) ──────────────────────────────

@dataclass
class KalmanPairsHistory:
    """Full trajectory produced by KalmanPairsTrader.fit()."""
    y1:             np.ndarray   # original series 1
    y2:             np.ndarray   # original series 2
    alphas:         np.ndarray   # intercept estimates θ[0]
    betas:          np.ndarray   # hedge ratio estimates θ[1]
    innovations:    np.ndarray   # raw Kalman innovations e_t
    innov_vars:     np.ndarray   # innovation variances S_t
    norm_innov:     np.ndarray   # standardised innovation e_t / √S_t
    zscore:         np.ndarray   # rolling z-score of innovations
    spread:         np.ndarray   # y1 − β·y2 − α  (posterior spread)

    @property
    def N(self) -> int:
        return len(self.y1)

    def signal(
        self,
        entry_z:  float = 2.0,
        exit_z:   float = 0.5,
        use_norm: bool  = True,
    ) -> np.ndarray:
        """
        Integer trading signal in {−1, 0, +1}.

        Uses the normalised innovation (e/√S) when use_norm=True,
        otherwise the rolling z-score.

        +1 → buy the spread  (spread is cheap: z < −entry_z)
        −1 → sell the spread (spread is expensive: z > +entry_z)
         0 → flat / exit zone

        State-machine: enter on |z| > entry_z, exit on |z| < exit_z.
        """
        z   = self.norm_innov if use_norm else self.zscore
        n   = len(z)
        sig = np.zeros(n, dtype=int)
        pos = 0
        for i in range(n):
            if not np.isfinite(z[i]):
                sig[i] = pos
                continue
            if pos == 0:
                if z[i] < -entry_z:
                    pos = 1
                elif z[i] > entry_z:
                    pos = -1
            else:
                if abs(z[i]) < exit_z:
                    pos = 0
            sig[i] = pos
        return sig


class KalmanPairsTrader:
    """
    Kalman-filter–based pairs trading engine.

    Supports:
    - Batch fitting: process a full historical series and return history.
    - Online updating: stream new observations one at a time.
    - Rolling z-score: longer-memory signal for smoother entry/exit.

    Usage — batch
    -------------
    >>> trader = KalmanPairsTrader(delta=1e-4, R=1e-2)
    >>> hist   = trader.fit(y1, y2, zscore_window=30)
    >>> signal = hist.signal(entry_z=2.0, exit_z=0.5)

    Usage — online
    --------------
    >>> trader.reset()
    >>> for y1_t, y2_t in stream:
    ...     innov, S, z = trader.step(y1_t, y2_t)
    ...     trade_signal = 1 if z < -2 else -1 if z > 2 else 0
    """

    def __init__(
        self,
        delta:         float = 1e-4,
        R:             float = 1e-2,
        P0:            float = 1.0,
        zscore_window: int   = 30,
    ) -> None:
        self.delta         = delta
        self.R_param       = R
        self.P0            = P0
        self.zscore_window = zscore_window

        self._kf = KalmanFilter(delta=delta, R=R, P0=P0)
        # Online history for rolling z-score
        self._innov_buf: List[float] = []

    # ── Batch fitting ─────────────────────────────────────────────────────────

    def fit(
        self,
        y1:            np.ndarray,
        y2:            np.ndarray,
        zscore_window: Optional[int] = None,
    ) -> KalmanPairsHistory:
        """
        Fit the Kalman filter to the full (y1, y2) series.

        Parameters
        ----------
        y1, y2         : price series, shape (N,).
        zscore_window  : rolling window for z-score; overrides constructor default.

        Returns
        -------
        KalmanPairsHistory with full trajectory arrays.
        """
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        if len(y1) != len(y2):
            raise ValueError("y1 and y2 must have equal length")

        w = zscore_window or self.zscore_window
        n = len(y1)

        alphas      = np.zeros(n)
        betas       = np.zeros(n)
        innovations = np.zeros(n)
        innov_vars  = np.zeros(n)

        # Reset KF before batch run
        self._kf.reset(self.P0)

        for t in range(n):
            e, S          = self._kf.update(y1[t], y2[t])
            alphas[t]     = self._kf.alpha
            betas[t]      = self._kf.beta
            innovations[t] = e
            innov_vars[t]  = S

        # Normalised innovation (already a z-score in the Kalman sense)
        norm_innov = innovations / np.sqrt(np.maximum(innov_vars, 1e-30))

        # Rolling window z-score for longer-memory signal
        zscore = self._rolling_zscore(innovations, w)

        # Posterior spread: y1 - β*y2 - α  (posterior, not prior prediction)
        spread = y1 - betas * y2 - alphas

        return KalmanPairsHistory(
            y1          = y1,
            y2          = y2,
            alphas      = alphas,
            betas       = betas,
            innovations = innovations,
            innov_vars  = innov_vars,
            norm_innov  = norm_innov,
            zscore      = zscore,
            spread      = spread,
        )

    # ── Online step ───────────────────────────────────────────────────────────

    def step(
        self,
        y1_t: float,
        y2_t: float,
    ) -> Tuple[float, float, float]:
        """
        Process one new observation; return (innovation, innov_var, z_score).

        The z_score uses the rolling buffer of the last `zscore_window`
        innovations, so it requires at least 2 observations to be non-NaN.
        """
        e, S = self._kf.update(y1_t, y2_t)
        self._innov_buf.append(e)

        w = self.zscore_window
        if len(self._innov_buf) < 2:
            z = np.nan
        else:
            window = np.array(self._innov_buf[-w:])
            mu     = window.mean()
            sig    = window.std(ddof=1)
            z      = (e - mu) / sig if sig > 0 else 0.0

        # Keep buffer bounded
        if len(self._innov_buf) > w * 3:
            self._innov_buf = self._innov_buf[-w:]

        return float(e), float(S), float(z)

    # ── State access ──────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Current intercept estimate."""
        return self._kf.alpha

    @property
    def beta(self) -> float:
        """Current hedge ratio estimate."""
        return self._kf.beta

    def reset(self) -> None:
        """Reset filter state and online innovation buffer."""
        self._kf.reset(self.P0)
        self._innov_buf.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
        n   = len(series)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = series[i - window + 1 : i + 1]
            mu    = chunk.mean()
            sig   = chunk.std(ddof=1)
            out[i] = (series[i] - mu) / sig if sig > 0 else 0.0
        return out


# ── Hyperparameter selection via cross-validation ────────────────────────────

def select_delta(
    y1:      np.ndarray,
    y2:      np.ndarray,
    deltas:  np.ndarray = None,
    R:       float = 1e-2,
) -> Tuple[float, np.ndarray]:
    """
    Select the process-noise hyperparameter δ that maximises the
    log-likelihood of the innovation sequence.

    Under the Kalman model, the innovations are Gaussian:
        e_t ~ N(0, S_t)

    Log-likelihood:
        ℓ = −½ Σ_t [log(2πS_t) + e_t² / S_t]

    Parameters
    ----------
    y1, y2  : training series.
    deltas  : grid of δ values to search (default log-spaced 1e-6 … 1e-2).
    R       : fixed measurement noise.

    Returns
    -------
    (best_delta, log_likelihoods) : best δ and the ℓ curve over the grid.
    """
    if deltas is None:
        deltas = np.logspace(-6, -2, 40)

    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    lls = np.zeros(len(deltas))

    for i, d in enumerate(deltas):
        kf   = KalmanFilter(delta=float(d), R=R)
        ll   = 0.0
        for t in range(len(y1)):
            e, S = kf.update(y1[t], y2[t])
            ll  += -0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)
        lls[i] = ll

    best = float(deltas[np.argmax(lls)])
    return best, lls
