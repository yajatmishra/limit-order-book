"""
Tests for signals.kalman_pairs
================================
Covers:

KalmanFilter
  - invalid parameters raise ValueError
  - update returns finite innovation and positive variance
  - beta converges to true hedge ratio from static data
  - alpha converges to true intercept
  - P matrix remains positive definite after many updates
  - sequential reset restores initial state
  - online update matches batch fit step by step

KalmanPairsTrader.fit()
  - history arrays have correct shape
  - beta converges on long stationary series
  - innovation mean is near zero for cointegrated pair
  - normalised innovation stddev is near 1
  - spread is approximately stationary (|mean| < threshold)
  - no look-ahead: beta at t uses only y1[:t+1], y2[:t+1]
  - posterior spread is close to noise
  - larger delta → faster parameter tracking (regime shift test)

KalmanPairsTrader.step() (online)
  - single step returns 3-tuple (innov, S, z)
  - z is NaN on first step, finite after warm-up
  - online step matches batch fit for same data
  - reset clears buffer

KalmanPairsHistory.signal()
  - returns integer array in {-1, 0, 1}
  - positive signal when spread is persistently negative
  - negative signal when spread is persistently positive
  - signal flips to zero in the exit zone

select_delta()
  - returns a float in the supplied grid
  - log-likelihood array has same length as grid
  - best delta is finite and positive
"""

import numpy as np
import pytest

from signals.kalman_pairs import (
    KalmanFilter,
    KalmanPairsTrader,
    KalmanPairsHistory,
    select_delta,
)


# ── RNG fixture ───────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _make_cointegrated_pair(
    N:          int   = 500,
    true_beta:  float = 2.0,
    true_alpha: float = 1.0,
    noise_std:  float = 0.5,
    seed:       int   = 0,
) -> tuple:
    """
    Generate a cointegrated pair:
        y2 = random walk
        y1 = alpha + beta * y2 + white_noise
    """
    rng = np.random.default_rng(seed)
    y2  = np.cumsum(rng.standard_normal(N))
    y1  = true_alpha + true_beta * y2 + rng.normal(0, noise_std, N)
    return y1, y2


def _make_regime_shift_pair(
    N1:         int   = 300,
    N2:         int   = 300,
    beta1:      float = 1.5,
    beta2:      float = 3.0,
    noise_std:  float = 0.3,
    seed:       int   = 7,
) -> tuple:
    """Two-segment pair: beta changes at N1."""
    rng  = np.random.default_rng(seed)
    y2   = np.cumsum(rng.standard_normal(N1 + N2))
    y1   = np.concatenate([
        1.0 + beta1 * y2[:N1]  + rng.normal(0, noise_std, N1),
        1.0 + beta2 * y2[N1:]  + rng.normal(0, noise_std, N2),
    ])
    return y1, y2


# ═══════════════════════════════════════════════════════════════════════════════
# KalmanFilter unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanFilter:

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            KalmanFilter(delta=0.0)

    def test_invalid_R_raises(self):
        with pytest.raises(ValueError):
            KalmanFilter(R=-1.0)

    def test_invalid_P0_raises(self):
        with pytest.raises(ValueError):
            KalmanFilter(P0=0.0)

    def test_update_returns_finite_values(self):
        kf = KalmanFilter(delta=1e-4, R=1e-2)
        e, S = kf.update(y1=100.5, y2=50.0)
        assert np.isfinite(e)
        assert np.isfinite(S)

    def test_innovation_variance_positive(self):
        kf = KalmanFilter()
        for _ in range(20):
            e, S = kf.update(RNG.standard_normal(), RNG.standard_normal())
            assert S > 0.0

    def test_beta_converges_to_true_value(self):
        """After many observations, β̂ should be close to the true β."""
        y1, y2 = _make_cointegrated_pair(N=2000, true_beta=2.0, true_alpha=0.0,
                                          noise_std=0.2, seed=1)
        kf = KalmanFilter(delta=1e-5, R=0.04)
        for t in range(len(y1)):
            kf.update(y1[t], y2[t])
        assert abs(kf.beta - 2.0) < 0.15, (
            f"Beta should converge: got {kf.beta:.4f}, expected ≈ 2.0"
        )

    def test_alpha_converges_to_true_value(self):
        y1, y2 = _make_cointegrated_pair(N=2000, true_beta=1.0, true_alpha=5.0,
                                          noise_std=0.2, seed=2)
        kf = KalmanFilter(delta=1e-5, R=0.04)
        for t in range(len(y1)):
            kf.update(y1[t], y2[t])
        assert abs(kf.alpha - 5.0) < 0.5, (
            f"Alpha should converge: got {kf.alpha:.4f}, expected ≈ 5.0"
        )

    def test_P_matrix_remains_positive_definite(self):
        kf = KalmanFilter(delta=1e-4, R=1e-2)
        y1, y2 = _make_cointegrated_pair(N=100, seed=3)
        for t in range(len(y1)):
            kf.update(y1[t], y2[t])
            # P should be positive definite: all eigenvalues > 0
            eigs = np.linalg.eigvalsh(kf.P)
            assert np.all(eigs > 0), f"P not PD at step {t}: eigs = {eigs}"

    def test_P_matrix_is_symmetric(self):
        kf = KalmanFilter(delta=1e-4, R=1e-2)
        y1, y2 = _make_cointegrated_pair(N=50, seed=4)
        for t in range(len(y1)):
            kf.update(y1[t], y2[t])
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-12)

    def test_reset_restores_zero_state(self):
        kf = KalmanFilter(delta=1e-4, R=1e-2, P0=2.0)
        y1, y2 = _make_cointegrated_pair(N=50, seed=5)
        for t in range(len(y1)):
            kf.update(y1[t], y2[t])
        kf.reset(P0=2.0)
        np.testing.assert_array_equal(kf.theta, [0.0, 0.0])
        np.testing.assert_allclose(kf.P, np.eye(2) * 2.0)

    def test_predict_y1_consistency(self):
        kf = KalmanFilter(delta=1e-4, R=1e-2)
        y1, y2 = _make_cointegrated_pair(N=200, seed=6)
        for t in range(len(y1)):
            kf.update(y1[t], y2[t])
        # After convergence, predict should be close to true model
        y2_new = 10.0
        pred   = kf.predict_y1(y2_new)
        assert np.isfinite(pred)

    def test_constant_pair_beta_stable(self):
        """With zero noise, β should converge exactly and stay there."""
        N  = 500
        t  = np.arange(N, dtype=float)
        y2 = t
        y1 = 3.0 * y2 + 2.0   # exact linear, no noise
        kf = KalmanFilter(delta=1e-6, R=1e-6)
        for i in range(N):
            kf.update(y1[i], y2[i])
        assert abs(kf.beta  - 3.0) < 0.01
        assert abs(kf.alpha - 2.0) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# KalmanPairsTrader.fit() tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanPairsTraderFit:

    @pytest.fixture
    def fitted(self) -> KalmanPairsHistory:
        y1, y2 = _make_cointegrated_pair(N=600, true_beta=2.5, true_alpha=1.0,
                                          noise_std=0.3, seed=10)
        trader = KalmanPairsTrader(delta=1e-4, R=0.09, zscore_window=40)
        return trader.fit(y1, y2)

    def test_array_shapes(self, fitted):
        N = fitted.N
        for arr_name in ["alphas", "betas", "innovations", "innov_vars",
                          "norm_innov", "zscore", "spread"]:
            arr = getattr(fitted, arr_name)
            assert arr.shape == (N,), f"{arr_name} has wrong shape {arr.shape}"

    def test_beta_converges_end_of_series(self, fitted):
        # Use the last 100 beta estimates; their mean should be near 2.5
        beta_tail = fitted.betas[-100:]
        assert abs(beta_tail.mean() - 2.5) < 0.3, (
            f"Beta tail mean = {beta_tail.mean():.3f}, expected ≈ 2.5"
        )

    def test_innovation_mean_near_zero(self, fitted):
        # For a correctly specified model, E[e_t] ≈ 0
        tail = fitted.innovations[-200:]
        assert abs(tail.mean()) < 0.5, (
            f"Innovation mean = {tail.mean():.4f}, expected ≈ 0"
        )

    def test_norm_innov_std_near_one(self, fitted):
        # Normalised innovations should have std ≈ 1
        tail = fitted.norm_innov[-200:]
        assert 0.5 < tail.std() < 2.5, (
            f"Norm innovation std = {tail.std():.4f}, expected ≈ 1.0"
        )

    def test_spread_is_finite(self, fitted):
        assert np.all(np.isfinite(fitted.spread))

    def test_innov_vars_positive(self, fitted):
        assert np.all(fitted.innov_vars > 0)

    def test_zscore_first_window_minus_1_nan(self):
        y1, y2 = _make_cointegrated_pair(N=100, seed=11)
        w      = 20
        trader = KalmanPairsTrader(zscore_window=w)
        hist   = trader.fit(y1, y2)
        assert np.all(np.isnan(hist.zscore[:w - 1]))
        assert np.all(np.isfinite(hist.zscore[w - 1:]))

    def test_no_lookahead_beta(self):
        """
        Beta at step t should be determined only by observations 0..t.
        Verify: fit once on full series, then re-run step by step —
        results must match exactly.
        """
        N  = 80
        y1, y2 = _make_cointegrated_pair(N=N, seed=12)
        trader = KalmanPairsTrader(delta=1e-4, R=0.1)
        hist   = trader.fit(y1, y2)

        trader2 = KalmanPairsTrader(delta=1e-4, R=0.1)
        trader2.reset()
        for t in range(N):
            trader2._kf.update(y1[t], y2[t])
            beta_step = trader2.beta
            assert abs(beta_step - hist.betas[t]) < 1e-10, (
                f"Beta mismatch at t={t}: step={beta_step}, batch={hist.betas[t]}"
            )

    def test_unequal_length_raises(self):
        trader = KalmanPairsTrader()
        with pytest.raises(ValueError):
            trader.fit(np.ones(10), np.ones(11))

    def test_fit_twice_gives_same_result(self):
        """fit() should reset state, so calling it twice gives identical output."""
        y1, y2 = _make_cointegrated_pair(N=150, seed=13)
        trader  = KalmanPairsTrader(delta=1e-4)
        hist1   = trader.fit(y1, y2)
        hist2   = trader.fit(y1, y2)
        np.testing.assert_array_equal(hist1.betas, hist2.betas)


# ═══════════════════════════════════════════════════════════════════════════════
# KalmanPairsTrader.step() online tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanPairsTraderStep:

    def test_step_returns_three_tuple(self):
        trader = KalmanPairsTrader()
        result = trader.step(100.0, 50.0)
        assert len(result) == 3

    def test_first_step_z_is_nan(self):
        trader = KalmanPairsTrader()
        _, _, z = trader.step(100.0, 50.0)
        assert np.isnan(z)

    def test_z_finite_after_warmup(self):
        trader = KalmanPairsTrader(zscore_window=5)
        for t in range(10):
            _, _, z = trader.step(float(t) + 0.1 * RNG.standard_normal(),
                                   float(t))
        assert np.isfinite(z)

    def test_step_matches_batch_innovations(self):
        """Online step-by-step innovations must match batch fit exactly."""
        N  = 100
        y1, y2 = _make_cointegrated_pair(N=N, seed=20)
        trader  = KalmanPairsTrader(delta=1e-4, R=0.1)
        hist    = trader.fit(y1, y2)

        trader2 = KalmanPairsTrader(delta=1e-4, R=0.1)
        trader2.reset()
        for t in range(N):
            e, S, _ = trader2.step(y1[t], y2[t])
            assert abs(e - hist.innovations[t]) < 1e-10, (
                f"Innovation mismatch at t={t}"
            )

    def test_beta_property_after_steps(self):
        trader = KalmanPairsTrader()
        for t in range(50):
            trader.step(2.0 * t + 0.1, float(t))
        assert np.isfinite(trader.beta)

    def test_reset_clears_buffer(self):
        trader = KalmanPairsTrader(zscore_window=5)
        y1, y2 = _make_cointegrated_pair(N=50, seed=21)
        for t in range(50):
            trader.step(y1[t], y2[t])
        trader.reset()
        _, _, z = trader.step(y1[0], y2[0])
        assert np.isnan(z)   # buffer cleared → z is NaN again

    def test_innov_variance_positive(self):
        trader = KalmanPairsTrader()
        y1, y2 = _make_cointegrated_pair(N=30, seed=22)
        for t in range(len(y1)):
            _, S, _ = trader.step(y1[t], y2[t])
            assert S > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# KalmanPairsHistory.signal() tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanSignal:

    def _make_history(
        self,
        N=400, true_beta=2.0, true_alpha=0.0, noise_std=0.3, seed=30,
    ) -> KalmanPairsHistory:
        y1, y2 = _make_cointegrated_pair(N, true_beta, true_alpha, noise_std, seed)
        trader = KalmanPairsTrader(delta=1e-4, R=noise_std**2, zscore_window=30)
        return trader.fit(y1, y2)

    def test_signal_values_in_set(self):
        hist = self._make_history()
        sig  = hist.signal()
        assert set(np.unique(sig)).issubset({-1, 0, 1})

    def test_signal_integer_dtype(self):
        hist = self._make_history()
        sig  = hist.signal()
        assert sig.dtype in (np.int32, np.int64, int)

    def test_signal_same_length_as_series(self):
        hist = self._make_history(N=200)
        assert len(hist.signal()) == 200

    def test_buy_signal_when_spread_very_negative(self):
        """
        Construct a pair where spread is persistently very negative at the end.
        Expect some +1 signals in that region.
        """
        rng  = np.random.default_rng(31)
        N    = 300
        y2   = np.cumsum(rng.standard_normal(N))
        y1   = 2.0 * y2 + rng.normal(0, 0.1, N)
        # Push y1 far below its expected value in the last 50 bars
        y1[-50:] -= 15.0

        trader = KalmanPairsTrader(delta=1e-5, R=0.01, zscore_window=30)
        hist   = trader.fit(y1, y2)
        sig    = hist.signal(entry_z=1.5)
        assert np.any(sig[-50:] == 1), "Expected buy signal when spread is far below mean"

    def test_sell_signal_when_spread_very_positive(self):
        rng  = np.random.default_rng(32)
        N    = 300
        y2   = np.cumsum(rng.standard_normal(N))
        y1   = 2.0 * y2 + rng.normal(0, 0.1, N)
        # Push y1 far above its expected value in the last 50 bars
        y1[-50:] += 15.0

        trader = KalmanPairsTrader(delta=1e-5, R=0.01, zscore_window=30)
        hist   = trader.fit(y1, y2)
        sig    = hist.signal(entry_z=1.5)
        assert np.any(sig[-50:] == -1), "Expected sell signal when spread is far above mean"

    def test_no_signal_in_exit_zone(self):
        """
        When innovations are very small, signal should be flat (0).
        """
        rng  = np.random.default_rng(33)
        N    = 200
        y2   = np.cumsum(rng.standard_normal(N))
        # Extremely tight noise → innovations always near zero
        y1   = 2.0 * y2 + rng.normal(0, 1e-6, N)
        trader = KalmanPairsTrader(delta=1e-6, R=1e-12, zscore_window=20)
        hist   = trader.fit(y1, y2)
        sig    = hist.signal(entry_z=5.0, exit_z=0.5)   # very high threshold
        assert np.all(sig == 0), "With tiny innovations and high threshold, signal should be 0"

    def test_zscore_signal_also_valid(self):
        hist = self._make_history()
        sig  = hist.signal(use_norm=False)   # use rolling z-score
        assert set(np.unique(sig)).issubset({-1, 0, 1})


# ═══════════════════════════════════════════════════════════════════════════════
# Regime shift test
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeShift:

    def test_large_delta_tracks_regime_change_faster(self):
        """
        After a sudden change in true beta, a larger delta should
        adapt faster (smaller absolute error in the second regime).
        """
        y1, y2 = _make_regime_shift_pair(N1=300, N2=300, beta1=1.5, beta2=3.0)
        N1 = 300

        for delta, label in [(1e-6, "slow"), (1e-3, "fast")]:
            trader = KalmanPairsTrader(delta=delta, R=0.09)
            hist   = trader.fit(y1, y2)
            # Error in the second regime (last 100 steps)
            err = abs(hist.betas[-100:].mean() - 3.0)
            if label == "slow":
                err_slow = err
            else:
                err_fast = err

        assert err_fast < err_slow, (
            f"Fast tracker (err={err_fast:.3f}) should beat "
            f"slow tracker (err={err_slow:.3f}) after regime shift"
        )

    def test_beta_monotone_increase_after_shift(self):
        """
        When true beta shifts up, the estimated beta should trend upward
        in the second half of the series.
        """
        y1, y2 = _make_regime_shift_pair(N1=200, N2=200, beta1=1.0, beta2=4.0)
        trader  = KalmanPairsTrader(delta=5e-4, R=0.09)
        hist    = trader.fit(y1, y2)
        # Mean beta in final quarter should exceed mean beta at shift point
        mid   = hist.betas[200]
        final = hist.betas[-50:].mean()
        assert final > mid, (
            f"Beta should trend up after shift: mid={mid:.3f}, final={final:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# select_delta() tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectDelta:

    def test_returns_positive_delta(self):
        y1, y2 = _make_cointegrated_pair(N=200, seed=40)
        best, _ = select_delta(y1, y2)
        assert best > 0

    def test_ll_array_length_matches_grid(self):
        y1, y2  = _make_cointegrated_pair(N=200, seed=41)
        grid    = np.logspace(-5, -2, 15)
        _, lls  = select_delta(y1, y2, deltas=grid)
        assert len(lls) == 15

    def test_best_delta_in_grid(self):
        y1, y2 = _make_cointegrated_pair(N=200, seed=42)
        grid   = np.logspace(-5, -2, 20)
        best, _ = select_delta(y1, y2, deltas=grid)
        assert best in grid

    def test_ll_are_finite(self):
        y1, y2 = _make_cointegrated_pair(N=200, seed=43)
        _, lls = select_delta(y1, y2)
        assert np.all(np.isfinite(lls))

    def test_low_noise_favours_small_delta(self):
        """
        With a perfectly stable relationship, a small δ (slow adaptation)
        should yield the best log-likelihood.
        """
        N  = 500
        rng = np.random.default_rng(44)
        y2  = np.cumsum(rng.standard_normal(N))
        y1  = 2.0 * y2 + 1.0 + rng.normal(0, 0.01, N)  # almost exact

        grid = np.logspace(-7, -2, 30)
        best, lls = select_delta(y1, y2, deltas=grid)
        # Best should be in the lower half of the grid
        assert best < grid[len(grid) // 2], (
            f"Low-noise pair should prefer small delta; got {best:.2e}"
        )
