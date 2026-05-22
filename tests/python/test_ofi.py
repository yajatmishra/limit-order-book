"""
Tests for microstructure.ofi
=============================
Covers:
  - Single-level OFI: bid improved / worsened / unchanged
  - Ask-side symmetry
  - Simultaneous bid-ask moves
  - Zero-length and minimal inputs
  - Multi-level OFI: aggregation and custom weights
  - Normalized OFI: scaling and zero-guard
  - Rolling OFI: NaN head, correct values, sum mode
  - Price-impact regression: known-slope data, R² = 1
  - ofi_from_book_snapshots convenience wrapper
"""

import numpy as np
import pytest

from microstructure.ofi import (
    compute_ofi,
    compute_multi_level_ofi,
    normalized_ofi,
    rolling_ofi,
    ofi_price_impact_regression,
    ofi_from_book_snapshots,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _book(bp, bq, ap, aq):
    """Package scalar tick values into length-1 arrays."""
    return (np.array([bp], dtype=float),
            np.array([bq], dtype=float),
            np.array([ap], dtype=float),
            np.array([aq], dtype=float))


def _two_tick_ofi(bp0, bq0, ap0, aq0, bp1, bq1, ap1, aq1):
    """Compute single OFI value for a pair of ticks."""
    bp = np.array([bp0, bp1], dtype=float)
    bq = np.array([bq0, bq1], dtype=float)
    ap = np.array([ap0, ap1], dtype=float)
    aq = np.array([aq0, aq1], dtype=float)
    result = compute_ofi(bp, bq, ap, aq)
    assert result.shape == (1,)
    return result[0]


# ── Single-level OFI ───────────────────────────────────────────────────────────

class TestComputeOfi:

    def test_output_length(self):
        bp = np.arange(5, dtype=float)
        bq = np.ones(5)
        ap = bp + 1.0
        aq = np.ones(5)
        assert compute_ofi(bp, bq, ap, aq).shape == (4,)

    def test_empty_on_single_tick(self):
        result = compute_ofi(
            np.array([100.0]), np.array([10.0]),
            np.array([101.0]), np.array([10.0]),
        )
        assert result.shape == (0,)

    def test_bid_improves_positive_ofi(self):
        # Bid price rises: e_bid = Q_t^b > 0, e_ask = 0
        val = _two_tick_ofi(100, 10, 101, 10,   # tick 0
                             101, 15, 101, 10)  # tick 1: bid price ↑
        assert val > 0, "Bid improvement should give positive OFI"

    def test_bid_worsens_negative_ofi(self):
        # Bid price falls: e_bid = −Q_{t-1}^b < 0
        val = _two_tick_ofi(101, 10, 102, 10,
                             100, 10, 102, 10)  # bid ↓
        assert val < 0, "Bid deterioration should give negative OFI"

    def test_ask_improves_negative_ofi(self):
        # Ask price falls (improvement for sellers) → e_ask > 0 → OFI < 0
        val = _two_tick_ofi(100, 10, 102, 15,
                             100, 10, 101, 15)  # ask ↓
        assert val < 0, "Ask improvement should give negative OFI"

    def test_ask_worsens_positive_ofi(self):
        # Ask price rises (less selling pressure) → e_ask < 0 → OFI > 0
        val = _two_tick_ofi(100, 10, 101, 10,
                             100, 10, 102, 10)  # ask ↑
        assert val > 0, "Ask deterioration should give positive OFI"

    def test_unchanged_book_zero_ofi(self):
        # No changes at all: e_bid = Q - Q = 0, e_ask = Q - Q = 0
        val = _two_tick_ofi(100, 10, 101, 10,
                             100, 10, 101, 10)
        assert val == pytest.approx(0.0)

    def test_same_price_size_change_bid(self):
        # Bid price unchanged, size increases: e_bid = Q_t - Q_{t-1} > 0
        val = _two_tick_ofi(100, 5, 101, 10,
                             100, 8, 101, 10)
        assert val == pytest.approx(3.0)   # e_bid = 8 - 5 = 3

    def test_same_price_size_change_ask(self):
        # Ask price unchanged, size decreases: e_ask = Q_t - Q_{t-1} < 0 → OFI > 0
        val = _two_tick_ofi(100, 10, 101, 10,
                             100, 10, 101, 6)
        assert val == pytest.approx(4.0)   # OFI = 0 - (6 - 10) = 4

    def test_simultaneous_bid_up_ask_up(self):
        # Bid improves, ask worsens: both push OFI positive
        val = _two_tick_ofi(100, 10, 101, 10,
                             101, 12, 102, 10)
        assert val > 0

    def test_ofi_antisymmetry(self):
        # Swap bid and ask role: should give opposite sign
        ofi_buy  = _two_tick_ofi(100, 10, 101, 10, 101, 10, 101, 10)
        ofi_sell = _two_tick_ofi(100, 10, 101, 10, 100, 10, 100, 10)
        # bid improved → positive; ask improved (price fell) → negative
        assert ofi_buy > 0
        assert ofi_sell < 0

    def test_series_shape(self):
        N = 20
        bp = 100.0 + np.cumsum(np.random.default_rng(0).standard_normal(N))
        bq = np.abs(np.random.default_rng(1).standard_normal(N)) + 1
        ap = bp + 0.5
        aq = bq.copy()
        ofi = compute_ofi(bp, bq, ap, aq)
        assert ofi.shape == (N - 1,)
        assert np.all(np.isfinite(ofi))

    def test_large_size_move(self):
        # Giant order appears on bid side: OFI should be large and positive
        val = _two_tick_ofi(100, 1, 101, 1,
                             100, 10_000, 101, 1)
        assert val == pytest.approx(9_999.0)   # e_bid = 10000 - 1 = 9999


# ── Multi-level OFI ────────────────────────────────────────────────────────────

class TestMultiLevelOfi:

    def _build_book(self, N, L, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.uniform(99, 101, size=(N, L))
        sizes = rng.uniform(1, 20, size=(N, L))
        bids = np.stack([base,           sizes], axis=2)
        asks = np.stack([base + 0.5,     sizes], axis=2)
        return bids, asks

    def test_output_shape(self):
        bids, asks = self._build_book(10, 3)
        ofi = compute_multi_level_ofi(bids, asks)
        assert ofi.shape == (9,)

    def test_uniform_weights_equal_mean(self):
        bids, asks = self._build_book(8, 4)
        ofi_uniform = compute_multi_level_ofi(bids, asks)

        # Manually compute per-level OFIs and average
        L = 4
        per_level = np.stack([
            compute_ofi(bids[:, l, 0], bids[:, l, 1],
                        asks[:, l, 0], asks[:, l, 1])
            for l in range(L)
        ], axis=1)
        expected = per_level.mean(axis=1)
        np.testing.assert_allclose(ofi_uniform, expected)

    def test_custom_weights(self):
        bids, asks = self._build_book(6, 3)
        w = np.array([0.5, 0.3, 0.2])
        ofi = compute_multi_level_ofi(bids, asks, weights=w)
        assert ofi.shape == (5,)

    def test_wrong_weight_shape_raises(self):
        bids, asks = self._build_book(5, 3)
        with pytest.raises(ValueError):
            compute_multi_level_ofi(bids, asks, weights=np.array([0.5, 0.5]))

    def test_single_level_matches_single_fn(self):
        bids, asks = self._build_book(7, 1)
        multi = compute_multi_level_ofi(bids, asks, weights=np.array([1.0]))
        single = compute_ofi(bids[:, 0, 0], bids[:, 0, 1],
                             asks[:, 0, 0], asks[:, 0, 1])
        np.testing.assert_allclose(multi, single)


# ── Normalized OFI ─────────────────────────────────────────────────────────────

class TestNormalizedOfi:

    def test_scales_correctly(self):
        ofi = np.array([10.0, -5.0, 20.0])
        vol = np.array([100.0, 50.0, 200.0])
        result = normalized_ofi(ofi, vol)
        np.testing.assert_allclose(result, [0.1, -0.1, 0.1])

    def test_zero_volume_guard(self):
        ofi = np.array([5.0])
        vol = np.array([0.0])
        result = normalized_ofi(ofi, vol, min_vol=1.0)
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(5.0)

    def test_output_shape(self):
        ofi = np.ones(8)
        vol = np.full(8, 2.0)
        assert normalized_ofi(ofi, vol).shape == (8,)


# ── Rolling OFI ───────────────────────────────────────────────────────────────

class TestRollingOfi:

    def test_first_window_minus_one_are_nan(self):
        ofi = np.arange(10, dtype=float)
        result = rolling_ofi(ofi, window=4)
        assert np.all(np.isnan(result[:3]))
        assert np.all(np.isfinite(result[3:]))

    def test_mean_value(self):
        ofi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_ofi(ofi, window=3)
        assert result[2] == pytest.approx(2.0)   # mean(1,2,3)
        assert result[3] == pytest.approx(3.0)   # mean(2,3,4)
        assert result[4] == pytest.approx(4.0)   # mean(3,4,5)

    def test_sum_mode(self):
        ofi = np.array([1.0, 2.0, 3.0, 4.0])
        result = rolling_ofi(ofi, window=2, agg="sum")
        assert result[1] == pytest.approx(3.0)
        assert result[3] == pytest.approx(7.0)

    def test_window_one_is_identity(self):
        ofi = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(rolling_ofi(ofi, window=1), ofi)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            rolling_ofi(np.ones(5), window=0)

    def test_full_series_length_preserved(self):
        ofi = np.linspace(0, 1, 20)
        result = rolling_ofi(ofi, window=5)
        assert result.shape == (20,)


# ── OFI price-impact regression ───────────────────────────────────────────────

class TestOfiPriceImpactRegression:

    def test_perfect_fit(self):
        # ΔS = 0.5 · OFI exactly → R² = 1, β = 0.5
        ofi = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
        ds  = 0.5 * ofi
        beta, r2 = ofi_price_impact_regression(ofi, ds)
        assert beta == pytest.approx(0.5, abs=1e-10)
        assert r2   == pytest.approx(1.0, abs=1e-10)

    def test_zero_ofi_no_crash(self):
        ofi = np.zeros(5)
        ds  = np.ones(5)
        beta, r2 = ofi_price_impact_regression(ofi, ds)
        assert np.isfinite(beta)

    def test_r2_between_0_and_1(self):
        rng = np.random.default_rng(7)
        ofi = rng.standard_normal(100)
        ds  = 0.3 * ofi + 0.1 * rng.standard_normal(100)
        _, r2 = ofi_price_impact_regression(ofi, ds)
        assert 0.0 <= r2 <= 1.0

    def test_negative_impact_coefficient(self):
        ofi = np.array([1.0, 2.0, 3.0])
        ds  = -0.2 * ofi
        beta, _ = ofi_price_impact_regression(ofi, ds)
        assert beta < 0


# ── Convenience wrapper ────────────────────────────────────────────────────────

class TestOfiFromBookSnapshots:

    def test_basic(self):
        snaps = np.array([
            [100.0, 10.0, 101.0, 10.0],
            [100.5, 12.0, 101.5,  9.0],
            [100.0,  8.0, 101.0, 11.0],
        ])
        ofi = ofi_from_book_snapshots(snaps)
        assert ofi.shape == (2,)
        assert np.all(np.isfinite(ofi))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            ofi_from_book_snapshots(np.ones((5, 3)))
