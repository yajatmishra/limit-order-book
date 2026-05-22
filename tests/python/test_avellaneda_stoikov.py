"""
Tests for microstructure.avellaneda_stoikov
============================================
Covers:
  - ASParams validation (non-positive parameters rejected)
  - reservation_price: formula correctness, zero-inventory identity, sign
  - optimal_half_spread: formula, monotonicity in γ / σ / T-t / k
  - optimal_quotes: bid < reservation < ask, symmetry around reservation price
  - arrival_rate: decay with depth, non-negative
  - AvellanedaStoikov.simulate: runs to completion, inventory bounds,
    fill counts, PnL series properties, determinism with fixed seed
  - quote_grid: shape, monotone reservation price in inventory
  - Integration: higher risk aversion → wider spread; more time → wider spread
"""

import numpy as np
import pytest

from microstructure.avellaneda_stoikov import (
    ASParams,
    reservation_price,
    optimal_half_spread,
    optimal_quotes,
    arrival_rate,
    AvellanedaStoikov,
    SimResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def base_params():
    return ASParams(gamma=0.1, sigma=2.0, k=1.5, A=140.0, T=1.0)


@pytest.fixture
def base_model(base_params):
    return AvellanedaStoikov(base_params)


# ── ASParams validation ────────────────────────────────────────────────────────

class TestASParams:

    def test_valid_construction(self, base_params):
        assert base_params.gamma == 0.1
        assert base_params.T     == 1.0

    @pytest.mark.parametrize("bad", [
        dict(gamma=0.0, sigma=2.0, k=1.5, A=140.0, T=1.0),
        dict(gamma=0.1, sigma=0.0, k=1.5, A=140.0, T=1.0),
        dict(gamma=0.1, sigma=2.0, k=0.0, A=140.0, T=1.0),
        dict(gamma=0.1, sigma=2.0, k=1.5, A=0.0,   T=1.0),
        dict(gamma=0.1, sigma=2.0, k=1.5, A=140.0, T=0.0),
        dict(gamma=-1.0, sigma=2.0, k=1.5, A=140.0, T=1.0),
    ])
    def test_non_positive_raises(self, bad):
        with pytest.raises(ValueError):
            ASParams(**bad)


# ── reservation_price ─────────────────────────────────────────────────────────

class TestReservationPrice:

    def test_zero_inventory_equals_mid(self):
        r = reservation_price(mid=100.0, inventory=0, gamma=0.1,
                              sigma=2.0, time_remaining=0.5)
        assert r == pytest.approx(100.0)

    def test_long_inventory_below_mid(self):
        r = reservation_price(mid=100.0, inventory=5, gamma=0.1,
                              sigma=2.0, time_remaining=1.0)
        assert r < 100.0, "Long inventory should push reservation price below mid"

    def test_short_inventory_above_mid(self):
        r = reservation_price(mid=100.0, inventory=-5, gamma=0.1,
                              sigma=2.0, time_remaining=1.0)
        assert r > 100.0, "Short inventory should push reservation price above mid"

    def test_exact_formula(self):
        mid, q, gamma, sigma, tau = 100.0, 3.0, 0.2, 1.5, 0.8
        expected = mid - q * gamma * sigma ** 2 * tau
        result   = reservation_price(mid, q, gamma, sigma, tau)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_zero_time_equals_mid(self):
        r = reservation_price(mid=100.0, inventory=10, gamma=0.1,
                              sigma=2.0, time_remaining=0.0)
        assert r == pytest.approx(100.0)

    def test_larger_gamma_larger_discount(self):
        r_lo = reservation_price(100.0, 5, gamma=0.05, sigma=2.0, time_remaining=1.0)
        r_hi = reservation_price(100.0, 5, gamma=0.20, sigma=2.0, time_remaining=1.0)
        assert r_hi < r_lo, "Higher risk aversion → lower reservation price when long"


# ── optimal_half_spread ───────────────────────────────────────────────────────

class TestOptimalHalfSpread:

    def test_non_negative(self, base_params):
        delta = optimal_half_spread(base_params.gamma, base_params.sigma,
                                    time_remaining=0.5, k=base_params.k)
        assert delta >= 0.0

    def test_exact_formula(self):
        gamma, sigma, tau, k = 0.1, 2.0, 0.5, 1.5
        expected = 0.5 * gamma * sigma ** 2 * tau + (1.0 / gamma) * np.log(1.0 + gamma / k)
        result   = optimal_half_spread(gamma, sigma, tau, k)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_increases_with_sigma(self):
        d_lo = optimal_half_spread(0.1, 1.0, 0.5, 1.5)
        d_hi = optimal_half_spread(0.1, 3.0, 0.5, 1.5)
        assert d_hi > d_lo, "Higher volatility → wider optimal spread"

    def test_increases_with_gamma(self):
        d_lo = optimal_half_spread(0.05, 2.0, 0.5, 1.5)
        d_hi = optimal_half_spread(0.50, 2.0, 0.5, 1.5)
        assert d_hi > d_lo, "Higher risk aversion → wider optimal spread"

    def test_increases_with_time_remaining(self):
        d_short = optimal_half_spread(0.1, 2.0, 0.1, 1.5)
        d_long  = optimal_half_spread(0.1, 2.0, 1.0, 1.5)
        assert d_long > d_short, "More time remaining → wider spread"

    def test_at_zero_time_only_market_power_term(self):
        gamma, k = 0.1, 1.5
        d_zero = optimal_half_spread(gamma, 2.0, time_remaining=0.0, k=k)
        market_power = (1.0 / gamma) * np.log(1.0 + gamma / k)
        assert d_zero == pytest.approx(market_power, rel=1e-12)

    def test_decreases_with_k(self):
        # Higher k → arrivals more sensitive to depth → tighter spread
        d_lo_k = optimal_half_spread(0.1, 2.0, 0.5, k=0.5)
        d_hi_k = optimal_half_spread(0.1, 2.0, 0.5, k=5.0)
        assert d_hi_k < d_lo_k, "Higher k → tighter optimal spread"


# ── optimal_quotes ────────────────────────────────────────────────────────────

class TestOptimalQuotes:

    def test_bid_below_ask(self, base_params):
        bid, ask, r, d = optimal_quotes(100.0, 0, base_params, 0.5)
        assert bid < ask

    def test_symmetric_around_reservation(self, base_params):
        mid = 100.0
        for q in [-3, 0, 3]:
            bid, ask, r, d = optimal_quotes(mid, q, base_params, 0.5)
            assert bid == pytest.approx(r - d, rel=1e-12)
            assert ask == pytest.approx(r + d, rel=1e-12)

    def test_total_spread_formula(self, base_params):
        bid, ask, r, d = optimal_quotes(100.0, 0, base_params, time_remaining=0.8)
        gamma, sigma, k, tau = (base_params.gamma, base_params.sigma,
                                 base_params.k, 0.8)
        expected_spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert (ask - bid) == pytest.approx(expected_spread, rel=1e-10)

    def test_zero_inventory_symmetric_around_mid(self, base_params):
        bid, ask, r, _ = optimal_quotes(100.0, 0, base_params, 0.5)
        # reservation price = mid when q = 0
        assert r == pytest.approx(100.0)
        assert bid < 100.0
        assert ask > 100.0

    def test_long_inventory_shifts_both_quotes_down(self, base_params):
        bid0, ask0, r0, _ = optimal_quotes(100.0, 0,  base_params, 0.5)
        bid5, ask5, r5, _ = optimal_quotes(100.0, 5,  base_params, 0.5)
        assert r5 < r0
        assert bid5 < bid0
        assert ask5 < ask0

    def test_reservation_moves_linearly_in_inventory(self, base_params):
        # r decreases linearly in q: slope = −γσ²(T−t)
        tau   = 0.6
        slope = -base_params.gamma * base_params.sigma ** 2 * tau
        _, _, r0, _ = optimal_quotes(100.0, 0, base_params, tau)
        _, _, r1, _ = optimal_quotes(100.0, 1, base_params, tau)
        _, _, r2, _ = optimal_quotes(100.0, 2, base_params, tau)
        assert (r1 - r0) == pytest.approx(slope, rel=1e-10)
        assert (r2 - r1) == pytest.approx(slope, rel=1e-10)


# ── arrival_rate ──────────────────────────────────────────────────────────────

class TestArrivalRate:

    def test_at_zero_depth_is_A(self):
        assert arrival_rate(0.0, A=140.0, k=1.5) == pytest.approx(140.0)

    def test_decays_with_depth(self):
        r0 = arrival_rate(0.0, 100.0, 1.5)
        r1 = arrival_rate(1.0, 100.0, 1.5)
        r2 = arrival_rate(2.0, 100.0, 1.5)
        assert r1 < r0
        assert r2 < r1

    def test_negative_depth_clamped_to_A(self):
        # Depth cannot be negative (quote inside midprice) → clamp to 0
        r = arrival_rate(-1.0, A=100.0, k=1.5)
        assert r == pytest.approx(100.0)

    def test_exponential_decay_rate(self):
        A, k = 100.0, 2.0
        d = 0.5
        expected = A * np.exp(-k * d)
        assert arrival_rate(d, A, k) == pytest.approx(expected, rel=1e-12)

    def test_non_negative_always(self):
        for d in [0, 0.1, 1.0, 10.0, 100.0]:
            assert arrival_rate(d, A=50.0, k=1.0) >= 0.0


# ── AvellanedaStoikov.simulate ────────────────────────────────────────────────

class TestSimulate:

    def test_runs_without_error(self, base_model):
        result = base_model.simulate(n_steps=200, seed=42)
        assert isinstance(result, SimResult)

    def test_history_length(self, base_model):
        result = base_model.simulate(n_steps=300, seed=0)
        assert len(result.history) == 300

    def test_inventory_within_limit(self, base_model):
        result = base_model.simulate(n_steps=500, max_inventory=5, seed=7)
        invs = result.inventory_series
        assert np.all(invs >= -5), "Inventory must not breach lower limit"
        assert np.all(invs <=  5), "Inventory must not breach upper limit"

    def test_fill_count_non_negative(self, base_model):
        result = base_model.simulate(n_steps=200)
        assert result.fill_count >= 0

    def test_deterministic_with_same_seed(self, base_model):
        r1 = base_model.simulate(n_steps=100, seed=99)
        r2 = base_model.simulate(n_steps=100, seed=99)
        assert r1.final_pnl == pytest.approx(r2.final_pnl)

    def test_different_seed_different_result(self, base_model):
        r1 = base_model.simulate(n_steps=200, seed=1)
        r2 = base_model.simulate(n_steps=200, seed=2)
        # Not guaranteed to differ, but very unlikely to be identical
        assert r1.fill_count != r2.fill_count or r1.final_pnl != r2.final_pnl

    def test_pnl_series_length(self, base_model):
        result = base_model.simulate(n_steps=150)
        assert len(result.pnl_series) == 150

    def test_all_mid_prices_positive(self, base_model):
        result = base_model.simulate(n_steps=200, s0=50.0, seed=42)
        # Starting at $50 with σ=2 and short horizon, should remain positive
        mids = np.array([s.mid for s in result.history])
        assert np.all(mids > 0)

    def test_bid_always_below_ask(self, base_model):
        result = base_model.simulate(n_steps=200, seed=42)
        for state in result.history:
            assert state.bid < state.ask, "Bid must always be below ask"

    def test_max_inventory_tracked(self, base_model):
        result = base_model.simulate(n_steps=300, max_inventory=4, seed=5)
        actual_max = int(result.inventory_series.__abs__().max())
        assert result.max_inventory == actual_max

    def test_initial_inventory_zero_and_q0(self, base_model):
        result = base_model.simulate(n_steps=100, q0=0, seed=1)
        assert result.history[0].inventory == 0

    def test_zero_fills_when_arrival_rate_very_low(self):
        # A is tiny → almost no fills
        p = ASParams(gamma=0.1, sigma=2.0, k=1.5, A=1e-6, T=1.0)
        model = AvellanedaStoikov(p)
        result = model.simulate(n_steps=200, seed=42)
        assert result.fill_count == 0


# ── quote_grid ────────────────────────────────────────────────────────────────

class TestQuoteGrid:

    def test_output_shape(self, base_model):
        invs = np.arange(-3, 4)
        grid = base_model.quote_grid(invs, time_remaining=0.5)
        assert grid.shape == (7, 4)

    def test_bid_below_ask_all_rows(self, base_model):
        invs = np.arange(-5, 6)
        grid = base_model.quote_grid(invs, time_remaining=0.5)
        assert np.all(grid[:, 0] < grid[:, 1])   # bid < ask

    def test_reservation_monotone_decreasing_in_inventory(self, base_model):
        invs = np.arange(-4, 5, dtype=float)
        grid = base_model.quote_grid(invs, time_remaining=0.5)
        reservations = grid[:, 2]
        # Reservation price = mid - q*γσ²τ → strictly decreasing in q
        diffs = np.diff(reservations)
        assert np.all(diffs < 0), "Reservation price must decrease with inventory"

    def test_constant_spread_across_inventory(self, base_model):
        # Half-spread δ* does not depend on q, only on t
        invs = np.arange(-5, 6, dtype=float)
        grid = base_model.quote_grid(invs, time_remaining=0.5)
        half_spreads = grid[:, 3]
        np.testing.assert_allclose(half_spreads, half_spreads[0], rtol=1e-12)


# ── Integration / sanity checks ───────────────────────────────────────────────

class TestIntegration:

    def test_higher_gamma_wider_spread(self):
        tau = 0.5
        d_lo = optimal_half_spread(gamma=0.05, sigma=2.0, time_remaining=tau, k=1.5)
        d_hi = optimal_half_spread(gamma=0.50, sigma=2.0, time_remaining=tau, k=1.5)
        assert d_hi > d_lo

    def test_higher_sigma_wider_spread(self):
        tau = 0.5
        d_lo = optimal_half_spread(gamma=0.1, sigma=1.0, time_remaining=tau, k=1.5)
        d_hi = optimal_half_spread(gamma=0.1, sigma=4.0, time_remaining=tau, k=1.5)
        assert d_hi > d_lo

    def test_simulation_pnl_finite(self, base_model):
        result = base_model.simulate(n_steps=500, seed=0)
        assert np.isfinite(result.final_pnl)

    def test_spread_collapses_at_end_of_day(self, base_params):
        # Near T=0, the risk term vanishes; only market-power term remains
        gamma, k = base_params.gamma, base_params.k
        tau_end = 1e-6
        d_end   = optimal_half_spread(gamma, base_params.sigma, tau_end, k)
        market_power = (1.0 / gamma) * np.log(1.0 + gamma / k)
        assert d_end == pytest.approx(market_power, rel=1e-3)

    def test_as_total_spread_equals_gamma_sigma_sq_tau_plus_market_power(self, base_params):
        tau = 0.7
        _, ask, _, d = optimal_quotes(100.0, 0, base_params, tau)
        gamma, sigma, k = base_params.gamma, base_params.sigma, base_params.k
        expected_total = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert (2 * d) == pytest.approx(expected_total, rel=1e-10)
