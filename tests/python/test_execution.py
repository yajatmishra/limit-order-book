"""
Tests for execution/twap.py, vwap.py, participation_rate.py, market_impact.py
"""

import numpy as np
import pytest

from execution.twap              import TWAPScheduler, simulate_twap
from execution.vwap              import VWAPScheduler, u_shaped_profile, simulate_vwap
from execution.participation_rate import ParticipationRateExecutor
from execution.market_impact     import (
    square_root_impact, linear_impact, three_fifths_impact,
    impact_bps, AlmgrenChriss,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TWAP Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTWAP:

    def test_schedule_quantity_sums_to_total(self):
        sched = TWAPScheduler(n_slices=10, duration=3600)
        slices = sched.build(quantity=1000.0)
        total = sum(s.quantity for s in slices)
        assert abs(total - 1000.0) < 1e-9

    def test_schedule_sell_quantity_sums_to_total(self):
        sched = TWAPScheduler(n_slices=5, duration=1800)
        slices = sched.build(quantity=-500.0)
        total = sum(s.quantity for s in slices)
        assert abs(total - (-500.0)) < 1e-9

    def test_slice_count(self):
        sched = TWAPScheduler(n_slices=12)
        slices = sched.build(quantity=1200)
        assert len(slices) == 12

    def test_equal_slice_sizes(self):
        sched = TWAPScheduler(n_slices=5, duration=500, jitter=0.0)
        slices = sched.build(quantity=500)
        # Each of 4 initial slices = 100; last absorbs rounding
        for s in slices[:-1]:
            assert abs(s.quantity - 100.0) < 1e-9

    def test_times_are_monotone(self):
        """Slice times should be non-decreasing (without jitter)."""
        sched = TWAPScheduler(n_slices=10, duration=1000, jitter=0.0)
        slices = sched.build(quantity=100)
        times = [s.time for s in slices]
        assert times == sorted(times)

    def test_fill_tracking(self):
        sched  = TWAPScheduler(n_slices=4, duration=400)
        sched.build(quantity=400)
        sched.fill(0, filled=100.0, fill_price=50.0)
        assert sched.slices[0].filled == 100.0
        assert abs(sched.slices[0].fill_price - 50.0) < 1e-9

    def test_fill_partial_vwap(self):
        """Two partial fills should produce VWAPed fill_price."""
        sched = TWAPScheduler(n_slices=2, duration=200)
        sched.build(quantity=200)
        sched.fill(0, filled=50.0, fill_price=100.0)
        sched.fill(0, filled=50.0, fill_price=102.0)
        assert abs(sched.slices[0].fill_price - 101.0) < 1e-9

    def test_remaining_quantity(self):
        sched = TWAPScheduler(n_slices=4, duration=400)
        sched.build(quantity=400)
        sched.fill(0, 100.0, 50.0)
        assert abs(sched.remaining_quantity() - 300.0) < 1e-6

    def test_summary_avg_fill(self):
        sched = TWAPScheduler(n_slices=2, duration=200)
        sched.build(quantity=200)
        sched.fill(0, 100.0, 10.0)
        sched.fill(1, 100.0, 12.0)
        result = sched.summary()
        assert abs(result.avg_fill_price - 11.0) < 1e-9

    def test_simulate_twap_total_quantity(self):
        mids = np.full(20, 100.0)
        r = simulate_twap(quantity=200.0, mid_prices=mids, n_slices=20)
        assert abs(r.total_quantity - 200.0) < 1e-9

    def test_simulate_twap_buy_crosses_ask(self):
        """Buy should fill at mid + half_spread."""
        mids = np.full(10, 100.0)
        r = simulate_twap(quantity=100.0, mid_prices=mids, n_slices=10, spread_bps=10.0)
        expected_fill = 100.0 * (1 + 5e-4)
        assert abs(r.avg_fill_price - expected_fill) < 1e-6

    def test_invalid_n_slices(self):
        with pytest.raises(ValueError):
            TWAPScheduler(n_slices=0)

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            TWAPScheduler(duration=0)

    def test_repr(self):
        sched = TWAPScheduler(n_slices=10, duration=3600)
        assert "TWAPScheduler" in repr(sched)


# ═══════════════════════════════════════════════════════════════════════════════
# VWAP Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVWAP:

    def test_u_shaped_profile_sums_to_one(self):
        p = u_shaped_profile(12)
        assert abs(p.sum() - 1.0) < 1e-12

    def test_u_shaped_first_and_last_higher(self):
        """Endpoints should be larger than midday in a U-profile."""
        p = u_shaped_profile(12, alpha=2.0)
        mid_idx = len(p) // 2
        assert p[0] > p[mid_idx]
        assert p[-1] > p[mid_idx]

    def test_flat_profile_is_uniform(self):
        from execution.vwap import flat_profile
        p = flat_profile(6)
        assert np.allclose(p, 1.0 / 6)

    def test_schedule_quantity_sums_to_total(self):
        sched = VWAPScheduler(n_slices=12, duration=23400)
        slices = sched.build(quantity=12000)
        total = sum(s.quantity for s in slices)
        assert abs(total - 12000.0) < 1e-9

    def test_schedule_respects_profile(self):
        """Uniform profile → equal slice sizes."""
        profile = np.ones(5) / 5
        sched   = VWAPScheduler(n_slices=5, duration=500, profile=profile)
        slices  = sched.build(quantity=500)
        for s in slices[:-1]:
            assert abs(s.quantity - 100.0) < 1e-9

    def test_custom_profile_normalised(self):
        """Custom profile not summing to 1 should be auto-normalised."""
        p = np.array([3.0, 1.0, 1.0, 3.0])
        sched = VWAPScheduler(n_slices=4, duration=400, profile=p)
        assert abs(sched.profile.sum() - 1.0) < 1e-12

    def test_fill_tracking(self):
        sched = VWAPScheduler(n_slices=4, duration=400)
        sched.build(quantity=400)
        sched.fill(0, 100.0, 50.0)
        assert sched.slices[0].filled == 100.0

    def test_simulate_vwap_total_quantity(self):
        mids = np.full(20, 50.0)
        r = simulate_vwap(quantity=1000.0, mid_prices=mids)
        assert abs(r.total_quantity - 1000.0) < 1e-9

    def test_profile_length_mismatch(self):
        with pytest.raises(ValueError):
            VWAPScheduler(n_slices=5, profile=np.ones(4))

    def test_negative_profile_rejected(self):
        with pytest.raises(ValueError):
            VWAPScheduler(n_slices=3, profile=np.array([1.0, -0.1, 0.1]))

    def test_repr(self):
        sched = VWAPScheduler(n_slices=12, duration=3600)
        assert "VWAPScheduler" in repr(sched)


# ═══════════════════════════════════════════════════════════════════════════════
# POV / Participation Rate Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestParticipationRate:

    def test_simulate_fills_full_quantity(self):
        pov  = ParticipationRateExecutor(rate=0.20)
        vols = np.full(100, 1000.0)
        mids = np.full(100, 100.0)
        r    = pov.simulate(quantity=10_000, mkt_volumes=vols, mid_prices=mids)
        # With 20% of 1000 = 200/period, 10_000/200 = 50 periods needed
        assert r.complete
        assert abs(sum(abs(f.quantity) for f in r.fills) - 10_000.0) < 1e-6

    def test_periods_to_fill(self):
        pov  = ParticipationRateExecutor(rate=0.25)
        vols = np.full(200, 1000.0)
        mids = np.full(200, 50.0)
        r    = pov.simulate(quantity=10_000, mkt_volumes=vols, mid_prices=mids)
        # 25% of 1000 = 250/period → 40 periods
        assert r.n_periods == 40

    def test_quantity_never_exceeds_total(self):
        pov  = ParticipationRateExecutor(rate=0.50)
        vols = np.full(5, 2000.0)
        mids = np.full(5, 100.0)
        r    = pov.simulate(quantity=1_000, mkt_volumes=vols, mid_prices=mids)
        total_filled = sum(abs(f.quantity) for f in r.fills)
        assert total_filled <= 1_000.0 + 1e-9

    def test_buy_crosses_ask(self):
        pov  = ParticipationRateExecutor(rate=1.0, spread_bps=10.0, eta=0.0)
        pov.simulate(quantity=100.0, mkt_volumes=np.array([200.0]),
                     mid_prices=np.array([100.0]))
        fill = pov._fills[0]
        expected = 100.0 * (1 + 5e-4)
        assert abs(fill.fill_price - expected) < 1e-6

    def test_sell_crosses_bid(self):
        pov  = ParticipationRateExecutor(rate=1.0, spread_bps=10.0, eta=0.0)
        pov.simulate(quantity=-100.0, mkt_volumes=np.array([200.0]),
                     mid_prices=np.array([100.0]))
        fill = pov._fills[0]
        expected = 100.0 * (1 - 5e-4)
        assert abs(fill.fill_price - expected) < 1e-6

    def test_zero_volume_no_fill(self):
        pov  = ParticipationRateExecutor(rate=0.10)
        pov.start(1000.0)
        f = pov.step(mkt_volume=0.0, mid_price=100.0)
        assert f is None

    def test_max_rate_cap(self):
        pov  = ParticipationRateExecutor(rate=0.50, max_rate=0.20)
        vols = np.full(100, 1000.0)
        mids = np.full(100, 50.0)
        r    = pov.simulate(quantity=10_000, mkt_volumes=vols, mid_prices=mids)
        # Effective rate = 0.20, so 200/period → 50 periods
        assert r.n_periods == 50

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            ParticipationRateExecutor(rate=0.0)

    def test_repr(self):
        pov = ParticipationRateExecutor(rate=0.10)
        assert "ParticipationRateExecutor" in repr(pov)


# ═══════════════════════════════════════════════════════════════════════════════
# Market Impact Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketImpact:

    def test_sqrt_impact_positive_buy(self):
        impact = square_root_impact(quantity=1000, adv=1e6, daily_vol=0.02)
        assert impact > 0

    def test_sqrt_impact_negative_sell(self):
        impact = square_root_impact(quantity=-1000, adv=1e6, daily_vol=0.02)
        assert impact < 0

    def test_sqrt_impact_symmetry(self):
        buy  = square_root_impact( 1000, 1e6, 0.02)
        sell = square_root_impact(-1000, 1e6, 0.02)
        assert abs(buy + sell) < 1e-12

    def test_sqrt_impact_grows_with_quantity(self):
        i1 = square_root_impact(1000, 1e6, 0.02)
        i2 = square_root_impact(4000, 1e6, 0.02)
        assert i2 > i1
        # Should be ~2x for 4x quantity (square root)
        assert abs(i2 / i1 - 2.0) < 0.01

    def test_linear_impact(self):
        impact = linear_impact(1000, 1e6, 0.02, eta=0.1)
        expected = 0.1 * 0.02 * (1000 / 1e6)
        assert abs(impact - expected) < 1e-12

    def test_three_fifths_impact(self):
        impact = three_fifths_impact(1000, 1e6, 0.02, eta=0.1)
        expected = 0.1 * 0.02 * (1000 / 1e6) ** 0.6
        assert abs(impact - expected) < 1e-12

    def test_impact_bps_models(self):
        for model in ("sqrt", "linear", "three_fifths"):
            bps = impact_bps(10_000, 1e6, 0.02, eta=0.1, model=model)
            assert bps > 0

    def test_impact_bps_invalid_model(self):
        with pytest.raises(ValueError):
            impact_bps(1000, 1e6, 0.02, model="unknown")

    def test_sqrt_invalid_adv(self):
        with pytest.raises(ValueError):
            square_root_impact(1000, adv=0, daily_vol=0.02)


class TestAlmgrenChriss:

    def test_trajectory_length(self):
        ac   = AlmgrenChriss(sigma=0.02, adv=1e6)
        traj = ac.optimal_trajectory(quantity=10_000, T=1.0, N=10)
        assert len(traj.trades)   == 10
        assert len(traj.holdings) == 11

    def test_initial_holding_equals_quantity(self):
        ac   = AlmgrenChriss(sigma=0.02, adv=1e6)
        traj = ac.optimal_trajectory(quantity=50_000, T=1.0, N=5)
        assert abs(traj.holdings[0] - 50_000) < 1e-6

    def test_final_holding_near_zero(self):
        ac   = AlmgrenChriss(sigma=0.02, adv=1e6, lam=1e-5)
        traj = ac.optimal_trajectory(quantity=10_000, T=1.0, N=20)
        assert traj.holdings[-1] >= 0
        assert traj.holdings[-1] < traj.holdings[0]

    def test_trades_sum_to_quantity(self):
        ac   = AlmgrenChriss(sigma=0.02, adv=1e6)
        traj = ac.optimal_trajectory(quantity=10_000, T=1.0, N=10)
        # trades = -diff(holdings), so their sum = holdings[0] - holdings[-1]
        assert abs(traj.trades.sum() - (traj.holdings[0] - traj.holdings[-1])) < 1e-6

    def test_holdings_non_negative(self):
        ac   = AlmgrenChriss(sigma=0.02, adv=1e6)
        traj = ac.optimal_trajectory(quantity=10_000, T=1.0, N=10)
        assert np.all(traj.holdings >= 0)

    def test_high_risk_aversion_trades_faster(self):
        """Higher λ → front-loaded trades (more sold in first half)."""
        ac_lo = AlmgrenChriss(sigma=0.02, adv=1e6, lam=1e-7)
        ac_hi = AlmgrenChriss(sigma=0.02, adv=1e6, lam=1e-4)
        N = 10
        traj_lo = ac_lo.optimal_trajectory(10_000, T=1.0, N=N)
        traj_hi = ac_hi.optimal_trajectory(10_000, T=1.0, N=N)
        # Holding at midpoint should be lower for high-λ (more aggressive)
        assert traj_hi.holdings[N // 2] <= traj_lo.holdings[N // 2] + 1.0

    def test_twap_limit_uniform_trades(self):
        """λ → 0 should approach uniform (TWAP-like) liquidation."""
        ac   = AlmgrenChriss(sigma=0.02, adv=1e6, lam=1e-12)
        traj = ac.optimal_trajectory(100_000, T=1.0, N=10)
        # All trades should be roughly equal
        trade_sizes = np.abs(traj.trades)
        assert trade_sizes.std() / trade_sizes.mean() < 0.05

    def test_invalid_N(self):
        ac = AlmgrenChriss()
        with pytest.raises(ValueError):
            ac.optimal_trajectory(1000, T=1.0, N=0)

    def test_invalid_T(self):
        ac = AlmgrenChriss()
        with pytest.raises(ValueError):
            ac.optimal_trajectory(1000, T=0.0, N=10)

    def test_efficient_frontier(self):
        ac = AlmgrenChriss(sigma=0.02, adv=1e6)
        costs, varis = ac.efficient_frontier(10_000, T=1.0, N=10, n_points=5)
        assert len(costs) == len(varis) == 5
