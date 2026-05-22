"""
Tests for risk/kelly_sizer.py, circuit_breakers.py,
          position_tracker.py, pnl_reporter.py
"""

import time
import numpy as np
import pytest

from risk.kelly_sizer     import KellySizer, KellyResult, multi_asset_kelly
from risk.circuit_breakers import (
    RiskGate, OrderEvent, CheckResult,
    MaxDrawdownBreaker, DailyLossBreaker,
    PositionLimitBreaker, VelocityLimitBreaker,
    NotionalLimitBreaker,
)
from risk.position_tracker import PositionTracker, SymbolState
from risk.pnl_reporter     import PnLReporter, _drawdown_series, _var_cvar


# ═══════════════════════════════════════════════════════════════════════════════
# KellySizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellySizer:

    def test_full_kelly_gaussian(self):
        """f* = μ / σ² for Gaussian returns."""
        sizer = KellySizer(fraction=1.0)
        result = sizer.size_from_moments(mu=0.001, sigma=0.02)
        expected_full = 0.001 / (0.02 ** 2)
        assert abs(result.full_kelly - expected_full) < 1e-10

    def test_fractional_kelly_half(self):
        sizer  = KellySizer(fraction=0.5)
        result = sizer.size_from_moments(mu=0.001, sigma=0.02)
        assert abs(result.fractional_kelly - result.full_kelly * 0.5) < 1e-10

    def test_size_in_shares(self):
        """size = fractional_kelly * equity / price."""
        sizer  = KellySizer(fraction=0.5)
        result = sizer.size_from_moments(mu=0.001, sigma=0.02,
                                          equity=100_000, price=100.0)
        expected = result.fractional_kelly * 100_000 / 100.0
        assert abs(result.final_size - expected) < 1e-6

    def test_max_position_constraint(self):
        sizer  = KellySizer(fraction=1.0, max_position=50.0)
        result = sizer.size_from_moments(mu=0.01, sigma=0.02,
                                          equity=1e6, price=1.0)
        assert abs(result.final_size) <= 50.0 + 1e-9
        assert result.constrained

    def test_max_leverage_constraint(self):
        sizer  = KellySizer(fraction=1.0, max_leverage=2.0)
        result = sizer.size_from_moments(mu=0.01, sigma=0.02,
                                          equity=100_000, price=100.0)
        leverage = abs(result.final_size) * 100.0 / 100_000
        assert leverage <= 2.0 + 1e-9

    def test_negative_edge_returns_negative_position(self):
        """Negative expected return should produce negative (short) Kelly."""
        sizer  = KellySizer(fraction=0.5)
        result = sizer.size_from_moments(mu=-0.001, sigma=0.02)
        assert result.final_size < 0

    def test_min_edge_filters_low_edge(self):
        """Edge below min_edge_bps should return zero size."""
        sizer  = KellySizer(fraction=0.5, min_edge_bps=100.0)
        result = sizer.size_from_moments(mu=0.00001, sigma=0.02, price=100.0)
        assert result.final_size == 0.0
        assert result.constrained

    def test_binary_kelly_even_odds(self):
        """At p=0.5, even odds → f* = 0."""
        sizer  = KellySizer(fraction=1.0)
        result = sizer.size_binary(p_win=0.5, payoff=1.0, equity=1000)
        assert abs(result.full_kelly) < 1e-12

    def test_binary_kelly_edge(self):
        """p=0.6, b=1: f* = (0.6*1 - 0.4)/1 = 0.2."""
        sizer  = KellySizer(fraction=1.0)
        result = sizer.size_binary(p_win=0.6, payoff=1.0, equity=1000)
        assert abs(result.full_kelly - 0.2) < 1e-10

    def test_sharpe_based_sizing(self):
        sizer  = KellySizer(fraction=0.5)
        result = sizer.size_from_sharpe(sharpe_ratio=1.0, sigma=0.02)
        # μ = SR * σ = 0.02, so f* = 0.02 / 0.02² = 50
        assert abs(result.full_kelly - 50.0) < 1e-9

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            KellySizer(fraction=0.0)
        with pytest.raises(ValueError):
            KellySizer(fraction=1.1)

    def test_invalid_sigma(self):
        sizer = KellySizer()
        with pytest.raises(ValueError):
            sizer.size_from_moments(mu=0.001, sigma=0.0)

    def test_invalid_binary_p(self):
        sizer = KellySizer()
        with pytest.raises(ValueError):
            sizer.size_binary(p_win=0.0, payoff=1.0)

    def test_repr(self):
        result = KellySizer().size_from_moments(mu=0.001, sigma=0.02)
        assert "KellyResult" in repr(result)


class TestMultiAssetKelly:

    def test_single_asset(self):
        """For 1 asset: f* = μ / σ²."""
        mu  = np.array([0.01])
        cov = np.array([[0.04]])
        w   = multi_asset_kelly(mu, cov, fraction=1.0)
        assert abs(w[0] - 0.01 / 0.04) < 1e-9

    def test_two_assets_uncorrelated(self):
        mu  = np.array([0.01, 0.005])
        cov = np.diag([0.04, 0.01])
        w   = multi_asset_kelly(mu, cov, fraction=1.0)
        # tolerance relaxed slightly for regularisation noise (1e-10 added to diag)
        assert abs(w[0] - 0.01 / 0.04)  < 1e-7
        assert abs(w[1] - 0.005 / 0.01) < 1e-7

    def test_fraction_scales_weights(self):
        mu  = np.array([0.01, 0.01])
        cov = np.diag([0.04, 0.04])
        w1  = multi_asset_kelly(mu, cov, fraction=1.0)
        w5  = multi_asset_kelly(mu, cov, fraction=0.5)
        assert np.allclose(w5, 0.5 * w1)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            multi_asset_kelly(np.array([0.01, 0.02]), np.eye(3))


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _event(symbol="SPY", quantity=100, price=100.0, ts=None):
    return OrderEvent(symbol=symbol, quantity=quantity, price=price,
                      timestamp=ts or time.time())


class TestMaxDrawdownBreaker:

    def test_allows_when_below_limit(self):
        b = MaxDrawdownBreaker(max_dd=1000)
        b.update_pnl(500)
        b.update_pnl(400)     # drawdown = 100
        assert b.check().allowed

    def test_blocks_when_at_limit(self):
        b = MaxDrawdownBreaker(max_dd=100)
        b.update_pnl(200)
        b.update_pnl(100)     # drawdown = 100 >= 100
        result = b.check()
        assert not result.allowed
        assert "MaxDrawdown" in result.reason

    def test_stays_halted_after_recovery(self):
        b = MaxDrawdownBreaker(max_dd=100)
        b.update_pnl(200)
        b.update_pnl(50)      # drawdown = 150 → halt
        b.check()
        b.update_pnl(300)     # recovery — but still halted
        assert not b.check().allowed

    def test_reset_clears_halt(self):
        b = MaxDrawdownBreaker(max_dd=100)
        b.update_pnl(200)
        b.update_pnl(50)
        b.check()
        b.reset()
        assert b.check().allowed

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            MaxDrawdownBreaker(max_dd=0)

    def test_status_keys(self):
        b = MaxDrawdownBreaker(max_dd=500)
        s = b.status()
        assert {"hwm", "pnl", "drawdown", "limit", "halted"} == set(s.keys())


class TestDailyLossBreaker:

    def test_allows_below_limit(self):
        b = DailyLossBreaker(daily_loss_limit=1000)
        b.update_pnl(-900)
        assert b.check().allowed

    def test_blocks_at_limit(self):
        b = DailyLossBreaker(daily_loss_limit=1000)
        b.update_pnl(-1000)
        result = b.check()
        assert not result.allowed
        assert "DailyLoss" in result.reason

    def test_daily_reset_clears_halt(self):
        b = DailyLossBreaker(daily_loss_limit=500)
        b.update_pnl(-600)
        b.check()
        b.daily_reset()
        assert b.check().allowed

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            DailyLossBreaker(daily_loss_limit=0)


class TestPositionLimitBreaker:

    def test_allows_within_limit(self):
        b = PositionLimitBreaker(max_position=1000)
        b.update_position("SPY", 500)
        assert b.check(_event("SPY", 400)).allowed

    def test_blocks_over_limit(self):
        b = PositionLimitBreaker(max_position=1000)
        b.update_position("SPY", 800)
        result = b.check(_event("SPY", 300))
        assert not result.allowed
        assert "PositionLimit" in result.reason

    def test_sell_reducing_position_allowed(self):
        b = PositionLimitBreaker(max_position=1000)
        b.update_position("SPY", 1000)
        assert b.check(_event("SPY", -200)).allowed

    def test_independent_symbols(self):
        b = PositionLimitBreaker(max_position=100)
        b.update_position("AAPL", 90)
        b.update_position("MSFT", 90)
        assert b.check(_event("AAPL", 5)).allowed
        assert b.check(_event("MSFT", 5)).allowed

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            PositionLimitBreaker(max_position=0)


class TestVelocityLimitBreaker:

    def test_allows_within_limit(self):
        b = VelocityLimitBreaker(max_trades=5, window_seconds=60)
        for _ in range(4):
            assert b.check(_event()).allowed

    def test_blocks_over_limit(self):
        b = VelocityLimitBreaker(max_trades=3, window_seconds=60)
        ts = time.time()
        for i in range(3):
            b.check(_event(ts=ts + i * 0.1))
        result = b.check(_event(ts=ts + 0.5))
        assert not result.allowed
        assert "VelocityLimit" in result.reason

    def test_window_expiry_allows_again(self):
        b  = VelocityLimitBreaker(max_trades=2, window_seconds=1.0)
        t0 = 1_000_000.0
        b.check(_event(ts=t0))
        b.check(_event(ts=t0 + 0.1))
        # Both in window; third blocked
        assert not b.check(_event(ts=t0 + 0.2)).allowed
        # After window expires
        assert b.check(_event(ts=t0 + 1.5)).allowed

    def test_reset_clears_history(self):
        b = VelocityLimitBreaker(max_trades=1, window_seconds=60)
        b.check(_event())
        b.reset()
        assert b.check(_event()).allowed


class TestNotionalLimitBreaker:

    def test_allows_below_limit(self):
        b = NotionalLimitBreaker(max_notional=50_000)
        assert b.check(_event(quantity=100, price=100.0)).allowed  # 10_000

    def test_blocks_over_limit(self):
        b = NotionalLimitBreaker(max_notional=5_000)
        result = b.check(_event(quantity=100, price=100.0))  # 10_000
        assert not result.allowed

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            NotionalLimitBreaker(max_notional=0)


class TestRiskGate:

    def test_all_pass(self):
        gate = RiskGate([
            PositionLimitBreaker(max_position=10_000),
            NotionalLimitBreaker(max_notional=100_000),
        ])
        assert gate.check(_event(quantity=100, price=50.0)).allowed

    def test_first_failure_returned(self):
        pos_b = PositionLimitBreaker(max_position=50)
        not_b = NotionalLimitBreaker(max_notional=1_000_000)
        gate  = RiskGate([pos_b, not_b])
        pos_b.update_position("SPY", 50)
        result = gate.check(_event("SPY", quantity=10, price=100.0))
        assert not result.allowed
        assert "PositionLimit" in result.reason

    def test_check_all_returns_all_results(self):
        gate = RiskGate([
            PositionLimitBreaker(max_position=10_000),
            NotionalLimitBreaker(max_notional=100_000),
        ])
        results = gate.check_all(_event(quantity=100, price=50.0))
        assert len(results) == 2

    def test_add_breaker_fluent(self):
        gate = RiskGate([]).add(NotionalLimitBreaker(1e6))
        assert gate.check(_event(quantity=1, price=1.0)).allowed

    def test_reset_delegates(self):
        gate = RiskGate([VelocityLimitBreaker(max_trades=1)])
        gate.check(_event())
        gate.reset()
        assert gate.check(_event()).allowed

    def test_repr(self):
        gate = RiskGate([NotionalLimitBreaker(1e6)])
        assert "RiskGate" in repr(gate)

    def test_status_list(self):
        gate = RiskGate([NotionalLimitBreaker(1e6)])
        s    = gate.status()
        assert isinstance(s, list) and len(s) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# PositionTracker Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionTrackerFIFO:

    def test_buy_creates_position(self):
        t = PositionTracker(method="fifo")
        t.fill("AAPL", 100, 150.0)
        assert t.position("AAPL") == 100.0

    def test_sell_reduces_position(self):
        t = PositionTracker(method="fifo")
        t.fill("AAPL", 100, 150.0)
        t.fill("AAPL", -40,  152.0)
        assert t.position("AAPL") == 60.0

    def test_flat_position_after_full_sell(self):
        t = PositionTracker(method="fifo")
        t.fill("AAPL", 100, 150.0)
        t.fill("AAPL", -100, 155.0)
        assert abs(t.position("AAPL")) < 1e-9

    def test_realised_pnl_fifo(self):
        t = PositionTracker(method="fifo")
        t.fill("AAPL", 100, 100.0)
        f = t.fill("AAPL", -100, 110.0)
        assert abs(f.realised_pnl - 1000.0) < 1e-9
        assert abs(t.realised_pnl("AAPL") - 1000.0) < 1e-9

    def test_partial_sell_realised_pnl(self):
        t = PositionTracker(method="fifo")
        t.fill("SPY", 200, 400.0)
        t.fill("SPY", -100, 410.0)
        assert abs(t.realised_pnl("SPY") - 1000.0) < 1e-9

    def test_fifo_lot_ordering(self):
        """FIFO: sell closes the oldest lot first."""
        t = PositionTracker(method="fifo")
        t.fill("X", 50, 10.0)   # lot 1: 50@10
        t.fill("X", 50, 20.0)   # lot 2: 50@20
        f = t.fill("X", -50, 25.0)  # closes lot 1 first: P&L = 50*(25-10)
        assert abs(f.realised_pnl - 750.0) < 1e-9

    def test_unrealised_pnl(self):
        t = PositionTracker(method="fifo")
        t.fill("MSFT", 100, 300.0)
        t.mark_to_market({"MSFT": 310.0})
        assert abs(t.unrealised_pnl("MSFT") - 1000.0) < 1e-9

    def test_total_pnl_is_sum(self):
        t = PositionTracker(method="fifo")
        t.fill("GOOG", 10, 2000.0)
        t.fill("GOOG", -5, 2100.0)       # realised = 500
        t.mark_to_market({"GOOG": 2050.0})  # unrealised = 5*(2050-2000) = 250
        assert abs(t.total_pnl("GOOG") - 750.0) < 1e-9

    def test_gross_exposure(self):
        t = PositionTracker(method="fifo")
        t.fill("A", 100, 50.0)
        t.fill("B", 200, 30.0)
        t.mark_to_market({"A": 50.0, "B": 30.0})
        assert abs(t.gross_exposure() - (100*50 + 200*30)) < 1e-9

    def test_multiple_symbols_independent(self):
        t = PositionTracker(method="fifo")
        t.fill("X", 100, 10.0)
        t.fill("Y", 200, 20.0)
        assert t.position("X") == 100.0
        assert t.position("Y") == 200.0

    def test_unknown_symbol_returns_zero(self):
        t = PositionTracker()
        assert t.position("UNKNOWN") == 0.0

    def test_blotter_records_fills(self):
        t = PositionTracker()
        t.fill("A", 100, 10.0)
        t.fill("A", -50, 12.0)
        assert len(t.blotter()) == 2

    def test_reset_clears_state(self):
        t = PositionTracker()
        t.fill("A", 100, 10.0)
        t.reset()
        assert t.position("A") == 0.0
        assert len(t.blotter()) == 0

    def test_repr(self):
        t = PositionTracker()
        assert "PositionTracker" in repr(t)


class TestPositionTrackerAvgCost:

    def test_avg_cost_updates_on_buy(self):
        t = PositionTracker(method="avg_cost")
        t.fill("A", 100, 100.0)
        t.fill("A", 100, 120.0)
        assert abs(t.avg_cost("A") - 110.0) < 1e-9

    def test_realised_pnl_avg_cost(self):
        t = PositionTracker(method="avg_cost")
        t.fill("A", 100, 100.0)
        t.fill("A", 100, 120.0)    # avg cost = 110
        f = t.fill("A", -100, 130.0)   # realised = 100*(130-110)
        assert abs(f.realised_pnl - 2000.0) < 1e-9

    def test_flat_after_full_close(self):
        t = PositionTracker(method="avg_cost")
        t.fill("B", 50, 200.0)
        t.fill("B", -50, 210.0)
        assert abs(t.position("B")) < 1e-9
        assert t.avg_cost("B") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PnL Reporter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPnLReporter:

    def _flat_returns(self, n=252, daily_ret=0.001):
        return np.full(n, daily_ret)

    def _random_returns(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        return rng.normal(0.0005, 0.01, n)

    def test_positive_sharpe_for_positive_mean(self):
        r      = self._flat_returns()
        report = PnLReporter(periods_per_year=252).report(r)
        assert report.sharpe > 0

    def test_zero_mean_zero_sharpe(self):
        rng = np.random.default_rng(0)
        r   = rng.normal(0.0, 0.01, 1000)
        report = PnLReporter().report(r)
        assert abs(report.sharpe) < 1.0    # roughly zero given noise

    def test_annualised_return_formula(self):
        r      = np.full(252, 0.001)
        report = PnLReporter(periods_per_year=252).report(r)
        expected = (1.001 ** 252) - 1.0
        assert abs(report.total_return - expected) < 1e-9

    def test_max_drawdown_zero_for_monotone_up(self):
        r      = np.full(100, 0.01)
        report = PnLReporter().report(r)
        assert report.max_drawdown < 1e-6

    def test_max_drawdown_nonzero_for_crash(self):
        r = np.concatenate([np.full(50, 0.01), np.full(50, -0.01)])
        report = PnLReporter().report(r)
        assert report.max_drawdown > 0

    def test_sortino_greater_than_sharpe_for_positive_skew(self):
        """Sortino > Sharpe when most returns are positive."""
        r = np.array([0.01] * 90 + [-0.001] * 10)
        report = PnLReporter().report(r)
        assert report.sortino >= report.sharpe - 1e-9

    def test_var_95_negative(self):
        r      = self._random_returns()
        report = PnLReporter().report(r)
        assert report.var_95 < 0       # 5th percentile of losses is negative

    def test_cvar_worse_than_var(self):
        r      = self._random_returns()
        report = PnLReporter().report(r)
        assert report.cvar_95 <= report.var_95 + 1e-9

    def test_win_rate_with_trade_pnls(self):
        trade_pnls = np.array([100, -50, 200, -30, 150])
        report = PnLReporter().report(np.zeros(5), trade_pnls=trade_pnls)
        assert abs(report.win_rate - 3/5) < 1e-9

    def test_profit_factor(self):
        trade_pnls = np.array([100.0, -50.0, 200.0, -50.0])
        report = PnLReporter().report(np.zeros(4), trade_pnls=trade_pnls)
        # Profit factor = 300 / 100 = 3.0
        assert abs(report.profit_factor - 3.0) < 1e-9

    def test_info_ratio_vs_identical_benchmark(self):
        """IR vs identical benchmark should be ~0."""
        r      = self._random_returns()
        report = PnLReporter().report(r, benchmark=r)
        assert abs(report.info_ratio) < 1e-9

    def test_nan_free_report(self):
        r      = self._random_returns()
        report = PnLReporter().report(r)
        assert np.isfinite(report.sharpe)
        assert np.isfinite(report.sortino)
        assert np.isfinite(report.max_drawdown)

    def test_summary_string(self):
        r      = self._flat_returns()
        report = PnLReporter().report(r)
        s      = report.summary()
        assert "Sharpe" in s
        assert "drawdown" in s.lower()

    def test_repr(self):
        r      = self._flat_returns()
        report = PnLReporter().report(r)
        assert "PnLReport" in repr(report)


class TestDrawdownHelpers:

    def test_monotone_up_zero_drawdown(self):
        wealth = np.cumprod(np.full(50, 1.01))
        dd     = _drawdown_series(wealth)
        assert np.all(dd < 1e-9)

    def test_crash_then_recovery(self):
        # 50% crash then full recovery
        wealth = np.array([1.0, 0.5, 0.5, 1.0, 1.0])
        dd     = _drawdown_series(wealth)
        assert abs(dd[1] - 0.5) < 1e-9
        assert abs(dd[-1]) < 1e-9


class TestVarCVar:

    def test_var_95_is_5th_percentile(self):
        rng = np.random.default_rng(0)
        r   = rng.normal(0, 0.01, 10_000)
        var, _ = _var_cvar(r, 0.95)
        assert abs(var - np.percentile(r, 5)) < 1e-12

    def test_cvar_worse_than_var(self):
        rng = np.random.default_rng(0)
        r   = rng.normal(0, 0.01, 10_000)
        var95, cvar95 = _var_cvar(r, 0.95)
        assert cvar95 <= var95


class TestRollingMetrics:

    def test_rolling_sharpe_length(self):
        r     = np.random.default_rng(0).normal(0.001, 0.01, 100)
        rs    = PnLReporter.rolling_sharpe(r, window=20)
        assert len(rs) == 100

    def test_rolling_sharpe_nan_before_window(self):
        r  = np.random.default_rng(0).normal(0.001, 0.01, 50)
        rs = PnLReporter.rolling_sharpe(r, window=20)
        assert np.all(np.isnan(rs[:19]))
        assert np.all(np.isfinite(rs[19:]))

    def test_rolling_drawdown_length(self):
        r   = np.random.default_rng(0).normal(0.001, 0.01, 100)
        rdd = PnLReporter.rolling_drawdown(r, window=20)
        assert len(rdd) == 100

    def test_rolling_drawdown_non_negative(self):
        r   = np.random.default_rng(0).normal(0.0, 0.01, 200)
        rdd = PnLReporter.rolling_drawdown(r, window=30)
        assert np.all(rdd[29:] >= 0)
