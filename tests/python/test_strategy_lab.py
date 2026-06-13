"""Tests for the strategy comparison lab."""

from __future__ import annotations

import numpy as np

from backtester.strategy_lab import generate_market, run_comparison


def test_market_has_expected_shape() -> None:
    market = generate_market(n_snaps=500, seed=1)
    assert len(market.snapshots) == 500
    assert market.mids.shape == (500,)
    assert market.imbalance.shape == (500,)
    assert np.all(market.imbalance >= -1.0) and np.all(market.imbalance <= 1.0)
    # every snapshot has five levels per side
    assert all(len(s.bids) == 5 and len(s.asks) == 5 for s in market.snapshots)


def test_run_comparison_returns_four_strategies() -> None:
    comp = run_comparison(n_snaps=1_000, seed=28)
    names = [r.name for r in comp.runs]
    assert names == ["Buy & Hold", "MA Crossover", "Mean Reversion", "OFI Momentum"]
    for run in comp.runs:
        assert run.result.equity_curve.size == 1_000
        assert np.isfinite(run.tearsheet.report.sharpe)


def test_active_strategies_are_profitable_on_default_seed() -> None:
    # On the structured synthetic market, every active strategy should beat zero.
    comp = run_comparison(seed=28)
    by_name = {r.name: r.tearsheet.report for r in comp.runs}
    for name in ("MA Crossover", "Mean Reversion", "OFI Momentum"):
        assert by_name[name].total_return > 0.0, f"{name} should be profitable"
        assert by_name[name].sharpe > 0.0, f"{name} Sharpe should be positive"


def test_run_comparison_is_deterministic() -> None:
    a = run_comparison(n_snaps=600, seed=5)
    b = run_comparison(n_snaps=600, seed=5)
    for ra, rb in zip(a.runs, b.runs):
        assert ra.result.equity_curve[-1] == rb.result.equity_curve[-1]
