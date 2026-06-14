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


def _mean_sharpe_by_name(seeds: range, n_snaps: int = 800) -> dict:
    acc: dict = {}
    for seed in seeds:
        for run in run_comparison(n_snaps=n_snaps, seed=seed).runs:
            acc.setdefault(run.name, []).append(run.tearsheet.report.sharpe)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def test_momentum_strategies_have_realistic_edge() -> None:
    # Averaged over out-of-sample seeds, the trend and order-flow strategies have
    # a positive, believable Sharpe. The upper bound guards against a regression
    # to frequency-inflated annualization (the Sharpe-of-20 artifact).
    mean = _mean_sharpe_by_name(range(200, 206))
    assert 0.5 < mean["MA Crossover"] < 5.0
    assert 0.5 < mean["OFI Momentum"] < 5.0
    assert all(abs(s) < 8.0 for s in mean.values()), "Sharpe magnitudes must stay believable"


def test_mean_reversion_loses_on_trending_asset() -> None:
    # Honest result, not a bug: fading a trending, momentum-driven asset reliably
    # loses, and it cannot coexist with a profitable momentum strategy at the same
    # horizon. Forcing it positive would require look-ahead or curve-fitting.
    mean = _mean_sharpe_by_name(range(200, 206))
    assert mean["Mean Reversion"] < 0.0


def test_run_comparison_is_deterministic() -> None:
    a = run_comparison(n_snaps=600, seed=5)
    b = run_comparison(n_snaps=600, seed=5)
    for ra, rb in zip(a.runs, b.runs):
        assert ra.result.equity_curve[-1] == rb.result.equity_curve[-1]
