"""
Strategy Comparison Panel
=========================
Builds a single Plotly figure comparing several strategies run on the same
synthetic market: an overlaid equity-curve chart on top and a ranked metrics
table below.

Public API
----------
build_strategy_comparison_figure(comparison, title=None) -> go.Figure
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow running this file directly (python dashboard/strategy_panel.py) by putting
# the python/ package root on the path before importing sibling packages.
_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from backtester.strategy_lab import ComparisonResult  # noqa: E402

# ── Palette (matches the dashboard theme) ───────────────────────────────────────
_BG      = "rgba(15, 23, 42, 0.00)"
_GRID    = "rgba(255, 255, 255, 0.06)"
_TEXT    = "#94a3b8"
_TEXT_HI = "#f1f5f9"
_AMBER   = "#fbbf24"
_ZERO    = "rgba(255, 255, 255, 0.20)"
_PANEL   = "#1e293b"
_BORDER  = "#334155"


def build_strategy_comparison_figure(
    comparison: ComparisonResult,
    title:      Optional[str] = None,
) -> go.Figure:
    """Overlay each strategy's equity curve and tabulate its key metrics."""
    runs = comparison.runs

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.66, 0.34],
        vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Cumulative return (%)", None),
    )

    # ── Row 1: equity curves as percent return from the starting equity ──────────
    for run in runs:
        eq = np.asarray(run.result.equity_curve, dtype=float)
        if eq.size == 0 or eq[0] == 0:
            continue
        pct = (eq / eq[0] - 1.0) * 100.0
        x = np.arange(eq.size)
        fig.add_trace(
            go.Scatter(
                x=x, y=pct, mode="lines", name=run.name,
                line=dict(color=run.color, width=2.2),
                hovertemplate=f"<b>{run.name}</b><br>tick %{{x}}<br>%{{y:.2f}}%<extra></extra>",
            ),
            row=1, col=1,
        )
    fig.add_hline(y=0, line=dict(color=_ZERO, width=1, dash="dot"), row=1, col=1)

    # ── Row 2: ranked metrics table (best Sharpe first) ──────────────────────────
    ordered = sorted(runs, key=lambda r: r.tearsheet.report.sharpe, reverse=True)
    names, rets, sharpe, sortino, maxdd, fills = [], [], [], [], [], []
    name_colors = []
    for run in ordered:
        r = run.tearsheet.report
        names.append(run.name)
        name_colors.append(run.color)
        rets.append(f"{r.total_return:+.1%}")
        sharpe.append(f"{r.sharpe:.2f}")
        sortino.append(f"{r.sortino:.2f}")
        maxdd.append(f"{r.max_drawdown:.2%}")
        fills.append(f"{run.tearsheet.n_fills:,}")

    fig.add_trace(
        go.Table(
            columnwidth=[2.2, 1.4, 1.2, 1.2, 1.2, 1.0],
            header=dict(
                values=["<b>Strategy</b>", "<b>Total Return</b>", "<b>Sharpe</b>",
                        "<b>Sortino</b>", "<b>Max DD</b>", "<b>Fills</b>"],
                fill_color=_PANEL,
                font=dict(color=_TEXT_HI, size=12),
                align="left", height=28,
                line_color=_BORDER,
            ),
            cells=dict(
                values=[names, rets, sharpe, sortino, maxdd, fills],
                fill_color=[name_colors,
                            ["rgba(0,0,0,0)"] * len(names)],
                font=dict(
                    color=[["#0f172a"] * len(names)] + [[_TEXT_HI] * len(names)] * 5,
                    size=12,
                ),
                align="left", height=26,
                line_color=_BORDER,
            ),
        ),
        row=2, col=1,
    )

    # ── Layout ───────────────────────────────────────────────────────────────────
    fig.update_xaxes(title_text="snapshot (tick)", showgrid=True, gridcolor=_GRID,
                     zeroline=False, color=_TEXT, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False,
                     color=_TEXT, ticksuffix="%", row=1, col=1)
    fig.update_layout(
        title=dict(
            text=title or "Strategy Comparison  ·  synthetic market (seed 28)",
            font=dict(color=_AMBER, size=16, family="monospace"),
            x=0.01, xanchor="left",
        ),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="Inter, system-ui, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
                    font=dict(color=_TEXT_HI, size=11)),
        margin=dict(l=50, r=24, t=70, b=40),
        hovermode="x unified",
    )
    # subplot title styling
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT, size=12)
    return fig


if __name__ == "__main__":
    # Regenerate the static comparison image used in the README.
    # Requires kaleido:  pip install kaleido
    import os

    from backtester.strategy_lab import run_comparison

    comp = run_comparison()
    figure = build_strategy_comparison_figure(comp)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "images")
    figure.write_image(os.path.join(out_dir, "strategy_comparison.png"),
                       width=1100, height=760, scale=2)
    figure.write_html(os.path.join(out_dir, "strategy_comparison.html"),
                      include_plotlyjs="cdn")
    print("wrote docs/images/strategy_comparison.png and .html")
