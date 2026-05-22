"""
P&L Panel
==========
Three-row panel:

  Row 1 (60 %) : Equity curve with fill-below for drawdown periods
                 highlighted in semi-transparent red.
  Row 2 (25 %) : Per-period returns as a bar chart (green / red).
  Row 3 (15 %) : Running drawdown as a red area chart.

A metrics annotation box in the top-right corner shows:
  Sharpe | Sortino | Calmar | MaxDD | Ann. Return | PSR | DSR

Public API
----------
build_pnl_figure(result, tearsheet, title=None) -> go.Figure
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.engine    import EngineResult
from backtester.tearsheet import Tearsheet


# ── Palette ────────────────────────────────────────────────────────────────────
_EQUITY_LINE = "rgba(147, 197, 253, 0.90)"  # blue-300
_EQUITY_FILL = "rgba(147, 197, 253, 0.10)"
_DD_FILL     = "rgba(239,  68,  68, 0.25)"
_DD_LINE     = "rgba(239,  68,  68, 0.80)"
_RET_GREEN   = "rgba( 34, 197,  94, 0.75)"
_RET_RED     = "rgba(239,  68,  68, 0.75)"
_ZERO        = "rgba(255, 255, 255, 0.20)"
_BG          = "rgba(15,  23,  42, 0.00)"
_GRID        = "rgba(255, 255, 255, 0.06)"
_TEXT        = "#94a3b8"
_AMBER       = "#fbbf24"


def build_pnl_figure(
    result:    EngineResult,
    tearsheet: Tearsheet,
    title:     Optional[str] = None,
) -> go.Figure:
    """
    Build the three-row P&L panel.

    Parameters
    ----------
    result    : EngineResult from BacktestEngine.run().
    tearsheet : Tearsheet computed from result.
    title     : panel title.

    Returns
    -------
    go.Figure
    """
    eq  = result.equity_curve
    ret = result.returns
    ts  = np.arange(len(eq))
    r   = tearsheet.report

    # Drawdown series from equity curve
    dd  = _drawdown(eq)

    fig = make_subplots(
        rows             = 3,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = [0.55, 0.25, 0.20],
        vertical_spacing = 0.04,
        subplot_titles   = ("Equity Curve", "Period Returns", "Drawdown"),
    )

    # ── Row 1: equity curve ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        name           = "Equity",
        x              = ts,
        y              = eq,
        mode           = "lines",
        line           = dict(color=_EQUITY_LINE, width=1.8),
        fill           = "tozeroy",
        fillcolor      = _EQUITY_FILL,
        hovertemplate  = "Bar %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Initial equity reference line
    fig.add_hline(
        y             = tearsheet.initial_equity,
        line_color    = _ZERO,
        line_dash     = "dot",
        line_width    = 1,
        row=1, col=1,
    )

    # ── Row 2: period returns ──────────────────────────────────────────────────
    if len(ret) > 0:
        ret_ts    = ts[1:]   # returns are diff(equity), aligned to bar[1:]
        ret_color = [_RET_GREEN if v >= 0 else _RET_RED for v in ret]
        fig.add_trace(go.Bar(
            name          = "Return",
            x             = ret_ts,
            y             = ret,
            marker_color  = ret_color,
            hovertemplate = "Bar %{x}<br>Return: %{y:.4%}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color=_ZERO, line_width=0.8, row=2, col=1)

    # ── Row 3: drawdown ────────────────────────────────────────────────────────
    if len(dd) > 0:
        fig.add_trace(go.Scatter(
            name           = "Drawdown",
            x              = ts,
            y              = -dd,    # negative so it fills downward
            mode           = "lines",
            fill           = "tozeroy",
            fillcolor      = _DD_FILL,
            line           = dict(color=_DD_LINE, width=1.2),
            hovertemplate  = "Bar %{x}<br>DD: %{customdata:.2%}<extra></extra>",
            customdata     = dd,
        ), row=3, col=1)
        fig.add_hline(y=0, line_color=_ZERO, line_width=0.8, row=3, col=1)

    # ── Metrics annotation ─────────────────────────────────────────────────────
    metrics = _metrics_text(r, tearsheet)
    fig.add_annotation(
        x           = 0.99,
        y           = 0.99,
        xref        = "paper",
        yref        = "paper",
        text        = metrics,
        showarrow   = False,
        align       = "left",
        font        = dict(family="monospace", size=11, color=_AMBER),
        bgcolor     = "rgba(15,23,42,0.75)",
        bordercolor = _AMBER,
        borderwidth = 1,
        borderpad   = 6,
        xanchor     = "right",
        yanchor     = "top",
    )

    # ── Axis labels ───────────────────────────────────────────────────────────
    _style_axis(fig, row=1, ytitle="Equity ($)", ytickfmt="$,.0f")
    _style_axis(fig, row=2, ytitle="Return",     ytickfmt=".2%")
    _style_axis(fig, row=3, ytitle="Drawdown",   ytickfmt=".1%", xtitle="Bar")

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        title         = dict(
            text = title or "Strategy P&L",
            font = dict(size=13, color=_TEXT),
        ),
        paper_bgcolor = _BG,
        plot_bgcolor  = _BG,
        showlegend    = False,
        margin        = dict(l=20, r=20, t=50, b=30),
        font          = dict(color=_TEXT),
    )

    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _drawdown(equity: np.ndarray) -> np.ndarray:
    """Peak-to-trough drawdown series as positive fractions."""
    if len(equity) == 0:
        return np.array([])
    hwm = np.maximum.accumulate(equity)
    return (hwm - equity) / np.maximum(hwm, 1e-12)


def _metrics_text(r, t: Tearsheet) -> str:
    """Format key metrics as an HTML table for the annotation."""
    def _pct(v): return f"{v:+.2%}" if np.isfinite(v) else "N/A"
    def _f(v, fmt=".3f"): return format(v, fmt) if np.isfinite(v) else "N/A"

    total_pnl = t.final_equity - t.initial_equity
    pnl_str   = f"${total_pnl:+,.0f}"

    lines = [
        f"Ann. Return : {_pct(r.ann_return)}",
        f"Ann. Vol    : {_pct(r.ann_volatility)}",
        f"Sharpe      : {_f(r.sharpe)}",
        f"Sortino     : {_f(r.sortino)}",
        f"Calmar      : {_f(r.calmar)}",
        f"Max DD      : {_pct(r.max_drawdown)}",
        f"PSR         : {_f(t.psr)}",
        f"DSR         : {_f(t.dsr)}",
        f"Total P&L   : {pnl_str}",
        f"Fills       : {t.n_fills}",
    ]
    return "<br>".join(lines)


def _style_axis(
    fig,
    row:      int,
    xtitle:   str = "",
    ytitle:   str = "",
    ytickfmt: str = "",
) -> None:
    fig.update_xaxes(
        title_text = xtitle,
        gridcolor  = _GRID,
        color      = _TEXT,
        tickfont   = dict(color=_TEXT),
        row        = row, col=1,
    )
    fig.update_yaxes(
        title_text  = ytitle,
        tickformat  = ytickfmt,
        gridcolor   = _GRID,
        color       = _TEXT,
        tickfont    = dict(color=_TEXT),
        row         = row, col=1,
    )
