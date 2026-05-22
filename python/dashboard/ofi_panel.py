"""
Order-Flow Imbalance (OFI) Panel
==================================
Two-subplot panel:

  Top    : Rolling OFI (window W ticks) as a filled area chart.
           Positive OFI (net buying pressure) is coloured green;
           negative OFI (net selling pressure) is coloured red.
           A thin black zero-line and ±1-sigma bands are overlaid.

  Bottom : Normalised OFI vs mid-price change scatter (β estimate).
           Each point is one tick; the OLS regression line is shown.

Public API
----------
build_ofi_figure(ofi_series, mid_prices=None, window=30, title=None) -> go.Figure
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Palette ────────────────────────────────────────────────────────────────────
_GREEN_FILL  = "rgba( 34, 197,  94, 0.20)"
_GREEN_LINE  = "rgba( 34, 197,  94, 0.85)"
_RED_FILL    = "rgba(239,  68,  68, 0.20)"
_RED_LINE    = "rgba(239,  68,  68, 0.85)"
_BLUE_LINE   = "rgba(147, 197, 253, 0.80)"   # regression line
_ZERO_LINE   = "rgba(255, 255, 255, 0.25)"
_SIGMA_BAND  = "rgba(255, 255, 255, 0.06)"
_BG          = "rgba(15,  23,  42, 0.00)"
_GRID        = "rgba(255, 255, 255, 0.06)"
_TEXT        = "#94a3b8"


def build_ofi_figure(
    ofi_series:  np.ndarray,
    mid_prices:  Optional[np.ndarray] = None,
    window:      int                   = 30,
    title:       Optional[str]         = None,
) -> go.Figure:
    """
    Build the two-subplot OFI panel.

    Parameters
    ----------
    ofi_series  : raw per-tick OFI series, shape (M,).
    mid_prices  : corresponding mid-price series, shape (M+1,) or (M,).
                  If provided, the bottom subplot shows ΔMid vs OFI.
    window      : rolling window length (ticks) for the top chart.
    title       : panel title.

    Returns
    -------
    go.Figure with two subplots.
    """
    ofi = np.asarray(ofi_series, dtype=float)
    M   = len(ofi)
    x   = np.arange(M)

    # Rolling OFI (causal — no look-ahead)
    roll_ofi = _rolling_mean(ofi, window)
    sigma    = float(np.nanstd(roll_ofi))

    # ── Mid-price changes for scatter (if available) ──────────────────────────
    have_mid = mid_prices is not None and len(mid_prices) > 1
    if have_mid:
        mp = np.asarray(mid_prices, dtype=float)
        # align: ofi[t] = f(book[t-1], book[t]) → compare with Δmid[t]
        n_delta = min(M, len(mp) - 1)
        delta_mid = np.diff(mp[:n_delta + 1])   # shape (n_delta,)
        ofi_reg   = ofi[:n_delta]
        beta, r2  = _ols(ofi_reg, delta_mid)
    else:
        delta_mid = np.array([])
        ofi_reg   = np.array([])
        beta, r2  = 0.0, 0.0

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_rows = 2 if have_mid else 1
    row_heights = [0.65, 0.35] if have_mid else [1.0]
    fig = make_subplots(
        rows               = n_rows,
        cols               = 1,
        shared_xaxes       = False,
        row_heights        = row_heights,
        vertical_spacing   = 0.10,
        subplot_titles     = (
            [f"Rolling OFI (window={window})",
             f"ΔMid vs OFI  [β={beta:.4f}, R²={r2:.3f}]"]
            if have_mid else
            [f"Rolling OFI (window={window})"]
        ),
    )

    # ── Top subplot: rolling OFI ──────────────────────────────────────────────
    # Split into positive and negative sections
    pos_ofi = np.where(roll_ofi > 0, roll_ofi, 0.0)
    neg_ofi = np.where(roll_ofi < 0, roll_ofi, 0.0)

    fig.add_trace(go.Scatter(
        name       = "OFI⁺ (buy pressure)",
        x          = x, y = pos_ofi,
        mode       = "lines",
        fill       = "tozeroy",
        fillcolor  = _GREEN_FILL,
        line       = dict(color=_GREEN_LINE, width=1.0),
        hovertemplate = "Tick %{x}<br>OFI=%{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name       = "OFI⁻ (sell pressure)",
        x          = x, y = neg_ofi,
        mode       = "lines",
        fill       = "tozeroy",
        fillcolor  = _RED_FILL,
        line       = dict(color=_RED_LINE, width=1.0),
        hovertemplate = "Tick %{x}<br>OFI=%{y:.1f}<extra></extra>",
    ), row=1, col=1)

    # ±1-sigma bands
    if sigma > 0:
        for sign, name in [(+1, "+1σ"), (-1, "−1σ")]:
            fig.add_hline(
                y            = sign * sigma,
                line_dash    = "dot",
                line_color   = _SIGMA_BAND,
                line_width   = 1,
                annotation_text      = name,
                annotation_font_color = _TEXT,
                annotation_font_size  = 10,
                annotation_position   = "right",
                row=1, col=1,
            )

    # Zero line
    fig.add_hline(
        y          = 0,
        line_color = _ZERO_LINE,
        line_width = 1.2,
        row=1, col=1,
    )

    # ── Bottom subplot: scatter + OLS ─────────────────────────────────────────
    if have_mid and len(ofi_reg) > 0:
        fig.add_trace(go.Scatter(
            name   = "Tick",
            x      = ofi_reg,
            y      = delta_mid,
            mode   = "markers",
            marker = dict(
                size    = 3,
                color   = ofi_reg,
                colorscale = "RdYlGn",
                opacity = 0.55,
                showscale = False,
            ),
            hovertemplate = "OFI=%{x:.0f}<br>ΔMid=%{y:.5f}<extra></extra>",
        ), row=2, col=1)

        # OLS regression line
        x_fit = np.linspace(float(ofi_reg.min()), float(ofi_reg.max()), 60)
        y_fit = beta * x_fit
        fig.add_trace(go.Scatter(
            name      = f"OLS (β={beta:.4f})",
            x         = x_fit,
            y         = y_fit,
            mode      = "lines",
            line      = dict(color=_BLUE_LINE, width=2, dash="dash"),
            hoverinfo = "skip",
        ), row=2, col=1)

        fig.update_xaxes(title_text="OFI (shares)", row=2, col=1,
                         gridcolor=_GRID, color=_TEXT, tickfont=dict(color=_TEXT))
        fig.update_yaxes(title_text="ΔMid ($)", row=2, col=1,
                         gridcolor=_GRID, color=_TEXT, tickfont=dict(color=_TEXT),
                         tickformat=".5f")

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_xaxes(
        title_text = "Tick",
        gridcolor  = _GRID,
        color      = _TEXT,
        tickfont   = dict(color=_TEXT),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text = "OFI (shares)",
        gridcolor  = _GRID,
        color      = _TEXT,
        tickfont   = dict(color=_TEXT),
        row=1, col=1,
    )

    fig.update_layout(
        title          = dict(
            text = title or "Order-Flow Imbalance",
            font = dict(size=13, color=_TEXT),
        ),
        paper_bgcolor  = _BG,
        plot_bgcolor   = _BG,
        showlegend     = True,
        legend         = dict(
            font        = dict(color=_TEXT, size=10),
            bgcolor     = "rgba(0,0,0,0)",
            orientation = "h",
            y           = -0.08,
        ),
        margin         = dict(l=20, r=20, t=50, b=40),
        font           = dict(color=_TEXT),
    )

    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean; first (window-1) entries are NaN."""
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = arr[i - window + 1 : i + 1].mean()
    return out


def _ols(x: np.ndarray, y: np.ndarray):
    """Return (beta, R²) from OLS y ~ beta*x (no intercept)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 0.0, 0.0
    beta   = float(np.dot(x, y) / max(np.dot(x, x), 1e-30))
    y_hat  = beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2
