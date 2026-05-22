"""
Regime Panel
=============
Two-subplot panel driven by the Gaussian HMM in signals/hmm_regime.py.

  Top (65 %) : Mid-price time series.  Each bar is coloured by the
               Viterbi-decoded regime:
               Regime 0 (low mean / stress)  → red/orange
               Regime 1 (mid)                → yellow
               Regime K-1 (high / calm)      → green
               Regime boundaries are marked with vertical dashed lines.

  Bottom (35 %): Stacked area chart of posterior state probabilities
                 γ_t(k) from the forward-backward algorithm — shows
                 the smoothed regime confidence at each bar.

Public API
----------
build_regime_figure(
    mid_prices, returns=None, n_states=2, title=None
) -> go.Figure
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals.hmm_regime import GaussianHMM


# ── Colour scales ─────────────────────────────────────────────────────────────
# Maps regime index → display colour.  Regime 0 = bearish = red.
_REGIME_COLORS = [
    "rgba(239,  68,  68, 0.85)",   # red    — bearish / high-stress
    "rgba(251, 191,  36, 0.85)",   # amber  — neutral
    "rgba( 34, 197,  94, 0.85)",   # green  — bullish / low-stress
    "rgba(147, 197, 253, 0.85)",   # blue   — extra state if K>3
]
_REGIME_FILL = [
    "rgba(239,  68,  68, 0.12)",
    "rgba(251, 191,  36, 0.12)",
    "rgba( 34, 197,  94, 0.12)",
    "rgba(147, 197, 253, 0.12)",
]
_REGIME_NAMES = ["Bearish", "Neutral", "Bullish", "State 4"]
_BG    = "rgba(15, 23, 42, 0.00)"
_GRID  = "rgba(255, 255, 255, 0.06)"
_TEXT  = "#94a3b8"


def build_regime_figure(
    mid_prices: np.ndarray,
    returns:    Optional[np.ndarray] = None,
    n_states:   int                  = 2,
    title:      Optional[str]        = None,
) -> go.Figure:
    """
    Fit a Gaussian HMM and render the regime panel.

    Parameters
    ----------
    mid_prices : per-bar mid-price, shape (N,).
    returns    : per-bar return series, shape (N,) or (N-1,).
                 If None, derived from np.diff(log(mid_prices)).
    n_states   : number of HMM states (default 2).
    title      : panel title.

    Returns
    -------
    go.Figure
    """
    mp   = np.asarray(mid_prices, dtype=float)
    N    = len(mp)
    x    = np.arange(N)

    # ── Fit HMM ───────────────────────────────────────────────────────────────
    if returns is not None:
        r = np.asarray(returns, dtype=float)
        if len(r) == N - 1:
            # Pad a leading zero so r aligns with mid_prices
            r = np.concatenate([[0.0], r])
        r = r[:N]
    else:
        log_mp = np.log(np.maximum(mp, 1e-12))
        r = np.concatenate([[0.0], np.diff(log_mp)])

    r = np.where(np.isfinite(r), r, 0.0)

    n_states = min(n_states, max(2, N // 50))   # guard: need enough obs
    hmm      = GaussianHMM(n_states=n_states, seed=42)
    hmm_result = hmm.fit(r)
    states   = hmm.predict(r)       # Viterbi path (N,) int
    proba    = hmm.predict_proba(r) # posterior (N, K)

    K = hmm_result.n_states

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows             = 2,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = [0.65, 0.35],
        vertical_spacing = 0.06,
        subplot_titles   = (
            f"Mid-Price by Regime  [{K}-state HMM]",
            "Posterior Regime Probability  γ_t(k)",
        ),
    )

    # ── Row 1: mid-price segments coloured by regime ──────────────────────────
    # We draw one Scatter trace per regime, masking the other bars to NaN.
    for k in range(K):
        name  = _REGIME_NAMES[k] if k < len(_REGIME_NAMES) else f"State {k}"
        mu_k  = float(hmm_result.means[k])
        std_k = float(hmm_result.stds[k])

        mask = np.where(states == k, mp, np.nan)
        col  = _REGIME_COLORS[k % len(_REGIME_COLORS)]
        fill = _REGIME_FILL[k % len(_REGIME_FILL)]

        fig.add_trace(go.Scatter(
            name           = name,
            x              = x,
            y              = mask,
            mode           = "lines",
            connectgaps    = False,
            line           = dict(color=col, width=1.5),
            hovertemplate  = (
                f"<b>{name}</b>  μ={mu_k:.4f}, σ={std_k:.4f}<br>"
                "Bar %{x}<br>Mid: $%{y:.4f}<extra></extra>"
            ),
        ), row=1, col=1)

    # Regime transition vertical lines (only first 30 to avoid clutter)
    transitions = np.where(np.diff(states) != 0)[0] + 1
    for t_idx in transitions[:30]:
        fig.add_vline(
            x           = int(t_idx),
            line_color  = "rgba(255,255,255,0.12)",
            line_dash   = "dot",
            line_width  = 1,
            row=1, col=1,
        )

    # ── Row 2: stacked area of posterior probabilities ─────────────────────────
    # Plotly stacked area: each trace fills to the running cumulative.
    cum = np.zeros(N)
    for k in range(K):
        name  = _REGIME_NAMES[k] if k < len(_REGIME_NAMES) else f"State {k}"
        col   = _REGIME_COLORS[k % len(_REGIME_COLORS)]
        fill  = _REGIME_FILL[k % len(_REGIME_FILL)]
        lower = cum.copy()
        cum  += proba[:, k]
        upper = cum.copy()

        # Fill between lower and upper using a closed polygon trick
        x_closed = np.concatenate([x, x[::-1]])
        y_closed = np.concatenate([upper, lower[::-1]])

        fig.add_trace(go.Scatter(
            name          = name,
            x             = x_closed,
            y             = y_closed,
            mode          = "lines",
            fill          = "toself",
            fillcolor     = fill,
            line          = dict(color=col, width=0.8),
            hovertemplate = (
                f"<b>{name}</b>  γ_t=%{{customdata:.3f}}<br>"
                "Bar %{x}<extra></extra>"
            ),
            customdata    = np.concatenate([proba[:, k], proba[::-1, k]]),
            showlegend    = False,
        ), row=2, col=1)

    # ── HMM parameter annotation ──────────────────────────────────────────────
    hmm_text = "<br>".join(
        f"μ{k}={hmm_result.means[k]:.4f}  σ{k}={hmm_result.stds[k]:.4f}"
        for k in range(K)
    ) + f"<br>LL={hmm_result.log_likelihood:.1f}  iters={hmm_result.n_iter}"

    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text       = hmm_text,
        showarrow  = False,
        align      = "left",
        font       = dict(family="monospace", size=10, color=_TEXT),
        bgcolor    = "rgba(15,23,42,0.75)",
        bordercolor = "rgba(255,255,255,0.15)",
        borderwidth = 1,
        borderpad   = 5,
        xanchor     = "left",
        yanchor     = "top",
    )

    # ── Axes and layout ────────────────────────────────────────────────────────
    fig.update_xaxes(gridcolor=_GRID, color=_TEXT, tickfont=dict(color=_TEXT))
    fig.update_yaxes(gridcolor=_GRID, color=_TEXT, tickfont=dict(color=_TEXT))
    fig.update_yaxes(title_text="Mid ($)", tickformat=".2f",  row=1, col=1)
    fig.update_yaxes(title_text="P(regime)", tickformat=".2f",
                     range=[0, 1.0], row=2, col=1)
    fig.update_xaxes(title_text="Bar", row=2, col=1)

    fig.update_layout(
        title         = dict(
            text = title or "Market Regime (Gaussian HMM)",
            font = dict(size=13, color=_TEXT),
        ),
        paper_bgcolor = _BG,
        plot_bgcolor  = _BG,
        showlegend    = True,
        legend        = dict(
            font        = dict(color=_TEXT, size=11),
            bgcolor     = "rgba(0,0,0,0)",
            orientation = "h",
            y           = -0.07,
        ),
        margin        = dict(l=20, r=20, t=50, b=40),
        font          = dict(color=_TEXT),
    )

    return fig
