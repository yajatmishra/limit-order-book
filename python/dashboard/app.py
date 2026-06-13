"""
Limit Order Book — Plotly Dash Dashboard
==========================================
Entry point.  Run with:

    cd limit-order-book/python
    python dashboard/app.py            # development server on :8050
    python dashboard/app.py --port 8080 --debug

Layout
------
  ┌──────────────────────────────────────────────────┐
  │  LIMIT ORDER BOOK  ·  Session Replay Dashboard    │
  │  [symbol] [equity] [sharpe] [maxdd] [fills] ···  │
  ├────────────────────┬─────────────────────────────┤
  │  LOB Depth Chart   │  P&L Panel                   │
  │  (slider → tick)   │  (equity + returns + DD)     │
  ├────────────────────┼─────────────────────────────┤
  │  OFI Panel         │  Regime Panel                │
  │  (rolling OFI)     │  (HMM + posterior prob)      │
  └────────────────────┴─────────────────────────────┘

Data flow
---------
  1. generate_session() builds a synthetic ITCH replay at startup.
  2. Static figures (P&L, OFI, Regime) are built once and stored in dcc.Store.
  3. The LOB depth chart is updated on every slider tick via a Dash callback.
  4. A "Play" interval auto-advances the slider for animation mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY   = os.path.join(_HERE, "..")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

from dashboard.replay_generator import generate_session
from dashboard.lob_depth_chart  import build_lob_depth_figure
from dashboard.ofi_panel        import build_ofi_figure
from dashboard.pnl_panel        import build_pnl_figure
from dashboard.regime_panel     import build_regime_figure


# ═══════════════════════════════════════════════════════════════════════════════
# Constants & theme
# ═══════════════════════════════════════════════════════════════════════════════

_DARK_BG    = "#0f172a"   # slate-900
_PANEL_BG   = "#1e293b"   # slate-800
_BORDER     = "#334155"   # slate-700
_TEXT_MAIN  = "#f1f5f9"   # slate-100
_TEXT_MUT   = "#94a3b8"   # slate-400
_AMBER      = "#fbbf24"   # amber-400
_GREEN      = "#22c55e"   # green-500
_RED        = "#ef4444"   # red-500
_BLUE       = "#93c5fd"   # blue-300

_PANEL_STYLE = {
    "backgroundColor": _PANEL_BG,
    "border":          f"1px solid {_BORDER}",
    "borderRadius":    "8px",
    "padding":         "8px",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation  (runs once at startup)
# ═══════════════════════════════════════════════════════════════════════════════

print("⟳  Generating synthetic session replay …", flush=True)
_t0 = time.time()
SESSION = generate_session(n_snaps=2_000, seed=42)
print(f"✓  Session ready in {time.time() - _t0:.1f}s  "
      f"({len(SESSION.snapshots):,} snapshots, "
      f"{SESSION.result.snapshots_processed:,} processed, "
      f"{SESSION.tearsheet.n_fills} fills)", flush=True)

# Build static figures (expensive, so done once)
print("⟳  Building static figures …", flush=True)
FIG_PNL    = build_pnl_figure(SESSION.result, SESSION.tearsheet)
FIG_OFI    = build_ofi_figure(SESSION.ofi_series, SESSION.mid_prices, window=30)
FIG_REGIME = build_regime_figure(SESSION.mid_prices,
                                  SESSION.result.returns, n_states=2)
# Initial LOB depth at tick 0
FIG_LOB_0  = build_lob_depth_figure(SESSION.snapshots[0], n_levels=5)
print("✓  Figures ready.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# KPI cards
# ═══════════════════════════════════════════════════════════════════════════════

def _kpi(label: str, value: str, colour: str = _TEXT_MAIN) -> html.Div:
    return html.Div([
        html.Span(label, style={"fontSize": "10px", "color": _TEXT_MUT,
                                "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.Div(value,  style={"fontSize": "16px", "color": colour,
                                "fontWeight": "700", "fontFamily": "monospace"}),
    ], className="kpi-cell", style={"padding": "0 14px", "borderRight": f"1px solid {_BORDER}"})


def _build_kpi_bar() -> html.Div:
    t = SESSION.tearsheet
    r = t.report

    def pct(v):  return f"{v:+.2%}" if np.isfinite(v) else "N/A"
    def flt(v):  return f"{v:.3f}"  if np.isfinite(v) else "N/A"

    pnl       = t.final_equity - t.initial_equity
    pnl_color = _GREEN if pnl >= 0 else _RED
    sr_color  = _GREEN if r.sharpe >= 1 else (_AMBER if r.sharpe >= 0 else _RED)

    return html.Div([
        html.Div([
            html.Span(className="pulse"),
            "LIMIT ORDER BOOK",
        ], className="brand", style={
            "fontSize": "18px", "fontWeight": "900", "color": _AMBER,
            "fontFamily": "monospace", "letterSpacing": "0.12em",
            "paddingRight": "20px", "display": "flex", "alignItems": "center",
        }),
        _kpi("Symbol",     SESSION.result.symbol),
        _kpi("Total P&L",  f"${pnl:+,.0f}",       colour=pnl_color),
        _kpi("Ann. Ret",   pct(r.ann_return),       colour=pnl_color),
        _kpi("Ann. Vol",   pct(r.ann_volatility)),
        _kpi("Sharpe",     flt(r.sharpe),            colour=sr_color),
        _kpi("Sortino",    flt(r.sortino),           colour=sr_color),
        _kpi("Max DD",     pct(r.max_drawdown),      colour=_RED),
        _kpi("PSR",        flt(t.psr),               colour=_AMBER),
        _kpi("Fills",      str(t.n_fills)),
        _kpi("Bars",       f"{len(SESSION.snapshots):,}"),
    ], className="kpi-bar", style={
        "display":         "flex",
        "alignItems":      "center",
        "backgroundColor": _PANEL_BG,
        "border":          f"1px solid {_BORDER}",
        "borderRadius":    "8px",
        "padding":         "10px 12px",
        "marginBottom":    "10px",
        "overflowX":       "auto",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# App layout
# ═══════════════════════════════════════════════════════════════════════════════

_DESCRIPTION = ("Interactive limit order book session replay — live depth, "
                "order-flow imbalance, P&L and HMM regime panels.")

app = dash.Dash(
    __name__,
    title = "Limit Order Book: Session Replay",
    update_title = None,   # no "Updating..." title flicker on callbacks
    meta_tags = [
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": _DESCRIPTION},
        {"name": "theme-color", "content": _DARK_BG},
        # Open Graph — nicer link previews when the deployed URL is shared
        {"property": "og:title", "content": "Limit Order Book: Session Replay"},
        {"property": "og:description", "content": _DESCRIPTION},
        {"property": "og:type", "content": "website"},
    ],
)
server = app.server   # expose Flask for gunicorn

N_SNAPS = len(SESSION.snapshots)


# ═══════════════════════════════════════════════════════════════════════════════
# Layout helpers  (must be defined before app.layout)
# ═══════════════════════════════════════════════════════════════════════════════

def _btn_style(active: bool = False) -> dict:
    return {
        "backgroundColor": (_AMBER if active else _PANEL_BG),
        "color":           (_DARK_BG if active else _TEXT_MAIN),
        "border":          f"1px solid {_BORDER}",
        "borderRadius":    "5px",
        "padding":         "4px 10px",
        "fontSize":        "12px",
        "cursor":          "pointer",
        "fontWeight":      "600",
    }


def _slider_marks(n: int) -> dict:
    step = max(1, n // 8)
    return {i: {"label": str(i), "style": {"color": _TEXT_MUT, "fontSize": "10px"}}
            for i in range(0, n, step)}


app.layout = html.Div([

    # ── Header KPI bar ────────────────────────────────────────────────────────
    _build_kpi_bar(),

    # ── LOB animation controls ────────────────────────────────────────────────
    html.Div([
        html.Span("LOB Snapshot", style={"color": _TEXT_MUT, "fontSize": "12px",
                                          "marginRight": "10px"}),
        html.Button("◀◀", id="btn-prev", n_clicks=0,
                    style=_btn_style()),
        html.Button("▶ Play", id="btn-play", n_clicks=0,
                    style=_btn_style(active=True)),
        html.Button("▶▶", id="btn-next", n_clicks=0,
                    style=_btn_style()),
        dcc.Slider(
            id    = "lob-slider",
            min   = 0,
            max   = N_SNAPS - 1,
            value = 0,
            step  = 1,
            marks = _slider_marks(N_SNAPS),
            tooltip = {"placement": "bottom", "always_visible": False},
            updatemode = "drag",
        ),
        dcc.Interval(id="play-interval", interval=200, disabled=True),
        dcc.Store(id="play-state", data={"playing": False}),
    ], className="controls-bar", style={
        "display":         "flex",
        "alignItems":      "center",
        "gap":             "8px",
        "backgroundColor": _PANEL_BG,
        "border":          f"1px solid {_BORDER}",
        "borderRadius":    "8px",
        "padding":         "8px 14px",
        "marginBottom":    "10px",
    }),

    # ── 2 × 2 panel grid (grid template defined in assets/dashboard.css so the
    #    mobile breakpoint can collapse it to a single column) ─────────────────
    html.Div([
        # Top-left: LOB depth (dynamic) — wrapped in a loading spinner
        html.Div([
            dcc.Loading(
                type="default", color=_AMBER,
                children=dcc.Graph(
                    id="graph-lob",
                    figure=FIG_LOB_0,
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "100%"}),
                style={"height": "100%"},
            ),
        ], style={**_PANEL_STYLE, "gridArea": "lob"}),

        # Top-right: P&L (static)
        html.Div([
            dcc.Graph(id="graph-pnl",
                      figure=FIG_PNL,
                      config={"displayModeBar": True, "responsive": True,
                               "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                      style={"height": "100%"}),
        ], style={**_PANEL_STYLE, "gridArea": "pnl"}),

        # Bottom-left: OFI (static)
        html.Div([
            dcc.Graph(id="graph-ofi",
                      figure=FIG_OFI,
                      config={"displayModeBar": False, "responsive": True},
                      style={"height": "100%"}),
        ], style={**_PANEL_STYLE, "gridArea": "ofi"}),

        # Bottom-right: Regime (static)
        html.Div([
            dcc.Graph(id="graph-regime",
                      figure=FIG_REGIME,
                      config={"displayModeBar": False, "responsive": True},
                      style={"height": "100%"}),
        ], style={**_PANEL_STYLE, "gridArea": "regime"}),

    ], className="panel-grid"),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Div([
        html.Span("Synthetic ITCH replay · 2,000 snapshots · seed = 42 · "
                  "offline demo — no live market data"),
        html.Span([
            html.A("GitHub", href="https://github.com/yajatmishra/limit-order-book",
                   target="_blank"),
            " · ",
            html.A("Methodology",
                   href="https://github.com/yajatmishra/limit-order-book/blob/main/METHODOLOGY.md",
                   target="_blank"),
        ]),
    ], className="app-footer"),

], style={
    "backgroundColor": _DARK_BG,
    "minHeight":       "100vh",
    "padding":         "12px 16px",
    "fontFamily":      "Inter, system-ui, sans-serif",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

# ── LOB depth chart — updates on slider move ──────────────────────────────────
@app.callback(
    Output("graph-lob", "figure"),
    Input("lob-slider", "value"),
    prevent_initial_call=True,
)
def update_lob(tick: int) -> go.Figure:
    tick = max(0, min(tick, N_SNAPS - 1))
    snap = SESSION.snapshots[tick]
    return build_lob_depth_figure(snap, n_levels=5)


# ── Play / pause — toggles interval + button label ────────────────────────────
@app.callback(
    Output("play-interval", "disabled"),
    Output("btn-play",      "children"),
    Output("play-state",    "data"),
    Input("btn-play",       "n_clicks"),
    State("play-state",     "data"),
    prevent_initial_call=True,
)
def toggle_play(n: int, state: dict):
    playing = not state.get("playing", False)
    label   = "⏸ Pause" if playing else "▶ Play"
    return not playing, label, {"playing": playing}


# ── Auto-advance slider while playing ─────────────────────────────────────────
@app.callback(
    Output("lob-slider", "value"),
    Input("play-interval", "n_intervals"),
    Input("btn-prev",      "n_clicks"),
    Input("btn-next",      "n_clicks"),
    State("lob-slider",    "value"),
    prevent_initial_call=True,
)
def advance_slider(
    _intervals: int,
    _prev:      int,
    _next:      int,
    current:    int,
) -> int:
    ctx = callback_context
    if not ctx.triggered:
        return current
    trigger = ctx.triggered[0]["prop_id"]
    if "btn-prev" in trigger:
        return max(0, current - 1)
    if "btn-next" in trigger:
        return min(N_SNAPS - 1, current + 1)
    # play-interval
    return (current + 5) % N_SNAPS


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limit Order Book Dashboard")
    parser.add_argument("--port",  type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--host",  default="0.0.0.0")
    args = parser.parse_args()

    print(f"\n🚀  Dashboard starting on http://localhost:{args.port}/\n")
    app.run(
        host      = args.host,
        port      = args.port,
        debug     = args.debug,
        use_reloader = False,   # disable reloader so session is built once
    )
