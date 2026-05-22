"""
LOB Depth Chart Panel
======================
Renders a symmetric "mountain chart" of the limit order book at a
single point in time.  Bids (green) fan out to the left; asks (red)
fan out to the right.  Each bar represents one price level; bar length
encodes the resting quantity.

Public API
----------
build_lob_depth_figure(snapshot, n_levels=5, title=None) -> go.Figure
    Return a Plotly Figure ready to embed in a Dash dcc.Graph.
"""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go

from backtester.engine import LOBSnapshot, DepthLevel


# ── Colour palette (matches Sigma Edge dark theme) ────────────────────────────
_BID_FILL   = "rgba( 34, 197,  94, 0.25)"   # green-400, 25 % opacity
_BID_LINE   = "rgba( 34, 197,  94, 0.90)"
_ASK_FILL   = "rgba(239,  68,  68, 0.25)"   # red-400, 25 % opacity
_ASK_LINE   = "rgba(239,  68,  68, 0.90)"
_MID_LINE   = "rgba(251, 191,  36, 0.80)"   # amber-400
_BG         = "rgba(15,  23,  42, 0.00)"    # transparent → parent bg shows
_GRID       = "rgba(255, 255, 255, 0.06)"
_TEXT       = "#94a3b8"                      # slate-400


def build_lob_depth_figure(
    snapshot: LOBSnapshot,
    n_levels: int            = 5,
    title:    Optional[str]  = None,
) -> go.Figure:
    """
    Build a LOB depth chart for a single snapshot.

    Parameters
    ----------
    snapshot : LOBSnapshot emitted by any DataSource.
    n_levels : number of price levels to show on each side (default 5).
    title    : figure title; defaults to "LOB Depth — {symbol} @{ts}".

    Returns
    -------
    go.Figure  (dark theme, transparent background)
    """
    bids = snapshot.bids[:n_levels]
    asks = snapshot.asks[:n_levels]
    mid  = snapshot.mid
    ts_s = snapshot.timestamp_ns / 1e9

    # ── Coordinate construction ───────────────────────────────────────────────
    # We plot price on the Y-axis and signed quantity on the X-axis.
    # Bids  → negative X (left branch)
    # Asks  → positive X (right branch)

    bid_prices = [d.price    for d in bids]
    bid_qty    = [-d.quantity for d in bids]   # negative → left
    ask_prices = [d.price    for d in asks]
    ask_qty    = [ d.quantity for d in asks]

    # Cumulative depth for step-fill area
    bid_cum = _cumulate(bids, negate=True)
    ask_cum = _cumulate(asks, negate=False)

    fig = go.Figure()

    # ── Bid bars ──────────────────────────────────────────────────────────────
    if bids:
        fig.add_trace(go.Bar(
            name          = "Bid",
            x             = bid_qty,
            y             = bid_prices,
            orientation   = "h",
            marker_color  = _BID_FILL,
            marker_line   = dict(color=_BID_LINE, width=1.2),
            hovertemplate = (
                "<b>BID</b><br>"
                "Price: $%{y:.4f}<br>"
                "Qty: %{customdata:,}<extra></extra>"
            ),
            customdata    = [d.quantity for d in bids],
        ))

    # ── Ask bars ──────────────────────────────────────────────────────────────
    if asks:
        fig.add_trace(go.Bar(
            name          = "Ask",
            x             = ask_qty,
            y             = ask_prices,
            orientation   = "h",
            marker_color  = _ASK_FILL,
            marker_line   = dict(color=_ASK_LINE, width=1.2),
            hovertemplate = (
                "<b>ASK</b><br>"
                "Price: $%{y:.4f}<br>"
                "Qty: %{customdata:,}<extra></extra>"
            ),
            customdata    = [d.quantity for d in asks],
        ))

    # ── Cumulative depth fill (step chart overlay) ────────────────────────────
    if bid_cum:
        cum_bid_px, cum_bid_qty = zip(*bid_cum)
        fig.add_trace(go.Scatter(
            name          = "Bid depth",
            x             = cum_bid_qty,
            y             = cum_bid_px,
            mode          = "lines",
            line          = dict(color=_BID_LINE, width=1.5, shape="hv"),
            fill          = "tozerox",
            fillcolor     = _BID_FILL,
            hoverinfo     = "skip",
        ))
    if ask_cum:
        cum_ask_px, cum_ask_qty = zip(*ask_cum)
        fig.add_trace(go.Scatter(
            name          = "Ask depth",
            x             = cum_ask_qty,
            y             = cum_ask_px,
            mode          = "lines",
            line          = dict(color=_ASK_LINE, width=1.5, shape="hv"),
            fill          = "tozerox",
            fillcolor     = _ASK_FILL,
            hoverinfo     = "skip",
        ))

    # ── Mid-price horizontal rule ─────────────────────────────────────────────
    if mid is not None:
        fig.add_hline(
            y            = mid,
            line_color   = _MID_LINE,
            line_dash    = "dash",
            line_width   = 1.5,
            annotation_text = f"Mid ${mid:.4f}",
            annotation_font_color = _MID_LINE,
            annotation_position   = "right",
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    spread = snapshot.spread_bps
    sp_str = f"{spread:.1f} bps" if spread is not None else "—"
    auto_title = (
        title or
        f"LOB Depth — {snapshot.symbol}  |  Spread {sp_str}"
    )

    all_qty = [abs(d.quantity) for d in bids + asks]
    max_qty = max(all_qty) * 1.15 if all_qty else 1

    fig.update_layout(
        title           = dict(text=auto_title, font=dict(size=13, color=_TEXT)),
        paper_bgcolor   = _BG,
        plot_bgcolor    = _BG,
        barmode         = "overlay",
        bargap          = 0.15,
        showlegend      = True,
        legend          = dict(
            font         = dict(color=_TEXT, size=11),
            bgcolor      = "rgba(0,0,0,0)",
            orientation  = "h",
            y            = -0.12,
        ),
        margin          = dict(l=20, r=20, t=40, b=40),
        xaxis           = dict(
            title         = "Resting Quantity",
            range         = [-max_qty, max_qty],
            tickformat    = ",d",
            zeroline      = True,
            zerolinecolor = "rgba(255,255,255,0.15)",
            gridcolor     = _GRID,
            color         = _TEXT,
            tickfont      = dict(color=_TEXT),
        ),
        yaxis           = dict(
            title         = "Price ($)",
            tickformat    = ".4f",
            gridcolor     = _GRID,
            color         = _TEXT,
            tickfont      = dict(color=_TEXT),
        ),
        font            = dict(color=_TEXT),
    )

    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cumulate(
    levels: List[DepthLevel],
    negate: bool,
) -> List[tuple]:
    """
    Convert a list of depth levels to (price, cumulative_qty) pairs for
    the step-fill area chart.  Asks accumulate positively, bids negatively.
    """
    result = []
    cum = 0
    for d in levels:
        cum += d.quantity
        result.append((d.price, -cum if negate else cum))
    return result
