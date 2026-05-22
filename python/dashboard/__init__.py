"""
Limit Order Book – Dashboard package
================================
Plotly Dash dashboard for visualising a live or replayed trading session.

Modules
-------
  replay_generator  : Synthetic ITCH session replay → EngineResult + Tearsheet
  lob_depth_chart   : LOB depth (mountain) chart panel
  ofi_panel         : Order-Flow Imbalance time-series panel
  pnl_panel         : Equity-curve / drawdown panel with tearsheet metrics
  regime_panel      : HMM regime overlay on mid-price
  app               : Dash application entry point  (run with `python app.py`)
"""

from .replay_generator import generate_session
from .lob_depth_chart  import build_lob_depth_figure
from .ofi_panel        import build_ofi_figure
from .pnl_panel        import build_pnl_figure
from .regime_panel     import build_regime_figure

__all__ = [
    "generate_session",
    "build_lob_depth_figure",
    "build_ofi_figure",
    "build_pnl_figure",
    "build_regime_figure",
]
