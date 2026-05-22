"""
backtester
==========
Event-driven backtesting framework with ITCH 5.0 / SHM data sources,
market-order fill simulation, portfolio accounting, and tearsheet generation.

Public API
----------
  Data sources
    DataSource          – abstract base
    ShmDataSource       – live POSIX shared-memory reader (C++ ShmWriter)
    SnapshotSource      – replay a list of LOBSnapshot objects
    ItchReplayer        – pure-Python ITCH 5.0 binary parser

  Data types
    DepthLevel          – one price level (price, quantity, order_count)
    LOBSnapshot         – point-in-time LOB state
    Order               – order submitted by a strategy

  Strategy
    Strategy            – abstract base (implement on_snapshot)

  Engine
    BacktestEngine      – wires DataSource → Strategy → Portfolio
    EngineResult        – aggregate backtest output

  Portfolio
    Fill                – canonical executed trade record
    Commission          – per-share commission model
    ZeroCommission      – no-cost commission model
    Portfolio           – cash + positions + equity curve

  Tearsheet
    Tearsheet           – full performance tearsheet (text / dict output)

  Helpers
    build_itch_add      – build synthetic ITCH 'A' message bytes
    build_itch_delete   – build synthetic ITCH 'D' message bytes
    build_itch_execute  – build synthetic ITCH 'E' message bytes
"""

from backtester.engine import (
    BacktestEngine,
    DataSource,
    DepthLevel,
    EngineResult,
    ItchReplayer,
    LOBSnapshot,
    Order,
    ShmDataSource,
    SnapshotSource,
    Strategy,
    build_itch_add,
    build_itch_delete,
    build_itch_execute,
)
from backtester.portfolio import (
    Commission,
    Fill,
    Portfolio,
    ZeroCommission,
)
from backtester.tearsheet import Tearsheet

__all__ = [
    # engine
    "BacktestEngine",
    "DataSource",
    "DepthLevel",
    "EngineResult",
    "ItchReplayer",
    "LOBSnapshot",
    "Order",
    "ShmDataSource",
    "SnapshotSource",
    "Strategy",
    "build_itch_add",
    "build_itch_delete",
    "build_itch_execute",
    # portfolio
    "Commission",
    "Fill",
    "Portfolio",
    "ZeroCommission",
    # tearsheet
    "Tearsheet",
]
