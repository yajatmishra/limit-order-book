"""
Backtester Portfolio
====================
Manages cash, positions, and P&L during a backtest.  Wraps the risk-package
PositionTracker to inherit FIFO/avg-cost accounting and adds:

  - Cash balance (reduced on buys, increased on sells)
  - Equity = cash + mark-to-market value of all positions
  - Equity-curve history and per-snapshot return series
  - Commission model (per-share or per-trade)
  - Margin / leverage enforcement (optional)

Fill protocol
-------------
The engine calls `portfolio.fill(fill)` with a Fill dataclass; the portfolio
updates positions, cash, and records the event.

`portfolio.mark(symbol, price)` updates mark-to-market prices between fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from risk.position_tracker import PositionTracker


# ── Fill (canonical trade record) ────────────────────────────────────────────

@dataclass
class Fill:
    """An executed trade.  Produced by the engine's fill simulator."""
    symbol:    str
    quantity:  float      # signed (+ buy, - sell)
    price:     float      # fill price per unit
    timestamp: int        # nanoseconds since midnight

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price

    @property
    def cash_impact(self) -> float:
        """Cash change from this fill (negative for buys, positive for sells)."""
        return -self.quantity * self.price

    def __repr__(self) -> str:
        side = "BUY" if self.quantity > 0 else "SELL"
        return (f"Fill({side} {abs(self.quantity):.0f} {self.symbol} "
                f"@ {self.price:.4f})")


# ── Commission models ─────────────────────────────────────────────────────────

class Commission:
    """Per-share commission: rate cents per share, min per trade."""
    def __init__(self, per_share: float = 0.005, min_per_trade: float = 1.0) -> None:
        self.per_share     = per_share
        self.min_per_trade = min_per_trade

    def compute(self, fill: Fill) -> float:
        return max(abs(fill.quantity) * self.per_share, self.min_per_trade)


class ZeroCommission(Commission):
    def compute(self, fill: Fill) -> float:
        return 0.0


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Strategy portfolio: tracks cash, positions, and equity during a backtest.

    Parameters
    ----------
    initial_cash : starting cash balance (default $1,000,000).
    commission   : Commission model (default $0.005/share, $1 minimum).
    max_leverage : maximum |gross_exposure| / equity ratio.
                   Orders exceeding this are rejected.  0 = unlimited.

    Usage
    -----
    >>> port = Portfolio(initial_cash=1_000_000)
    >>> port.fill(Fill("AAPL", 100, 150.0, ts))
    >>> port.mark("AAPL", 152.0)
    >>> print(port.equity())
    """

    def __init__(
        self,
        initial_cash: float      = 1_000_000.0,
        commission:   Commission  = None,
        max_leverage: float       = 0.0,
    ) -> None:
        self.initial_cash = initial_cash
        self._cash        = float(initial_cash)
        self.commission   = commission or Commission()
        self.max_leverage = max_leverage

        self.tracker   = PositionTracker(method="fifo")
        self._prices:  Dict[str, float] = {}
        self._fills:   List[Fill]       = []
        self._commissions: float        = 0.0

        # Equity curve snapshots (appended by engine after each market event)
        self._equity_history: List[float] = []

    # ── Fill processing ───────────────────────────────────────────────────────

    def fill(self, fill: Fill) -> bool:
        """
        Process a fill.  Returns False if rejected by leverage limit.

        Deducts commission from cash; updates PositionTracker.
        """
        if self.max_leverage > 0:
            new_pos  = self.tracker.position(fill.symbol) + fill.quantity
            new_exp  = abs(new_pos) * fill.price
            cur_gross = self.gross_exposure() + abs(fill.quantity) * fill.price
            eq = max(self.equity(), 1.0)
            if cur_gross / eq > self.max_leverage:
                return False   # reject

        comm  = self.commission.compute(fill)
        self._cash        -= fill.notional * (1 if fill.quantity > 0 else -1)
        self._cash        -= comm
        self._commissions += comm

        self.tracker.fill(fill.symbol, fill.quantity, fill.price)
        self._fills.append(fill)
        return True

    # ── Mark-to-market ────────────────────────────────────────────────────────

    def mark(self, symbol: str, price: float) -> None:
        """Update mark-to-market price for a symbol."""
        self._prices[symbol] = float(price)
        self.tracker.update_price(symbol, price)

    def mark_all(self, prices: Dict[str, float]) -> None:
        for sym, px in prices.items():
            self.mark(sym, px)

    # ── Queries ───────────────────────────────────────────────────────────────

    def cash(self) -> float:
        return self._cash

    def position(self, symbol: str) -> float:
        return self.tracker.position(symbol)

    def position_value(self, symbol: str) -> float:
        pos = self.tracker.position(symbol)
        px  = self._prices.get(symbol, 0.0)
        return pos * px

    def gross_exposure(self) -> float:
        return sum(abs(self.tracker.position(s)) * self._prices.get(s, 0.0)
                   for s in self.tracker.symbols())

    def net_exposure(self) -> float:
        return sum(self.tracker.position(s) * self._prices.get(s, 0.0)
                   for s in self.tracker.symbols())

    def equity(self) -> float:
        """Cash + mark-to-market value of all open positions.

        _cash already reflects every fill's notional (buys reduce cash, sells
        increase cash), so equity = cash + current market value of open positions.
        There is no need to add realised P&L separately — it is already in cash.
        """
        position_value = sum(
            self.tracker.position(s) * self._prices.get(s, 0.0)
            for s in self.tracker.symbols()
        )
        return self._cash + position_value

    def total_pnl(self) -> float:
        return self.equity() - self.initial_cash

    def total_commissions(self) -> float:
        return self._commissions

    def leverage(self) -> float:
        eq = self.equity()
        return self.gross_exposure() / max(abs(eq), 1.0)

    def snapshot_equity(self) -> None:
        """Append current equity to history (called by the engine)."""
        self._equity_history.append(self.equity())

    def equity_curve(self) -> np.ndarray:
        return np.array(self._equity_history, dtype=float)

    def returns(self) -> np.ndarray:
        eq = self.equity_curve()
        if len(eq) < 2:
            return np.array([], dtype=float)
        return np.diff(eq) / np.maximum(eq[:-1], 1e-12)

    def fills(self) -> List[Fill]:
        return list(self._fills)

    def reset(self) -> None:
        """Reset to initial state (useful for parameter sweeps)."""
        self._cash        = self.initial_cash
        self._commissions = 0.0
        self.tracker.reset()
        self._prices.clear()
        self._fills.clear()
        self._equity_history.clear()

    def __repr__(self) -> str:
        return (f"Portfolio(equity={self.equity():.2f}, "
                f"cash={self._cash:.2f}, "
                f"positions={len(self.tracker.symbols())})")
