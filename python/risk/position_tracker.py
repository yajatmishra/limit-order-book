"""
Position Tracker
================
Maintains real-time position state, mark-to-market P&L, and exposure for
a portfolio of instruments.

Accounting methods
------------------
  FIFO (First In, First Out) — default
    When reducing a position, the oldest lots are closed first.  Realised P&L
    is computed against the cost basis of the oldest open lot.

  Average cost
    A single average cost basis per symbol is maintained.  Each new buy updates
    the average; sells realise P&L vs. the running average.

Features
--------
  - Per-symbol open position, average cost, realised P&L
  - Portfolio-level mark-to-market (unrealised) P&L
  - Net and gross exposure
  - Trade blotter (full fill history)
  - P&L decomposition: realised + unrealised = total P&L

Usage
-----
>>> tracker = PositionTracker(method="fifo")
>>> tracker.fill(symbol="AAPL", quantity=100, price=150.0)   # buy
>>> tracker.fill(symbol="AAPL", quantity=-50, price=152.0)   # partial sell
>>> tracker.mark_to_market({"AAPL": 153.0})
>>> pnl = tracker.total_pnl()
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import DefaultDict, Deque, Dict, List, Optional, Tuple


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Lot:
    """A single FIFO cost-basis lot."""
    quantity:  float
    cost:      float   # price paid per unit

    @property
    def notional(self) -> float:
        return self.quantity * self.cost


@dataclass
class Fill:
    """One trade record."""
    symbol:    str
    quantity:  float   # signed
    price:     float
    realised_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


@dataclass
class SymbolState:
    """Real-time position state for one symbol."""
    symbol:        str
    net_position:  float = 0.0
    avg_cost:      float = 0.0
    realised_pnl:  float = 0.0
    last_price:    float = 0.0
    lots:          List[Lot] = field(default_factory=list)   # FIFO queue

    @property
    def unrealised_pnl(self) -> float:
        if abs(self.net_position) < 1e-12:
            return 0.0
        return self.net_position * (self.last_price - self.avg_cost)

    @property
    def total_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl

    @property
    def notional(self) -> float:
        return abs(self.net_position) * self.last_price

    def __repr__(self) -> str:
        return (f"SymbolState({self.symbol}, pos={self.net_position:.2f}, "
                f"avg_cost={self.avg_cost:.4f}, "
                f"unrealised={self.unrealised_pnl:.2f}, "
                f"realised={self.realised_pnl:.2f})")


# ── PositionTracker ───────────────────────────────────────────────────────────

class PositionTracker:
    """
    Real-time position and P&L tracker.

    Parameters
    ----------
    method : "fifo" (default) or "avg_cost" — lot-matching convention.

    Usage
    -----
    >>> tracker = PositionTracker()
    >>> tracker.fill("SPY", 500, 450.0)       # buy 500 @ 450
    >>> tracker.fill("SPY", -200, 452.0)      # sell 200 @ 452
    >>> tracker.mark_to_market({"SPY": 451.0})
    >>> print(tracker.total_pnl())
    """

    def __init__(self, method: str = "fifo") -> None:
        if method not in ("fifo", "avg_cost"):
            raise ValueError("method must be 'fifo' or 'avg_cost'")
        self.method = method
        self._state:  Dict[str, SymbolState] = {}
        self._blotter: List[Fill] = []

    # ── Fill recording ────────────────────────────────────────────────────────

    def fill(
        self,
        symbol:   str,
        quantity: float,
        price:    float,
    ) -> Fill:
        """
        Record a fill (trade execution).

        Parameters
        ----------
        symbol   : instrument identifier.
        quantity : signed quantity (+= buy, -= sell).
        price    : execution price per unit.

        Returns
        -------
        Fill with realised P&L (non-zero only for position-reducing trades).
        """
        if abs(quantity) < 1e-12:
            return Fill(symbol=symbol, quantity=0.0, price=price)

        if symbol not in self._state:
            self._state[symbol] = SymbolState(symbol=symbol)

        st = self._state[symbol]

        if self.method == "fifo":
            rpnl = self._fill_fifo(st, quantity, price)
        else:
            rpnl = self._fill_avg_cost(st, quantity, price)

        f = Fill(symbol=symbol, quantity=quantity, price=price, realised_pnl=rpnl)
        self._blotter.append(f)
        return f

    def _fill_fifo(self, st: SymbolState, quantity: float, price: float) -> float:
        """FIFO lot matching.  Returns realised P&L."""
        rpnl = 0.0
        if quantity > 0:                 # buy: add a new lot
            st.lots.append(Lot(quantity=quantity, cost=price))
            st.net_position += quantity
        else:                            # sell: consume oldest lots
            to_fill = -quantity          # positive amount to consume
            while to_fill > 1e-9 and st.lots:
                lot = st.lots[0]
                if lot.quantity <= to_fill:
                    rpnl   += lot.quantity * (price - lot.cost)
                    to_fill -= lot.quantity
                    st.net_position -= lot.quantity
                    st.lots.pop(0)
                else:
                    rpnl   += to_fill * (price - lot.cost)
                    lot.quantity -= to_fill
                    st.net_position -= to_fill
                    to_fill = 0.0
            # Short-selling: any residual creates a short lot
            if to_fill > 1e-9:
                st.lots.append(Lot(quantity=-to_fill, cost=price))
                st.net_position -= to_fill

        # Recompute avg_cost from lots
        total_qty = sum(lot.quantity for lot in st.lots)
        if abs(total_qty) > 1e-12:
            st.avg_cost = (sum(lot.quantity * lot.cost for lot in st.lots)
                           / total_qty)
        else:
            st.avg_cost = 0.0

        st.realised_pnl += rpnl
        return rpnl

    def _fill_avg_cost(self, st: SymbolState, quantity: float, price: float) -> float:
        """Average-cost lot matching.  Returns realised P&L."""
        rpnl = 0.0
        prev_pos = st.net_position

        if quantity > 0:                 # buy
            new_pos  = prev_pos + quantity
            if prev_pos >= 0:            # flat or long → grow position
                if abs(new_pos) > 1e-12:
                    st.avg_cost = ((prev_pos * st.avg_cost + quantity * price)
                                   / new_pos)
            else:                        # covering a short
                if quantity <= -prev_pos:
                    rpnl = quantity * (st.avg_cost - price)
                else:
                    rpnl = (-prev_pos) * (st.avg_cost - price)
                    # Residual goes long at new price
                    residual = quantity + prev_pos
                    st.avg_cost = price if residual > 0 else 0.0
        else:                            # sell
            if prev_pos > 0:             # reducing long
                reduce = min(-quantity, prev_pos)
                rpnl   = reduce * (price - st.avg_cost)
                if -quantity > prev_pos:
                    # Going short with residual
                    st.avg_cost = price
            else:                        # going more short
                new_pos = prev_pos + quantity
                if abs(new_pos) > 1e-12:
                    st.avg_cost = ((prev_pos * st.avg_cost + quantity * price)
                                   / new_pos)

        st.net_position += quantity
        if abs(st.net_position) < 1e-9:
            st.avg_cost = 0.0
        st.realised_pnl += rpnl
        return rpnl

    # ── Mark-to-market ────────────────────────────────────────────────────────

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        """Update last prices for all or some symbols."""
        for sym, px in prices.items():
            if sym in self._state:
                self._state[sym].last_price = float(px)

    def update_price(self, symbol: str, price: float) -> None:
        """Update the mark price for a single symbol."""
        if symbol in self._state:
            self._state[symbol].last_price = float(price)

    # ── Queries ───────────────────────────────────────────────────────────────

    def position(self, symbol: str) -> float:
        """Current net position for a symbol (0 if not tracked)."""
        return self._state[symbol].net_position if symbol in self._state else 0.0

    def avg_cost(self, symbol: str) -> float:
        return self._state[symbol].avg_cost if symbol in self._state else 0.0

    def unrealised_pnl(self, symbol: Optional[str] = None) -> float:
        if symbol is not None:
            return self._state[symbol].unrealised_pnl if symbol in self._state else 0.0
        return sum(s.unrealised_pnl for s in self._state.values())

    def realised_pnl(self, symbol: Optional[str] = None) -> float:
        if symbol is not None:
            return self._state[symbol].realised_pnl if symbol in self._state else 0.0
        return sum(s.realised_pnl for s in self._state.values())

    def total_pnl(self, symbol: Optional[str] = None) -> float:
        return self.realised_pnl(symbol) + self.unrealised_pnl(symbol)

    def gross_exposure(self) -> float:
        """Sum of |position| * last_price across all symbols."""
        return sum(s.notional for s in self._state.values())

    def net_exposure(self) -> float:
        """Net exposure: Σ position * last_price (signed)."""
        return sum(s.net_position * s.last_price for s in self._state.values())

    def symbols(self) -> List[str]:
        return list(self._state.keys())

    def state(self, symbol: str) -> Optional[SymbolState]:
        return self._state.get(symbol)

    def all_states(self) -> Dict[str, SymbolState]:
        return dict(self._state)

    def blotter(self) -> List[Fill]:
        return list(self._blotter)

    def reset(self) -> None:
        self._state.clear()
        self._blotter.clear()

    def __repr__(self) -> str:
        n   = len(self._state)
        pnl = self.total_pnl()
        return f"PositionTracker(symbols={n}, total_pnl={pnl:.2f}, method={self.method!r})"
