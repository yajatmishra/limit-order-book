"""
Circuit Breakers / Risk Gates
==============================
Real-time risk checks that halt trading or reject orders when pre-defined
limits are breached.  Designed to be composed: a RiskGate chains multiple
breakers and returns a structured verdict on each order event.

Breaker types
-------------
  MaxDrawdown     : halt if peak-to-trough P&L drawdown exceeds threshold.
  DailyLoss       : halt if intra-day loss exceeds daily loss limit.
  PositionLimit   : reject if |position| would exceed max_position.
  VelocityLimit   : reject if trade count in a rolling window exceeds max_trades.
  NightlyReset    : clears daily counters (call at session start).

Each breaker implements the Breaker protocol:
  .check(event)  →  CheckResult(allowed, reason)
  .reset()       →  reset daily/session state
  .status()      →  dict of current state

The RiskGate composes N breakers; order is allowed only if all pass.

Reference:
  Hull (2018). "Risk Management and Financial Institutions." Wiley, Ch. 26.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


# ── Event and result types ────────────────────────────────────────────────────

@dataclass
class OrderEvent:
    """Minimal representation of an incoming order for risk checks."""
    symbol:    str
    quantity:  float    # signed (+= buy, -= sell)
    price:     float
    timestamp: float = field(default_factory=time.time)

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


@dataclass
class CheckResult:
    """Result of a circuit-breaker check."""
    allowed: bool
    reason:  str    # empty string if allowed; description if rejected

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:
        status = "OK" if self.allowed else f"BLOCKED({self.reason})"
        return f"CheckResult({status})"


# ── Individual breakers ───────────────────────────────────────────────────────

class MaxDrawdownBreaker:
    """
    Halt if running P&L drawdown from high-water mark exceeds `max_dd`.

    Parameters
    ----------
    max_dd : maximum allowed drawdown (absolute, in the same units as P&L).
             E.g. 10_000 halts if P&L drops more than $10k from its peak.
    """

    def __init__(self, max_dd: float) -> None:
        if max_dd <= 0:
            raise ValueError("max_dd must be > 0")
        self.max_dd  = max_dd
        self._hwm    = 0.0   # high-water mark
        self._pnl    = 0.0
        self._halted = False

    def update_pnl(self, pnl: float) -> None:
        """Push latest cumulative P&L value."""
        self._pnl = float(pnl)
        if self._pnl > self._hwm:
            self._hwm = self._pnl

    def check(self, event: Optional[OrderEvent] = None) -> CheckResult:
        if self._halted:
            return CheckResult(False, "MaxDrawdown: strategy halted (manual reset required)")
        dd = self._hwm - self._pnl
        if dd >= self.max_dd:
            self._halted = True
            return CheckResult(False,
                f"MaxDrawdown: drawdown={dd:.2f} >= limit={self.max_dd:.2f}")
        return CheckResult(True, "")

    def reset(self) -> None:
        """Full reset — use at start of new session."""
        self._hwm    = 0.0
        self._pnl    = 0.0
        self._halted = False

    def status(self) -> Dict[str, Any]:
        return {"hwm": self._hwm, "pnl": self._pnl,
                "drawdown": self._hwm - self._pnl,
                "limit": self.max_dd, "halted": self._halted}


class DailyLossBreaker:
    """
    Halt if intra-day cumulative P&L falls below `−daily_loss_limit`.

    Parameters
    ----------
    daily_loss_limit : absolute daily loss cap (positive).
                       E.g. 5_000 → halt if day P&L < −5000.
    """

    def __init__(self, daily_loss_limit: float) -> None:
        if daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be > 0")
        self.limit   = daily_loss_limit
        self._day_pnl = 0.0
        self._halted  = False

    def update_pnl(self, day_pnl: float) -> None:
        self._day_pnl = float(day_pnl)

    def check(self, event: Optional[OrderEvent] = None) -> CheckResult:
        if self._halted:
            return CheckResult(False, "DailyLoss: halted for today")
        if self._day_pnl <= -self.limit:
            self._halted = True
            return CheckResult(False,
                f"DailyLoss: day_pnl={self._day_pnl:.2f} <= -{self.limit:.2f}")
        return CheckResult(True, "")

    def daily_reset(self) -> None:
        """Call at session start to clear daily counters."""
        self._day_pnl = 0.0
        self._halted  = False

    reset = daily_reset   # alias

    def status(self) -> Dict[str, Any]:
        return {"day_pnl": self._day_pnl, "limit": -self.limit, "halted": self._halted}


class PositionLimitBreaker:
    """
    Reject an order if |current_position + order_quantity| > max_position.

    Parameters
    ----------
    max_position : maximum absolute position in units.
    """

    def __init__(self, max_position: float) -> None:
        if max_position <= 0:
            raise ValueError("max_position must be > 0")
        self.max_position = max_position
        self._positions: Dict[str, float] = {}

    def update_position(self, symbol: str, position: float) -> None:
        self._positions[symbol] = float(position)

    def check(self, event: OrderEvent) -> CheckResult:
        current = self._positions.get(event.symbol, 0.0)
        new_pos = current + event.quantity
        if abs(new_pos) > self.max_position:
            return CheckResult(False,
                f"PositionLimit: |{new_pos:.2f}| > {self.max_position:.2f} "
                f"for {event.symbol}")
        return CheckResult(True, "")

    def reset(self) -> None:
        self._positions.clear()

    def status(self) -> Dict[str, Any]:
        return {"positions": dict(self._positions), "limit": self.max_position}


class VelocityLimitBreaker:
    """
    Reject if the number of trades in the last `window_seconds` exceeds
    `max_trades`.  Implements a sliding-window token bucket.

    Parameters
    ----------
    max_trades     : maximum number of trades allowed in the window.
    window_seconds : rolling window duration (default 60 s).
    """

    def __init__(self, max_trades: int, window_seconds: float = 60.0) -> None:
        if max_trades < 1:
            raise ValueError("max_trades must be >= 1")
        self.max_trades = max_trades
        self.window     = window_seconds
        self._timestamps: deque = deque()

    def check(self, event: OrderEvent) -> CheckResult:
        now    = event.timestamp
        cutoff = now - self.window
        # Remove expired entries
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_trades:
            return CheckResult(False,
                f"VelocityLimit: {len(self._timestamps)} trades in "
                f"{self.window:.0f}s >= limit {self.max_trades}")

        self._timestamps.append(now)
        return CheckResult(True, "")

    def reset(self) -> None:
        self._timestamps.clear()

    def status(self) -> Dict[str, Any]:
        return {"recent_trades": len(self._timestamps),
                "limit": self.max_trades, "window_s": self.window}


class NotionalLimitBreaker:
    """
    Reject if a single order's notional exceeds `max_notional`.

    Parameters
    ----------
    max_notional : maximum single-order notional (price × quantity).
    """

    def __init__(self, max_notional: float) -> None:
        if max_notional <= 0:
            raise ValueError("max_notional must be > 0")
        self.max_notional = max_notional

    def check(self, event: OrderEvent) -> CheckResult:
        if event.notional > self.max_notional:
            return CheckResult(False,
                f"NotionalLimit: {event.notional:.2f} > {self.max_notional:.2f}")
        return CheckResult(True, "")

    def reset(self) -> None:
        pass

    def status(self) -> Dict[str, Any]:
        return {"limit": self.max_notional}


# ── RiskGate — composite circuit breaker ─────────────────────────────────────

class RiskGate:
    """
    Composes multiple circuit breakers into a single pre-trade risk check.

    An order is allowed only if all breakers pass.  The first failing breaker
    is reported in the result.

    Usage
    -----
    >>> gate = RiskGate([
    ...     MaxDrawdownBreaker(max_dd=10_000),
    ...     DailyLossBreaker(daily_loss_limit=5_000),
    ...     PositionLimitBreaker(max_position=50_000),
    ...     VelocityLimitBreaker(max_trades=100, window_seconds=60),
    ... ])
    >>> result = gate.check(order_event)
    >>> if result.allowed:
    ...     send_order(order_event)
    """

    def __init__(self, breakers: List) -> None:
        self._breakers = list(breakers)

    def check(self, event: OrderEvent) -> CheckResult:
        """Run all breakers; return first failure or OK."""
        for breaker in self._breakers:
            result = breaker.check(event)
            if not result.allowed:
                return result
        return CheckResult(True, "")

    def check_all(self, event: OrderEvent) -> List[CheckResult]:
        """Run all breakers and return all results (not short-circuit)."""
        return [b.check(event) for b in self._breakers]

    def reset(self) -> None:
        for b in self._breakers:
            b.reset()

    def daily_reset(self) -> None:
        """Reset daily counters (call at session start)."""
        for b in self._breakers:
            if hasattr(b, "daily_reset"):
                b.daily_reset()

    def status(self) -> List[Dict[str, Any]]:
        return [{"breaker": type(b).__name__, **b.status()} for b in self._breakers]

    def add(self, breaker) -> "RiskGate":
        """Append a breaker and return self (fluent interface)."""
        self._breakers.append(breaker)
        return self

    def __repr__(self) -> str:
        names = [type(b).__name__ for b in self._breakers]
        return f"RiskGate([{', '.join(names)}])"
