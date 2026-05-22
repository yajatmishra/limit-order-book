"""
Transaction Cost Analysis (TCA)
================================
Estimates execution costs for a trade schedule and evaluates slippage.

Components
----------
1. Half-spread cost
   Crossing the bid-ask spread costs ½·spread per trade (one-way).
   cost_spread = ½ · spread · |trade_size|

2. Market impact (Almgren square-root law, 2005)
   Temporary impact of a trade of size Q in a stock with ADV (average
   daily volume) and σ (daily volatility):

     impact = η · σ · (|Q| / ADV)^(1/2)

   where η is a market-impact coefficient (default 0.1 from empirical
   calibrations on US equities).

3. Slippage
   Realised slippage = execution_price − arrival_price (for a buy).
   Negative slippage (favourable fills) can occur in passive strategies.

References
----------
  Almgren et al. (2005). "Direct estimation of equity market impact."
  Risk Magazine, 18(7), 58-62.

  Kissell (2014). "The Science of Algorithmic Trading and Portfolio
  Management." Academic Press.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class TradeCost:
    """Cost breakdown for a single trade."""
    spread_cost:  float   # half-spread component
    impact_cost:  float   # market impact component
    slippage:     float   # realised price slippage (signed)
    total_cost:   float   # spread + impact + |slippage| (if measured)

    @property
    def total_bps(self) -> float:
        """Total cost in basis points (requires price context)."""
        return self.total_cost * 1e4

    def __repr__(self) -> str:
        return (f"TradeCost(spread={self.spread_cost:.4f}, "
                f"impact={self.impact_cost:.4f}, "
                f"slippage={self.slippage:.4f}, "
                f"total={self.total_cost:.4f})")


@dataclass
class TCAResult:
    """Aggregate TCA across a sequence of trades."""
    n_trades:          int
    total_notional:    float   # sum of |trade_size| * price
    total_spread_cost: float
    total_impact_cost: float
    total_slippage:    float
    total_cost:        float
    avg_cost_bps:      float   # total_cost / total_notional * 1e4
    trade_costs:       list    # list[TradeCost]

    def __repr__(self) -> str:
        return (f"TCAResult(n={self.n_trades}, "
                f"avg_cost_bps={self.avg_cost_bps:.2f}, "
                f"total_cost={self.total_cost:.4f})")


# ── TCA engine ────────────────────────────────────────────────────────────────

class TCA:
    """
    Transaction Cost Analysis engine.

    Parameters
    ----------
    half_spread    : bid-ask half-spread as a fraction of price (default 5 bps).
    eta            : market impact coefficient η in the square-root law.
                     Default 0.1 (Almgren et al. 2005 calibration).
    daily_vol      : daily return volatility used for impact scaling.
                     If None, must be provided per-trade via compute().
    adv            : average daily volume (shares or notional).
                     If None, must be provided per-trade.

    Usage
    -----
    >>> tca = TCA(half_spread=5e-4, eta=0.1, daily_vol=0.02, adv=1e6)
    >>> cost = tca.compute(trade_size=10_000, arrival_price=100.0)
    >>> result = tca.analyse(trade_sizes, arrival_prices, exec_prices)
    """

    def __init__(
        self,
        half_spread: float = 5e-4,
        eta:         float = 0.1,
        daily_vol:   Optional[float] = None,
        adv:         Optional[float] = None,
    ) -> None:
        if half_spread < 0:
            raise ValueError("half_spread must be >= 0")
        if eta < 0:
            raise ValueError("eta must be >= 0")
        self.half_spread = half_spread
        self.eta         = eta
        self.daily_vol   = daily_vol
        self.adv         = adv

    # ── Single trade ──────────────────────────────────────────────────────────

    def compute(
        self,
        trade_size:    float,
        arrival_price: float,
        exec_price:    Optional[float] = None,
        daily_vol:     Optional[float] = None,
        adv:           Optional[float] = None,
    ) -> TradeCost:
        """
        Compute execution cost for one trade.

        Parameters
        ----------
        trade_size    : signed trade quantity (+ = buy, − = sell).
        arrival_price : mid-price at time of order arrival.
        exec_price    : actual execution price (optional; if provided,
                        slippage is computed directly; otherwise 0).
        daily_vol     : per-trade override for daily vol.
        adv           : per-trade override for average daily volume.

        Returns
        -------
        TradeCost breakdown.
        """
        q    = float(trade_size)
        p    = float(arrival_price)
        vol  = daily_vol  if daily_vol  is not None else self.daily_vol
        adv_ = adv        if adv        is not None else self.adv

        # Half-spread cost
        spread_cost = self.half_spread * abs(q) * p

        # Market impact
        if vol is not None and adv_ is not None and adv_ > 0:
            participation = abs(q) / float(adv_)
            impact_cost   = float(self.eta) * float(vol) * np.sqrt(participation) * abs(q) * p
        else:
            impact_cost = 0.0

        # Slippage (signed: positive = adverse, negative = favourable)
        if exec_price is not None:
            direction = 1.0 if q >= 0 else -1.0
            slippage  = direction * (float(exec_price) - p) * abs(q)
        else:
            slippage = 0.0

        total = spread_cost + impact_cost + abs(slippage)

        return TradeCost(
            spread_cost = spread_cost,
            impact_cost = impact_cost,
            slippage    = slippage,
            total_cost  = total,
        )

    # ── Batch analysis ────────────────────────────────────────────────────────

    def analyse(
        self,
        trade_sizes:    np.ndarray,
        arrival_prices: np.ndarray,
        exec_prices:    Optional[np.ndarray] = None,
        daily_vols:     Optional[np.ndarray] = None,
        advs:           Optional[np.ndarray] = None,
    ) -> TCAResult:
        """
        Analyse a sequence of trades.

        Parameters
        ----------
        trade_sizes    : shape (N,) signed trade quantities.
        arrival_prices : shape (N,) mid-prices at order arrival.
        exec_prices    : shape (N,) actual execution prices (optional).
        daily_vols     : shape (N,) per-trade daily vols (optional).
        advs           : shape (N,) per-trade ADV (optional).

        Returns
        -------
        TCAResult with aggregate statistics.
        """
        sizes  = np.asarray(trade_sizes,    dtype=float)
        prices = np.asarray(arrival_prices, dtype=float)
        N      = len(sizes)

        if len(prices) != N:
            raise ValueError("trade_sizes and arrival_prices must have equal length")

        ep  = np.asarray(exec_prices,  dtype=float) if exec_prices  is not None else [None] * N
        dv  = np.asarray(daily_vols,   dtype=float) if daily_vols   is not None else [None] * N
        av  = np.asarray(advs,         dtype=float) if advs          is not None else [None] * N

        costs       = []
        total_notional = 0.0

        for i in range(N):
            c = self.compute(
                trade_size    = sizes[i],
                arrival_price = prices[i],
                exec_price    = ep[i]  if ep[i]  is not None else None,
                daily_vol     = dv[i]  if dv[i]  is not None else None,
                adv           = av[i]  if av[i]  is not None else None,
            )
            costs.append(c)
            total_notional += abs(sizes[i]) * prices[i]

        total_spread = sum(c.spread_cost for c in costs)
        total_impact = sum(c.impact_cost for c in costs)
        total_slip   = sum(c.slippage    for c in costs)
        total_cost   = sum(c.total_cost  for c in costs)

        avg_bps = (total_cost / total_notional * 1e4) if total_notional > 0 else 0.0

        return TCAResult(
            n_trades          = N,
            total_notional    = total_notional,
            total_spread_cost = total_spread,
            total_impact_cost = total_impact,
            total_slippage    = total_slip,
            total_cost        = total_cost,
            avg_cost_bps      = avg_bps,
            trade_costs       = costs,
        )

    # ── Impact model alone ───────────────────────────────────────────────────

    @staticmethod
    def market_impact_bps(
        participation: float,
        daily_vol:     float,
        eta:           float = 0.1,
    ) -> float:
        """
        Square-root market impact in basis points.

        impact_bps = η · σ · √(|Q|/ADV) · 1e4

        Parameters
        ----------
        participation : |Q| / ADV
        daily_vol     : annualised-daily return vol
        eta           : impact coefficient (default 0.1)
        """
        return float(eta * daily_vol * np.sqrt(max(participation, 0.0)) * 1e4)

    @staticmethod
    def half_spread_bps(spread: float, price: float) -> float:
        """Convert absolute half-spread to basis points."""
        return float(spread / price * 1e4) if price > 0 else 0.0
