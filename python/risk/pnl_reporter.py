"""
P&L Reporter
============
Computes standard performance metrics from a strategy's return series.

Metrics
-------
  Return metrics
    - Cumulative return
    - Annualised return (geometric)
    - Annualised volatility

  Risk-adjusted ratios
    - Sharpe ratio (annualised)
    - Sortino ratio (downside deviation)
    - Calmar ratio (annualised return / max drawdown)
    - Information ratio (vs. benchmark)

  Drawdown
    - Maximum drawdown (peak-to-trough)
    - Drawdown duration (trading periods)
    - Average drawdown

  Trade statistics (if trade list provided)
    - Win rate, profit factor
    - Average win / average loss
    - Expectancy per trade

  Risk measures
    - VaR (Historical, 95% and 99%)
    - CVaR / Expected Shortfall

References
----------
  Sharpe (1994). "The Sharpe Ratio." Journal of Portfolio Management.
  Sortino & Lee (1994). "Performance measurement in a downside risk framework."
  Bailey & Lopez de Prado (2012). "The Sharpe Ratio Efficient Frontier."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PnLReport:
    """Full performance report for a strategy."""
    # Return metrics
    total_return:     float
    ann_return:       float
    ann_volatility:   float

    # Ratios
    sharpe:           float
    sortino:          float
    calmar:           float
    info_ratio:       float    # vs. benchmark; 0.0 if no benchmark

    # Drawdown
    max_drawdown:     float    # expressed as positive fraction
    max_dd_duration:  int      # trading periods
    avg_drawdown:     float

    # Risk
    var_95:           float    # VaR at 95% confidence (negative loss)
    var_99:           float
    cvar_95:          float    # Expected Shortfall at 95%
    cvar_99:          float

    # Trade stats (NaN if not available)
    win_rate:         float
    profit_factor:    float
    avg_win:          float
    avg_loss:         float
    expectancy:       float

    # Meta
    n_periods:        int
    periods_per_year: int

    def __repr__(self) -> str:
        return (f"PnLReport(Sharpe={self.sharpe:.2f}, "
                f"Sortino={self.sortino:.2f}, "
                f"MaxDD={self.max_drawdown:.2%}, "
                f"Calmar={self.calmar:.2f})")

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "══════════════ P&L Report ══════════════",
            f"  Periods          : {self.n_periods} ({self.periods_per_year}/yr)",
            f"  Cumulative return: {self.total_return:+.2%}",
            f"  Ann. return      : {self.ann_return:+.2%}",
            f"  Ann. volatility  : {self.ann_volatility:.2%}",
            "──────────────────────────────────────",
            f"  Sharpe ratio     : {self.sharpe:.3f}",
            f"  Sortino ratio    : {self.sortino:.3f}",
            f"  Calmar ratio     : {self.calmar:.3f}",
            f"  Info ratio       : {self.info_ratio:.3f}",
            "──────────────────────────────────────",
            f"  Max drawdown     : {self.max_drawdown:.2%}",
            f"  Max DD duration  : {self.max_dd_duration} periods",
            f"  Avg drawdown     : {self.avg_drawdown:.2%}",
            "──────────────────────────────────────",
            f"  VaR 95 / 99      : {self.var_95:.2%} / {self.var_99:.2%}",
            f"  CVaR 95 / 99     : {self.cvar_95:.2%} / {self.cvar_99:.2%}",
            "──────────────────────────────────────",
            f"  Win rate         : {self.win_rate:.1%}",
            f"  Profit factor    : {self.profit_factor:.2f}",
            f"  Avg win / loss   : {self.avg_win:.4f} / {self.avg_loss:.4f}",
            f"  Expectancy       : {self.expectancy:.4f}",
            "════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ── Helper functions ──────────────────────────────────────────────────────────

def _drawdown_series(cumulative_returns: np.ndarray) -> np.ndarray:
    """
    Compute drawdown series from cumulative return (wealth) index.
    Returns array of drawdowns as positive fractions.
    """
    hwm = np.maximum.accumulate(cumulative_returns)
    dd  = (hwm - cumulative_returns) / np.maximum(hwm, 1e-12)
    return dd


def _max_drawdown_duration(dd_series: np.ndarray) -> int:
    """Longest consecutive streak of non-zero drawdown (in periods)."""
    max_dur = 0
    current = 0
    for d in dd_series:
        if d > 1e-8:
            current += 1
            max_dur  = max(max_dur, current)
        else:
            current  = 0
    return max_dur


def _var_cvar(returns: np.ndarray, confidence: float) -> Tuple[float, float]:
    """Historical VaR and CVaR (Expected Shortfall) at `confidence` level."""
    r    = returns[np.isfinite(returns)]
    if len(r) == 0:
        return np.nan, np.nan
    var  = float(np.percentile(r, (1.0 - confidence) * 100))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var
    return var, cvar


# ── PnL Reporter ──────────────────────────────────────────────────────────────

class PnLReporter:
    """
    Compute a full performance report from a return series.

    Parameters
    ----------
    periods_per_year : trading periods per year (252 = daily, 252*6.5*60 = minute).
    risk_free_rate   : annualised risk-free rate for Sharpe adjustment (default 0).

    Usage
    -----
    >>> reporter = PnLReporter(periods_per_year=252)
    >>> report   = reporter.report(daily_returns)
    >>> print(report.summary())
    """

    def __init__(
        self,
        periods_per_year: int   = 252,
        risk_free_rate:   float = 0.0,
    ) -> None:
        self.periods_per_year = periods_per_year
        self.risk_free_rate   = risk_free_rate

    def report(
        self,
        returns:    np.ndarray,
        benchmark:  Optional[np.ndarray] = None,
        trade_pnls: Optional[np.ndarray] = None,
    ) -> PnLReport:
        """
        Generate the full performance report.

        Parameters
        ----------
        returns    : per-period return series (fractional), shape (T,).
        benchmark  : per-period benchmark returns for Info Ratio (optional).
        trade_pnls : signed P&L per closed trade for trade statistics (optional).

        Returns
        -------
        PnLReport.
        """
        r  = np.asarray(returns, dtype=float)
        r  = r[np.isfinite(r)]
        T  = len(r)
        scale = np.sqrt(self.periods_per_year)
        rf_pp = self.risk_free_rate / self.periods_per_year

        # Cumulative wealth index (starting at 1)
        wealth    = np.cumprod(1.0 + r) if T > 0 else np.array([1.0])
        total_ret = float(wealth[-1] - 1.0) if T > 0 else 0.0
        ann_ret   = float(wealth[-1] ** (self.periods_per_year / max(T, 1)) - 1.0) if T > 0 else 0.0
        ann_vol   = float(r.std(ddof=1) * scale) if T > 1 else 0.0

        # Sharpe
        excess = r - rf_pp
        sharpe = float((excess.mean() / max(r.std(ddof=1), 1e-12)) * scale) if T > 1 else 0.0

        # Sortino (downside deviation)
        downside = np.where(r < 0, r, 0.0)
        dd_std   = np.sqrt(np.mean(downside ** 2))
        sortino  = float((r.mean() / max(dd_std, 1e-12)) * scale) if T > 1 else 0.0

        # Drawdown
        dd_series = _drawdown_series(wealth)
        max_dd    = float(dd_series.max())
        avg_dd    = float(dd_series.mean())
        max_dd_dur = _max_drawdown_duration(dd_series)

        # Calmar
        calmar = float(ann_ret / max(max_dd, 1e-12))

        # Information ratio
        if benchmark is not None:
            bm  = np.asarray(benchmark, dtype=float)[:T]
            active = r - bm
            ir  = float((active.mean() / max(active.std(ddof=1), 1e-12)) * scale)
        else:
            ir = 0.0

        # VaR / CVaR
        var95, cvar95 = _var_cvar(r, 0.95)
        var99, cvar99 = _var_cvar(r, 0.99)

        # Trade statistics
        if trade_pnls is not None:
            tp = np.asarray(trade_pnls, dtype=float)
            tp = tp[np.isfinite(tp)]
            if len(tp) > 0:
                wins   = tp[tp > 0]
                losses = tp[tp < 0]
                win_rate   = float(len(wins) / len(tp))
                avg_win    = float(wins.mean())  if len(wins)   > 0 else 0.0
                avg_loss   = float(losses.mean()) if len(losses) > 0 else 0.0
                pf_denom   = abs(losses.sum()) if len(losses) > 0 else 1e-12
                profit_factor = float(wins.sum() / pf_denom) if len(losses) > 0 else np.inf
                expectancy = float(tp.mean())
            else:
                win_rate = profit_factor = avg_win = avg_loss = expectancy = np.nan
        else:
            win_rate = profit_factor = avg_win = avg_loss = expectancy = np.nan

        return PnLReport(
            total_return     = total_ret,
            ann_return       = ann_ret,
            ann_volatility   = ann_vol,
            sharpe           = sharpe,
            sortino          = sortino,
            calmar           = calmar,
            info_ratio       = ir,
            max_drawdown     = max_dd,
            max_dd_duration  = max_dd_dur,
            avg_drawdown     = avg_dd,
            var_95           = var95,
            var_99           = var99,
            cvar_95          = cvar95,
            cvar_99          = cvar99,
            win_rate         = win_rate,
            profit_factor    = profit_factor,
            avg_win          = avg_win,
            avg_loss         = avg_loss,
            expectancy       = expectancy,
            n_periods        = T,
            periods_per_year = self.periods_per_year,
        )

    # ── Per-period metrics (useful for live monitoring) ───────────────────────

    @staticmethod
    def rolling_sharpe(
        returns: np.ndarray,
        window:  int,
        scale:   float = np.sqrt(252),
    ) -> np.ndarray:
        """Rolling Sharpe ratio over `window` periods."""
        r   = np.asarray(returns, dtype=float)
        n   = len(r)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = r[i - window + 1 : i + 1]
            mu    = chunk.mean()
            sig   = chunk.std(ddof=1)
            out[i] = (mu / sig * scale) if sig > 1e-12 else 0.0
        return out

    @staticmethod
    def rolling_drawdown(
        returns: np.ndarray,
        window:  int,
    ) -> np.ndarray:
        """Rolling maximum drawdown over `window` periods."""
        r   = np.asarray(returns, dtype=float)
        n   = len(r)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk  = r[i - window + 1 : i + 1]
            wealth = np.cumprod(1.0 + chunk)
            out[i] = float(_drawdown_series(wealth).max())
        return out
