"""
Backtester Tearsheet
====================
Wraps PnLReporter + SharpeDeflator to produce a complete, publication-quality
strategy tearsheet from an EngineResult or raw inputs.

Output formats
--------------
  .text()   – ASCII block suitable for terminal / logging
  .dict()   – flat dict of all metrics (JSON-serialisable, floats only)
  .report   – underlying PnLReport dataclass
  .psr      – Probabilistic Sharpe Ratio (scalar 0-1)
  .dsr      – Deflated Sharpe Ratio (scalar 0-1)

Usage
-----
>>> result   = engine.run()           # EngineResult from BacktestEngine
>>> sheet    = Tearsheet.from_result(result, periods_per_year=252*6.5*3600)
>>> print(sheet.text())
>>> metrics  = sheet.dict()

References
----------
  Bailey & Lopez de Prado (2014). "The Deflated Sharpe Ratio."
  Sharpe (1994). "The Sharpe Ratio." Journal of Portfolio Management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from risk.pnl_reporter import PnLReport, PnLReporter
from validation.sharpe_deflator import SharpeDeflator


# ── Tearsheet ─────────────────────────────────────────────────────────────────

@dataclass
class Tearsheet:
    """
    Full strategy tearsheet combining PnL metrics with Sharpe reliability tests.

    Attributes
    ----------
    report          : PnLReport from PnLReporter
    psr             : Probabilistic Sharpe Ratio ∈ [0,1]
    dsr             : Deflated Sharpe Ratio ∈ [0,1] (N-trial corrected)
    min_track_len   : minimum track length (periods) to justify observed SR
    n_trials        : number of trials used for DSR
    symbol          : instrument label
    initial_equity  : starting equity value
    final_equity    : ending equity value
    total_commissions : total commissions paid (if available)
    n_fills         : number of fills executed
    """

    report:            PnLReport
    psr:               float
    dsr:               float
    min_track_len:     int
    n_trials:          int
    symbol:            str
    initial_equity:    float
    final_equity:      float
    total_commissions: float
    n_fills:           int

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_result(
        cls,
        result,                         # EngineResult from BacktestEngine
        periods_per_year: int   = 252,
        risk_free_rate:   float = 0.0,
        sr_benchmark:     float = 0.0,
        n_trials:         int   = 1,
        initial_equity:   float = 1_000_000.0,
        total_commissions: float = 0.0,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> "Tearsheet":
        """
        Build a Tearsheet from a BacktestEngine EngineResult.

        Parameters
        ----------
        result            : EngineResult returned by BacktestEngine.run().
        periods_per_year  : annualisation factor (252 for daily, larger for intraday).
        risk_free_rate    : annualised risk-free rate (default 0).
        sr_benchmark      : SR₀ for PSR hypothesis test (default 0 = SR > 0).
        n_trials          : N distinct strategy variants tried (for DSR).
        initial_equity    : starting equity (for total P&L display).
        total_commissions : total commissions charged (passed in from portfolio).
        benchmark_returns : optional benchmark return series for Info Ratio.
        """
        return cls.from_returns(
            returns           = result.returns,
            symbol            = result.symbol,
            periods_per_year  = periods_per_year,
            risk_free_rate    = risk_free_rate,
            sr_benchmark      = sr_benchmark,
            n_trials          = n_trials,
            initial_equity    = initial_equity,
            final_equity      = float(result.equity_curve[-1]) if len(result.equity_curve) > 0 else initial_equity,
            total_commissions = total_commissions,
            n_fills           = len(result.fills),
            benchmark_returns = benchmark_returns,
        )

    @classmethod
    def from_returns(
        cls,
        returns:           np.ndarray,
        symbol:            str   = "STRAT",
        periods_per_year:  int   = 252,
        risk_free_rate:    float = 0.0,
        sr_benchmark:      float = 0.0,
        n_trials:          int   = 1,
        initial_equity:    float = 1_000_000.0,
        final_equity:      Optional[float] = None,
        total_commissions: float = 0.0,
        n_fills:           int   = 0,
        trade_pnls:        Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> "Tearsheet":
        """
        Build a Tearsheet directly from a return series.

        Parameters
        ----------
        returns          : per-period fractional return series, shape (T,).
        symbol           : instrument / strategy label.
        periods_per_year : annualisation factor.
        risk_free_rate   : annualised risk-free rate.
        sr_benchmark     : SR₀ for PSR null hypothesis.
        n_trials         : number of strategy variants tried (for DSR).
        initial_equity   : starting equity (display only).
        final_equity     : ending equity (display only).  Inferred from returns if None.
        total_commissions: total commissions paid.
        n_fills          : number of fills executed.
        trade_pnls       : signed P&L per closed trade for trade statistics.
        benchmark_returns: optional benchmark returns for Info Ratio.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]

        # ── PnL report ────────────────────────────────────────────────────────
        reporter = PnLReporter(
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        report = reporter.report(
            returns    = r,
            benchmark  = benchmark_returns,
            trade_pnls = trade_pnls,
        )

        # ── Sharpe deflation ──────────────────────────────────────────────────
        deflator = SharpeDeflator(
            sr_benchmark     = sr_benchmark,
            n_trials         = n_trials,
            periods_per_year = periods_per_year,
        )
        try:
            defl_result = deflator.compute(r)
            psr  = defl_result.psr
            dsr  = defl_result.dsr
            mtrl = defl_result.mtrl
        except (ValueError, ZeroDivisionError):
            # Too few observations or zero-variance returns
            psr  = float("nan")
            dsr  = float("nan")
            mtrl = float("nan")

        # Derive final equity from returns if not supplied
        if final_equity is None and len(r) > 0:
            wealth     = np.cumprod(1.0 + r)
            final_equity = initial_equity * float(wealth[-1])
        elif final_equity is None:
            final_equity = initial_equity

        return cls(
            report            = report,
            psr               = float(psr),
            dsr               = float(dsr),
            min_track_len     = int(np.ceil(mtrl)) if np.isfinite(mtrl) else 0,
            n_trials          = n_trials,
            symbol            = symbol,
            initial_equity    = initial_equity,
            final_equity      = final_equity,
            total_commissions = total_commissions,
            n_fills           = n_fills,
        )

    # ── Output methods ────────────────────────────────────────────────────────

    def text(self) -> str:
        """Return a full ASCII tearsheet string."""
        r   = self.report
        pnl = self.final_equity - self.initial_equity

        lines = [
            "╔══════════════════════════════════════════════╗",
            f"║  LIMIT ORDER BOOK  ·  Strategy Tearsheet     ║",
            f"║  Symbol : {self.symbol:<35s}║",
            "╠══════════════════════════════════════════════╣",
            "║  EQUITY                                      ║",
            f"║   Initial equity    : ${self.initial_equity:>18,.2f}    ║",
            f"║   Final equity      : ${self.final_equity:>18,.2f}    ║",
            f"║   Total P&L         : ${pnl:>+18,.2f}    ║",
            f"║   Commissions paid  : ${self.total_commissions:>18,.2f}    ║",
            f"║   Number of fills   : {self.n_fills:>22d}    ║",
            "╠══════════════════════════════════════════════╣",
            "║  RETURNS                                     ║",
            f"║   Periods / year    : {r.periods_per_year:>22d}    ║",
            f"║   Observations      : {r.n_periods:>22d}    ║",
            f"║   Cumulative return : {r.total_return:>+22.2%}    ║",
            f"║   Ann. return       : {r.ann_return:>+22.2%}    ║",
            f"║   Ann. volatility   : {r.ann_volatility:>22.2%}    ║",
            "╠══════════════════════════════════════════════╣",
            "║  RISK-ADJUSTED RATIOS                        ║",
            f"║   Sharpe ratio      : {r.sharpe:>22.4f}    ║",
            f"║   Sortino ratio     : {r.sortino:>22.4f}    ║",
            f"║   Calmar ratio      : {r.calmar:>22.4f}    ║",
            f"║   Info ratio        : {r.info_ratio:>22.4f}    ║",
            "╠══════════════════════════════════════════════╣",
            "║  SHARPE RELIABILITY                          ║",
            f"║   PSR (SR > {self.report.sharpe:.2f})    : {self.psr:>22.4f}    ║",
            f"║   DSR (N={self.n_trials:>4d} trials)   : {self.dsr:>22.4f}    ║",
            f"║   Min track length  : {self.min_track_len:>22d}    ║",
            "╠══════════════════════════════════════════════╣",
            "║  DRAWDOWN                                    ║",
            f"║   Max drawdown      : {r.max_drawdown:>22.2%}    ║",
            f"║   Max DD duration   : {r.max_dd_duration:>22d}    ║",
            f"║   Avg drawdown      : {r.avg_drawdown:>22.2%}    ║",
            "╠══════════════════════════════════════════════╣",
            "║  RISK MEASURES                               ║",
            f"║   VaR  95% / 99%    : {r.var_95:>10.2%} / {r.var_99:>10.2%}    ║",
            f"║   CVaR 95% / 99%    : {r.cvar_95:>10.2%} / {r.cvar_99:>10.2%}    ║",
            "╠══════════════════════════════════════════════╣",
            "║  TRADE STATISTICS                            ║",
            f"║   Win rate          : {_fmt_pct(r.win_rate):>22s}    ║",
            f"║   Profit factor     : {_fmt_float(r.profit_factor, '.4f'):>22s}    ║",
            f"║   Avg win / loss    : {_fmt_float(r.avg_win, '.4f'):>10s} / {_fmt_float(r.avg_loss, '.4f'):>10s}    ║",
            f"║   Expectancy        : {_fmt_float(r.expectancy, '.4f'):>22s}    ║",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def dict(self) -> Dict[str, float]:
        """
        Return all metrics as a flat dict (all values are Python floats or ints).
        Suitable for JSON serialisation / pandas DataFrame construction.
        """
        r = self.report
        return {
            # equity
            "initial_equity":    self.initial_equity,
            "final_equity":      self.final_equity,
            "total_pnl":         self.final_equity - self.initial_equity,
            "total_commissions": self.total_commissions,
            "n_fills":           float(self.n_fills),
            # returns
            "n_periods":         float(r.n_periods),
            "periods_per_year":  float(r.periods_per_year),
            "total_return":      r.total_return,
            "ann_return":        r.ann_return,
            "ann_volatility":    r.ann_volatility,
            # ratios
            "sharpe":            r.sharpe,
            "sortino":           r.sortino,
            "calmar":            r.calmar,
            "info_ratio":        r.info_ratio,
            # sharpe reliability
            "psr":               self.psr,
            "dsr":               self.dsr,
            "min_track_len":     float(self.min_track_len),
            "n_trials":          float(self.n_trials),
            # drawdown
            "max_drawdown":      r.max_drawdown,
            "max_dd_duration":   float(r.max_dd_duration),
            "avg_drawdown":      r.avg_drawdown,
            # risk
            "var_95":            r.var_95,
            "var_99":            r.var_99,
            "cvar_95":           r.cvar_95,
            "cvar_99":           r.cvar_99,
            # trade stats
            "win_rate":          r.win_rate     if np.isfinite(r.win_rate)     else float("nan"),
            "profit_factor":     r.profit_factor if np.isfinite(r.profit_factor) else float("nan"),
            "avg_win":           r.avg_win       if np.isfinite(r.avg_win)       else float("nan"),
            "avg_loss":          r.avg_loss      if np.isfinite(r.avg_loss)      else float("nan"),
            "expectancy":        r.expectancy    if np.isfinite(r.expectancy)    else float("nan"),
        }

    def __repr__(self) -> str:
        return (f"Tearsheet({self.symbol}, "
                f"SR={self.report.sharpe:.3f}, "
                f"PSR={self.psr:.3f}, "
                f"MaxDD={self.report.max_drawdown:.2%})")


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_skew(r: np.ndarray) -> float:
    if len(r) < 3:
        return 0.0
    mu  = r.mean()
    sig = r.std(ddof=1)
    if sig < 1e-12:
        return 0.0
    return float(np.mean(((r - mu) / sig) ** 3))


def _safe_kurt(r: np.ndarray) -> float:
    """Excess kurtosis (Fisher, i.e. Normal = 0)."""
    if len(r) < 4:
        return 0.0
    mu  = r.mean()
    sig = r.std(ddof=1)
    if sig < 1e-12:
        return 0.0
    return float(np.mean(((r - mu) / sig) ** 4)) - 3.0


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}" if np.isfinite(v) else "N/A"


def _fmt_float(v: float, fmt: str = ".4f") -> str:
    return format(v, fmt) if np.isfinite(v) else "N/A"
