"""
Daily OHLCV Downloader
========================
Fetches end-of-day OHLCV data from Yahoo Finance (v8 JSON API) for one or
more tickers and saves them as Parquet files (falling back to CSV if
``pyarrow`` / ``fastparquet`` are not installed).

Every downloaded file is registered in the local DataCatalog.

Data
----
  Source  : Yahoo Finance public API  (no key required)
  Interval: 1 day (adjustable: "1d", "1wk", "1mo")
  Fields  : Open, High, Low, Close, Volume, Adj Close
  Format  : Parquet (preferred) or CSV

File naming
-----------
  raw/daily/{SYMBOL}_{start}_{end}.parquet
  e.g.  raw/daily/AAPL_2020-01-01_2024-12-31.parquet

Example
-------
  >>> dl = DailyDownloader(dest_dir="raw/daily")
  >>> dl.download("AAPL", "2020-01-01", "2024-12-31")
  >>> dl.download_many(["AAPL", "MSFT", "NVDA"], "2023-01-01", "2024-12-31")

CLI
---
  python -m data.download_daily --symbols AAPL MSFT NVDA \\
         --from 2020-01-01 --to 2024-12-31
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

log = logging.getLogger(__name__)

# ── Yahoo Finance API constants ────────────────────────────────────────────────
_YF_BASE    = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_YF_BASE_v2 = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    ),
    "Accept":     "application/json",
}
_RETRY_DELAY_S = [2, 5, 15]    # back-off schedule (seconds)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_to_epoch(iso: str) -> int:
    """Convert "YYYY-MM-DD" to a UTC Unix timestamp at 00:00:00."""
    d = date.fromisoformat(iso)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _fetch_json(url: str, retries: int = 3) -> dict:
    """HTTP GET with retries; returns parsed JSON dict."""
    req = urllib.request.Request(url, headers=_HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:          # rate-limited
                delay = _RETRY_DELAY_S[min(attempt, len(_RETRY_DELAY_S) - 1)]
                log.warning("Rate-limited; waiting %ds …", delay)
                time.sleep(delay)
            else:
                raise
        except urllib.error.URLError as exc:
            if attempt == retries - 1:
                raise
            delay = _RETRY_DELAY_S[min(attempt, len(_RETRY_DELAY_S) - 1)]
            log.warning("Network error (%s); retrying in %ds …", exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def _parse_yf_response(data: dict) -> "pd.DataFrame":
    """
    Parse a Yahoo Finance v8 JSON response into a pandas DataFrame.

    Returns a DataFrame with columns:
        date (index), open, high, low, close, adj_close, volume
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required: pip install pandas") from e

    result = data["chart"]["result"]
    if not result:
        error  = data["chart"].get("error") or {}
        raise ValueError(f"Yahoo Finance error: {error.get('description', 'empty result')}")

    r         = result[0]
    timestamps = r["timestamp"]
    q         = r["indicators"]["quote"][0]
    adj       = r["indicators"].get("adjclose", [{}])[0]

    df = pd.DataFrame({
        "open":      q.get("open",   [None] * len(timestamps)),
        "high":      q.get("high",   [None] * len(timestamps)),
        "low":       q.get("low",    [None] * len(timestamps)),
        "close":     q.get("close",  [None] * len(timestamps)),
        "adj_close": adj.get("adjclose", [None] * len(timestamps)),
        "volume":    q.get("volume", [None] * len(timestamps)),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True).normalize())

    df.index.name = "date"
    df = df.dropna(how="all")
    df = df.sort_index()
    return df


def _save(df: "pd.DataFrame", path: Path) -> str:
    """Save DataFrame; Parquet if available, else CSV.  Returns format used."""
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=True)
        return "parquet"
    except (ImportError, Exception):
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=True)
        return "csv"


# ── DailyDownloader ────────────────────────────────────────────────────────────

class DailyDownloader:
    """
    Download and cache daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    dest_dir  : local directory for Parquet/CSV files.
    catalog   : DataCatalog instance (or None to skip registration).
    interval  : Yahoo Finance bar interval ("1d", "1wk", "1mo").
    retries   : HTTP retry count per request.
    """

    def __init__(
        self,
        dest_dir:  Union[str, Path] = "raw/daily",
        catalog    = None,
        interval:  str  = "1d",
        retries:   int  = 3,
    ) -> None:
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.catalog  = catalog
        self.interval = interval
        self.retries  = retries

    # ── Single-ticker download ─────────────────────────────────────────────────

    def download(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        force:    bool = False,
    ) -> Optional[Path]:
        """
        Download OHLCV bars for one ticker.

        Parameters
        ----------
        symbol : ticker symbol (e.g. "AAPL").
        start  : start date inclusive, "YYYY-MM-DD".
        end    : end date inclusive, "YYYY-MM-DD".
        force  : re-download even if file exists.

        Returns
        -------
        Path to saved file, or None on failure.
        """
        sym = symbol.upper()
        stem = f"{sym}_{start}_{end}"
        # Try both extensions
        for ext in (".parquet", ".csv"):
            candidate = self.dest_dir / (stem + ext)
            if candidate.exists() and not force:
                log.info("Already exists: %s", candidate.name)
                return candidate

        log.info("Downloading %s  %s → %s …", sym, start, end)

        t1 = _ts_to_epoch(start)
        t2 = _ts_to_epoch(end) + 86400   # include the end day

        url = _YF_BASE.format(symbol=urllib.parse.quote(sym))
        params = {
            "period1":  t1,
            "period2":  t2,
            "interval": self.interval,
            "events":   "history",
            "includeAdjustedClose": "true",
        }
        full_url = url + "?" + urllib.parse.urlencode(params)

        try:
            data = _fetch_json(full_url, retries=self.retries)
        except Exception as exc:
            # Try the backup Yahoo Finance host
            log.warning("Primary host failed (%s); trying backup …", exc)
            url2      = _YF_BASE_v2.format(symbol=urllib.parse.quote(sym))
            full_url2 = url2 + "?" + urllib.parse.urlencode(params)
            try:
                data = _fetch_json(full_url2, retries=self.retries)
            except Exception as exc2:
                log.error("Both hosts failed for %s: %s", sym, exc2)
                return None

        try:
            df = _parse_yf_response(data)
        except Exception as exc:
            log.error("Parse error for %s: %s", sym, exc)
            return None

        if df.empty:
            log.warning("No data returned for %s between %s and %s", sym, start, end)
            return None

        dest_stem = self.dest_dir / stem
        fmt  = _save(df, dest_stem)
        path = dest_stem.with_suffix(f".{fmt}" if fmt != "parquet" else ".parquet")
        if fmt == "csv":
            path = dest_stem.with_suffix(".csv")

        log.info("Saved %s  [%d rows]  → %s", sym, len(df), path.name)
        self._register(
            symbol    = sym,
            date_str  = start,
            path      = path,
            source    = full_url,
            n_rows    = len(df),
            start     = start,
            end       = end,
        )
        return path

    # ── Multi-ticker ──────────────────────────────────────────────────────────

    def download_many(
        self,
        symbols:   List[str],
        start:     str,
        end:       str,
        force:     bool = False,
        delay_s:   float = 0.5,
    ) -> Dict[str, Optional[Path]]:
        """
        Download multiple tickers, pausing `delay_s` seconds between
        requests to avoid rate-limiting.

        Returns
        -------
        Dict mapping symbol → Path (or None on failure).
        """
        results: Dict[str, Optional[Path]] = {}
        for i, sym in enumerate(symbols):
            results[sym] = self.download(sym, start, end, force=force)
            if i < len(symbols) - 1:
                time.sleep(delay_s)
        return results

    # ── Load helper ───────────────────────────────────────────────────────────

    @staticmethod
    def load(path: Union[str, Path]) -> "pd.DataFrame":
        """
        Load a previously downloaded daily file (Parquet or CSV).

        Returns a DataFrame with columns:
            date (index), open, high, low, close, adj_close, volume
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required: pip install pandas") from e

        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

    # ── Catalog helper ────────────────────────────────────────────────────────

    def _register(
        self,
        symbol:   str,
        date_str: str,
        path:     Path,
        source:   str,
        n_rows:   int,
        start:    str,
        end:      str,
    ) -> None:
        if self.catalog is None:
            return
        try:
            from data.data_catalog import CatalogEntry
        except ImportError:
            return

        entry = CatalogEntry(
            data_type   = "daily",
            date        = date_str,
            symbol      = symbol,
            filename    = path.name,
            path        = str(path.resolve()),
            size_bytes  = path.stat().st_size,
            source_url  = source,
            download_ts = CatalogEntry.now_ts(),
            tags        = "ohlcv,daily,yahoo",
            notes       = f"rows={n_rows}, start={start}, end={end}",
        )
        self.catalog.add(entry, replace=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog        = "download_daily",
        description = "Download daily OHLCV data from Yahoo Finance",
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        metavar="TICKER",
                        help="One or more ticker symbols (e.g. AAPL MSFT)")
    parser.add_argument("--from", dest="date_from", required=True,
                        metavar="YYYY-MM-DD", help="Start date (inclusive)")
    parser.add_argument("--to",   dest="date_to",   required=True,
                        metavar="YYYY-MM-DD", help="End date (inclusive)")
    parser.add_argument("--dest", default="raw/daily",
                        help="Local destination directory (default: raw/daily)")
    parser.add_argument("--interval", default="1d",
                        choices=["1d", "1wk", "1mo"],
                        help="Bar interval (default: 1d)")
    parser.add_argument("--force",  action="store_true",
                        help="Re-download even if file exists")
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="Seconds between requests (default: 0.5)")
    parser.add_argument("--no-catalog", action="store_true",
                        help="Skip catalog registration")

    args = parser.parse_args()

    catalog = None
    if not args.no_catalog:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.data_catalog import DataCatalog
            catalog = DataCatalog()
        except Exception as exc:
            log.warning("Could not open catalog: %s", exc)

    dl = DailyDownloader(
        dest_dir = args.dest,
        catalog  = catalog,
        interval = args.interval,
    )

    results = dl.download_many(
        symbols  = args.symbols,
        start    = args.date_from,
        end      = args.date_to,
        force    = args.force,
        delay_s  = args.delay,
    )

    print()
    for sym, path in results.items():
        status = f"✓  {path}" if path else "✗  FAILED"
        print(f"  {sym:<12}  {status}")
    n_ok = sum(1 for p in results.values() if p)
    print(f"\n  {n_ok}/{len(results)} downloads successful.\n")
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    _main()
