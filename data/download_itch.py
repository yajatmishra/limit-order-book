"""
NASDAQ TotalView-ITCH 5.0 Downloader
=======================================
Fetches historical ITCH 5.0 binary files from the official NASDAQ EMI FTP
server and registers them in the local DataCatalog.

Source
------
  URL  : https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/
  Auth : anonymous FTP (no credentials needed)
  Files: MMDDYYYY.NASDAQ_ITCH50.gz   (gzip-compressed binary)
         MMDDYYYY.NASDAQ_ITCH50       (uncompressed, approx 12–20 GB/day)

File naming convention
----------------------
  NASDAQ stores files by US-locale date: MMDDYYYY.
  This downloader accepts ISO dates (YYYY-MM-DD) and converts internally.

Example
-------
  >>> dl = ItchDownloader(dest_dir="raw/itch")
  >>> dl.list_available()                             # print remote listing
  >>> dl.download("2024-01-15")                       # single day
  >>> dl.download_range("2024-01-02", "2024-01-05")   # date range
  >>> dl.download("2024-01-15", verify_sha256=True)   # with checksum

CLI
---
  python -m data.download_itch --date 2024-01-15
  python -m data.download_itch --from 2024-01-02 --to 2024-01-05
  python -m data.download_itch --list
"""

from __future__ import annotations

import ftplib
import gzip
import hashlib
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── NASDAQ EMI FTP constants ──────────────────────────────────────────────────
_FTP_HOST   = "emi.nasdaq.com"
_FTP_DIR    = "/ITCH/Nasdaq ITCH/"
_SUFFIX_GZ  = ".NASDAQ_ITCH50.gz"
_SUFFIX_RAW = ".NASDAQ_ITCH50"
_CHUNK      = 1 << 20     # 1 MiB read chunks


def _iso_to_ftp(iso: str) -> str:
    """Convert "YYYY-MM-DD" → "MMDDYYYY" (NASDAQ naming convention)."""
    d = date.fromisoformat(iso)
    return d.strftime("%m%d%Y")


def _ftp_to_iso(ftp_name: str) -> Optional[str]:
    """Convert "MMDDYYYY.NASDAQ_ITCH50.gz" → "YYYY-MM-DD", or None."""
    stem = ftp_name.split(".")[0]
    if len(stem) != 8:
        return None
    try:
        d = datetime.strptime(stem, "%m%d%Y")
        return d.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_bar(desc: str, total: int) -> Callable[[int], None]:
    """
    Return a simple terminal progress callback.
    Falls back to a no-op if ``tqdm`` is not installed.
    """
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, desc=desc, unit="B",
                   unit_scale=True, unit_divisor=1024,
                   ncols=80, leave=True)

        def update(n: int) -> None:
            bar.update(n)
            if bar.n >= bar.total:
                bar.close()

        return update
    except ImportError:
        last: list = [0, time.time()]

        def update(n: int) -> None:
            last[0] += n
            if time.time() - last[1] > 2:
                pct = 100 * last[0] / total if total else 0
                mb  = last[0] / 1024 ** 2
                print(f"  {desc}: {mb:.1f} MB  ({pct:.1f}%)", flush=True)
                last[1] = time.time()

        return update


# ── ItchDownloader ─────────────────────────────────────────────────────────────

class ItchDownloader:
    """
    Download NASDAQ TotalView-ITCH 5.0 historical binary files.

    Parameters
    ----------
    dest_dir     : local directory where .gz files are saved.
                   Created automatically if it does not exist.
    catalog      : DataCatalog instance for registration.
                   Pass ``None`` to skip catalog logging.
    keep_gz      : if True (default), keep the compressed file on disk.
                   If False, decompress in-place and delete the .gz.
    timeout      : FTP connection timeout in seconds (default 30).
    retry        : number of automatic retries on network failure (default 3).
    """

    def __init__(
        self,
        dest_dir:  Union[str, Path] = "raw/itch",
        catalog    = None,
        keep_gz:   bool = True,
        timeout:   int  = 30,
        retry:     int  = 3,
    ) -> None:
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.catalog  = catalog
        self.keep_gz  = keep_gz
        self.timeout  = timeout
        self.retry    = retry

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_available(self) -> List[Tuple[str, str, int]]:
        """
        Return list of (ftp_filename, iso_date, size_bytes) for all files
        visible on the NASDAQ EMI FTP server.

        Prints a formatted table to stdout.
        """
        records = []
        with ftplib.FTP(timeout=self.timeout) as ftp:
            ftp.connect(_FTP_HOST)
            ftp.login()
            ftp.cwd(_FTP_DIR)
            lines = []
            ftp.retrlines("LIST", lines.append)

        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            fname = parts[-1]
            if not fname.endswith(_SUFFIX_GZ) and not fname.endswith(_SUFFIX_RAW):
                continue
            try:
                size = int(parts[4])
            except (ValueError, IndexError):
                size = 0
            iso = _ftp_to_iso(fname) or "unknown"
            records.append((fname, iso, size))

        records.sort(key=lambda r: r[1])

        print(f"\n{'Filename':<40} {'Date':<12} {'Size':>10}")
        print("─" * 65)
        for fname, iso, size in records:
            gb = size / 1024 ** 3
            print(f"  {fname:<38} {iso:<12} {gb:>8.2f} GB")
        print(f"\n  Total: {len(records)} files\n")
        return records

    def download(
        self,
        iso_date:       str,
        force:          bool = False,
        verify_sha256:  bool = False,
        decompress:     bool = False,
    ) -> Optional[Path]:
        """
        Download one day's ITCH file.

        Parameters
        ----------
        iso_date      : trading date in "YYYY-MM-DD" format.
        force         : re-download even if the file already exists locally.
        verify_sha256 : compute and log the SHA-256 after download.
        decompress    : decompress the .gz file (WARNING: ~8× larger on disk).

        Returns
        -------
        Path to the downloaded file, or None if the download failed.
        """
        ftp_name = _iso_to_ftp(iso_date) + _SUFFIX_GZ
        dest     = self.dest_dir / ftp_name

        if dest.exists() and not force:
            log.info("Already exists: %s", dest)
            return dest

        for attempt in range(1, self.retry + 1):
            try:
                self._ftp_download(ftp_name, dest)
                break
            except (ftplib.Error, OSError, EOFError) as exc:
                log.warning("Attempt %d/%d failed: %s", attempt, self.retry, exc)
                if attempt == self.retry:
                    log.error("All retries exhausted for %s", ftp_name)
                    return None
                time.sleep(2 ** attempt)

        sha = None
        if verify_sha256:
            print(f"  Computing SHA-256 for {dest.name} …", flush=True)
            sha = _sha256_file(dest)
            print(f"  SHA-256: {sha}", flush=True)

        if decompress or not self.keep_gz:
            dest = self._decompress(dest)

        self._register(
            iso_date   = iso_date,
            dest       = dest,
            ftp_name   = ftp_name,
            sha256     = sha,
            source_url = f"ftp://{_FTP_HOST}{_FTP_DIR}{ftp_name}",
        )
        return dest

    def download_range(
        self,
        start:        str,
        end:          str,
        skip_weekends: bool = True,
        **kwargs,
    ) -> List[Path]:
        """
        Download all ITCH files between `start` and `end` (inclusive).

        Parameters
        ----------
        start, end    : ISO-8601 dates.
        skip_weekends : skip Saturdays and Sundays (default True).
        **kwargs      : forwarded to :meth:`download`.

        Returns
        -------
        List of successfully downloaded Paths.
        """
        d0   = date.fromisoformat(start)
        d1   = date.fromisoformat(end)
        results: List[Path] = []
        d    = d0
        while d <= d1:
            if skip_weekends and d.weekday() >= 5:
                d += timedelta(days=1)
                continue
            path = self.download(d.isoformat(), **kwargs)
            if path:
                results.append(path)
            d += timedelta(days=1)
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ftp_download(self, ftp_name: str, dest: Path) -> None:
        """Stream-download `ftp_name` from the NASDAQ FTP to `dest`."""
        tmp = dest.with_suffix(".part")
        start_byte = tmp.stat().st_size if tmp.exists() else 0

        with ftplib.FTP(timeout=self.timeout) as ftp:
            ftp.connect(_FTP_HOST)
            ftp.login()
            ftp.cwd(_FTP_DIR)
            ftp.set_pasv(True)

            # Get file size for progress bar
            try:
                total = ftp.size(ftp_name)
            except ftplib.Error:
                total = 0

            progress = _progress_bar(ftp_name, total or 1)

            mode = "ab" if start_byte else "wb"
            with open(tmp, mode) as fh:
                def _write(data: bytes) -> None:
                    fh.write(data)
                    progress(len(data))

                if start_byte:
                    ftp.retrbinary(f"RETR {ftp_name}", _write,
                                   rest=start_byte)
                else:
                    ftp.retrbinary(f"RETR {ftp_name}", _write)

        tmp.rename(dest)
        log.info("Downloaded: %s  (%s bytes)", dest.name, dest.stat().st_size)

    def _decompress(self, gz_path: Path) -> Path:
        """Decompress a .gz file in-place; delete the .gz; return the raw path."""
        raw_path = gz_path.with_suffix("")
        print(f"  Decompressing {gz_path.name} → {raw_path.name} …", flush=True)
        with gzip.open(gz_path, "rb") as src, open(raw_path, "wb") as dst:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
        gz_path.unlink()
        print(f"  Done → {raw_path.stat().st_size / 1024**3:.2f} GB", flush=True)
        return raw_path

    def _register(
        self,
        iso_date:   str,
        dest:       Path,
        ftp_name:   str,
        sha256:     Optional[str],
        source_url: str,
    ) -> None:
        """Add the downloaded file to the catalog (if one is attached)."""
        if self.catalog is None:
            return
        try:
            from data.data_catalog import CatalogEntry
        except ImportError:
            return

        entry = CatalogEntry(
            data_type   = "itch",
            date        = iso_date,
            filename    = dest.name,
            path        = str(dest.resolve()),
            size_bytes  = dest.stat().st_size,
            sha256      = sha256,
            source_url  = source_url,
            download_ts = CatalogEntry.now_ts(),
            tags        = "nasdaq,itch5",
        )
        self.catalog.add(entry, replace=True)
        log.info("Registered in catalog: %s", dest.name)


# ── Type alias ────────────────────────────────────────────────────────────────
from typing import Union


# ── CLI entry point ───────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog        = "download_itch",
        description = "Download NASDAQ TotalView-ITCH 5.0 historical files",
    )
    parser.add_argument("--dest",   default="raw/itch",
                        help="Local destination directory (default: raw/itch)")
    parser.add_argument("--list",   action="store_true",
                        help="List all available files on the FTP server")
    parser.add_argument("--date",   metavar="YYYY-MM-DD",
                        help="Download a single date")
    parser.add_argument("--from",   dest="date_from", metavar="YYYY-MM-DD",
                        help="Start of date range")
    parser.add_argument("--to",     dest="date_to",   metavar="YYYY-MM-DD",
                        help="End of date range")
    parser.add_argument("--decompress",  action="store_true",
                        help="Decompress .gz files after download")
    parser.add_argument("--verify-sha",  action="store_true",
                        help="Compute and log SHA-256 checksum after download")
    parser.add_argument("--force",  action="store_true",
                        help="Re-download even if file exists locally")
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

    dl = ItchDownloader(
        dest_dir = args.dest,
        catalog  = catalog,
    )

    if args.list:
        dl.list_available()
        return

    if args.date:
        path = dl.download(
            args.date,
            force         = args.force,
            verify_sha256 = args.verify_sha,
            decompress    = args.decompress,
        )
        if path:
            print(f"\n  ✓  {path}")
        else:
            print(f"\n  ✗  Download failed for {args.date}", file=sys.stderr)
            sys.exit(1)

    elif args.date_from and args.date_to:
        paths = dl.download_range(
            args.date_from,
            args.date_to,
            force         = args.force,
            verify_sha256 = args.verify_sha,
            decompress    = args.decompress,
        )
        print(f"\n  ✓  {len(paths)} file(s) downloaded.")

    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
