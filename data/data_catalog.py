"""
Data Catalog
=============
SQLite-backed registry of every dataset downloaded to the local store.
Supports ITCH binary files, daily OHLCV Parquet files, and arbitrary
user-registered datasets.

Schema
------
  entries
    id            INTEGER PRIMARY KEY
    data_type     TEXT     NOT NULL  -- "itch" | "daily" | "custom"
    date          TEXT     NOT NULL  -- ISO-8601 date the data *represents*
    symbol        TEXT              -- ticker (daily / custom) or NULL (itch)
    filename      TEXT     NOT NULL UNIQUE
    path          TEXT     NOT NULL  -- absolute local path
    size_bytes    INTEGER
    sha256        TEXT              -- hex digest, NULL if not verified
    source_url    TEXT              -- original download URL
    download_ts   TEXT     NOT NULL  -- ISO-8601 datetime of download
    tags          TEXT              -- comma-separated free-form tags
    notes         TEXT

Usage
-----
>>> catalog = DataCatalog("sigma-edge/data/catalog.db")
>>> catalog.add(entry)
>>> results = catalog.find(data_type="itch", date="2024-01-15")
>>> catalog.status()
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union


# ── Entry ─────────────────────────────────────────────────────────────────────

@dataclass
class CatalogEntry:
    """One row in the catalog."""
    data_type:   str
    date:        str          # "YYYY-MM-DD"
    filename:    str
    path:        str
    download_ts: str          # "YYYY-MM-DDTHH:MM:SS"
    symbol:      Optional[str]  = None
    size_bytes:  Optional[int]  = None
    sha256:      Optional[str]  = None
    source_url:  Optional[str]  = None
    tags:        Optional[str]  = None
    notes:       Optional[str]  = None
    id:          Optional[int]  = field(default=None, repr=False)

    @classmethod
    def now_ts(cls) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    def exists(self) -> bool:
        """True if the file is present on disk."""
        return os.path.isfile(self.path)

    def verify_size(self) -> bool:
        """True if on-disk size matches recorded size_bytes."""
        if self.size_bytes is None:
            return True
        try:
            return os.path.getsize(self.path) == self.size_bytes
        except OSError:
            return False

    def compute_sha256(self) -> str:
        """Compute and return the SHA-256 hex digest of the file."""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def __repr__(self) -> str:
        return (f"CatalogEntry({self.data_type}/{self.date}"
                f"{' ' + self.symbol if self.symbol else ''}"
                f", {self.filename})")


# ── DataCatalog ────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type   TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    symbol      TEXT,
    filename    TEXT    NOT NULL UNIQUE,
    path        TEXT    NOT NULL,
    size_bytes  INTEGER,
    sha256      TEXT,
    source_url  TEXT,
    download_ts TEXT    NOT NULL,
    tags        TEXT,
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_type_date ON entries (data_type, date);
CREATE INDEX IF NOT EXISTS idx_entries_symbol    ON entries (symbol);
"""


class DataCatalog:
    """
    SQLite-backed catalog of downloaded market data.

    Parameters
    ----------
    db_path : path to the SQLite database file.
              Defaults to ``{project_root}/data/catalog.db``.

    Usage
    -----
    >>> cat = DataCatalog()
    >>> cat.add(CatalogEntry(data_type="itch", date="2024-01-15",
    ...         filename="20240115.NASDAQ_ITCH50.gz",
    ...         path="/data/raw/itch/20240115.NASDAQ_ITCH50.gz",
    ...         download_ts=CatalogEntry.now_ts()))
    >>> for e in cat.find(data_type="itch"):
    ...     print(e)
    >>> print(cat.status())
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            db_path = Path(__file__).parent / "catalog.db"
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_SQL)
        self._conn.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, entry: CatalogEntry, replace: bool = False) -> int:
        """
        Insert a new entry (or replace if `replace=True`).

        Returns the row ``id``.
        """
        op = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        cur = self._conn.execute(
            f"""
            {op} INTO entries
              (data_type, date, symbol, filename, path,
               size_bytes, sha256, source_url, download_ts, tags, notes)
            VALUES
              (:data_type, :date, :symbol, :filename, :path,
               :size_bytes, :sha256, :source_url, :download_ts, :tags, :notes)
            """,
            {k: v for k, v in asdict(entry).items() if k != "id"},
        )
        self._conn.commit()
        return cur.lastrowid

    def remove(self, filename: str) -> bool:
        """Delete by filename.  Returns True if a row was deleted."""
        cur = self._conn.execute(
            "DELETE FROM entries WHERE filename = ?", (filename,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, filename: str) -> Optional[CatalogEntry]:
        """Return entry by filename, or None."""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE filename = ?", (filename,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def update_sha256(self, filename: str, sha256: str) -> None:
        """Persist a verified checksum."""
        self._conn.execute(
            "UPDATE entries SET sha256 = ? WHERE filename = ?",
            (sha256, filename),
        )
        self._conn.commit()

    # ── Query ─────────────────────────────────────────────────────────────────

    def find(
        self,
        data_type: Optional[str] = None,
        date:      Optional[str] = None,
        symbol:    Optional[str] = None,
        tags:      Optional[str] = None,
    ) -> List[CatalogEntry]:
        """
        Return entries matching all non-None filters.

        Parameters
        ----------
        data_type : "itch" | "daily" | "custom" | None (any).
        date      : exact ISO-8601 date string, e.g. "2024-01-15".
        symbol    : ticker symbol, e.g. "AAPL".
        tags      : LIKE pattern matched against the tags column.
        """
        clauses: List[str] = []
        params:  List     = []

        if data_type is not None:
            clauses.append("data_type = ?")
            params.append(data_type)
        if date is not None:
            clauses.append("date = ?")
            params.append(date)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if tags is not None:
            clauses.append("tags LIKE ?")
            params.append(f"%{tags}%")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows  = self._conn.execute(
            f"SELECT * FROM entries {where} ORDER BY date, filename",
            params,
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def dates(self, data_type: str = "itch") -> List[str]:
        """Return sorted list of all dates present for `data_type`."""
        rows = self._conn.execute(
            "SELECT DISTINCT date FROM entries WHERE data_type=? ORDER BY date",
            (data_type,),
        ).fetchall()
        return [r[0] for r in rows]

    def symbols(self) -> List[str]:
        """Return sorted list of all distinct symbols (daily data)."""
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM entries "
            "WHERE symbol IS NOT NULL ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]

    def __iter__(self) -> Iterator[CatalogEntry]:
        rows = self._conn.execute(
            "SELECT * FROM entries ORDER BY data_type, date, filename"
        ).fetchall()
        return iter(_row_to_entry(r) for r in rows)

    def __len__(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM entries"
        ).fetchone()[0]

    # ── Integrity checks ──────────────────────────────────────────────────────

    def verify(self, recompute_sha256: bool = False) -> Dict[str, List[str]]:
        """
        Scan every entry for problems.

        Returns a dict with keys "missing", "size_mismatch", "hash_mismatch".
        Pass ``recompute_sha256=True`` to recheck hashes (slow for large files).
        """
        issues: Dict[str, List[str]] = {
            "missing":        [],
            "size_mismatch":  [],
            "hash_mismatch":  [],
        }
        for entry in self:
            if not entry.exists():
                issues["missing"].append(entry.filename)
                continue
            if not entry.verify_size():
                issues["size_mismatch"].append(entry.filename)
            if recompute_sha256 and entry.sha256:
                actual = entry.compute_sha256()
                if actual != entry.sha256:
                    issues["hash_mismatch"].append(entry.filename)
        return issues

    # ── Status summary ────────────────────────────────────────────────────────

    def status(self) -> str:
        """Return a human-readable status summary string."""
        rows = self._conn.execute(
            """
            SELECT  data_type,
                    COUNT(*)        AS n,
                    COALESCE(SUM(size_bytes), 0) AS total_bytes,
                    MIN(date)       AS earliest,
                    MAX(date)       AS latest
            FROM    entries
            GROUP   BY data_type
            ORDER   BY data_type
            """
        ).fetchall()

        if not rows:
            return "Catalog is empty."

        lines = [
            "══════════════════════════════════════════════════",
            "  Data Catalog Status",
            "══════════════════════════════════════════════════",
        ]
        for r in rows:
            gb = r["total_bytes"] / 1024 ** 3
            lines += [
                f"  {r['data_type']:<10}  {r['n']:>5} files  "
                f"{gb:>7.2f} GB  "
                f"{r['earliest']} → {r['latest']}",
            ]
        lines.append(f"  DB location: {self._db}")
        lines.append("══════════════════════════════════════════════════")
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DataCatalog":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_entry(row: sqlite3.Row) -> CatalogEntry:
    return CatalogEntry(
        id          = row["id"],
        data_type   = row["data_type"],
        date        = row["date"],
        symbol      = row["symbol"],
        filename    = row["filename"],
        path        = row["path"],
        size_bytes  = row["size_bytes"],
        sha256      = row["sha256"],
        source_url  = row["source_url"],
        download_ts = row["download_ts"],
        tags        = row["tags"],
        notes       = row["notes"],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog        = "data_catalog",
        description = "Inspect and manage the Sigma Edge data catalog",
    )
    parser.add_argument("--db", default=None, help="Path to catalog.db")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status",  help="Print summary statistics")
    sub.add_parser("list",    help="List all entries")
    sub.add_parser("verify",  help="Check all files exist and sizes match")

    find_p = sub.add_parser("find", help="Query entries")
    find_p.add_argument("--type",   dest="data_type")
    find_p.add_argument("--date")
    find_p.add_argument("--symbol")

    rm_p = sub.add_parser("remove", help="Delete an entry by filename")
    rm_p.add_argument("filename")

    args = parser.parse_args()

    with DataCatalog(args.db) as cat:
        if args.cmd == "status" or args.cmd is None:
            print(cat.status())

        elif args.cmd == "list":
            for e in cat:
                status = "✓" if e.exists() else "✗"
                print(f"  [{status}] {e.data_type:<8} {e.date}  {e.filename}")

        elif args.cmd == "find":
            results = cat.find(
                data_type = getattr(args, "data_type", None),
                date      = getattr(args, "date", None),
                symbol    = getattr(args, "symbol", None),
            )
            for e in results:
                print(e)

        elif args.cmd == "verify":
            issues = cat.verify()
            any_issue = any(issues.values())
            if not any_issue:
                print("✓  All files present and sizes match.")
            for kind, files in issues.items():
                for f in files:
                    print(f"  [{kind}] {f}")

        elif args.cmd == "remove":
            ok = cat.remove(args.filename)
            print("Removed." if ok else "Not found.")


if __name__ == "__main__":
    _main()
