"""Resumable, chunked downloads cached as parquet files.

Each source is downloaded in chunks (a day, a week, a month or a year). A chunk is saved as
``data/raw/<source>/chunks/<key>.parquet``, so an interrupted run resumes where it stopped and a
re-run only fetches what is missing.

When is a cached chunk trusted? Only once it is *final*: it covers its whole grid cell and it was
fetched more than FRESH_DAYS after the cell ended (judged by the file's modification time). A
chunk fetched earlier may have been empty or incomplete (e.g. tomorrow's forecasts, or today's
outturn), so every run fetches it again until it becomes final.

Chunk boundaries sit on a fixed calendar grid, so their keys do not depend on the requested date
range and re-runs with different ranges never create overlapping chunks. A chunk always starts at
the start of its grid cell (fetching slightly more than asked for); a chunk cut short by the end
date is saved as ``<key>_partial`` and superseded once the complete chunk exists.

After downloading, all chunks of a source are merged into ``data/raw/<source>.parquet``; that
merged file is what the rest of the code reads (see ``load``).
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import RAW_DIR

GRID_ANCHOR = dt.date(2018, 1, 1)  # a Monday; day/week chunks are aligned to it
FRESH_DAYS = 3  # data is assumed not to change once its period ended more than this many days ago


@dataclass(frozen=True)
class Chunk:
    key: str
    start: dt.date  # first date to request (inclusive) = start of the grid cell
    end: dt.date  # last date to request (inclusive) = end of the grid cell, unless cut short
    grid_end: dt.date  # end of the chunk's grid cell
    extra: dict = field(default_factory=dict, hash=False, compare=False)  # e.g. {"zone": "FR"}

    @property
    def complete(self) -> bool:
        return self.end == self.grid_end

    @property
    def filename(self) -> str:
        return f"{self.key}.parquet" if self.complete else f"{self.key}_partial.parquet"

    def is_final(self, fetched_on: dt.date) -> bool:
        """Whether a copy fetched on ``fetched_on`` can be trusted forever (see module docstring)."""
        return self.complete and fetched_on > self.grid_end + dt.timedelta(days=FRESH_DAYS)


def make_chunks(start: dt.date, end: dt.date, freq: str | int, prefix: str = "", **extra) -> list[Chunk]:
    """Split [start, end] into grid-aligned chunks.

    ``freq`` is a number of days (aligned to GRID_ANCHOR), "month" or "year". Every chunk starts
    at its grid cell's start (even if that is before ``start``); only the end is clipped to ``end``.
    """
    cells: list[tuple[dt.date, dt.date]] = []
    if freq == "year":
        cells = [(dt.date(y, 1, 1), dt.date(y, 12, 31)) for y in range(start.year, end.year + 1)]
    elif freq == "month":
        y, m = start.year, start.month
        while dt.date(y, m, 1) <= end:
            nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
            cells.append((dt.date(y, m, 1), nxt - dt.timedelta(days=1)))
            y, m = nxt.year, nxt.month
    else:
        days = int(freq)
        cell_start = GRID_ANCHOR + dt.timedelta(days=((start - GRID_ANCHOR).days // days) * days)
        while cell_start <= end:
            cells.append((cell_start, cell_start + dt.timedelta(days=days - 1)))
            cell_start += dt.timedelta(days=days)

    return [
        Chunk(
            key=f"{prefix}{cell_start.isoformat()}",
            start=cell_start,
            end=min(cell_end, end),
            grid_end=cell_end,
            extra=extra,
        )
        for cell_start, cell_end in cells
    ]


def _fetched_on(path: Path) -> dt.date | None:
    return dt.date.fromtimestamp(path.stat().st_mtime) if path.exists() else None


def download(
    source: str,
    chunks: list[Chunk],
    fetch: Callable[[Chunk], pd.DataFrame],
    workers: int = 4,
) -> list[tuple[str, str]]:
    """Fetch every chunk that is not yet final. Returns a list of (chunk key, error)."""
    folder = RAW_DIR / source / "chunks"
    folder.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    todo = [c for c in chunks if not ((d := _fetched_on(folder / c.filename)) and c.is_final(d))]
    print(f"[{source}] {len(chunks)} chunks, {len(chunks) - len(todo)} cached, {len(todo)} to fetch", flush=True)

    def run(chunk: Chunk) -> int:
        df = fetch(chunk)
        tmp = folder / f"{chunk.filename}.tmp"
        df.reset_index(drop=True).to_parquet(tmp, index=False)
        tmp.replace(folder / chunk.filename)  # atomic, so a crash never leaves half a file
        return len(df)

    failures: list[tuple[str, str]] = []
    empty_past: list[str] = []  # empty although the period is long over: a source gap (or an API hiccup)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, c): c for c in todo}
        for i, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            try:
                if future.result() == 0 and chunk.grid_end < today - dt.timedelta(days=FRESH_DAYS):
                    empty_past.append(chunk.key)
            except Exception as exc:  # keep going; failed chunks are retried on the next run
                failures.append((chunk.key, repr(exc)[:300]))
            if i % 100 == 0 or i == len(todo):
                print(f"[{source}] {i}/{len(todo)} fetched, {len(failures)} failed", flush=True)
    if empty_past:
        print(f"[{source}] {len(empty_past)} past chunks came back empty: {sorted(empty_past)[:10]}", flush=True)
    return failures


def combine(source: str) -> pd.DataFrame:
    """Merge all cached chunks of a source into data/raw/<source>.parquet and return it."""
    folder = RAW_DIR / source / "chunks"
    files = sorted(folder.glob("*.parquet"))
    # A partial chunk is superseded once the complete version of the same chunk exists.
    files = [
        p for p in files
        if not (p.stem.endswith("_partial") and (folder / f"{p.stem.removesuffix('_partial')}.parquet").exists())
    ]
    frames = [f for f in (pd.read_parquet(p) for p in files) if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = df.drop_duplicates(ignore_index=True)
    save(source, df)
    return df


def save(source: str, df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DIR / f"{source}.parquet", index=False)


def load(source: str) -> pd.DataFrame:
    """Read a merged raw dataset, e.g. ``load("n2ex")``."""
    return pd.read_parquet(RAW_DIR / f"{source}.parquet")
