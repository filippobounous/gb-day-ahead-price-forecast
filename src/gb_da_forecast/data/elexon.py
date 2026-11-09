"""Elexon Insights Solution API (BMRS). Free, no key. Docs: https://bmrs.elexon.co.uk/api-documentation

Datasets:
    ndf            National Demand Forecast, half-hourly. Two kinds of publication share this
                   dataset: a vintage every 30 minutes covering from now until ~05:00 UK tomorrow,
                   and a day-ahead release at ~08:45 UK covering 05:00 tomorrow to 05:00 the day
                   after. Every vintage is kept; ``asof.as_of_cutoff`` stitches the two together
                   per period, as known at the 09:00 cutoff.
    windfor        Forecast of transmission-connected wind generation, hourly, ~8 vintages a day,
                   each covering roughly the next two days.
    fuelhh         Outturn generation by fuel type, including interconnector flows (INT*). Quirk:
                   in 2018-2022 the last period of each day carries the next day's settlementDate,
                   so always use ``startTime``, never (settlementDate, settlementPeriod).
    system_prices  Imbalance (cash-out) price per settlement period; buy and sell prices are equal.
                   The API returns the latest settlement run: since 2024 ``createdDateTime`` is
                   ~25 h after the period, so only periods older than that are safe as features.
    mid            Market Index Data: volume-weighted price of short-term exchange trades per
                   settlement period and exchange (APXMIDP = EPEX). The EPEX day-ahead auction is
                   reportedly not included, so this mostly reflects later trading.

All timestamps are UTC. ``startTime`` is the start of the settlement period (or hour).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ._http import get_json
from ._store import Chunk, combine, download, make_chunks

BASE = "https://data.elexon.co.uk/bmrs/api/v1"


def _get(path: str, **params) -> pd.DataFrame:
    payload = get_json(f"{BASE}{path}", params={**params, "format": "json"})
    return pd.DataFrame(payload["data"] if isinstance(payload, dict) else payload)


def _to_utc(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _vintages_published_on(path: str):
    """Fetcher for forecast datasets: every vintage published on one UTC day (the API's max window)."""

    def fetch(chunk: Chunk) -> pd.DataFrame:
        day = chunk.start.isoformat()
        df = _get(path, publishDateTimeFrom=f"{day}T00:00Z", publishDateTimeTo=f"{day}T23:59Z")
        return _to_utc(df, "publishTime", "startTime")

    return fetch


def _fetch_fuelhh(chunk: Chunk) -> pd.DataFrame:
    df = _get(
        "/datasets/FUELHH",
        settlementDateFrom=chunk.start.isoformat(),
        settlementDateTo=chunk.end.isoformat(),
    )
    return _to_utc(df, "publishTime", "startTime")


def _fetch_system_prices(chunk: Chunk) -> pd.DataFrame:
    df = _get(f"/balancing/settlement/system-prices/{chunk.start.isoformat()}")
    return _to_utc(df, "startTime", "createdDateTime")


def _fetch_mid(chunk: Chunk) -> pd.DataFrame:
    df = _get("/datasets/MID", **{"from": f"{chunk.start}T00:00Z", "to": f"{chunk.end}T23:30Z"})
    return _to_utc(df, "startTime")


def collect_ndf(start: dt.date, end: dt.date, workers: int = 4):
    failures = download("ndf", make_chunks(start, end, 1), _vintages_published_on("/datasets/NDF"), workers)
    return combine("ndf"), failures


def collect_windfor(start: dt.date, end: dt.date, workers: int = 4):
    failures = download("windfor", make_chunks(start, end, 1), _vintages_published_on("/datasets/WINDFOR"), workers)
    return combine("windfor"), failures


def collect_fuelhh(start: dt.date, end: dt.date, workers: int = 4):
    failures = download("fuelhh", make_chunks(start, end, 7), _fetch_fuelhh, workers)  # API max: 7 days
    return combine("fuelhh"), failures


def collect_system_prices(start: dt.date, end: dt.date, workers: int = 4):
    failures = download("system_prices", make_chunks(start, end, 1), _fetch_system_prices, workers)
    return combine("system_prices"), failures


def collect_mid(start: dt.date, end: dt.date, workers: int = 4):
    failures = download("mid", make_chunks(start, end, 7), _fetch_mid, workers)  # API max: 7 days
    return combine("mid"), failures
