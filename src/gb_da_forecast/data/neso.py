"""NESO data portal (CKAN API). Free, no key. https://www.neso.energy/data-portal

Datasets:
    n2ex      N2EX (Nord Pool) GB hourly day-ahead auction price in £/MWh, from 2021-09-20. This
              is the forecasting target, used as an open proxy for EPEX DA1 (see README).
              NESO publishes it as (date, "HH:00 - HH:00") labelled in UTC: every date has 24
              rows, including clock-change days, and hour-to-hour price changes line up with
              Elexon's UTC-stamped EPEX index at zero shift in both summer and winter
              (``python -m gb_da_forecast.data.check``). Stored with ``start_utc`` and
              ``delivery_day``; a delivery day runs 23:00-23:00 UK time, so it spans two UTC dates.
    embedded  Forecasts of wind and solar generation connected to distribution networks. NESO
              cannot see this generation, so it lowers the demand NESO has to meet. A new vintage
              is published every hour covering ~14 days ahead. Its ``Forecast_Datetime`` is in UK
              local time (in summer each vintage's first period starts ~72 min before the stamp,
              in winter ~12 min; the stamps also skip 01:xx on the spring clock-change night), so it
              is converted to UTC. Only vintages stamped 04:00-10:59 UK time that target the next
              two days are kept, which brackets the 09:00 UK forecast cutoff.
"""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from ..config import UK_TZ
from ..timeutils import delivery_day_of, settlement_period_start
from ._http import get_json
from ._store import Chunk, combine, download, make_chunks

API = "https://api.neso.energy/api/3/action"
N2EX_RESOURCE = "4f27eea5-7038-4f73-9740-e3e4ad47c26a"  # "N2EX GB Day-Ahead Price"
EMBEDDED_PACKAGE = "embedded-wind-and-solar-forecasts"


def _sql(query: str) -> pd.DataFrame:
    result = get_json(f"{API}/datastore_search_sql", params={"sql": query})["result"]
    return pd.DataFrame(result["records"])


def _fetch_n2ex(chunk: Chunk) -> pd.DataFrame:
    df = _sql(
        f'SELECT "Date", "Delivery Period", "Price" FROM "{N2EX_RESOURCE}" '
        f"WHERE \"Date\" >= '{chunk.start}' AND \"Date\" <= '{chunk.end}'"
    )
    if df.empty:
        return df
    utc_date = pd.to_datetime(df["Date"])
    utc_hour = df["Delivery Period"].str[:2].astype(int)
    out = pd.DataFrame({"start_utc": (utc_date + pd.to_timedelta(utc_hour, unit="h")).dt.tz_localize("UTC")})
    out["delivery_day"] = delivery_day_of(out["start_utc"])
    out["price_gbp_mwh"] = df["Price"].astype(float)
    out["source_label"] = df["Date"].astype(str).str[:10] + " " + df["Delivery Period"]  # as published
    return out


def collect_n2ex(start: dt.date, end: dt.date, workers: int = 2):
    failures = download("n2ex", make_chunks(start, end, "year"), _fetch_n2ex, workers)
    df = combine("n2ex").sort_values("start_utc", ignore_index=True)
    return df, failures


def _embedded_archives() -> dict[int, str]:
    """Map year -> resource id of NESO's yearly 'Embedded Solar and Wind Forecast Archive'."""
    package = get_json(f"{API}/package_show", params={"id": EMBEDDED_PACKAGE})["result"]
    archives = {}
    for resource in package["resources"]:
        match = re.search(r"Archive (\d{4})", resource.get("name") or "")
        if match and resource.get("datastore_active"):
            archives[int(match.group(1))] = resource["id"]
    return archives


def collect_embedded(start: dt.date, end: dt.date, workers: int = 2):
    archives = _embedded_archives()

    def fetch(chunk: Chunk) -> pd.DataFrame:
        resource = archives[chunk.start.year]  # monthly chunks never span two years
        df = _sql(
            'SELECT "Forecast_Datetime", "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", '
            '"EMBEDDED_WIND_FORECAST", "EMBEDDED_WIND_CAPACITY", '
            '"EMBEDDED_SOLAR_FORECAST", "EMBEDDED_SOLAR_CAPACITY" '
            f'FROM "{resource}" '
            f"WHERE \"Forecast_Datetime\" >= '{chunk.start}' "
            f"AND \"Forecast_Datetime\" < '{chunk.end + dt.timedelta(days=1)}' "
            "AND date_part('hour', \"Forecast_Datetime\"::timestamp) BETWEEN 4 AND 10 "
            'AND "SETTLEMENT_DATE"::date <= "Forecast_Datetime"::date + 2'
        )
        if df.empty:
            return df
        df.columns = df.columns.str.lower()
        # UK local time -> UTC. Stamps between 04:00 and 10:59 are never ambiguous or missing.
        df["forecast_time"] = pd.to_datetime(df.pop("forecast_datetime")).dt.tz_localize(UK_TZ).dt.tz_convert("UTC")
        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
        df["start_utc"] = settlement_period_start(df["settlement_date"], df["settlement_period"])
        value_cols = [c for c in df.columns if c.startswith("embedded_")]
        df[value_cols] = df[value_cols].astype(float)
        return df

    failures = download("embedded", make_chunks(start, end, "month"), fetch, workers)
    return combine("embedded"), failures
