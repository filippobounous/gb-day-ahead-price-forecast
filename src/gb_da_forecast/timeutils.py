"""GB market time conventions.

A GB settlement day runs from 00:00 to 24:00 UK local time in 30-minute settlement periods
numbered from 1. Because of daylight saving it has 46 periods on the day clocks go forward and
50 on the day they go back, so (date, period) must be converted via local midnight rather than
by assuming period 1 starts at 00:00 UTC.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import UK_TZ

# The forecast for delivery day D is made at 09:00 UK time on D-1: 20 minutes before the EPEX DA1
# gate closure (09:20), leaving time to turn it into bids. Only data published by then may be used.
FORECAST_TIME_LOCAL = dt.time(9, 0)


def forecast_cutoff(delivery_dates: pd.Series) -> pd.Series:
    """UTC time at which the forecast for each delivery day is made (09:00 UK time the day before)."""
    local = pd.to_datetime(delivery_dates) - pd.Timedelta(days=1) + pd.Timedelta(
        hours=FORECAST_TIME_LOCAL.hour, minutes=FORECAST_TIME_LOCAL.minute
    )
    return local.dt.tz_localize(UK_TZ).dt.tz_convert("UTC")


def delivery_day_bounds(delivery_dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """UTC [start, end) of each auction delivery day.

    GB day-ahead auction delivery day D runs from 23:00 UK time on D-1 to 23:00 UK time on D
    (i.e. the CET calendar day), as the N2EX data shows: the first delivery day (21 Sep 2021)
    starts at 22:00 UTC on 20 Sep. It is 23, 24 or 25 hours long.
    """
    days = pd.to_datetime(delivery_dates).dt.tz_localize(None).dt.normalize()
    start = (days - pd.Timedelta(hours=1)).dt.tz_localize(UK_TZ).dt.tz_convert("UTC")
    end = (days + pd.Timedelta(hours=23)).dt.tz_localize(UK_TZ).dt.tz_convert("UTC")
    return start, end


def delivery_day_of(start_utc: pd.Series) -> pd.Series:
    """Delivery day (a date) that each period belongs to: 23:00 UK on D-1 up to 23:00 UK on D -> D."""
    local = pd.to_datetime(start_utc, utc=True).dt.tz_convert(UK_TZ)
    return (local + pd.Timedelta(hours=1)).dt.date


def settlement_period_start(settlement_date: pd.Series, settlement_period: pd.Series) -> pd.Series:
    """UTC start time of each (settlement date, settlement period) pair."""
    dates = pd.to_datetime(settlement_date).dt.tz_localize(None).dt.normalize()
    unique = pd.Series(dates.unique())
    # UK clocks change at 01:00/02:00, so local midnight is never ambiguous or missing.
    midnight_utc = dict(zip(unique, unique.dt.tz_localize(UK_TZ).dt.tz_convert("UTC")))
    start = dates.map(midnight_utc)
    return start + pd.to_timedelta((settlement_period.astype(int) - 1) * 30, unit="min")
