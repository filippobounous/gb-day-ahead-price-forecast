"""Point-in-time ("as-of") selection of forecasts: the main guard against look-ahead.

The forecast for delivery day D is made at 09:00 UK time on D-1 (``timeutils.forecast_cutoff``),
so every input must be a value that had been published by then.

Forecast datasets publish many vintages with different horizons, so "take the latest vintage
before the cutoff" is not enough. Elexon's national demand forecast, for example, has a
half-hourly vintage that only runs until ~05:00 tomorrow, plus a separate 08:45 release covering
05:00 tomorrow to 05:00 the day after; the first six hours of delivery day D are only in the
former. So for every target period we take the latest value published at or before the cutoff of
the delivery day that period belongs to.
"""

from __future__ import annotations

import pandas as pd

from .timeutils import delivery_day_of, forecast_cutoff


def as_of_cutoff(
    df: pd.DataFrame,
    publish_col: str,
    time_col: str,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Keep, for every target period (and group in ``by``), the latest row published by its cutoff.

    Adds ``delivery_day``, ``cutoff`` and ``age`` (how old the chosen value was at the cutoff) so
    stale fallbacks can be spotted or excluded.
    """
    times = pd.Series(df[time_col].unique())
    day_of = dict(zip(times, delivery_day_of(times)))
    days = pd.Series(sorted(set(day_of.values())))
    cutoff_of = dict(zip(days, forecast_cutoff(pd.Series(pd.to_datetime(days)))))

    out = df.assign(delivery_day=df[time_col].map(day_of))
    out["cutoff"] = out["delivery_day"].map(cutoff_of)
    out = out[out[publish_col] <= out["cutoff"]]
    out = out.sort_values(publish_col).drop_duplicates([*(by or []), time_col], keep="last")
    out["age"] = out["cutoff"] - out[publish_col]
    return out.sort_values([*(by or []), time_col], ignore_index=True)
