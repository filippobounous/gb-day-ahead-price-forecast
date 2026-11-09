"""Sanity checks on the collected raw data.

    python -m gb_da_forecast.data.check

Answers the questions the model depends on:
  1. Is the target complete (23/24/25 hourly prices per delivery day)?
  2. Are the N2EX hour labels really UTC? Hour-to-hour price changes are compared with Elexon's
     UTC-stamped EPEX market index at shifts of -1, 0 and +1 hour. If the labels were UK local
     time, the best shift in summer would be one hour off.
  3. For each delivery day, how many of its periods have a demand / wind / embedded forecast
     published by the 09:00 UK cutoff on D-1 (using the same as-of rule as the features)?
  4. Where are the gaps in every other source?
"""

from __future__ import annotations

import pandas as pd

from ..asof import as_of_cutoff
from ..timeutils import delivery_day_bounds
from ._store import load


def _expected_periods(days: pd.Series, minutes: int) -> pd.Series:
    start, end = delivery_day_bounds(pd.Series(pd.to_datetime(days)))
    return ((end - start) / pd.Timedelta(minutes=minutes)).astype(int).set_axis(days)


def _target(n2ex: pd.DataFrame) -> pd.Series:
    per_day = n2ex.groupby("delivery_day").size()
    expected = _expected_periods(pd.Series(per_day.index), 60)
    incomplete = per_day[per_day != expected]
    print(f"N2EX: {len(n2ex):,} hourly prices, delivery days {per_day.index.min()} -> {per_day.index.max()}")
    print(f"  incomplete delivery days: {len(incomplete)} {incomplete.to_dict()}")
    all_days = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D").date
    print(f"  missing delivery days: {len(set(all_days) - set(per_day.index))}")
    p = n2ex["price_gbp_mwh"]
    print(f"  price £/MWh: mean {p.mean():.1f}, min {p.min():.1f}, max {p.max():.1f}, negative hours {(p < 0).sum()}")
    yearly = n2ex.groupby(n2ex["start_utc"].dt.year)["price_gbp_mwh"].agg(["mean", "std"]).round(1)
    print("  yearly mean / std:", {y: tuple(r) for y, r in yearly.iterrows()})
    return per_day


def _n2ex_time_basis(n2ex: pd.DataFrame, mid: pd.DataFrame) -> None:
    epex = mid[(mid["dataProvider"] == "APXMIDP") & (mid["volume"].astype(float) > 0)]
    epex = epex.set_index("startTime")["price"].astype(float).sort_index().resample("1h").mean()
    n2 = n2ex.set_index("start_utc")["price_gbp_mwh"].sort_index()
    bst = n2.index.tz_convert("Europe/London").map(lambda t: t.dst() != pd.Timedelta(0))
    print("N2EX label check: corr of hour-to-hour changes with the EPEX index, by shift of the N2EX labels")
    for season, mask in {"summer (BST)": bst, "winter (GMT)": ~bst}.items():
        cells = []
        for shift_h in (-1, 0, 1):
            s = n2[mask].copy()
            s.index = s.index + pd.Timedelta(hours=shift_h)
            j = pd.concat([s.diff().rename("n2ex"), epex.diff().rename("epex")], axis=1, join="inner").dropna()
            cells.append(f"{shift_h:+d}h: {j.corr().iloc[0, 1]:.3f}")
        print(f"  {season:13s} " + "   ".join(cells) + "   (0h best => labels are UTC)")


def _forecast_coverage(name: str, df: pd.DataFrame, publish_col: str, time_col: str, minutes: int, days) -> None:
    chosen = as_of_cutoff(df, publish_col, time_col)
    chosen = chosen[chosen["delivery_day"].isin(set(days))]
    got = chosen.groupby("delivery_day").size().reindex(days, fill_value=0)
    expected = _expected_periods(pd.Series(days), minutes)
    full = got.values == expected.values
    stale = chosen["age"] > pd.Timedelta(hours=24)
    print(
        f"  {name:9s} full coverage on {full.sum():,}/{len(days):,} delivery days; "
        f"value age at cutoff: median {chosen['age'].median()}, {stale.mean():.2%} older than 24h"
    )
    short = got[~full]
    if len(short):
        print(f"            short days (periods found): {dict(list(short.items())[:8])}")


def _gaps(name: str, times: pd.Series, freq: str) -> None:
    t = pd.Series(pd.to_datetime(times, utc=True).unique()).sort_values()
    expected = pd.date_range(t.min(), t.max(), freq=freq)
    missing = expected.difference(pd.DatetimeIndex(t))
    longest = t.diff().max()
    print(
        f"  {name:24s} {t.min():%Y-%m-%d %H:%M} -> {t.max():%Y-%m-%d %H:%M}  "
        f"missing {len(missing):>6,} of {len(expected):>9,} {freq:>5s} steps; longest jump {longest}"
    )


def _empty_publish_days(name: str, df: pd.DataFrame, publish_col: str, since) -> None:
    days = set(df[publish_col].dt.tz_convert("Europe/London").dt.date)
    span = pd.date_range(since, max(days), freq="D").date
    empty = sorted(set(span) - days)
    print(f"  {name:24s} publish days with no vintage since {since}: {len(empty)} {[str(d) for d in empty[:8]]}")


def main() -> None:
    n2ex, mid = load("n2ex"), load("mid")
    per_day = _target(n2ex)
    days = list(per_day.index)
    print()
    _n2ex_time_basis(n2ex, mid)

    print("\nForecast coverage at 09:00 UK time on D-1 (latest value per period published by the cutoff):")
    ndf, windfor, embedded = load("ndf"), load("windfor"), load("embedded")
    _forecast_coverage("ndf", ndf, "publishTime", "startTime", 30, days)
    _forecast_coverage("windfor", windfor, "publishTime", "startTime", 60, days)
    _forecast_coverage("embedded", embedded, "forecast_time", "start_utc", 30, days)
    target_start = str(days[0] - pd.Timedelta(days=1))
    _empty_publish_days("ndf", ndf, "publishTime", target_start)
    _empty_publish_days("windfor", windfor, "publishTime", target_start)
    _empty_publish_days("embedded", embedded, "forecast_time", target_start)

    print("\nGaps in outturn and other series:")
    fuel = load("fuelhh")
    _gaps("fuelhh", fuel["startTime"], "30min")
    _gaps("mid (EPEX)", mid.loc[(mid["dataProvider"] == "APXMIDP"), "startTime"], "30min")
    _gaps("system_prices", load("system_prices")["startTime"], "30min")
    _gaps("pv_live", load("pv_live")["datetime_gmt"], "30min")
    cont = load("continental_prices")
    for zone, g in cont.groupby("zone"):
        _gaps(f"continental {zone}", g["start_utc"].dt.floor("h"), "1h")
    weather = load("weather")
    _gaps(f"weather ({weather['location'].nunique()} points)", weather["start_utc"], "1h")
    gas = load("gas_sap")
    _gaps("gas_sap (daily)", pd.to_datetime(gas["gas_day"]), "1D")
    bh = load("bank_holidays")
    print(f"  {'bank_holidays':24s} {bh['date'].min()} -> {bh['date'].max()}, {len(bh)} rows")


if __name__ == "__main__":
    main()
