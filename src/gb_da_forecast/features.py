"""Build the modelling table: one row per delivery hour, with every input as known at 09:00 UK on D-1.

    from gb_da_forecast.features import build_dataset
    df = build_dataset()            # all delivery days with data
    df = build_dataset(until=date)  # also add rows for a future delivery day (target missing)

Rows are delivery hours (``start_utc``), grouped into delivery days (23:00-23:00 UK). Columns:

  target        ``price``: the N2EX hourly day-ahead price (£/MWh); missing for days not yet auctioned.
  calendar      hour of day (UK local), day of week, bank holiday, month, smooth day-of-year.
  fundamentals  demand, transmission wind and embedded wind/solar forecasts for that hour, as
                published by the cutoff (``asof.as_of_cutoff``), and residual demand.
  gas           realised gas price of gas day D-3 (``config.GAS_SAP_LAG_DAYS``) and its 7-day mean.
  price lags    prices of delivery days D-1, D-2, D-7 at the same hour, and D-1's mean/min/max/last hour.
  context       optional groups, kept only if they help on the validation year (see backtest.py):
                continental day-ahead prices of D-1, imbalance prices and generation mix of the
                last 24 hours available at the cutoff.

Residual demand is demand minus *transmission* wind. NESO's national demand is measured on the
transmission network, so embedded (distribution-connected) wind and solar already reduce it; they
are kept as separate inputs but not subtracted again.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .asof import as_of_cutoff
from .config import GAS_SAP_LAG_DAYS, UK_TZ
from .data._store import load
from .timeutils import delivery_day_bounds, delivery_day_of, forecast_cutoff

GAS_REF_DAYS = 7  # the gas price used to normalise prices is a 7-day mean, robust to one-day dips

CALENDAR = ["hour", "dow", "is_holiday", "month", "doy_sin", "doy_cos"]
FUNDAMENTALS = [
    "demand_fc", "wind_fc", "emb_solar_fc", "emb_wind_fc", "residual_fc",
    "residual_fc_day_mean", "residual_fc_day_max", "residual_fc_day_min",
]
GAS = ["gas_sap", "gas_ref"]
PRICE_LAGS = ["p_d1", "p_d2", "p_d7", "p_d1_mean", "p_d1_min", "p_d1_max", "p_d1_last"]
CONTINENTAL = ["cont_fr_d1", "cont_nl_d1", "cont_be_d1", "cont_no2_d1", "cont_dk1_d1", "cont_de_d1"]
IMBALANCE = ["imb_price_24h", "imb_price_24h_max", "imb_niv_24h"]
GEN_MIX = ["gen_interconnect_24h", "gen_ccgt_24h", "gen_wind_24h", "gen_nuclear_24h"]

FEATURE_GROUPS = {
    "calendar": CALENDAR,
    "fundamentals": FUNDAMENTALS,
    "gas": GAS,
    "price_lags": PRICE_LAGS,
    "continental": CONTINENTAL,
    "imbalance": IMBALANCE,
    "gen_mix": GEN_MIX,
}
# Columns measured in money: divided by the gas price when the model works in "units of gas".
PRICE_LIKE = PRICE_LAGS + CONTINENTAL + ["imb_price_24h", "imb_price_24h_max"]


def _hour_grid(first_day: dt.date, last_day: dt.date) -> pd.DataFrame:
    """Every delivery hour from first_day to last_day (23, 24 or 25 per day)."""
    days = pd.Series(pd.date_range(first_day, last_day, freq="D"))
    starts, ends = delivery_day_bounds(days)
    hours = [pd.date_range(s, e, freq="h", inclusive="left") for s, e in zip(starts, ends)]
    grid = pd.DataFrame({"start_utc": hours[0].append(hours[1:])})
    grid["delivery_day"] = delivery_day_of(grid["start_utc"])
    return grid


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    local = df["start_utc"].dt.tz_convert(UK_TZ)
    holidays = load("bank_holidays")
    holidays = set(pd.to_datetime(holidays.loc[holidays["division"] == "england-and-wales", "date"]).dt.date)
    doy = local.dt.day_of_year
    return df.assign(
        hour=local.dt.hour,
        dow=local.dt.day_of_week,  # of the local date: Sunday 23:00 behaves like Sunday night
        is_holiday=local.dt.date.isin(holidays).astype(int),
        month=local.dt.month,
        doy_sin=np.sin(2 * np.pi * doy / 365.25),
        doy_cos=np.cos(2 * np.pi * doy / 365.25),
    )


def _hourly_as_of(df: pd.DataFrame, publish_col: str, time_col: str, value_col: str) -> pd.Series:
    """Point-in-time forecast per period, averaged to hours, indexed by hour start (UTC)."""
    chosen = as_of_cutoff(df, publish_col, time_col)
    return chosen.groupby(chosen[time_col].dt.floor("h"))[value_col].mean()


def _fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    demand = _hourly_as_of(load("ndf"), "publishTime", "startTime", "demand")
    wind = _hourly_as_of(load("windfor"), "publishTime", "startTime", "generation")
    embedded = load("embedded")
    solar = _hourly_as_of(embedded, "forecast_time", "start_utc", "embedded_solar_forecast")
    emb_wind = _hourly_as_of(embedded, "forecast_time", "start_utc", "embedded_wind_forecast")

    out = df.copy()
    idx = out["start_utc"]
    out["demand_fc"] = idx.map(demand)
    out["wind_fc"] = idx.map(wind)
    out["emb_solar_fc"] = idx.map(solar)
    out["emb_wind_fc"] = idx.map(emb_wind)
    out["residual_fc"] = out["demand_fc"] - out["wind_fc"]
    day = out.groupby("delivery_day")["residual_fc"]
    out["residual_fc_day_mean"] = day.transform("mean")
    out["residual_fc_day_max"] = day.transform("max")
    out["residual_fc_day_min"] = day.transform("min")
    return out


def _gas(df: pd.DataFrame) -> pd.DataFrame:
    sap = load("gas_sap")
    daily = pd.Series(sap["sap_gbp_mwh"].to_numpy(), index=pd.to_datetime(sap["gas_day"])).sort_index()
    last_day = max(daily.index.max(), pd.Timestamp(df["delivery_day"].max()))
    full = daily.reindex(pd.date_range(daily.index.min(), last_day, freq="D"))
    available = full.ffill()  # only ever carries an older value forward (ONS publishes with a lag)
    ref = full.rolling(GAS_REF_DAYS, min_periods=3).mean().ffill()

    out = df.copy()
    lagged = pd.to_datetime(out["delivery_day"]) - pd.Timedelta(days=GAS_SAP_LAG_DAYS)
    out["gas_sap"] = lagged.map(available)
    out["gas_ref"] = lagged.map(ref)  # mean of gas days D-9 .. D-3
    out["gas_sap_prev"] = (lagged - pd.Timedelta(days=1)).map(available)  # for the gas-scaled naive
    last_real = pd.Series(full.dropna().index, index=full.dropna().index).reindex(full.index).ffill()
    out["gas_age_days"] = (lagged - lagged.map(last_real)).dt.days + GAS_SAP_LAG_DAYS
    return out


def _price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Prices of earlier *delivery days* (all cleared by D-2, so known at the cutoff)."""
    out = df.copy()
    by_hour = out.groupby(["delivery_day", "hour"])["price"].mean()  # 25-hour day: two 01:00s
    daily = out.groupby("delivery_day")["price"].agg(["mean", "min", "max"])
    last_hour = by_hour.xs(22, level="hour")  # 22:00-23:00 UK, the last hour of a delivery day

    def shifted(days: int) -> pd.Series:
        return out["delivery_day"] - dt.timedelta(days=days)

    for k in (1, 2, 7):
        keys = pd.MultiIndex.from_arrays([shifted(k), out["hour"]])
        out[f"p_d{k}"] = by_hour.reindex(keys).to_numpy()
    d1 = shifted(1)
    out["p_d1_mean"] = d1.map(daily["mean"])
    out["p_d1_min"] = d1.map(daily["min"])
    out["p_d1_max"] = d1.map(daily["max"])
    out["p_d1_last"] = d1.map(last_hour)
    return out


def _continental(df: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead prices of delivery day D-1 at the same hour (their auction closes 11:00 UK on D-2)."""
    cont = load("continental_prices")
    hourly = cont.groupby(["zone", cont["start_utc"].dt.floor("h")])["price_eur_mwh"].mean().reset_index()
    hourly["delivery_day"] = delivery_day_of(hourly["start_utc"])
    hourly["hour"] = hourly["start_utc"].dt.tz_convert(UK_TZ).dt.hour
    table = hourly.groupby(["zone", "delivery_day", "hour"])["price_eur_mwh"].mean()

    out = df.copy()
    keys = pd.MultiIndex.from_arrays([out["delivery_day"] - dt.timedelta(days=1), out["hour"]])
    for zone, col in zip(["FR", "NL", "BE", "NO2", "DK1", "DE-LU"], CONTINENTAL):
        out[col] = table.xs(zone, level="zone").reindex(keys).to_numpy()
    return out


def _last_24h_as_of(
    df: pd.DataFrame,
    series: pd.DataFrame,
    available_col: str,
    cols: list[str],
    how: str = "mean",
    age_col: str | None = None,
) -> pd.DataFrame:
    """24-hour statistics of outturn data as known at each delivery day's cutoff.

    ``series`` is indexed by period start (UTC) and has an ``available_col`` saying when each
    period's value was published. For each cutoff: keep only the periods published by then, find
    the latest of them, and aggregate the kept periods in the 24 hours ending there. Filtering
    period by period matters: some periods are re-published months later, and must neither be
    used early nor hold the window back. ``age_col`` records how old the window's end was at the
    cutoff (hours), as a diagnostic.
    """
    starts = series.index.to_numpy()
    available = series[available_col].to_numpy()
    values = series[cols].to_numpy(float)
    agg = np.nanmean if how == "mean" else np.nanmax
    days = sorted(df["delivery_day"].unique())
    cutoffs = forecast_cutoff(pd.Series(pd.to_datetime(days))).dt.tz_localize(None).to_numpy()
    starts_naive = pd.DatetimeIndex(starts).tz_localize(None).to_numpy()
    available_naive = pd.DatetimeIndex(available).tz_localize(None).to_numpy()
    rows = []
    for day, cutoff in zip(days, cutoffs):
        known = available_naive <= cutoff
        row = {"delivery_day": day}
        if known.any():
            end = starts_naive[known].max()
            window = known & (starts_naive > end - np.timedelta64(24, "h")) & (starts_naive <= end)
            row |= dict(zip(cols, agg(values[window], axis=0)))
            if age_col:
                row[age_col] = (cutoff - end) / np.timedelta64(1, "h")
        rows.append(row)
    return df.merge(pd.DataFrame(rows), on="delivery_day", how="left")


def _imbalance(df: pd.DataFrame) -> pd.DataFrame:
    sp = load("system_prices").sort_values("startTime").drop_duplicates("startTime", keep="last")
    series = pd.DataFrame(
        {
            "imb_price_24h": sp["systemSellPrice"].astype(float).to_numpy(),
            "imb_niv_24h": sp["netImbalanceVolume"].astype(float).to_numpy(),
            "created": sp["createdDateTime"].to_numpy(),
        },
        index=pd.DatetimeIndex(sp["startTime"]),
    )
    out = _last_24h_as_of(df, series, "created", ["imb_price_24h", "imb_niv_24h"], age_col="imb_age_h")
    peak = series.rename(columns={"imb_price_24h": "imb_price_24h_max"})
    return _last_24h_as_of(out, peak, "created", ["imb_price_24h_max"], how="max")


def _gen_mix(df: pd.DataFrame) -> pd.DataFrame:
    fuel = load("fuelhh")
    wide = fuel.pivot_table(index="startTime", columns="fuelType", values="generation", aggfunc="mean")
    published = fuel.groupby("startTime")["publishTime"].max()
    interconnectors = [c for c in wide.columns if c.startswith("INT")]
    series = pd.DataFrame(
        {
            "gen_interconnect_24h": wide[interconnectors].sum(axis=1, min_count=1),  # net imports, MW
            "gen_ccgt_24h": wide.get("CCGT"),
            "gen_wind_24h": wide.get("WIND"),
            "gen_nuclear_24h": wide.get("NUCLEAR"),
            "published": published,
        }
    ).sort_index()
    return _last_24h_as_of(df, series, "published", GEN_MIX, age_col="gen_age_h")


def build_dataset(until: dt.date | None = None) -> pd.DataFrame:
    """Assemble the full modelling table (see module docstring)."""
    n2ex = load("n2ex")
    first = n2ex["delivery_day"].min()
    last = max(n2ex["delivery_day"].max(), until or first)
    df = _hour_grid(first, last).merge(n2ex[["start_utc", "price_gbp_mwh"]], on="start_utc", how="left")
    df = df.rename(columns={"price_gbp_mwh": "price"})
    df = _calendar(df)
    df = _price_lags(df)
    df = _fundamentals(df)
    df = _gas(df)
    df = _continental(df)
    df = _imbalance(df)
    df = _gen_mix(df)
    df["cutoff"] = df["delivery_day"].map(
        dict(zip(df["delivery_day"].unique(), forecast_cutoff(pd.Series(pd.to_datetime(df["delivery_day"].unique())))))
    )
    return df.sort_values("start_utc", ignore_index=True)
