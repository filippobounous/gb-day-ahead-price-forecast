"""Tests for the parts where a silent mistake would leak the future or misalign time.

    .venv/bin/python -m pytest -q
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_da_forecast.asof import as_of_cutoff
from gb_da_forecast.battery import Battery, plan, profit
from gb_da_forecast.config import RAW_DIR
from gb_da_forecast.timeutils import (
    delivery_day_bounds,
    delivery_day_of,
    forecast_cutoff,
    settlement_period_start,
)

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


# --- time conventions ----------------------------------------------------------------------


@pytest.mark.parametrize("day, hours", [("2025-03-30", 23), ("2025-10-26", 25), ("2025-07-01", 24)])
def test_delivery_day_length(day, hours):
    start, end = delivery_day_bounds(pd.Series(pd.to_datetime([day])))
    assert (end.iloc[0] - start.iloc[0]) == pd.Timedelta(hours=hours)


def test_delivery_day_runs_23_to_23_uk():
    start, end = delivery_day_bounds(pd.Series(pd.to_datetime(["2025-07-02"])))
    assert start.iloc[0] == ts("2025-07-01 22:00")  # 23:00 BST
    assert end.iloc[0] == ts("2025-07-02 22:00")
    days = delivery_day_of(pd.Series([ts("2025-07-01 21:00"), ts("2025-07-01 22:00")]))
    assert list(days) == [dt.date(2025, 7, 1), dt.date(2025, 7, 2)]


def test_forecast_cutoff_is_0900_uk_on_the_day_before():
    cut = forecast_cutoff(pd.Series(pd.to_datetime(["2025-07-02", "2025-12-02"])))
    assert list(cut) == [ts("2025-07-01 08:00"), ts("2025-12-01 09:00")]  # BST, then GMT


def test_settlement_periods_on_clock_change_days():
    starts = settlement_period_start(
        pd.Series(["2025-10-26", "2025-10-26", "2025-03-30"]), pd.Series([1, 50, 46])
    )
    assert list(starts) == [ts("2025-10-25 23:00"), ts("2025-10-26 23:30"), ts("2025-03-30 22:30")]


# --- point-in-time selection ---------------------------------------------------------------


def test_as_of_stitches_vintages_and_ignores_later_ones():
    # Delivery day 2023-03-02 (winter): runs 2023-03-01 23:00 -> 2023-03-02 23:00 UTC, cutoff 03-01 09:00.
    early = pd.date_range(ts("2023-03-01 23:00"), periods=12, freq="30min")  # up to 04:30
    late = pd.date_range(ts("2023-03-02 05:00"), periods=36, freq="30min")
    rows = (
        [("2023-03-01 08:15", t, 1.0) for t in early]  # half-hourly vintage, short horizon
        + [("2023-03-01 08:45", t, 2.0) for t in late]  # day-ahead release
        + [("2023-03-01 09:15", t, 9.0) for t in list(early) + list(late)]  # after the cutoff
    )
    df = pd.DataFrame(rows, columns=["published", "start", "value"])
    df["published"] = pd.to_datetime(df["published"]).dt.tz_localize(UTC)
    chosen = as_of_cutoff(df, "published", "start")
    assert len(chosen) == 48
    assert (chosen["published"] <= chosen["cutoff"]).all()
    assert chosen["value"].tolist() == [1.0] * 12 + [2.0] * 36


# --- battery -------------------------------------------------------------------------------


def test_battery_worked_example():
    battery = Battery(power_mw=1, energy_mwh=2, round_trip_efficiency=1.0)
    actual = np.full(24, 80.0)
    actual[[3, 4, 18, 19, 20]] = [40, 45, 150, 140, 100]
    forecast = actual.copy()
    forecast[[18, 19, 20]] = [120, 150, 145]  # thinks the peak is an hour later
    best = profit(*plan(actual, battery), actual)
    got = profit(*plan(forecast, battery), actual)
    assert best == pytest.approx(205)
    assert got == pytest.approx(155)


def test_battery_respects_its_limits():
    battery = Battery()
    prices = np.sin(np.linspace(0, 4 * np.pi, 24)) * 50 + 60
    charge, discharge = plan(prices, battery)
    eta = np.sqrt(battery.round_trip_efficiency)
    soc = np.cumsum(eta * charge - discharge / eta)
    assert charge.max() <= battery.power_mw + 1e-9 and discharge.max() <= battery.power_mw + 1e-9
    assert soc.min() >= -1e-9 and soc.max() <= battery.energy_mwh + 1e-9
    assert abs(soc[-1]) < 1e-9  # empty at the end of the day
    assert discharge.sum() / eta <= battery.energy_mwh + 1e-9  # at most one cycle


# --- features on the real data (skipped if it has not been collected) ----------------------

needs_data = pytest.mark.skipif(not (RAW_DIR / "n2ex.parquet").exists(), reason="run collect first")


@needs_data
def test_features_use_only_information_from_before_the_cutoff():
    from gb_da_forecast.data._store import load
    from gb_da_forecast.features import build_dataset

    df = build_dataset()
    day = dt.date(2025, 7, 2)
    rows = df[df["delivery_day"] == day]
    assert len(rows) == 24

    # yesterday's price at the same hour is the price of delivery day D-1
    prev = df[df["delivery_day"] == day - dt.timedelta(days=1)].set_index("hour")["price"]
    assert np.allclose(rows["p_d1"], rows["hour"].map(prev))

    # the demand forecast equals a direct filter on publish time <= cutoff
    ndf = load("ndf")
    cutoff = rows["cutoff"].iloc[0]
    hour = rows["start_utc"].iloc[10]
    direct = ndf[(ndf["publishTime"] <= cutoff) & (ndf["startTime"].dt.floor("h") == hour)]
    direct = direct.sort_values("publishTime").groupby("startTime").tail(1)["demand"].mean()
    assert rows["demand_fc"].iloc[10] == pytest.approx(direct)

    # gas: gas day D-3
    gas = load("gas_sap").set_index("gas_day")["sap_gbp_mwh"]
    assert rows["gas_sap"].iloc[0] == pytest.approx(gas[day - dt.timedelta(days=3)])

    # outturn windows end before the cutoff, and are fresh (late re-publications must not freeze them)
    days = df.groupby("delivery_day")[["imb_age_h", "gen_age_h"]].first()
    assert (days >= 0).all().all()
    assert (days["gen_age_h"] <= 2).mean() > 0.99
    assert (days["imb_age_h"] <= 48).mean() > 0.97  # >48 h only where Elexon re-published late
