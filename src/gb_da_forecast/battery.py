"""Battery value of a price forecast: how much money a battery schedule built on it makes.

For each delivery day:
  1. plan the charge/discharge schedule that maximises profit *if the forecast were right*;
  2. settle that schedule at the prices that actually cleared (realised profit);
  3. plan again on the actual prices (perfect foresight: the most that was possible);
  capture rate = realised profit / perfect-foresight profit, summed over all days.

A forecast that gets the daily *shape* right (which hours are cheap and which are dear) scores
well even if its level is off; average error (MAE) cannot see that distinction. (Scaling a whole
day by one factor leaves the schedule unchanged; a constant offset can change it slightly, because
with efficiency losses the battery sells less energy than it buys.)

The planning step is a small linear program (scipy's HiGHS solver). Battery: 5 kW / 10 kWh
(a 2-hour home battery), 90% round-trip efficiency, at most one full cycle per delivery day,
empty at the start and end of each day. Simplifications: day-ahead market only, no household
load or solar, no degradation cost, hourly resolution, the battery does not move prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    power_mw: float = 0.005  # 5 kW
    energy_mwh: float = 0.010  # 10 kWh
    round_trip_efficiency: float = 0.90
    max_cycles_per_day: float = 1.0


def plan(prices: np.ndarray, battery: Battery = Battery()) -> tuple[np.ndarray, np.ndarray]:
    """Profit-maximising schedule for one day of hourly prices.

    Returns (charge, discharge): MWh drawn from / delivered to the grid in each hour.
    Variables: charge c_t, discharge d_t and state of charge s_t (end of hour t), for each hour.
      s_t = s_(t-1) + eta * c_t - d_t / eta     (eta = sqrt(round-trip efficiency), s_(-1) = 0)
      0 <= c_t, d_t <= power * 1h,   0 <= s_t <= energy,   s_(last) = 0
      sum_t d_t / eta <= cycles * energy        (at most one full cycle a day)
    Objective: maximise sum_t price_t * (d_t - c_t).
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    eta = np.sqrt(battery.round_trip_efficiency)
    cost = np.concatenate([p, -p, np.zeros(n)])  # linprog minimises cost = p.c - p.d

    a_eq = np.zeros((n, 3 * n))
    for t in range(n):
        a_eq[t, 2 * n + t] = 1.0  # s_t
        if t > 0:
            a_eq[t, 2 * n + t - 1] = -1.0  # - s_(t-1)
        a_eq[t, t] = -eta  # - eta c_t
        a_eq[t, n + t] = 1.0 / eta  # + d_t / eta
    a_ub = np.concatenate([np.zeros(n), np.full(n, 1.0 / eta), np.zeros(n)])[None, :]
    b_ub = [battery.max_cycles_per_day * battery.energy_mwh]
    bounds = [(0, battery.power_mw)] * (2 * n) + [(0, battery.energy_mwh)] * (n - 1) + [(0, 0)]

    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:n], res.x[n : 2 * n]


def profit(charge: np.ndarray, discharge: np.ndarray, prices: np.ndarray) -> float:
    return float(np.dot(prices, discharge - charge))


def evaluate(predictions: pd.DataFrame, models: list[str], battery: Battery = Battery()) -> pd.DataFrame:
    """Daily realised and perfect-foresight profits (£ per battery) for each model column.

    ``predictions`` needs columns delivery_day, start_utc, price (actual) and one per model.
    """
    rows = []
    for day, part in predictions.sort_values("start_utc").groupby("delivery_day"):
        actual = part["price"].to_numpy(float)
        best = profit(*plan(actual, battery), actual)
        row = {"delivery_day": day, "perfect": best}
        for m in models:
            forecast = part[m].to_numpy(float)
            row[m] = profit(*plan(np.nan_to_num(forecast, nan=np.nanmean(forecast)), battery), actual)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(daily: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Capture rate and £ per battery per year for each model."""
    days = len(daily)
    perfect = daily["perfect"].sum()
    out = pd.DataFrame(
        {
            "capture_rate": [daily[m].sum() / perfect for m in models],
            "profit_gbp_per_year": [daily[m].sum() / days * 365 for m in models],
        },
        index=models,
    )
    out.loc["perfect foresight"] = [1.0, perfect / days * 365]
    return out
