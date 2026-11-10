"""Forecast the hourly prices of one delivery day, exactly as it would be done at 09:00 UK on D-1.

    python -m gb_da_forecast.collect                     # refresh the data first
    python -m gb_da_forecast.forecast 2026-11-11         # forecast delivery day 11 Nov
    python -m gb_da_forecast.forecast 2026-11-10 --csv out.csv

The model and its settings are the headline choice frozen by the backtest (results/run_info.json).
It is trained on every delivery day before D, and fed the inputs as published by the cutoff. If the
day has already been auctioned, the actual prices are shown alongside.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from .backtest import RESULTS_DIR, _make
from .config import UK_TZ
from .features import FEATURE_GROUPS, build_dataset


def forecast_day(day: dt.date) -> pd.DataFrame:
    choices = json.loads((RESULTS_DIR / "run_info.json").read_text())["choices"]
    df = build_dataset(until=day)
    train = df[(df["delivery_day"] < day) & df["price"].notna()]
    rows = df[df["delivery_day"] == day]
    if rows.empty:
        raise SystemExit(f"No delivery hours built for {day}")

    inputs = [f for g in ("fundamentals", "gas") for f in FEATURE_GROUPS[g]]
    missing = rows[inputs].isna().mean()
    if missing.any():
        print(f"Warning: some inputs are missing for {day} (data not collected up to the cutoff?):")
        print(missing[missing > 0].round(2).to_string())

    model = _make(choices["headline"], choices)().fit(train)
    out = pd.DataFrame(
        {
            "start_uk": rows["start_utc"].dt.tz_convert(UK_TZ).dt.strftime("%Y-%m-%d %H:%M"),
            "forecast_gbp_mwh": model.predict(rows).round(2),
            "actual_gbp_mwh": rows["price"].round(2),
        }
    )
    print(f"Delivery day {day} (23:00 UK on {day - dt.timedelta(days=1)} to 23:00 on {day}); "
          f"model: {choices['headline']}; forecast as at {rows['cutoff'].iloc[0].tz_convert(UK_TZ):%Y-%m-%d %H:%M} UK")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("day", type=dt.date.fromisoformat, help="delivery day, YYYY-MM-DD")
    parser.add_argument("--csv", help="also write the forecast to this CSV file")
    args = parser.parse_args()
    out = forecast_day(args.day)
    print(out.to_string(index=False))
    if args.csv:
        out.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
