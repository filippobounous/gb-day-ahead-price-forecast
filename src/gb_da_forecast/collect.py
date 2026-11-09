"""Download every raw dataset into data/raw/.

    python -m gb_da_forecast.collect                     # everything, default date ranges
    python -m gb_da_forecast.collect --only n2ex ndf     # a subset
    python -m gb_da_forecast.collect --start 2023-01-01  # override every source's start date

Re-running is cheap: finished chunks are cached and only missing or recent ones are fetched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

from .data import elexon, external, neso

TARGET_ERA = dt.date(2021, 1, 1)  # N2EX prices start 2021-09-20; earlier months give room for lags
LONG_HISTORY = dt.date(2018, 1, 1)  # cheap sources: take everything back to 2018

# name: (collector, default start date, description)
SOURCES = {
    "n2ex": (neso.collect_n2ex, TARGET_ERA, "TARGET - N2EX GB hourly day-ahead price (NESO)"),
    "ndf": (elexon.collect_ndf, TARGET_ERA, "National demand forecast, every vintage (Elexon)"),
    "windfor": (elexon.collect_windfor, TARGET_ERA, "Transmission wind forecast, every vintage (Elexon)"),
    "embedded": (neso.collect_embedded, TARGET_ERA, "Embedded wind + solar forecast, morning vintages (NESO)"),
    "gas_sap": (external.collect_gas_sap, LONG_HISTORY, "GB gas System Average Price, daily (ONS)"),
    "bank_holidays": (external.collect_bank_holidays, LONG_HISTORY, "UK bank holidays (gov.uk)"),
    "fuelhh": (elexon.collect_fuelhh, LONG_HISTORY, "Generation by fuel + interconnectors, outturn (Elexon)"),
    "system_prices": (elexon.collect_system_prices, TARGET_ERA, "Imbalance price (Elexon)"),
    "mid": (elexon.collect_mid, LONG_HISTORY, "Market index price, EPEX/N2EX short-term trades (Elexon)"),
    "continental_prices": (external.collect_continental_prices, LONG_HISTORY, "FR/NL/BE/DE/NO2/DK1 day-ahead prices (energy-charts)"),
    "weather": (external.collect_weather, TARGET_ERA, "Weather at 9 onshore/offshore points (Open-Meteo)"),
    "pv_live": (external.collect_pv_live, LONG_HISTORY, "National solar PV outturn (Sheffield Solar)"),
}

TIME_COLUMNS = ["start_utc", "startTime", "datetime_gmt", "date", "gas_day"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(SOURCES), help="sources to collect (default: all)")
    parser.add_argument("--start", type=dt.date.fromisoformat, help="override every source's start date")
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today() + dt.timedelta(days=2))
    args = parser.parse_args()

    summary = []
    for name in args.only or SOURCES:
        collector, default_start, description = SOURCES[name]
        start = args.start or default_start
        print(f"\n=== {name}: {description} [{start} -> {args.end}]", flush=True)
        t0 = time.time()
        try:
            df, failures = collector(start, args.end)
        except Exception as exc:
            print(f"[{name}] FAILED: {exc!r}", flush=True)
            summary.append((name, 0, "-", "-", f"error: {exc!r}"[:60], time.time() - t0))
            continue
        time_col = next((c for c in TIME_COLUMNS if c in df.columns), None)
        lo, hi = (df[time_col].min(), df[time_col].max()) if time_col and len(df) else ("-", "-")
        summary.append((name, len(df), lo, hi, f"{len(failures)} failed chunks", time.time() - t0))
        for key, err in failures[:5]:
            print(f"  failed {key}: {err}", flush=True)

    print("\n=== Summary")
    for name, rows, lo, hi, status, secs in summary:
        print(f"{name:20s} {rows:>10,} rows  {str(lo)[:16]:16s} -> {str(hi)[:16]:16s}  {status:16s} {secs:6.0f}s")


if __name__ == "__main__":
    main()
