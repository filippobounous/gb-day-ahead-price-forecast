"""Project-wide constants: paths and time zone."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GB_DA_DATA_DIR", REPO_ROOT / "data"))  # override e.g. for tests
RAW_DIR = DATA_DIR / "raw"

# GB settlement days, auction delivery days and the 09:20 gate closure are all in UK local time.
# Every collected timestamp is converted to a UTC column (start_utc, publishTime, forecast_time...);
# UK-local concepts are derived from UTC only where needed (see timeutils).

# Gas price feature for delivery day D: the realised System Average Price of gas day D-3 (05:00 on
# D-3 to 05:00 on D-2), which was certainly published by the 09:00 D-1 cutoff. Gas day D-2 ended
# only 4 hours before the cutoff and its publication time is unverified, so it is a sensitivity
# check, not the default (see README, "Gas price").
GAS_SAP_LAG_DAYS = 3
UK_TZ = "Europe/London"
