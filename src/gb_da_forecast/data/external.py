"""Other open data sources.

    gas_sap             Daily System Average Price (SAP) of gas in GB: the average price of gas
                        traded on the On-the-day Commodity Market for each gas day (05:00-05:00).
                        From the ONS real-time indicators workbook (source: National Gas
                        Transmission), published weekly with about a week's delay.
    bank_holidays       UK bank holidays per nation (gov.uk), 2019 onwards.
    continental_prices  Day-ahead prices of zones linked to GB by interconnectors (FR, NL, BE, NO2,
                        DK1), plus DE-LU as the main continental reference market (no direct GB link
                        yet). From energy-charts.info (CC BY 4.0). Their auction closes at 11:00 UK time
                        (results by ~12:00), after the GB 09:20 auction, so only earlier days can be used as features.
                        15-minute resolution from Oct 2025.
    weather             Hourly weather for a few onshore and offshore points (Open-Meteo historical
                        forecast API, free for non-commercial use). NOT safe as a feature: the archive
                        is stitched from the first hours of successive model runs, i.e. runs started
                        after the 09:00 D-1 cutoff. Kept for exploration only; Open-Meteo's Previous
                        Runs API (2024 onwards) would give honest day-ahead weather forecasts.
    pv_live             Estimated national solar PV generation, half-hourly (Sheffield Solar PV_Live).
"""

from __future__ import annotations

import datetime as dt
import io
import re

import pandas as pd

from ._http import get_bytes, get_json
from ._store import Chunk, combine, download, make_chunks, save

# --- Gas -------------------------------------------------------------------------------------

ONS_SAP_PAGE = (
    "https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas"
)


def collect_gas_sap(start: dt.date | None = None, end: dt.date | None = None, workers: int = 1):
    """Download the latest ONS workbook (it always contains the full history since 2018)."""
    page = get_bytes(ONS_SAP_PAGE).decode("utf-8", "replace")
    links = list(dict.fromkeys(re.findall(r'(/file\?uri=[^"]+?\.xlsx)', page)))  # newest edition first
    content = None
    for link in links:  # the "/current/" alias sometimes 404s; fall back to the dated editions
        try:
            content = get_bytes("https://www.ons.gov.uk" + link)
            break
        except Exception:
            continue
    if content is None:
        raise RuntimeError(f"Could not download any SAP workbook linked from {ONS_SAP_PAGE}")
    workbook = pd.read_excel(io.BytesIO(content), sheet_name=None, header=4)
    table = next(df for name, df in workbook.items() if name.startswith("1."))  # "1.Daily SAP Gas"
    df = pd.DataFrame(
        {
            "gas_day": pd.to_datetime(table["Date"], errors="coerce").dt.date,
            "sap_p_kwh": pd.to_numeric(table["SAP actual day"], errors="coerce"),
        }
    ).dropna()
    df["sap_gbp_mwh"] = df["sap_p_kwh"] * 10  # 1 p/kWh = £10/MWh
    save("gas_sap", df)
    return df, []


# --- Bank holidays ---------------------------------------------------------------------------


def collect_bank_holidays(start: dt.date | None = None, end: dt.date | None = None, workers: int = 1):
    data = get_json("https://www.gov.uk/bank-holidays.json")
    rows = [
        {"division": division, "date": dt.date.fromisoformat(e["date"]), "title": e["title"]}
        for division, content in data.items()
        for e in content["events"]
    ]
    df = pd.DataFrame(rows)
    save("bank_holidays", df)
    return df, []


# --- Continental day-ahead prices ------------------------------------------------------------

CONTINENTAL_ZONES = ["FR", "NL", "BE", "NO2", "DK1", "DE-LU"]  # GB's neighbours + the German reference


def _fetch_continental(chunk: Chunk) -> pd.DataFrame:
    zone = chunk.extra["zone"]
    data = get_json(
        "https://api.energy-charts.info/price",
        params={"bzn": zone, "start": chunk.start.isoformat(), "end": chunk.end.isoformat()},
    )
    return pd.DataFrame(
        {
            "zone": zone,
            "start_utc": pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
            "price_eur_mwh": pd.to_numeric(pd.Series(data["price"]), errors="coerce"),
        }
    )


def collect_continental_prices(start: dt.date, end: dt.date, workers: int = 2):
    chunks = [c for z in CONTINENTAL_ZONES for c in make_chunks(start, end, "year", prefix=f"{z}_", zone=z)]
    failures = download("continental_prices", chunks, _fetch_continental, workers)
    return combine("continental_prices"), failures


# --- Weather ---------------------------------------------------------------------------------

WEATHER_LOCATIONS = {  # name: (latitude, longitude)
    "london": (51.51, -0.13),
    "birmingham": (52.48, -1.90),
    "manchester": (53.48, -2.24),
    "newcastle": (54.98, -1.61),
    "glasgow": (55.86, -4.25),
    "cardiff": (51.48, -3.18),
    "dogger_bank": (54.75, 1.90),  # offshore wind, North Sea
    "east_anglia_offshore": (52.60, 2.50),  # offshore wind, southern North Sea
    "moray_firth": (58.10, -3.00),  # offshore wind, Scotland
}
WEATHER_VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]


def _fetch_weather(chunk: Chunk) -> pd.DataFrame:
    end = min(chunk.end, dt.date.today())
    if chunk.start > end:
        return pd.DataFrame()
    name = chunk.extra["location"]
    lat, lon = WEATHER_LOCATIONS[name]
    data = get_json(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk.start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        },
    )
    df = pd.DataFrame(data["hourly"])
    df.insert(0, "start_utc", pd.to_datetime(df.pop("time"), utc=True))
    df.insert(0, "location", name)
    return df


def collect_weather(start: dt.date, end: dt.date, workers: int = 2):
    chunks = [
        c for loc in WEATHER_LOCATIONS for c in make_chunks(start, end, "year", prefix=f"{loc}_", location=loc)
    ]
    failures = download("weather", chunks, _fetch_weather, workers)
    return combine("weather"), failures


# --- Solar outturn ---------------------------------------------------------------------------


def _fetch_pv_live(chunk: Chunk) -> pd.DataFrame:
    end = min(chunk.end, dt.date.today())
    if chunk.start > end:
        return pd.DataFrame()
    data = get_json(
        "https://api.pvlive.uk/pvlive/api/v4/gsp/0",  # GSP 0 = national total
        params={"start": f"{chunk.start}T00:00:00Z", "end": f"{end}T23:30:00Z", "data_format": "json"},
    )
    df = pd.DataFrame(data["data"], columns=data["meta"])
    # PV_Live labels each half hour by its END time; keep the raw label and convert when building features.
    df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
    return df


def collect_pv_live(start: dt.date, end: dt.date, workers: int = 2):
    failures = download("pv_live", make_chunks(start, end, "year"), _fetch_pv_live, workers)
    return combine("pv_live"), failures
