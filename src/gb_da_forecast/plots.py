"""Charts used in the walkthrough notebook. One consistent, quiet style: thin lines, recessive grid,
a fixed colour per model, one y-axis per chart (two measures = two stacked panels, never two axes)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
ACTUAL = "#3a3a38"  # the observed price: neutral, so the model colours stay distinct
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Fixed colour per model, so a model looks the same in every chart.
MODEL_COLOURS = {
    "naive_d1": SERIES[0],
    "lasso": SERIES[1],
    "gbm": SERIES[2],
    "average": SERIES[3],
    "naive_d7": SERIES[4],
    "naive_d1_gas": SERIES[6],
}
LABELS = {
    "naive_d1": "Naive: same hour yesterday",
    "naive_d7": "Naive: same hour last week",
    "naive_d1_gas": "Naive: yesterday x gas change",
    "lasso_raw": "Lasso (price)",
    "lasso_ratio": "Lasso (price / gas)",
    "gbm_raw": "Boosted trees (price)",
    "gbm_ratio": "Boosted trees (price / gas)",
    "average": "Average of Lasso and trees",
}


FEATURE_LABELS = {
    "residual_fc": "Residual demand forecast (this hour)",
    "residual_fc_day_mean": "Residual demand forecast (day mean)",
    "residual_fc_day_max": "Residual demand forecast (day max)",
    "residual_fc_day_min": "Residual demand forecast (day min)",
    "demand_fc": "Demand forecast",
    "wind_fc": "Transmission wind forecast",
    "emb_solar_fc": "Embedded solar forecast",
    "emb_wind_fc": "Embedded wind forecast",
    "gas_sap": "Gas price (gas day D-3)",
    "gas_ref": "Gas price (7-day mean)",
    "p_d1": "Price, same hour, D-1",
    "p_d2": "Price, same hour, D-2",
    "p_d7": "Price, same hour, D-7",
    "p_d1_mean": "Price, D-1 average",
    "p_d1_min": "Price, D-1 minimum",
    "p_d1_max": "Price, D-1 maximum",
    "p_d1_last": "Price, last hour of D-1",
    "hour": "Hour of day",
    "dow": "Day of week",
    "is_holiday": "Bank holiday",
    "month": "Month",
    "doy_sin": "Season (sine)",
    "doy_cos": "Season (cosine)",
    "cont_fr_d1": "France price, D-1",
    "cont_nl_d1": "Netherlands price, D-1",
    "cont_be_d1": "Belgium price, D-1",
    "cont_no2_d1": "Norway (NO2) price, D-1",
    "cont_dk1_d1": "Denmark (DK1) price, D-1",
    "cont_de_d1": "Germany price, D-1",
    "imb_price_24h": "Imbalance price, last 24 h",
    "imb_price_24h_max": "Imbalance price max, last 24 h",
    "imb_niv_24h": "Net imbalance volume, last 24 h",
    "gen_interconnect_24h": "Net interconnector imports, last 24 h",
    "gen_ccgt_24h": "Gas (CCGT) generation, last 24 h",
    "gen_wind_24h": "Wind generation, last 24 h",
    "gen_nuclear_24h": "Nuclear generation, last 24 h",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def colour(model: str) -> str:
    return MODEL_COLOURS.get(model, MODEL_COLOURS.get(model.split("_")[0], SERIES[5]))


def label(model: str) -> str:
    return LABELS.get(model, model)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def price_and_gas(df: pd.DataFrame):
    """Daily average power price (top) and the gas price used by the model (bottom), shared time axis."""
    daily = df.groupby(pd.to_datetime(df["delivery_day"])).agg(price=("price", "mean"), gas=("gas_sap", "first"))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[3, 2])
    top.plot(daily.index, daily["price"], color=SERIES[0], linewidth=1.0)
    top.set_title("Daily average N2EX day-ahead price (£/MWh)")
    bottom.plot(daily.index, daily["gas"], color=SERIES[1], linewidth=1.0)
    bottom.set_title("Gas System Average Price, gas day D-3 (£/MWh of gas)")
    fig.tight_layout()
    return fig


def ratio_over_time(df: pd.DataFrame):
    """Monthly median of price / gas: the market's implied heat rate."""
    ratio = (df["price"] / df["gas_ref"]).groupby(pd.to_datetime(df["delivery_day"]).dt.to_period("M")).median()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(ratio.index.to_timestamp(), ratio.to_numpy(), color=SERIES[0], marker="o", markersize=3)
    ax.set_ylim(0, max(4, float(ratio.max()) * 1.1))
    ax.set_title("Monthly median of power price / gas price")
    ax.set_ylabel("ratio")
    fig.tight_layout()
    return fig


def merit_order(df: pd.DataFrame):
    """Price / gas against forecast residual demand: the merit-order 'hockey stick'."""
    d = df[["residual_fc", "price", "gas_ref"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hb = ax.hexbin(d["residual_fc"] / 1000, d["price"] / d["gas_ref"], gridsize=60, mincnt=3, cmap="Blues",
                   extent=(d["residual_fc"].quantile(0.001) / 1000, d["residual_fc"].quantile(0.999) / 1000, -1, 6))
    ax.set_xlabel("Forecast residual demand: demand - transmission wind (GW)")
    ax.set_ylabel("Price / gas price")
    ax.set_title("The merit order: price climbs steeply once gas plants are needed")
    ax.grid(False)
    fig.colorbar(hb, ax=ax, label="hours")
    fig.tight_layout()
    return fig


def daily_shape_by_season(df: pd.DataFrame):
    """Average price by hour of day, one line per season (2023 onwards, to avoid the 2022 crisis level)."""
    d = df[pd.to_datetime(df["delivery_day"]) >= "2023-01-01"]
    season = d["month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                             6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"})
    shape = d.groupby([season, d["hour"]])["price"].mean().unstack(0)
    order = ["Winter", "Spring", "Summer", "Autumn"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, s in enumerate(order):
        ax.plot(shape.index, shape[s], color=SERIES[i], label=s)
    # direct labels at the line ends, nudged apart so they never overlap
    ends = shape.iloc[-1][order].sort_values()
    gap = (shape.max().max() - shape.min().min()) * 0.06
    placed = []
    for s, v in ends.items():
        y = max(v, placed[-1] + gap) if placed else v
        placed.append(y)
        ax.annotate(s, (23, v), xytext=(23.4, y), textcoords="data", color=TEXT_2, fontsize=9, va="center")
    ax.set_xlim(-0.5, 25)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (UK time)")
    ax.set_ylabel("£/MWh")
    ax.set_title("Average price by hour of day and season (2023 onwards)")
    ax.legend(loc="upper left", ncols=4)
    fig.tight_layout()
    return fig


def mae_by_hour(table: pd.DataFrame, models: list[str]):
    fig, ax = plt.subplots(figsize=(9, 4))
    for m in models:
        ax.plot(table.index, table[m], color=colour(m), label=label(m), marker="o", markersize=3)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (UK time)")
    ax.set_ylabel("MAE (£/MWh)")
    ax.set_ylim(bottom=0)
    ax.set_title("Where the errors are: mean absolute error by hour (test year)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def mae_by_month(table: pd.DataFrame, models: list[str]):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(table))
    for m in models:
        ax.plot(x, table[m], color=colour(m), label=label(m), marker="o", markersize=4)
    ax.set_xticks(x, table.index, rotation=45, ha="right")
    ax.set_ylabel("MAE (£/MWh)")
    ax.set_ylim(bottom=0)
    ax.set_title("Mean absolute error by month (test year)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def sample_week(test: pd.DataFrame, models: list[str], start: str, days: int = 7):
    t = test.set_index("start_utc").sort_index()
    w = t.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=days)]
    local = w.index.tz_convert("Europe/London").tz_localize(None)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(local, w["price"], color=ACTUAL, linewidth=2.0, label="Actual price")
    for m in models:
        ax.plot(local, w[m], color=colour(m), label=label(m), linewidth=1.4)
    ax.set_ylabel("£/MWh")
    top = float(np.nanmax(w[["price", *models]].to_numpy()))
    ax.set_ylim(top=top * 1.2)  # headroom for the legend
    ax.set_title(f"Forecasts made at 09:00 the day before vs actual prices, week of {start}")
    ax.legend(loc="upper left", ncols=len(models) + 1)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def battery_capture(summary: pd.DataFrame, models: list[str]):
    """Share of the perfect-foresight battery profit each forecast captures (one bar per model)."""
    s = summary.loc[models, "capture_rate"] * 100
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(models)))
    y = np.arange(len(models))[::-1]
    ax.barh(y, s.to_numpy(), color=[colour(m) for m in models], height=0.6)
    for yi, v in zip(y, s.to_numpy()):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=TEXT_2)
    ax.set_yticks(y, [label(m) for m in models])
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of perfect-foresight battery profit captured (test year)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Battery value: share of the possible profit captured")
    fig.tight_layout()
    return fig


def hypothesis_effects(table: pd.DataFrame):
    """Round 2: change in validation MAE for each single change vs the round-1 baseline (left = better)."""
    d = table.drop(index=[i for i in table.index if i.startswith("B0")]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.42 * len(d)))
    y = np.arange(len(d))
    colours = [SERIES[2] if a else GRID for a in d["adopted"].astype(bool)]
    ax.barh(y, d["mae_change_pct"], color=colours, height=0.6, edgecolor=TEXT_2, linewidth=0.4)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.axvline(-1, color=TEXT_2, linewidth=0.8, linestyle=":")
    for yi, (mae, cap) in zip(y, d[["mae_change_pct", "capture_change_pp"]].to_numpy()):
        ax.annotate(f"{mae:+.1f}% MAE, {cap:+.1f} pp battery", (max(mae, 0), yi), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT_2, fontsize=8)
    names = [f"{n}  [adopted]" if a else n for n, a in zip(d.index, d["adopted"].astype(bool))]
    ax.set_yticks(y, names)
    for tick, a in zip(ax.get_yticklabels(), d["adopted"].astype(bool)):
        if a:
            tick.set_fontweight("bold")
    ax.set_xlabel("Change in validation MAE vs round-1 model (%); dotted line = the 1% adoption threshold")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    lo = min(-2.0, float(d["mae_change_pct"].min()) - 0.5)
    ax.set_xlim(lo, max(2.0, float(d["mae_change_pct"].max())) + 3.5)
    ax.set_title("Round 2: effect of each change (validation year)")
    fig.tight_layout()
    return fig


def importance(imp: pd.DataFrame, top: int = 15):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 + 0.32 * len(d)))
    ax.barh(d["feature"].map(feature_label), d["importance"], xerr=d["std"], color=SERIES[2], height=0.6,
            error_kw={"ecolor": TEXT_2, "elinewidth": 0.8})
    ax.set_xlabel("Increase in test MAE (£/MWh) when the input is shuffled")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("What the boosted trees rely on (permutation importance)")
    fig.tight_layout()
    return fig
