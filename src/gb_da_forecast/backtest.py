"""Walk-forward backtest: simulate forecasting every delivery day at 09:00 D-1.

    python -m gb_da_forecast.backtest     # 25-100 minutes depending on the machine; writes results/

Protocol, so that the reported test results are honest:

  validation year  1 Nov 2024 - 31 Oct 2025. Used to make every choice: which optional input
                   groups to keep, the boosted trees' settings, and which model is the headline.
                   Models are retrained every 3 months here (to keep tuning fast).
  test year        1 Nov 2025 - latest delivery day. Evaluated with the choices frozen.
                   Models are retrained every month on all delivery days before that month.

At every step a model is trained only on delivery days before the period it forecasts, and its
inputs are those known at the 09:00 D-1 cutoff (features.py), so nothing from the future leaks in.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance

from . import battery
from .config import REPO_ROOT
from .evaluate import diebold_mariano, mae_by, metrics
from .features import FEATURE_GROUPS, build_dataset
from .models import AverageModel, GBMModel, LassoModel, NaiveModel, to_model_units

warnings.filterwarnings("ignore", category=ConvergenceWarning)

VALID_START = dt.date(2024, 11, 1)
TEST_START = dt.date(2025, 11, 1)
RESULTS_DIR = REPO_ROOT / "results"

BASE_GROUPS = ["calendar", "fundamentals", "gas", "price_lags"]
OPTIONAL_GROUPS = ["continental", "imbalance", "gen_mix"]
TARGETS = ["raw", "ratio"]
GBM_GRID = [
    dict(loss=loss, learning_rate=lr, max_iter=iters, max_leaf_nodes=leaves, min_samples_leaf=40)
    for loss, (lr, iters), leaves in itertools.product(
        ["absolute_error", "squared_error"], [(0.05, 600), (0.1, 300)], [15, 63]
    )
]


def features_for(groups: list[str]) -> list[str]:
    return [f for g in groups for f in FEATURE_GROUPS[g]]


def walk_forward(df: pd.DataFrame, make_model, start: dt.date, end: dt.date, months: int) -> pd.Series:
    """Forecast every row with delivery day in [start, end], retraining every ``months`` months."""
    days = pd.to_datetime(df["delivery_day"])
    starts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{months}MS")
    preds = pd.Series(np.nan, index=df.index)
    for i, period_start in enumerate(starts):
        period_end = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp(end) + pd.Timedelta(days=1)
        train = df[(days < period_start) & df["price"].notna()]
        rows = (days >= period_start) & (days < period_end)
        if rows.any():
            preds[rows] = make_model().fit(train).predict(df[rows])
    return preds[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]


def _mae(pred: pd.Series, df: pd.DataFrame) -> float:
    return float((pred - df.loc[pred.index, "price"]).abs().mean())


def run_validation(df: pd.DataFrame, log) -> dict:
    """Make every modelling choice on the validation year."""
    end = TEST_START - dt.timedelta(days=1)
    choices: dict = {}

    # 1. Optional input groups: compare the base set with each optional group added, and all of them.
    candidates = {"base": BASE_GROUPS}
    candidates |= {f"base+{g}": BASE_GROUPS + [g] for g in OPTIONAL_GROUPS}
    candidates["base+all"] = BASE_GROUPS + OPTIONAL_GROUPS
    rows = []
    for (name, groups), target in itertools.product(candidates.items(), TARGETS):
        pred = walk_forward(df, lambda: GBMModel(features_for(groups), target), VALID_START, end, months=3)
        rows.append({"inputs": name, "target": target, "MAE": _mae(pred, df)})
        log(f"  inputs {name:16s} target {target:5s} MAE {rows[-1]['MAE']:.3f}")
    groups_table = pd.DataFrame(rows)
    groups_table.to_csv(RESULTS_DIR / "validation_input_groups.csv", index=False)
    for target in TARGETS:
        best = groups_table[groups_table["target"] == target].sort_values("MAE").iloc[0]["inputs"]
        choices[f"groups_{target}"] = candidates[best]

    # 2. Boosted-tree settings, on the chosen inputs.
    rows = []
    for target, params in itertools.product(TARGETS, GBM_GRID):
        feats = features_for(choices[f"groups_{target}"])
        pred = walk_forward(df, lambda: GBMModel(feats, target, **params), VALID_START, end, months=3)
        rows.append({"target": target, **params, "MAE": _mae(pred, df)})
        log(f"  GBM {target:5s} {params} MAE {rows[-1]['MAE']:.3f}")
    tuning = pd.DataFrame(rows)
    tuning.to_csv(RESULTS_DIR / "validation_gbm_tuning.csv", index=False)
    for target in TARGETS:
        best = tuning[tuning["target"] == target].sort_values("MAE").iloc[0]
        choices[f"gbm_params_{target}"] = {k: best[k] for k in GBM_GRID[0]}
        choices[f"gbm_params_{target}"] = {
            k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
            for k, v in choices[f"gbm_params_{target}"].items()
        }

    # 3. Compare model families on the validation year and pick the headline model.
    preds = {}
    for target in TARGETS:
        feats = features_for(choices[f"groups_{target}"])
        params = choices[f"gbm_params_{target}"]
        preds[f"lasso_{target}"] = walk_forward(df, lambda: LassoModel(feats, target), VALID_START, end, months=3)
        preds[f"gbm_{target}"] = walk_forward(df, lambda: GBMModel(feats, target, **params), VALID_START, end, months=3)
    for kind in ["d1", "d7", "d1_gas"]:
        preds[f"naive_{kind}"] = walk_forward(df, lambda: NaiveModel(kind), VALID_START, end, months=3)
    valid = pd.DataFrame(preds)
    best_lasso = min(["lasso_raw", "lasso_ratio"], key=lambda m: _mae(valid[m], df))
    best_gbm = min(["gbm_raw", "gbm_ratio"], key=lambda m: _mae(valid[m], df))
    valid["average"] = (valid[best_lasso] + valid[best_gbm]) / 2
    table = metrics(df.loc[valid.index, "price"], valid)
    table.to_csv(RESULTS_DIR / "validation_models.csv")
    log(table.round(3).to_string())
    choices["best_lasso"], choices["best_gbm"] = best_lasso, best_gbm
    learned = table.drop(index=[c for c in table.index if c.startswith("naive")])
    choices["headline"] = learned["MAE"].idxmin()
    return choices


def _make(name: str, choices: dict):
    """Factory for each model in the test-year line-up, using the frozen validation choices."""
    if name.startswith("naive_"):
        return lambda: NaiveModel(name.removeprefix("naive_"))
    family, _, target = name.partition("_")
    if family in ("lasso", "gbm"):
        feats = features_for(choices[f"groups_{target}"])
        if family == "lasso":
            return lambda: LassoModel(feats, target)
        return lambda: GBMModel(feats, target, **choices[f"gbm_params_{target}"])
    if name == "average":
        return lambda: AverageModel(_make(choices["best_lasso"], choices)(), _make(choices["best_gbm"], choices)())
    raise ValueError(name)


def run_test(df: pd.DataFrame, choices: dict, log) -> pd.DataFrame:
    end = df.loc[df["price"].notna(), "delivery_day"].max()
    names = ["naive_d1", "naive_d7", "naive_d1_gas", "lasso_raw", "lasso_ratio", "gbm_raw", "gbm_ratio", "average"]
    preds = {}
    for name in names:
        t0 = time.time()
        preds[name] = walk_forward(df, _make(name, choices), TEST_START, end, months=1)
        log(f"  test {name:14s} done in {time.time() - t0:.0f}s")
    out = df.loc[preds["naive_d1"].index, ["start_utc", "delivery_day", "hour", "price"]].copy()
    for name, pred in preds.items():
        out[name] = pred
    return out[out["price"].notna()]


def summarise_test(test: pd.DataFrame, df: pd.DataFrame, choices: dict, log) -> None:
    models = [c for c in test.columns if c not in ("start_utc", "delivery_day", "hour", "price")]
    actual = test["price"]
    table = metrics(actual, test[models])
    table.to_csv(RESULTS_DIR / "test_metrics.csv")
    log(table.round(3).to_string())
    mae_by(actual, test[models], test["hour"]).to_csv(RESULTS_DIR / "test_mae_by_hour.csv")
    month = pd.to_datetime(test["delivery_day"]).dt.to_period("M").astype(str)
    mae_by(actual, test[models], month).to_csv(RESULTS_DIR / "test_mae_by_month.csv")

    headline = choices["headline"]
    rows = []
    for other in [m for m in models if m != headline]:
        stat, p = diebold_mariano(actual, test[headline], test[other], test["delivery_day"])
        rows.append({"model_a": headline, "model_b": other, "dm_stat": stat, "p_value": p})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "test_diebold_mariano.csv", index=False)

    daily = battery.evaluate(test, models)
    daily.to_parquet(RESULTS_DIR / "battery_daily.parquet", index=False)
    summary = battery.summarise(daily, models)
    summary.to_csv(RESULTS_DIR / "battery_summary.csv")
    log(summary.round(3).to_string())

    # What the models learned: fit once on everything before the test year.
    train = df[(pd.to_datetime(df["delivery_day"]) < pd.Timestamp(TEST_START)) & df["price"].notna()]
    target = choices["best_gbm"].removeprefix("gbm_")
    gbm = _make(choices["best_gbm"], choices)().fit(train)
    test_rows = df.loc[test.index]
    X, y = to_model_units(test_rows, gbm.features, target)
    scale = test_rows["gas_ref"].to_numpy(float) if target == "ratio" else np.ones(len(test_rows))
    price = test_rows["price"].to_numpy(float)

    def neg_mae_gbp(estimator, X_, y_):  # score in £/MWh whatever units the model works in
        return -np.mean(np.abs(estimator.predict(X_) * scale - price))

    # Importance = how much the test MAE (£/MWh) rises when one input is randomly shuffled.
    imp = permutation_importance(gbm.model, X, y, scoring=neg_mae_gbp, n_repeats=5, random_state=0)
    pd.DataFrame({"feature": gbm.features, "importance": imp.importances_mean, "std": imp.importances_std}).sort_values(
        "importance", ascending=False
    ).to_csv(RESULTS_DIR / "gbm_permutation_importance.csv", index=False)
    lasso = _make(choices["best_lasso"], choices)().fit(train)
    pd.DataFrame({h: lasso.coefficients(h) for h in range(24)}).to_csv(RESULTS_DIR / "lasso_coefficients.csv")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    df = build_dataset()
    log(f"dataset: {len(df):,} delivery hours, {df['delivery_day'].min()} -> {df['delivery_day'].max()}")
    log("validation year: choosing inputs, settings and the headline model")
    choices = run_validation(df, log)
    log(f"choices: {choices}")
    log("test year: walk-forward with monthly retraining")
    test = run_test(df, choices, log)
    test.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)
    summarise_test(test, df, choices, log)
    info = {
        "validation_period": [str(VALID_START), str(TEST_START - dt.timedelta(days=1))],
        "test_period": [str(TEST_START), str(test["delivery_day"].max())],
        "choices": choices,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    log("done")


if __name__ == "__main__":
    main()
