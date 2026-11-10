"""Round 2: pre-registered hypotheses, decided on the validation year, then one more test-year run.

    python -m gb_da_forecast.experiments            # writes results/round2_* (20-60 min, machine-dependent)
    python -m gb_da_forecast.experiments --smoke    # 1-month check of the machinery, no test year

The hypotheses and every decision rule applied here are written in docs/round2_hypotheses.md, which
was committed before this script was first run. In short:

  B0       the round-1 headline (boosted trees on price / gas), the reference for every comparison
  H1       drop the imbalance-price inputs                       (simplification: adopt if not worse)
  H2a-d    bigger trees: 127 / 255 leaves, 1,200 trees, both      (adopt if >= 1% better)
  H3a-b    weight recent training data more (half-life 1 / 2 y)   (adopt if >= 1% better)
  H4       add yesterday's full 24-hour price shape               (adopt if >= 1% better)
  H6       train only on delivery days from 2023                  (adopt if >= 1% better)
  H5       blend w x trees + (1 - w) x Lasso, tested on top of the final trees
"Better" is validation MAE; every adoption also requires that validation battery capture does not
fall by more than 0.5 percentage points.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from . import battery
from .backtest import RESULTS_DIR, TEST_START, VALID_START, features_for, walk_forward
from .evaluate import diebold_mariano, metrics
from .features import build_dataset
from .models import GBMModel, LassoModel

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MAE_GAIN = 0.01  # an added change must improve validation MAE by at least 1%
SIMPLIFICATION_TOLERANCE = 0.01  # a simplification may be at most 1% worse
CAPTURE_TOLERANCE = 0.005  # battery capture may fall by at most 0.5 percentage points
BLEND_WEIGHTS = [0.5, 0.6, 0.7, 0.8, 0.9]  # weight on the trees in H5


def baseline_spec() -> dict:
    """The round-1 headline, read from the frozen round-1 choices."""
    choices = json.loads((RESULTS_DIR / "run_info.json").read_text())["choices"]
    return {
        "groups": list(choices["groups_ratio"]),
        "target": "ratio",
        "params": dict(choices["gbm_params_ratio"]),
        "half_life_days": None,
        "train_start": None,
    }


def single_changes(base: dict) -> dict[str, dict]:
    """Each variant changes exactly one thing from the baseline."""
    return {
        "B0 baseline": {},
        "H1 drop imbalance": {"groups": [g for g in base["groups"] if g != "imbalance"]},
        "H2a 127 leaves": {"params": {"max_leaf_nodes": 127}},
        "H2b 255 leaves": {"params": {"max_leaf_nodes": 255}},
        "H2c 1200 trees": {"params": {"max_iter": 1200}},
        "H2d 127 leaves + 1200 trees": {"params": {"max_leaf_nodes": 127, "max_iter": 1200}},
        "H3a recency half-life 1 year": {"half_life_days": 365},
        "H3b recency half-life 2 years": {"half_life_days": 730},
        "H4 add D-1 price shape": {"groups": base["groups"] + ["d1_shape"]},
        "H6 train from 2023": {"train_start": dt.date(2023, 1, 1)},
    }


def apply(spec: dict, change: dict) -> dict:
    out = copy.deepcopy(spec)
    for key, value in change.items():
        if key == "params":
            out["params"].update(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def trees_factory(spec: dict):
    feats = features_for(spec["groups"])
    return lambda: GBMModel(
        feats, spec["target"], half_life_days=spec["half_life_days"], train_start=spec["train_start"], **spec["params"]
    )


def score(df: pd.DataFrame, preds: dict[str, pd.Series]) -> pd.DataFrame:
    """MAE, RMSE, bias and battery capture for each set of predictions (same rows)."""
    rows = next(iter(preds.values())).index
    frame = df.loc[rows, ["start_utc", "delivery_day", "price"]].copy()
    for name, p in preds.items():
        frame[name] = p
    frame = frame[frame["price"].notna()]
    table = metrics(frame["price"], frame[list(preds)], reference=list(preds)[0])
    daily = battery.evaluate(frame, list(preds))
    table["capture"] = [daily[m].sum() / daily["perfect"].sum() for m in preds]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="one validation month, B0 and H1 only, no test year")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    df = build_dataset()
    valid_end = TEST_START - dt.timedelta(days=1)
    if args.smoke:
        valid_end = VALID_START + dt.timedelta(days=30)
    base = baseline_spec()
    changes = single_changes(base)
    if args.smoke:
        changes = {k: v for k, v in changes.items() if k in ("B0 baseline", "H1 drop imbalance")}

    # 1. Each single change on the validation year.
    preds, specs = {}, {}
    for name, change in changes.items():
        specs[name] = apply(base, change)
        preds[name] = walk_forward(df, trees_factory(specs[name]), VALID_START, valid_end, months=3)
        log(f"validation {name:32s} MAE {(preds[name] - df.loc[preds[name].index, 'price']).abs().mean():.3f}")
    table = score(df, preds)
    b0 = table.loc["B0 baseline"]
    table["mae_change_pct"] = (table["MAE"] / b0["MAE"] - 1) * 100
    table["capture_change_pp"] = (table["capture"] - b0["capture"]) * 100

    def passes(name: str) -> bool:
        r = table.loc[name]
        capture_ok = b0["capture"] - r["capture"] <= CAPTURE_TOLERANCE
        if name.startswith("H1"):
            return r["MAE"] <= b0["MAE"] * (1 + SIMPLIFICATION_TOLERANCE) and capture_ok
        return r["MAE"] <= b0["MAE"] * (1 - MAE_GAIN) and capture_ok

    table["passes_rule"] = [False if n.startswith("B0") else passes(n) for n in table.index]
    adopted = {}
    for hyp in ["H1", "H2", "H3", "H4", "H6"]:
        winners = [n for n in table.index if n.startswith(hyp) and table.loc[n, "passes_rule"]]
        if winners:
            adopted[hyp] = min(winners, key=lambda n: table.loc[n, "MAE"])
    table["adopted"] = [n in adopted.values() for n in table.index]
    log("single changes:\n" + table.round(4).to_string())
    log(f"adopted: {adopted}")

    # 2. Combine the adopted changes (if more than one), else take the single winner or the baseline.
    if not adopted:
        final_name = "B0 baseline"
    elif len(adopted) == 1:
        final_name = next(iter(adopted.values()))
    else:
        combined = copy.deepcopy(base)
        if "H1" in adopted:
            combined["groups"] = [g for g in combined["groups"] if g != "imbalance"]
        if "H4" in adopted:
            combined["groups"] = combined["groups"] + ["d1_shape"]
        for hyp in ("H2", "H3", "H6"):
            if hyp in adopted:
                change = {k: v for k, v in changes[adopted[hyp]].items() if k != "groups"}
                combined = apply(combined, change)
        specs["C combined"] = combined
        preds["C combined"] = walk_forward(df, trees_factory(combined), VALID_START, valid_end, months=3)
        comb = score(df, {"B0 baseline": preds["B0 baseline"], "C combined": preds["C combined"]}).loc["C combined"]
        table.loc["C combined", ["MAE", "RMSE", "bias", "capture"]] = comb[["MAE", "RMSE", "bias", "capture"]]
        table.loc["C combined", "mae_change_pct"] = (comb["MAE"] / b0["MAE"] - 1) * 100
        table.loc["C combined", "capture_change_pp"] = (comb["capture"] - b0["capture"]) * 100
        best_single = min(adopted.values(), key=lambda n: table.loc[n, "MAE"])
        final_name = "C combined" if comb["MAE"] <= table.loc[best_single, "MAE"] else best_single
        log(f"combined MAE {comb['MAE']:.3f} vs best single ({best_single}) {table.loc[best_single, 'MAE']:.3f}")
    final = specs[final_name]
    table.to_csv(args.out / "round2_validation.csv")
    log(f"final trees: {final_name}")

    # 3. H5: blend the final trees with a Lasso on the same input groups.
    lasso_feats = features_for(final["groups"])
    lasso_valid = walk_forward(df, lambda: LassoModel(lasso_feats, "ratio"), VALID_START, valid_end, months=3)
    blends = {"trees only": preds[final_name], "lasso only": lasso_valid}
    for w in BLEND_WEIGHTS:
        blends[f"w={w:.1f}"] = w * preds[final_name] + (1 - w) * lasso_valid
    blend_table = score(df, blends)
    trees = blend_table.loc["trees only"]
    ok = blend_table[
        blend_table.index.str.startswith("w=")
        & (blend_table["capture"] >= trees["capture"] + CAPTURE_TOLERANCE)
        & (blend_table["MAE"] <= trees["MAE"] * (1 + MAE_GAIN))
    ]
    blend_w = float(ok["capture"].idxmax().removeprefix("w=")) if len(ok) else None
    blend_table["adopted"] = [len(ok) > 0 and i == ok["capture"].idxmax() for i in blend_table.index]
    blend_table.to_csv(args.out / "round2_blend_validation.csv")
    log("H5 blends:\n" + blend_table.round(4).to_string())
    log(f"H5 adopted weight on trees: {blend_w}")

    info = {
        "plan": "docs/round2_hypotheses.md",
        "validation_period": [str(VALID_START), str(valid_end)],
        "baseline": {k: str(v) if isinstance(v, dt.date) else v for k, v in base.items()},
        "adopted_single_changes": adopted,
        "final_trees": final_name,
        "final_spec": {k: str(v) if isinstance(v, dt.date) else v for k, v in final.items()},
        "blend_weight_on_trees": blend_w,
        "headline": "round2_blend" if blend_w is not None else "round2_trees",
    }
    if args.smoke:
        (args.out / "round2_info.json").write_text(json.dumps(info, indent=2))
        log("smoke run done (no test year)")
        return

    # 4. The test year, once, with everything frozen.
    end = df.loc[df["price"].notna(), "delivery_day"].max()
    test_preds = {"round2_trees": walk_forward(df, trees_factory(final), TEST_START, end, months=1)}
    log("test: round-2 trees done")
    if blend_w is not None:
        lasso_test = walk_forward(df, lambda: LassoModel(lasso_feats, "ratio"), TEST_START, end, months=1)
        test_preds["round2_lasso"] = lasso_test
        test_preds["round2_blend"] = blend_w * test_preds["round2_trees"] + (1 - blend_w) * lasso_test
        log("test: round-2 Lasso and blend done")
    round1 = pd.read_parquet(RESULTS_DIR / "test_predictions.parquet")[["start_utc", "naive_d1", "gbm_ratio", "average"]]
    round1 = round1.rename(columns={"gbm_ratio": "round1_headline", "average": "round1_average"})
    frame = df.loc[test_preds["round2_trees"].index, ["start_utc", "delivery_day", "hour", "price"]].copy()
    for name, p in test_preds.items():
        frame[name] = p
    frame = frame[frame["price"].notna()].merge(round1, on="start_utc", how="left")
    frame.to_parquet(args.out / "round2_test_predictions.parquet", index=False)
    models = ["naive_d1", "round1_headline", "round1_average", *test_preds]
    test_table = metrics(frame["price"], frame[models])
    daily = battery.evaluate(frame, models)
    test_table["capture"] = [daily[m].sum() / daily["perfect"].sum() for m in models]
    test_table["gbp_per_battery_year"] = [daily[m].sum() / len(daily) * 365 for m in models]
    test_table.to_csv(args.out / "round2_test_metrics.csv")
    log("test year:\n" + test_table.round(4).to_string())
    headline = info["headline"]
    dm_rows = []
    for other in ["round1_headline", "round1_average", "naive_d1"]:
        stat, p = diebold_mariano(frame["price"], frame[headline], frame[other], frame["delivery_day"])
        dm_rows.append({"model_a": headline, "model_b": other, "dm_stat": stat, "p_value": p})
    pd.DataFrame(dm_rows).to_csv(args.out / "round2_test_diebold_mariano.csv", index=False)
    log("DM:\n" + pd.DataFrame(dm_rows).round(4).to_string())
    info["test_period"] = [str(TEST_START), str(end)]
    info["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    (args.out / "round2_info.json").write_text(json.dumps(info, indent=2))
    log("done")


if __name__ == "__main__":
    main()
