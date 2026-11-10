"""Generate and execute notebooks/walkthrough.ipynb.

    python notebooks/build_walkthrough.py     # needs the collected data and results/ (about 1 minute)

This script is the single source of the notebook's text and code: edit the cells here and re-run it,
rather than editing the .ipynb, so that a regeneration never overwrites changes. Each entry of CELLS
is ("md", text) for a markdown cell or ("code", text) for a code cell.
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

OUT = Path(__file__).resolve().parent / "walkthrough.ipynb"

CELLS = [
    ("md", """
# GB day-ahead price forecast: walkthrough

This notebook explains, step by step, how the hourly GB day-ahead price is forecast at **09:00 UK
time on D-1** for every hour of **delivery day D** (23:00 UK on D-1 to 23:00 UK on D), and how well it
works. It is written to be read top to bottom; every chart has a short "how to read it" note.

**To reproduce:** `python -m gb_da_forecast.collect` (download data, ~20 min), then
`python -m gb_da_forecast.backtest` (25-100 min depending on the machine, writes `results/`), then run this notebook (~1 min;
it only reads `results/` and the collected data).
"""),
    ("md", """
## Results in brief

Test year: **1 Nov 2025 to 10 Nov 2026** (375 delivery days). Every forecast was made as at 09:00 UK on D-1, models were retrained monthly, and every choice was made beforehand on the validation year (Nov 2024 to Oct 2025).

| Model | MAE (£/MWh) | vs naive | Battery capture |
|---|---|---|---|
| Naive: same hour yesterday | 20.79 | - | 79.0% |
| Lasso (price) | 14.41 | 31% better | 90.7% |
| Lasso (price / gas) | 12.87 | 38% better | 92.6% |
| Boosted trees (price) | 11.70 | 44% better | 92.8% |
| Boosted trees (price / gas), the round-1 headline | 10.27 | 51% better | 92.9% |
| Average of Lasso and trees | 10.69 | 49% better | 94.2% |
| **Final model (round 2): boosted trees (price / gas) without imbalance inputs** | **10.34** | **50% better** | 93.0% |

* The round-1 headline, chosen on the validation year by a rule fixed in advance, **halves the naive error**, and its advantage over every other round-1 model is statistically significant (Diebold-Mariano p < 0.05; p < 0.001 against all but the average).
* **Price / gas beats raw price** in both model families (trees £10.27 vs £11.70; Lasso £12.87 vs £14.41), as the merit-order economics suggested.
* For a 5 kW / 10 kWh home battery trading day-ahead only, scheduling on the learned models' forecasts earns about **£194-202 a year against £169** on the naive forecast, out of £214 possible with perfect foresight.
* The dominant input is the **forecast residual demand** (demand minus wind): the merit order in one number.

* **Round 2** (section 9) tested six pre-registered hypotheses: only dropping the imbalance inputs was adopted (same accuracy, simpler model); the rest moved the error by under 1% or made it worse. The result is the **final model** (last row), the one the forecast command uses. It is statistically indistinguishable from the round-1 headline (p = 0.32), on which the detailed analysis in sections 6-8 was done. Its advantage over the average of Lasso and trees is weaker (p = 0.07), and the average captures more battery value.

To forecast a new day: `python -m gb_da_forecast.forecast YYYY-MM-DD` (after refreshing the data with `collect`).
"""),
    ("code", """
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gb_da_forecast import plots
from gb_da_forecast.config import REPO_ROOT
from gb_da_forecast.features import FEATURE_GROUPS, build_dataset

warnings.filterwarnings("ignore")
plots.set_style()
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
RESULTS = REPO_ROOT / "results"
run_info = json.loads((RESULTS / "run_info.json").read_text())
df = build_dataset()
print(f"{len(df):,} delivery hours, delivery days {df['delivery_day'].min()} to {df['delivery_day'].max()}")
"""),
    ("md", """
## 1. The question

At 09:00 on D-1 we must forecast all 23-25 hourly prices of delivery day D, 20 minutes before the
EPEX 60-minute auction (DA1) closes at 09:20. The first hour of D starts 14 hours after the forecast,
the last one 37 hours after.

| When (example: D = Wed 11 Nov) | What happens |
|---|---|
| Mon, 09:20 / 09:50 | EPEX DA1 / N2EX auctions for delivery day Tue (prices of D-1 become known) |
| **Tue, 09:00** | **Forecast made for Wed** (the cutoff) |
| Tue, 09:20 / 09:50 | Auctions for Wed close |
| Tue 23:00 to Wed 23:00 | Delivery |

**Target.** Historical EPEX DA1 prices are not openly available, so the target is the **N2EX** GB
hourly day-ahead price (Nord Pool's auction for the same product and delivery hours, closing 30
minutes later), published free by NESO. The two have separate order books, and N2EX is coupled with
Norway through the North Sea Link cable while EPEX DA1 is not, so the prices should be close but are
not identical; every accuracy figure here is for N2EX. Swapping in EPEX data only changes the loader.
"""),
    ("md", """
## 2. The data and the economics behind it

Power prices follow the **merit order**: plants are lined up from cheapest to run (wind, solar,
nuclear) to most expensive (old gas "peakers"), and the price each hour is set by the most expensive
plant needed. Two things therefore drive the price:

* **how far up the stack we go**: demand minus wind and solar (*residual demand*);
* **how expensive the gas part of the stack is**: the gas price.
"""),
    ("code", """
plots.price_and_gas(df);
"""),
    ("md", """
**How to read it.** Top: the daily average power price; bottom: the gas price the model uses
(gas day D-3, the latest value certainly published by the cutoff). The two move together: the 2022
energy crisis, the calm 2023-2025, and the climb in late 2026 all show up in both.
"""),
    ("code", """
plots.ratio_over_time(df);
"""),
    ("md", """
**How to read it.** Each point is the median, over a month, of *power price ÷ gas price*. While the
price level moved between ~£70 and ~£200/MWh, this ratio stayed around 2.3-3.2. It is the market's
**implied heat rate**: roughly how many units of gas the price-setting plant burns per unit of power,
plus carbon. The spikes in 2022 are months when the GB gas price briefly collapsed (an LNG glut), so
the ratio's denominator was tiny.

This stability is why the models also try predicting **price ÷ gas** and multiplying back: the model
then learns something stable (which plant sets the price at each hour) instead of having to cope
with price levels it has never seen. The idea follows the structural "bid stack" view of power prices
(Coulon & Howison 2009) and the econometric result that, in a gas-dominated market, the gas
coefficient is the plants' heat rate (Ghelasi & Ziel 2024); the implied market heat rate is also a
standard industry metric (e.g. CPUC 2019).
"""),
    ("md", """
**Why this ratio and not something else?**

| Option | Verdict |
|---|---|
| Raw price | The baseline form. Fine for a linear model (a gas weight scales it); trees struggle when the price level moves |
| Spark spread (price - h x gas) | Additive: needs a fixed heat rate h, and does not let the peak premium grow with gas |
| Log price | Compresses spikes and is multiplicative, but impossible with zero or negative prices (591 negative hours here) |
| asinh(price) | Handles negatives and spikes (Uniejewski, Weron & Ziel 2018), but on its own does not remove gas-driven level shifts |
| Price ÷ recent average price | Purely statistical: reacts to level changes from any cause, but always lags them |
| **Price ÷ gas** | Economically motivated (implied heat rate), stable across regimes, handles negative prices (the ratio just goes negative) |

Two details: the ratio divides by the **7-day mean** gas price (gas days D-9 to D-3), because the
single-day price occasionally collapses (to £4/MWh in June 2022); and errors in the ratio are
multiplied back by the gas price, so they grow when gas is expensive, which is realistic. Both the raw
and the ratio forms are run, and the validation year decides.
"""),
    ("code", """
plots.merit_order(df);
"""),
    ("md", """
**How to read it.** Every hour since Sept 2021 is a point (darker = more hours). Horizontal axis:
forecast residual demand (national demand minus transmission wind, as known at the cutoff). Vertical:
price ÷ gas. Below ~5 GW the ratio sits near zero (wind and nuclear set the price, sometimes negative);
from ~10 GW gas plants set it and the ratio is 2-3; at very high residual demand older, less
efficient plants are needed and it climbs to 4+. This **hockey stick** is the non-linear shape a
straight-line model cannot follow but decision trees can.
"""),
    ("code", """
plots.daily_shape_by_season(df);
"""),
    ("md", """
**How to read it.** The average price at each hour of the day, by season, since 2023. Morning and
evening peaks follow demand; the midday dip in spring and summer is solar. This is why the hour of day
and the solar forecast are inputs.
"""),
    ("md", """
## 3. Inputs: what is known at 09:00 on D-1

Each row of the modelling table is one delivery hour. Every input is something that had been
published by the cutoff:

| Group | Inputs | Why |
|---|---|---|
| Calendar | hour, day of week, bank holiday, month, smooth day-of-year | daily and weekly patterns |
| Fundamentals | demand, transmission wind, embedded solar and wind forecasts for that hour; residual demand and its daily mean/min/max | how far up the merit order each hour goes |
| Gas | gas price of gas day D-3, its 7-day mean | cost of the price-setting plant |
| Price lags | same hour on delivery days D-1, D-2, D-7; D-1's mean, min, max and last hour | persistence; captures what we cannot see (outages, interconnector conditions) |
| Optional context | continental prices of D-1, imbalance prices and generation mix of the last 24 h | kept only if they help on the validation year |

**Point-in-time ("as-of") rule.** Forecast publishers re-issue forecasts many times a day (each
edition is a *vintage*). For each half hour of D we take the latest value published by 09:00 on D-1
(`asof.as_of_cutoff`). Not "the latest vintage": Elexon's demand forecast has a half-hourly vintage
that only reaches ~05:00 tomorrow and a separate 08:45 release for 05:00-05:00, so the two are
stitched per period. Residual demand is demand minus *transmission* wind only: NESO's national
demand is measured on the transmission network, so embedded solar and wind already reduce it.
"""),
    ("code", """
day = df[df["delivery_day"] == pd.Timestamp("2026-07-15").date()]
cols = ["start_utc", "hour", "price", "p_d1", "p_d7", "demand_fc", "wind_fc", "emb_solar_fc", "residual_fc", "gas_sap"]
day[cols].head(8)
"""),
    ("md", """
**How to read it.** The first eight hours of delivery day 15 July 2026 (note it starts at 22:00 UTC =
23:00 BST on 14 July). `price` is the target; everything to its right was known at 09:00 BST on 14 July.
"""),
    ("md", """
## 4. The models: a ladder from simple to flexible

Each rung must beat the one below it to justify its complexity.

1. **Naive baselines** (no learning): same hour yesterday; same hour last week; yesterday scaled by
   the change in gas. Surprisingly hard to beat, and the bar every model is measured against
   (Lago et al. 2021).
2. **Lasso regression**: price = weighted sum of inputs, one model per hour of the day. The Lasso
   penalty sets the weights of useless inputs to exactly zero, so it selects inputs itself. Fully
   readable. A simplified version of the field's "LEAR" benchmark (Uniejewski et al. 2016; named by
   Lago et al. 2021), which also transforms prices and uses all 24 hourly prices of several past days.
   Weakness: straight lines only.
3. **Gradient-boosted decision trees**: hundreds of small if-then trees, each correcting the previous
   ones' errors. Learns bends (the hockey stick) and interactions automatically and copes with missing
   inputs. The strongest practical family on tabular data (M5 competition, Makridakis et al. 2022;
   Grinsztajn et al. 2022). Weakness: cannot extrapolate beyond the range it was trained on, which
   the price ÷ gas target addresses.
4. **Average of Lasso and trees**: combining forecasts often beats each one.

Each learned model comes in two target forms: **price** (£/MWh) and **price ÷ gas** (with all
money-valued inputs also divided by gas, so the model works in "units of gas").

Neural networks, ARIMA and pre-trained foundation models were considered and not used: too little
data for neural networks to pay off (~45,000 hourly rows), poor use of outside drivers for
ARIMA-type models, and opacity for all of them (see README).
"""),
    ("md", """
## 5. How it is tested: a walk-forward backtest

The backtest replays history as if the forecaster had been running live:

| Period | Used for | Retraining |
|---|---|---|
| Sept 2021 - Oct 2024 | training only | - |
| **Validation year**: Nov 2024 - Oct 2025 | every choice: which optional inputs, tree settings, headline model | every 3 months |
| **Test year**: Nov 2025 - latest day | the final evaluation, choices frozen beforehand | every month |

For every forecast month, the model is trained only on delivery days before that month. Making the
choices on the validation year and only then scoring the test year keeps the test results honest:
tuning on the test year would be another form of looking ahead. (The backtest was run twice: a code
review after the first run found a bug in the imbalance inputs, which was fixed and everything
re-run. No choice was made from test results, but the headline moved from £10.37 to £10.27, so
differences of a few tenths of a pound between models are within the noise.)

A third test-year run came later (section 9): six hypotheses raised by the results were written down
with fixed decision rules and committed before running, decided on the validation year only, and the
test year was scored once more for the single adopted change.
"""),
    ("code", """
groups = pd.read_csv(RESULTS / "validation_input_groups.csv").pivot(index="inputs", columns="target", values="MAE")
print("Validation-year MAE (£/MWh) of boosted trees with each set of input groups:")
groups.sort_values("ratio")
"""),
    ("md", """
**What this shows.** Each row is a set of inputs; each column a target form; each cell the validation-year MAE of the boosted trees. Continental prices and the generation mix each helped on their own; imbalance prices on their own made the error worse (ratio form: £9.79 vs £9.56 for the base set); all three together gave the lowest error for both target forms (£9.22 ratio, £9.24 raw), so all were kept. Only these five combinations were tried: continental + generation mix *without* imbalance was not, and is the obvious simplification to test next. The price / gas form beat the raw form in every row.
"""),
    ("code", """
tuning = pd.read_csv(RESULTS / "validation_gbm_tuning.csv")
print("Boosted-tree settings tried on the validation year (best 5 per target):")
tuning.sort_values("MAE").groupby("target").head(5).sort_values(["target", "MAE"])
"""),
    ("md", """
**Settings, in plain words.** `loss`: what the trees minimise, absolute errors (robust to spikes) or
squared errors. `learning_rate` and `max_iter`: how much each new tree corrects, and how many trees.
`max_leaf_nodes`: how many if-then rules per tree. `min_samples_leaf`: no rule may rest on fewer than
40 hours of data.
"""),
    ("code", """
valid = pd.read_csv(RESULTS / "validation_models.csv", index_col=0)
print("Validation year, all models (MAE, RMSE, bias in £/MWh; rMAE relative to 'same hour yesterday'):")
valid.sort_values("MAE")
"""),
    ("code", """
print("Choices frozen for the test year:")
print(json.dumps(run_info["choices"], indent=2))
"""),
    ("md", """
**What was chosen, and why.** All input groups; trees with up to 63 rules each and absolute-error loss (learning rate 0.05 and 600 trees for the ratio form; 0.1 and 300 for the raw form). The best Lasso and the best trees were both the **price / gas** versions, and the headline model is the **boosted trees on price / gas** (validation MAE £9.12 against £19.16 for naive, 52% better). How much each choice mattered on the validation year: tree settings moved the MAE by up to £0.32 (ratio form) or £0.98 (raw form); input groups by up to about £0.6; the target form by £0.33 (best raw £9.44 vs best ratio £9.12). So settings, inputs and target form all mattered by similar, modest amounts. The best settings sat at the edge of the small grid (the largest trees), which prompted hypothesis H2 in round 2 (section 9): larger trees helped by under 1%.
"""),
    ("md", """
## 6. Test-year results

**How to read the metrics.** **MAE**: on average, how far each hourly forecast is from the actual price
(£/MWh). **RMSE**: like MAE but punishes big misses more; RMSE much larger than MAE means a few spikes
dominate. **bias**: average of forecast minus actual (positive = too high). **rMAE**: MAE divided by the
naive "same hour yesterday" MAE; 0.70 = 30% better than naive.
"""),
    ("code", """
test = pd.read_parquet(RESULTS / "test_predictions.parquet")
metrics_table = pd.read_csv(RESULTS / "test_metrics.csv", index_col=0)
print(f"Test year: {run_info['test_period'][0]} to {run_info['test_period'][1]}, {test['delivery_day'].nunique()} delivery days")
metrics_table.sort_values("MAE")
"""),
    ("md", """
**Reading the table.** The headline model's forecasts are on average £10.27/MWh from the actual hourly price, half the naive error. RMSE (£15.6) is about 1.5 times its MAE, against 1.6 times for the naive forecast: big misses still exist, but they matter less. The bias of about -£2.2 means the trees forecast slightly too low on average, and this happens in 10 of the 13 months, so it is not only the autumn 2026 price rise. About a quarter of it comes from spikes: on the 5% most expensive hours the error is £18.1 and the forecast is £11.3 too low on average (**spikes are under-predicted**, typical of models that minimise average error). Round 2 (section 9) traced about half of the bias to the **gas input lagging a rising market**: the model multiplies by a gas price that is 3-9 days old, and on days when gas then rose by more than 3% the forecasts were £3.9 too low on average, against £0.5 when gas fell. Re-running the final model with a gas price with no lag at all (not available in practice) cut the bias from -£2.2 to -£1.1 and the MAE from £10.2 to £9.9. The other half is not explained by gas: forecasts were still £1.7 too low on days when gas barely moved. The test MAE (£10.3) is higher than on the validation year (£9.1) because 2026 prices were higher and more volatile (mean £111/MWh), and errors grow with the price level.
"""),
    ("code", """
dm = pd.read_csv(RESULTS / "test_diebold_mariano.csv")
dm["verdict"] = np.where(dm["p_value"] < 0.05, np.where(dm["dm_stat"] < 0, "A better (significant)", "B better (significant)"), "no significant difference")
dm
"""),
    ("md", """
**How to read it.** The Diebold-Mariano test asks whether the difference in error between the headline
model (A) and each other model (B) is larger than chance. It compares daily average errors and allows
for the fact that consecutive days' errors are related. p < 0.05: the difference is unlikely to be luck.
"""),
    ("code", """
by_hour = pd.read_csv(RESULTS / "test_mae_by_hour.csv", index_col=0)
show = ["naive_d1", run_info["choices"]["best_lasso"], run_info["choices"]["best_gbm"]]
plots.mae_by_hour(by_hour, show);
"""),
    ("md", """
**How to read it.** The naive forecast is worst in the daytime (08:00-15:00, about £25), because yesterday's sun and wind are a poor guide to today's. The models are flatter: the hardest hours are still midday (solar uncertainty) and the 17:00-18:00 evening peak (tight supply, spikes). The Lasso keeps up with the trees overnight but falls behind during the day, where the price relationships bend most.
"""),
    ("code", """
by_month = pd.read_csv(RESULTS / "test_mae_by_month.csv", index_col=0)
plots.mae_by_month(by_month, show);
"""),
    ("md", """
**How to read it.** The boosted trees beat the naive forecast in every month; the Lasso did not always (the price / gas Lasso is worse than naive in November 2026, and the raw Lasso in December 2025 and February 2026). Errors rise in March-April and September-October 2026: windy, volatile shoulder seasons and, in autumn, a fast-rising gas price. November 2026 covers only 10 days, and its last six use a gas price up to nine days old, because the free ONS gas series is published with a lag (a live system would read National Gas directly). Both price / gas models multiply by the gas price, so a stale gas price hurts them; the Lasso visibly so.
"""),
    ("code", """
plots.sample_week(test, ["naive_d1", run_info["choices"]["headline"]], start="2026-10-19");
"""),
    ("md", """
**How to read it.** 19-25 October 2026, a typical week: the model's error is 48% below the naive forecast's, close to the test-year median week (46%). Black is the actual price, blue the naive forecast (yesterday's prices) and green the model, each made at 09:00 the day before. Monday to Friday the model tracks the daily shape and the evening peaks. On Saturday 24 October prices collapsed to about £10-50 as the wind picked up. The naive forecast repeated Friday's £130-190; the model, reading the wind forecast, predicted a fall but not its full depth (£60-90 overnight, about £140 in the morning against £50 actual). The remaining error is weather and market uncertainty that the information available at 09:00 on Friday did not resolve.
"""),
    ("md", """
## 7. Battery value: is the forecast useful for decisions?

Capture uses price forecasts to decide **when to charge and discharge batteries**, and the money
depends on picking the right cheap and expensive hours, which MAE does not measure. A forecast with the
right daily *shape* but the wrong level (say, every hour 20% too high) has a large MAE yet produces
exactly the perfect schedule; a forecast with a lower MAE that puts the evening peak an hour late
discharges at the wrong time. (A constant £ offset, unlike a percentage one, can change the schedule
slightly, because with efficiency losses the battery sells less energy than it buys.)

**Method, for each test day:** (1) plan the profit-maximising schedule of a 5 kW / 10 kWh home battery
(90% round-trip efficiency, at most one cycle a day, empty at the start and end of the day) *as if the
forecast were right*, by solving a small linear program; (2) settle that schedule at the actual prices;
(3) plan again on the actual prices (perfect foresight, the most that was possible). **Capture rate** =
realised profit ÷ perfect-foresight profit over the test year.

**Worked example** (no losses, 1 MW / 2 MWh): actual prices £40 and £45 at 03:00-04:00, £150, £140,
£100 at 18:00-20:00. Perfect foresight buys at 03-04 and sells at 18-19: profit £205. A forecast that
thinks the peak is 19-20 sells at £140 + £100: profit £155, a **76%** capture rate.

Simplifications: day-ahead market only (no intraday or imbalance re-trading), no household load or
solar, no degradation cost, hourly resolution, the battery does not move prices. It measures how useful
a forecast is for decisions; it is not a revenue estimate.
"""),
    ("code", """
battery = pd.read_csv(RESULTS / "battery_summary.csv", index_col=0)
order = battery.drop(index="perfect foresight").sort_values("capture_rate", ascending=False).index.tolist()
plots.battery_capture(battery, order);
battery.assign(capture_rate=lambda t: t["capture_rate"] * 100).rename(
    columns={"capture_rate": "capture rate (%)", "profit_gbp_per_year": "£ per battery per year"})
"""),
    ("md", """
**How to read it.** All learned models capture 91-94% of the perfect-foresight profit, against 79-80% for the naive forecasts. For one 5 kW / 10 kWh battery trading day-ahead only, that is about £194-202 a year against £169 for naive, out of £214 possible: roughly **£25-33 more per battery per year** from better forecasts, which across a fleet of thousands of batteries adds up. The ranking differs from MAE: the average of Lasso and trees captures the most (94.2%) although the trees alone have the lowest MAE, because it gets the order of cheap and expensive hours right slightly more often. The 'same hour last week' naive forecast beats 'same hour yesterday' here despite a much worse MAE (£27.4 vs £20.8). And 'yesterday x gas change' captures exactly the same 79.0% as 'yesterday': multiplying a whole day by one number changes the level but not which hours are cheap or dear, so the battery schedule is identical. Level accuracy and shape accuracy are different things, which is why this measure exists.
"""),
    ("md", """
## 8. What the models learned

**Permutation importance** (boosted trees): shuffle one input at random across the test year and
measure how much the MAE gets worse. A big increase means the model relies on that input; zero means
it could do without it. Error bars show the spread over five shuffles.
"""),
    ("code", """
imp = pd.read_csv(RESULTS / "gbm_permutation_importance.csv")
plots.importance(imp);
"""),
    ("md", """
**How to read it.** Forecast residual demand dominates: shuffling it raises the test MAE by about £11/MWh, more than all other inputs combined. That is the merit order. Next come yesterday's price at the same hour and in its last hour, and the embedded solar forecast (the midday dip). The gas price scores low because in the price / gas model the gas level is divided out of the target and multiplied back afterwards: it does its work outside the trees. Imbalance prices barely matter, consistent with the input-group comparison.
"""),
    ("code", """
coefs = pd.read_csv(RESULTS / "lasso_coefficients.csv", index_col=0)
hour18 = coefs["18"].loc[lambda s: s.abs() > 0].sort_values(key=abs, ascending=False)
print(f"Lasso ({run_info['choices']['best_lasso']}), 18:00 model: {len(hour18)} of {len(coefs)} inputs kept")
hour18.head(12).to_frame("weight per standard deviation")
"""),
    ("md", """
**How to read it.** For numeric inputs, each weight is the change in the prediction (here in units of the gas price) for a one standard-deviation increase in that input, holding the others fixed; for the day-of-week indicators (`dow_1` to `dow_6`) it is the difference from Monday. Inputs not listed were set to exactly zero by the Lasso penalty (the 18:00 model keeps 12 of 39; across all hours the median model keeps 20). At 18:00 the largest weights are residual demand (+), the wind forecast (-), the previous day's French and Belgian prices (+) and yesterday's 18:00 price (+). The signs make economic sense: more residual demand means a higher price, more wind a lower one, and a pricier continent means GB imports less power, which raises the GB price.
"""),
    ("md", """
## 9. Round 2: testing hypotheses raised by round 1

Studying round 1's results raised six specific questions. Each became a hypothesis with a fixed
decision rule, **written down and committed to git before anything was run**
(`docs/round2_hypotheses.md`), so the rules could not be bent to fit the results:

| # | Hypothesis | What prompted it |
|---|---|---|
| H1 | Imbalance prices add nothing; drop them | Alone they made the validation error worse |
| H2 | The trees are too small (127 / 255 leaves, 1,200 trees) | The best settings sat at the edge of the grid |
| H3 | Weighting recent data more fixes the low bias | Forecasts about £2 too low in 10 of 13 months |
| H4 | Yesterday's full 24-hour price shape helps | Battery value depends on shape; the LEAR benchmark uses it |
| H5 | A weighted trees/Lasso blend improves battery value | The 50/50 average had the best battery capture |
| H6 | The 2021-22 crisis teaches wrong lessons; train from 2023 | Prices behaved differently in 2022 |

**Rules.** Each hypothesis changes one thing from the round-1 model and is judged on the validation
year only. A change that adds something is adopted if it cuts validation MAE by at least 1% without
cutting battery capture by more than 0.5 percentage points; the simplification (H1) is adopted if it is
no more than 1% worse. Adopted changes are combined; the blend (H5) is tested on top. Then the test year
is run once more (its third scoring: round 1 scored it twice). H3 and H5 were prompted by test-year
results (the low bias, the average's battery capture); the decisions used only the validation year.
"""),
    ("code", """
r2 = pd.read_csv(RESULTS / "round2_validation.csv", index_col=0)
r2_info = json.loads((RESULTS / "round2_info.json").read_text())
plots.hypothesis_effects(r2);
r2[["MAE", "capture", "mae_change_pct", "capture_change_pp", "passes_rule", "adopted"]]
"""),
    ("md", """
**What this shows.** Each bar is one change, measured against the round-1 model on the validation year (left = better; the dotted line is the 1% threshold a change had to beat). Only **H1 (drop the imbalance-price inputs)** was adopted: it left the error unchanged (+0.0%) and the battery capture almost unchanged (-0.1 points), so the simpler model wins. **H2 (bigger trees)** helped, but by at most 0.6% (127 leaves and 1,200 trees), below the threshold: the round-1 settings were not holding the model back. **H4 (yesterday's 24-hour shape)** helped by 0.7%, also below the threshold. **H3 (weight recent data more)** made the error slightly worse and did not reduce the low bias, so recency weighting is not the fix; following this up found that about half of the bias comes from the gas input lagging rising gas prices (section 6, metrics). **H6 (train only from 2023)** made the error 1.4% worse and the battery capture 0.8 points worse: the 2021-22 crisis years, expressed as price / gas, still teach the model useful patterns, and more data beats a cleaner regime here.
"""),
    ("code", """
blend = pd.read_csv(RESULTS / "round2_blend_validation.csv", index_col=0)
print("H5: blending the final trees with a Lasso (validation year):")
blend[["MAE", "capture", "adopted"]]
"""),
    ("md", """
**What this shows.** Blending the trees with a Lasso does raise battery capture (to 93.9% with 60% trees, from 93.2%), but at the cost of a 6% higher MAE; the blends that keep the MAE within 1% (90% trees) barely change capture (+0.1 points). No weight met both conditions of the rule, so the blend was not adopted. The trade-off is real, though: if battery value were the only goal, a 60/40 blend would be the choice.
"""),
    ("code", """
print("Round-2 choices:", json.dumps({k: r2_info[k] for k in ["adopted_single_changes", "final_trees", "blend_weight_on_trees", "headline"]}, indent=2))
r2_test = pd.read_csv(RESULTS / "round2_test_metrics.csv", index_col=0)
r2_test.rename(index={"naive_d1": "naive: same hour yesterday"})
"""),
    ("code", """
pd.read_csv(RESULTS / "round2_test_diebold_mariano.csv")
"""),
    ("md", """
**Round-2 test result.** The round-2 model (round 1 minus the imbalance inputs) scores MAE £10.34 against the round-1 model's £10.27, and captures 93.0% of the battery value against 92.9%. The Diebold-Mariano test cannot tell the two apart (p = 0.32), and both remain far better than the naive forecast (p < 0.0001). **Conclusion of round 2:** the model is not under-tuned (bigger trees, more inputs and recency weighting each change the error by under 1%), the crisis years are worth keeping, and the imbalance inputs can be dropped without losing accuracy, which removes the messiest data source. The round-2 model is the **final model**. Every number in sections 6-8 refers to the round-1 headline; the two are statistically indistinguishable. Against the round-1 average of Lasso and trees the final model's MAE advantage is weaker than round 1's (p = 0.07 against 0.03), and the average captures more battery value (94.2% against 93.0%): MAE was the selection criterion fixed in advance, and on a battery-value criterion the average would be the candidate.
"""),
    ("md", """
## 10. Limitations and next steps

* **Target proxy.** N2EX stands in for EPEX DA1. The method transfers unchanged; the price gap between the two auctions is unmeasured without EPEX data.
* **Gas data: about half of the bias.** The model uses gas day D-3 from the free ONS series, which itself lags about a week. When gas rises after that, forecasts are too low (about £4/MWh on days when gas rose more than 3%). With a gas price with no lag at all the bias halves (-£2.2 to -£1.1) and the MAE falls from £10.2 to £9.9, which is the ceiling on what fresher gas data can buy. Gas day D-2 (National Gas directly) gains only about £0.1; the forward-looking day-ahead gas price is the better route. The remaining bias is not explained by gas. The forecast command warns when the gas price is stale.
* **Missing drivers.** No carbon price (GOV.UK fortnightly auction prices could be added) and no EUR/GBP rate for the continental prices. Weather enters only through NESO and Elexon forecasts; Open-Meteo's Previous Runs API (2024 onwards) could add temperature honestly.
* **Spikes.** The most expensive hours are under-predicted. A probabilistic version (the same trees with a quantile loss) would give the battery optimiser ranges rather than single numbers.
* **What round 2 ruled out.** Bigger trees, yesterday's 24-hour shape, recency weighting and a trees/Lasso blend each changed the validation error by under 1% (or made it worse); training only on post-crisis data made it worse.
* **Imbalance data.** Elexon's API serves the latest settlement run, so before 2024 the feature uses imbalance prices ~1 hour old at the cutoff and from 2024 ~25 hours old. Round 2 showed the model does not need them.
* **Next steps.** The same pipeline forecasts DA2 (half-hourly, 15:30) and intraday prices with a later cutoff; and a model of the imbalance price and system direction would let the optimiser treat imbalance as a fourth market.
"""),
    ("md", """
## References

1. Lago, Marcjasz, De Schutter & Weron (2021), *Forecasting day-ahead electricity prices: A review of
   state-of-the-art algorithms, best practices and an open-access benchmark*, Applied Energy 293, 116983.
2. Ziel & Weron (2018), *Day-ahead electricity price forecasting with high-dimensional structures:
   Univariate vs. multivariate modeling frameworks*, Energy Economics 70, 396-420.
3. Weron (2014), *Electricity price forecasting: A review of the state-of-the-art with a look into the
   future*, International Journal of Forecasting 30(4), 1030-1081.
4. Makridakis, Spiliotis & Assimakopoulos (2022), *M5 accuracy competition: Results, findings, and
   conclusions*, International Journal of Forecasting 38(4), 1346-1364.
5. Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on
   typical tabular data?*, NeurIPS Datasets and Benchmarks. arXiv:2207.08815.
6. Coulon & Howison (2009), *Stochastic behaviour of the electricity bid stack: from fundamental drivers
   to power prices*, Journal of Energy Markets.
7. Ghelasi & Ziel (2024), *From day-ahead to mid and long-term horizons with econometric electricity
   price forecasting models*, arXiv:2406.00326.
8. California Public Utilities Commission (2019), *Implied Market Heat Rate and Electric Cost
   Overcharge in 2018*.
9. Uniejewski, Weron & Ziel (2018), *Variance stabilizing transformations for electricity spot price
   forecasting*, IEEE Transactions on Power Systems.
10. Diebold & Mariano (1995), *Comparing predictive accuracy*, Journal of Business & Economic Statistics 13(3), 253-263.
11. Uniejewski, Nowotarski & Weron (2016), *Automated variable selection and shrinkage for day-ahead
    electricity price forecasting*, Energies 9(8), 621.
"""),
]


def main() -> None:
    cells = [
        (nbf.v4.new_markdown_cell if kind == "md" else nbf.v4.new_code_cell)(text.strip("\n"))
        for kind, text in CELLS
    ]
    kernel = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": kernel})
    NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
