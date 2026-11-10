# Round 2: hypotheses (written before running)

Round 1 (`results/`, commit 1e59275) chose inputs, settings and the headline model on the validation
year and ran the test year once. Studying its results raised six specific questions. They are tested
here with the rules below, **fixed before any round-2 result was seen** (this file is committed before
`python -m gb_da_forecast.experiments` is run; the git history shows the order).

## Protocol

- **Baseline (B0):** the round-1 headline: boosted trees on price / gas, all input groups, absolute-error
  loss, learning rate 0.05, 600 trees, up to 63 leaves, at least 40 hours per leaf.
- **One change at a time:** each hypothesis changes exactly one thing from B0.
- **Decided on the validation year only** (Nov 2024 - Oct 2025, retrained every 3 months, as in round 1).
  Two measures: MAE, and battery capture rate (5 kW / 10 kWh battery, as in `battery.py`).
- **Test year run once more** at the end, with the round-2 choices frozen. Round-1 and round-2 test
  results are both reported, and the round-2 headline is compared with the round-1 headline
  (Diebold-Mariano test).

## Hypotheses

| # | Hypothesis | Evidence from round 1 | Change from B0 |
|---|---|---|---|
| H1 | Imbalance prices add nothing; dropping them gives a simpler model with the same accuracy | Alone they raised validation MAE (£9.56 -> £9.79); lowest importance; messiest data (late re-publications) | Input groups: base + continental + generation mix |
| H2 | The trees are too small | The best settings sat at the edge of the grid (largest trees tried) | (a) 127 leaves; (b) 255 leaves; (c) 1,200 trees; (d) 127 leaves and 1,200 trees |
| H3 | The model lags a moving market; weighting recent data more reduces the bias | Trees about £2 too low in 10 of 13 test months | Training rows weighted by recency, weight halving every (a) 365 days, (b) 730 days |
| H4 | Yesterday's full daily price shape helps forecast today's shape | The battery value depends on shape; the literature's LEAR benchmark uses all 24 hourly prices of D-1 | Add 24 inputs: the price of each hour of delivery day D-1 |
| H5 | A weighted average of trees and Lasso improves battery value without hurting accuracy | The 50/50 average had the best battery capture (94.2%) but a worse MAE | Final forecast = w x trees + (1 - w) x Lasso, w in {0.5, 0.6, 0.7, 0.8, 0.9} |
| H6 | The 2021-22 gas crisis teaches the model the wrong lessons | Prices and the price / gas ratio behaved differently in 2022 | Train only on delivery days from 1 Jan 2023 |

## Decision rules

- **H2, H3, H4, H6 (changes that add something):** adopt only if validation MAE improves by at least 1%
  **and** validation battery capture falls by no more than 0.5 percentage points. Where several variants
  of one hypothesis pass (H2a-d, H3a-b), take the one with the lowest validation MAE.
- **H1 (a simplification):** adopt if validation MAE is no more than 1% worse **and** battery capture
  falls by no more than 0.5 percentage points (simpler wins a tie).
- **Combination:** apply all adopted changes together and run it on the validation year. If the
  combination is worse (MAE) than the best single adopted change, use that single change instead.
- **H5 (blending, tested on top of the final trees):** adopt the weight w that gives the highest
  validation battery capture among weights that improve capture by at least 0.5 percentage points and
  make MAE no more than 1% worse than the trees alone. The Lasso uses the final input groups.
- **Headline:** the final trees, or the blend if H5 is adopted.

## Results

Recorded in `results/round2_*.csv` and summarised below after the run. (Everything above this heading
is unchanged from the pre-registered version.)

### Validation year (each change vs B0: MAE £9.12, battery capture 93.3%)

| Change | MAE change | Battery capture change | Rule met? |
|---|---|---|---|
| H1 drop imbalance inputs | +0.0% | -0.1 pp | **Yes (simplification): adopted** |
| H2a 127 leaves | -0.2% | +0.4 pp | No (< 1%) |
| H2b 255 leaves | +0.1% | +0.1 pp | No |
| H2c 1,200 trees | -0.4% | +0.1 pp | No (< 1%) |
| H2d 127 leaves + 1,200 trees | -0.6% | +0.3 pp | No (< 1%) |
| H3a recency half-life 1 year | +0.8% | +0.1 pp | No (worse) |
| H3b recency half-life 2 years | +0.2% | +0.0 pp | No (worse) |
| H4 add D-1 24-hour shape | -0.7% | -0.2 pp | No (< 1%) |
| H6 train from 2023 | +1.4% | -0.8 pp | No (worse) |

Only H1 was adopted, so no combination step was needed. **H5** (blend on top of the H1 trees): the best
battery capture was 93.9% at 60% trees (+0.7 pp) but with MAE 6.1% worse; at 90% trees MAE was within
0.1% but capture rose only 0.1 pp. No weight met both conditions: not adopted.

### Test year (run once with the round-2 choice)

| Model | MAE | Battery capture | £ per battery per year |
|---|---|---|---|
| Naive: same hour yesterday | 20.79 | 79.0% | 169 |
| Round-1 headline | 10.27 | 92.9% | 199 |
| **Round-2 model (round 1 without imbalance inputs)** | **10.34** | 93.0% | 199 |

Diebold-Mariano, round 2 vs round 1: p = 0.32 (no detectable difference); vs naive: p < 0.0001.

### Conclusions

- The round-1 model was not under-tuned: bigger trees, yesterday's full price shape and recency
  weighting each moved the validation error by under 1%.
- The 2021-22 crisis years are worth training on (dropping them made things worse): in price / gas
  terms they still teach the model useful patterns.
- The imbalance inputs can be dropped with no loss of accuracy, removing the messiest data source.
- H3 failing prompted a follow-up on the low bias (about -£2/MWh): it is not the model lagging the
  market. On test days when gas rose by more than 3% after the model's gas input (gas day D-3, 7-day
  mean), forecasts were £3.9/MWh too low on average, against £0.5 when gas fell. The bias comes mainly
  from the gas input's lag in a rising market (plus some right-skew from spikes), so the remedy is
  fresher gas data, not a modelling change.
