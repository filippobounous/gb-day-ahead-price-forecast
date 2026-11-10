# GB day-ahead (DA1) price forecast

Forecasting the hourly GB day-ahead auction price for every hour of delivery day D, as it would
be done at 09:00 UK time on D-1, 20 minutes before the EPEX 60-minute auction (DA1) closes at 09:20.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,dev]"
python -m gb_da_forecast.collect            # download all raw data into data/raw/ (~20 min the first time)
python -m gb_da_forecast.data.check         # coverage and sanity checks
python -m gb_da_forecast.backtest           # validation + test backtest, writes results/ (~25 min)
python -m gb_da_forecast.forecast 2026-11-11  # forecast one delivery day as at 09:00 the day before
python -m pytest -q                         # tests of the time logic, as-of selection, battery, features
```

Then open `notebooks/walkthrough.ipynb`: a step-by-step explanation of the data, the models and the
results, with a "how to read it" note under every chart.

Re-running `collect` only fetches what is missing or not yet final. `--only <source ...>` and
`--start YYYY-MM-DD` restrict it. (A run with an `--end` earlier than before temporarily shrinks
the current year's data until the next default run.)

## Code map

| Module | Purpose |
|---|---|
| `config.py` | Paths, time zone and fixed modelling choices (e.g. the gas price lag) |
| `timeutils.py` | GB time conventions: settlement periods, delivery days (23:00-23:00 UK), the 09:00 D-1 cutoff |
| `asof.py` | Point-in-time selection of input forecasts: the guard against look-ahead |
| `data/elexon.py`, `data/neso.py`, `data/external.py` | One fetch function per source, documenting what each dataset means |
| `data/_store.py`, `data/_http.py` | Plumbing: chunked, resumable downloads with retries |
| `collect.py` | Command line entry point for downloading |
| `data/check.py` | Coverage, gap and time-basis checks |
| `features.py` | The modelling table: one row per delivery hour, every input as known at the cutoff |
| `models.py` | Naive baselines, Lasso (one model per hour), gradient-boosted trees, their average |
| `backtest.py` | Walk-forward backtest: validation year for choices, untouched test year |
| `evaluate.py` | MAE, RMSE, bias, relative MAE, Diebold-Mariano test |
| `battery.py` | Battery value of a forecast (linear-program schedule, capture rate) |
| `forecast.py` | Command line forecast of one delivery day |
| `plots.py` | Charts used in the notebook |

## Key terms

| Term | Meaning |
|---|---|
| **Delivery day D** | The day whose prices are forecast. A GB day-ahead delivery day runs **23:00 UK on D-1 to 23:00 UK on D** (the CET calendar day, a convention from when GB's auctions were coupled to Europe's), so it has 23, 24 or 25 hours. |
| **D-1, D-2, ...** | Days counted back from the delivery day. "Prices for D-1" means the prices of *delivery day* D-1, not trades made on D-1. |
| **Cutoff** | 09:00 UK time on D-1, when the forecast for D is made. Only information published by then may be used. |
| **Settlement period** | A half hour, the unit of GB settlement. Numbered from 1 at UK-local midnight, so a settlement day has 46, 48 or 50 periods. |
| **Vintage** | One published edition of a forecast: a snapshot released at a given time with values for many future periods. Publishers re-issue forecasts many times a day, so each future period appears in many vintages. |
| **As-of selection** | For each period of D, taking the latest value published by the cutoff (`asof.as_of_cutoff`). |
| **Gas day** | The gas industry's day: 05:00 to 05:00 UK time. |
| **SAP** | System Average Price of gas: the realised average price of gas traded on National Gas's On-the-day Commodity Market for one gas day. A price that happened, not a forecast. |
| **N2EX / EPEX** | Two power exchanges (Nord Pool's GB market and EPEX SPOT). Each runs its own GB hourly day-ahead auction for the same delivery day, with separate order books. |
| **EPEX market index (`MID`)** | Elexon's Market Index Data for EPEX (`APXMIDP`): per half hour, the volume and volume-weighted average price of qualifying short-term EPEX trades for delivery in that half hour, mostly continuous intraday trading. Elexon uses it in imbalance pricing as the fallback price when there are no balancing actions to price from. |
| **Imbalance price** | The price at which Elexon settles each party's difference between contracted and metered volume, per half hour. Derived from the marginal cost of the balancing actions NESO took in that period; a single price for being long or short since 2015. |
| **energy-charts** | A free data service from Fraunhofer ISE that republishes European market data (CC BY 4.0); used here for neighbouring countries' day-ahead prices. |

## Auction timeline

Example: delivery day **D = Wednesday 11 Nov 2026** (winter, so UK time = UTC). It runs from
23:00 Tuesday to 23:00 Wednesday.

| When | What happens |
|---|---|
| Mon 9 Nov, 09:20 / 09:50 | EPEX DA1 / N2EX auctions for delivery day **Tue 10 Nov** (23:00 Mon to 23:00 Tue). Results out by ~10:00 Monday |
| **Tue 10 Nov, 09:00** | **Cutoff: we forecast all 24 hourly prices of Wed 11 Nov** |
| Tue 10 Nov, 09:20 | EPEX DA1 gate closes for Wed 11 Nov |
| Tue 10 Nov, 09:50 | N2EX gate closes for Wed 11 Nov; results by ~10:00 |
| Tue 10 Nov, ~11:00 | Continental day-ahead auction for Wed 11 Nov (00:00-24:00 CET = 23:00 Tue to 23:00 Wed UK) |
| Tue 23:00 to Wed 23:00 | Delivery |

In each auction, participants submit for every hour how much they would buy or sell at each price
before gate closure. The exchange stacks sell orders from cheapest up and buy orders from highest
down; where the curves cross is that hour's **clearing price**, paid or received by everyone matched
(pay-as-clear).

At the cutoff (Tue 09:00) the prices of delivery day Tue 10 Nov and earlier are known (set by
Monday's auction, i.e. "cleared on D-2"); Wednesday's prices are the target. The first hour of D
starts 14 hours after the cutoff and the last one 37 hours after.

## Data

### Target

Historical EPEX DA1 prices are not openly available (EPEX sells them; ENTSO-E stopped publishing GB
data after Brexit). The target is therefore the **N2EX GB hourly day-ahead auction price**, which
NESO republishes for free from delivery day 21 Sep 2021. It is the other GB hourly day-ahead
auction: same product and delivery hours, pay-as-clear, closing at 09:50 instead of 09:20. Both
auctions clear against the same fundamentals half an hour apart, so their prices should be close,
but the gap cannot be measured without EPEX data. The code is target-agnostic: swapping in EPEX
DA1 history only changes the loader.

Established from the data (`python -m gb_da_forecast.data.check`):

- **NESO's N2EX hour labels are in UTC.** NESO labels each price with a date and an hour but no
  time zone. Every date has 24 rows, including clock-change days, which already suggests UTC. To
  confirm, the N2EX series was placed on the timeline three ways (labels read as UTC, then moved one
  hour earlier and later) and, for each, its hour-to-hour price changes were correlated (Pearson)
  with those of Elexon's EPEX market index (hourly mean of the half-hourly index, zero-volume
  periods dropped; Elexon timestamps are UTC). Price changes are used rather than levels because
  prices are sticky from hour to hour, so a misaligned series still correlates well in levels
  (0.937 vs 0.923); changes such as the jump into the evening peak only line up when the
  timestamps do.

  | Season | Labels moved -1h | Labels read as UTC | Labels moved +1h |
  |---|---|---|---|
  | Summer (BST, UK = UTC+1) | 0.40 | **0.56** | 0.37 |
  | Winter (GMT, UK = UTC) | 0.16 | **0.54** | 0.23 |

  If the labels were UK local time, the -1h column would win in summer. Winter is the control, where
  UTC and UK time coincide and the method correctly picks 0h. The correlations are well below 1
  because the two series are different markets (an auction the day before vs trading close to
  delivery); only the comparison across shifts matters.
- **A delivery day runs 23:00-23:00 UK time:** the first one starts at 22:00 UTC (23:00 BST) on
  20 Sep 2021, and the latest auction ends at 23:00 UTC.
- All 1,877 delivery days are complete. Yearly mean price (£/MWh): 2022 204, 2023 94, 2024 73,
  2025 81, 2026 111, so the level shifts a lot with the gas price.

### Inputs and what is known at 09:00 on D-1

Every *input* forecast (demand, wind, solar) is taken, for each half hour of delivery day D, as
the most recent value its publisher had released by 09:00 UK on D-1 (`asof.as_of_cutoff`).
Publishing continues after 09:00 right up to delivery; anything released after the cutoff is
ignored, because it would not have been available.

"The latest vintage" is not enough, because vintages have different horizons. Elexon's demand
forecast, for example, publishes a new vintage every 30 minutes covering until ~05:00 tomorrow, plus
a separate day-ahead release at 08:45 UK covering 05:00 tomorrow to 05:00 the day after. For
delivery day 2 Mar 2023 (23:00 1 Mar to 23:00 2 Mar UTC):

| Vintage published (UTC, 1 Mar) | Covers | Used for |
|---|---|---|
| 08:15 | 08:30 today to 04:30 tomorrow | The first 12 half hours of D (23:00-04:30) |
| 08:45 (day-ahead release) | 05:00 tomorrow to 04:30 the day after | The remaining 36 half hours (05:00-22:30) |
| 09:15 | 09:30 today to 04:30 tomorrow | Nothing: published after the cutoff |

| Source | Dataset | Use | Known at the cutoff |
|---|---|---|---|
| Elexon `NDF` | National demand forecast | Demand for D | Stitched per half hour from two publications (above) |
| Elexon `WINDFOR` | Transmission-connected wind forecast | Wind for D | ~8 vintages a day, hourly values |
| NESO embedded forecast | Wind and solar on distribution networks | Solar, small wind for D | A new vintage every hour |
| ONS / National Gas | Gas SAP, daily | Fuel cost | Realised SAP of gas day D-3 (see below) |
| gov.uk | Bank holidays | Calendar | Known in advance |
| N2EX (target) | Hourly day-ahead price | Lagged prices | Delivery days up to and including D-1 (cleared on D-2) |
| Elexon `FUELHH`, `MID`; PV_Live | Generation mix and interconnector flows, short-term traded price, solar outturn | Lagged context | Up to ~08:30 on D-1 |
| Elexon system prices | Imbalance price | Lagged context | Only periods >25 h old: stored values are a later settlement run |
| energy-charts | FR, NL, BE, NO2, DK1 (+ DE-LU) day-ahead prices | Lagged context | Delivery days up to D-1 (they clear ~11:00 UK, after GB's auction) |
| Open-Meteo | Weather at 9 onshore and offshore points | Exploration only | Not as-of: see below |

### Gas price

SAP is a realised price for a whole gas day (05:00-05:00 UK). At the 09:00 D-1 cutoff:

| Gas day | Runs | Status at 09:00 on D-1 |
|---|---|---|
| D-1 | 05:00 D-1 to 05:00 D | Still in progress: its SAP does not exist yet |
| D-2 | 05:00 D-2 to 05:00 D-1 | Ended 4 hours earlier. Computable, but publication by 09:00 is unverified |
| **D-3** | 05:00 D-3 to 05:00 D-2 | Ended more than a day earlier: certainly published. **Default** (`config.GAS_SAP_LAG_DAYS`) |

D-3 is chosen because it was guaranteed to be known, at the cost of being a day older. In our data
the correlation between the daily average power price of D and SAP is 0.896 using gas day D-2 and
0.869 using D-3, so D-2 is reported as a sensitivity check. Gas usually moves little from day to day
(median change £0.9/MWh) but occasionally jumps (10% of days move by more than £5.7/MWh).

Traders would use the **day-ahead gas price** (traded on D-1 for gas day D), which looks forward
rather than back, but that series is not freely available; past SAP is the proxy.

### Timing traps (avoiding look-ahead)

- **Use the forecast that existed at the cutoff**, not the latest one or the outturn. All vintages
  are stored and selected by publish time, per target period (see the demand example above).
- **The cutoff is in UK local time.** In summer 09:00 BST is 08:00 UTC; a cutoff fixed in UTC
  would leak an extra hour of information for half the year.
- **Timestamps are not all UTC at source.** N2EX labels are UTC; NESO's embedded forecast stamps are
  UK local time. Both are converted to UTC on download.
- **"Prices for D-1" means delivery day D-1, not UTC date D-1.** NESO labels N2EX rows by UTC date.
  In the timeline above, "UTC date Tue 10 Nov" includes 23:00-24:00 Tuesday, which belongs to
  delivery day Wed 11 Nov, the target itself (two hours in summer). Taking "all rows labelled
  Tuesday" as yesterday's prices would leak part of the answer, so every price is tagged with its
  delivery day.
- **Elexon's day-ahead wind and solar forecast (`DGWS`) is published ~16:45**, after the auction,
  so it is not used. **Continental day-ahead prices clear after GB's DA1**, so only earlier days
  are usable.
- **Open-Meteo's historical forecast archive** is built from the first hours of each model run,
  i.e. runs that started after the cutoff, so it is look-ahead and not used as a feature. Open-Meteo's
  Previous Runs API (2024 onwards) would provide honest weather forecasts.

### Data gaps (all in the sources, not chunk edges; handled as missing values)

- Demand forecast: 13 delivery days (mostly late 2021) lack the 08:45 release, so 36 of 48 periods
  are missing.
- Embedded forecast: 3 delivery days (11-13 May 2025, a NESO outage).
- EPEX market index: 2,312 missing half-hours (including a 4-day gap). Generation mix: 87.
  Imbalance prices: 11.

### Known limitations

- **No free daily carbon (UKA) price.** Daily UK ETS allowance prices come from ICE futures and are
  licensed (e.g. via Montel or Argus). GOV.UK does publish the fortnightly UK ETS auction clearing
  prices [3], which could be added as a coarse series; carbon also has a fixed £18/t Carbon Price
  Support and a £22/t auction reserve price [4]. Carbon is a smaller part of a gas plant's marginal
  cost than fuel: at ~0.37 tCO2/MWh and, say, £40/t + £18/t, it adds ~£21/MWh, against ~£174/MWh of
  fuel at the latest SAP (£87/MWh of gas, 50% efficiency). Slow carbon drifts are absorbed by
  regular retraining.
- **Gas is the main price driver.** Gas set the GB power price about 84% of the time in recent years
  despite producing under half the electricity [1], and over 80% of hours in 2021 [2]. In our data the
  daily power-gas correlation is 0.91 (0.97 on monthly means), and the ratio of power price to gas
  price stays between 2.6 and 2.9 every year, roughly what a gas plant's efficiency plus carbon would
  imply. This is why the model works with the gas price explicitly.
- The ONS gas series is published weekly with about a week's lag. Fine for backtesting; a live
  system would read National Gas's data portal directly.
- Continental prices are in EUR and no exchange rate is collected.

## Modelling

### Inputs

One row per delivery hour (`features.py`), every input as known at 09:00 UK on D-1:

| Group | Inputs | Why |
|---|---|---|
| Calendar | Hour, day of week, bank holiday, month, smooth day-of-year | Daily and weekly patterns |
| Fundamentals | Demand, transmission wind, embedded solar and wind forecasts for the hour; residual demand (demand minus transmission wind) and its daily mean/min/max | How far up the merit order each hour goes |
| Gas | Gas SAP of gas day D-3; its 7-day mean (gas days D-9 to D-3) | Cost of the price-setting plant |
| Price lags | Same hour on delivery days D-1, D-2, D-7; D-1's mean, min, max and last hour | Persistence; captures outages and interconnector conditions we cannot see |
| Optional context | Continental prices of D-1; imbalance price and generation mix of the last 24 h available | Kept only if they help on the validation year |

Residual demand subtracts only *transmission* wind: NESO's national demand is measured on the
transmission network, so embedded solar and wind already reduce it. The ratio target divides by the
7-day mean gas price rather than a single day, because single-day SAP occasionally collapses (an LNG
glut in 2022 took it to £4/MWh).

### Models

A ladder from simple to flexible; each rung must beat the one below to justify itself.

| Model | How it works | Pros | Cons |
|---|---|---|---|
| Naive baselines | Same hour yesterday; same hour last week; yesterday x gas change | Free, hard to beat, the standard bar | Ignore weather and demand |
| Lasso (one per hour) | Weighted sum of inputs; the penalty sets useless weights to zero | Readable, fast; the field's "LEAR" benchmark [8, 9] | Straight lines only |
| Gradient-boosted trees | Hundreds of small if-then trees, each correcting the previous ones | Learns the merit-order bend and interactions; handles missing inputs; strongest on tabular data [10, 11] | Cannot extrapolate beyond the training range |
| Average | Mean of the best Lasso and best trees | Combining forecasts often beats each | Less readable |

Not used: neural networks (too little data, about 45,000 hourly rows, to repay their tuning cost and
opacity), ARIMA-type models (poor use of outside drivers such as wind forecasts), pre-trained
time-series foundation models (weak with outside drivers, opaque). scikit-learn's
`HistGradientBoostingRegressor` implements the same algorithm family as LightGBM without a system
dependency.

### Why price / gas

When a gas plant sets the price, price ~ heat rate x gas price + carbon cost, so price / gas is
roughly the **implied market heat rate** of the price-setting plant: a standard industry metric [14],
the one-fuel version of the structural bid-stack view of power prices [12], and what an econometric
model's gas coefficient estimates in a gas-dominated market [13]. It depends on residual demand and
the hour, not on the gas level: the yearly median stayed between 2.6 and 2.9 while yearly mean prices
ranged from £73 to £204/MWh. Predicting the ratio (with all money-valued inputs also divided by gas)
lets the trees learn something stable instead of having to extrapolate to new price levels; the
prediction is multiplied back by the gas price. Alternatives (raw price, spark spread, log price,
asinh [15], scaling by recent prices) are discussed in the notebook; raw price and the ratio are both
run and the validation year decides.

### Testing

Walk-forward backtest (`backtest.py`): each forecast period is predicted by a model trained only on
delivery days before it.

| Period | Used for | Retraining |
|---|---|---|
| Sept 2021 - Oct 2024 | Training only | - |
| Validation year (Nov 2024 - Oct 2025) | Every choice: optional inputs, tree settings, headline model | Every 3 months |
| Test year (Nov 2025 - latest day) | Final evaluation, choices frozen, touched once | Every month |

Metrics: MAE, RMSE, bias, MAE relative to the naive "same hour yesterday" forecast, a Diebold-Mariano
test [16] for whether differences are larger than chance, and the **battery value**: a 5 kW / 10 kWh
home battery (90% round-trip efficiency, at most one cycle a day) is scheduled on each forecast by a
small linear program and settled at actual prices; the capture rate is its profit as a share of the
perfect-foresight profit. It rewards getting the cheap and expensive hours right, which is what a
battery optimiser needs and MAE does not measure.

### Results

Test year 1 Nov 2025 to 10 Nov 2026 (375 delivery days), forecasts as at 09:00 UK on D-1, models
retrained monthly, all choices frozen on the validation year (`results/`, explained step by step in
`notebooks/walkthrough.ipynb`):

| Model | MAE (£/MWh) | RMSE | vs naive (MAE) | Battery capture | £ per battery per year |
|---|---|---|---|---|---|
| Naive: same hour yesterday | 20.79 | 33.94 | - | 79.0% | 169 |
| Naive: same hour last week | 27.41 | 42.89 | 32% worse | 79.8% | 171 |
| Lasso (price) | 14.40 | 20.58 | 31% better | 90.7% | 194 |
| Lasso (price / gas) | 12.90 | 18.60 | 38% better | 92.5% | 198 |
| Boosted trees (price) | 11.24 | 16.84 | 46% better | 93.4% | 200 |
| **Boosted trees (price / gas): headline** | **10.37** | 15.77 | **50% better** | 93.1% | 199 |
| Average of Lasso and trees | 10.68 | 15.71 | 49% better | 94.2% | 202 |
| Perfect foresight | - | - | - | 100% | 214 |

- The headline model was chosen on the validation year before the test year was touched; its MAE
  advantage is statistically significant (Diebold-Mariano, p < 0.001) over every model except the
  average of Lasso and trees (p = 0.11).
- Price / gas beats raw price in both model families, as the merit-order economics suggested.
- All settings and input choices were made on the validation year (headline validation MAE £9.02 vs
  naive £19.16). All optional input groups were kept; tree settings mattered much less than inputs
  and target form.
- Forecast residual demand is by far the most important input (shuffling it raises the MAE by about
  £14/MWh). Errors are largest at midday (solar) and at the 17:00-18:00 peak; the most expensive 5% of
  hours are under-predicted by about £9.5/MWh on average.
- Battery value: a 5 kW / 10 kWh battery scheduled on the model forecasts earns about £25-33 a year
  more in the day-ahead market than on the naive forecast, out of £214 possible.

Limitations and next steps are listed at the end of the notebook (target proxy, gas data lag, no
carbon price, spike under-prediction, light tuning; DA2 and imbalance forecasting as extensions).

## References

1. UCL (2022), *Electricity prices dictated by gas producers who provide less than half of UK electricity*. https://www.ucl.ac.uk/news/headlines/2022/sep/electricity-prices-dictated-gas-producers-who-provide-less-half-uk-electricity
2. Zakeri, Staffell, Grubb et al. (2023), *The role of natural gas in setting electricity prices in Europe*, Energy Reports. https://discovery.ucl.ac.uk/id/eprint/10178434/
3. GOV.UK, *Functioning of the UK carbon market*. https://www.gov.uk/government/collections/functioning-of-the-uk-carbon-market
4. GOV.UK, *Taking part in the UK Emissions Trading Scheme markets*. https://www.gov.uk/government/publications/taking-part-in-the-uk-emissions-trading-scheme-markets/taking-part-in-the-uk-emissions-trading-scheme-markets
5. Nord Pool, *About the day-ahead GB auction* (gate closure 09:50, results by 10:00). https://support.nordpoolgroup.com/support/solutions/articles/8000088463-about-the-day-ahead-gb-auction
6. NESO, *Day-Ahead Power Exchange prices (Nord Pool)*. https://www.neso.energy/data-portal/day-ahead-power-exchange-prices-nordpool
7. Elexon, *CP1359* (products included in the EPEX market index). https://assets.elexon.co.uk/wp-content/uploads/sites/11/2012/02/28171703/CP1359.pdf
8. Ziel & Weron (2018), *Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks*, Energy Economics 70, 396-420.
9. Lago, Marcjasz, De Schutter & Weron (2021), *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*, Applied Energy 293, 116983.
10. Makridakis, Spiliotis & Assimakopoulos (2022), *M5 accuracy competition: Results, findings, and conclusions*, International Journal of Forecasting 38(4), 1346-1364.
11. Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2207.08815
12. Coulon & Howison (2009), *Stochastic behaviour of the electricity bid stack: from fundamental drivers to power prices*, Journal of Energy Markets. https://people.maths.ox.ac.uk/~howison/papers/CoulonHowison211008.pdf
13. Ghelasi & Ziel (2024), *From day-ahead to mid and long-term horizons with econometric electricity price forecasting models*, arXiv:2406.00326. https://arxiv.org/abs/2406.00326
14. California Public Utilities Commission (2019), *Implied Market Heat Rate and Electric Cost Overcharge in 2018*. https://www.cpuc.ca.gov/-/media/cpuc-website/files/uploadedfiles/cpucwebsite/content/news_room/newsupdates/2019/imhr-and-surplus-cost-analysis-6-17-2019-v2-.pdf
15. Uniejewski, Weron & Ziel (2018), *Variance stabilizing transformations for electricity spot price forecasting*, IEEE Transactions on Power Systems.
16. Diebold & Mariano (1995), *Comparing predictive accuracy*, Journal of Business & Economic Statistics 13(3), 253-263.
