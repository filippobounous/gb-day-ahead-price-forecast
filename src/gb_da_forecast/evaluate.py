"""Forecast accuracy metrics.

    MAE    mean absolute error (£/MWh): "on average each hourly forecast is off by £X". Headline.
    RMSE   root mean squared error: punishes big misses more. RMSE >> MAE means a few spikes
           dominate the errors.
    bias   mean error (forecast - actual): positive = systematically too high.
    rMAE   MAE relative to the naive "same hour yesterday" forecast: 0.70 = 30% better than naive;
           above 1 = not worth having.
    Diebold-Mariano test: is the difference in MAE between two models larger than chance? Based on
           daily average absolute-error differences, with a variance that allows for day-to-day
           correlation (Newey-West). p < 0.05 = the difference is unlikely to be luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def metrics(actual: pd.Series, predictions: pd.DataFrame, reference: str = "naive_d1") -> pd.DataFrame:
    err = predictions.sub(actual, axis=0)
    table = pd.DataFrame(
        {
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err**2).mean()),
            "bias": err.mean(),
        }
    )
    table["rMAE"] = table["MAE"] / table.loc[reference, "MAE"]
    return table


def mae_by(actual: pd.Series, predictions: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    return predictions.sub(actual, axis=0).abs().groupby(by.to_numpy()).mean()


def diebold_mariano(actual: pd.Series, pred_a: pd.Series, pred_b: pd.Series, days: pd.Series) -> tuple[float, float]:
    """DM statistic and two-sided p-value for H0: models A and B have equal MAE.

    Negative statistic = A has the lower error.
    """
    d = ((pred_a - actual).abs() - (pred_b - actual).abs()).groupby(days.to_numpy()).mean().to_numpy()
    n = len(d)
    x = d - d.mean()
    lags = int(n ** (1 / 3))
    var = x @ x / n + 2 * sum((1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n for k in range(1, lags + 1))
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))
