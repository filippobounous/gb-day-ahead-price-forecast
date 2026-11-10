"""The forecasting models, from simplest to most flexible.

Every model has the same interface: ``fit(train_df)`` then ``predict(df)`` -> £/MWh per row, where
the frames come from ``features.build_dataset``. All inputs in those frames are already known at
the 09:00 D-1 cutoff, so a model can never see the future through its inputs.

    NaiveModel   no learning: copy an earlier price ("same hour yesterday", "same hour last week",
                 "yesterday scaled by the change in gas"). The bar every model must clear.
    LassoModel   linear regression with a Lasso penalty, one model per hour of the day; the
                 "LEAR" benchmark of the price-forecasting literature (Ziel & Weron 2018,
                 Lago et al. 2021). Readable: every input gets one weight.
    GBMModel     gradient-boosted decision trees (scikit-learn's HistGradientBoostingRegressor,
                 the same algorithm family as LightGBM). Learns bends and interactions, e.g.
                 the merit-order hockey stick, and handles missing inputs natively.

Target forms (``target``):
    "raw"    predict the price in £/MWh.
    "ratio"  predict price / gas price (the market's implied heat rate), then multiply back.
             All money-valued inputs are divided by the gas price too, so the model works in
             "units of gas" and does not need to extrapolate when the gas price moves to a level
             it has not seen (see README, "Why price / gas").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import PRICE_LIKE

GAS_COL = "gas_ref"  # the gas price used for the ratio target (7-day mean, see features.py)


def to_model_units(df: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Inputs and target in the units the model is trained in (£/MWh, or multiples of the gas price)."""
    X = df[features].astype(float).copy()
    y = df["price"].astype(float)
    if target == "ratio":
        gas = df[GAS_COL].astype(float)
        for col in set(features) & set(PRICE_LIKE):
            X[col] = X[col] / gas
        y = y / gas
    elif target != "raw":
        raise ValueError(f"unknown target {target!r}")
    return X, y


def from_model_units(pred: np.ndarray, df: pd.DataFrame, target: str) -> np.ndarray:
    return pred * df[GAS_COL].to_numpy(float) if target == "ratio" else pred


class NaiveModel:
    """Copy an earlier price. ``kind``: "d1" (same hour of D-1), "d7" (D-7), "d1_gas" (D-1 x gas change)."""

    def __init__(self, kind: str = "d1"):
        self.kind = kind

    def fit(self, train: pd.DataFrame) -> "NaiveModel":
        return self  # nothing to learn

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d1 = df["p_d1"].fillna(df["p_d1_mean"])  # the missing 01:00 on the spring clock-change day
        if self.kind == "d1":
            pred = d1
        elif self.kind == "d7":
            pred = df["p_d7"].fillna(d1)
        elif self.kind == "d1_gas":
            pred = d1 * (df["gas_sap"] / df["gas_sap_prev"]).fillna(1.0)
        else:
            raise ValueError(self.kind)
        return pred.to_numpy(float)


class LassoModel:
    """One Lasso regression per hour of the day. The penalty strength is chosen by time-ordered
    cross-validation inside the training data only (never on the period being forecast)."""

    def __init__(self, features: list[str], target: str = "raw"):
        self.features = [f for f in features if f != "hour"]  # one model per hour instead
        self.target = target
        self.models: dict[int, Pipeline] = {}

    def _pipeline(self) -> Pipeline:
        categorical = [c for c in ["dow"] if c in self.features]
        numeric = [c for c in self.features if c not in categorical and c != "month"]  # season via doy_sin/cos
        prep = ColumnTransformer(
            [
                ("dow", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ]
        )
        lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5), alphas=30, max_iter=20000, tol=1e-3)
        return Pipeline([("prep", prep), ("lasso", lasso)])

    def fit(self, train: pd.DataFrame) -> "LassoModel":
        train = train[train["price"].notna()]
        for hour, part in train.groupby("hour"):
            X, y = to_model_units(part, self.features, self.target)
            self.models[hour] = self._pipeline().fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for hour, part in df.groupby("hour"):
            X, _ = to_model_units(part, self.features, self.target)
            pred = self.models[hour].predict(X)
            out[df.index.get_indexer(part.index)] = from_model_units(pred, part, self.target)
        return out

    def coefficients(self, hour: int) -> pd.Series:
        """Weights of the hour's model on standardised inputs: the change in the target (in model
        units) for a one-standard-deviation change in the input. Zero means the Lasso dropped it."""
        pipe = self.models[hour]
        names = pipe.named_steps["prep"].get_feature_names_out()
        return pd.Series(pipe.named_steps["lasso"].coef_, index=[n.split("__", 1)[1] for n in names])


GBM_DEFAULTS = dict(
    loss="absolute_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40
)


class GBMModel:
    """Gradient-boosted trees: hundreds of small trees, each correcting the errors of the previous ones."""

    def __init__(self, features: list[str], target: str = "raw", **params):
        self.features = features
        self.target = target
        self.params = {**GBM_DEFAULTS, **params}
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, train: pd.DataFrame) -> "GBMModel":
        train = train[train["price"].notna()]
        X, y = to_model_units(train, self.features, self.target)
        self.model = HistGradientBoostingRegressor(early_stopping=False, random_state=0, **self.params)
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = to_model_units(df, self.features, self.target)
        return from_model_units(self.model.predict(X), df, self.target)


class AverageModel:
    """Equal-weight average of several models' forecasts (forecast combination)."""

    def __init__(self, *models):
        self.models = models

    def fit(self, train: pd.DataFrame) -> "AverageModel":
        for m in self.models:
            m.fit(train)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(df) for m in self.models], axis=0)
