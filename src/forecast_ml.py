"""ML forecasting models: Prophet, XGBoost, LightGBM.

Same ``(series, horizon) -> np.ndarray`` signature as
``src.forecast_models``. The tree models (XGBoost, LightGBM) share one
feature set -- time index, cyclical month encoding, lag_1, lag_12 -- and
forecast multiple steps ahead recursively (each prediction feeds back as a
lag feature for the next step), which is standard practice for tree-based
time series models.
"""

from __future__ import annotations

import logging
import warnings
from typing import Protocol

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

_FEATURE_COLUMNS = ["time_idx", "month_sin", "month_cos", "lag_1", "lag_12"]
# Deliberately conservative given n around 25-35 observations: shallow trees, few of them.
_TREE_KWARGS = {"n_estimators": 50, "max_depth": 3, "random_state": 42}


class _SklearnLikeRegressor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def _month_cyclical(month: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.sin(2 * np.pi * np.asarray(month) / 12), np.cos(2 * np.pi * np.asarray(month) / 12)


def _build_training_features(series: pd.Series) -> pd.DataFrame:
    """Time index, cyclical month, lag_1, lag_12, target -- rows with an undefined lag_12 dropped."""
    n = len(series)
    sin_m, cos_m = _month_cyclical(series.index.month)
    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "month_sin": sin_m,
            "month_cos": cos_m,
            "lag_1": series.shift(1).to_numpy(),
            "lag_12": series.shift(12).to_numpy(),
            "y": series.to_numpy(dtype=float),
        },
        index=series.index,
    )
    return df.dropna()


def _recursive_tree_forecast(series: pd.Series, horizon: int, fit_fn) -> np.ndarray:
    """Fit a tree regressor on lag/seasonal features, then forecast forward step by step.

    Raises:
        ValueError: if fewer than 13 months are available (the minimum
            needed for one training row with a defined lag_12 feature).
    """
    train_df = _build_training_features(series)
    if train_df.empty:
        raise ValueError("Need at least 13 months of history for lag-12 features to be defined.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # benign "X does not have valid feature names" from sklearn
        model: _SklearnLikeRegressor = fit_fn(
            train_df[_FEATURE_COLUMNS].to_numpy(), train_df["y"].to_numpy()
        )

        working = series.astype(float).copy()
        predictions = np.empty(horizon)
        for step in range(horizon):
            t = len(series) + step
            next_date = working.index[-1] + pd.DateOffset(months=1)
            sin_m, cos_m = _month_cyclical(next_date.month)
            lag_1 = working.iloc[-1]
            lag_12 = working.iloc[-12]
            x_next = np.array([[t, sin_m, cos_m, lag_1, lag_12]])
            pred = float(model.predict(x_next)[0])
            predictions[step] = pred
            working.loc[next_date] = pred

    return predictions


def forecast_xgboost(series: pd.Series, horizon: int) -> np.ndarray:
    import xgboost as xgb

    return _recursive_tree_forecast(
        series, horizon, lambda x, y: xgb.XGBRegressor(**_TREE_KWARGS).fit(x, y)
    )


def forecast_lightgbm(series: pd.Series, horizon: int) -> np.ndarray:
    import lightgbm as lgb

    # min_child_samples/min_data_in_leaf lowered from LightGBM's default (20) since
    # training sets here are as small as ~12 rows -- the default would forbid every
    # split and silently degrade to a constant predictor.
    return _recursive_tree_forecast(
        series,
        horizon,
        lambda x, y: lgb.LGBMRegressor(
            **_TREE_KWARGS, min_child_samples=3, min_data_in_leaf=3, verbosity=-1
        ).fit(x, y),
    )


def forecast_prophet(series: pd.Series, horizon: int) -> np.ndarray:
    """Prophet with yearly seasonality only (monthly data has no weekly/daily signal).

    ``uncertainty_samples=0`` disables Prophet's internal interval sampling
    -- this codebase computes uncertainty bands uniformly for every model
    via residual bootstrapping (see ``src.forecast_uncertainty``), not via
    each library's own (and differently-valid-at-this-sample-size) interval
    machinery.

    ``iter=200`` caps the underlying cmdstan MAP optimizer's iteration
    budget. Without it, fit time is very data-dependent: on real (noisy)
    monthly billing data it converges quickly (~0.5s), but on unusually
    clean/regular series (verified with a near-perfect synthetic
    trend+seasonal signal) the optimizer can grind for 10-60x longer
    without materially changing the fit -- confirmed on the real data here
    that ``iter=200`` produces numerically identical forecasts to the
    uncapped default while removing that worst case.
    """
    from prophet import Prophet

    df = pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=float)})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df, iter=200)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
    return forecast["yhat"].to_numpy()[-horizon:]
