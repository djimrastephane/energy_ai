"""Cross-validation, metrics, model selection, and final forecast assembly.

Walk-forward (expanding-window), 1-step-ahead cross-validation compares
every model in ``MODEL_REGISTRY`` -- naive/seasonal-naive/linear-trend/
Holt-Winters/SARIMA (``src.forecast_models``) and Prophet/XGBoost/LightGBM
(``src.forecast_ml``) -- on equal footing. With ~25-35 months of history,
multi-step CV at longer horizons wouldn't have enough folds to be
reliable, so CV only ever validates 1-step accuracy; the horizon argument
to :func:`generate_forecast` only controls how far the final *chosen*
model is projected forward, independent of what CV measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.forecast_ml import forecast_lightgbm, forecast_prophet, forecast_xgboost
from src.forecast_models import (
    forecast_holt_winters,
    forecast_linear_trend,
    forecast_naive,
    forecast_sarima,
    forecast_seasonal_naive,
)
from src.forecast_uncertainty import bootstrap_forecast_bands
from src.utils import get_logger

logger = get_logger(__name__)

ForecastFn = Callable[[pd.Series, int], np.ndarray]

MODEL_REGISTRY: dict[str, ForecastFn] = {
    "Naive": forecast_naive,
    "Seasonal Naive": forecast_seasonal_naive,
    "Linear Trend": forecast_linear_trend,
    "Holt-Winters": forecast_holt_winters,
    "SARIMA": forecast_sarima,
    "Prophet": forecast_prophet,
    "XGBoost": forecast_xgboost,
    "LightGBM": forecast_lightgbm,
}

DEFAULT_MIN_TRAIN_SIZE = 24


@dataclass
class ModelCVResult:
    name: str
    mae: float
    rmse: float
    mape: float | None
    n_folds: int
    residuals: np.ndarray = field(repr=False)


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Mean absolute percentage error, skipping points where actual is ~0 (percentage undefined).

    Returns None if every actual value is ~0 (no valid points to average).
    """
    mask = np.abs(actual) > 1e-6
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def walk_forward_cv(
    name: str, series: pd.Series, model_fn: ForecastFn, min_train_size: int = DEFAULT_MIN_TRAIN_SIZE
) -> ModelCVResult | None:
    """Expanding-window, 1-step-ahead walk-forward CV for one model.

    Returns None (logged) if there isn't enough history for even one fold,
    or if the model fails on every fold it's given -- excluded from
    comparison rather than crashing the whole evaluation.
    """
    n = len(series)
    if n <= min_train_size:
        logger.info("%s: not enough history (%d months) for a %d-month CV window", name, n, min_train_size)
        return None

    actuals: list[float] = []
    predictions: list[float] = []
    for train_end in range(min_train_size, n):
        train = series.iloc[:train_end]
        try:
            pred = model_fn(train, 1)[0]
        except Exception as exc:  # noqa: BLE001 -- one bad fold shouldn't sink the whole CV run
            logger.debug("%s: fold at train_end=%d failed: %s", name, train_end, exc)
            continue
        actuals.append(float(series.iloc[train_end]))
        predictions.append(float(pred))

    if not actuals:
        logger.warning("%s: produced no valid CV folds", name)
        return None

    actual_arr = np.array(actuals)
    predicted_arr = np.array(predictions)
    residuals = actual_arr - predicted_arr

    return ModelCVResult(
        name=name,
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        mape=_safe_mape(actual_arr, predicted_arr),
        n_folds=len(actuals),
        residuals=residuals,
    )


def evaluate_all_models(
    series: pd.Series, min_train_size: int = DEFAULT_MIN_TRAIN_SIZE
) -> list[ModelCVResult]:
    """Run walk-forward CV for every model in the registry, dropping any that fail entirely."""
    results = []
    for name, model_fn in MODEL_REGISTRY.items():
        result = walk_forward_cv(name, series, model_fn, min_train_size)
        if result is not None:
            results.append(result)
    return results


def select_best_model(results: list[ModelCVResult]) -> ModelCVResult:
    """Lowest mean CV MAE wins; RMSE breaks ties.

    Raises:
        ValueError: if no model produced any valid CV fold.
    """
    if not results:
        raise ValueError("No model produced any valid cross-validation folds on this series.")
    return min(results, key=lambda r: (r.mae, r.rmse))


@dataclass
class ForecastResult:
    model_name: str
    comparison: pd.DataFrame  # every model's CV metrics, sorted best (lowest MAE) first
    forecast_dates: pd.DatetimeIndex
    point: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray


def _comparison_table(results: list[ModelCVResult]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"model": r.name, "mae": r.mae, "rmse": r.rmse, "mape": r.mape, "n_folds": r.n_folds}
            for r in results
        ]
    )
    return df.sort_values("mae").reset_index(drop=True)


def generate_forecast(
    series: pd.Series,
    horizon: int,
    model_name: str = "auto",
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    min_value: float | None = 0.0,
) -> ForecastResult:
    """Cross-validate every model, pick (or honor a forced) model, refit on the full series, and forecast.

    Args:
        min_value: physical floor applied to the point forecast and every
            band (default 0.0, since this app's series are always
            non-negative kWh/£ quantities). Without this, the residual
            bootstrap can and does push a low-season month's P10 band
            below zero when residual spread is large relative to the point
            forecast -- verified on the real data, e.g. a naive bootstrap
            put July's P10 at -190 kWh. Pass ``None`` to disable if this is
            ever reused for a quantity that can legitimately be negative.

    Raises:
        ValueError: if no model produced a valid CV fold, or if
            ``model_name`` isn't ``"auto"`` and isn't a model that produced
            valid folds on this series.
    """
    cv_results = evaluate_all_models(series, min_train_size)
    comparison = _comparison_table(cv_results)

    if model_name == "auto":
        chosen = select_best_model(cv_results)
    else:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {model_name!r}; choose from {list(MODEL_REGISTRY)} or 'auto'.")
        chosen = next((r for r in cv_results if r.name == model_name), None)
        if chosen is None:
            raise ValueError(f"{model_name!r} did not produce any valid CV folds on this series.")

    point_forecast = np.asarray(MODEL_REGISTRY[chosen.name](series, horizon), dtype=float)
    p10, p50, p90 = bootstrap_forecast_bands(point_forecast, chosen.residuals)

    if min_value is not None:
        point_forecast = np.maximum(point_forecast, min_value)
        p10 = np.maximum(p10, min_value)
        p50 = np.maximum(p50, min_value)
        p90 = np.maximum(p90, min_value)

    forecast_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

    return ForecastResult(
        model_name=chosen.name,
        comparison=comparison,
        forecast_dates=forecast_dates,
        point=point_forecast,
        p10=p10,
        p50=p50,
        p90=p90,
    )
