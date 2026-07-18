"""Classical/statistical forecasting models: naive, seasonal naive, linear
trend, Holt-Winters, SARIMA.

Every model exposes the same signature -- ``(series, horizon) -> np.ndarray``
of ``horizon`` future point forecasts -- so ``src.forecast_evaluation`` can
treat all of them (plus the ML models in ``src.forecast_ml``)
interchangeably for cross-validation and final forecasting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import get_logger

logger = get_logger(__name__)

SEASONAL_PERIOD = 12
MIN_SEASONAL_OBS = 2 * SEASONAL_PERIOD  # two full cycles, needed for a reliable seasonal fit

_SARIMA_CANDIDATE_ORDERS = [
    ((1, 1, 1), (1, 1, 0, SEASONAL_PERIOD)),
    ((1, 1, 0), (0, 1, 1, SEASONAL_PERIOD)),
    ((0, 1, 1), (1, 1, 0, SEASONAL_PERIOD)),
]


def _as_indexed_series(series: pd.Series) -> pd.Series:
    """Attach an explicit monthly-start DatetimeIndex freq (avoids statsmodels inference warnings)."""
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    return s


def forecast_naive(series: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every future month."""
    return np.full(horizon, float(series.iloc[-1]))


def forecast_seasonal_naive(series: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last 12 observed months, cycling forward for the requested horizon.

    Raises:
        ValueError: if fewer than 12 months of history are available.
    """
    if len(series) < SEASONAL_PERIOD:
        raise ValueError(f"Seasonal naive needs at least {SEASONAL_PERIOD} months; got {len(series)}.")
    last_cycle = series.iloc[-SEASONAL_PERIOD:].to_numpy(dtype=float)
    return np.array([last_cycle[i % SEASONAL_PERIOD] for i in range(horizon)])


def forecast_linear_trend(series: pd.Series, horizon: int) -> np.ndarray:
    """Ordinary least squares on the time index (pure trend, no seasonality), extrapolated forward."""
    n = len(series)
    x_train = np.column_stack([np.ones(n), np.arange(n)])
    y = series.to_numpy(dtype=float)
    model = sm.OLS(y, x_train).fit()
    x_future = np.column_stack([np.ones(horizon), np.arange(n, n + horizon)])
    return model.predict(x_future)


def forecast_holt_winters(series: pd.Series, horizon: int) -> np.ndarray:
    """Exponential smoothing with additive trend and additive yearly seasonality.

    Raises:
        ValueError: if fewer than ``MIN_SEASONAL_OBS`` months are available.
    """
    if len(series) < MIN_SEASONAL_OBS:
        raise ValueError(f"Holt-Winters needs at least {MIN_SEASONAL_OBS} months; got {len(series)}.")
    s = _as_indexed_series(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            s, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD
        ).fit()
        forecast = model.forecast(horizon)
    return forecast.to_numpy()


def forecast_sarima(series: pd.Series, horizon: int) -> np.ndarray:
    """SARIMA, order chosen by AIC over a small fixed candidate grid (not an exhaustive search).

    The grid is deliberately small (3 candidates) so this stays fast enough
    to run inside every cross-validation fold -- selecting a order per fold,
    appropriate to whatever slice of data that fold has, rather than a
    single order fixed from the full series and potentially mismatched to
    shorter early folds.

    Raises:
        ValueError: if fewer than ``MIN_SEASONAL_OBS`` months are available,
            or if no candidate order converges.
    """
    if len(series) < MIN_SEASONAL_OBS:
        raise ValueError(f"SARIMA needs at least {MIN_SEASONAL_OBS} months; got {len(series)}.")
    s = _as_indexed_series(series)

    best_fit, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, seasonal_order in _SARIMA_CANDIDATE_ORDERS:
            try:
                fit = SARIMAX(
                    s,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            except Exception as exc:  # noqa: BLE001 -- one candidate failing to converge isn't fatal
                logger.debug("SARIMA order %s/%s failed to converge: %s", order, seasonal_order, exc)
                continue
            if fit.aic < best_aic:
                best_fit, best_aic = fit, fit.aic

        if best_fit is None:
            raise ValueError("No SARIMA candidate order converged for this series.")
        forecast = best_fit.forecast(horizon)
    return forecast.to_numpy()
