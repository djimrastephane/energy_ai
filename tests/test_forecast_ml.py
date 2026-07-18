import numpy as np
import pandas as pd
import pytest

from src.forecast_ml import (
    _build_training_features,
    _recursive_tree_forecast,
    forecast_lightgbm,
    forecast_prophet,
    forecast_xgboost,
)


def _synthetic_series(n=36, trend_per_month=2.0, seasonal_amplitude=50.0, noise_std=1.0, seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    trend = trend_per_month * np.arange(n)
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (idx.month - 1) / 12)
    noise = rng.normal(0, noise_std, n)
    return pd.Series(300 + trend + seasonal + noise, index=idx)


def test_build_training_features_drops_rows_without_lag12():
    series = pd.Series(np.arange(1.0, 15.0), index=pd.date_range("2022-01-01", periods=14, freq="MS"))

    df = _build_training_features(series)

    assert len(df) == 2  # only positions 12, 13 have a defined lag_12
    assert df["lag_12"].tolist() == [1.0, 2.0]
    assert df["lag_1"].tolist() == [12.0, 13.0]


def test_recursive_tree_forecast_uses_correct_lag_features_and_recurses():
    series = pd.Series(np.arange(1.0, 25.0), index=pd.date_range("2022-01-01", periods=24, freq="MS"))
    captured_x = []

    class _FakeModel:
        def predict(self, x):
            captured_x.append(x.copy())
            return np.array([x[0, 3] + 1.0])  # "next = lag_1 + 1", matching this series' true pattern

    result = _recursive_tree_forecast(series, horizon=3, fit_fn=lambda x, y: _FakeModel())

    assert result.tolist() == pytest.approx([25.0, 26.0, 27.0])
    first_call_x = captured_x[0]
    assert first_call_x[0, 3] == pytest.approx(24.0)  # lag_1 = last observed value
    assert first_call_x[0, 4] == pytest.approx(13.0)  # lag_12 = value from 12 months back
    # Second call must see the first prediction fed back in as the new lag_1.
    assert captured_x[1][0, 3] == pytest.approx(25.0)


def test_recursive_tree_forecast_raises_on_insufficient_data():
    series = pd.Series(np.arange(1.0, 11.0), index=pd.date_range("2022-01-01", periods=10, freq="MS"))

    with pytest.raises(ValueError, match="at least 13"):
        _recursive_tree_forecast(series, horizon=3, fit_fn=lambda x, y: None)


def test_forecast_xgboost_returns_correct_length_and_finite_values():
    series = _synthetic_series(n=30)
    result = forecast_xgboost(series, horizon=4)

    assert len(result) == 4
    assert np.all(np.isfinite(result))


def test_forecast_lightgbm_returns_correct_length_and_finite_values():
    series = _synthetic_series(n=30)
    result = forecast_lightgbm(series, horizon=4)

    assert len(result) == 4
    assert np.all(np.isfinite(result))


def test_forecast_prophet_returns_correct_length_and_reasonable_values():
    series = _synthetic_series(n=30)
    result = forecast_prophet(series, horizon=5)

    assert len(result) == 5
    assert np.all(np.isfinite(result))
    assert result.min() > series.min() - 3 * series.std()
    assert result.max() < series.max() + 3 * series.std()
