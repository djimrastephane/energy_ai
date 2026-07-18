import numpy as np
import pandas as pd
import pytest

from src.forecast_models import (
    forecast_holt_winters,
    forecast_linear_trend,
    forecast_naive,
    forecast_sarima,
    forecast_seasonal_naive,
)


def _synthetic_series(n=36, trend_per_month=2.0, seasonal_amplitude=50.0, noise_std=1.0, seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    trend = trend_per_month * np.arange(n)
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (idx.month - 1) / 12)
    noise = rng.normal(0, noise_std, n)
    return pd.Series(300 + trend + seasonal + noise, index=idx)


def test_forecast_naive_repeats_last_value():
    series = pd.Series([10.0, 20.0, 30.0])
    result = forecast_naive(series, horizon=4)
    assert result.tolist() == [30.0, 30.0, 30.0, 30.0]


def test_forecast_seasonal_naive_cycles_last_12_months():
    series = pd.Series(np.arange(1.0, 13.0))  # 1..12
    result = forecast_seasonal_naive(series, horizon=14)
    assert result[:12].tolist() == list(range(1, 13))
    assert result[12] == 1.0  # cycles back around
    assert result[13] == 2.0


def test_forecast_seasonal_naive_raises_on_insufficient_data():
    with pytest.raises(ValueError, match="at least 12"):
        forecast_seasonal_naive(pd.Series([1.0, 2.0]), horizon=3)


def test_forecast_linear_trend_recovers_known_slope():
    n = 24
    y = pd.Series(100.0 + 5.0 * np.arange(n))  # exact linear trend, no noise

    result = forecast_linear_trend(y, horizon=3)

    # Next points should continue the exact trend: 100 + 5*24, 100 + 5*25, 100 + 5*26
    expected = [100.0 + 5.0 * n, 100.0 + 5.0 * (n + 1), 100.0 + 5.0 * (n + 2)]
    assert result == pytest.approx(expected, abs=1e-6)


def test_forecast_holt_winters_raises_on_insufficient_data():
    with pytest.raises(ValueError, match="at least 24"):
        forecast_holt_winters(_synthetic_series(n=18), horizon=3)


def test_forecast_holt_winters_directionally_reasonable():
    series = _synthetic_series(n=36)
    result = forecast_holt_winters(series, horizon=6)

    assert len(result) == 6
    assert np.all(np.isfinite(result))
    # Forecasts should be in the ballpark of the observed range, not wildly off.
    assert result.min() > series.min() - 3 * series.std()
    assert result.max() < series.max() + 3 * series.std()


def test_forecast_sarima_raises_on_insufficient_data():
    with pytest.raises(ValueError, match="at least 24"):
        forecast_sarima(_synthetic_series(n=18), horizon=3)


def test_forecast_sarima_returns_correct_length_and_finite_values():
    series = _synthetic_series(n=36)
    result = forecast_sarima(series, horizon=5)

    assert len(result) == 5
    assert np.all(np.isfinite(result))
