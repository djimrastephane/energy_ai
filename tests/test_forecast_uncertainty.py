import numpy as np
import pytest

from src.forecast_uncertainty import bootstrap_forecast_bands


def test_bands_are_ordered_at_every_horizon_step():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 5, 20)
    point_forecast = np.array([100.0, 105.0, 110.0, 115.0])

    p10, p50, p90 = bootstrap_forecast_bands(point_forecast, residuals)

    assert np.all(p10 <= p50)
    assert np.all(p50 <= p90)


def test_band_width_grows_with_sqrt_horizon():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 5, 200)
    point_forecast = np.array([100.0] * 9)  # flat point forecast isolates the width effect

    p10, p50, p90 = bootstrap_forecast_bands(point_forecast, residuals, n_boot=20000, seed=1)

    width_h1 = p90[0] - p10[0]
    width_h4 = p90[3] - p10[3]  # h=4 -> sqrt(4)=2x scale
    width_h9 = p90[8] - p10[8]  # h=9 -> sqrt(9)=3x scale

    assert width_h4 == pytest.approx(2 * width_h1, rel=0.1)
    assert width_h9 == pytest.approx(3 * width_h1, rel=0.1)


def test_degenerate_case_with_too_few_residuals_returns_point_forecast():
    point_forecast = np.array([50.0, 60.0])

    p10, p50, p90 = bootstrap_forecast_bands(point_forecast, residuals=np.array([1.0]))

    assert p10.tolist() == point_forecast.tolist()
    assert p50.tolist() == point_forecast.tolist()
    assert p90.tolist() == point_forecast.tolist()


def test_reproducible_with_fixed_seed():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 5, 20)
    point_forecast = np.array([100.0, 105.0])

    result_a = bootstrap_forecast_bands(point_forecast, residuals, seed=7)
    result_b = bootstrap_forecast_bands(point_forecast, residuals, seed=7)

    for a, b in zip(result_a, result_b, strict=True):
        assert a.tolist() == b.tolist()
