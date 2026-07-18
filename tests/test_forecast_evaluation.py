import numpy as np
import pandas as pd
import pytest

from src.forecast_evaluation import (
    ModelCVResult,
    _safe_mape,
    evaluate_all_models,
    generate_forecast,
    select_best_model,
    walk_forward_cv,
)


def _synthetic_series(n=36, trend_per_month=2.0, seasonal_amplitude=50.0, noise_std=1.0, seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    trend = trend_per_month * np.arange(n)
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (idx.month - 1) / 12)
    noise = rng.normal(0, noise_std, n)
    return pd.Series(300 + trend + seasonal + noise, index=idx)


def test_safe_mape_skips_near_zero_actuals():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([5.0, 11.0, 18.0])

    result = _safe_mape(actual, predicted)

    # Only the last two points (actual != 0) contribute: |11-10|/10=0.1, |18-20|/20=0.1 -> mean 10%
    assert result == pytest.approx(10.0)


def test_safe_mape_returns_none_when_all_actuals_are_zero():
    assert _safe_mape(np.array([0.0, 0.0]), np.array([1.0, 2.0])) is None


def test_walk_forward_cv_fold_count_matches_expected():
    series = pd.Series(np.arange(30.0))

    result = walk_forward_cv("linear", series, lambda s, h: np.full(h, s.iloc[-1]), min_train_size=24)

    assert result is not None
    assert result.n_folds == 30 - 24  # one fold per train_end from 24..29


def test_walk_forward_cv_returns_none_for_short_series():
    series = pd.Series(np.arange(10.0))

    result = walk_forward_cv("linear", series, lambda s, h: np.full(h, s.iloc[-1]), min_train_size=24)

    assert result is None


def test_walk_forward_cv_excludes_a_model_that_always_raises():
    series = pd.Series(np.arange(30.0))

    def _broken(s, h):
        raise RuntimeError("simulated model failure")

    result = walk_forward_cv("broken", series, _broken, min_train_size=24)

    assert result is None


def test_select_best_model_picks_lowest_mae():
    good = ModelCVResult(name="good", mae=0.1, rmse=0.2, mape=1.0, n_folds=10, residuals=np.zeros(10))
    bad = ModelCVResult(name="bad", mae=5.0, rmse=6.0, mape=50.0, n_folds=10, residuals=np.ones(10) * 5)

    assert select_best_model([bad, good]).name == "good"


def test_select_best_model_raises_on_empty_results():
    with pytest.raises(ValueError, match="No model"):
        select_best_model([])


def test_evaluate_all_models_on_real_registry_returns_results_for_sufficient_history():
    series = _synthetic_series(n=36)

    results = evaluate_all_models(series)

    names = {r.name for r in results}
    # All 8 models should have enough history (36 months) to produce at least one fold.
    assert names == {
        "Naive",
        "Seasonal Naive",
        "Linear Trend",
        "Holt-Winters",
        "SARIMA",
        "Prophet",
        "XGBoost",
        "LightGBM",
    }
    assert all(r.n_folds > 0 for r in results)


def test_generate_forecast_end_to_end_with_auto_selection():
    series = _synthetic_series(n=36)

    result = generate_forecast(series, horizon=6, model_name="auto")

    assert result.model_name in {
        "Naive",
        "Seasonal Naive",
        "Linear Trend",
        "Holt-Winters",
        "SARIMA",
        "Prophet",
        "XGBoost",
        "LightGBM",
    }
    assert len(result.forecast_dates) == 6
    assert len(result.point) == 6
    assert np.all(result.p10 <= result.p50)
    assert np.all(result.p50 <= result.p90)
    assert result.forecast_dates[0] == series.index[-1] + pd.DateOffset(months=1)
    # Comparison table sorted best-first.
    assert result.comparison["mae"].is_monotonic_increasing


def test_generate_forecast_honors_forced_model_choice():
    series = _synthetic_series(n=36)

    result = generate_forecast(series, horizon=3, model_name="Naive")

    assert result.model_name == "Naive"
    # Naive forecast = last observed value, repeated.
    assert result.point.tolist() == pytest.approx([series.iloc[-1]] * 3)


def test_generate_forecast_raises_on_unknown_model_name():
    series = _synthetic_series(n=36)

    with pytest.raises(ValueError, match="Unknown model"):
        generate_forecast(series, horizon=3, model_name="NotARealModel")


def test_generate_forecast_clips_bands_at_zero_by_default():
    # Low final value + high noise -> an unclipped residual bootstrap would push
    # the low end of the band negative, which is physically meaningless for kWh.
    idx = pd.date_range("2021-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(5)
    values = np.abs(rng.normal(20, 60, 30))  # high relative noise vs. a low level
    series = pd.Series(values, index=idx)

    clipped = generate_forecast(series, horizon=3, model_name="Naive", min_value=0.0)
    assert np.all(clipped.p10 >= 0.0)
    assert np.all(clipped.point >= 0.0)

    unclipped = generate_forecast(series, horizon=3, model_name="Naive", min_value=None)
    assert unclipped.p10.min() < 0.0  # confirms the clip above is actually doing something
