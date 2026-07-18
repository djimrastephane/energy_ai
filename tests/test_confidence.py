import numpy as np
import pandas as pd

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.confidence import (
    rate_anomaly,
    rate_changepoint,
    rate_data_quality,
    rate_forecast,
    rate_weather_model,
)
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult
from src.preprocessing import PreprocessingReport


def _report(missing=None, conflicts=None, outliers=None) -> PreprocessingReport:
    return PreprocessingReport(
        n_files_loaded=1,
        source_files=["a.csv"],
        date_range=(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-01")),
        n_months=36,
        missing_months=missing or [],
        duplicates_removed=0,
        conflicts=conflicts or [],
        outlier_warnings=outliers or [],
    )


def test_rate_data_quality_low_on_missing_months():
    rating = rate_data_quality(_report(missing=[pd.Timestamp("2024-05-01")]), n_months=36)
    assert rating.level == "Low"


def test_rate_data_quality_low_on_short_history():
    rating = rate_data_quality(_report(), n_months=8)
    assert rating.level == "Low"


def test_rate_data_quality_medium_on_outlier_warnings_only():
    rating = rate_data_quality(_report(outliers=["Jan 2024: unusually high"]), n_months=36)
    assert rating.level == "Medium"


def test_rate_data_quality_high_when_clean_and_long():
    rating = rate_data_quality(_report(), n_months=36)
    assert rating.level == "High"


def _energy_result(r_squared, heating_p, cooling_p=float("nan"), dw=2.0) -> EnergySignatureResult:
    return EnergySignatureResult(
        intercept=1.5,
        intercept_se=0.2,
        heating_slope=1.6,
        heating_se=0.1,
        heating_pvalue=heating_p,
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=cooling_p,
        r_squared=r_squared,
        adj_r_squared=r_squared - 0.01,
        durbin_watson=dw,
        n_obs=34,
        fitted=pd.Series(dtype=float),
        resid=pd.Series(dtype=float),
    )


def test_rate_weather_model_low_when_none():
    assert rate_weather_model(None).level == "Low"


def test_rate_weather_model_low_on_weak_fit():
    rating = rate_weather_model(_energy_result(r_squared=0.1, heating_p=0.6))
    assert rating.level == "Low"


def test_rate_weather_model_medium_on_moderate_r_squared():
    rating = rate_weather_model(_energy_result(r_squared=0.45, heating_p=0.01))
    assert rating.level == "Medium"


def test_rate_weather_model_medium_on_autocorrelated_residuals():
    rating = rate_weather_model(_energy_result(r_squared=0.7, heating_p=0.001, dw=0.8))
    assert rating.level == "Medium"


def test_rate_weather_model_high_on_strong_significant_fit():
    rating = rate_weather_model(_energy_result(r_squared=0.73, heating_p=0.0001, dw=1.6))
    assert rating.level == "High"


def _forecast_result(mae, n_folds=11, band_ratio=0.3) -> ForecastResult:
    p50 = np.array([300.0] * 6)
    half_width = band_ratio * p50 / 2
    comparison = pd.DataFrame(
        [{"model": "Seasonal Naive", "mae": mae, "rmse": mae * 1.2, "mape": 20.0, "n_folds": n_folds}]
    )
    return ForecastResult(
        model_name="Seasonal Naive",
        comparison=comparison,
        forecast_dates=pd.date_range("2026-08-01", periods=6, freq="MS"),
        point=p50,
        p10=p50 - half_width,
        p50=p50,
        p90=p50 + half_width,
    )


def test_rate_forecast_low_when_none():
    assert rate_forecast(None, history_mean=300.0).level == "Low"


def test_rate_forecast_low_on_large_relative_error():
    rating = rate_forecast(_forecast_result(mae=150.0), history_mean=300.0)
    assert rating.level == "Low"


def test_rate_forecast_medium_on_moderate_error():
    rating = rate_forecast(_forecast_result(mae=60.0), history_mean=300.0)
    assert rating.level == "Medium"


def test_rate_forecast_high_on_small_error_and_narrow_band():
    rating = rate_forecast(_forecast_result(mae=20.0, band_ratio=0.2), history_mean=300.0)
    assert rating.level == "High"


def test_rate_forecast_low_on_few_folds_even_with_good_mae():
    rating = rate_forecast(_forecast_result(mae=20.0, n_folds=3), history_mean=300.0)
    assert rating.level == "Low"


def test_rate_anomaly_levels_by_method_count():
    base = {"date": pd.Timestamp("2024-12-01"), "direction": "spike", "rank_context": "1st highest of 35"}
    assert rate_anomaly(Anomaly(methods=["a", "b", "c"], **base)).level == "High"
    assert rate_anomaly(Anomaly(methods=["a", "b"], **base)).level == "Medium"
    assert rate_anomaly(Anomaly(methods=["a"], **base)).level == "Low"


def test_rate_changepoint_levels_by_method():
    base = {"date": pd.Timestamp("2024-12-01"), "magnitude_kwh": 50.0, "direction": "increase"}
    assert rate_changepoint(ChangePoint(method="both", **base)).level == "High"
    assert rate_changepoint(ChangePoint(method="pelt", **base)).level == "Medium"
