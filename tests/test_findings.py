import numpy as np
import pandas as pd

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.energy_signature import EnergySignatureResult
from src.findings import (
    finding_biggest_change,
    finding_forecast_outlook,
    finding_seasonality,
    finding_weather,
    finding_yoy_trend,
    generate_findings,
)
from src.forecast_evaluation import ForecastResult
from src.kpis import KPIComparison


def test_finding_yoy_trend_none_without_previous_period():
    kpis = [KPIComparison(label="Total consumption", current=100.0, previous=None, pct_change=None, unit="kWh")]
    assert finding_yoy_trend(kpis) is None


def test_finding_yoy_trend_reports_direction_and_magnitude():
    kpis = [
        KPIComparison(label="Total consumption", current=4400.0, previous=4000.0, pct_change=10.0, unit="kWh"),
        KPIComparison(label="Total cost", current=600.0, previous=550.0, pct_change=9.0, unit="£"),
    ]
    finding = finding_yoy_trend(kpis)
    assert finding is not None
    assert "10.0% higher" in finding.narrative
    assert finding.confidence == "High"


def test_finding_seasonality_none_without_result():
    assert finding_seasonality(None) is None


def test_finding_seasonality_reports_strength_bands():
    from src.decomposition import stl_decompose

    n = 36
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    seasonal = 100.0 * np.cos(2 * np.pi * (idx.month - 1) / 12)
    values = 300 + seasonal + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"month_start": idx, "consumption_kwh": values})

    stl_result = stl_decompose(df)
    finding = finding_seasonality(stl_result)

    assert finding is not None
    assert finding.category == "seasonality"
    assert finding.confidence == "High"  # strong, low-noise synthetic seasonality


def _energy_result(r_squared=0.73, heating_p=0.0001, cooling_p=float("nan")) -> EnergySignatureResult:
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    return EnergySignatureResult(
        intercept=1.5,
        intercept_se=0.2,
        heating_slope=1.66,
        heating_se=0.1,
        heating_pvalue=heating_p,
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=cooling_p,
        r_squared=r_squared,
        adj_r_squared=r_squared - 0.01,
        durbin_watson=1.8,
        n_obs=34,
        fitted=pd.Series(np.zeros(12), index=idx),
        resid=pd.Series(np.zeros(12), index=idx),
    )


def test_finding_weather_none_without_result():
    assert finding_weather(None, unit_rate=0.3) is None


def test_finding_weather_mentions_fraction_and_heating_sensitivity():
    finding = finding_weather(_energy_result(), unit_rate=0.3)
    assert finding is not None
    assert "three quarters" in finding.narrative
    assert "1.66 kWh/day" in finding.narrative
    assert finding.confidence == "High"


def test_finding_weather_reports_no_significant_sensitivity():
    finding = finding_weather(_energy_result(r_squared=0.4, heating_p=0.6), unit_rate=0.3)
    assert finding is not None
    assert "No statistically significant heating sensitivity" in finding.narrative


def test_finding_weather_defaults_to_energy_noun_and_generic_attribution():
    """Default fuel ('total') must not claim 'electric heating' -- that's a specific claim only
    correct when the regression was actually fitted on electricity-only data."""
    finding = finding_weather(_energy_result(), unit_rate=0.3)
    assert finding is not None
    assert "energy consumption" in finding.narrative
    assert "electric heating" not in finding.narrative


def test_finding_weather_names_electricity_when_fuel_is_electricity():
    finding = finding_weather(_energy_result(), unit_rate=0.3, fuel="electricity")
    assert finding is not None
    assert "electricity consumption" in finding.narrative
    assert "electric heating" in finding.narrative


def test_finding_weather_names_gas_when_fuel_is_gas():
    finding = finding_weather(_energy_result(), unit_rate=0.3, fuel="gas")
    assert finding is not None
    assert "gas consumption" in finding.narrative
    assert "gas heating" in finding.narrative
    assert "electric" not in finding.narrative.lower()


def test_finding_biggest_change_none_without_events():
    assert finding_biggest_change([], [], None) is None


def test_finding_biggest_change_prefers_anomaly_over_changepoint():
    anomaly = Anomaly(
        date=pd.Timestamp("2024-12-01"),
        methods=["rolling_zscore", "stl_esd", "isolation_forest"],
        direction="spike",
        rank_context="1st highest of 35 months",
    )
    cp = ChangePoint(date=pd.Timestamp("2024-05-01"), method="both", magnitude_kwh=50.0, direction="increase")

    finding = finding_biggest_change([anomaly], [cp], None)

    assert finding is not None
    assert "December 2024" in finding.narrative
    assert "unexplained increase" in finding.narrative
    assert finding.confidence == "High"


def test_finding_biggest_change_mentions_weather_when_residual_is_large():
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    resid = pd.Series(np.array([1.0] * 11 + [50.0]), index=idx)  # last point is a big outlier
    energy_result = _energy_result()
    energy_result.resid = resid
    anomaly = Anomaly(
        date=idx[-1], methods=["rolling_zscore", "stl_esd"], direction="spike", rank_context="1st highest of 12 months"
    )

    finding = finding_biggest_change([anomaly], [], energy_result)

    assert "after adjusting for weather" in finding.narrative


def test_finding_biggest_change_falls_back_to_changepoints():
    cp = ChangePoint(date=pd.Timestamp("2024-05-01"), method="pelt", magnitude_kwh=-40.0, direction="decrease")

    finding = finding_biggest_change([], [cp], None)

    assert finding is not None
    assert "May 2024" in finding.narrative
    assert finding.confidence == "Medium"


def _forecast_result(mae=20.0) -> ForecastResult:
    p50 = np.array([300.0] * 6)
    comparison = pd.DataFrame(
        [{"model": "Seasonal Naive", "mae": mae, "rmse": mae * 1.2, "mape": 15.0, "n_folds": 11}]
    )
    return ForecastResult(
        model_name="Seasonal Naive",
        comparison=comparison,
        forecast_dates=pd.date_range("2026-08-01", periods=6, freq="MS"),
        point=p50,
        p10=p50 - 30,
        p50=p50,
        p90=p50 + 30,
    )


def test_finding_forecast_outlook_none_without_result():
    assert finding_forecast_outlook(None, history_mean=300.0) is None


def test_finding_forecast_outlook_reports_totals_and_model():
    finding = finding_forecast_outlook(_forecast_result(), history_mean=300.0)
    assert finding is not None
    assert "Seasonal Naive" in finding.narrative
    assert "1,800" in finding.narrative  # 300*6


def test_generate_findings_sorts_by_confidence_and_drops_none():
    kpis = [KPIComparison(label="Total consumption", current=100.0, previous=90.0, pct_change=11.1, unit="kWh")]
    findings = generate_findings(
        kpis=kpis,
        stl_result=None,
        energy_result=None,
        unit_rate=0.3,
        anomalies=[],
        changepoints=[],
        forecast_result=None,
        history_mean=300.0,
    )
    assert len(findings) == 1
    assert findings[0].category == "trend"


def test_generate_findings_includes_all_available_categories():
    kpis = [KPIComparison(label="Total consumption", current=100.0, previous=90.0, pct_change=11.1, unit="kWh")]
    anomaly = Anomaly(
        date=pd.Timestamp("2024-12-01"),
        methods=["rolling_zscore", "stl_esd", "isolation_forest"],
        direction="spike",
        rank_context="1st highest of 35 months",
    )
    findings = generate_findings(
        kpis=kpis,
        stl_result=None,
        energy_result=_energy_result(),
        unit_rate=0.3,
        anomalies=[anomaly],
        changepoints=[],
        forecast_result=_forecast_result(),
        history_mean=300.0,
    )
    categories = {f.category for f in findings}
    assert categories == {"trend", "anomaly", "weather", "forecast"}
    # Confidence-sorted: High-confidence findings should appear before any Medium/Low ones.
    ranks = {"High": 3, "Medium": 2, "Low": 1}
    levels = [ranks[f.confidence] for f in findings]
    assert levels == sorted(levels, reverse=True)
