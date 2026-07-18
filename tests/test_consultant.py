import numpy as np
import pandas as pd

from src.anomalies import Anomaly
from src.confidence import ConfidenceRating
from src.consultant import (
    ConsultantContext,
    answer_benchmark,
    answer_bill_change,
    answer_carbon,
    answer_forecast,
    answer_fuel_focus,
    answer_last_month_anomaly,
    answer_savings,
    answer_winter_comparison,
)
from src.findings import Finding
from src.forecast_evaluation import ForecastResult
from src.recommendations import NO_RECOMMENDATIONS_MESSAGE, Recommendation
from src.report import AnalystReport


def _confidence_dict():
    ok = ConfidenceRating("High", "clean")
    return {"data_quality": ok, "weather_model": ok, "forecast": ok, "anomaly_detection": ok}


def _analyst_report(**overrides) -> AnalystReport:
    defaults = dict(
        executive_summary="",
        overall_assessment="Consumption is running at normal, expected levels compared to a year ago.",
        findings=[],
        biggest_finding=None,
        recommendations=[],
        largest_saving=None,
        confidence=_confidence_dict(),
        limitations=[],
        monitoring_priorities=[],
        fuel_mix_finding=None,
        weather_sensitivity_finding=None,
        largest_cost_driver=None,
        weather_vs_behavioural_summary=None,
        per_fuel_findings={},
    )
    defaults.update(overrides)
    return AnalystReport(**defaults)


def _clean_df(n=24, start="2023-01-01", kwh=300.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    df = pd.DataFrame({"month_start": idx, "consumption_kwh": np.full(n, kwh), "cost_gbp": np.full(n, kwh * 0.3)})
    df["year"] = df["month_start"].dt.year
    df["is_partial_year"] = df.groupby("year")["month_start"].transform("size") < 12
    return df


def _ctx(**overrides) -> ConsultantContext:
    defaults = dict(
        analyst_report=_analyst_report(),
        clean=_clean_df(),
        fuel_frames={},
        fuel_energy_results={},
        anomalies=[],
        forecast_12mo=None,
        multi_fuel_forecasts=None,
        weather_enabled=False,
    )
    defaults.update(overrides)
    return ConsultantContext(**defaults)


def _finding(title="X", narrative="narrative text", confidence="High") -> Finding:
    return Finding(title=title, narrative=narrative, evidence=["evidence line"], confidence=confidence, confidence_reason="reason", category="fuel")


def _recommendation(title, action, saving=None, confidence="High") -> Recommendation:
    return Recommendation(title=title, action=action, estimated_saving_gbp=saving, evidence=["ev"], confidence=confidence, confidence_reason="reason", rationale="rationale")


# --- answer_bill_change -------------------------------------------------------------------


def test_answer_bill_change_uses_overall_assessment_at_minimum():
    ans = answer_bill_change(_ctx())
    assert "normal, expected levels" in ans.answer
    assert ans.confidence == "High"


def test_answer_bill_change_includes_cost_driver_and_weather_summary():
    report = _analyst_report(
        largest_cost_driver="Gas (70% of combined cost)",
        weather_vs_behavioural_summary="2025: actual 4000 kWh vs weather-adjusted 3800 kWh -- exceeds expectation.",
    )
    ans = answer_bill_change(_ctx(analyst_report=report))
    assert "Gas (70% of combined cost)" in ans.answer
    assert "exceeds expectation" in ans.answer
    assert len(ans.evidence) >= 2


# --- answer_winter_comparison --------------------------------------------------------------


def test_answer_winter_comparison_none_with_insufficient_history():
    ans = answer_winter_comparison(_ctx(clean=_clean_df(n=6, start="2024-01-01")))
    assert "Not enough winters" in ans.answer
    assert ans.confidence is None


def test_answer_winter_comparison_reports_real_change():
    idx = pd.date_range("2023-09-01", periods=29, freq="MS")
    kwh = []
    for m in idx:
        if m in (pd.Timestamp("2023-12-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")):
            kwh.append(300.0)
        elif m in (pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")):
            kwh.append(400.0)
        else:
            kwh.append(50.0)
    df = pd.DataFrame({"month_start": idx, "consumption_kwh": kwh, "cost_gbp": [v * 0.3 for v in kwh]})

    ans = answer_winter_comparison(_ctx(clean=df))
    assert "Winter 2024/2025" in ans.answer
    assert "winter 2023/2024" in ans.answer
    assert ans.confidence == "High"


# --- answer_fuel_focus ----------------------------------------------------------------------


def test_answer_fuel_focus_uses_recommendation_when_it_fired():
    rec = _recommendation("Focus on gas", "Gas accounts for 82% of annual energy...")
    ans = answer_fuel_focus(_ctx(analyst_report=_analyst_report(recommendations=[rec])))
    assert ans.answer == rec.action
    assert ans.confidence == "High"


def test_answer_fuel_focus_honest_when_findings_exist_but_no_clear_winner():
    report = _analyst_report(fuel_mix_finding=_finding("Fuel mix", "Gas is 55%..."), weather_sensitivity_finding=_finding("Weather sensitivity by fuel", "Gas is 52%..."))
    ans = answer_fuel_focus(_ctx(analyst_report=report))
    assert "no confident single recommendation" in ans.answer
    assert ans.confidence == "Medium"


def test_answer_fuel_focus_not_enough_data():
    ans = answer_fuel_focus(_ctx())
    assert "needs both Electricity and Gas" in ans.answer
    assert ans.confidence is None


# --- answer_forecast --------------------------------------------------------------------------


def _forecast_result(model="Seasonal Naive", p10=90.0, p50=100.0, p90=110.0) -> ForecastResult:
    dates = pd.date_range("2026-08-01", periods=1, freq="MS")
    return ForecastResult(
        model_name=model, comparison=pd.DataFrame(), forecast_dates=dates,
        point=np.array([p50]), p10=np.array([p10]), p50=np.array([p50]), p90=np.array([p90]),
    )


def test_answer_forecast_none_when_no_forecast():
    ans = answer_forecast(_ctx())
    assert "Not enough history" in ans.answer


def test_answer_forecast_converts_to_gbp():
    df = _clean_df(n=12, kwh=100.0)  # unit rate 0.3
    ans = answer_forecast(_ctx(clean=df, forecast_12mo=_forecast_result()))
    assert "Seasonal Naive" in ans.answer
    assert "£30" in ans.answer  # 100 kWh * 0.3 = £30 likely bill


def test_answer_forecast_includes_per_fuel_breakdown_when_available():
    elec_df = _clean_df(n=12, kwh=100.0)
    gas_df = _clean_df(n=12, kwh=300.0)
    total_df = _clean_df(n=12, kwh=400.0)
    fuel_frames = {"electricity": elec_df, "gas": gas_df, "total": total_df}
    multi_forecasts = {
        "electricity": _forecast_result(p50=1200.0),
        "gas": _forecast_result(p50=3600.0),
    }
    ans = answer_forecast(_ctx(clean=total_df, forecast_12mo=_forecast_result(p50=4800.0), fuel_frames=fuel_frames, multi_fuel_forecasts=multi_forecasts))
    assert "Broken down by fuel" in ans.answer
    assert any("forecast sum" in e.lower() for e in ans.evidence)


# --- answer_benchmark -------------------------------------------------------------------------


def test_answer_benchmark_no_fuel_data():
    ans = answer_benchmark(_ctx())
    assert "Not enough fuel-level data" in ans.answer


def test_answer_benchmark_reports_band_for_available_fuels():
    ans = answer_benchmark(_ctx(fuel_frames={"electricity": _clean_df(n=12, kwh=100.0)}))
    assert "Electricity usage is" in ans.answer
    assert "UK household" in ans.answer


# --- answer_carbon ----------------------------------------------------------------------------


def test_answer_carbon_no_data():
    ans = answer_carbon(_ctx())
    assert "Not enough complete-year data" in ans.answer


def test_answer_carbon_reports_real_split():
    fuel_frames = {"electricity": _clean_df(n=12, kwh=100.0), "gas": _clean_df(n=12, kwh=300.0)}
    ans = answer_carbon(_ctx(fuel_frames=fuel_frames))
    assert "tonnes CO2e" in ans.answer
    assert "electricity" in ans.answer
    assert "gas" in ans.answer


# --- answer_last_month_anomaly ------------------------------------------------------------------


def test_answer_last_month_normal_when_not_flagged():
    ans = answer_last_month_anomaly(_ctx())
    assert "looks normal" in ans.answer


def test_answer_last_month_flags_real_anomaly():
    df = _clean_df(n=6, start="2024-08-01")
    last_month = df["month_start"].max()
    anomaly = Anomaly(date=last_month, methods=["stl_esd", "rolling_zscore"], direction="spike", rank_context="1st highest of 6 months")
    ans = answer_last_month_anomaly(_ctx(clean=df, anomalies=[anomaly]))
    assert "spike" in ans.answer
    assert ans.confidence == "Medium"  # 2 methods


def test_answer_last_month_no_data():
    ans = answer_last_month_anomaly(_ctx(clean=pd.DataFrame()))
    assert ans.answer == "No data available."
    assert ans.confidence is None


# --- answer_savings ---------------------------------------------------------------------------


def test_answer_savings_none_recommendations():
    ans = answer_savings(_ctx())
    assert ans.answer == NO_RECOMMENDATIONS_MESSAGE


def test_answer_savings_prefers_largest_saving():
    rec = _recommendation("Review heating schedule", "Reduce winter heating by 10%...", saving=120.0)
    ans = answer_savings(_ctx(analyst_report=_analyst_report(largest_saving=rec, recommendations=[rec])))
    assert "£120.00" in ans.answer


def test_answer_savings_skips_collect_more_data_when_alternative_exists():
    collect_more = _recommendation("Collect more historical data", "Keep tracking for another year...")
    fuel_focus = _recommendation("Focus on gas", "Gas accounts for 82% of annual energy...")
    ans = answer_savings(_ctx(analyst_report=_analyst_report(recommendations=[collect_more, fuel_focus])))
    assert ans.answer == fuel_focus.action


def test_answer_savings_falls_back_to_collect_more_data_when_its_the_only_one():
    collect_more = _recommendation("Collect more historical data", "Keep tracking for another year...")
    ans = answer_savings(_ctx(analyst_report=_analyst_report(recommendations=[collect_more])))
    assert ans.answer == collect_more.action
