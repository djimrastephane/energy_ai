import numpy as np
import pandas as pd

from src.cost_engine import (
    compute_combined_cost_breakdown,
    compute_cost_breakdown,
    forecast_bill_by_fuel,
)
from src.forecast_evaluation import ForecastResult


def _clean_df(n=24, start="2023-01-01", kwh=300.0, unit_rate=0.3) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    consumption = np.full(n, kwh)
    df = pd.DataFrame(
        {"month_start": idx, "consumption_kwh": consumption, "cost_gbp": consumption * unit_rate}
    )
    df["year"] = df["month_start"].dt.year
    months_per_year = df.groupby("year")["month_start"].transform("size")
    df["is_partial_year"] = months_per_year < 12
    return df


def test_compute_cost_breakdown_none_on_empty_df():
    assert compute_cost_breakdown(pd.DataFrame(), "electricity") is None


def test_compute_cost_breakdown_basic_arithmetic():
    breakdown = compute_cost_breakdown(_clean_df(n=12, kwh=300.0, unit_rate=0.3), "electricity")

    assert breakdown is not None
    assert breakdown.fuel == "electricity"
    assert breakdown.annual_total_gbp == 12 * 300.0 * 0.3
    assert breakdown.effective_unit_rate_gbp_per_kwh == 0.3
    assert "Not enough complete calendar years" in breakdown.trend_description


def test_compute_cost_breakdown_reports_yoy_trend_when_two_years_present():
    idx = pd.date_range("2023-01-01", periods=24, freq="MS")
    consumption = np.where(idx.year == 2023, 300.0, 330.0)  # 2024 costs more
    df = pd.DataFrame({"month_start": idx, "consumption_kwh": consumption, "cost_gbp": consumption * 0.3})
    df["year"] = df["month_start"].dt.year
    df["is_partial_year"] = df.groupby("year")["month_start"].transform("size") < 12

    breakdown = compute_cost_breakdown(df, "gas")

    assert breakdown is not None
    assert "up" in breakdown.trend_description


def test_compute_combined_cost_breakdown_skips_missing_fuels():
    result = compute_combined_cost_breakdown({"electricity": _clean_df(n=12), "gas": pd.DataFrame(), "total": _clean_df(n=12)})

    assert set(result["fuel"]) == {"electricity", "total"}


def _forecast_result(model_name: str, p10: float, p50: float, p90: float) -> ForecastResult:
    dates = pd.date_range("2026-08-01", periods=1, freq="MS")
    return ForecastResult(
        model_name=model_name,
        comparison=pd.DataFrame(),
        forecast_dates=dates,
        point=np.array([p50]),
        p10=np.array([p10]),
        p50=np.array([p50]),
        p90=np.array([p90]),
    )


def test_forecast_bill_by_fuel_converts_kwh_to_gbp_per_fuel():
    results = {"electricity": _forecast_result("Seasonal Naive", 90.0, 100.0, 110.0)}
    rates = {"electricity": 0.30}

    comparison = forecast_bill_by_fuel(results, rates)

    row = comparison.per_fuel.iloc[0]
    assert row["fuel"] == "electricity"
    assert row["likely_gbp"] == 100.0 * 0.30
    assert comparison.summed_likely_gbp is None  # gas missing, no combined comparison


def test_forecast_bill_by_fuel_flags_large_divergence_between_summed_and_total():
    results = {
        "electricity": _forecast_result("Seasonal Naive", 90.0, 100.0, 110.0),
        "gas": _forecast_result("Holt-Winters", 190.0, 200.0, 210.0),
        "total": _forecast_result("SARIMA", 250.0, 260.0, 270.0),  # much lower than 100+200=300 combined
    }
    rates = {"electricity": 1.0, "gas": 1.0, "total": 1.0}

    comparison = forecast_bill_by_fuel(results, rates)

    assert comparison.summed_likely_gbp == 300.0
    assert comparison.total_likely_gbp == 260.0
    assert comparison.divergence_note is not None
    assert "don't always agree" in comparison.divergence_note


def test_forecast_bill_by_fuel_no_divergence_note_when_close():
    results = {
        "electricity": _forecast_result("Seasonal Naive", 90.0, 100.0, 110.0),
        "gas": _forecast_result("Holt-Winters", 190.0, 200.0, 210.0),
        "total": _forecast_result("SARIMA", 290.0, 300.0, 310.0),  # matches 100+200 exactly
    }
    rates = {"electricity": 1.0, "gas": 1.0, "total": 1.0}

    comparison = forecast_bill_by_fuel(results, rates)

    assert comparison.divergence_pct == 0.0
    assert comparison.divergence_note is None
