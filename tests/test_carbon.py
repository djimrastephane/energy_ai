import numpy as np
import pandas as pd
import pytest

from config import SETTINGS
from src.carbon import (
    combined_annual_emissions,
    estimate_annual_emissions,
    estimate_monthly_emissions,
    forecast_emissions,
    weather_adjusted_annual_emissions,
)
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult


def _clean_df(n=24, start="2023-01-01", kwh=300.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    df = pd.DataFrame({"month_start": idx, "consumption_kwh": np.full(n, kwh), "cost_gbp": np.full(n, kwh * 0.3)})
    df["year"] = df["month_start"].dt.year
    df["is_partial_year"] = df.groupby("year")["month_start"].transform("size") < 12
    return df


def test_estimate_monthly_emissions_uses_correct_factor():
    result = estimate_monthly_emissions(_clean_df(n=1, kwh=100.0), "electricity")
    assert result["kg_co2e"].iloc[0] == pytest.approx(100.0 * SETTINGS.carbon.electricity_kg_co2e_per_kwh)


def test_estimate_monthly_emissions_gas_uses_different_factor():
    elec = estimate_monthly_emissions(_clean_df(n=1, kwh=100.0), "electricity")
    gas = estimate_monthly_emissions(_clean_df(n=1, kwh=100.0), "gas")
    assert elec["kg_co2e"].iloc[0] != gas["kg_co2e"].iloc[0]


def test_estimate_monthly_emissions_rejects_total_fuel():
    with pytest.raises(ValueError, match="aren't supported"):
        estimate_monthly_emissions(_clean_df(n=1), "total")


def test_estimate_monthly_emissions_empty_on_empty_df():
    result = estimate_monthly_emissions(pd.DataFrame(), "electricity")
    assert result.empty
    assert list(result.columns) == ["month_start", "kg_co2e"]


def test_estimate_annual_emissions_excludes_partial_years():
    df = _clean_df(n=15, start="2023-10-01")  # 2023: Oct-Dec (partial), 2024: Jan-Dec (complete), 2025: partial start
    result = estimate_annual_emissions(df, "electricity")
    assert list(result["year"]) == [2024]


def test_estimate_annual_emissions_non_negative_and_component_correct():
    result = estimate_annual_emissions(_clean_df(n=12, kwh=300.0), "gas")
    expected_kwh = 12 * 300.0
    assert (result["kg_co2e"] >= 0).all()
    assert result["kg_co2e"].iloc[0] == pytest.approx(expected_kwh * SETTINGS.carbon.gas_kg_co2e_per_kwh)
    assert result["tonnes_co2e"].iloc[0] == pytest.approx(result["kg_co2e"].iloc[0] / 1000)


def test_estimate_annual_emissions_total_fuel_returns_empty():
    result = estimate_annual_emissions(_clean_df(n=12), "total")
    assert result.empty


def test_combined_annual_emissions_equals_component_sum():
    fuel_dfs = {"electricity": _clean_df(n=12, kwh=100.0), "gas": _clean_df(n=12, kwh=300.0)}
    combined = combined_annual_emissions(fuel_dfs)

    row = combined.iloc[0]
    assert row["combined_kg_co2e"] == pytest.approx(row["electricity_kg_co2e"] + row["gas_kg_co2e"])
    assert row["combined_tonnes_co2e"] == pytest.approx(row["combined_kg_co2e"] / 1000)


def test_combined_annual_emissions_handles_missing_fuel_without_double_counting():
    combined = combined_annual_emissions({"electricity": _clean_df(n=12, kwh=100.0)})
    row = combined.iloc[0]
    assert row["combined_kg_co2e"] == pytest.approx(row["electricity_kg_co2e"])
    assert pd.isna(row["gas_kg_co2e"])


def test_combined_annual_emissions_empty_when_no_fuels():
    assert combined_annual_emissions({}).empty


def _annual_comparison_inputs(n=12, elevation_kwh=50.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    days = idx.days_in_month
    baseline_kwh = 300.0
    consumption = np.full(n, baseline_kwh) + elevation_kwh
    merged_df = pd.DataFrame(
        {"month_start": idx, "consumption_kwh": consumption, "cost_gbp": consumption * 0.3, "days_in_month": days}
    )
    result = EnergySignatureResult(
        intercept=0.0,
        intercept_se=0.0,
        heating_slope=0.0,
        heating_se=0.0,
        heating_pvalue=float("nan"),
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=float("nan"),
        r_squared=0.5,
        adj_r_squared=0.5,
        durbin_watson=2.0,
        n_obs=n,
        fitted=pd.Series(baseline_kwh / days, index=idx),
        resid=pd.Series((consumption - baseline_kwh) / days, index=idx),
    )
    return merged_df, result


def test_weather_adjusted_annual_emissions_applies_factor_to_both_columns():
    merged_df, result = _annual_comparison_inputs()
    emissions = weather_adjusted_annual_emissions(merged_df, result, "gas")

    row = emissions.iloc[0]
    factor = SETTINGS.carbon.gas_kg_co2e_per_kwh
    assert row["actual_kg_co2e"] == pytest.approx(row["actual_kwh"] * factor)
    assert row["weather_predicted_kg_co2e"] == pytest.approx(row["weather_predicted_kwh"] * factor)


def test_forecast_emissions_converts_kwh_bands_to_kg_co2e():
    dates = pd.date_range("2026-08-01", periods=1, freq="MS")
    forecast = ForecastResult(
        model_name="Seasonal Naive",
        comparison=pd.DataFrame(),
        forecast_dates=dates,
        point=np.array([100.0]),
        p10=np.array([90.0]),
        p50=np.array([100.0]),
        p90=np.array([110.0]),
    )
    emissions = forecast_emissions(forecast, "electricity")

    factor = SETTINGS.carbon.electricity_kg_co2e_per_kwh
    assert emissions["likely_kg_co2e"] == pytest.approx(100.0 * factor)
    assert emissions["best_kg_co2e"] < emissions["likely_kg_co2e"] < emissions["worst_kg_co2e"]
