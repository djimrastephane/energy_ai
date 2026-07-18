import numpy as np
import pandas as pd

from src.comparisons import (
    compare_annual_totals,
    compare_monthly,
    compare_seasonality,
    compare_weather_adjusted_annual,
    compare_weather_sensitivity,
)
from src.decomposition import STLResult
from src.energy_signature import EnergySignatureResult


def _clean_df(n=24, start="2023-01-01", kwh=300.0, unit_rate=0.3) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    consumption = np.full(n, kwh)
    df = pd.DataFrame(
        {
            "month_start": idx,
            "consumption_kwh": consumption,
            "cost_gbp": consumption * unit_rate,
        }
    )
    df["year"] = df["month_start"].dt.year
    months_per_year = df.groupby("year")["month_start"].transform("size")
    df["is_partial_year"] = months_per_year < 12
    return df


def test_compare_annual_totals_merges_fuels_by_year():
    fuel_dfs = {
        "electricity": _clean_df(n=24, kwh=100.0),
        "gas": _clean_df(n=24, kwh=300.0),
    }
    result = compare_annual_totals(fuel_dfs)

    assert list(result["year"]) == [2023, 2024]
    assert result.loc[result["year"] == 2023, "electricity_kwh"].iloc[0] == 1200.0
    assert result.loc[result["year"] == 2023, "gas_kwh"].iloc[0] == 3600.0


def test_compare_annual_totals_missing_fuel_yields_nan_not_crash():
    fuel_dfs = {"electricity": _clean_df(n=24), "gas": pd.DataFrame()}
    result = compare_annual_totals(fuel_dfs)

    assert "electricity_kwh" in result.columns
    assert "gas_kwh" not in result.columns  # empty fuel contributes nothing, doesn't crash


def test_compare_annual_totals_empty_input_returns_empty_frame():
    result = compare_annual_totals({})
    assert result.empty


def test_compare_monthly_outer_joins_and_preserves_missing_months():
    elec = _clean_df(n=3, start="2023-01-01", kwh=100.0)
    gas = _clean_df(n=2, start="2023-02-01", kwh=300.0)  # starts one month later
    result = compare_monthly({"electricity": elec, "gas": gas})

    assert len(result) == 3  # Jan, Feb, Mar -- Jan's gas_kwh is NaN, not dropped
    jan_row = result[result["month_start"] == pd.Timestamp("2023-01-01")].iloc[0]
    assert jan_row["electricity_kwh"] == 100.0
    assert pd.isna(jan_row["gas_kwh"])


def _stl_result(seasonal_strength: float, trend_strength: float) -> STLResult:
    idx = pd.date_range("2023-01-01", periods=12, freq="MS")
    placeholder = pd.Series(np.zeros(12), index=idx)
    return STLResult(
        observed=placeholder,
        trend=placeholder,
        seasonal=placeholder,
        resid=placeholder,
        seasonal_strength=seasonal_strength,
        trend_strength=trend_strength,
    )


def test_compare_seasonality_tabulates_per_fuel():
    result = compare_seasonality({"gas": _stl_result(0.9, 0.1), "electricity": _stl_result(0.4, 0.5)})

    assert set(result["fuel"]) == {"gas", "electricity"}
    gas_row = result[result["fuel"] == "gas"].iloc[0]
    assert gas_row["seasonal_strength"] == 0.9


def test_compare_seasonality_skips_none_results():
    result = compare_seasonality({"gas": _stl_result(0.9, 0.1), "electricity": None})
    assert list(result["fuel"]) == ["gas"]


def _energy_result(heating_slope: float, heating_p: float, r_squared: float) -> EnergySignatureResult:
    return EnergySignatureResult(
        intercept=1.0,
        intercept_se=0.1,
        heating_slope=heating_slope,
        heating_se=0.1,
        heating_pvalue=heating_p,
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=float("nan"),
        r_squared=r_squared,
        adj_r_squared=r_squared - 0.01,
        durbin_watson=1.8,
        n_obs=12,
        fitted=pd.Series(dtype=float),
        resid=pd.Series(dtype=float),
    )


def test_compare_weather_sensitivity_none_without_both_fuels():
    assert compare_weather_sensitivity({"electricity": _energy_result(0.2, 0.001, 0.4)}) is None
    assert compare_weather_sensitivity({}) is None


def test_compare_weather_sensitivity_none_when_neither_significant():
    fuels = {
        "electricity": _energy_result(0.2, 0.6, 0.1),
        "gas": _energy_result(1.4, 0.7, 0.1),
    }
    assert compare_weather_sensitivity(fuels) is None


def test_compare_weather_sensitivity_matches_real_data_split():
    """Locks in the real-data payoff check from this session: gas ~89%, electricity ~11%."""
    fuels = {
        "electricity": _energy_result(heating_slope=0.189, heating_p=0.0003, r_squared=0.34),
        "gas": _energy_result(heating_slope=1.472, heating_p=0.0000, r_squared=0.69),
    }
    finding = compare_weather_sensitivity(fuels)

    assert finding is not None
    assert finding.confidence == "High"
    assert "gas accounts for 89%" in finding.narrative
    assert "electricity accounts for 11%" in finding.narrative
    assert "gas-related" in finding.narrative


def test_compare_weather_sensitivity_medium_confidence_when_only_one_significant():
    fuels = {
        "electricity": _energy_result(heating_slope=0.05, heating_p=0.8, r_squared=0.05),
        "gas": _energy_result(heating_slope=1.5, heating_p=0.0001, r_squared=0.7),
    }
    finding = compare_weather_sensitivity(fuels)

    assert finding is not None
    assert finding.confidence == "Medium"


def _annual_comparison_inputs(n=24, start="2023-01-01", elevation_kwh=0.0, elevated_year=None):
    idx = pd.date_range(start, periods=n, freq="MS")
    days = idx.days_in_month
    baseline_kwh = 300.0
    consumption = np.full(n, baseline_kwh)
    if elevated_year is not None:
        consumption[idx.year == elevated_year] += elevation_kwh
    merged_df = pd.DataFrame(
        {"month_start": idx, "consumption_kwh": consumption, "cost_gbp": consumption * 0.3, "days_in_month": days}
    )
    fitted_avg_daily = baseline_kwh / days
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
        fitted=pd.Series(fitted_avg_daily, index=idx),
        resid=pd.Series((consumption - baseline_kwh) / days, index=idx),
    )
    return merged_df, result


def test_compare_weather_adjusted_annual_stacks_fuels_with_fuel_column():
    elec_merged, elec_result = _annual_comparison_inputs(elevated_year=2023, elevation_kwh=20.0)
    gas_merged, gas_result = _annual_comparison_inputs(elevated_year=2023, elevation_kwh=80.0)

    result = compare_weather_adjusted_annual(
        {"electricity": (elec_merged, elec_result), "gas": (gas_merged, gas_result)}
    )

    assert set(result["fuel"]) == {"electricity", "gas"}
    assert "difference_kwh" in result.columns
    gas_2023 = result[(result["fuel"] == "gas") & (result["year"] == 2023)].iloc[0]
    assert gas_2023["difference_kwh"] > 0  # gas was deliberately elevated that year


def test_compare_weather_adjusted_annual_empty_when_no_complete_years():
    merged, result = _annual_comparison_inputs(n=6)  # under a year, no complete calendar year
    combined = compare_weather_adjusted_annual({"electricity": (merged, result)})
    assert combined.empty
