import numpy as np
import pandas as pd
import pytest

from src.energy_signature import (
    EnergySignatureResult,
    annual_weather_adjusted_comparison,
    fit_energy_signature,
    interpret_energy_signature,
    partial_dependence,
)


def _synthetic_signature_df(
    n=24, intercept=5.0, heating_slope=0.8, noise_std=0.05, seed=0, with_cdd_variation=False
):
    rng = np.random.default_rng(seed)
    hdd = rng.uniform(0, 15, n)
    # cdd deliberately has no true effect on kwh either way -- with_cdd_variation only
    # controls whether it has *any* variance, to exercise the "not estimable" branch
    # (mirroring the real all-zero Aberdeen data) separately from the "estimated but
    # not significant" branch.
    cdd = rng.uniform(0, 3, n) if with_cdd_variation else np.zeros(n)
    kwh = intercept + heating_slope * hdd + rng.normal(0, noise_std, n)
    return pd.DataFrame(
        {
            "month_start": pd.date_range("2022-01-01", periods=n, freq="MS"),
            "avg_daily_kwh": kwh,
            "avg_daily_hdd": hdd,
            "avg_daily_cdd": cdd,
        }
    )


def test_fit_energy_signature_recovers_known_relationship():
    df = _synthetic_signature_df()

    result = fit_energy_signature(df)

    assert result.intercept == pytest.approx(5.0, abs=0.1)
    assert result.heating_slope == pytest.approx(0.8, abs=0.05)
    assert result.heating_pvalue < 0.05
    assert result.r_squared > 0.9


def test_fit_energy_signature_raises_on_insufficient_data():
    df = _synthetic_signature_df(n=4)

    with pytest.raises(ValueError, match="at least 6"):
        fit_energy_signature(df)


def test_interpret_energy_signature_reports_significant_heating_and_not_estimable_cooling():
    df = _synthetic_signature_df()  # cdd all zero, mirrors the real Aberdeen data
    result = fit_energy_signature(df)

    assert np.isnan(result.cooling_pvalue)
    assert result.cooling_slope == 0.0

    text = interpret_energy_signature(result, unit_rate_gbp_per_kwh=0.30)

    assert "statistically significant effect" in text  # heating
    assert "couldn't be estimated" in text  # cooling: no variation in the data


def test_interpret_energy_signature_reports_not_significant_when_cdd_has_no_true_effect():
    df = _synthetic_signature_df(with_cdd_variation=True)
    result = fit_energy_signature(df)

    assert not np.isnan(result.cooling_pvalue)  # estimable now that cdd actually varies

    text = interpret_energy_signature(result, unit_rate_gbp_per_kwh=0.30)

    assert "couldn't be estimated" not in text


def test_interpret_energy_signature_default_fuel_does_not_claim_electric_heating():
    """Default fuel ('total') combines electricity and gas -- it must not assert 'electric
    heating' specifically, since that's only correct when fitted on electricity-only data."""
    df = _synthetic_signature_df()
    result = fit_energy_signature(df)

    text = interpret_energy_signature(result, unit_rate_gbp_per_kwh=0.30)

    assert "electric heating" not in text
    assert "Fuel selector" in text


def test_interpret_energy_signature_names_electric_heating_for_electricity_fuel():
    df = _synthetic_signature_df()
    result = fit_energy_signature(df)

    text = interpret_energy_signature(result, unit_rate_gbp_per_kwh=0.30, fuel="electricity")

    assert "electric heating" in text


def test_interpret_energy_signature_names_gas_heating_for_gas_fuel():
    df = _synthetic_signature_df()
    result = fit_energy_signature(df)

    text = interpret_energy_signature(result, unit_rate_gbp_per_kwh=0.30, fuel="gas")

    assert "gas heating" in text
    assert "electric heating" not in text


def test_partial_dependence_matches_manual_linear_calc():
    df = _synthetic_signature_df()
    result = fit_energy_signature(df)

    pdp = partial_dependence(result, df, "avg_daily_hdd")

    manual = result.intercept + result.heating_slope * pdp["avg_daily_hdd"] + result.cooling_slope * df[
        "avg_daily_cdd"
    ].mean()
    assert pdp["predicted_avg_daily_kwh"].to_numpy() == pytest.approx(manual.to_numpy())


def _annual_comparison_inputs(elevated_year=2024, elevation_kwh=50.0, n=30, start="2023-09-01"):
    """2023 (4mo, partial), 2024 (12mo, full), 2025 (12mo, full), 2026 (2mo, partial).

    ``fitted`` is set to exactly track a flat 300 kWh/month baseline (via
    avg_daily_kwh = 300/days_in_month), so any month's residual is fully
    controlled by how far its ``consumption_kwh`` deviates from 300 --
    ``elevated_year`` gets ``elevation_kwh`` added to every one of its months.
    """
    idx = pd.date_range(start, periods=n, freq="MS")
    days = idx.days_in_month
    baseline_kwh = 300.0
    consumption = np.full(n, baseline_kwh)
    consumption[idx.year == elevated_year] += elevation_kwh

    merged_df = pd.DataFrame(
        {
            "month_start": idx,
            "consumption_kwh": consumption,
            "cost_gbp": consumption * 0.3,
            "days_in_month": days,
        }
    )
    fitted_avg_daily = baseline_kwh / days  # "weather predicts a flat 300 kWh/month"
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


def test_annual_weather_adjusted_comparison_excludes_incomplete_years():
    merged_df, result = _annual_comparison_inputs()

    table = annual_weather_adjusted_comparison(merged_df, result)

    assert set(table["year"]) == {2024, 2025}  # 2023 (4mo) and 2026 (2mo) excluded


def test_annual_weather_adjusted_comparison_matches_independent_sums():
    merged_df, result = _annual_comparison_inputs(elevated_year=2024, elevation_kwh=50.0)

    table = annual_weather_adjusted_comparison(merged_df, result)
    row_2024 = table[table["year"] == 2024].iloc[0]
    row_2025 = table[table["year"] == 2025].iloc[0]

    expected_actual_2024 = merged_df.loc[merged_df["month_start"].dt.year == 2024, "consumption_kwh"].sum()
    assert row_2024["actual_kwh"] == pytest.approx(expected_actual_2024)
    # 2025 wasn't elevated, so actual should equal the flat 300/month baseline exactly.
    assert row_2025["actual_kwh"] == pytest.approx(300.0 * 12)
    assert row_2025["difference_kwh"] == pytest.approx(0.0, abs=1e-6)


def test_annual_weather_adjusted_comparison_flags_elevated_year_as_behavioural():
    merged_df, result = _annual_comparison_inputs(elevated_year=2024, elevation_kwh=50.0)

    table = annual_weather_adjusted_comparison(merged_df, result)
    row_2024 = table[table["year"] == 2024].iloc[0]
    row_2025 = table[table["year"] == 2025].iloc[0]

    assert row_2024["difference_kwh"] == pytest.approx(50.0 * 12, rel=0.01)
    assert "behavioural change" in row_2024["interpretation"]
    assert "well explained by weather" in row_2025["interpretation"]


def test_annual_weather_adjusted_comparison_empty_when_no_complete_year():
    idx = pd.date_range("2024-01-01", periods=6, freq="MS")
    merged_df = pd.DataFrame(
        {
            "month_start": idx,
            "consumption_kwh": np.full(6, 300.0),
            "days_in_month": idx.days_in_month,
        }
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
        n_obs=6,
        fitted=pd.Series(10.0, index=idx),
        resid=pd.Series(0.0, index=idx),
    )

    table = annual_weather_adjusted_comparison(merged_df, result)

    assert table.empty
    assert list(table.columns) == ["year", "actual_kwh", "weather_predicted_kwh", "difference_kwh", "interpretation"]
