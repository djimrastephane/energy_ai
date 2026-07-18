import numpy as np
import pandas as pd

from src.anomalies import Anomaly
from src.cross_fuel_anomalies import cross_fuel_anomaly_insights
from src.energy_signature import EnergySignatureResult


def _clean_df(n=6, start="2024-08-01", kwh=100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame({"month_start": idx, "consumption_kwh": np.full(n, kwh), "cost_gbp": np.full(n, kwh * 0.3)})


def _anomaly(date, methods, direction) -> Anomaly:
    return Anomaly(date=pd.Timestamp(date), methods=methods, direction=direction, rank_context="1st highest")


def _energy_result(resid: pd.Series) -> EnergySignatureResult:
    return EnergySignatureResult(
        intercept=1.0,
        intercept_se=0.1,
        heating_slope=1.0,
        heating_se=0.1,
        heating_pvalue=0.001,
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=float("nan"),
        r_squared=0.6,
        adj_r_squared=0.59,
        durbin_watson=1.8,
        n_obs=len(resid),
        fitted=pd.Series(dtype=float),
        resid=resid,
    )


def test_electricity_only_flags_appliance_or_occupancy():
    fuel_anomalies = {"electricity": [_anomaly("2024-12-01", ["stl_esd"], "spike")], "gas": []}
    fuel_clean = {"electricity": _clean_df(), "gas": _clean_df()}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean)

    assert len(insights) == 1
    assert insights[0].pattern == "electricity_only"
    assert "appliance or occupancy" in insights[0].narrative


def test_gas_only_flags_heating_event():
    fuel_anomalies = {"electricity": [], "gas": [_anomaly("2024-12-01", ["stl_esd", "rolling_zscore"], "spike")]}
    fuel_clean = {"electricity": _clean_df(), "gas": _clean_df()}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean)

    assert insights[0].pattern == "gas_only"
    assert "heating event" in insights[0].narrative
    assert insights[0].confidence == "Medium"  # 2 methods


def test_total_only_when_no_fuel_breakdown_available():
    fuel_anomalies = {"total": [_anomaly("2024-12-01", ["stl_esd"], "spike")]}
    fuel_clean = {"total": _clean_df()}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean)

    assert insights[0].pattern == "total_only"
    assert "isn't available to attribute" in insights[0].narrative


def test_both_same_direction_weather_driven_when_residuals_small():
    idx = pd.date_range("2024-08-01", periods=6, freq="MS")
    small_resid = pd.Series(np.zeros(6), index=idx)  # exactly at weather-adjusted expectation
    fuel_anomalies = {
        "electricity": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
        "gas": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
    }
    fuel_clean = {"electricity": _clean_df(), "gas": _clean_df()}
    fuel_energy = {"electricity": _energy_result(small_resid), "gas": _energy_result(small_resid)}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean, fuel_energy)

    assert insights[0].pattern == "both"
    assert "weather-driven" in insights[0].narrative
    assert insights[0].confidence == "High"


def test_both_same_direction_partially_unexplained_when_residual_material():
    idx = pd.date_range("2024-08-01", periods=6, freq="MS")  # Aug, Sep, Oct, Nov, Dec, Jan
    # Mostly-zero residuals except December (index 4), a clear outlier relative to the rest.
    values = np.array([0.1, -0.1, 0.05, -0.05, 20.0, 0.0])
    resid = pd.Series(values, index=idx)
    fuel_anomalies = {
        "electricity": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
        "gas": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
    }
    fuel_clean = {"electricity": _clean_df(), "gas": _clean_df()}
    fuel_energy = {"electricity": _energy_result(resid), "gas": _energy_result(resid)}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean, fuel_energy)

    assert insights[0].pattern == "both"
    assert "worth investigating further" in insights[0].narrative
    assert insights[0].confidence == "Medium"


def test_both_no_weather_data_stays_medium_and_honest_about_the_gap():
    fuel_anomalies = {
        "electricity": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
        "gas": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
    }
    fuel_clean = {"electricity": _clean_df(), "gas": _clean_df()}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean)  # no fuel_energy_results

    assert insights[0].pattern == "both"
    assert "wasn't run" in insights[0].narrative
    assert insights[0].confidence == "Medium"


def test_december_2024_real_pattern_gas_dominant_conflicting_directions():
    """Regression test locking in the real December 2024 finding from manual verification:
    electricity *dropped* (1 method, weak) while gas *spiked* (3 methods, strong) -- opposite
    directions, so this must NOT be classified as a shared "both moved together" driver, and
    must be attributed to gas specifically with High confidence, honestly noting the conflict.
    """
    fuel_anomalies = {
        "electricity": [_anomaly("2024-12-01", ["stl_esd"], "drop")],
        "gas": [_anomaly("2024-12-01", ["rolling_zscore", "stl_esd", "isolation_forest"], "spike")],
    }
    fuel_clean = {
        "electricity": _clean_df(kwh=149.0),
        "gas": _clean_df(kwh=589.0),
    }

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean)

    assert len(insights) == 1
    insight = insights[0]
    assert insight.month == pd.Timestamp("2024-12-01")
    assert insight.pattern == "both"
    assert "opposite directions" in insight.narrative
    assert "gas-specific event" in insight.narrative
    assert insight.confidence == "High"
    assert any("conflict" in e.lower() for e in insight.evidence)


def test_conflicting_directions_equal_strength_is_low_confidence():
    fuel_anomalies = {
        "electricity": [_anomaly("2024-12-01", ["stl_esd"], "drop")],
        "gas": [_anomaly("2024-12-01", ["stl_esd"], "spike")],
    }
    fuel_clean = {"electricity": _clean_df(), "gas": _clean_df()}

    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean)

    assert insights[0].pattern == "both"
    assert "conflicting evidence" in insights[0].narrative
    assert insights[0].confidence == "Low"


def test_empty_inputs_return_empty_list():
    assert cross_fuel_anomaly_insights({}, {}) == []
