"""Multi-fuel comparisons: annual, monthly, seasonality, weather sensitivity,
weather-adjusted annual, and cross-fuel anomaly attribution.

Every function here takes *already-computed* per-fuel results (from
`src.kpis`, `src.decomposition`, `src.energy_signature`, `src.anomalies` --
each already generic over Electricity/Gas/Total via the sidebar's Fuel
selector) and diffs/compares them. Nothing in this module fits a new model
or runs a new statistical test -- it only cross-references results that
already exist, per the "reuse the existing statistical engine, never
duplicate logic" principle this phase is scoped to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.confidence import Confidence
from src.decomposition import STLResult
from src.energy_signature import EnergySignatureResult, annual_weather_adjusted_comparison
from src.findings import Finding
from src.ingestion import EnergyType
from src.kpis import full_year_comparison


def _merge_all(frames: list[pd.DataFrame], on: str) -> pd.DataFrame:
    """Outer-join a list of per-fuel frames on ``on`` -- missing fuel/month combinations
    show up as NaN rather than silently dropping rows, so a missing stream is visible,
    not hidden."""
    if not frames:
        return pd.DataFrame(columns=[on])
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=on, how="outer")
    return combined.sort_values(on).reset_index(drop=True)


def compare_annual_totals(fuel_clean_dfs: dict[EnergyType, pd.DataFrame]) -> pd.DataFrame:
    """Per-complete-calendar-year kWh/£/YoY% for every fuel with data, side by side.

    Reuses ``src.kpis.full_year_comparison`` once per fuel -- no new
    aggregation logic. A fuel absent from ``fuel_clean_dfs`` (or with no
    complete year) simply contributes no columns rather than raising.
    """
    frames = []
    for fuel, df in fuel_clean_dfs.items():
        if df.empty:
            continue
        yearly = full_year_comparison(df)
        if yearly.empty:
            continue
        frames.append(
            yearly.rename(
                columns={
                    "total_kwh": f"{fuel}_kwh",
                    "total_cost_gbp": f"{fuel}_cost_gbp",
                    "yoy_kwh_pct": f"{fuel}_yoy_kwh_pct",
                    "yoy_cost_pct": f"{fuel}_yoy_cost_pct",
                }
            )
        )
    return _merge_all(frames, on="year")


def compare_monthly(fuel_clean_dfs: dict[EnergyType, pd.DataFrame]) -> pd.DataFrame:
    """Wide monthly kWh/£ table, one pair of columns per fuel present."""
    frames = []
    for fuel, df in fuel_clean_dfs.items():
        if df.empty:
            continue
        frames.append(
            df[["month_start", "consumption_kwh", "cost_gbp"]].rename(
                columns={"consumption_kwh": f"{fuel}_kwh", "cost_gbp": f"{fuel}_cost_gbp"}
            )
        )
    return _merge_all(frames, on="month_start")


def compare_seasonality(fuel_stl_results: dict[EnergyType, STLResult | None]) -> pd.DataFrame:
    """Seasonal/trend strength side by side per fuel -- tabulates STL results already computed."""
    rows = [
        {"fuel": fuel, "seasonal_strength": result.seasonal_strength, "trend_strength": result.trend_strength}
        for fuel, result in fuel_stl_results.items()
        if result is not None
    ]
    return pd.DataFrame(rows, columns=["fuel", "seasonal_strength", "trend_strength"])


@dataclass
class WeatherSensitivityShares:
    """The numeric core of ``compare_weather_sensitivity``, exposed separately so other
    modules (e.g. ``src.recommendations.recommend_fuel_focus``) can consume the same shares
    without re-deriving them or parsing the Finding's prose narrative."""

    gas_share_pct: float
    electricity_share_pct: float
    gas_significant: bool
    electricity_significant: bool
    dominant_fuel: EnergyType


def compute_weather_sensitivity_shares(
    fuel_energy_results: Mapping[EnergyType, EnergySignatureResult | None],
) -> WeatherSensitivityShares | None:
    """Normalized share of the two fuels' combined heating-degree-day slope.

    Returns ``None`` under the same conditions ``compare_weather_sensitivity``
    returns ``None`` for (missing fuel, or neither fuel significant).
    """
    elec = fuel_energy_results.get("electricity")
    gas = fuel_energy_results.get("gas")
    if elec is None or gas is None:
        return None

    elec_significant = not np.isnan(elec.heating_pvalue) and elec.heating_pvalue < 0.05
    gas_significant = not np.isnan(gas.heating_pvalue) and gas.heating_pvalue < 0.05
    if not (elec_significant or gas_significant):
        return None

    elec_slope = max(elec.heating_slope, 0.0) if elec_significant else 0.0
    gas_slope = max(gas.heating_slope, 0.0) if gas_significant else 0.0
    total_slope = elec_slope + gas_slope
    if total_slope <= 0:
        return None

    gas_share = gas_slope / total_slope * 100
    elec_share = elec_slope / total_slope * 100
    return WeatherSensitivityShares(
        gas_share_pct=gas_share,
        electricity_share_pct=elec_share,
        gas_significant=gas_significant,
        electricity_significant=elec_significant,
        dominant_fuel="gas" if gas_share >= elec_share else "electricity",
    )


def compare_weather_sensitivity(
    fuel_energy_results: Mapping[EnergyType, EnergySignatureResult | None],
) -> Finding | None:
    """Task 5 -- weather attribution: which fuel's usage responds more strongly to weather.

    Requires both an electricity and a gas ``EnergySignatureResult`` (each
    already fitted independently elsewhere in the app via
    ``fit_energy_signature`` -- this does not fit anything new). Reports two
    complementary numbers: each fuel's own R-squared ("weather explains X%
    of this fuel's variation"), and a normalized share of the two fuels'
    combined heating-degree-day slope (a sharper "which fuel actually moves
    more per degree of cold" comparison, via :func:`compute_weather_sensitivity_shares`).
    Returns ``None`` if neither fuel shows a statistically significant
    heating response -- there is nothing to attribute.
    """
    elec = fuel_energy_results.get("electricity")
    gas = fuel_energy_results.get("gas")
    shares = compute_weather_sensitivity_shares(fuel_energy_results)
    if elec is None or gas is None or shares is None:
        return None

    narrative = (
        f"Weather explains {gas.r_squared:.0%} of the variation in gas consumption and "
        f"{elec.r_squared:.0%} of the variation in electricity consumption. Comparing heating "
        f"sensitivity directly, gas accounts for {shares.gas_share_pct:.0f}% of the two fuels' "
        f"combined heating-driven response and electricity accounts for "
        f"{shares.electricity_share_pct:.0f}%. Most heating demand therefore appears to be "
        f"{shares.dominant_fuel}-related."
    )
    evidence = [
        f"Gas: heating slope {gas.heating_slope:.3f} kWh/day per HDD (p={gas.heating_pvalue:.4f}), R-squared {gas.r_squared:.0%}",
        f"Electricity: heating slope {elec.heating_slope:.3f} kWh/day per HDD (p={elec.heating_pvalue:.4f}), R-squared {elec.r_squared:.0%}",
    ]
    confidence: Confidence = "High" if (shares.electricity_significant and shares.gas_significant) else "Medium"
    confidence_reason = (
        "Both fuels show a statistically significant heating response."
        if confidence == "High"
        else "Only one fuel shows a statistically significant heating response; the split is "
        "still arithmetically valid but rests on a single significant estimate."
    )

    return Finding(
        title="Weather sensitivity by fuel",
        narrative=narrative,
        evidence=evidence,
        confidence=confidence,
        confidence_reason=confidence_reason,
        category="weather",
    )


def compare_weather_adjusted_annual(
    fuel_merged_and_results: dict[EnergyType, tuple[pd.DataFrame, EnergySignatureResult]],
) -> pd.DataFrame:
    """Task 7 -- observed vs. weather-adjusted vs. behavioural difference, per year, per fuel.

    Thin wrapper: calls the existing
    ``src.energy_signature.annual_weather_adjusted_comparison`` once per
    fuel and stacks the results with a ``fuel`` column. That function
    already computes exactly this (actual/weather-predicted/difference/
    interpretation); nothing new is modeled here.
    """
    frames = []
    for fuel, (merged_df, result) in fuel_merged_and_results.items():
        comparison = annual_weather_adjusted_comparison(merged_df, result)
        if comparison.empty:
            continue
        comparison = comparison.copy()
        comparison.insert(0, "fuel", fuel)
        frames.append(comparison)
    if not frames:
        return pd.DataFrame(
            columns=["fuel", "year", "actual_kwh", "weather_predicted_kwh", "difference_kwh", "interpretation"]
        )
    return pd.concat(frames, ignore_index=True)
