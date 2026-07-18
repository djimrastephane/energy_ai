"""Task 8 -- carbon analytics: estimated CO2e emissions from electricity and gas.

Emission factors are static, cited config constants (``config.CarbonConfig``)
-- not a live API, consistent with the rest of Phase 4's "not live data"
approach to benchmarking. Only "electricity" and "gas" are supported: for
"total", the electricity/gas split is unknown, so a blended factor can't be
honestly computed without knowing the mix -- callers need both fuels
available (same gate as ``src.fuel.finding_fuel_mix``'s minimum-months
check) rather than guessing at a blended figure.
"""

from __future__ import annotations

import pandas as pd

from config import SETTINGS
from src.energy_signature import EnergySignatureResult, annual_weather_adjusted_comparison
from src.forecast_evaluation import ForecastResult
from src.ingestion import EnergyType

_SUPPORTED_FUELS = ("electricity", "gas")


def _factor(fuel: EnergyType) -> float:
    if fuel == "electricity":
        return SETTINGS.carbon.electricity_kg_co2e_per_kwh
    if fuel == "gas":
        return SETTINGS.carbon.gas_kg_co2e_per_kwh
    raise ValueError(
        f"Carbon estimates aren't supported for fuel={fuel!r} -- the electricity/gas split is "
        "unknown for 'total', so a blended factor can't be honestly computed."
    )


def estimate_monthly_emissions(clean_df: pd.DataFrame, fuel: EnergyType) -> pd.DataFrame:
    """Monthly kg CO2e = consumption_kwh * the fuel's emission factor.

    Only "electricity"/"gas" are supported (see module docstring). Returns
    an empty typed frame if ``clean_df`` is empty.
    """
    if fuel not in _SUPPORTED_FUELS:
        raise ValueError(f"Carbon estimates aren't supported for fuel={fuel!r}; use 'electricity' or 'gas'.")
    if clean_df.empty:
        return pd.DataFrame(columns=["month_start", "kg_co2e"])
    factor = _factor(fuel)
    return pd.DataFrame({"month_start": clean_df["month_start"], "kg_co2e": clean_df["consumption_kwh"] * factor})


def estimate_annual_emissions(clean_df: pd.DataFrame, fuel: EnergyType) -> pd.DataFrame:
    """Per-*complete*-calendar-year kg CO2e / tonnes CO2e, plus YoY % change.

    Mirrors ``src.kpis.full_year_comparison``'s complete-year-only
    convention (excludes rows where ``is_partial_year``) so a partial year
    in progress is never presented as a full-year total.
    """
    if fuel not in _SUPPORTED_FUELS or clean_df.empty:
        return pd.DataFrame(columns=["year", "kg_co2e", "tonnes_co2e", "yoy_pct"])
    complete = clean_df[~clean_df["is_partial_year"]]
    if complete.empty:
        return pd.DataFrame(columns=["year", "kg_co2e", "tonnes_co2e", "yoy_pct"])
    factor = _factor(fuel)
    yearly = complete.groupby("year")["consumption_kwh"].sum().reset_index()
    yearly["kg_co2e"] = yearly["consumption_kwh"] * factor
    yearly["tonnes_co2e"] = yearly["kg_co2e"] / 1000
    yearly["yoy_pct"] = yearly["kg_co2e"].pct_change() * 100
    return yearly[["year", "kg_co2e", "tonnes_co2e", "yoy_pct"]]


def combined_annual_emissions(fuel_clean_dfs: dict[EnergyType, pd.DataFrame]) -> pd.DataFrame:
    """Electricity + Gas annual emissions side by side, plus their component-sum combined total.

    Never applies a single blended factor to a mixed total -- ``combined_kg_co2e`` is always
    ``electricity_kg_co2e + gas_kg_co2e``, each computed from that fuel's own emission factor.
    """
    elec = estimate_annual_emissions(fuel_clean_dfs.get("electricity", pd.DataFrame()), "electricity")
    gas = estimate_annual_emissions(fuel_clean_dfs.get("gas", pd.DataFrame()), "gas")
    if elec.empty and gas.empty:
        return pd.DataFrame(
            columns=["year", "electricity_kg_co2e", "gas_kg_co2e", "combined_kg_co2e", "combined_tonnes_co2e"]
        )
    elec = elec.rename(columns={"kg_co2e": "electricity_kg_co2e"})[["year", "electricity_kg_co2e"]]
    gas = gas.rename(columns={"kg_co2e": "gas_kg_co2e"})[["year", "gas_kg_co2e"]]
    combined = elec.merge(gas, on="year", how="outer").sort_values("year").reset_index(drop=True)
    combined["combined_kg_co2e"] = combined[["electricity_kg_co2e", "gas_kg_co2e"]].sum(axis=1, skipna=True)
    combined["combined_tonnes_co2e"] = combined["combined_kg_co2e"] / 1000
    return combined


def weather_adjusted_annual_emissions(
    merged_df: pd.DataFrame, energy_result: EnergySignatureResult, fuel: EnergyType
) -> pd.DataFrame:
    """Weather-adjusted annual emissions: reuses the *existing*
    ``annual_weather_adjusted_comparison`` (actual vs. weather-predicted kWh) and applies the
    fuel's emission factor to both columns -- no new model.
    """
    factor = _factor(fuel)
    comparison = annual_weather_adjusted_comparison(merged_df, energy_result)
    if comparison.empty:
        return comparison
    result = comparison.copy()
    result["actual_kg_co2e"] = result["actual_kwh"] * factor
    result["weather_predicted_kg_co2e"] = result["weather_predicted_kwh"] * factor
    result["difference_kg_co2e"] = result["difference_kwh"] * factor
    return result


def forecast_emissions(forecast_result: ForecastResult, fuel: EnergyType) -> dict[str, float]:
    """Forecast kg CO2e (best/likely/worst): reuses the existing forecast's P10/P50/P90 kWh
    and applies the fuel's emission factor -- no new forecasting model.
    """
    factor = _factor(fuel)
    return {
        "best_kg_co2e": float(forecast_result.p10.sum()) * factor,
        "likely_kg_co2e": float(forecast_result.p50.sum()) * factor,
        "worst_kg_co2e": float(forecast_result.p90.sum()) * factor,
    }
