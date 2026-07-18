"""Carbon tab (Task 8): estimated CO2e emissions from electricity and gas.

Requires both Electricity and Gas data (same gate as the Fuel Breakdown
tab) -- "Total" alone can't be split into a blended emission factor
honestly, so this tab says so plainly rather than guessing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SETTINGS
from src.carbon import combined_annual_emissions, weather_adjusted_annual_emissions
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType

_UNAVAILABLE_MESSAGE = (
    "Carbon estimates need both 'Electricity Use' and 'Gas Use' exports -- the electricity/gas "
    "split can't be recovered from a 'Total Use'-only file, so a blended emission factor can't "
    "be honestly computed. Add both fuel exports to unlock this tab."
)


def render_carbon(
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame],
    fuel_merged: dict[EnergyType, pd.DataFrame],
    fuel_energy_results: dict[EnergyType, EnergySignatureResult],
    weather_enabled: bool,
) -> None:
    elec_df = fuel_clean_dfs.get("electricity", pd.DataFrame())
    gas_df = fuel_clean_dfs.get("gas", pd.DataFrame())
    if elec_df.empty or gas_df.empty:
        st.info(_UNAVAILABLE_MESSAGE)
        return

    st.caption(
        f"Electricity: {SETTINGS.carbon.electricity_kg_co2e_per_kwh} kgCO2e/kWh "
        "(DESNZ/DEFRA GHG Conversion Factors, 2024 edition, location-based grid electricity -- "
        "declining year over year with grid decarbonisation, so treat as a periodically-"
        f"refreshed estimate). Gas: {SETTINGS.carbon.gas_kg_co2e_per_kwh} kgCO2e/kWh "
        "(DEFRA GHG Conversion Factors, natural gas combustion, stable across recent editions)."
    )

    annual = combined_annual_emissions(fuel_clean_dfs)
    if annual.empty:
        st.info("Not enough complete calendar years yet for an annual emissions estimate.")
        return

    display = annual.rename(
        columns={
            "year": "Year",
            "electricity_kg_co2e": "Electricity (kg CO2e)",
            "gas_kg_co2e": "Gas (kg CO2e)",
            "combined_kg_co2e": "Combined (kg CO2e)",
            "combined_tonnes_co2e": "Combined (tonnes CO2e)",
        }
    ).round(1)
    st.dataframe(display, hide_index=True, width="stretch")

    latest = annual.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Electricity emissions", f"{latest['electricity_kg_co2e']:,.0f} kg CO2e")
    c2.metric("Gas emissions", f"{latest['gas_kg_co2e']:,.0f} kg CO2e")
    c3.metric("Combined emissions", f"{latest['combined_tonnes_co2e']:,.2f} tonnes CO2e")

    if not weather_enabled:
        st.info("Turn on 'Weather adjustment' in the sidebar to see weather-adjusted emissions.")
        return

    st.divider()
    st.subheader("Weather-adjusted emissions")
    st.caption("Was the change in emissions weather, or behaviour? Reuses the existing weather-adjusted comparison.")
    for fuel in ("electricity", "gas"):
        merged = fuel_merged.get(fuel)
        result = fuel_energy_results.get(fuel)
        if merged is None or result is None:
            continue
        emissions = weather_adjusted_annual_emissions(merged, result, fuel)
        if emissions.empty:
            continue
        st.write(f"**{fuel.capitalize()}**")
        display = emissions.rename(
            columns={
                "year": "Year",
                "actual_kg_co2e": "Actual (kg CO2e)",
                "weather_predicted_kg_co2e": "Weather-predicted (kg CO2e)",
                "difference_kg_co2e": "Difference (kg CO2e)",
                "interpretation": "Interpretation",
            }
        )[["Year", "Actual (kg CO2e)", "Weather-predicted (kg CO2e)", "Difference (kg CO2e)", "Interpretation"]].round(1)
        st.dataframe(display, hide_index=True, width="stretch")
