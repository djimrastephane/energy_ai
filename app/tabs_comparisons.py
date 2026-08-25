"""Comparisons tab (Tasks 2/5/6/7): Electricity vs. Gas vs. Total, side by side.

Every number here is already computed elsewhere in the app (once per fuel,
via ``app.multi_fuel.compute_all_fuel_analysis``) -- this tab only compares
them via ``src.comparisons``/``src.cross_fuel_anomalies``, reusing the exact
same functions, never refitting or re-detecting anything.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts import render_chart
from app.charts_comparisons import annual_comparison_bar
from src.anomalies import Anomaly
from src.comparisons import (
    compare_annual_totals,
    compare_monthly,
    compare_seasonality,
    compare_weather_adjusted_annual,
    compare_weather_sensitivity,
)
from src.cross_fuel_anomalies import cross_fuel_anomaly_insights
from src.decomposition import STLResult
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult
from src.ingestion import EnergyType


def render_comparisons(
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame],
    fuel_stl_results: dict[EnergyType, STLResult | None],
    fuel_anomalies: dict[EnergyType, list[Anomaly]],
    fuel_merged: dict[EnergyType, pd.DataFrame],
    fuel_energy_results: dict[EnergyType, EnergySignatureResult],
    weather_enabled: bool,
    multi_fuel_forecasts: dict[EnergyType, ForecastResult] | None,
) -> None:
    st.caption(
        "Electricity, Gas, and Total side by side -- every number here is already computed "
        "elsewhere in the app (once per fuel); this tab only compares them."
    )

    st.subheader("Annual totals")
    annual = compare_annual_totals(fuel_clean_dfs)
    if annual.empty:
        st.info("Not enough complete calendar years yet for an annual comparison.")
    else:
        render_chart(annual_comparison_bar(annual))
        with st.expander("View annual table"):
            st.dataframe(annual, hide_index=True, width="stretch")

    st.subheader("Monthly")
    monthly = compare_monthly(fuel_clean_dfs)
    if not monthly.empty:
        with st.expander("View monthly table"):
            st.dataframe(monthly, hide_index=True, width="stretch")

    st.subheader("Seasonality")
    seasonality = compare_seasonality(fuel_stl_results)
    if seasonality.empty:
        st.info("Not enough contiguous history yet for a seasonality comparison.")
    else:
        st.dataframe(seasonality, hide_index=True, width="stretch")
        st.caption("Seasonal/trend strength from STL decomposition, 0-100%, run independently per fuel.")

    st.divider()
    st.subheader("Weather sensitivity")
    if not weather_enabled:
        st.info("Turn on 'Weather adjustment' in the sidebar to see which fuel responds more to weather.")
    else:
        finding = compare_weather_sensitivity(fuel_energy_results)
        if finding:
            st.write(finding.narrative)
            st.caption(f"Confidence: {finding.confidence} -- {finding.confidence_reason}")
        else:
            st.info("Not enough significant weather sensitivity in either fuel to attribute heating demand.")

        st.subheader("Weather-adjusted annual comparison")
        fuel_merged_and_results = {
            fuel: (fuel_merged[fuel], fuel_energy_results[fuel])
            for fuel in fuel_merged
            if fuel in fuel_energy_results
        }
        weather_adjusted = compare_weather_adjusted_annual(fuel_merged_and_results)
        if weather_adjusted.empty:
            st.info("No complete calendar year has full weather coverage yet for this comparison.")
        else:
            display = weather_adjusted.rename(
                columns={
                    "fuel": "Fuel",
                    "year": "Year",
                    "actual_kwh": "Actual (kWh)",
                    "weather_predicted_kwh": "Weather-predicted (kWh)",
                    "difference_kwh": "Difference (kWh)",
                    "interpretation": "Interpretation",
                }
            ).round(0)
            st.dataframe(display, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Cross-fuel anomaly attribution")
    st.caption(
        "Cross-references each fuel's independently-detected anomalies to say which fuel is "
        "responsible for a flagged month -- e.g. an electricity-only spike suggests an appliance "
        "or occupancy change, a gas-only spike suggests a heating event, and both moving together "
        "suggests weather (unless the weather-adjusted model says otherwise)."
    )
    insights = cross_fuel_anomaly_insights(fuel_anomalies, fuel_clean_dfs, fuel_energy_results if weather_enabled else None)
    if not insights:
        st.success("No anomalies flagged in any fuel -- consumption looks stable once trend/season is removed.")
    else:
        for insight in insights:
            with st.expander(f"[{insight.confidence}] {insight.narrative}"):
                st.write("**Evidence:**")
                for e in insight.evidence:
                    st.write(f"- {e}")

    st.divider()
    st.subheader("Forecast comparison")
    if multi_fuel_forecasts:
        rows = [
            {"Fuel": fuel.capitalize(), "Model": result.model_name, "Likely (kWh)": f"{result.p50.sum():,.0f}"}
            for fuel, result in multi_fuel_forecasts.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("£ bill comparison for these same forecasts is on the Cost Intelligence tab.")
    else:
        st.info(
            "Click 'Compute multi-fuel forecast comparison' on the Cost Intelligence tab to see "
            "each fuel's forecast side by side here too (shared across both tabs, computed once)."
        )
