"""Cost Intelligence tab (Task 4): per-fuel and combined billing breakdown, forecast bills,
and benchmarking against published UK/Scotland averages (Task 12, folded in here rather than
its own tab -- it's inherently a cost/consumption-scale comparison).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from tabs_phase3 import generate_multi_fuel_forecast_cached

from src.benchmarking import compare_to_benchmark
from src.cost_engine import compute_combined_cost_breakdown, forecast_bill_by_fuel
from src.ingestion import EnergyType


def render_cost_intelligence(fuel_clean_dfs: dict[EnergyType, pd.DataFrame], weather_enabled: bool) -> None:
    st.caption(
        "Standing charges aren't broken out in the OVO billing exports (only total £/month is "
        "available), so they're omitted here rather than estimated -- the effective £/kWh rate "
        "below still reflects your real blended cost."
    )

    breakdown = compute_combined_cost_breakdown(fuel_clean_dfs)
    if breakdown.empty:
        st.info("No cost data available yet.")
        return

    display = breakdown.rename(
        columns={
            "fuel": "Fuel",
            "annual_total_gbp": "Annual total (£)",
            "monthly_avg_gbp": "Monthly average (£)",
            "effective_unit_rate_gbp_per_kwh": "Effective rate (£/kWh)",
            "trend": "YoY trend",
        }
    ).round(2)
    st.dataframe(display, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Benchmark vs. published averages")
    st.caption(
        "Never a single 'Energy Score' -- a categorical band (Below/Average/Above average) "
        "against Ofgem/DESNZ published figures, with a +/-15% tolerance for 'Average'."
    )
    region = st.radio(
        "Compare against", ["uk", "scotland"], format_func=lambda r: "UK average" if r == "uk" else "Scotland average", horizontal=True
    )
    cols = st.columns(2)
    for col, fuel in zip(cols, ("electricity", "gas"), strict=True):
        df = fuel_clean_dfs.get(fuel)
        with col:
            if df is None or df.empty:
                st.info(f"No {fuel} data available.")
                continue
            annualized_kwh = float(df["consumption_kwh"].sum()) / len(df) * 12
            result = compare_to_benchmark(
                annualized_kwh, fuel, region, n_months=len(df), weather_adjusted=weather_enabled
            )
            if result is None:
                st.info(f"No {region} benchmark is available for {fuel} -- see the source note in config.py.")
                continue
            st.metric(
                f"{fuel.capitalize()} vs. {region.upper()}",
                result.band,
                help=(
                    f"Reference: {result.reference_kwh:,.0f} kWh/year ({result.source}). Your annualized "
                    f"usage: {result.annual_kwh:,.0f} kWh/year. Confidence: {result.confidence}."
                ),
            )

    st.divider()
    st.subheader("Forecast bill by fuel")
    if st.button("Compute multi-fuel forecast comparison"):
        st.session_state["multi_fuel_forecast_requested"] = True
        # Restart immediately so main() computes the comparison on a run where the flag is
        # already set -- removes the old cross-tab render-order dependency (audit finding F7).
        st.rerun()

    if not st.session_state.get("multi_fuel_forecast_requested"):
        st.info(
            "Runs the full cross-validated forecast model suite for Electricity, Gas, and Total "
            "(up to 8 models each) -- opt-in rather than automatic, since it's the slowest "
            "computation in the app. Click the button above to run it."
        )
        return

    forecast_results = generate_multi_fuel_forecast_cached(fuel_clean_dfs, horizon=12)
    if not forecast_results:
        st.warning("No fuel had enough history for a cross-validated forecast.")
        return

    unit_rates = {
        fuel: (df["cost_gbp"].sum() / df["consumption_kwh"].sum()) if not df.empty else 0.0
        for fuel, df in fuel_clean_dfs.items()
    }
    comparison = forecast_bill_by_fuel(forecast_results, unit_rates)
    bill_display = comparison.per_fuel.rename(
        columns={
            "fuel": "Fuel",
            "model": "Model",
            "best_gbp": "Best case (£)",
            "likely_gbp": "Most likely (£)",
            "worst_gbp": "Worst plausible (£)",
        }
    ).round(0)
    st.dataframe(bill_display, hide_index=True, width="stretch")
    if comparison.divergence_note:
        st.info(comparison.divergence_note)
