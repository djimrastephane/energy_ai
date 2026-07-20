"""Cost Intelligence tab (Task 4): per-fuel and combined billing breakdown, forecast bills,
and benchmarking against published UK/Scotland averages (Task 12, folded in here rather than
its own tab -- it's inherently a cost/consumption-scale comparison).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from app.tabs_forecast import generate_multi_fuel_forecast_cached
from config import SETTINGS, BillingConfig
from src.benchmarking import compare_to_benchmark
from src.billing import bill_breakdown_frame
from src.cost_engine import compute_combined_cost_breakdown, forecast_bill_by_fuel
from src.ingestion import EnergyType

_BILL_FUEL_LABELS: dict[EnergyType, str] = {
    "total": "Combined (both standing charges)",
    "electricity": "Electricity",
    "gas": "Gas",
}


def _render_bill_breakdown(
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame], billing_config: BillingConfig
) -> None:
    """Per-month consumption + standing + VAT = total bill, per fuel, from the user-supplied
    tariff facts in ``config.BillingConfig`` -- estimated and labelled, never silently blended."""
    st.subheader("Bill breakdown: consumption + standing charge + VAT")
    st.caption(
        "Bills run 6th to 5th (the April bill covers 6 Apr - 5 May). Standing charges "
        f"({billing_config.electricity_standing_gbp_per_day * 100:.2f}p/day electricity, "
        f"{billing_config.gas_standing_gbp_per_day * 100:.2f}p/day gas, ex VAT) and "
        f"{billing_config.vat_rate:.0%} VAT come from the sidebar's Tariff section, not from the "
        "exports; the exported cost is treated as the consumption charge excluding VAT "
        "(config.BillingConfig documents that assumption)."
    )
    fuel_options: tuple[EnergyType, ...] = ("total", "electricity", "gas")
    available = [f for f in fuel_options if not fuel_clean_dfs.get(f, pd.DataFrame()).empty]
    if not available:
        return
    fuel = st.radio(
        "Fuel for bill breakdown",
        available,
        format_func=lambda f: _BILL_FUEL_LABELS[f],
        horizontal=True,
        key="bill_breakdown_fuel",
    )
    frame = bill_breakdown_frame(fuel_clean_dfs[fuel], fuel, billing_config)
    display = frame.assign(month=frame["month_start"].dt.strftime("%b %Y")).rename(
        columns={
            "month": "Month",
            "billing_period": "Billing period",
            "consumption_cost_gbp": "Consumption (£)",
            "standing_charge_gbp": "Standing (£)",
            "vat_gbp": f"VAT {billing_config.vat_rate:.0%} (£)",
            "total_bill_gbp": "Total bill (£)",
        }
    )[
        ["Month", "Billing period", "Consumption (£)", "Standing (£)",
         f"VAT {billing_config.vat_rate:.0%} (£)", "Total bill (£)"]
    ].round(2)
    st.dataframe(display.iloc[::-1], hide_index=True, width="stretch")


def render_cost_intelligence(
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame],
    weather_enabled: bool,
    billing_config: BillingConfig | None = None,
) -> None:
    billing_config = billing_config or SETTINGS.billing
    st.caption(
        "The OVO exports carry consumption cost only; standing charges and VAT below are "
        "estimated from your supplied tariff details (see the bill breakdown section)."
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
    _render_bill_breakdown(fuel_clean_dfs, billing_config)

    st.divider()
    st.subheader("Benchmark vs. published averages")
    st.caption(
        "Never a single 'Energy Score' -- a categorical band (Below/Average/Above average) "
        "against Ofgem/DESNZ published figures, with a +/-15% tolerance for 'Average'."
    )
    regions: list[Literal["uk", "scotland"]] = ["uk", "scotland"]
    region = st.radio(
        "Compare against", regions, format_func=lambda r: "UK average" if r == "uk" else "Scotland average", horizontal=True
    )
    cols = st.columns(2)
    benchmark_fuels: tuple[EnergyType, ...] = ("electricity", "gas")
    for col, fuel in zip(cols, benchmark_fuels, strict=True):
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
            "best_gbp": "Lower estimate (£)",
            "likely_gbp": "Expected (£)",
            "worst_gbp": "Upper estimate (£)",
        }
    ).round(0)
    st.dataframe(bill_display, hide_index=True, width="stretch")
    if comparison.divergence_note:
        st.info(comparison.divergence_note)
