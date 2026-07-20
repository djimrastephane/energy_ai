"""Fuel Breakdown tab: electricity vs. gas, when both are available.

Reuses ``src.fuel`` (cross-checking, combining, and the fuel-mix finding) --
no new statistics. The rest of the app becomes fuel-aware not through this
tab but through the sidebar's fuel selector, which swaps which cleaned
DataFrame flows into every other tab; this tab is the one place electricity
and gas are shown side by side.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts_fuel import fuel_comparison_bar, fuel_mix_annual_stacked_bar, fuel_share_area
from src.fuel import combine_fuel_frames, finding_fuel_mix

_UNAVAILABLE_MESSAGE = (
    "Fuel-level analysis needs both 'Electricity Use' and 'Gas Use' exports. Add them to "
    "`data/raw/` (or upload them in the sidebar) alongside the existing 'Total Use' files to "
    "unlock this tab."
)


def render_fuel_breakdown(
    electricity_df: pd.DataFrame, gas_df: pd.DataFrame, cross_check_warnings: list[str]
) -> None:
    if electricity_df.empty or gas_df.empty:
        st.info(_UNAVAILABLE_MESSAGE)
        return

    combined = combine_fuel_frames(electricity_df, gas_df)
    if combined.empty:
        st.info("Electricity and Gas exports don't share any overlapping months.")
        return

    total_elec_kwh = float(combined["electricity_kwh"].sum())
    total_gas_kwh = float(combined["gas_kwh"].sum())
    total_kwh = total_elec_kwh + total_gas_kwh
    c1, c2, c3 = st.columns(3)
    c1.metric("Electricity share", f"{total_elec_kwh / total_kwh * 100:.0f}%" if total_kwh > 0 else "n/a")
    c2.metric("Gas share", f"{total_gas_kwh / total_kwh * 100:.0f}%" if total_kwh > 0 else "n/a")
    c3.metric("Combined energy", f"{total_kwh:,.0f} kWh")

    if cross_check_warnings:
        st.warning(
            f"{len(cross_check_warnings)} month(s) where Electricity + Gas doesn't match Total -- "
            "see the Data Quality tab for details."
        )
    else:
        st.success("Electricity + Gas matches Total for every overlapping month.")

    # Each chart gets its own one-line takeaway rather than stacking three charts behind a
    # single shared paragraph (UX audit finding: this was the app's clearest "collection of
    # charts" moment). Every sentence below is simple arithmetic on ``combined`` -- counts,
    # min/max -- not a new statistic.
    st.plotly_chart(fuel_comparison_bar(combined), width="stretch")
    gas_higher_months = int((combined["gas_kwh"] > combined["electricity_kwh"]).sum())
    elec_higher_months = int((combined["electricity_kwh"] > combined["gas_kwh"]).sum())
    higher_fuel, higher_months = (
        ("Gas", gas_higher_months) if gas_higher_months >= elec_higher_months else ("Electricity", elec_higher_months)
    )
    st.caption(f"{higher_fuel} used more energy than the other fuel in {higher_months} of the last {len(combined)} months.")

    st.plotly_chart(fuel_share_area(combined), width="stretch")
    min_share = combined["electricity_share_pct"].min()
    max_share = combined["electricity_share_pct"].max()
    st.caption(
        f"Electricity's share of combined energy ranges from about {min_share:.0f}% to "
        f"{max_share:.0f}% depending on the month -- the wider that range, the more one fuel "
        "swings with the seasons."
    )

    st.plotly_chart(fuel_mix_annual_stacked_bar(combined), width="stretch")
    st.caption(
        "Each bar splits that year's total energy between electricity (bottom) and gas (top) -- "
        "bar height shows whether total usage is rising or falling year to year; the split within "
        "each bar shows whether the fuel mix is shifting."
    )

    st.divider()
    st.subheader("Fuel mix")
    finding = finding_fuel_mix(combined)
    if finding:
        st.write(finding.narrative)
        st.caption(f"Confidence: {finding.confidence} -- {finding.confidence_reason}")
        with st.expander("Evidence"):
            for e in finding.evidence:
                st.write(f"- {e}")
    else:
        st.info("Not enough overlapping history yet (need at least 6 months of both fuels).")

    st.caption(
        "To see the weather-adjusted heating sensitivity for a single fuel (e.g. how much of "
        "the heating signal is really gas vs. electricity), switch the sidebar's Fuel selector "
        "to 'Electricity only' or 'Gas only' and check the Weather Impact tab."
    )
