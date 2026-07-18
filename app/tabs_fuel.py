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
from charts_fuel import fuel_comparison_bar, fuel_share_area

from src.fuel import combine_fuel_frames, finding_fuel_mix

_UNAVAILABLE_MESSAGE = (
    "Fuel-level analysis needs both 'Electricity Use' and 'Gas Use' OVO exports. Add them to "
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

    if cross_check_warnings:
        st.warning(
            f"{len(cross_check_warnings)} month(s) where Electricity + Gas doesn't match Total -- "
            "see the Data Quality tab for details."
        )
    else:
        st.success("Electricity + Gas matches Total for every overlapping month.")

    st.plotly_chart(fuel_comparison_bar(combined), width="stretch")
    st.plotly_chart(fuel_share_area(combined), width="stretch")

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
        "to 'Electricity only' or 'Gas only' and check the Weather Adjustment tab."
    )
