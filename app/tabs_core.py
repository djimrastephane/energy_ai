"""Phase 1 tab renderers: Consumption Analysis, Statistical Analysis, Data Quality.

Split out of ``streamlit_app.py`` to keep that module under the project's
preferred ~300-line-per-file guideline. The old Executive Summary tab was
redesigned into an evidence-based briefing -- see ``app/tabs_briefing.py``.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import streamlit as st

from app.charts import (
    annual_totals_bar,
    monthly_consumption_bar,
    render_chart,
    rolling_average_line,
    year_over_year_overlay,
)
from src.preprocessing import PreprocessingReport
from src.statistics import describe, interpret


def render_consumption_analysis(period_df: pd.DataFrame) -> None:
    # Each chart gets its own one-line takeaway rather than stacking four charts behind a
    # single shared caption (UX audit finding: this tab, alongside Fuel breakdown, was the
    # app's clearest "collection of charts" moment). The one computed fact below (highest/
    # lowest month) is a plain min/max over data already loaded, not a new statistic.
    render_chart(monthly_consumption_bar(period_df))
    highest = period_df.loc[period_df["consumption_kwh"].idxmax()]
    lowest = period_df.loc[period_df["consumption_kwh"].idxmin()]
    st.caption(
        f"Highest month in the selected period: {highest['month_start']:%B %Y} "
        f"({highest['consumption_kwh']:,.0f} kWh). Lowest: {lowest['month_start']:%B %Y} "
        f"({lowest['consumption_kwh']:,.0f} kWh)."
    )

    col1, col2 = st.columns(2)
    with col1:
        render_chart(annual_totals_bar(period_df))
    with col2:
        render_chart(year_over_year_overlay(period_df))
    st.caption(
        "Annual totals on the left, each year's monthly shape overlaid on the right -- lines "
        "sitting close together mean a stable year-to-year pattern; lines drifting apart mean "
        "usage is trending up or down."
    )

    render_chart(rolling_average_line(period_df, window=3))
    st.caption(
        "A 3-month rolling average smooths out month-to-month noise so the underlying trend is "
        "easier to see through the normal seasonal swings."
    )

    st.caption(
        "Calendar heatmaps and consumption duration curves require daily or sub-daily meter "
        "readings. The source data here is monthly, so those chart types are omitted rather "
        "than approximated from monthly totals."
    )


def _summary_to_frame(summary) -> pd.DataFrame:
    rows = [(k.replace("_", " ").title(), v) for k, v in dataclasses.asdict(summary).items()]
    df = pd.DataFrame(rows, columns=["Statistic", "Value"])
    df["Value"] = df["Value"].apply(lambda v: round(v, 3) if isinstance(v, float) else v)
    return df


def render_statistical_analysis(period_df: pd.DataFrame) -> None:
    st.subheader("Consumption (kWh)")
    kwh_summary = describe(period_df["consumption_kwh"])
    st.write(interpret(kwh_summary, label="monthly consumption (kWh)"))
    st.dataframe(_summary_to_frame(kwh_summary), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Cost (£)")
    cost_summary = describe(period_df["cost_gbp"])
    st.write(interpret(cost_summary, label="monthly cost (£)"))
    st.dataframe(_summary_to_frame(cost_summary), hide_index=True, width="stretch")


def render_data_quality(
    report: PreprocessingReport, fuel_cross_check_warnings: list[str] | None = None
) -> None:
    st.subheader("Ingestion & validation report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Files loaded", report.n_files_loaded)
    c2.metric("Months of data", report.n_months)
    date_range_str = (
        f"{report.date_range[0]:%b %Y} - {report.date_range[1]:%b %Y}"
        if report.date_range
        else "n/a"
    )
    c3.metric("Date range", date_range_str)
    # Filenames are user-supplied (uploads) -- backtick-wrap so markdown in a name
    # renders literally instead of altering the page (audit finding F9).
    st.write("**Source files:** " + ", ".join(f"`{name}`" for name in report.source_files))

    st.divider()
    if report.missing_months:
        st.error(
            f"{len(report.missing_months)} missing month(s): "
            + ", ".join(m.strftime("%b %Y") for m in report.missing_months)
        )
    else:
        st.success("No missing months detected within the available date range.")

    if report.duplicates_removed:
        st.warning(f"Removed {report.duplicates_removed} exact duplicate row(s).")
    else:
        st.success("No exact duplicate rows found.")

    if report.conflicts:
        st.warning("Conflicting values across source files were resolved:")
        for c in report.conflicts:
            st.write(f"- {c}")
    else:
        st.success("No conflicting month values across source files.")

    if report.outlier_warnings:
        st.warning("Plausibility warnings (values kept, flagged for review):")
        for w in report.outlier_warnings:
            st.write(f"- {w}")
    else:
        st.success("No implausible cost/consumption values detected.")

    if fuel_cross_check_warnings is not None:
        st.divider()
        st.write("**Electricity + Gas vs. Total cross-check**")
        if fuel_cross_check_warnings:
            st.warning("Mismatches found:")
            for w in fuel_cross_check_warnings:
                st.write(f"- {w}")
        else:
            st.success("Electricity + Gas matches Total for every overlapping month.")
