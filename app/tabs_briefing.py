"""Executive Briefing tab: an evidence-based summary, not a KPI dump.

Every sentence here is pulled from ``src.report.AnalystReport`` -- the same
object the AI Analyst page (``app/tabs_analyst.py``) renders in full, so
the two surfaces can never contradict each other.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.forecast_evaluation import ForecastResult
from src.kpis import KPIComparison, compute_kpis
from src.report import NO_SAVINGS_MESSAGE, AnalystReport
from src.utils import format_gbp

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


def _fmt(value: float, unit: str) -> str:
    if unit == "£":
        return format_gbp(value)
    if unit == "£/kWh":
        return f"£{value:.3f}/kWh"
    return f"{value:,.0f} {unit}".strip()


def _render_kpi_strip(kpis: list[KPIComparison]) -> None:
    headline = [k for k in kpis if k.label in ("Total consumption", "Total cost")]
    if not headline:
        return
    cols = st.columns(len(headline))
    for col, kpi in zip(cols, headline, strict=True):
        with col:
            delta = f"{kpi.pct_change:+.1f}%" if kpi.pct_change is not None else None
            st.metric(f"{kpi.label} (trailing 12mo)", _fmt(kpi.current, kpi.unit), delta)


def _confidence_badge(col, label: str, rating) -> None:
    with col:
        icon = _CONFIDENCE_ICON[rating.level]
        st.metric(label, f"{icon} {rating.level}", help=rating.reason)


def render_executive_briefing(
    clean: pd.DataFrame,
    analyst_report: AnalystReport,
    forecast_12mo: ForecastResult | None,
    forecast_error: str | None,
) -> None:
    st.caption(
        "An evidence-based briefing, not a statistics dump -- every statement below traces to a "
        "specific number computed elsewhere in the app. See the AI Analyst tab for the full report "
        "with evidence and methodology, or the individual analysis tabs for the underlying detail."
    )

    kpis = compute_kpis(clean, months=12)
    _render_kpi_strip(kpis)

    st.divider()
    st.subheader("Overall assessment")
    st.write(analyst_report.overall_assessment)

    st.subheader("Biggest finding")
    if analyst_report.biggest_finding:
        st.write(analyst_report.biggest_finding.narrative)
        st.caption(
            f"Confidence: {analyst_report.biggest_finding.confidence} -- "
            f"{analyst_report.biggest_finding.confidence_reason}"
        )
    else:
        st.write("No standout finding identified in the current data.")

    st.subheader("Largest saving opportunity")
    if analyst_report.largest_saving:
        st.write(analyst_report.largest_saving.action)
        st.caption(
            f"Confidence: {analyst_report.largest_saving.confidence} -- "
            f"{analyst_report.largest_saving.confidence_reason}"
        )
    else:
        st.write(NO_SAVINGS_MESSAGE)

    st.divider()
    st.subheader("Forecast: expected annual bill")
    if forecast_error:
        st.info(f"Forecast unavailable: {forecast_error}")
    elif forecast_12mo is not None:
        unit_rate = clean["cost_gbp"].sum() / clean["consumption_kwh"].sum()
        best = float(forecast_12mo.p10.sum())  # lower kWh = lower bill = best case for the wallet
        likely = float(forecast_12mo.p50.sum())
        worst = float(forecast_12mo.p90.sum())  # higher kWh = higher bill = worst plausible case
        tooltip = "From resampling the model's own past forecast errors -- a plausible range, not a guarantee."
        c1, c2, c3 = st.columns(3)
        c1.metric("Best case", format_gbp(best * unit_rate), help=tooltip)
        c2.metric("Most likely", format_gbp(likely * unit_rate), help=tooltip)
        c3.metric("Worst plausible", format_gbp(worst * unit_rate), help=tooltip)
        st.caption(
            f"Based on the {forecast_12mo.model_name} model (auto-selected by cross-validation). "
            "See the Forecasting tab for the full model comparison and chart."
        )
    else:
        st.info("Not enough history for a cross-validated forecast yet.")

    if (
        analyst_report.fuel_mix_finding
        or analyst_report.largest_cost_driver
        or analyst_report.weather_vs_behavioural_summary
    ):
        st.divider()
        st.subheader("Household energy profile")
        if analyst_report.largest_cost_driver:
            st.write(f"**Largest cost driver:** {analyst_report.largest_cost_driver}")
        if analyst_report.fuel_mix_finding:
            st.write(f"**Fuel mix:** {analyst_report.fuel_mix_finding.narrative}")
        if analyst_report.weather_vs_behavioural_summary:
            st.write(f"**Weather vs. behavioural impact:** {analyst_report.weather_vs_behavioural_summary}")
        st.caption("See the Comparisons tab for the full electricity-vs-gas breakdown.")

    st.divider()
    st.subheader("Confidence")
    c1, c2, c3, c4 = st.columns(4)
    _confidence_badge(c1, "Data quality", analyst_report.confidence["data_quality"])
    _confidence_badge(c2, "Forecast", analyst_report.confidence["forecast"])
    _confidence_badge(c3, "Weather model", analyst_report.confidence["weather_model"])
    _confidence_badge(c4, "Anomaly detection", analyst_report.confidence["anomaly_detection"])
    st.caption(
        "See the AI Analyst tab for the full report, including every finding's evidence and "
        "every recommendation's rationale."
    )
