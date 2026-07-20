"""AI Analyst page: the full deterministic report.

Every number here comes from ``src.report.AnalystReport`` and the analysis
objects that feed it -- no LLM, no free-form generation. A later phase may
add an LLM purely to improve the *wording* of these already-computed
conclusions, never to invent new ones (see the module docstring on
``src.report``).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.decomposition import STLResult
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult
from src.ingestion import EnergyType
from src.narrative import monthly_narrative
from src.preprocessing import PreprocessingReport
from src.recommendations import NO_RECOMMENDATIONS_MESSAGE
from src.report import AnalystReport, build_analyst_report
from src.utils import format_gbp

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


def build_report(
    clean: pd.DataFrame,
    report: PreprocessingReport,
    stl_result: STLResult | None,
    merged: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    changepoints: list[ChangePoint],
    anomalies: list[Anomaly],
    forecast_result: ForecastResult | None,
    fuel: str = "total",
    fuel_clean_dfs: Mapping[EnergyType, pd.DataFrame] | None = None,
    fuel_stl_results: Mapping[EnergyType, STLResult | None] | None = None,
    fuel_energy_results: Mapping[EnergyType, EnergySignatureResult | None] | None = None,
    fuel_anomalies: Mapping[EnergyType, list[Anomaly]] | None = None,
) -> AnalystReport:
    """Thin wrapper around ``build_analyst_report``, deliberately uncached.

    Report assembly is pure orchestration over already-computed objects
    (no new statistics) and runs in well under a second -- caching it would
    add the risk and overhead of hashing nested dataclasses that contain
    pandas Series/DataFrames for no real benefit.
    """
    return build_analyst_report(
        clean,
        report,
        stl_result,
        merged,
        energy_result,
        changepoints,
        anomalies,
        forecast_result,
        fuel,
        fuel_clean_dfs,
        fuel_stl_results,
        fuel_energy_results,
        fuel_anomalies,
    )


def _render_finding_list(findings: list) -> None:
    for f in findings:
        with st.expander(f"[{f.confidence}] {f.title}"):
            st.write(f.narrative)
            st.caption(f"Confidence: {f.confidence} -- {f.confidence_reason}")
            st.write("**Evidence:**")
            for e in f.evidence:
                st.write(f"- {e}")


def _render_findings(analyst_report: AnalystReport) -> None:
    st.header("Key Findings")
    _render_finding_list(analyst_report.findings)


def _render_household_profile(analyst_report: AnalystReport) -> None:
    if not (
        analyst_report.fuel_mix_finding
        or analyst_report.weather_sensitivity_finding
        or analyst_report.per_fuel_findings
    ):
        return

    st.header("Household Energy Profile")
    if analyst_report.largest_cost_driver:
        st.write(f"**Largest cost driver:** {analyst_report.largest_cost_driver}")
    if analyst_report.fuel_mix_finding:
        st.subheader("Fuel mix")
        st.write(analyst_report.fuel_mix_finding.narrative)
        st.caption(
            f"Confidence: {analyst_report.fuel_mix_finding.confidence} -- "
            f"{analyst_report.fuel_mix_finding.confidence_reason}"
        )
    if analyst_report.weather_sensitivity_finding:
        st.subheader("Weather sensitivity by fuel")
        st.write(analyst_report.weather_sensitivity_finding.narrative)
        st.caption(
            f"Confidence: {analyst_report.weather_sensitivity_finding.confidence} -- "
            f"{analyst_report.weather_sensitivity_finding.confidence_reason}"
        )

    electricity_findings = analyst_report.per_fuel_findings.get("electricity", [])
    if electricity_findings:
        st.subheader("Electricity Findings")
        _render_finding_list(electricity_findings)

    gas_findings = analyst_report.per_fuel_findings.get("gas", [])
    if gas_findings:
        st.subheader("Gas Findings")
        _render_finding_list(gas_findings)


def _render_recommendations(analyst_report: AnalystReport) -> None:
    st.header("Recommendations")
    if not analyst_report.recommendations:
        st.info(NO_RECOMMENDATIONS_MESSAGE)
        return
    for r in analyst_report.recommendations:
        with st.expander(f"[{r.confidence}] {r.title}"):
            st.write(r.action)
            if r.estimated_saving_gbp is not None:
                st.metric("Estimated annual saving", format_gbp(r.estimated_saving_gbp))
            st.caption(f"Confidence: {r.confidence} -- {r.confidence_reason}")
            st.write(f"**Rationale:** {r.rationale}")
            st.write("**Evidence:**")
            for e in r.evidence:
                st.write(f"- {e}")


def _render_monthly_history(
    clean: pd.DataFrame,
    merged: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    anomalies: list[Anomaly],
    analyst_report: AnalystReport,
) -> None:
    st.header("Monthly Narrative History")
    st.caption("A plain-English narrative for every month, generated the same way as the briefing above.")
    months = sorted(clean["month_start"].unique(), reverse=True)
    if not months:
        return
    forecast_rating = analyst_report.confidence.get("forecast")
    selected = st.selectbox("Month", months, format_func=lambda m: pd.Timestamp(m).strftime("%B %Y"))
    st.write(monthly_narrative(pd.Timestamp(selected), clean, merged, energy_result, anomalies, forecast_rating))

    with st.expander("View all months"):
        rows = [
            {
                "Month": pd.Timestamp(m).strftime("%B %Y"),
                "Narrative": monthly_narrative(pd.Timestamp(m), clean, merged, energy_result, anomalies, forecast_rating),
            }
            for m in months
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_ai_analyst(
    clean: pd.DataFrame,
    merged: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    anomalies: list[Anomaly],
    analyst_report: AnalystReport,
) -> None:
    st.caption(
        "A structured, deterministic report generated entirely from the analysis already computed "
        "elsewhere in this app -- no language model, no free-form generation. Every conclusion is "
        "traceable to a specific finding or recommendation with its own evidence."
    )

    st.header("Executive Summary")
    st.write(analyst_report.executive_summary)

    _render_findings(analyst_report)

    _render_household_profile(analyst_report)

    st.header("Largest Changes")
    st.write(analyst_report.biggest_finding.narrative if analyst_report.biggest_finding else "No standout change identified.")

    _render_recommendations(analyst_report)

    st.header("Confidence")
    for key, rating in analyst_report.confidence.items():
        icon = _CONFIDENCE_ICON[rating.level]
        st.write(f"{icon} **{key.replace('_', ' ').title()}**: {rating.level} -- {rating.reason}")

    st.header("Limitations")
    for item in analyst_report.limitations:
        st.write(f"- {item}")

    st.header("Future Monitoring Priorities")
    for item in analyst_report.monitoring_priorities:
        st.write(f"- {item}")

    _render_monthly_history(clean, merged, energy_result, anomalies, analyst_report)
