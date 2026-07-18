"""Deterministic report assembly: the single source of truth behind the Executive Briefing
and the AI Analyst page.

Pure orchestration -- every number here was already computed by an
existing module (``src.kpis``, ``src.confidence``, ``src.findings``,
``src.recommendations``, ``src.investigation``). This module doesn't
compute any new statistic; it just wires the pieces together once so the
two UI surfaces that consume it can't drift out of sync with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.confidence import (
    ConfidenceRating,
    rate_anomaly,
    rate_data_quality,
    rate_forecast,
    rate_weather_model,
)
from src.decomposition import STLResult
from src.energy_signature import EnergySignatureResult
from src.findings import Finding, generate_findings
from src.forecast_evaluation import ForecastResult
from src.investigation import build_investigation_checklist
from src.kpis import compute_kpis
from src.preprocessing import PreprocessingReport
from src.recommendations import NO_RECOMMENDATIONS_MESSAGE, Recommendation, generate_recommendations
from src.utils import get_logger

logger = get_logger(__name__)

NO_SAVINGS_MESSAGE = "No specific savings opportunity meets the evidence threshold right now."


@dataclass
class AnalystReport:
    executive_summary: str
    overall_assessment: str
    findings: list[Finding]
    biggest_finding: Finding | None
    recommendations: list[Recommendation]
    largest_saving: Recommendation | None
    confidence: dict[str, ConfidenceRating]
    limitations: list[str]
    monitoring_priorities: list[str]


def _overall_assessment(pct_change: float | None, data_quality_rating: ConfidenceRating) -> str:
    if data_quality_rating.level == "Low":
        return f"High uncertainty in this assessment: {data_quality_rating.reason}"
    if pct_change is None:
        return "Not enough history yet for a year-on-year assessment."
    if abs(pct_change) < 5:
        return "Consumption is running at normal, expected levels compared to a year ago."
    if pct_change > 0:
        return f"Consumption is higher than expected, up {pct_change:.0f}% on the prior 12 months."
    return f"Consumption is lower than expected, down {abs(pct_change):.0f}% on the prior 12 months."


def _limitations(
    n_months: int,
    report: PreprocessingReport,
    energy_result: EnergySignatureResult | None,
    anomalies: list[Anomaly],
) -> list[str]:
    items = []
    if n_months < 36:
        items.append(
            f"Only {n_months} months of billing history are available -- statistical power "
            "improves with more history, especially for the weather and forecast models."
        )
    items.append(
        "Forecast cross-validation is 1-step-ahead only; longer-horizon projections aren't "
        "themselves independently validated at that horizon."
    )
    if any(len(a.methods) == 1 for a in anomalies):
        items.append(
            "The STL-residual anomaly test has a measured, elevated false-positive rate at this "
            "sample size -- single-method anomaly flags are leads, not conclusions."
        )
    if energy_result is None:
        items.append("Weather adjustment is off, so findings can't separate weather-driven changes from behavioural ones.")
    if report.missing_months:
        items.append(f"{len(report.missing_months)} month(s) are missing from the billing data.")
    return items


def _monitoring_priorities(
    data_quality_rating: ConfidenceRating,
    weather_rating: ConfidenceRating,
    forecast_rating: ConfidenceRating,
    anomalies: list[Anomaly],
) -> list[str]:
    items = []
    if data_quality_rating.level != "High":
        items.append(f"Data quality: {data_quality_rating.reason}")
    if weather_rating.level != "High":
        items.append(f"Weather model: {weather_rating.reason}")
    if forecast_rating.level != "High":
        items.append(f"Forecast reliability: {forecast_rating.reason}")
    single_method = [a for a in anomalies if len(a.methods) == 1]
    if single_method:
        items.append(
            f"{len(single_method)} month(s) flagged by only one anomaly detection method -- "
            "worth a second look if a similar pattern repeats."
        )
    if not items:
        items.append("No specific concerns identified -- continue routine monitoring.")
    return items


def build_analyst_report(
    clean: pd.DataFrame,
    report: PreprocessingReport,
    stl_result: STLResult | None,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    changepoints: list[ChangePoint],
    anomalies: list[Anomaly],
    forecast_result: ForecastResult | None,
    fuel: str = "total",
) -> AnalystReport:
    """Assemble the full deterministic report from already-computed analysis outputs.

    ``fuel`` (one of ``"total"``/``"electricity"``/``"gas"``) is which fuel
    ``energy_result``/``merged_df`` were actually computed on -- passed through
    so weather-related findings/checklist items name the right fuel instead of
    assuming electricity.
    """
    n_months = len(clean)
    unit_rate = clean["cost_gbp"].sum() / clean["consumption_kwh"].sum()
    history_mean = float(clean["consumption_kwh"].mean())

    kpis = compute_kpis(clean, months=12)
    total_kpi = next((k for k in kpis if k.label == "Total consumption"), None)
    pct_change = total_kpi.pct_change if total_kpi else None

    data_quality_rating = rate_data_quality(report, n_months)
    weather_rating = rate_weather_model(energy_result)
    forecast_rating = rate_forecast(forecast_result, history_mean)

    findings = generate_findings(
        kpis=kpis,
        stl_result=stl_result,
        energy_result=energy_result,
        unit_rate=unit_rate,
        anomalies=anomalies,
        changepoints=changepoints,
        forecast_result=forecast_result,
        history_mean=history_mean,
        fuel=fuel,
    )
    biggest_finding = next((f for f in findings if f.title == "Biggest finding"), None)
    forecast_finding = next((f for f in findings if f.category == "forecast"), None)

    top_anomaly = anomalies[0] if anomalies else None
    top_anomaly_checklist = (
        build_investigation_checklist(top_anomaly.date, clean, merged_df, energy_result, fuel)
        if top_anomaly
        else None
    )
    top_anomaly_rating = rate_anomaly(top_anomaly) if top_anomaly else None

    recommendations = generate_recommendations(
        clean_df=clean,
        merged_df=merged_df,
        energy_result=energy_result,
        forecast_rating=forecast_rating,
        n_months=n_months,
        top_anomaly=top_anomaly,
        top_anomaly_checklist=top_anomaly_checklist,
        top_anomaly_rating=top_anomaly_rating,
    )
    savings_candidates = [r for r in recommendations if r.estimated_saving_gbp is not None]
    largest_saving = max(savings_candidates, key=lambda r: r.estimated_saving_gbp) if savings_candidates else None

    overall_assessment = _overall_assessment(pct_change, data_quality_rating)
    summary_parts = [overall_assessment]
    if biggest_finding:
        summary_parts.append(biggest_finding.narrative)
    summary_parts.append(largest_saving.action if largest_saving else NO_SAVINGS_MESSAGE)
    if forecast_finding:
        summary_parts.append(forecast_finding.narrative)
    if not recommendations:
        summary_parts.append(NO_RECOMMENDATIONS_MESSAGE)

    return AnalystReport(
        executive_summary=" ".join(summary_parts),
        overall_assessment=overall_assessment,
        findings=findings,
        biggest_finding=biggest_finding,
        recommendations=recommendations,
        largest_saving=largest_saving,
        confidence={
            "data_quality": data_quality_rating,
            "weather_model": weather_rating,
            "forecast": forecast_rating,
            "anomaly_detection": top_anomaly_rating
            or ConfidenceRating("High", "No anomalies were detected in this data."),
        },
        limitations=_limitations(n_months, report, energy_result, anomalies),
        monitoring_priorities=_monitoring_priorities(data_quality_rating, weather_rating, forecast_rating, anomalies),
    )
