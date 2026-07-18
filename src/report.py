"""Deterministic report assembly: the single source of truth behind the Executive Briefing
and the AI Analyst page.

Pure orchestration -- every number here was already computed by an
existing module (``src.kpis``, ``src.confidence``, ``src.findings``,
``src.recommendations``, ``src.investigation``). This module doesn't
compute any new statistic; it just wires the pieces together once so the
two UI surfaces that consume it can't drift out of sync with each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.comparisons import compare_weather_sensitivity, compute_weather_sensitivity_shares
from src.confidence import (
    ConfidenceRating,
    rate_anomaly,
    rate_data_quality,
    rate_forecast,
    rate_weather_model,
)
from src.decomposition import STLResult
from src.energy_signature import EnergySignatureResult, annual_weather_adjusted_comparison
from src.findings import Finding, generate_findings
from src.forecast_evaluation import ForecastResult
from src.fuel import combine_fuel_frames, compute_fuel_shares, finding_fuel_mix
from src.ingestion import EnergyType
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
    fuel_mix_finding: Finding | None = None
    weather_sensitivity_finding: Finding | None = None
    largest_cost_driver: str | None = None
    weather_vs_behavioural_summary: str | None = None
    per_fuel_findings: dict[EnergyType, list[Finding]] = field(default_factory=dict)


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
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame] | None = None,
    fuel_stl_results: dict[EnergyType, STLResult | None] | None = None,
    fuel_energy_results: dict[EnergyType, EnergySignatureResult | None] | None = None,
    fuel_anomalies: dict[EnergyType, list[Anomaly]] | None = None,
) -> AnalystReport:
    """Assemble the full deterministic report from already-computed analysis outputs.

    ``fuel`` (one of ``"total"``/``"electricity"``/``"gas"``) is which fuel
    ``energy_result``/``merged_df`` were actually computed on -- passed through
    so weather-related findings/checklist items name the right fuel instead of
    assuming electricity.

    The ``fuel_*`` dicts (keyed by ``EnergyType``) are optional and carry all
    three fuels' already-computed results (the app loads all three
    regardless of the sidebar's Fuel selector) -- when provided, they power
    the household-level Task 10/11 sections (fuel mix, weather attribution,
    largest cost driver, per-fuel findings) without fitting or detecting
    anything new. Omit them (leave ``None``) to get the same single-fuel
    report this function has always produced.
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

    # Task 10/11 household-level sections -- only computed when the caller supplies all three
    # fuels' already-computed results; each piece reuses an existing function, nothing new is
    # fitted or detected here.
    fuel_shares = None
    fuel_mix_finding: Finding | None = None
    weather_shares = None
    weather_sensitivity_finding: Finding | None = None
    largest_cost_driver: str | None = None
    per_fuel_findings: dict[EnergyType, list[Finding]] = {}

    if fuel_clean_dfs:
        elec_df, gas_df = fuel_clean_dfs.get("electricity"), fuel_clean_dfs.get("gas")
        if elec_df is not None and gas_df is not None and not elec_df.empty and not gas_df.empty:
            combined_fuel_df = combine_fuel_frames(elec_df, gas_df)
            fuel_shares = compute_fuel_shares(combined_fuel_df)
            fuel_mix_finding = finding_fuel_mix(combined_fuel_df)
            if fuel_shares is not None:
                cost_dominant = (
                    "gas"
                    if fuel_shares.gas_share_cost_pct >= fuel_shares.electricity_share_cost_pct
                    else "electricity"
                )
                cost_share = (
                    fuel_shares.gas_share_cost_pct
                    if cost_dominant == "gas"
                    else fuel_shares.electricity_share_cost_pct
                )
                largest_cost_driver = f"{cost_dominant.capitalize()} ({cost_share:.0f}% of combined cost)"

        if fuel_energy_results:
            weather_shares = compute_weather_sensitivity_shares(fuel_energy_results)
            weather_sensitivity_finding = compare_weather_sensitivity(fuel_energy_results)

        for pf in ("electricity", "gas"):
            pf_clean = fuel_clean_dfs.get(pf)
            if pf_clean is None or pf_clean.empty:
                continue
            per_fuel_findings[pf] = generate_findings(
                kpis=compute_kpis(pf_clean, months=12),
                stl_result=(fuel_stl_results or {}).get(pf),
                energy_result=(fuel_energy_results or {}).get(pf),
                unit_rate=pf_clean["cost_gbp"].sum() / pf_clean["consumption_kwh"].sum(),
                anomalies=(fuel_anomalies or {}).get(pf, []),
                changepoints=[],
                forecast_result=None,
                history_mean=float(pf_clean["consumption_kwh"].mean()),
                fuel=pf,
            )

    weather_vs_behavioural_summary: str | None = None
    if merged_df is not None and energy_result is not None:
        annual_comparison = annual_weather_adjusted_comparison(merged_df, energy_result)
        if not annual_comparison.empty:
            latest = annual_comparison.iloc[-1]
            weather_vs_behavioural_summary = (
                f"{int(latest['year'])}: actual {latest['actual_kwh']:,.0f} kWh vs. weather-adjusted "
                f"{latest['weather_predicted_kwh']:,.0f} kWh ({latest['difference_kwh']:+,.0f} kWh) -- "
                f"{latest['interpretation']}"
            )

    recommendations = generate_recommendations(
        clean_df=clean,
        merged_df=merged_df,
        energy_result=energy_result,
        forecast_rating=forecast_rating,
        n_months=n_months,
        top_anomaly=top_anomaly,
        top_anomaly_checklist=top_anomaly_checklist,
        top_anomaly_rating=top_anomaly_rating,
        fuel_shares=fuel_shares,
        weather_shares=weather_shares,
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
        fuel_mix_finding=fuel_mix_finding,
        weather_sensitivity_finding=weather_sensitivity_finding,
        largest_cost_driver=largest_cost_driver,
        weather_vs_behavioural_summary=weather_vs_behavioural_summary,
        per_fuel_findings=per_fuel_findings,
    )
