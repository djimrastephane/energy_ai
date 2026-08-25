"""AI Energy Consultant: a bounded, deterministic Q&A layer over analysis already computed
elsewhere in the app.

This module explains analysis that's already been run -- it never runs new
analysis. Every handler below pulls from objects already computed in
``app/streamlit_app.py``'s ``main()`` (``AnalystReport``, the per-fuel
result dicts, the forecast) and only re-presents them; nothing here fits a
model, detects an anomaly, or computes a new statistic. Without an LLM
(deliberately -- see ``docs/roadmap.md``), "understands any question" isn't
honest, so this supports a fixed set of ~8 canonical questions via
``route_question``'s keyword matching, falling back to ``None`` (not a
guess) when nothing matches well enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import SETTINGS, BillingConfig
from src.anomalies import Anomaly
from src.benchmarking import compare_to_benchmark
from src.carbon import combined_annual_emissions
from src.confidence import Confidence, rate_anomaly
from src.cost_engine import forecast_bill_by_fuel
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult
from src.ingestion import EnergyType
from src.kpis import winter_over_winter_comparison
from src.monthly_comparison import DisplayMode
from src.recommendations import NO_RECOMMENDATIONS_MESSAGE
from src.report import AnalystReport
from src.utils import format_gbp, safe_divide
from src.weather_interpretation import UnusualMonthInterpretation


@dataclass
class ConsultantAnswer:
    question: str
    answer: str
    evidence: list[str]
    confidence: Confidence | None
    related_tab: str | None  # lets the UI show "see full detail on the X tab"


@dataclass
class ConsultantContext:
    """Everything a handler might need, built once in ``streamlit_app.main()`` from objects it
    already computes -- visiting the Consultant tab triggers zero new computation."""

    analyst_report: AnalystReport
    clean: pd.DataFrame  # currently-selected fuel, for "last month" lookups
    fuel_frames: dict[EnergyType, pd.DataFrame]
    fuel_energy_results: dict[EnergyType, EnergySignatureResult]
    anomalies: list[Anomaly]  # selected fuel's anomalies, for the "last month" question
    forecast_12mo: ForecastResult | None
    multi_fuel_forecasts: dict[EnergyType, ForecastResult] | None
    weather_enabled: bool
    # Which fuel the selected-fuel answers describe (e.g. "Gas only") -- displayed by the UI
    # so "your forecast" is never silently a single-fuel number (audit finding F6).
    fuel_label: str = ""
    # Weather-context interpretations for the selected fuel's flagged months, keyed by
    # month_start -- built in main() when weather adjustment is on, None otherwise.
    weather_interpretations: dict[pd.Timestamp, UnusualMonthInterpretation] | None = None
    # The month-comparison selection currently shown on "How did this month compare?" --
    # mirrored from the same session state that page renders, so month-comparison answers
    # (src.consultant_month) always describe what the user is looking at, never stale state.
    selected_comparison_month: pd.Timestamp | None = None
    comparison_mode: DisplayMode = "same_month_last_year"
    comparison_fuel: EnergyType = "total"
    # Per-fuel merged consumption+weather frames (needed to split a month's change into
    # weather-explained and unexplained parts); None when weather is off.
    fuel_merged: dict[EnergyType, pd.DataFrame | None] | None = None
    # The sidebar's Tariff settings (standing charges, VAT) for full-bill estimates;
    # None falls back to the config defaults.
    billing_config: BillingConfig | None = None
    # Whether the sidebar forced a specific forecasting model rather than letting
    # cross-validation pick -- feeds answer_forecast's "auto-selected" vs. "forced via the
    # sidebar" wording, so it stays true once model_choice actually reaches forecast_12mo.
    forecast_model_forced: bool = False


def answer_bill_change(ctx: ConsultantContext) -> ConsultantAnswer:
    report = ctx.analyst_report
    parts = [report.overall_assessment]
    evidence: list[str] = []
    if report.largest_cost_driver:
        parts.append(f"Largest cost driver: {report.largest_cost_driver}.")
        evidence.append(f"Largest cost driver: {report.largest_cost_driver}")
    if report.weather_vs_behavioural_summary:
        parts.append(report.weather_vs_behavioural_summary)
        evidence.append(report.weather_vs_behavioural_summary)
    if report.biggest_finding:
        parts.append(report.biggest_finding.narrative)
        evidence.extend(report.biggest_finding.evidence)
    return ConsultantAnswer(
        question="Why did my bill change?",
        answer=" ".join(parts),
        evidence=evidence or ["Overall assessment computed from a trailing 12-month KPI comparison."],
        confidence=report.confidence["data_quality"].level,
        related_tab="Executive Summary",
    )


def answer_winter_comparison(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "What changed compared with last winter?"
    result = winter_over_winter_comparison(ctx.clean)
    if result is None:
        return ConsultantAnswer(
            question=question,
            answer="Not enough winters of history yet to compare -- need at least two complete "
            "December/January/February seasons.",
            evidence=[],
            confidence=None,
            related_tab=None,
        )
    latest, previous = result["latest_season"], result["previous_season"]
    kwh_change = result["kwh_pct_change"] or 0.0
    direction = "more" if kwh_change >= 0 else "less"
    answer = (
        f"Winter {latest - 1}/{latest} used {abs(kwh_change):.0f}% {direction} energy than winter "
        f"{previous - 1}/{previous} ({result['latest_kwh']:,.0f} kWh vs {result['previous_kwh']:,.0f} kWh), "
        f"costing {format_gbp(result['latest_cost_gbp'])} vs {format_gbp(result['previous_cost_gbp'])}."
    )
    evidence = [
        f"Winter {latest - 1}/{latest}: {result['latest_kwh']:,.0f} kWh, {format_gbp(result['latest_cost_gbp'])}",
        f"Winter {previous - 1}/{previous}: {result['previous_kwh']:,.0f} kWh, {format_gbp(result['previous_cost_gbp'])}",
    ]
    return ConsultantAnswer(question=question, answer=answer, evidence=evidence, confidence="High", related_tab="Comparisons")


def answer_fuel_focus(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Should I focus on reducing gas or electricity?"
    report = ctx.analyst_report
    fuel_focus_rec = next((r for r in report.recommendations if r.title.startswith("Focus on")), None)
    if fuel_focus_rec:
        return ConsultantAnswer(
            question=question,
            answer=fuel_focus_rec.action,
            evidence=fuel_focus_rec.evidence,
            confidence=fuel_focus_rec.confidence,
            related_tab="Comparisons",
        )
    if report.fuel_mix_finding or report.weather_sensitivity_finding:
        parts: list[str] = []
        evidence: list[str] = []
        for finding in (report.fuel_mix_finding, report.weather_sensitivity_finding):
            if finding:
                parts.append(finding.narrative)
                evidence.extend(finding.evidence)
        parts.append(
            "Neither fuel clearly dominates both consumption share and weather sensitivity, so "
            "there's no confident single recommendation to focus on one over the other."
        )
        return ConsultantAnswer(question=question, answer=" ".join(parts), evidence=evidence, confidence="Medium", related_tab="Comparisons")
    return ConsultantAnswer(
        question=question,
        answer="Not enough fuel-level data available yet -- this needs both Electricity and Gas exports.",
        evidence=[],
        confidence=None,
        related_tab="Fuel Breakdown",
    )


def answer_forecast(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "What's my forecast for next year?"
    if ctx.forecast_12mo is None:
        return ConsultantAnswer(
            question=question,
            answer="Not enough history yet for a cross-validated forecast.",
            evidence=[],
            confidence=None,
            related_tab="Forecasting",
        )
    unit_rate = safe_divide(ctx.clean["cost_gbp"].sum(), ctx.clean["consumption_kwh"].sum())
    likely_gbp = float(ctx.forecast_12mo.p50.sum()) * unit_rate
    best_gbp = float(ctx.forecast_12mo.p10.sum()) * unit_rate
    worst_gbp = float(ctx.forecast_12mo.p90.sum()) * unit_rate
    months = len(ctx.forecast_12mo.forecast_dates)
    selection_note = "forced via the sidebar" if ctx.forecast_model_forced else "auto-selected by cross-validation"
    # Whole pounds (bootstrap bands don't support penny precision), and "energy cost" rather
    # than "bill" -- the billing exports carry consumption cost only, no standing charges.
    answer = (
        f"Over the next {months} months, the {ctx.forecast_12mo.model_name} model ({selection_note}) "
        f"predicts a most-likely energy cost of £{likely_gbp:,.0f} (plausible "
        f"range £{best_gbp:,.0f}-£{worst_gbp:,.0f}), excluding standing charges."
    )
    evidence = [f"Model: {ctx.forecast_12mo.model_name}", f"P50 forecast: {ctx.forecast_12mo.p50.sum():,.0f} kWh"]
    if ctx.multi_fuel_forecasts:
        fuel_unit_rates = {
            fuel: safe_divide(df["cost_gbp"].sum(), df["consumption_kwh"].sum()) for fuel, df in ctx.fuel_frames.items()
        }
        comparison = forecast_bill_by_fuel(ctx.multi_fuel_forecasts, fuel_unit_rates)
        if comparison.summed_likely_gbp is not None:
            answer += f" Broken down by fuel: Electricity + Gas together = {format_gbp(comparison.summed_likely_gbp)}."
            evidence.append(f"Electricity + Gas forecast sum: {format_gbp(comparison.summed_likely_gbp)}")
    return ConsultantAnswer(
        question=question, answer=answer, evidence=evidence, confidence=ctx.analyst_report.confidence["forecast"].level, related_tab="Forecasting"
    )


def answer_benchmark(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "How does my usage compare to average?"
    parts: list[str] = []
    evidence: list[str] = []
    confidence: Confidence | None = None
    for fuel in ("electricity", "gas"):
        df = ctx.fuel_frames.get(fuel)
        if df is None or df.empty:
            continue
        annualized_kwh = float(df["consumption_kwh"].sum()) / len(df) * 12
        result = compare_to_benchmark(annualized_kwh, fuel, "uk", n_months=len(df), weather_adjusted=ctx.weather_enabled)
        if result is None:
            continue
        parts.append(
            f"{fuel.capitalize()} usage is {result.band.lower()} for a UK household "
            f"({result.annual_kwh:,.0f} kWh/year vs a {result.reference_kwh:,.0f} kWh/year reference)."
        )
        evidence.append(f"{fuel.capitalize()}: {result.source}")
        confidence = confidence or result.confidence
    if not parts:
        return ConsultantAnswer(
            question=question,
            answer="Not enough fuel-level data to benchmark against published averages.",
            evidence=[],
            confidence=None,
            related_tab="Cost Intelligence",
        )
    return ConsultantAnswer(question=question, answer=" ".join(parts), evidence=evidence, confidence=confidence, related_tab="Cost Intelligence")


def answer_carbon(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "What's my carbon footprint?"
    annual = combined_annual_emissions(ctx.fuel_frames)
    if annual.empty:
        return ConsultantAnswer(
            question=question,
            answer="Not enough complete-year data (or missing Electricity/Gas exports) to estimate a carbon footprint.",
            evidence=[],
            confidence=None,
            related_tab="Carbon",
        )
    latest = annual.iloc[-1]
    answer = (
        f"In {int(latest['year'])} (the most recent complete year), estimated emissions were "
        f"{latest['combined_tonnes_co2e']:.2f} tonnes CO2e -- {latest['electricity_kg_co2e']:,.0f} kg from "
        f"electricity and {latest['gas_kg_co2e']:,.0f} kg from gas."
    )
    evidence = [
        f"Electricity factor: {SETTINGS.carbon.electricity_kg_co2e_per_kwh} kgCO2e/kWh",
        f"Gas factor: {SETTINGS.carbon.gas_kg_co2e_per_kwh} kgCO2e/kWh",
    ]
    return ConsultantAnswer(question=question, answer=answer, evidence=evidence, confidence="Medium", related_tab="Carbon")


def answer_last_month_anomaly(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Was last month's usage normal, or an anomaly?"
    if ctx.clean.empty:
        return ConsultantAnswer(question=question, answer="No data available.", evidence=[], confidence=None, related_tab=None)
    last_month = ctx.clean["month_start"].max()
    label = last_month.strftime("%B %Y")
    # If the latest month is the current calendar month, its figures may still be
    # accumulating (audit finding F1) -- say so rather than judging a partial month
    # as if it were complete.
    in_progress_caveat = ""
    if last_month.to_period("M") == pd.Timestamp.now().to_period("M"):
        in_progress_caveat = (
            f" Note: {label} is the current month, so its figures may be month-to-date "
            "rather than complete -- judge it with caution."
        )
    matching = [a for a in ctx.anomalies if a.date == last_month]
    if not matching:
        return ConsultantAnswer(
            question=question,
            answer=f"{label} looks normal -- it wasn't flagged as unusual by any of the three "
            f"anomaly-detection methods.{in_progress_caveat}",
            evidence=[],
            confidence="Medium",
            related_tab="Unusual Months",
        )
    anomaly = matching[0]
    rating = rate_anomaly(anomaly)
    answer = (
        f"{label} was flagged as a {anomaly.direction}, detected by {len(anomaly.methods)} of 3 methods "
        f"({', '.join(anomaly.methods)}) -- {anomaly.rank_context}.{in_progress_caveat}"
    )
    return ConsultantAnswer(
        question=question,
        answer=answer,
        evidence=[f"Detected by: {', '.join(anomaly.methods)}", anomaly.rank_context],
        confidence=rating.level,
        related_tab="Unusual Months",
    )


def answer_severe_weather(ctx: ConsultantContext) -> ConsultantAnswer:
    """Could snow/wind/rain explain the flagged months? Answer structure: direct answer,
    weather facts (evidence), energy evidence, plausible interpretation, limitation --
    always hedged, never asserting home-working/occupancy as fact."""
    question = "Could severe weather explain my unusual months?"
    if not ctx.weather_enabled:
        return ConsultantAnswer(
            question=question,
            answer="Weather data isn't loaded -- turn on 'Weather adjustment' in the sidebar "
            "first, then ask again. Severe-weather context (snow, wind, heavy rain) comes from "
            "the same Open-Meteo fetch as the temperature model.",
            evidence=[],
            confidence=None,
            related_tab="Weather Impact",
        )
    if not ctx.anomalies:
        return ConsultantAnswer(
            question=question,
            answer="No months are currently flagged as unusual for this fuel, so there's "
            "nothing severe weather would need to explain.",
            evidence=[],
            confidence="Medium",
            related_tab="Unusual Months",
        )
    interpretations = ctx.weather_interpretations or {}
    top = ctx.anomalies[0]  # already sorted by method agreement, then date
    interp = interpretations.get(top.date)
    if interp is None:
        return ConsultantAnswer(
            question=question,
            answer=f"{top.date.strftime('%B %Y')} is flagged as unusual, but weather context "
            "couldn't be computed for it (weather data may not cover that month).",
            evidence=[],
            confidence="Low",
            related_tab="Unusual Months",
        )
    parts = [interp.headline]
    others = [a for a in ctx.anomalies if a.date != top.date]
    if others:
        summaries = []
        for a in others[:3]:
            other = interpretations.get(a.date)
            label = other.weather_facts[0] if other and other.weather_facts else "no weather context"
            summaries.append(f"{a.date.strftime('%B %Y')} ({a.direction}; {label})")
        parts.append("Other flagged months: " + "; ".join(summaries) + ".")
    parts.append(interp.limitation)
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=[*interp.weather_facts, *interp.energy_evidence],
        confidence=interp.confidence,
        related_tab="Unusual Months",
    )


def answer_savings(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Where can I realistically save money?"
    report = ctx.analyst_report
    # Prefer a recommendation with a quantified £ saving; failing that, prefer anything actionable
    # over "Collect more historical data" (real advice, but not itself a savings action) -- only
    # fall back to that one if it's the only recommendation that fired.
    rec = report.largest_saving
    if rec is None:
        actionable = [r for r in report.recommendations if r.title != "Collect more historical data"]
        candidates = actionable or report.recommendations
        rec = candidates[0] if candidates else None
    if rec is None:
        return ConsultantAnswer(question=question, answer=NO_RECOMMENDATIONS_MESSAGE, evidence=[], confidence=None, related_tab="AI Analyst")
    answer = rec.action
    if rec.estimated_saving_gbp is not None:
        answer += f" Estimated saving: {format_gbp(rec.estimated_saving_gbp)}/year."
    return ConsultantAnswer(question=question, answer=answer, evidence=rec.evidence, confidence=rec.confidence, related_tab="AI Analyst")
