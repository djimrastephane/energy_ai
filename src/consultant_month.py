"""Month-comparison questions for the AI Energy Consultant.

Split out of ``src.consultant`` to stay under the ~300-line guideline (the
same reactive-split pattern as ``consultant_router``). Every handler here
answers from the *currently selected* month, fuel, and comparison mode --
carried on ``ConsultantContext`` from the same session state the "How did
this month compare?" page renders, never from stale stored answers (the
audit-F2 lesson). Building a ``MonthlyComparison`` is row-lookup
arithmetic over the already-clean billing frame -- the same weight of work
as ``answer_winter_comparison``'s aggregation, not a new model.

Answer structure follows the design brief: direct answer, breakdown
(gas / electricity / weather / unexplained), meaning, suggested action
only when supported, confidence and limitation.
"""

from __future__ import annotations

from config import SETTINGS
from src.billing import bill_breakdown
from src.consultant import ConsultantAnswer, ConsultantContext
from src.ingestion import EnergyType
from src.monthly_comparison import (
    ComparisonMode,
    MonthlyComparison,
    build_monthly_comparison,
    fuel_contributions,
)
from src.monthly_narrative import build_monthly_comparison_narrative
from src.utils import format_gbp

_COMPARISON_TAB = "How did this month compare?"


def _build(
    ctx: ConsultantContext, fuel: EnergyType | None = None, mode: ComparisonMode | None = None
) -> MonthlyComparison | None:
    fuel = fuel if fuel is not None else ctx.comparison_fuel
    if mode is None:
        # "Long-term trend" isn't a single-month mode; fall back to the default comparison.
        ctx_mode = ctx.comparison_mode
        mode = ctx_mode if ctx_mode != "long_term" else "same_month_last_year"
    df = ctx.fuel_frames.get(fuel)
    if df is None or df.empty:
        return None
    merged = (ctx.fuel_merged or {}).get(fuel) if ctx.weather_enabled else None
    energy = ctx.fuel_energy_results.get(fuel) if ctx.weather_enabled else None
    return build_monthly_comparison(
        df, fuel, mode, ctx.selected_comparison_month, merged, energy
    )


def _contributions(ctx: ConsultantContext, mode: ComparisonMode | None = None):
    return fuel_contributions(
        _build(ctx, "total", mode), _build(ctx, "electricity", mode), _build(ctx, "gas", mode)
    )


def _selection_caption(ctx: ConsultantContext, comparison: MonthlyComparison) -> str:
    return (
        f"Answering for {comparison.selected_month.strftime('%B %Y')}, "
        f"{_fuel_phrase(comparison.fuel)}."
    )


def _fuel_phrase(fuel: EnergyType) -> str:
    return {"total": "combined energy", "electricity": "electricity only", "gas": "gas only"}[fuel]


def _breakdown_evidence(ctx: ConsultantContext, mode: ComparisonMode | None = None) -> list[str]:
    """Gas / electricity / weather / unexplained lines, only for values that exist."""
    evidence: list[str] = []
    contributions = _contributions(ctx, mode)
    if contributions is not None:
        evidence.append(f"Gas: {contributions.gas_change_kwh:+,.0f} kWh")
        evidence.append(f"Electricity: {contributions.electricity_change_kwh:+,.0f} kWh")
    total = _build(ctx, "total", mode)
    if total is not None and total.weather_explained_change_kwh is not None:
        evidence.append(f"Weather-related: {total.weather_explained_change_kwh:+,.0f} kWh")
        evidence.append(f"Remaining unexplained: {total.unexplained_change_kwh:+,.0f} kWh")
    return evidence


def _no_month_answer(question: str) -> ConsultantAnswer:
    return ConsultantAnswer(
        question=question,
        answer="No complete month is available to compare yet.",
        evidence=[],
        confidence=None,
        related_tab=_COMPARISON_TAB,
    )


def answer_month_vs_last_year(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "How did this month compare with last year?"
    comparison = _build(ctx, mode="same_month_last_year")
    if comparison is None:
        return _no_month_answer(question)
    contributions = _contributions(ctx, "same_month_last_year") if comparison.fuel == "total" else None
    narrative = build_monthly_comparison_narrative(comparison, contributions, ctx.weather_enabled)
    parts = [narrative.headline]
    if narrative.weather_effect:
        parts.append(narrative.weather_effect)
    if narrative.is_unusual:
        parts.append(narrative.is_unusual)
    if narrative.action:
        parts.append(narrative.action)
    parts.append(narrative.limitation)
    parts.append(_selection_caption(ctx, comparison))
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=_breakdown_evidence(ctx, "same_month_last_year")
        or [comparison.interpretation],
        confidence=comparison.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_month_vs_typical(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Did I use more energy than usual this month?"
    comparison = _build(ctx, mode="typical_month")
    if comparison is None:
        return _no_month_answer(question)
    parts = [comparison.interpretation]
    if comparison.same_month_low_kwh is not None and comparison.percentage_change is not None:
        month_name = comparison.selected_month.strftime("%B")
        parts.append(
            f"The typical {month_name} range is "
            f"{comparison.same_month_low_kwh:,.0f}-{comparison.same_month_high_kwh:,.0f} kWh "
            f"(median {comparison.same_month_median_kwh:,.0f}); this {month_name} used "
            f"{comparison.current_consumption_kwh:,.0f} kWh."
        )
    parts.append(_selection_caption(ctx, comparison))
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=[
            f"Previous {comparison.selected_month.strftime('%B')}s on record: "
            + (", ".join(str(y) for y in comparison.same_month_years) or "none")
        ],
        confidence=comparison.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_which_fuel_caused_change(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Which fuel caused the change?"
    total = _build(ctx, "total")
    if total is None or total.percentage_change is None:
        return ConsultantAnswer(
            question=question,
            answer="No comparison is available for the selected month, so there is no change to attribute.",
            evidence=[],
            confidence=None,
            related_tab=_COMPARISON_TAB,
        )
    contributions = _contributions(ctx)
    if contributions is None:
        return ConsultantAnswer(
            question=question,
            answer="Fuel attribution needs both Electricity and Gas exports covering the selected "
            "month and its comparison period.",
            evidence=[],
            confidence=None,
            related_tab="What drives my usage?",
        )
    narrative = build_monthly_comparison_narrative(total, contributions, ctx.weather_enabled)
    parts = [narrative.which_fuel or "Neither fuel dominates the change."]
    parts.append(total.interpretation)
    parts.append(_selection_caption(ctx, total))
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=_breakdown_evidence(ctx),
        confidence=total.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_was_it_weather(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Was the difference caused by weather?"
    if not ctx.weather_enabled:
        return ConsultantAnswer(
            question=question,
            answer="Weather data isn't loaded -- turn on 'Weather adjustment' in the sidebar, "
            "then ask again.",
            evidence=[],
            confidence=None,
            related_tab=_COMPARISON_TAB,
        )
    comparison = _build(ctx)
    if comparison is None or comparison.percentage_change is None:
        return _no_month_answer(question)
    weather_kwh = comparison.weather_explained_change_kwh
    unexplained_kwh = comparison.unexplained_change_kwh
    change_kwh = comparison.absolute_change_kwh
    if weather_kwh is None or unexplained_kwh is None or change_kwh is None:
        return ConsultantAnswer(
            question=question,
            answer=f"The weather model doesn't cover both {comparison.selected_month.strftime('%B %Y')} "
            "and its comparison period, so the split isn't available.",
            evidence=[],
            confidence="Low",
            related_tab=_COMPARISON_TAB,
        )
    if comparison.unexplained_change_is_meaningful:
        verdict = "No -- weather does not explain this change."
    elif weather_kwh * change_kwh > 0 and abs(weather_kwh) >= abs(unexplained_kwh):
        verdict = "Mostly yes -- the weather accounts for the larger share of the change."
    elif weather_kwh * change_kwh > 0:
        verdict = "Partly -- weather moved in the same direction, but explains less than half."
    else:
        verdict = "No -- the weather alone would have pushed usage the other way."
    answer = (
        f"{verdict} Of the {change_kwh:+,.0f} kWh change, {weather_kwh:+,.0f} kWh matches what the "
        f"temperature model expected and {unexplained_kwh:+,.0f} kWh does not. "
        + _selection_caption(ctx, comparison)
    )
    return ConsultantAnswer(
        question=question,
        answer=answer,
        evidence=_breakdown_evidence(ctx),
        confidence=comparison.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_best_or_worst_month(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Is this my best month on record?"
    best = _build(ctx, mode="best_month")
    worst = _build(ctx, mode="worst_month")
    if best is None:
        return _no_month_answer(question)
    if best.percentage_change is None:
        return ConsultantAnswer(
            question=question,
            answer=best.interpretation + " " + _selection_caption(ctx, best),
            evidence=[],
            confidence=best.confidence,
            related_tab=_COMPARISON_TAB,
        )
    parts = [best.interpretation]
    if worst is not None and worst.percentage_change is not None and best.percentage_change >= 0:
        parts.append(worst.interpretation)
    parts.append(_selection_caption(ctx, best))
    month_name = best.selected_month.strftime("%B")
    evidence = [
        f"Best recorded {month_name}: {best.comparison_consumption_kwh:,.0f} kWh"
        + (f" ({best.comparison_month.year})" if best.comparison_month is not None else ""),
    ]
    if worst is not None and worst.comparison_consumption_kwh is not None:
        evidence.append(
            f"Worst recorded {month_name}: {worst.comparison_consumption_kwh:,.0f} kWh"
            + (f" ({worst.comparison_month.year})" if worst.comparison_month is not None else "")
        )
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=evidence,
        confidence=best.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_why_expensive(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Why was this month expensive?"
    comparison = _build(ctx)
    if (
        comparison is None
        or comparison.cost_change_gbp is None
        or comparison.current_cost_gbp is None
        or comparison.comparison_cost_gbp is None
    ):
        return ConsultantAnswer(
            question=question,
            answer="No cost comparison is available for the selected month.",
            evidence=[],
            confidence=None,
            related_tab=_COMPARISON_TAB,
        )
    target = (
        comparison.comparison_month.strftime("%B %Y")
        if comparison.comparison_month is not None
        else f"a typical {comparison.selected_month.strftime('%B')}"
    )
    direction = "more" if comparison.cost_change_gbp >= 0 else "less"
    current_bill = bill_breakdown(
        comparison.selected_month, comparison.current_cost_gbp, comparison.fuel, ctx.billing_config
    )
    parts = [
        f"{comparison.selected_month.strftime('%B %Y')} cost "
        f"{format_gbp(abs(comparison.cost_change_gbp))} {direction} in consumption than {target} "
        f"({format_gbp(comparison.current_cost_gbp)} vs {format_gbp(comparison.comparison_cost_gbp)}).",
        f"The estimated full bill is {format_gbp(current_bill.total_bill_gbp)}: "
        f"{format_gbp(current_bill.consumption_cost_gbp)} consumption + "
        f"{format_gbp(current_bill.standing_charge_gbp)} standing charge + "
        f"{format_gbp(current_bill.vat_gbp)} VAT.",
    ]
    evidence = [
        f"{comparison.selected_month.strftime('%B %Y')}: {format_gbp(comparison.current_cost_gbp)} consumption",
        f"{target}: {format_gbp(comparison.comparison_cost_gbp)} consumption",
        f"Standing charge ({current_bill.days_in_period} days, "
        f"{current_bill.billing_period_start.strftime('%d %b')} - "
        f"{current_bill.billing_period_end.strftime('%d %b')}): {format_gbp(current_bill.standing_charge_gbp)}",
        f"VAT at {(ctx.billing_config or SETTINGS.billing).vat_rate:.0%}: {format_gbp(current_bill.vat_gbp)}",
    ]
    if comparison.cost_change_from_usage_gbp is not None and comparison.cost_change_from_rate_gbp is not None:
        usage, rate = comparison.cost_change_from_usage_gbp, comparison.cost_change_from_rate_gbp

        def _signed(value: float) -> str:
            return ("+" if value >= 0 else "-") + format_gbp(abs(value))

        if abs(usage) >= abs(rate):
            parts.append(
                f"Most of that came from usage ({_signed(usage)}); the effective price per kWh "
                f"contributed {_signed(rate)}."
            )
        else:
            parts.append(
                f"Most of that came from the effective price per kWh ({_signed(rate)}), not from "
                f"using more energy ({_signed(usage)}) -- a usage change alone can't explain this bill."
            )
        evidence.append(f"From usage: {_signed(usage)}; from effective rate: {_signed(rate)}")
    parts.append(
        "Standing charges are near-identical for the same calendar month, so they rarely "
        "explain a year-on-year change. " + _selection_caption(ctx, comparison)
    )
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=evidence,
        confidence=comparison.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_should_i_be_concerned(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "Should I be concerned?"
    comparison = _build(ctx)
    if comparison is None or comparison.percentage_change is None:
        return _no_month_answer(question)
    month_anomaly = next((a for a in ctx.anomalies if a.date == comparison.selected_month), None)
    if comparison.unexplained_change_is_meaningful or (
        month_anomaly is not None and len(month_anomaly.methods) >= 2
    ):
        lead = "This month is worth a closer look, though one month is never an emergency."
    elif comparison.judgement in ("Similar", "Better than comparable period"):
        lead = "No."
    else:
        lead = "Probably not."
    parts = [lead, comparison.interpretation]
    if month_anomaly is not None:
        parts.append(
            f"Anomaly detection: flagged by {len(month_anomaly.methods)} of 3 methods "
            f"({', '.join(month_anomaly.methods)})."
        )
    else:
        parts.append("Anomaly detection: not flagged by any of the three methods.")
    parts.append(_selection_caption(ctx, comparison))
    return ConsultantAnswer(
        question=question,
        answer=" ".join(parts),
        evidence=_breakdown_evidence(ctx) or [comparison.interpretation],
        confidence=comparison.confidence,
        related_tab=_COMPARISON_TAB,
    )


def answer_what_next(ctx: ConsultantContext) -> ConsultantAnswer:
    question = "What should I do next?"
    comparison = _build(ctx)
    if comparison is None:
        return _no_month_answer(question)
    contributions = _contributions(ctx) if comparison.fuel == "total" else None
    narrative = build_monthly_comparison_narrative(comparison, contributions, ctx.weather_enabled)
    action = narrative.action or (
        "Nothing specific this month. Keep collecting monthly exports -- every added month "
        "sharpens the typical range and the forecast."
    )
    answer = f"{action} {narrative.limitation} {_selection_caption(ctx, comparison)}"
    return ConsultantAnswer(
        question=question,
        answer=answer,
        evidence=[comparison.interpretation],
        confidence=comparison.confidence,
        related_tab=_COMPARISON_TAB,
    )
