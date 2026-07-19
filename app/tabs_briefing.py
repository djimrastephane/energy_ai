"""Home tab: the month-first executive summary.

Leads with the question a homeowner actually asks -- "how did this month
compare with the same month last year?" -- using the latest *complete*
month (never the in-progress one), then the existing evidence-based
briefing (assessment, biggest finding, saving opportunity, forecast)
below. Every number is pulled from the shared ``MonthContext`` built once
in ``main()`` and from ``src.report.AnalystReport`` -- the same objects
the comparison page, AI Analyst, and Consultant read, so the surfaces
can never contradict each other.

The trailing-12-month KPI strip that used to open this page lives under
Long-term trends now: a 12-month window answers "how is the year going",
not "how did this month compare", and it kept pulling attention first.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from tabs_month import MonthContext

from config import SETTINGS
from src.billing import standing_charge_for_months
from src.forecast_evaluation import ForecastResult
from src.report import NO_SAVINGS_MESSAGE, AnalystReport

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


def _confidence_badge(col, label: str, rating) -> None:
    with col:
        icon = _CONFIDENCE_ICON[rating.level]
        st.metric(label, f"{icon} {rating.level}", help=rating.reason)


def _contribution_caption(change_kwh: float | None, share_pct: float | None) -> str | None:
    if change_kwh is None:
        return None
    if share_pct is None:
        return f"{change_kwh:+,.0f} kWh of the change"
    if share_pct < 0:
        return f"{change_kwh:+,.0f} kWh -- moved against the overall change"
    return f"{change_kwh:+,.0f} kWh ({share_pct:.0f}% of the change)"


def _render_month_headline(ctx: MonthContext) -> None:
    """The three compact cards: Total energy, Gas, Electricity -- current month vs same
    month last year, judgement in words (an increase is never automatically red)."""
    st.subheader("This month compared with last year")
    if ctx.partial_note:
        st.info(ctx.partial_note)

    total = ctx.comparisons.get("total")
    if total is None or ctx.selected_month is None:
        st.info("Not enough data yet for a month-on-month comparison.")
        return

    narrative = ctx.narrative
    if narrative is not None:
        st.markdown(f"##### {narrative.headline}")
        st.caption(f"{total.judgement} · Confidence: {total.confidence}")

    month_label = ctx.selected_month.strftime("%b %Y")
    comparison_label = (
        total.comparison_month.strftime("%b %Y") if total.comparison_month is not None else "last year"
    )
    elec = ctx.comparisons.get("electricity")
    gas = ctx.comparisons.get("gas")
    contributions = ctx.contributions

    cols = st.columns(3)
    cards = [
        ("Total energy", total, None),
        ("Gas", gas, contributions.gas_change_kwh if contributions else None),
        ("Electricity", elec, contributions.electricity_change_kwh if contributions else None),
    ]
    shares = {
        "Gas": contributions.gas_share_pct if contributions else None,
        "Electricity": contributions.electricity_share_pct if contributions else None,
    }
    for col, (label, comparison, contribution_kwh) in zip(cols, cards, strict=True):
        with col:
            if comparison is None:
                st.metric(label, "no data")
                continue
            delta = (
                f"{comparison.percentage_change:+.0f}% vs {comparison_label}"
                if comparison.percentage_change is not None
                else None
            )
            st.metric(
                f"{label} -- {month_label}",
                f"{comparison.current_consumption_kwh:,.0f} kWh",
                delta,
                delta_color="off",
                help=(
                    f"{comparison_label}: {comparison.comparison_consumption_kwh:,.0f} kWh"
                    if comparison.comparison_consumption_kwh is not None
                    else None
                ),
            )
            if label != "Total energy":
                caption = _contribution_caption(contribution_kwh, shares.get(label))
                if caption:
                    st.caption(caption)

    # What explains the change? -- only when the weather model actually covers both months.
    if total.weather_explained_change_kwh is not None and total.absolute_change_kwh is not None:
        st.markdown("**What explains the change?**")
        st.markdown(
            f"- Weather-related: {total.weather_explained_change_kwh:+,.0f} kWh\n"
            f"- Remaining unexplained: {total.unexplained_change_kwh:+,.0f} kWh"
            + (
                f"\n- By fuel: gas {contributions.gas_change_kwh:+,.0f} kWh, "
                f"electricity {contributions.electricity_change_kwh:+,.0f} kWh"
                if contributions
                else ""
            )
        )

    if narrative is not None:
        meaning = narrative.weather_effect or narrative.is_unusual
        if meaning and narrative.which_fuel:
            meaning = f"{narrative.which_fuel} {meaning}"
        if meaning:
            st.markdown(f"**What this means:** {meaning}")
        if narrative.action:
            st.markdown(f"**What to do:** {narrative.action}")
    st.caption("Full breakdown, other comparison modes, and history: 'How did this month compare?'")


def render_executive_briefing(
    clean: pd.DataFrame,
    analyst_report: AnalystReport,
    forecast_12mo: ForecastResult | None,
    forecast_error: str | None,
    month_ctx: MonthContext,
    fuel: str = "total",
) -> None:
    st.caption(
        "An evidence-based briefing, not a statistics dump -- every statement below traces to a "
        "specific number computed elsewhere in the app. The full report with evidence and "
        "methodology is under Data and methods."
    )

    _render_month_headline(month_ctx)

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
    st.subheader("Forecast: expected annual energy cost")
    if forecast_error:
        st.info(f"Forecast unavailable: {forecast_error}")
    elif forecast_12mo is not None:
        unit_rate = clean["cost_gbp"].sum() / clean["consumption_kwh"].sum()
        lower = float(forecast_12mo.p10.sum())
        expected = float(forecast_12mo.p50.sum())
        upper = float(forecast_12mo.p90.sum())
        tooltip = (
            "From resampling the model's own past forecast errors -- bounds on what's plausible, "
            "not equally likely outcomes; values near the expected estimate are more likely. "
            "Consumption cost only; the caption below adds standing charges and VAT."
        )
        # Whole pounds: bootstrap uncertainty bands don't support penny precision.
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected", f"£{expected * unit_rate:,.0f}", help=tooltip)
        c2.metric("Lower estimate", f"£{lower * unit_rate:,.0f}", help=tooltip)
        c3.metric("Upper estimate", f"£{upper * unit_rate:,.0f}", help=tooltip)
        standing = standing_charge_for_months(list(forecast_12mo.forecast_dates), fuel)
        vat_rate = SETTINGS.billing.vat_rate
        total_bill = (expected * unit_rate + standing) * (1 + vat_rate)
        st.caption(
            f"Cards are consumption cost only. Adding ≈ £{standing:,.0f} standing charges and "
            f"{vat_rate:.0%} VAT gives an estimated total bill of ≈ £{total_bill:,.0f} (expected "
            "case). Produced by the best-performing statistical model in cross-validation -- see "
            "'What should I expect next?' for the chart, seasonal expectations, and model details."
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
        st.caption("See 'What drives my usage?' for the full electricity-vs-gas breakdown.")

    st.divider()
    st.subheader("Confidence")
    c1, c2, c3, c4 = st.columns(4)
    _confidence_badge(c1, "Data quality", analyst_report.confidence["data_quality"])
    _confidence_badge(c2, "Forecast", analyst_report.confidence["forecast"])
    _confidence_badge(c3, "Weather model", analyst_report.confidence["weather_model"])
    _confidence_badge(c4, "Anomaly detection", analyst_report.confidence["anomaly_detection"])
    st.caption(
        "The full report -- every finding's evidence and every recommendation's rationale -- is "
        "under Data and methods → Full report (AI Analyst)."
    )
