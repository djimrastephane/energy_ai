"""Home tab: the month-first executive summary.

Leads with the question a homeowner actually asks -- "how did this month
compare with the same month last year?" -- using the latest *complete*
month (never the in-progress one), then the single biggest finding, the
single largest saving opportunity (if any), and one forecast figure.
Deliberately short: a "30-second briefing" reads like one when it's four
things, not eight. The overall 12-month assessment, the full household
energy profile, and the per-area confidence grid are not repeated here --
they already live in the Full Report (``tabs_analyst.py``), one tab over
under Advanced, so nothing is lost, just not duplicated (UX audit finding:
Home stacked eight sections and ended on a wall of confidence badges).
Every number is pulled from the shared ``MonthContext`` built once
in ``main()`` and from ``src.report.AnalystReport`` -- the same objects
the comparison page, AI Analyst, and Consultant read, so the surfaces
can never contradict each other.

The trailing-12-month KPI strip that used to open this page lives under
Advanced now: a 12-month window answers "how is the year going", not "how
did this month compare", and it kept pulling attention first.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.tabs_month import MonthContext
from config import SETTINGS, BillingConfig
from src.billing import standing_charge_for_months
from src.forecast_evaluation import ForecastResult
from src.ingestion import EnergyType
from src.report import NO_SAVINGS_MESSAGE, AnalystReport


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
    fuel: EnergyType = "total",
    billing_config: BillingConfig | None = None,
) -> None:
    billing_config = billing_config or SETTINGS.billing
    st.caption(
        "An evidence-based briefing, not a statistics dump -- every statement below traces to a "
        "specific number computed elsewhere in the app. The full report with evidence and "
        "methodology is under Advanced."
    )

    _render_month_headline(month_ctx)

    st.divider()
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
    st.subheader("Forecast: expected energy cost")
    if forecast_error:
        st.info(f"Forecast unavailable: {forecast_error}")
    elif forecast_12mo is not None:
        horizon_months = len(forecast_12mo.forecast_dates)
        unit_rate = clean["cost_gbp"].sum() / clean["consumption_kwh"].sum()
        expected = float(forecast_12mo.p50.sum())
        consumption_cost = expected * unit_rate
        standing = standing_charge_for_months(list(forecast_12mo.forecast_dates), fuel, billing_config)
        vat_rate = billing_config.vat_rate
        total_bill = (consumption_cost + standing) * (1 + vat_rate)
        tooltip = (
            "The model's central estimate, including standing charges and VAT. See "
            "'What should I expect next?' for the lower/upper plausible range and how sure the "
            "model is."
        )
        # Whole pounds: bootstrap uncertainty bands don't support penny precision. Label names
        # the actual horizon (the sidebar's "Forecast horizon (months)" slider) rather than
        # assuming 12 -- this used to always be 12 regardless of the slider.
        st.metric(f"Expected {horizon_months}-month bill", f"£{total_bill:,.0f}", help=tooltip)
        st.caption(
            f"£{consumption_cost:,.0f} consumption + ≈ £{standing:,.0f} standing charges, "
            f"{vat_rate:.0%} VAT (expected case). See 'What should I expect next?' for the "
            "chart, plausible range, seasonal expectations, and model details."
        )
    else:
        st.info("Not enough history for a cross-validated forecast yet.")

    st.divider()
    st.caption(
        "See the full picture -- the overall 12-month assessment, the household energy profile "
        "(fuel mix, largest cost driver, weather vs. behaviour), and every confidence rating with "
        "its reason -- under Advanced → Full report (AI Analyst)."
    )
