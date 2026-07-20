"""'How did this month compare?' -- the primary month-comparison page, plus the shared
comparison state every other surface (Home, Consultant) reads.

The selected month / fuel / comparison mode live in ``st.session_state``
under module-level keys; ``build_month_context`` resolves them (with
defaults: latest complete month, combined energy, same month last year)
and builds the typed comparison objects once per rerun in ``main()``, so
the Home page, this page, and the Consultant all describe the exact same
comparison -- the same single-source-of-truth pattern the app already uses
for the per-fuel analysis dict (audit findings F2/F7).

A partial in-progress month is never a primary selection: it is excluded
from the month selector, flagged with a notice, and inspectable only as a
clearly-labelled month-to-date figure -- never annualized or extrapolated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd
import streamlit as st

from app.charts import year_over_year_overlay
from app.charts_month import same_month_history_bar, two_month_grouped_bar
from config import SETTINGS, BillingConfig
from src.anomalies import Anomaly
from src.billing import bill_breakdown
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType
from src.kpis import compute_kpis
from src.monthly_comparison import (
    COMPARISON_MODE_LABELS,
    ComparisonMode,
    DisplayMode,
    FuelContributions,
    MonthlyComparison,
    build_monthly_comparison,
    fuel_contributions,
    in_progress_month,
    latest_complete_month,
    month_to_date_note,
    selectable_months,
)
from src.monthly_narrative import MonthlyComparisonNarrative, build_monthly_comparison_narrative
from src.utils import format_gbp

MONTH_KEY = "month_cmp_selected"
FUEL_KEY = "month_cmp_fuel"
MODE_KEY = "month_cmp_mode"

LONG_TERM_MODE: Final = "long_term"
_MODE_OPTIONS: list[DisplayMode] = [*COMPARISON_MODE_LABELS, LONG_TERM_MODE]
MODE_DISPLAY_LABELS: dict[DisplayMode, str] = {m: label for m, label in COMPARISON_MODE_LABELS.items()}
MODE_DISPLAY_LABELS[LONG_TERM_MODE] = "Long-term trend"

_FUEL_OPTIONS: list[EnergyType] = ["total", "electricity", "gas"]
FUEL_DISPLAY_LABELS: dict[EnergyType, str] = {
    "total": "Combined energy",
    "electricity": "Electricity",
    "gas": "Gas",
}

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


@dataclass
class MonthContext:
    """Everything the month-comparison surfaces need, built once per rerun."""

    selected_month: pd.Timestamp | None
    mode: DisplayMode
    fuel: EnergyType
    comparisons: dict[EnergyType, MonthlyComparison | None]
    featured: MonthlyComparison | None  # comparisons[fuel]
    contributions: FuelContributions | None
    narrative: MonthlyComparisonNarrative | None
    partial_note: str | None
    mtd_month: pd.Timestamp | None
    selectable: list[pd.Timestamp]
    weather_enabled: bool
    billing: BillingConfig


def build_month_context(
    fuel_frames: dict[EnergyType, pd.DataFrame],
    fuel_merged: Mapping[EnergyType, pd.DataFrame | None],
    fuel_energy_results: Mapping[EnergyType, EnergySignatureResult | None],
    weather_enabled: bool,
    today: pd.Timestamp | None = None,
    billing_config: BillingConfig | None = None,
) -> MonthContext:
    """Resolve the shared month/fuel/mode state and build all per-fuel comparisons.

    Stale session values (e.g. a month that vanished after a re-upload) fall
    back to the defaults rather than raising -- the widget keys are simply
    reset so the next render shows the default selection too.
    """
    clean_total = fuel_frames.get("total", pd.DataFrame())
    reference = clean_total if not clean_total.empty else next(
        (df for df in fuel_frames.values() if not df.empty), pd.DataFrame()
    )
    months = selectable_months(reference, today)
    partial = in_progress_month(reference, today)
    note = month_to_date_note(reference, today)

    selected = st.session_state.get(MONTH_KEY)
    if selected not in months:
        selected = latest_complete_month(reference, today)
        st.session_state.pop(MONTH_KEY, None)
    fuel = st.session_state.get(FUEL_KEY)
    if fuel not in _FUEL_OPTIONS or fuel_frames.get(fuel, pd.DataFrame()).empty:
        fuel = "total" if not clean_total.empty else next(
            (f for f in _FUEL_OPTIONS if not fuel_frames.get(f, pd.DataFrame()).empty), "total"
        )
        st.session_state.pop(FUEL_KEY, None)
    mode = st.session_state.get(MODE_KEY)
    if mode not in _MODE_OPTIONS:
        mode = "same_month_last_year"
        st.session_state.pop(MODE_KEY, None)
    fuel = cast(EnergyType, fuel)  # both validated against their option lists above
    mode = cast(DisplayMode, mode)

    comparisons: dict[EnergyType, MonthlyComparison | None] = {}
    build_mode: ComparisonMode = mode if mode != LONG_TERM_MODE else "same_month_last_year"
    if selected is not None:
        for f in _FUEL_OPTIONS:
            df = fuel_frames.get(f, pd.DataFrame())
            comparisons[f] = (
                build_monthly_comparison(
                    df,
                    f,
                    build_mode,
                    selected,
                    fuel_merged.get(f) if weather_enabled else None,
                    fuel_energy_results.get(f) if weather_enabled else None,
                    today=today,
                )
                if not df.empty
                else None
            )
    else:
        comparisons = dict.fromkeys(_FUEL_OPTIONS)

    contributions = fuel_contributions(
        comparisons.get("total"), comparisons.get("electricity"), comparisons.get("gas")
    )
    featured = comparisons.get(fuel)
    narrative = (
        build_monthly_comparison_narrative(
            featured,
            contributions if fuel == "total" else None,
            weather_enabled=weather_enabled,
        )
        if featured is not None
        else None
    )

    return MonthContext(
        selected_month=selected,
        mode=mode,
        fuel=fuel,
        comparisons=comparisons,
        featured=featured,
        contributions=contributions,
        narrative=narrative,
        partial_note=note,
        mtd_month=partial,
        selectable=months,
        weather_enabled=weather_enabled,
        billing=billing_config if billing_config is not None else SETTINGS.billing,
    )


def _comparison_label(comparison: MonthlyComparison) -> str:
    if comparison.comparison_mode == "typical_month":
        return f"Typical {comparison.selected_month.strftime('%B')}"
    if comparison.comparison_month is not None:
        return comparison.comparison_month.strftime("%B %Y")
    return "comparison period"


_MODE_PURPOSE: dict[str, str] = {
    "same_month_last_year": "The default: the fairest like-for-like comparison, one year apart.",
    "previous_month": (
        "Short-term movement only. Adjacent months can differ because of seasonality -- "
        "January vs December says little about performance."
    ),
    "typical_month": "The selected month against the median of previous same-calendar months.",
    "best_month": "The selected month against the lowest recorded same-calendar month.",
    "worst_month": "The selected month against the highest recorded same-calendar month.",
    LONG_TERM_MODE: "Trailing 12-month totals and multi-year patterns -- context, not a verdict.",
}


def _render_controls(ctx: MonthContext) -> None:
    col_month, col_fuel, col_mode = st.columns([1.1, 1.2, 1.4])
    with col_month:
        st.selectbox(
            "Month",
            ctx.selectable,
            index=ctx.selectable.index(ctx.selected_month) if ctx.selected_month in ctx.selectable else 0,
            format_func=lambda m: m.strftime("%B %Y"),
            key=MONTH_KEY,
        )
    with col_fuel:
        available = [f for f in _FUEL_OPTIONS if ctx.comparisons.get(f) is not None or f == ctx.fuel]
        st.segmented_control(
            "Fuel",
            available,
            format_func=lambda f: FUEL_DISPLAY_LABELS[f],
            key=FUEL_KEY,
            default=ctx.fuel if FUEL_KEY not in st.session_state else None,
        )
    with col_mode:
        st.selectbox(
            "Compare against",
            _MODE_OPTIONS,
            index=_MODE_OPTIONS.index(ctx.mode),
            format_func=lambda m: MODE_DISPLAY_LABELS[m],
            key=MODE_KEY,
        )
    st.caption(_MODE_PURPOSE[ctx.mode])


def _render_month_to_date(ctx: MonthContext, fuel_frames: dict[EnergyType, pd.DataFrame]) -> None:
    if ctx.mtd_month is None:
        return
    st.info(ctx.partial_note)
    with st.expander(f"Month to date: {ctx.mtd_month.strftime('%B %Y')} (incomplete)"):
        df = fuel_frames.get(ctx.fuel, pd.DataFrame())
        rows = df[df["month_start"] == ctx.mtd_month] if not df.empty else pd.DataFrame()
        if rows.empty:
            st.write("No month-to-date figures for this fuel.")
            return
        row = rows.iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Month-to-date consumption", f"{row['consumption_kwh']:,.0f} kWh")
        c2.metric("Month-to-date cost", format_gbp(float(row["cost_gbp"])))
        prior = df[df["month_start"] == ctx.mtd_month - pd.DateOffset(years=1)]
        if not prior.empty:
            st.caption(
                f"For reference, the full {prior.iloc[0]['month_start'].strftime('%B %Y')} used "
                f"{prior.iloc[0]['consumption_kwh']:,.0f} kWh -- not directly comparable while "
                f"{ctx.mtd_month.strftime('%B %Y')} is still accumulating. No month-end estimate "
                "is shown because it would be an extrapolation, not a measurement."
            )


def _render_explanation(ctx: MonthContext) -> None:
    narrative = ctx.narrative
    if narrative is None:
        return
    st.subheader("What changed, and why")
    lines: list[tuple[str, str | None]] = [
        ("What changed", narrative.what_changed),
        ("Which fuel caused it", narrative.which_fuel),
        ("How weather affected it", narrative.weather_effect),
        ("Is it unusual", narrative.is_unusual),
        ("Suggested action", narrative.action),
    ]
    for label, text in lines:
        if text:
            st.markdown(f"**{label}:** {text}")
    if narrative.which_fuel is None and ctx.fuel == "total" and ctx.contributions is None:
        st.caption("Add Electricity and Gas exports to attribute the change to a fuel.")
    if narrative.weather_effect is None and not ctx.weather_enabled:
        st.caption("Turn on 'Weather adjustment' in the sidebar to see how much weather explains.")
    icon = _CONFIDENCE_ICON[narrative.confidence]
    st.caption(f"Confidence: {icon} {narrative.confidence} -- {narrative.confidence_reason}")
    st.caption(f"Limitation: {narrative.limitation}")


def _render_costs_and_carbon(ctx: MonthContext) -> None:
    featured = ctx.featured
    if featured is None or featured.current_cost_gbp is None:
        return
    st.subheader("Cost")
    comparison_label = _comparison_label(featured)

    current_bill = bill_breakdown(featured.selected_month, featured.current_cost_gbp, ctx.fuel, ctx.billing)
    comparison_bill = (
        bill_breakdown(featured.comparison_month, featured.comparison_cost_gbp, ctx.fuel, ctx.billing)
        if featured.comparison_month is not None and featured.comparison_cost_gbp is not None
        else None
    )

    period_label = (
        f"{current_bill.billing_period_start.strftime('%d %b')} - "
        f"{current_bill.billing_period_end.strftime('%d %b %Y')}"
    )
    cols = st.columns(4)
    cols[0].metric(
        "Consumption cost",
        format_gbp(current_bill.consumption_cost_gbp),
        help=f"From the billing export, excluding VAT. Billing period: {period_label}.",
    )
    cols[1].metric(
        "Standing charge",
        format_gbp(current_bill.standing_charge_gbp),
        help=(
            f"{current_bill.days_in_period} days x the daily rate(s) from the sidebar's Tariff "
            f"section ({ctx.billing.electricity_standing_gbp_per_day * 100:.2f}p electricity, "
            f"{ctx.billing.gas_standing_gbp_per_day * 100:.2f}p gas, ex VAT); the combined view "
            "pays both."
        ),
    )
    cols[2].metric(f"VAT ({ctx.billing.vat_rate:.0%})", format_gbp(current_bill.vat_gbp))
    delta = None
    if comparison_bill is not None:
        bill_change = current_bill.total_bill_gbp - comparison_bill.total_bill_gbp
        sign = "+" if bill_change >= 0 else "-"
        delta = f"{sign}{format_gbp(abs(bill_change))} vs {comparison_label}"
    cols[3].metric("Estimated total bill", format_gbp(current_bill.total_bill_gbp), delta, delta_color="off")

    if featured.cost_change_from_usage_gbp is not None and featured.cost_change_from_rate_gbp is not None:
        usage_part = featured.cost_change_from_usage_gbp
        rate_part = featured.cost_change_from_rate_gbp
        st.caption(
            f"Of the consumption-cost change vs {comparison_label}, "
            f"{'+' if usage_part >= 0 else '-'}{format_gbp(abs(usage_part))} came from using more or "
            f"less energy and {'+' if rate_part >= 0 else '-'}{format_gbp(abs(rate_part))} from a "
            "different effective price per kWh. Standing charges are near-identical for the same "
            "calendar month, so they rarely explain a year-on-year change."
        )
    st.caption(
        f"Bill covers the {period_label} billing period (bills run 6th to 5th). Standing charges "
        "and VAT are estimated from the rates in the sidebar's Tariff section, not read from the "
        "exports; the exported cost is treated as excluding VAT (see config.BillingConfig to "
        "flip that)."
    )

    elec = ctx.comparisons.get("electricity")
    gas = ctx.comparisons.get("gas")
    if elec is not None or gas is not None:
        st.subheader("Carbon")
        cols = st.columns(2)
        factors = SETTINGS.carbon
        for col, comparison, factor, label in (
            (cols[0], elec, factors.electricity_kg_co2e_per_kwh, "Electricity"),
            (cols[1], gas, factors.gas_kg_co2e_per_kwh, "Gas"),
        ):
            if comparison is None:
                continue
            current_kg = comparison.current_consumption_kwh * factor
            delta = None
            if comparison.comparison_consumption_kwh is not None:
                delta_kg = (comparison.current_consumption_kwh - comparison.comparison_consumption_kwh) * factor
                delta = f"{delta_kg:+,.0f} kg vs {_comparison_label(comparison)}"
            col.metric(f"{label} CO2e (estimated)", f"{current_kg:,.0f} kg", delta, delta_color="off")
        st.caption(
            f"Estimates use static UK factors ({factors.electricity_kg_co2e_per_kwh} kg/kWh "
            f"electricity, {factors.gas_kg_co2e_per_kwh} kg/kWh gas -- DESNZ/DEFRA 2024); real grid "
            "intensity varies, so treat these as rough, not precise."
        )


def _render_technical_details(ctx: MonthContext, anomalies: list[Anomaly]) -> None:
    featured = ctx.featured
    if featured is None:
        return
    with st.expander("Technical details"):
        st.markdown(
            "**Formulas.** Percentage change = (selected - comparison) / comparison x 100. "
            "Weather-expected kWh = fitted kWh/day from the degree-day regression x days in month "
            "(the regression is fitted once elsewhere; nothing is refitted here). "
            "Weather-explained change = expected(selected) - expected(comparison). "
            "Unexplained change = actual change - weather-explained change. "
            "Weather-adjusted kWh = actual - (expected - base load)."
        )
        rows = {
            "Selected month consumption (kWh)": f"{featured.current_consumption_kwh:,.1f}",
            "Comparison consumption (kWh)": (
                f"{featured.comparison_consumption_kwh:,.1f}"
                if featured.comparison_consumption_kwh is not None
                else "n/a"
            ),
            "Average daily (selected, kWh/day)": f"{featured.current_avg_daily_kwh:,.2f}",
        }
        if featured.current_weather_adjusted_kwh is not None:
            rows["Weather-adjusted (selected, kWh)"] = f"{featured.current_weather_adjusted_kwh:,.1f}"
        if featured.comparison_weather_adjusted_kwh is not None:
            rows["Weather-adjusted (comparison, kWh)"] = f"{featured.comparison_weather_adjusted_kwh:,.1f}"
        if featured.weather_explained_change_kwh is not None:
            rows["Weather-explained change (kWh)"] = f"{featured.weather_explained_change_kwh:+,.1f}"
            rows["Unexplained change (kWh)"] = f"{featured.unexplained_change_kwh:+,.1f}"
        if featured.current_heating_degree_days is not None:
            rows["Heating degree days (selected)"] = f"{featured.current_heating_degree_days:,.0f}"
        if featured.comparison_heating_degree_days is not None:
            rows["Heating degree days (comparison)"] = f"{featured.comparison_heating_degree_days:,.0f}"
        st.table(pd.DataFrame({"Value": rows}))

        thresholds = SETTINGS.month_comparison
        st.caption(
            f"Change categories: under {thresholds.little_change_pct:.0f}% = little, "
            f"{thresholds.little_change_pct:.0f}-{thresholds.large_change_pct:.0f}% = moderate, "
            f"over {thresholds.large_change_pct:.0f}% = large (practical materiality bands, not "
            "statistical significance -- a single month's change isn't formally testable at this "
            "sample size). An unexplained gap is called meaningful beyond "
            f"{thresholds.meaningful_residual_z} residual standard deviations of the weather fit."
        )

        month_anomaly = next((a for a in anomalies if a.date == featured.selected_month), None)
        if month_anomaly is not None:
            st.markdown(
                f"**Anomaly status:** flagged as a {month_anomaly.direction} by "
                f"{len(month_anomaly.methods)} of 3 methods ({', '.join(month_anomaly.methods)}) -- "
                f"{month_anomaly.rank_context}. See 'Did anything unusual happen?'."
            )
        else:
            st.markdown("**Anomaly status:** not flagged by any of the three detection methods.")


def _render_long_term(clean_df: pd.DataFrame) -> None:
    """The long-term mode: trailing 12-month KPIs and the multi-year overlay --
    the whole-period view, framed as context rather than a monthly verdict."""
    kpis = compute_kpis(clean_df, months=12)
    headline = [k for k in kpis if k.label in ("Total consumption", "Total cost")]
    if headline:
        cols = st.columns(len(headline))
        for col, kpi in zip(cols, headline, strict=True):
            delta = f"{kpi.pct_change:+.1f}% vs previous 12 months" if kpi.pct_change is not None else None
            value = format_gbp(kpi.current) if kpi.unit == "£" else f"{kpi.current:,.0f} kWh"
            col.metric(f"{kpi.label} (trailing 12 months)", value, delta, delta_color="off")
    # Explicit key: the same overlay chart also renders under Long-term trends, and two
    # identical figures would collide on Streamlit's auto-generated element ID.
    st.plotly_chart(year_over_year_overlay(clean_df), width="stretch", key="month_long_term_overlay")
    st.caption(
        "One line per year, aligned by calendar month. The 'Long-term trends' tab holds the "
        "full whole-period analysis: annual totals, fuel comparisons, and underlying trend."
    )


def render_month_comparison(
    ctx: MonthContext,
    fuel_frames: dict[EnergyType, pd.DataFrame],
    fuel_anomalies_all: dict[EnergyType, list[Anomaly]],
) -> None:
    st.caption(
        "The default comparison is the latest complete month against the same calendar month "
        "last year -- comparing January with December would mostly measure the seasons, not you."
    )
    if not ctx.selectable:
        st.info(
            "No complete month in the data yet -- the current month is still in progress, so "
            "there is nothing that can be fairly compared."
        )
        return

    _render_controls(ctx)
    _render_month_to_date(ctx, fuel_frames)

    if ctx.mode == LONG_TERM_MODE:
        _render_long_term(fuel_frames.get(ctx.fuel, fuel_frames.get("total", pd.DataFrame())))
        return

    featured = ctx.featured
    narrative = ctx.narrative
    if featured is None or narrative is None:
        st.info("No data for this fuel and month.")
        return

    st.markdown(f"#### {narrative.headline}")
    st.caption(
        f"{featured.judgement} · Confidence: {_CONFIDENCE_ICON[featured.confidence]} "
        f"{featured.confidence}"
    )

    if featured.percentage_change is not None:
        chart = two_month_grouped_bar(ctx.comparisons, _comparison_label(featured))
        if chart is not None:
            st.plotly_chart(chart, width="stretch")
        if featured.comparison_mode == "typical_month" and featured.same_month_low_kwh is not None:
            month_name = featured.selected_month.strftime("%B")
            st.caption(
                f"Typical {month_name} range across {len(featured.same_month_years)} previous "
                f"year(s): {featured.same_month_low_kwh:,.0f}-{featured.same_month_high_kwh:,.0f} kWh "
                f"(median {featured.same_month_median_kwh:,.0f}). This {month_name}: "
                f"{featured.current_consumption_kwh:,.0f} kWh."
            )
    else:
        st.info(featured.interpretation)

    history_chart = same_month_history_bar(
        fuel_frames.get(ctx.fuel, pd.DataFrame()), featured.selected_month, ctx.mtd_month
    )
    if history_chart is not None and len(featured.same_month_years) >= 1:
        st.subheader("This calendar month, year by year")
        st.plotly_chart(history_chart, width="stretch")

    _render_explanation(ctx)
    _render_costs_and_carbon(ctx)
    _render_technical_details(ctx, fuel_anomalies_all.get(ctx.fuel, []))
