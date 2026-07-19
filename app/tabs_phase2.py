"""Phase 2 tab renderers: Seasonality & Trend, Weather Adjustment, Change Points.

Split out of ``streamlit_app.py`` to keep that module under the project's
preferred ~300-line-per-file guideline.

Weather-loading and change-point detection are computed once in
``streamlit_app.main()`` (not inside these render functions) since the
Executive Briefing and AI Analyst pages need the same ``merged``/``result``/
``changepoints`` objects -- computing them here too would let the two
surfaces silently drift out of sync.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from charts import changepoint_timeline, energy_signature_scatter, stl_components_figure
from charts_seasonal import seasonal_calendar_profile_figure, seasonal_overview_figure
from tabs_weather_context import render_weather_impact_summary

from config import SETTINGS
from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.decomposition import STLResult, interpret_decomposition
from src.energy_signature import (
    EnergySignatureResult,
    annual_weather_adjusted_comparison,
    fit_energy_signature,
    interpret_energy_signature,
    partial_dependence,
)
from src.findings import finding_weather
from src.investigation import build_investigation_checklist
from src.seasonal_summary import seasonal_profile, summarize_seasonal_pattern
from src.weather import (
    compute_monthly_degree_days,
    fetch_daily_weather,
    merge_weather_with_consumption,
)
from src.weather_context import cooling_variation_negligible

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

_INFLUENCE_EXPLAINER = {
    "Strong": (
        "Seasonal influence is strong. Most month-to-month variation follows the normal "
        "winter and summer cycle."
    ),
    "Moderate": (
        "Seasonal influence is moderate. Some month-to-month variation follows the normal "
        "winter and summer cycle, but a meaningful share doesn't."
    ),
    "Low": (
        "Seasonal influence is low. Month-to-month variation mostly doesn't follow the "
        "normal winter and summer cycle."
    ),
}

_DEFINITIONS = [
    ("Trend", "The underlying direction after removing the normal seasonal cycle."),
    ("Seasonal effect", "The typical increase or decrease associated with each calendar month."),
    ("Residual", "The part not explained by trend or seasonality."),
]


@st.cache_data(show_spinner="Fetching historical weather from Open-Meteo...")
def load_weather_analysis(clean: pd.DataFrame) -> tuple[pd.DataFrame, EnergySignatureResult]:
    """Fetch weather, fit the energy signature, and return (merged_df, result).

    Cached because the fetch is a network call (mitigated further by
    ``src.weather``'s own on-disk cache) and this can otherwise re-run on
    every widget interaction. Raises ``WeatherFetchError`` or ``ValueError``
    on failure -- callers decide how to present that.
    """
    w = SETTINGS.weather
    start = clean["month_start"].min()
    end = clean["month_start"].max() + pd.offsets.MonthEnd(1)
    daily = fetch_daily_weather(w.latitude, w.longitude, start, end, w.timezone, SETTINGS.weather_cache_dir)
    monthly_dd = compute_monthly_degree_days(daily, w.base_heat_c, w.base_cool_c)
    merged = merge_weather_with_consumption(clean, monthly_dd)
    result = fit_energy_signature(merged)
    return merged, result


def render_seasonality(
    stl_result: STLResult | None,
    stl_error: str | None,
    clean: pd.DataFrame,
    anomalies: list[Anomaly],
) -> None:
    """Non-technical view: is usage seasonal, is there a persistent trend, did anything stand
    out -- readable in a few seconds without knowing what STL, trend, or residual mean. The
    full statistical decomposition (identical calculation to before this redesign) is retained
    unchanged in the "Advanced statistical decomposition" expander at the bottom.
    """
    st.caption("Uses full history regardless of the sidebar year filter (STL needs a contiguous series).")
    if stl_error:
        st.warning(stl_error)
        return

    summary = summarize_seasonal_pattern(stl_result, anomalies)

    st.markdown(f"##### {summary.main_conclusion}")
    st.write(summary.why_it_matters)
    icon = _CONFIDENCE_ICON[summary.confidence]
    st.caption(f"Confidence: {icon} {summary.confidence} -- {summary.confidence_reason}")
    if summary.unusual_month is not None:
        st.info(summary.unusual_month_context)

    st.divider()
    st.plotly_chart(seasonal_overview_figure(clean, stl_result, anomalies), width="stretch")
    st.caption(
        "Bars are actual monthly consumption -- lighter bars belong to a year with incomplete "
        "data. The line is the underlying trend with the regular seasonal swing smoothed out. "
        "Triangles mark months that came in materially higher (▲) or lower (▼) than expected."
    )

    st.subheader("Which months run high or low?")
    profile = seasonal_profile(stl_result)
    st.plotly_chart(seasonal_calendar_profile_figure(profile), width="stretch")
    st.caption(
        "Positive bars are calendar months that typically run above the household's overall "
        "level; negative bars are months that typically run below it -- both once the "
        "long-term trend is removed. Averaged across every year in the data."
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Seasonal influence", summary.seasonal_band, help=_INFLUENCE_EXPLAINER[summary.seasonal_band])
    trend_detail = (
        "Change is within normal month-to-month variation -- no persistent direction."
        if summary.trend_label == "Stable"
        else f"Moving at roughly {abs(summary.trend_pct_per_year):.0f}% a year, beyond the normal seasonal swing."
    )
    c2.metric("Long-term trend", summary.trend_label, help=trend_detail)
    if summary.unusual_month is not None:
        c3.metric(
            "Largest unexplained deviation",
            summary.unusual_month.date.strftime("%b %Y"),
            f"{summary.unusual_month_kwh:+,.0f} kWh",
            help=summary.unusual_month_context,
        )
    else:
        c3.metric("Largest unexplained deviation", "None", help=summary.unusual_month_context)

    st.divider()
    with st.expander("Advanced statistical decomposition"):
        st.plotly_chart(stl_components_figure(stl_result), width="stretch")
        st.write(interpret_decomposition(stl_result))
        d1, d2 = st.columns(2)
        d1.metric("Seasonal strength", f"{stl_result.seasonal_strength:.0%}")
        d2.metric("Trend strength", f"{stl_result.trend_strength:.0%}")
        st.caption(
            "Method: STL (Seasonal-Trend decomposition using LOESS, robust fitting), 12-month "
            "period. Strength measures follow Hyndman & Athanasopoulos: "
            "1 - Var(residual) / Var(component + residual)."
        )
        st.markdown("**Definitions**")
        for term, definition in _DEFINITIONS:
            st.write(f"- **{term}:** {definition}")


def render_weather_adjustment(
    weather_enabled: bool,
    merged: pd.DataFrame | None,
    result: EnergySignatureResult | None,
    weather_error: str | None,
    fuel: str = "total",
    weather_context_df: pd.DataFrame | None = None,
) -> None:
    st.caption(f"Location: {SETTINGS.weather.location_label}. Uses full history regardless of the year filter.")
    if not weather_enabled:
        st.info(
            "Turn on 'Weather adjustment' in the sidebar to fetch historical temperature and see "
            "the weather-adjusted energy signature (requires internet access to Open-Meteo)."
        )
        return
    if weather_error:
        st.warning(f"Could not fetch or fit weather data: {weather_error}")
        return

    unit_rate = merged["cost_gbp"].sum() / merged["consumption_kwh"].sum()
    finding = finding_weather(result, unit_rate, fuel)
    if finding:
        st.write(finding.narrative)
        st.caption(f"Confidence: {finding.confidence} -- {finding.confidence_reason}")

    pdp_hdd = partial_dependence(result, merged, "avg_daily_hdd")
    pdp_cdd = partial_dependence(result, merged, "avg_daily_cdd")
    cooling_negligible = cooling_variation_negligible(merged)
    if cooling_negligible:
        # This household has heating but effectively zero cooling variation (a real finding
        # for an Aberdeen home, not a gap) -- keep the main view heating-only and say so,
        # rather than showing an empty cooling chart. CDD stays computed; the chart moves
        # into Advanced diagnostics.
        st.plotly_chart(energy_signature_scatter(merged, pdp_hdd, "avg_daily_hdd"), width="stretch")
        st.caption(
            "Warm-weather (cooling) electricity effects are too small to detect at this home "
            "-- cooling degree days are near zero across the whole history. Cooling detail "
            "remains available under Advanced diagnostics."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(energy_signature_scatter(merged, pdp_hdd, "avg_daily_hdd"), width="stretch")
        with col2:
            st.plotly_chart(energy_signature_scatter(merged, pdp_cdd, "avg_daily_cdd"), width="stretch")

    st.subheader("Weather-adjusted annual comparison")
    st.caption("Answers: is a year-on-year change more likely weather, or behaviour?")
    comparison = annual_weather_adjusted_comparison(merged, result)
    if comparison.empty:
        st.info("No complete calendar year has full weather coverage yet for this comparison.")
    else:
        display = comparison.rename(
            columns={
                "year": "Year",
                "actual_kwh": "Actual (kWh)",
                "weather_predicted_kwh": "Weather-predicted (kWh)",
                "difference_kwh": "Difference (kWh)",
                "interpretation": "Interpretation",
            }
        ).round(0)
        st.dataframe(display, hide_index=True, width="stretch")

    if weather_context_df is not None:
        st.divider()
        render_weather_impact_summary(weather_context_df)

    with st.expander("Advanced diagnostics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R-squared", f"{result.r_squared:.0%}")
        c2.metric("Adjusted R-squared", f"{result.adj_r_squared:.0%}")
        c3.metric("Durbin-Watson", f"{result.durbin_watson:.2f}")
        c4.metric("Months used", result.n_obs)
        st.write(interpret_energy_signature(result, unit_rate, fuel))
        if cooling_negligible:
            st.markdown("**Cooling (kept out of the main view -- negligible for this home):**")
            st.plotly_chart(energy_signature_scatter(merged, pdp_cdd, "avg_daily_cdd"), width="stretch")
            st.caption(
                "Cooling degree days (base 22°C) stay computed for reuse in other climates; "
                "for this household they carry no detectable signal."
            )


def _render_investigation_expander(
    label: str,
    date: pd.Timestamp,
    clean: pd.DataFrame,
    merged: pd.DataFrame | None,
    result: EnergySignatureResult | None,
    fuel: str = "total",
) -> None:
    checklist = build_investigation_checklist(date, clean, merged, result, fuel)
    with st.expander(f"Possible causes: {label}"):
        for item in checklist.items:
            marker = "✅" if item.checked else "⬜"
            st.write(f"{marker} **{item.label}** -- {item.reason}")


def render_change_points(
    stl_result: STLResult | None,
    stl_error: str | None,
    changepoints: list[ChangePoint],
    clean: pd.DataFrame,
    merged: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    fuel: str = "total",
) -> None:
    st.caption(
        "Detected on the deseasonalized series (trend + residual) so the regular winter/summer "
        "swing isn't mistaken for a behavioural shift. Uses full history regardless of the year filter."
    )
    if stl_error:
        st.warning(stl_error)
        return

    st.plotly_chart(changepoint_timeline(stl_result.deseasonalized, changepoints), width="stretch")

    if not changepoints:
        st.success("No change points detected -- consumption behaviour looks stable once seasonality is removed.")
        return

    rows = [
        {
            "Date": cp.date.strftime("%B %Y"),
            "Method": cp.method,
            "Direction": cp.direction,
            "Magnitude (kWh)": round(cp.magnitude_kwh, 1),
        }
        for cp in changepoints
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "'both' means PELT and CUSUM independently agree on that date -- higher-confidence than a "
        "single-method hit. Magnitude is the mean shift before vs. after that date."
    )

    for cp in changepoints:
        _render_investigation_expander(cp.date.strftime("%B %Y"), cp.date, clean, merged, energy_result, fuel)
