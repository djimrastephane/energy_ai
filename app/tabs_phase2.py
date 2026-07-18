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

from config import SETTINGS
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
from src.weather import (
    compute_monthly_degree_days,
    fetch_daily_temperature,
    merge_weather_with_consumption,
)


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
    daily = fetch_daily_temperature(w.latitude, w.longitude, start, end, w.timezone, SETTINGS.weather_cache_dir)
    monthly_dd = compute_monthly_degree_days(daily, w.base_heat_c, w.base_cool_c)
    merged = merge_weather_with_consumption(clean, monthly_dd)
    result = fit_energy_signature(merged)
    return merged, result


def render_seasonality(stl_result: STLResult | None, stl_error: str | None) -> None:
    st.caption("Uses full history regardless of the sidebar year filter (STL needs a contiguous series).")
    if stl_error:
        st.warning(stl_error)
        return
    st.plotly_chart(stl_components_figure(stl_result), width="stretch")
    st.write(interpret_decomposition(stl_result))


def render_weather_adjustment(
    weather_enabled: bool,
    merged: pd.DataFrame | None,
    result: EnergySignatureResult | None,
    weather_error: str | None,
    fuel: str = "total",
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

    col1, col2 = st.columns(2)
    with col1:
        pdp_hdd = partial_dependence(result, merged, "avg_daily_hdd")
        st.plotly_chart(energy_signature_scatter(merged, pdp_hdd, "avg_daily_hdd"), width="stretch")
    with col2:
        pdp_cdd = partial_dependence(result, merged, "avg_daily_cdd")
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

    with st.expander("Advanced diagnostics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R-squared", f"{result.r_squared:.0%}")
        c2.metric("Adjusted R-squared", f"{result.adj_r_squared:.0%}")
        c3.metric("Durbin-Watson", f"{result.durbin_watson:.2f}")
        c4.metric("Months used", result.n_obs)
        st.write(interpret_energy_signature(result, unit_rate, fuel))


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
