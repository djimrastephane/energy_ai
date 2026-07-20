"""AI Home Energy Intelligence Platform -- Streamlit app.

The analysis layers: data ingestion/validation, descriptive statistics, and
core KPIs; STL seasonality/trend decomposition, weather-adjusted "energy
signature" regression (Open-Meteo, opt-in via the sidebar toggle since it
needs a network call), and change-point detection; an 8-model forecasting
suite compared by walk-forward cross-validation with auto-selection and
P10/P50/P90 bands, plus anomaly detection (rolling z-score, STL-residual
ESD, Isolation Forest). A decision-support layer -- the findings/
recommendations/confidence engine (``src.report``) -- turns those outputs
into an Executive Briefing and a full AI Analyst report: no new statistics,
purely synthesis of what's already computed. Multi-fuel intelligence
(``src.comparisons``, ``src.cross_fuel_anomalies``, ``src.cost_engine``,
``src.carbon``, ``src.benchmarking``) compares the already-computed per-fuel
results against each other -- the sidebar's Fuel selector makes every module
generic over Electricity/Gas/Total.

Change-point detection, anomaly detection, and (if enabled) the weather fit
are computed once here in ``main()`` for the selected fuel (and, via
``app/multi_fuel.py``, for all three fuels), not inside individual tab
renderers, since the Executive Briefing, AI Analyst, Comparisons, Cost
Intelligence, and Carbon pages all need the exact same objects -- computing
them twice per fuel would let the surfaces silently drift out of sync.

Navigation is organized around user questions (Home / How did this month
compare? / What drives my usage? / ...), with the month-comparison journey
(``tabs_month.py``, backed by ``src.monthly_comparison``) as the primary
surface: latest complete month vs. the same calendar month last year.
Whole-period and method-level views remain available under Long-term trends
and Data and methods. Sidebar controls live in ``app/sidebar.py``; tab
rendering logic lives in the ``tabs_*`` modules, named for what they render
(``tabs_core.py`` for data quality/statistics/consumption, ``tabs_drivers.py``
for seasons/weather/usage shifts, ``tabs_forecast.py``/``tabs_anomalies.py``,
``tabs_briefing.py``/``tabs_analyst.py`` for decision support, ``tabs_fuel.py``/
``tabs_comparisons.py``/``tabs_cost.py``/``tabs_carbon.py`` for multi-fuel and
costs, and ``tabs_month.py`` for the month comparison) -- this module is just
top-level orchestration.
"""

from __future__ import annotations

import streamlit as st

from app.multi_fuel import compute_all_fuel_analysis
from app.report_html import build_household_report_html
from app.sidebar import render_sidebar
from app.tabs_analyst import build_report, render_ai_analyst
from app.tabs_anomalies import render_anomalies
from app.tabs_briefing import render_executive_briefing
from app.tabs_carbon import render_carbon
from app.tabs_comparisons import render_comparisons
from app.tabs_consultant import render_consultant
from app.tabs_core import (
    render_consumption_analysis,
    render_data_quality,
    render_statistical_analysis,
)
from app.tabs_cost import render_cost_intelligence
from app.tabs_drivers import (
    render_change_points,
    render_seasonality,
    render_weather_adjustment,
)
from app.tabs_forecast import (
    generate_forecast_cached,
    generate_multi_fuel_forecast_cached,
    render_forecasting,
)
from app.tabs_fuel import render_fuel_breakdown
from app.tabs_month import build_month_context, render_month_comparison
from app.tabs_weather_context import (
    build_weather_interpretations,
    load_weather_context,
)
from src.changepoints import detect_changepoints
from src.consultant import ConsultantContext
from src.weather import WeatherFetchError

st.set_page_config(page_title="AI Home Energy Intelligence Platform", layout="wide")


def main() -> None:
    st.title("AI Home Energy Intelligence Platform")
    st.caption(
        "An evidence-based energy analyst, not a statistics dashboard -- built on real energy provider "
        "monthly billing exports. Start with Home or 'How did this month compare?'; the other "
        "tabs hold the underlying analysis every conclusion there is traceable to."
    )

    (
        clean,
        report,
        selected_years,
        weather_enabled,
        horizon,
        model_choice,
        fuel,
        fuel_label,
        fuel_frames,
        fuel_cross_check_warnings,
        billing_config,
        weather_config,
    ) = render_sidebar()
    st.caption(f"**Fuel: {fuel_label}**")
    period_df = clean[clean["year"].isin(selected_years)].sort_values("month_start")
    if period_df.empty:
        st.warning("No data in the selected period. Adjust the year filter in the sidebar.")
        st.stop()
        return

    # One cached computation covers all three fuels; the selected fuel's results are pulled
    # from it rather than recomputed separately (previously the selected fuel's STL and
    # anomaly detection ran twice per rerun, uncached -- audit finding F8).
    (
        fuel_stl_results,
        fuel_stl_errors,
        fuel_anomalies_all,
        fuel_merged,
        fuel_energy_results,
        fuel_weather_errors,
    ) = compute_all_fuel_analysis(fuel_frames, weather_enabled, weather_config)

    stl_result = fuel_stl_results.get(fuel)
    stl_error = fuel_stl_errors.get(fuel)
    anomalies = fuel_anomalies_all.get(fuel, [])
    changepoints = detect_changepoints(stl_result.deseasonalized) if stl_result else []
    merged = fuel_merged.get(fuel) if weather_enabled else None
    energy_result = fuel_energy_results.get(fuel) if weather_enabled else None
    weather_error = fuel_weather_errors.get(fuel) if weather_enabled else None

    try:
        forecast_12mo, forecast_error = generate_forecast_cached(clean, 12, "auto"), None
    except ValueError as exc:
        forecast_12mo, forecast_error = None, str(exc)

    # Weather Context Engine (snow/wind/precipitation): computed once here so the Unusual
    # Months tab, the Weather Impact summary, and the Consultant all read the same objects.
    # Fuel-independent (weather is weather), and a disk-cache hit after the first fetch.
    weather_context_df = daily_weather = weather_interpretations = None
    if weather_enabled:
        try:
            weather_context_df, daily_weather = load_weather_context(clean, weather_config)
        except (WeatherFetchError, ValueError) as exc:
            st.sidebar.caption(f"Weather context unavailable: {exc}")
        if weather_context_df is not None and anomalies:
            weather_interpretations = build_weather_interpretations(
                anomalies, weather_context_df, energy_result, merged, fuel_anomalies_all, fuel_frames
            )

    # Owned here (not in session state) so every consumer sees the same, current-data-consistent
    # result: the Cost Intelligence button sets the request flag and reruns, and the cached
    # generator makes this a cache hit on every rerun after the first (audit finding F7).
    multi_fuel_forecasts = (
        generate_multi_fuel_forecast_cached(fuel_frames, horizon=12)
        if st.session_state.get("multi_fuel_forecast_requested")
        else None
    )

    analyst_report = build_report(
        clean,
        report,
        stl_result,
        merged,
        energy_result,
        changepoints,
        anomalies,
        forecast_12mo,
        fuel,
        fuel_frames,
        fuel_stl_results,
        fuel_energy_results,
        fuel_anomalies_all,
    )

    # The month-comparison state (selected month / fuel / mode) is resolved once here from
    # session state, so Home, the comparison page, and the Consultant describe the exact
    # same comparison objects on every rerun -- never stale, never diverging (audit F2/F7).
    month_ctx = build_month_context(
        fuel_frames,
        fuel_merged if weather_enabled else dict.fromkeys(fuel_frames),
        fuel_energy_results if weather_enabled else dict.fromkeys(fuel_frames),
        weather_enabled,
        billing_config=billing_config,
    )

    consultant_ctx = ConsultantContext(
        analyst_report=analyst_report,
        clean=clean,
        fuel_frames=fuel_frames,
        fuel_energy_results=fuel_energy_results,
        anomalies=anomalies,
        forecast_12mo=forecast_12mo,
        multi_fuel_forecasts=multi_fuel_forecasts,
        weather_enabled=weather_enabled,
        fuel_label=fuel_label,
        weather_interpretations=(
            {date: interp for date, (_cls, interp) in weather_interpretations.items()}
            if weather_interpretations
            else None
        ),
        selected_comparison_month=month_ctx.selected_month,
        comparison_mode=month_ctx.mode,
        comparison_fuel=month_ctx.fuel,
        fuel_merged=fuel_merged if weather_enabled else None,
        billing_config=billing_config,
    )

    # --- Household Energy Review download (sidebar, two-step) -----------------------------
    # Generate-on-click keeps the ~4.9 MB build (measured 0.17 s) off the per-rerun path.
    # The stored report carries a context fingerprint; if fuel/weather/data change, the
    # stored copy is discarded rather than served stale -- the session-state lesson from
    # audit findings F2/F7, applied preemptively.
    report_fingerprint = (
        fuel,
        weather_enabled,
        str(clean["month_start"].max()),
        len(clean),
        multi_fuel_forecasts is not None,
    )
    stored = st.session_state.get("household_report")
    if stored is not None and stored[0] != report_fingerprint:
        stored = None
        del st.session_state["household_report"]

    st.sidebar.divider()
    st.sidebar.subheader("Report")
    if stored is None:
        if st.sidebar.button("Generate Household Energy Review"):
            html = build_household_report_html(
                analyst_report,
                fuel_frames,
                fuel_merged,
                fuel_energy_results,
                forecast_12mo,
                weather_enabled,
                report,
            )
            st.session_state["household_report"] = (report_fingerprint, html)
            st.rerun()
        st.sidebar.caption(
            "Builds a self-contained HTML report (executive summary, findings, fuel mix, "
            "weather, forecast, recommendations, methodology, limitations). For weather and "
            "fuel-attribution sections, enable 'Weather adjustment' first."
        )
    else:
        period_slug = f"{clean['month_start'].min().year}-{clean['month_start'].max().year}"
        st.sidebar.download_button(
            "Download Household Energy Review (HTML)",
            data=stored[1],
            file_name=f"household_energy_review_{period_slug}.html",
            mime="text/html",
        )
        st.sidebar.caption(
            "Open in a browser; charts are interactive. Print → 'Save as PDF' produces the "
            "print-quality PDF version."
        )

    # Navigation is organized around the questions a homeowner actually asks; the
    # whole-period and method-level surfaces all remain, one level down (Long-term
    # trends / Data and methods) -- demoted, never deleted.
    (
        tab_home,
        tab_month,
        tab_drivers,
        tab_costs_carbon,
        tab_unusual,
        tab_forecast,
        tab_consultant,
        tab_long_term,
        tab_methods,
    ) = st.tabs(
        [
            "Home",
            "How did this month compare?",
            "What drives my usage?",
            "Costs and carbon",
            "Did anything unusual happen?",
            "What should I expect next?",
            "Ask the Energy Consultant",
            "Long-term trends",
            "Data and methods",
        ]
    )
    with tab_home:
        render_executive_briefing(
            clean, analyst_report, forecast_12mo, forecast_error, month_ctx, fuel, billing_config
        )
    with tab_month:
        render_month_comparison(month_ctx, fuel_frames, fuel_anomalies_all)
    with tab_drivers:
        sub_fuel, sub_seasons, sub_weather = st.tabs(
            ["Fuel breakdown", "Seasons", "Weather impact"]
        )
        with sub_fuel:
            render_fuel_breakdown(
                fuel_frames["electricity"], fuel_frames["gas"], fuel_cross_check_warnings or []
            )
        with sub_seasons:
            render_seasonality(stl_result, stl_error, clean, anomalies)
        with sub_weather:
            render_weather_adjustment(
                weather_enabled, merged, energy_result, weather_error, fuel, weather_context_df, weather_config
            )
    with tab_costs_carbon:
        sub_cost, sub_carbon = st.tabs(["Costs", "Carbon"])
        with sub_cost:
            render_cost_intelligence(fuel_frames, weather_enabled, billing_config)
        with sub_carbon:
            render_carbon(fuel_frames, fuel_merged, fuel_energy_results, weather_enabled)
    with tab_unusual:
        sub_anomalies, sub_shifts = st.tabs(["Unusual months", "Usage shifts"])
        with sub_anomalies:
            render_anomalies(
                clean,
                stl_error,
                anomalies,
                merged,
                energy_result,
                weather_interpretations,
                weather_context_df,
                daily_weather,
            )
        with sub_shifts:
            render_change_points(
                stl_result, stl_error, changepoints, clean, merged, energy_result, fuel
            )
    with tab_forecast:
        render_forecasting(clean, horizon, model_choice, fuel, billing_config)
    with tab_consultant:
        render_consultant(consultant_ctx)
    with tab_long_term:
        sub_consumption, sub_comparisons = st.tabs(
            ["Consumption over time", "Fuel comparisons (all years)"]
        )
        with sub_consumption:
            render_consumption_analysis(period_df)
        with sub_comparisons:
            render_comparisons(
                fuel_frames,
                fuel_stl_results,
                fuel_anomalies_all,
                fuel_merged,
                fuel_energy_results,
                weather_enabled,
                multi_fuel_forecasts,
            )
    with tab_methods:
        sub_analyst, sub_stats, sub_quality = st.tabs(
            ["Full report (AI Analyst)", "Statistical analysis", "Data quality"]
        )
        with sub_analyst:
            render_ai_analyst(clean, merged, energy_result, anomalies, analyst_report)
        with sub_stats:
            render_statistical_analysis(period_df)
        with sub_quality:
            render_data_quality(report, fuel_cross_check_warnings)


if __name__ == "__main__":
    main()
