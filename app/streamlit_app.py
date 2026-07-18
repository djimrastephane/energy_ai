"""AI Home Energy Intelligence Platform -- Streamlit app.

Phase 1: data ingestion/validation, descriptive statistics, core KPIs.
Phase 2: STL seasonality/trend decomposition, weather-adjusted "energy
signature" regression (Open-Meteo, opt-in via the sidebar toggle since it
needs a network call), and change-point detection. Phase 3: an 8-model
forecasting suite compared by walk-forward cross-validation with
auto-selection and P10/P50/P90 bands, plus anomaly detection (rolling
z-score, STL-residual ESD, Isolation Forest). Decision-support layer: a
findings/recommendations/confidence engine (``src.report``) turns those
outputs into an Executive Briefing and a full AI Analyst report -- no new
statistics, purely synthesis of what's already computed. Phase 4: multi-fuel
comparisons, cost intelligence, carbon estimates, and UK/Scotland
benchmarking (``src.comparisons``, ``src.cross_fuel_anomalies``,
``src.cost_engine``, ``src.carbon``, ``src.benchmarking``) -- the sidebar's
Fuel selector already made every earlier module generic over Electricity/
Gas/Total, so Phase 4 only adds what's genuinely new: comparing those
already-computed per-fuel results against each other.

Change-point detection, anomaly detection, and (if enabled) the weather fit
are computed once here in ``main()`` for the selected fuel (and, via
``app/multi_fuel.py``, for all three fuels), not inside individual tab
renderers, since the Executive Briefing, AI Analyst, Comparisons, Cost
Intelligence, and Carbon pages all need the exact same objects -- computing
them twice per fuel would let the surfaces silently drift out of sync.

Sidebar controls live in ``app/sidebar.py``. Tab rendering logic lives in
``tabs_core.py`` (Phase 1), ``tabs_phase2.py`` (Phase 2), ``tabs_phase3.py``
(Phase 3), ``tabs_briefing.py``/``tabs_analyst.py`` (decision-support), and
``tabs_fuel.py``/``tabs_comparisons.py``/``tabs_cost.py``/``tabs_carbon.py``
(Phase 4) -- this module is just top-level orchestration, kept under ~300
lines as the app grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in (PROJECT_ROOT, APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402
from multi_fuel import compute_all_fuel_analysis  # noqa: E402
from sidebar import render_sidebar  # noqa: E402
from tabs_analyst import build_report, render_ai_analyst  # noqa: E402
from tabs_briefing import render_executive_briefing  # noqa: E402
from tabs_carbon import render_carbon  # noqa: E402
from tabs_comparisons import render_comparisons  # noqa: E402
from tabs_consultant import render_consultant  # noqa: E402
from tabs_core import (  # noqa: E402
    render_consumption_analysis,
    render_data_quality,
    render_statistical_analysis,
)
from tabs_cost import render_cost_intelligence  # noqa: E402
from tabs_fuel import render_fuel_breakdown  # noqa: E402
from tabs_phase2 import (  # noqa: E402
    load_weather_analysis,
    render_change_points,
    render_seasonality,
    render_weather_adjustment,
)
from tabs_phase3 import generate_forecast_cached, render_anomalies, render_forecasting  # noqa: E402

from src.anomalies import detect_anomalies  # noqa: E402
from src.changepoints import detect_changepoints  # noqa: E402
from src.consultant import ConsultantContext  # noqa: E402
from src.decomposition import stl_decompose  # noqa: E402
from src.weather import WeatherFetchError  # noqa: E402

st.set_page_config(page_title="AI Home Energy Intelligence Platform", layout="wide")


def main() -> None:
    st.title("AI Home Energy Intelligence Platform")
    st.caption(
        "An evidence-based energy analyst, not a statistics dashboard -- built on real OVO Energy "
        "monthly billing exports. Start with Executive Summary or AI Analyst; the other tabs hold "
        "the underlying analysis every conclusion there is traceable to."
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
    ) = render_sidebar()
    st.caption(f"**Fuel: {fuel_label}**")
    period_df = clean[clean["year"].isin(selected_years)].sort_values("month_start")
    if period_df.empty:
        st.warning("No data in the selected period. Adjust the year filter in the sidebar.")
        st.stop()
        return

    try:
        stl_result, stl_error = stl_decompose(clean), None
    except ValueError as exc:
        stl_result, stl_error = None, str(exc)

    changepoints = detect_changepoints(stl_result.deseasonalized) if stl_result else []
    anomalies = detect_anomalies(clean, stl_result) if stl_result else []

    merged, energy_result, weather_error = None, None, None
    if weather_enabled:
        try:
            merged, energy_result = load_weather_analysis(clean)
        except (WeatherFetchError, ValueError) as exc:
            weather_error = str(exc)

    try:
        forecast_12mo, forecast_error = generate_forecast_cached(clean, 12, "auto"), None
    except ValueError as exc:
        forecast_12mo, forecast_error = None, str(exc)

    fuel_stl_results, fuel_anomalies_all, fuel_merged, fuel_energy_results = compute_all_fuel_analysis(
        fuel_frames, weather_enabled
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
    consultant_ctx = ConsultantContext(
        analyst_report=analyst_report,
        clean=clean,
        fuel_frames=fuel_frames,
        fuel_energy_results=fuel_energy_results,
        anomalies=anomalies,
        forecast_12mo=forecast_12mo,
        multi_fuel_forecasts=st.session_state.get("multi_fuel_forecasts"),
        weather_enabled=weather_enabled,
    )

    (
        tab_summary,
        tab_consultant,
        tab_analyst,
        tab_consumption,
        tab_fuel,
        tab_comparisons,
        tab_cost,
        tab_carbon,
        tab_stats,
        tab_seasonality,
        tab_weather,
        tab_changepoints,
        tab_forecast,
        tab_anomalies,
        tab_quality,
    ) = st.tabs(
        [
            "Executive Summary",
            "AI Consultant",
            "AI Analyst",
            "Consumption Analysis",
            "Fuel Breakdown",
            "Comparisons",
            "Cost Intelligence",
            "Carbon",
            "Statistical Analysis",
            "Seasonality & Trend",
            "Weather Adjustment",
            "Change Points",
            "Forecasting",
            "Anomaly Detection",
            "Data Quality",
        ]
    )
    with tab_summary:
        render_executive_briefing(clean, analyst_report, forecast_12mo, forecast_error)
    with tab_consultant:
        render_consultant(consultant_ctx)
    with tab_analyst:
        render_ai_analyst(clean, merged, energy_result, anomalies, analyst_report)
    with tab_consumption:
        render_consumption_analysis(period_df)
    with tab_fuel:
        render_fuel_breakdown(
            fuel_frames["electricity"], fuel_frames["gas"], fuel_cross_check_warnings or []
        )
    with tab_cost:
        render_cost_intelligence(fuel_frames, weather_enabled)
    with tab_comparisons:
        render_comparisons(
            fuel_frames,
            fuel_stl_results,
            fuel_anomalies_all,
            fuel_merged,
            fuel_energy_results,
            weather_enabled,
            st.session_state.get("multi_fuel_forecasts"),
        )
    with tab_carbon:
        render_carbon(fuel_frames, fuel_merged, fuel_energy_results, weather_enabled)
    with tab_stats:
        render_statistical_analysis(period_df)
    with tab_seasonality:
        render_seasonality(stl_result, stl_error)
    with tab_weather:
        render_weather_adjustment(weather_enabled, merged, energy_result, weather_error, fuel)
    with tab_changepoints:
        render_change_points(stl_result, stl_error, changepoints, clean, merged, energy_result, fuel)
    with tab_forecast:
        render_forecasting(clean, horizon, model_choice)
    with tab_anomalies:
        render_anomalies(clean, stl_error, anomalies, merged, energy_result)
    with tab_quality:
        render_data_quality(report, fuel_cross_check_warnings)


if __name__ == "__main__":
    main()
