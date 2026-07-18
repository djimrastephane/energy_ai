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
statistics, purely synthesis of what's already computed. Sidebar controls
for tariffs and reporting are still shown but disabled, labelled with the
phase that will implement them.

Change-point detection, anomaly detection, and (if enabled) the weather fit
are computed once here in ``main()``, not inside individual tab renderers,
since the Executive Briefing and AI Analyst pages need the exact same
objects the individual analysis tabs use -- computing them twice would let
the two surfaces silently drift out of sync.

Tab rendering logic lives in ``tabs_core.py`` (Phase 1), ``tabs_phase2.py``
(Phase 2), ``tabs_phase3.py`` (Phase 3), ``tabs_briefing.py`` and
``tabs_analyst.py`` (decision-support) -- this module is just sidebar +
data loading + orchestration, kept under ~300 lines as the app grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in (PROJECT_ROOT, APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from tabs_analyst import build_report, render_ai_analyst  # noqa: E402
from tabs_briefing import render_executive_briefing  # noqa: E402
from tabs_core import (  # noqa: E402
    render_consumption_analysis,
    render_data_quality,
    render_statistical_analysis,
)
from tabs_fuel import render_fuel_breakdown  # noqa: E402
from tabs_phase2 import (  # noqa: E402
    load_weather_analysis,
    render_change_points,
    render_seasonality,
    render_weather_adjustment,
)
from tabs_phase3 import generate_forecast_cached, render_anomalies, render_forecasting  # noqa: E402

from config import SETTINGS  # noqa: E402
from src.anomalies import detect_anomalies  # noqa: E402
from src.changepoints import detect_changepoints  # noqa: E402
from src.decomposition import stl_decompose  # noqa: E402
from src.forecast_evaluation import MODEL_REGISTRY  # noqa: E402
from src.fuel import cross_check_fuel_totals  # noqa: E402
from src.ingestion import (  # noqa: E402
    FUEL_FILE_PATTERNS,
    discover_csv_files,
    filter_sources_by_fuel,
    load_all,
)
from src.preprocessing import PreprocessingReport, run_pipeline  # noqa: E402
from src.weather import WeatherFetchError  # noqa: E402

_FUEL_DISPLAY = {
    "total": "Total (Electricity + Gas)",
    "electricity": "Electricity only",
    "gas": "Gas only",
}

st.set_page_config(page_title="AI Home Energy Intelligence Platform", layout="wide")


@st.cache_data(show_spinner="Loading data/raw/*.csv...")
def _load_default_data(pattern: str = FUEL_FILE_PATTERNS["total"]) -> tuple[pd.DataFrame, PreprocessingReport]:
    files = discover_csv_files(SETTINGS.raw_data_dir, pattern=pattern)
    raw = load_all(files)
    return run_pipeline(raw)


def _load_uploaded_data(uploaded_files) -> tuple[pd.DataFrame, PreprocessingReport]:
    raw = load_all(uploaded_files)
    return run_pipeline(raw)


def _load_all_fuels(uploaded) -> tuple[dict[str, pd.DataFrame], dict[str, PreprocessingReport]]:
    """Load Total/Electricity/Gas independently (each cached separately by pattern).

    A fuel with zero matching files/uploads yields an empty ``clean`` frame
    -- already how ``load_all([])``/``run_pipeline`` degrade -- so it's
    simply excluded from the sidebar's fuel choices below rather than
    special-cased here.
    """
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, PreprocessingReport] = {}
    for fuel, pattern in FUEL_FILE_PATTERNS.items():
        if uploaded:
            frames[fuel], reports[fuel] = _load_uploaded_data(filter_sources_by_fuel(uploaded, fuel))
        else:
            frames[fuel], reports[fuel] = _load_default_data(pattern)
    return frames, reports


def render_sidebar() -> tuple[
    pd.DataFrame,
    PreprocessingReport,
    list[int],
    bool,
    int,
    str,
    str,
    str,
    dict[str, pd.DataFrame],
    list[str] | None,
]:
    st.sidebar.title("Controls")
    st.sidebar.subheader("Data")
    uploaded = st.sidebar.file_uploader(
        "Upload OVO 'Total Use', 'Electricity Use', and/or 'Gas Use' CSV exports",
        type="csv",
        accept_multiple_files=True,
    )

    try:
        fuel_frames, fuel_reports = _load_all_fuels(uploaded)
    except Exception as exc:  # noqa: BLE001 -- ingestion errors must not crash the app
        st.sidebar.error(f"Could not load data: {exc}")
        st.stop()
        raise

    available_fuels = [f for f in _FUEL_DISPLAY if not fuel_frames[f].empty]
    if not available_fuels:
        st.sidebar.error("No valid data loaded.")
        st.stop()

    st.sidebar.subheader("Fuel")
    fuel = st.sidebar.selectbox(
        "Fuel to analyze", available_fuels, format_func=lambda f: _FUEL_DISPLAY[f]
    )
    if len(available_fuels) == 1:
        st.sidebar.caption(
            "Add 'Electricity Use'/'Gas Use' exports to data/raw/ (or upload them above) to "
            "unlock fuel-level analysis."
        )

    clean, report = fuel_frames[fuel], fuel_reports[fuel]
    if uploaded:
        st.sidebar.success(f"Loaded {report.n_files_loaded} uploaded file(s) for {_FUEL_DISPLAY[fuel]}")
    else:
        st.sidebar.caption(f"Using {report.n_files_loaded} file(s) from data/raw/ ({_FUEL_DISPLAY[fuel]})")
        if st.sidebar.button("Reload data/raw/"):
            _load_default_data.clear()
            st.rerun()

    fuel_cross_check_warnings: list[str] | None = None
    if all(not fuel_frames[f].empty for f in ("total", "electricity", "gas")):
        fuel_cross_check_warnings = cross_check_fuel_totals(
            fuel_frames["total"], fuel_frames["electricity"], fuel_frames["gas"]
        )

    st.sidebar.subheader("Analysis period")
    years = sorted(clean["year"].unique().tolist())
    selected_years = st.sidebar.multiselect("Years to include", years, default=years)

    st.sidebar.divider()
    st.sidebar.subheader("Weather")
    weather_enabled = st.sidebar.toggle("Weather adjustment", value=False)
    st.sidebar.caption(
        f"Location: {SETTINGS.weather.location_label}. Off by default since turning it on "
        "fetches historical temperature from Open-Meteo (free, no key) -- cached to disk "
        "after the first fetch. [Open-Meteo](https://open-meteo.com/)."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Forecasting")
    horizon = st.sidebar.slider("Forecast horizon (months)", 1, 24, 12)
    model_choice = st.sidebar.selectbox("Forecasting model", ["Auto (best by CV)", *MODEL_REGISTRY])
    st.sidebar.caption(
        "'Auto' picks the model with the lowest cross-validated error; forcing a specific "
        "model is useful to inspect how it individually performs."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Coming in later phases")
    st.sidebar.selectbox("Tariff", ["Detected from billing data"], disabled=True)
    st.sidebar.caption("Tariff selection -- Phase 4")
    st.sidebar.button("Download report (PDF / HTML / Excel)", disabled=True)
    st.sidebar.caption("Reporting -- Phase 6")

    return (
        clean,
        report,
        selected_years,
        weather_enabled,
        horizon,
        model_choice,
        fuel,
        _FUEL_DISPLAY[fuel],
        fuel_frames,
        fuel_cross_check_warnings,
    )


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

    analyst_report = build_report(
        clean, report, stl_result, merged, energy_result, changepoints, anomalies, forecast_12mo, fuel
    )

    (
        tab_summary,
        tab_analyst,
        tab_consumption,
        tab_fuel,
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
            "AI Analyst",
            "Consumption Analysis",
            "Fuel Breakdown",
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
    with tab_analyst:
        render_ai_analyst(clean, merged, energy_result, anomalies, analyst_report)
    with tab_consumption:
        render_consumption_analysis(period_df)
    with tab_fuel:
        render_fuel_breakdown(
            fuel_frames["electricity"], fuel_frames["gas"], fuel_cross_check_warnings or []
        )
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
