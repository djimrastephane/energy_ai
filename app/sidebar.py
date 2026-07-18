"""Sidebar controls and data loading -- split out of ``streamlit_app.py`` to keep that module
under the ~300-line guideline as Phase 4 adds more tabs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SETTINGS
from src.forecast_evaluation import MODEL_REGISTRY
from src.fuel import cross_check_fuel_totals
from src.ingestion import FUEL_FILE_PATTERNS, discover_csv_files, filter_sources_by_fuel, load_all
from src.preprocessing import PreprocessingReport, run_pipeline

_FUEL_DISPLAY = {
    "total": "Total (Electricity + Gas)",
    "electricity": "Electricity only",
    "gas": "Gas only",
}


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
