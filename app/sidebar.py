"""Sidebar controls and data loading -- split out of ``streamlit_app.py`` to keep that module
under the ~300-line guideline as Phase 4 adds more tabs.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import streamlit as st

from config import SETTINGS, BillingConfig, WeatherConfig
from src.forecast_evaluation import MODEL_REGISTRY
from src.fuel import cross_check_fuel_totals
from src.geocoding import GeocodingError, LocationCandidate, search_location
from src.ingestion import (
    FUEL_FILE_PATTERNS,
    EnergyType,
    discover_csv_files,
    filter_sources_by_fuel,
    load_all,
)
from src.preprocessing import PreprocessingReport, run_pipeline

_FUEL_DISPLAY: dict[EnergyType, str] = {
    "total": "Total (Electricity + Gas)",
    "electricity": "Electricity only",
    "gas": "Gas only",
}

_WEATHER_LOCATION_KEY = "weather_confirmed_location"  # WeatherConfig, once the user confirms a match
_WEATHER_QUERY_KEY = "weather_location_query"


@st.cache_data(show_spinner="Loading data/raw/*.csv...")
def _load_default_data(pattern: str = FUEL_FILE_PATTERNS["total"]) -> tuple[pd.DataFrame, PreprocessingReport]:
    files = discover_csv_files(SETTINGS.raw_data_dir, pattern=pattern)
    raw = load_all(files)
    return run_pipeline(raw)


def _load_uploaded_data(uploaded_files) -> tuple[pd.DataFrame, PreprocessingReport]:
    raw = load_all(uploaded_files)
    return run_pipeline(raw)


@st.cache_data(show_spinner="Searching for location...")
def _search_location_cached(query: str) -> list[LocationCandidate]:
    return search_location(query)


def _confirm_location(candidate: LocationCandidate) -> None:
    """Apply a geocoding candidate as the confirmed weather location.

    Runs as a button ``on_click`` callback -- Streamlit executes callbacks
    before the script reruns from the top, which is the only point a
    widget's own session-state key can be reassigned (assigning it after
    the widget has already rendered in the current run raises
    ``StreamlitAPIException``).
    """
    st.session_state[_WEATHER_LOCATION_KEY] = dataclasses.replace(
        SETTINGS.weather,
        latitude=candidate.latitude,
        longitude=candidate.longitude,
        location_label=candidate.label,
        timezone=candidate.timezone,
    )
    st.session_state[_WEATHER_QUERY_KEY] = candidate.label


def _load_all_fuels(
    uploaded,
) -> tuple[dict[EnergyType, pd.DataFrame], dict[EnergyType, PreprocessingReport]]:
    """Load Total/Electricity/Gas independently (each cached separately by pattern).

    A fuel with zero matching files/uploads yields an empty ``clean`` frame
    -- already how ``load_all([])``/``run_pipeline`` degrade -- so it's
    simply excluded from the sidebar's fuel choices below rather than
    special-cased here.
    """
    frames: dict[EnergyType, pd.DataFrame] = {}
    reports: dict[EnergyType, PreprocessingReport] = {}
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
    EnergyType,
    str,
    dict[EnergyType, pd.DataFrame],
    list[str] | None,
    BillingConfig,
    WeatherConfig,
]:
    st.sidebar.title("Controls")
    st.sidebar.subheader("Data")
    uploaded = st.sidebar.file_uploader(
        "Upload your 'Total Use', 'Electricity Use', and/or 'Gas Use' CSV exports",
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
    st.sidebar.subheader("Tariff")
    electricity_standing_p = st.sidebar.number_input(
        "Electricity standing charge (p/day)",
        min_value=0.0,
        max_value=200.0,
        value=SETTINGS.billing.electricity_standing_gbp_per_day * 100,
        step=0.01,
        format="%.2f",
    )
    gas_standing_p = st.sidebar.number_input(
        "Gas standing charge (p/day)",
        min_value=0.0,
        max_value=200.0,
        value=SETTINGS.billing.gas_standing_gbp_per_day * 100,
        step=0.01,
        format="%.2f",
    )
    vat_pct = st.sidebar.number_input(
        "VAT rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=SETTINGS.billing.vat_rate * 100,
        step=0.5,
        format="%.1f",
    )
    billing_config = dataclasses.replace(
        SETTINGS.billing,
        electricity_standing_gbp_per_day=electricity_standing_p / 100,
        gas_standing_gbp_per_day=gas_standing_p / 100,
        vat_rate=vat_pct / 100,
    )
    st.sidebar.caption(
        "Defaults are this household's energy tariff (July 2026) -- adjust if your provider "
        "charges differently. Used everywhere a full bill is estimated (consumption + "
        "standing charge + VAT); the exports themselves carry consumption cost only."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Weather")

    confirmed_location: WeatherConfig = st.session_state.get(_WEATHER_LOCATION_KEY, SETTINGS.weather)
    if _WEATHER_QUERY_KEY not in st.session_state:
        st.session_state[_WEATHER_QUERY_KEY] = confirmed_location.location_label

    location_query = st.sidebar.text_input(
        "Location (for weather adjustment)",
        key=_WEATHER_QUERY_KEY,
        help=(
            "Weather adjustment fits historical temperature for this location against your "
            "consumption. Type a city/town and confirm the right match below -- place names "
            "are often ambiguous (there are Manchesters in England, New Hampshire, and "
            "Tennessee), so a search result is never applied automatically."
        ),
    )
    location_confirmed = location_query == confirmed_location.location_label

    if not location_confirmed and location_query.strip():
        try:
            candidates = _search_location_cached(location_query)
        except GeocodingError as exc:
            candidates = []
            st.sidebar.error(str(exc))
        if candidates:
            options = {c.label: c for c in candidates}
            chosen_label = st.sidebar.radio(
                "Confirm the match", list(options), key="weather_candidate_choice"
            )
            # Widgets keyed to _WEATHER_QUERY_KEY can't be reassigned once instantiated in the
            # same run -- the mutation must happen in an on_click callback, which runs *before*
            # the script reruns (and the widget is re-created) from the top.
            st.sidebar.button(
                "Confirm location", on_click=_confirm_location, args=(options[chosen_label],)
            )
        else:
            st.sidebar.caption(f"No matches found for {location_query!r}.")

    weather_config: WeatherConfig = st.session_state.get(_WEATHER_LOCATION_KEY, SETTINGS.weather)
    weather_enabled = st.sidebar.toggle("Weather adjustment", value=False)
    pending_note = "" if location_confirmed else " (pending confirmation above)"
    st.sidebar.caption(
        f"Location: {weather_config.location_label}{pending_note}. Turning this on fetches "
        "historical temperature from Open-Meteo (free, no key) for this location -- cached "
        "to disk after the first fetch. [Open-Meteo](https://open-meteo.com/)."
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
        billing_config,
        weather_config,
    )
