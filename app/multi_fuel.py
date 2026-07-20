"""Computes STL/anomalies/weather-regression for all three fuels (Total, Electricity, Gas),
reusing the exact same functions ``main()`` uses -- called once per fuel. This powers the
household-level Comparisons/Cost Intelligence/Carbon tabs, the AI Analyst's per-fuel sections,
and (since the audit) the selected fuel's own analysis in ``main()``, which previously
recomputed the same STL/anomaly results a second time on every rerun (audit finding F8).

The whole computation is ``st.cache_data``-cached: anomaly detection alone (three Isolation
Forest fits) measured at ~160 ms per rerun uncached, which was ~40% of warm interaction
latency. Results are deterministic (fixed seeds throughout), so caching cannot change behavior.

Forecasting is deliberately NOT computed here -- walk-forward CV across up to 8 models is the
one genuinely expensive step in this pipeline, so the multi-fuel forecast comparison stays an
opt-in button (see ``tabs_forecast.generate_multi_fuel_forecast_cached``).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.tabs_drivers import load_weather_analysis
from src.anomalies import Anomaly, detect_anomalies
from src.decomposition import STLResult, stl_decompose
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType
from src.weather import WeatherFetchError

FuelAnalysis = tuple[
    dict[EnergyType, STLResult | None],
    dict[EnergyType, str],  # per-fuel STL error messages (surfaced in the UI, not swallowed)
    dict[EnergyType, list[Anomaly]],
    dict[EnergyType, pd.DataFrame],
    dict[EnergyType, EnergySignatureResult],
    dict[EnergyType, str],  # per-fuel weather error messages
]


@st.cache_data(show_spinner="Analyzing all fuels...")
def compute_all_fuel_analysis(
    fuel_frames: dict[EnergyType, pd.DataFrame], weather_enabled: bool
) -> FuelAnalysis:
    """Run STL + anomaly detection for every fuel with data, and (if ``weather_enabled``) the
    weather regression too. Reuses ``stl_decompose``/``detect_anomalies``/``load_weather_analysis``
    unmodified -- no new analysis, just called once per fuel. Error messages are captured per
    fuel rather than swallowed, so ``main()`` can surface the selected fuel's error exactly as
    it did when it computed these itself.
    """
    fuel_stl: dict[EnergyType, STLResult | None] = {}
    fuel_stl_errors: dict[EnergyType, str] = {}
    fuel_anomalies: dict[EnergyType, list[Anomaly]] = {}
    fuel_merged: dict[EnergyType, pd.DataFrame] = {}
    fuel_energy_results: dict[EnergyType, EnergySignatureResult] = {}
    fuel_weather_errors: dict[EnergyType, str] = {}

    for fuel, df in fuel_frames.items():
        if df.empty:
            continue
        try:
            fuel_stl[fuel] = stl_decompose(df)
        except ValueError as exc:
            fuel_stl[fuel] = None
            fuel_stl_errors[fuel] = str(exc)
        fuel_anomalies[fuel] = detect_anomalies(df, fuel_stl[fuel]) if fuel_stl[fuel] else []

        if weather_enabled:
            try:
                merged, result = load_weather_analysis(df)
                fuel_merged[fuel] = merged
                fuel_energy_results[fuel] = result
            except (WeatherFetchError, ValueError) as exc:
                fuel_weather_errors[fuel] = str(exc)

    return fuel_stl, fuel_stl_errors, fuel_anomalies, fuel_merged, fuel_energy_results, fuel_weather_errors
