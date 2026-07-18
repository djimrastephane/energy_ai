"""Eagerly computes STL/anomalies/weather-regression for all three fuels (Total, Electricity,
Gas), reusing the exact same functions ``main()`` already calls for the single selected fuel --
just called for the other fuels too. This is what powers the household-level Comparisons/Cost
Intelligence/Carbon tabs and the AI Analyst's per-fuel sections. Kept in its own module so
``streamlit_app.py`` doesn't grow past the ~300-line guideline as Phase 4 adds more tabs.

Forecasting is deliberately NOT computed here -- walk-forward CV across up to 8 models is the
one genuinely expensive step in this pipeline, so the multi-fuel forecast comparison stays an
opt-in button (see ``tabs_phase3.generate_multi_fuel_forecast_cached``) rather than running
eagerly on every page load for all three fuels.
"""

from __future__ import annotations

import pandas as pd
from tabs_phase2 import load_weather_analysis

from src.anomalies import Anomaly, detect_anomalies
from src.decomposition import STLResult, stl_decompose
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType
from src.weather import WeatherFetchError


def compute_all_fuel_analysis(
    fuel_frames: dict[EnergyType, pd.DataFrame], weather_enabled: bool
) -> tuple[
    dict[EnergyType, STLResult | None],
    dict[EnergyType, list[Anomaly]],
    dict[EnergyType, pd.DataFrame],
    dict[EnergyType, EnergySignatureResult],
]:
    """Run STL + anomaly detection for every fuel with data, and (if ``weather_enabled``) the
    weather regression too. Reuses ``stl_decompose``/``detect_anomalies``/``load_weather_analysis``
    unmodified -- no new analysis, just called once per fuel instead of once for the selected
    fuel only. ``load_weather_analysis`` is already ``@st.cache_data``-wrapped, so re-calling it
    for a fuel whose data hasn't changed (e.g. the currently-selected one, already computed
    elsewhere in ``main()``) is a cache hit, not a real recomputation.
    """
    fuel_stl: dict[EnergyType, STLResult | None] = {}
    fuel_anomalies: dict[EnergyType, list[Anomaly]] = {}
    fuel_merged: dict[EnergyType, pd.DataFrame] = {}
    fuel_energy_results: dict[EnergyType, EnergySignatureResult] = {}

    for fuel, df in fuel_frames.items():
        if df.empty:
            continue
        try:
            fuel_stl[fuel] = stl_decompose(df)
        except ValueError:
            fuel_stl[fuel] = None
        fuel_anomalies[fuel] = detect_anomalies(df, fuel_stl[fuel]) if fuel_stl[fuel] else []

        if weather_enabled:
            try:
                merged, result = load_weather_analysis(df)
                fuel_merged[fuel] = merged
                fuel_energy_results[fuel] = result
            except (WeatherFetchError, ValueError):
                pass

    return fuel_stl, fuel_anomalies, fuel_merged, fuel_energy_results
