"""Weather-context UI: loading, interpretation-building, and render helpers.

Split into its own module (rather than growing ``tabs_phase2.py``/
``tabs_phase3.py`` past the ~300-line guideline). The Unusual Months tab
renders per-anomaly weather context via these helpers; the Weather Impact
tab renders the history-wide severe-weather summary. All statistics come
from :mod:`src.weather_context`/:mod:`src.weather_interpretation` -- this
module only lays them out.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SETTINGS
from src.anomalies import Anomaly
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType
from src.weather import fetch_daily_weather
from src.weather_context import (
    WeatherContextClassification,
    add_relative_severity,
    aggregate_monthly_weather_context,
    classify_monthly_weather_context,
)
from src.weather_interpretation import UnusualMonthInterpretation, interpret_unusual_month

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


@st.cache_data(show_spinner="Computing weather context (snow, wind, precipitation)...")
def load_weather_context(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly context with severity columns, daily weather) for the consumption range.

    The underlying fetch is the same disk-cached call the energy-signature path uses, so
    on any ordinary rerun this is a cache hit at both layers (Streamlit's and the disk's)
    -- no network request and no recomputation.
    """
    w = SETTINGS.weather
    start = clean["month_start"].min()
    end = clean["month_start"].max() + pd.offsets.MonthEnd(1)
    daily = fetch_daily_weather(w.latitude, w.longitude, start, end, w.timezone, SETTINGS.weather_cache_dir)
    context = add_relative_severity(aggregate_monthly_weather_context(daily))
    return context, daily


def build_weather_interpretations(
    anomalies: list[Anomaly],
    weather_context_df: pd.DataFrame,
    energy_result: EnergySignatureResult | None,
    merged: pd.DataFrame | None,
    fuel_anomalies_all: dict[EnergyType, list[Anomaly]],
    fuel_frames: dict[EnergyType, pd.DataFrame],
) -> dict[pd.Timestamp, tuple[WeatherContextClassification, UnusualMonthInterpretation]]:
    """Classify + interpret every flagged month once, in ``main()``, so the Unusual Months
    tab and the Consultant read the exact same objects (the project's single-computation
    pattern). Fuel flags are ``None`` when that fuel's exports aren't loaded -- honestly
    unknown rather than assumed unchanged."""
    has_elec = not fuel_frames.get("electricity", pd.DataFrame()).empty
    has_gas = not fuel_frames.get("gas", pd.DataFrame()).empty
    elec_dates = {a.date for a in fuel_anomalies_all.get("electricity", [])}
    gas_dates = {a.date for a in fuel_anomalies_all.get("gas", [])}

    out: dict[pd.Timestamp, tuple[WeatherContextClassification, UnusualMonthInterpretation]] = {}
    for anomaly in anomalies:
        classification = classify_monthly_weather_context(weather_context_df, anomaly.date)
        interpretation = interpret_unusual_month(
            anomaly,
            classification,
            energy_result,
            merged,
            electricity_flagged=(anomaly.date in elec_dates) if has_elec else None,
            gas_flagged=(anomaly.date in gas_dates) if has_gas else None,
        )
        out[anomaly.date] = (classification, interpretation)
    return out


def render_anomaly_weather_context(
    classification: WeatherContextClassification, interpretation: UnusualMonthInterpretation
) -> None:
    """One flagged month's homeowner-facing weather context: facts, meaning, what remains
    unexplained, confidence + limitation. No hover needed for any conclusion."""
    st.markdown(f"**What the weather was like** -- {classification.label}")
    for fact in classification.facts:
        st.write(f"- {fact}")
    st.markdown("**What this may mean**")
    st.write(interpretation.headline)
    if interpretation.residual_monthly_kwh is not None:
        st.markdown("**What remains unexplained**")
        st.write(
            f"After adjusting for temperature, this month came in "
            f"{interpretation.residual_monthly_kwh:+,.0f} kWh versus the model's expectation."
        )
    icon = _CONFIDENCE_ICON[interpretation.confidence]
    st.caption(f"Confidence: {icon} {interpretation.confidence}. {interpretation.limitation}")


def render_severe_weather_sections(
    anomalies: list[Anomaly],
    interpretations: dict[pd.Timestamp, tuple[WeatherContextClassification, UnusualMonthInterpretation]],
    weather_context_df: pd.DataFrame,
    daily_weather: pd.DataFrame,
) -> None:
    """The Unusual Months tab's weather-context block: the top flagged month inline (the
    conclusion a homeowner most needs is never behind a click), the rest in expanders,
    and raw tables/thresholds in a single Advanced details expander."""
    flagged = [a for a in anomalies if a.date in interpretations]
    if not flagged:
        return
    st.subheader("Severe weather context")
    first, *rest = flagged
    st.markdown(f"#### {first.date.strftime('%B %Y')}")
    render_anomaly_weather_context(*interpretations[first.date])
    for anomaly in rest:
        with st.expander(f"Weather context: {anomaly.date.strftime('%B %Y')}"):
            render_anomaly_weather_context(*interpretations[anomaly.date])
    _render_advanced_weather_details(weather_context_df, daily_weather)


def _render_advanced_weather_details(weather_context_df: pd.DataFrame, daily_weather: pd.DataFrame) -> None:
    t = SETTINGS.weather_context
    with st.expander("Advanced weather details (all months, thresholds, severe days)"):
        display_cols = [
            "month_start", "snow_days", "heavy_snow_days", "snowfall_total_cm", "heavy_rain_days",
            "precipitation_total_mm", "strong_wind_days", "severe_gust_days", "max_wind_gust_kmh",
            "longest_severe_weather_run_days", "weather_context_complete",
        ]
        display = weather_context_df[display_cols].assign(
            month_start=weather_context_df["month_start"].dt.strftime("%B %Y")
        ).rename(
            columns={
                "month_start": "Month",
                "snow_days": "Snow days",
                "heavy_snow_days": "Heavy-snow days",
                "snowfall_total_cm": "Snowfall (cm)",
                "heavy_rain_days": "Heavy-rain days",
                "precipitation_total_mm": "Precipitation (mm)",
                "strong_wind_days": "Strong-wind days",
                "severe_gust_days": "Severe-gust days",
                "max_wind_gust_kmh": "Max gust (km/h)",
                "longest_severe_weather_run_days": "Longest severe run (days)",
                "weather_context_complete": "Complete coverage",
            }
        ).round(1)
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption(
            f"Day thresholds (config.py, documented conservative UK conventions): snow day > "
            f"{t.snow_day_cm:.0f} cm; heavy snow >= {t.heavy_snow_day_cm:.0f} cm; heavy rain >= "
            f"{t.heavy_rain_day_mm:.0f} mm; strong wind >= {t.strong_wind_day_kmh:.0f} km/h sustained; "
            f"severe gust >= {t.severe_gust_day_kmh:.0f} km/h. Severity bands compare each month "
            "against this household's own history (strict percentile rank + robust IQR) -- no "
            "statistical significance is claimed at this sample size."
        )
        severe_days = daily_weather[
            (daily_weather["snowfall_cm"] >= t.heavy_snow_day_cm)
            | (daily_weather["precipitation_mm"] >= t.heavy_rain_day_mm)
            | (daily_weather["wind_speed_max_kmh"] >= t.strong_wind_day_kmh)
            | (daily_weather["wind_gust_max_kmh"] >= t.severe_gust_day_kmh)
        ]
        if not severe_days.empty:
            st.markdown(f"**Individual severe-weather days ({len(severe_days)}):**")
            st.dataframe(
                severe_days.assign(date=severe_days["date"].dt.strftime("%Y-%m-%d")).rename(
                    columns={
                        "date": "Date",
                        "temp_mean_c": "Mean temp (°C)",
                        "snowfall_cm": "Snowfall (cm)",
                        "snow_depth_m": "Snow depth (m)",
                        "precipitation_mm": "Precipitation (mm)",
                        "wind_speed_max_kmh": "Max wind (km/h)",
                        "wind_gust_max_kmh": "Max gust (km/h)",
                    }
                ).round(1),
                hide_index=True,
                width="stretch",
            )


def render_weather_impact_summary(weather_context_df: pd.DataFrame) -> None:
    """Compact history-wide severe-weather summary for the Weather Impact tab."""
    notable = [
        classify_monthly_weather_context(weather_context_df, month)
        for month in weather_context_df["month_start"]
    ]
    notable = [c for c in notable if c.is_notable]
    st.subheader("Severe weather in your history")
    if not notable:
        st.write(
            "No months in your history stand out for snow, wind, or heavy rain against "
            "what's usual for this location."
        )
        return
    rows = [
        {
            "Month": c.month_start.strftime("%B %Y"),
            "Classification": c.label,
            "Headline fact": c.facts[0] if c.facts else "",
        }
        for c in notable
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Context for the Unusual Months tab, not part of the temperature model: the "
        "energy-signature regression stays driven by degree days alone."
    )
