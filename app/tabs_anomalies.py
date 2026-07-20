"""Unusual Months tab renderer: cross-referenced anomaly detection.

Anomaly detection is computed once in ``streamlit_app.main()`` (not here)
since the Executive Briefing and AI Analyst pages need the same list --
see ``app/tabs_drivers.py``'s module docstring for why.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts import anomaly_scatter
from app.tabs_weather_context import render_severe_weather_sections
from src.anomalies import Anomaly, interpret_anomalies
from src.energy_signature import EnergySignatureResult
from src.investigation import build_investigation_checklist
from src.weather_context import WeatherContextClassification
from src.weather_interpretation import UnusualMonthInterpretation


def render_anomalies(
    clean: pd.DataFrame,
    stl_error: str | None,
    anomalies: list[Anomaly],
    merged: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    weather_interpretations: dict[
        pd.Timestamp, tuple[WeatherContextClassification, UnusualMonthInterpretation]
    ]
    | None = None,
    weather_context_df: pd.DataFrame | None = None,
    daily_weather: pd.DataFrame | None = None,
) -> None:
    st.caption(
        "Detected on the deseasonalized STL residual, so a normal winter isn't mistaken for "
        "an anomaly. Uses full history regardless of the sidebar year filter."
    )
    if stl_error:
        st.warning(stl_error)
        return

    series = clean.set_index("month_start")["consumption_kwh"]
    st.plotly_chart(anomaly_scatter(series, anomalies), width="stretch")
    st.write(interpret_anomalies(anomalies))

    if not anomalies:
        st.caption(
            "Three independent methods (rolling z-score, STL-residual ESD, Isolation Forest) are "
            "cross-referenced -- a month flagged by 2 or more is meaningfully more likely to be "
            "real than one flagged by a single method."
        )
        return

    rows = [
        {
            "Date": a.date.strftime("%B %Y"),
            "Methods": ", ".join(a.methods),
            "Direction": a.direction,
            "Context": a.rank_context,
        }
        for a in anomalies
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "At ~35 months of history, a single method's nominal confidence level doesn't translate "
        "to its real-world false-positive rate (documented in src/esd.py for the ESD method "
        "specifically) -- treat single-method flags as worth a look, not statistically certain."
    )

    for a in anomalies:
        checklist = build_investigation_checklist(a.date, clean, merged, energy_result)
        with st.expander(f"Possible causes: {a.date.strftime('%B %Y')}"):
            for item in checklist.items:
                marker = "✅" if item.checked else "⬜"
                st.write(f"{marker} **{item.label}** -- {item.reason}")

    if weather_interpretations and weather_context_df is not None and daily_weather is not None:
        st.divider()
        render_severe_weather_sections(anomalies, weather_interpretations, weather_context_df, daily_weather)
    elif anomalies:
        st.caption(
            "Turn on 'Weather adjustment' in the sidebar to also see each flagged month's "
            "severe-weather context (snow, strong wind, heavy rain)."
        )
