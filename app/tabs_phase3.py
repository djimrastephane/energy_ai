"""Phase 3 tab renderers: Forecasting, Anomaly Detection.

Split out to match the ``tabs_core.py``/``tabs_phase2.py`` pattern.
Anomaly detection is computed once in ``streamlit_app.main()`` (not here)
since the Executive Briefing and AI Analyst pages need the same list --
see ``app/tabs_phase2.py``'s module docstring for why.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from charts_phase3 import anomaly_scatter, forecast_fan_chart, model_comparison_bar
from tabs_weather_context import render_severe_weather_sections

from src.anomalies import Anomaly, interpret_anomalies
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult, generate_forecast
from src.ingestion import EnergyType
from src.investigation import build_investigation_checklist
from src.weather_context import WeatherContextClassification
from src.weather_interpretation import UnusualMonthInterpretation


@st.cache_data(show_spinner="Cross-validating forecasting models...")
def generate_forecast_cached(clean: pd.DataFrame, horizon: int, model_name: str) -> ForecastResult:
    """Cached wrapper around ``src.forecast_evaluation.generate_forecast``.

    No network call is involved (unlike Phase 2's weather fetch) -- this is
    purely CPU-bound and fast enough (benchmarked well under 10s for the
    full 8-model comparison on the real data) to run automatically, but is
    still cached since Streamlit reruns the whole script on every widget
    interaction.
    """
    series = clean.set_index("month_start")["consumption_kwh"]
    return generate_forecast(series, horizon=horizon, model_name=model_name)


@st.cache_data(show_spinner="Running the full forecast model suite for Electricity, Gas, and Total...")
def generate_multi_fuel_forecast_cached(
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame], horizon: int = 12
) -> dict[EnergyType, ForecastResult]:
    """Run the same cross-validated forecast once per fuel -- opt-in (behind a button in the
    Cost Intelligence and Comparisons tabs) since this triples the walk-forward-CV cost of the
    single-fuel Forecasting tab. A fuel with insufficient history simply doesn't appear in the
    result rather than failing the whole comparison.
    """
    results: dict[EnergyType, ForecastResult] = {}
    for fuel, df in fuel_clean_dfs.items():
        if df.empty:
            continue
        try:
            results[fuel] = generate_forecast_cached(df, horizon, "auto")
        except ValueError:
            continue
    return results


def render_forecasting(clean: pd.DataFrame, horizon: int, model_choice: str) -> None:
    st.caption(
        "Uses full history regardless of the sidebar year filter. Cross-validation is "
        "1-step-ahead walk-forward, not multi-step -- ~35 months of data doesn't support "
        "reliable multi-step CV at longer horizons, so this only validates 1-month-ahead "
        "accuracy. The chart still projects the chosen model out to the full horizon below; "
        "that projection just isn't itself cross-validated at longer lead times."
    )
    model_name = "auto" if model_choice == "Auto (best by CV)" else model_choice

    try:
        result = generate_forecast_cached(clean, horizon, model_name)
    except ValueError as exc:
        st.warning(str(exc))
        return

    selection_note = " (auto-selected by CV)" if model_choice == "Auto (best by CV)" else " (forced)"
    st.subheader(f"Selected model: {result.model_name}{selection_note}")
    st.plotly_chart(forecast_fan_chart(clean, result), width="stretch")

    unit_rate = clean["cost_gbp"].sum() / clean["consumption_kwh"].sum()
    total_best = float(result.p10.sum())  # lower kWh = lower bill = the best case for the user's wallet
    total_likely = float(result.p50.sum())
    total_worst = float(result.p90.sum())  # higher kWh = higher bill = the worst plausible case
    tooltip = (
        "These come from resampling the model's own past forecast errors, scaled up the further "
        "ahead the month is -- a plausible range, not a guarantee. 'Best'/'Worst' refer to your "
        "bill (lower consumption is better for cost), not to forecast accuracy. £ figures cover "
        "energy consumption only -- standing charges aren't in the billing exports, so actual "
        "bills will be higher by that fixed daily amount."
    )
    # Whole pounds for forecast £: bootstrap bands don't support penny precision.
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Best plausible ({horizon}mo)", f"{total_best:,.0f} kWh", f"£{total_best * unit_rate:,.0f}", help=tooltip
    )
    c2.metric(
        f"Most likely ({horizon}mo)", f"{total_likely:,.0f} kWh", f"£{total_likely * unit_rate:,.0f}", help=tooltip
    )
    c3.metric(
        f"Worst plausible ({horizon}mo)", f"{total_worst:,.0f} kWh", f"£{total_worst * unit_rate:,.0f}", help=tooltip
    )

    st.subheader("Model comparison (cross-validated MAE, lower is better)")
    st.plotly_chart(model_comparison_bar(result.comparison, result.model_name), width="stretch")
    display = result.comparison.rename(
        columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)", "n_folds": "CV folds"}
    ).round(1)
    st.dataframe(display, hide_index=True, width="stretch")


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
