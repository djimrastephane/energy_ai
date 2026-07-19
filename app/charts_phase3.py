"""Phase 3 chart builders: forecasting and anomaly detection.

Split from ``charts.py`` to keep that module under the project's preferred
~300-line-per-file guideline, same pattern as ``tabs_core.py``/
``tabs_phase2.py``/``tabs_phase3.py``. Shares the same palette/layout
conventions as ``charts.py``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from charts import PALETTE, base_layout

from src.anomalies import Anomaly
from src.forecast_evaluation import ForecastResult


def forecast_fan_chart(history: pd.DataFrame, forecast_result: ForecastResult) -> go.Figure:
    """Historical consumption plus the forecast's plausible range and expected line.

    Presentation choices (per the forecast-page UX review): history is
    slightly faded so the eye lands on the forecast; an explicit
    "Forecast begins" marker separates measurement from projection; and
    the plausible-range band fades progressively with the horizon, since
    uncertainty grows the further ahead the month is -- the band should
    *look* less certain at 12 months out than at 1.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["month_start"],
            y=history["consumption_kwh"],
            mode="lines+markers",
            name="Actual (history)",
            line={"width": 2, "color": "rgba(42, 120, 214, 0.45)"},
            marker={"size": 5, "color": "rgba(42, 120, 214, 0.45)"},
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )

    # Plausible range as per-month segments with decreasing opacity -- one legend
    # entry for the first segment, the rest hidden from the legend.
    dates = list(forecast_result.forecast_dates)
    p10 = list(forecast_result.p10)
    p90 = list(forecast_result.p90)
    n_segments = max(len(dates) - 1, 1)
    for i in range(len(dates) - 1):
        alpha = 0.24 - (0.24 - 0.05) * (i / n_segments)
        fig.add_trace(
            go.Scatter(
                x=[dates[i], dates[i + 1]],
                y=[p90[i], p90[i + 1]],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[dates[i], dates[i + 1]],
                y=[p10[i], p10[i + 1]],
                mode="lines",
                name="Plausible range (fades = less certain)",
                showlegend=(i == 0),
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba(42, 120, 214, {alpha:.3f})",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast_result.forecast_dates,
            y=forecast_result.p50,
            mode="lines+markers",
            name="Expected",
            line={"width": 2.5, "color": PALETTE[1], "dash": "dash"},
            marker={"size": 6},
            hovertemplate="%{x|%b %Y}<br>expected %{y:,.0f} kWh<extra></extra>",
        )
    )

    if len(history) and len(dates):
        boundary = history["month_start"].max() + (dates[0] - history["month_start"].max()) / 2
        fig.add_vline(
            x=boundary.timestamp() * 1000,
            line={"width": 1.5, "dash": "dot", "color": "#8a8a8a"},
            annotation_text="Forecast begins",
            annotation_position="top",
        )
    return base_layout(fig, "Consumption: History and Forecast", "kWh")


def model_comparison_bar(comparison_df: pd.DataFrame, chosen_model: str) -> go.Figure:
    """CV MAE per model, sorted, with the chosen model highlighted -- shows the comparison, not just the winner."""
    colors = [PALETTE[0] if m == chosen_model else "#c7c7c7" for m in comparison_df["model"]]
    fig = go.Figure(
        go.Bar(
            x=comparison_df["model"],
            y=comparison_df["mae"],
            marker_color=colors,
            hovertemplate="%{x}<br>MAE %{y:,.1f} kWh<extra></extra>",
        )
    )
    return base_layout(fig, "Model Comparison (Cross-Validated MAE, Lower = Better)", "MAE (kWh)")


def anomaly_scatter(series: pd.Series, anomalies: list[Anomaly]) -> go.Figure:
    """Monthly consumption with flagged anomalies marked, styled by how many methods agree."""
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            mode="lines+markers",
            name="Consumption",
            line={"width": 2, "color": PALETTE[0]},
            marker={"size": 6},
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )

    by_date = {a.date: a for a in anomalies}
    tiers = {"high": {"x": [], "y": [], "text": []}, "low": {"x": [], "y": [], "text": []}}
    for date, value in series.items():
        anomaly = by_date.get(date)
        if anomaly is None:
            continue
        tier = "high" if len(anomaly.methods) >= 2 else "low"
        tiers[tier]["x"].append(date)
        tiers[tier]["y"].append(value)
        tiers[tier]["text"].append(f"{', '.join(anomaly.methods)}<br>{anomaly.rank_context}")

    if tiers["low"]["x"]:
        fig.add_trace(
            go.Scatter(
                x=tiers["low"]["x"],
                y=tiers["low"]["y"],
                mode="markers",
                name="Flagged (1 method)",
                marker={"size": 12, "color": PALETTE[3], "symbol": "circle-open", "line": {"width": 2}},
                hovertext=tiers["low"]["text"],
                hovertemplate="%{x|%b %Y}: %{y:,.0f} kWh<br>%{hovertext}<extra></extra>",
            )
        )
    if tiers["high"]["x"]:
        fig.add_trace(
            go.Scatter(
                x=tiers["high"]["x"],
                y=tiers["high"]["y"],
                mode="markers",
                name="Flagged (2+ methods)",
                marker={
                    "size": 14,
                    "color": PALETTE[2],
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "#ffffff"},
                },
                hovertext=tiers["high"]["text"],
                hovertemplate="%{x|%b %Y}: %{y:,.0f} kWh<br>%{hovertext}<extra></extra>",
            )
        )

    return base_layout(fig, "Consumption with Flagged Anomalies", "kWh")
