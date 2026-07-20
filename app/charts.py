"""Reusable Plotly chart builders for the Streamlit app.

Every chart in the UI is produced by one of these functions so there is a
single place that owns styling (palette, line/marker specs, gridlines) and
so chart logic is testable independently of Streamlit.

Colour palette is the validated categorical set from the project's dataviz
guidelines (CVD-safe in fixed order, never cycled): blue, green, magenta,
yellow. At most 4 years of data are ever plotted together, which stays
within the 4-series band the palette validates for all-pairs comparisons.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.decomposition import STLResult

PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
_GRID_COLOR = "#e5e5e0"
_MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def base_layout(fig: go.Figure, title: str, y_title: str, x_title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 60, "r": 20, "b": 40, "l": 60},
    )
    fig.update_xaxes(title=x_title, showgrid=False)
    fig.update_yaxes(title=y_title, showgrid=True, gridcolor=_GRID_COLOR, gridwidth=1)
    return fig


def monthly_consumption_bar(df: pd.DataFrame) -> go.Figure:
    """Monthly consumption (kWh) as a bar chart -- answers 'how has usage moved month to month?'."""
    fig = go.Figure(
        go.Bar(
            x=df["month_start"],
            y=df["consumption_kwh"],
            marker_color=PALETTE[0],
            hovertemplate="%{x|%B %Y}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )
    return base_layout(fig, "Monthly Consumption", "kWh")


def annual_totals_bar(df: pd.DataFrame) -> go.Figure:
    """Total consumption per calendar year -- partial years are visually distinguished."""
    yearly = df.groupby("year").agg(
        total_kwh=("consumption_kwh", "sum"), is_partial=("is_partial_year", "any")
    )
    colors = [PALETTE[0] if not partial else "#a9c8ee" for partial in yearly["is_partial"]]
    labels = [
        f"{y}{' (partial)' if p else ''}"
        for y, p in zip(yearly.index, yearly["is_partial"], strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=yearly["total_kwh"],
            marker_color=colors,
            hovertemplate="%{x}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )
    return base_layout(fig, "Annual Consumption Total", "kWh")


def year_over_year_overlay(df: pd.DataFrame) -> go.Figure:
    """One line per year, aligned by calendar month -- answers 'how does this year compare?'."""
    fig = go.Figure()
    years = sorted(df["year"].unique())
    for i, year in enumerate(years):
        year_df = df[df["year"] == year].sort_values("month_num")
        fig.add_trace(
            go.Scatter(
                x=year_df["month_name"],
                y=year_df["consumption_kwh"],
                name=str(year),
                mode="lines+markers",
                line={"width": 2, "color": PALETTE[i % len(PALETTE)]},
                marker={"size": 8},
                hovertemplate="%{x} " + str(year) + "<br>%{y:,.0f} kWh<extra></extra>",
            )
        )
    fig.update_xaxes(categoryorder="array", categoryarray=_MONTH_ORDER)
    return base_layout(fig, "Year-on-Year Comparison", "kWh")


def rolling_average_line(df: pd.DataFrame, window: int = 3) -> go.Figure:
    """Actual monthly consumption vs. a trailing rolling average -- smooths month-to-month noise."""
    df = df.sort_values("month_start")
    rolling = df["consumption_kwh"].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["month_start"],
            y=df["consumption_kwh"],
            name="Actual",
            mode="lines+markers",
            line={"width": 2, "color": PALETTE[0]},
            marker={"size": 8},
            hovertemplate="%{x|%B %Y}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["month_start"],
            y=rolling,
            name=f"{window}-month rolling average",
            mode="lines",
            line={"width": 2, "color": PALETTE[1], "dash": "dash"},
            hovertemplate="%{x|%B %Y}<br>%{y:,.0f} kWh (rolling)<extra></extra>",
        )
    )
    return base_layout(fig, f"Consumption with {window}-Month Rolling Average", "kWh")


def stl_components_figure(result: STLResult) -> go.Figure:
    """Observed / trend / seasonal / residual, stacked -- answers 'what's trend, what's calendar, what's noise?'."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.06,
    )
    components = [
        (result.observed, "lines", False),
        (result.trend, "lines", False),
        (result.seasonal, "lines", False),
        (result.resid, "markers", True),
    ]
    for row, (series, mode, is_resid) in enumerate(components, start=1):
        trace_kwargs = {"marker": {"size": 6, "color": PALETTE[0]}} if is_resid else {
            "line": {"width": 2, "color": PALETTE[0]}
        }
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(),
                mode=mode,
                showlegend=False,
                hovertemplate="%{x|%b %Y}<br>%{y:,.1f}<extra></extra>",
                **trace_kwargs,
            ),
            row=row,
            col=1,
        )
    fig.update_layout(
        title="STL Decomposition of Monthly Consumption",
        template="plotly_white",
        height=650,
        margin={"t": 60, "r": 20, "b": 40, "l": 60},
    )
    fig.update_yaxes(title_text="kWh", showgrid=True, gridcolor=_GRID_COLOR, gridwidth=1)
    return fig


def energy_signature_scatter(merged_df: pd.DataFrame, pdp_df: pd.DataFrame, feature: str) -> go.Figure:
    """Monthly consumption vs. a weather variable, with the fitted energy-signature line overlaid.

    Answers 'does colder/hotter weather actually explain higher usage, and by how much?'.
    ``feature`` is ``avg_daily_hdd`` or ``avg_daily_cdd``; ``pdp_df`` is the output of
    ``src.energy_signature.partial_dependence`` for that same feature.
    """
    label = "Heating Degree Days" if feature == "avg_daily_hdd" else "Cooling Degree Days"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_df[feature],
            y=merged_df["avg_daily_kwh"],
            mode="markers",
            name="Observed months",
            marker={"size": 9, "color": PALETTE[0], "line": {"width": 2, "color": "#ffffff"}},
            hovertemplate=f"{label}: " + "%{x:.1f}<br>%{y:.1f} kWh/day<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pdp_df[feature],
            y=pdp_df["predicted_avg_daily_kwh"],
            mode="lines",
            name="Fitted (other feature at its mean)",
            line={"width": 2, "color": PALETTE[1]},
            hovertemplate="predicted %{y:.1f} kWh/day<extra></extra>",
        )
    )
    return base_layout(fig, f"Energy Signature: Consumption vs. {label}", "Avg. daily kWh", label)


def changepoint_timeline(deseasonalized_series: pd.Series, changepoints: list[ChangePoint]) -> go.Figure:
    """Deseasonalized consumption with detected change points marked -- answers 'when did behaviour actually shift?'."""
    fig = go.Figure(
        go.Scatter(
            x=deseasonalized_series.index,
            y=deseasonalized_series.to_numpy(),
            mode="lines+markers",
            name="Deseasonalized (trend + residual)",
            line={"width": 2, "color": PALETTE[0]},
            marker={"size": 7},
            hovertemplate="%{x|%b %Y}<br>%{y:,.1f} kWh<extra></extra>",
        )
    )
    for cp in changepoints:
        style = "solid" if cp.method == "both" else "dash"
        color = PALETTE[3] if cp.direction == "increase" else PALETTE[2]
        fig.add_vline(
            x=cp.date.timestamp() * 1000,
            line={"width": 2, "dash": style, "color": color},
            annotation_text=f"{cp.date:%b %Y} ({cp.method})",
            annotation_position="top",
        )
    return base_layout(fig, "Detected Change Points", "kWh (deseasonalized)")


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
    tiers: dict[str, dict[str, list]] = {
        "high": {"x": [], "y": [], "text": []},
        "low": {"x": [], "y": [], "text": []},
    }
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
