"""Chart builders for the "How the Seasons Affect Usage" tab.

Split from ``charts.py`` per the project's ~300-line-per-file guideline.
Two figures: the plain-language primary view (actual consumption + the
already-computed smoothed trend + flagged unusual months) and the
calendar-month seasonal profile. Both render numbers
``src.decomposition``/``src.anomalies``/``src.seasonal_summary`` have
already computed -- no new statistics are introduced here.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.charts import PALETTE, base_layout
from src.anomalies import Anomaly
from src.decomposition import STLResult

_PARTIAL_YEAR_COLOR = "#a9c8ee"  # same lighter shade used by charts.annual_totals_bar


def _shade_winters(fig: go.Figure, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Light shading behind each December-February span in range, labelled once."""
    first = True
    year = start.year
    while True:
        winter_start = pd.Timestamp(year=year, month=12, day=1)
        winter_end = pd.Timestamp(year=year + 1, month=3, day=1)
        if winter_start > end:
            break
        if winter_end >= start:
            fig.add_vrect(
                x0=max(winter_start, start),
                x1=min(winter_end, end),
                fillcolor=PALETTE[0],
                opacity=0.06,
                line_width=0,
                layer="below",
                annotation_text="Winter (Dec-Feb)" if first else None,
                annotation_position="top left",
                annotation_font_size=10,
            )
            first = False
        year += 1


def seasonal_overview_figure(clean_df: pd.DataFrame, stl_result: STLResult, anomalies: list[Anomaly]) -> go.Figure:
    """Actual monthly consumption, the smoothed trend, and any materially unusual months.

    Replaces the four-panel STL chart as the primary view: answers "is this seasonal, is it
    trending, and did anything stand out?" without decomposition terminology. Winter months
    are lightly shaded so the calendar cycle reads visually. Months belonging to an incomplete
    calendar year are shown in a lighter bar shade (same convention as the Consumption Analysis
    tab's annual totals chart).
    """
    df = clean_df.sort_values("month_start")
    colors = [PALETTE[0] if not partial else _PARTIAL_YEAR_COLOR for partial in df["is_partial_year"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["month_start"],
            y=df["consumption_kwh"],
            name="Actual consumption",
            marker_color=colors,
            hovertemplate="%{x|%B %Y}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stl_result.trend.index,
            y=stl_result.trend.to_numpy(),
            name="Underlying trend (smoothed)",
            mode="lines",
            line={"width": 3, "color": PALETTE[1]},
            hovertemplate="%{x|%B %Y}<br>Trend: %{y:,.0f} kWh<extra></extra>",
        )
    )

    observed = stl_result.observed
    shown_legend: set[str] = set()
    for a in anomalies:
        if a.date not in observed.index:
            continue
        is_spike = a.direction == "spike"
        legend_name = "Higher than expected" if is_spike else "Lower than expected"
        fig.add_trace(
            go.Scatter(
                x=[a.date],
                y=[float(observed.loc[a.date])],
                mode="markers+text",
                name=legend_name,
                showlegend=legend_name not in shown_legend,
                marker={
                    "size": 14,
                    "symbol": "triangle-up" if is_spike else "triangle-down",
                    "color": PALETTE[3] if is_spike else PALETTE[2],
                    "line": {"width": 1.5, "color": "#ffffff"},
                },
                text=[a.date.strftime("%b %Y")],
                textposition="top center" if is_spike else "bottom center",
                textfont={"size": 10},
                hovertemplate=f"%{{x|%B %Y}}<br>%{{y:,.0f}} kWh<br>{a.rank_context}<extra></extra>",
            )
        )
        shown_legend.add(legend_name)

    if len(df) >= 1:
        _shade_winters(fig, df["month_start"].min(), df["month_start"].max())
    return base_layout(fig, "Actual Consumption, Trend, and Unusual Months", "kWh", "Month")


def seasonal_calendar_profile_figure(profile_df: pd.DataFrame) -> go.Figure:
    """Typical seasonal effect (kWh) for each calendar month.

    Positive bars sit above the household's overall typical month, negative bars below --
    shown by both color and a hatch pattern on the negative bars so the sign doesn't rely on
    color alone. ``profile_df`` must already be in January-December order (as returned by
    ``src.seasonal_summary.seasonal_profile``).
    """
    values = profile_df["typical_effect_kwh"]
    colors = [PALETTE[3] if v >= 0 else PALETTE[2] for v in values]
    patterns = ["" if v >= 0 else "/" for v in values]
    fig = go.Figure(
        go.Bar(
            x=profile_df["month_name"],
            y=values,
            marker={"color": colors, "pattern": {"shape": patterns}},
            hovertemplate="%{x}<br>%{y:+,.0f} kWh vs. typical month<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"width": 1, "color": "#888888"})
    fig.update_xaxes(categoryorder="array", categoryarray=profile_df["month_name"].tolist())
    return base_layout(
        fig,
        "Typical Seasonal Effect by Calendar Month",
        "kWh vs. typical month (+ above / − below)",
        "Calendar month",
    )
