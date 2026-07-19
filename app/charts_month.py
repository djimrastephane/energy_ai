"""Chart builders for the month-comparison journey.

Same conventions as ``app/charts.py`` (single palette, ``base_layout``): the
selected month always renders in the primary blue, the comparison period in
the muted light blue already used for partial years -- colour supports the
words, it never carries the verdict alone.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from charts import PALETTE, base_layout

from src.ingestion import EnergyType
from src.monthly_comparison import MonthlyComparison

_COMPARISON_COLOR = "#a9c8ee"  # muted primary -- same as charts.annual_totals_bar partial years
_FUEL_LABELS: dict[EnergyType, str] = {
    "electricity": "Electricity",
    "gas": "Gas",
    "total": "Combined",
}


def two_month_grouped_bar(
    comparisons: dict[EnergyType, MonthlyComparison | None],
    comparison_label: str,
) -> go.Figure | None:
    """The page's one primary chart: selected month vs. comparison period, grouped by fuel.

    ``comparison_label`` names the comparator ("June 2025", "Typical June", ...).
    Fuels without a comparison simply contribute no bars; returns ``None`` if
    nothing at all is plottable.
    """
    categories: list[str] = []
    current_values: list[float] = []
    comparison_values: list[float] = []
    selected_label = None
    for fuel in ("electricity", "gas", "total"):
        comparison = comparisons.get(fuel)
        if comparison is None or comparison.comparison_consumption_kwh is None:
            continue
        categories.append(_FUEL_LABELS[fuel])
        current_values.append(comparison.current_consumption_kwh)
        comparison_values.append(comparison.comparison_consumption_kwh)
        selected_label = comparison.selected_month.strftime("%B %Y")
    if not categories:
        return None

    fig = go.Figure(
        [
            go.Bar(
                x=categories,
                y=comparison_values,
                name=comparison_label,
                marker_color=_COMPARISON_COLOR,
                hovertemplate=comparison_label + ": %{y:,.0f} kWh<extra></extra>",
            ),
            go.Bar(
                x=categories,
                y=current_values,
                name=selected_label,
                marker_color=PALETTE[0],
                hovertemplate=selected_label + ": %{y:,.0f} kWh<extra></extra>",
            ),
        ]
    )
    fig.update_layout(barmode="group")
    return base_layout(fig, f"{selected_label} vs {comparison_label}", "kWh")


def same_month_history_bar(
    clean_df: pd.DataFrame, selected_month: pd.Timestamp, in_progress: pd.Timestamp | None = None
) -> go.Figure | None:
    """The selected calendar month across every year on record, selected year highlighted.

    Reveals whether the latest month continues an improving or worsening
    pattern. Only same-calendar-month observations appear -- January is never
    charted against July. An in-progress month is excluded entirely (a
    month-to-date bar next to full months would invite a false comparison).
    """
    month_name = selected_month.strftime("%B")
    rows = clean_df[clean_df["month_start"].dt.month == selected_month.month].sort_values("month_start")
    if in_progress is not None:
        rows = rows[rows["month_start"] != in_progress]
    if rows.empty:
        return None

    labels = [f"{month_name} {m.year}" for m in rows["month_start"]]
    colors = [
        PALETTE[0] if m == selected_month else _COMPARISON_COLOR for m in rows["month_start"]
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=rows["consumption_kwh"],
            marker_color=colors,
            hovertemplate="%{x}<br>%{y:,.0f} kWh<extra></extra>",
        )
    )
    return base_layout(fig, f"Every {month_name} on record", "kWh")
