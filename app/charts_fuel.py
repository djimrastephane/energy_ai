"""Fuel-level chart builders: electricity vs. gas comparison.

Split out for the same reason as ``charts_phase3.py`` -- one concern per
module, reusing ``charts.py``'s palette/layout conventions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from charts import PALETTE, base_layout


def fuel_comparison_bar(combined_df: pd.DataFrame) -> go.Figure:
    """Monthly electricity vs. gas consumption, grouped -- answers 'which fuel drives usage, and when?'."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=combined_df["month_start"],
            y=combined_df["electricity_kwh"],
            name="Electricity",
            marker_color=PALETTE[0],
            hovertemplate="%{x|%B %Y}<br>Electricity: %{y:,.0f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=combined_df["month_start"],
            y=combined_df["gas_kwh"],
            name="Gas",
            marker_color=PALETTE[1],
            hovertemplate="%{x|%B %Y}<br>Gas: %{y:,.0f} kWh<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    return base_layout(fig, "Monthly Electricity vs. Gas Consumption", "kWh")


def fuel_share_area(combined_df: pd.DataFrame) -> go.Figure:
    """Electricity's share of combined monthly kWh -- answers 'how does the fuel mix shift with the seasons?'."""
    fig = go.Figure(
        go.Scatter(
            x=combined_df["month_start"],
            y=combined_df["electricity_share_pct"],
            mode="lines+markers",
            name="Electricity share",
            line={"width": 2, "color": PALETTE[0]},
            marker={"size": 7},
            fill="tozeroy",
            fillcolor="rgba(42, 120, 214, 0.15)",
            hovertemplate="%{x|%B %Y}<br>%{y:.0f}% electricity<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 100])
    return base_layout(fig, "Electricity Share of Combined Consumption", "% electricity")
