"""Fuel-level chart builders: electricity vs. gas comparison.

Split out for the same reason as ``charts_forecast.py`` -- one concern per
module, reusing ``charts.py``'s palette/layout conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.charts import PALETTE, base_layout


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


def fuel_mix_annual_stacked_bar(combined_df: pd.DataFrame) -> go.Figure:
    """Annual electricity % vs. gas % of combined consumption, stacked -- answers 'has the fuel mix shifted year to year?'."""
    yearly = combined_df.copy()
    yearly["year"] = yearly["month_start"].dt.year
    grouped = (
        yearly.groupby("year")
        .agg(electricity_kwh=("electricity_kwh", "sum"), gas_kwh=("gas_kwh", "sum"))
        .reset_index()
    )
    total_kwh = grouped["electricity_kwh"] + grouped["gas_kwh"]
    grouped["electricity_pct"] = np.where(total_kwh > 0, grouped["electricity_kwh"] / total_kwh * 100, np.nan)
    grouped["gas_pct"] = np.where(total_kwh > 0, grouped["gas_kwh"] / total_kwh * 100, np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["year"].astype(str),
            y=grouped["electricity_pct"],
            name="Electricity",
            marker_color=PALETTE[0],
            hovertemplate="%{x}<br>Electricity: %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped["year"].astype(str),
            y=grouped["gas_pct"],
            name="Gas",
            marker_color=PALETTE[1],
            hovertemplate="%{x}<br>Gas: %{y:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(range=[0, 100])
    return base_layout(fig, "Annual Fuel Mix (% of Combined Consumption)", "% of combined kWh", "Year")
