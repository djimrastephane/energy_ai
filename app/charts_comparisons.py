"""Comparisons-tab chart builders. Reuses ``charts.py``'s palette/layout conventions."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from charts import PALETTE, base_layout

_FUEL_COLOR = {"electricity": PALETTE[0], "gas": PALETTE[1], "total": PALETTE[3]}
_FUEL_LABEL = {"electricity": "Electricity", "gas": "Gas", "total": "Total"}


def annual_comparison_bar(annual_df: pd.DataFrame) -> go.Figure:
    """Grouped annual kWh per fuel -- answers 'which fuel's usage is actually changing year to year?'."""
    fig = go.Figure()
    for fuel in ("electricity", "gas", "total"):
        col = f"{fuel}_kwh"
        if col not in annual_df.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=annual_df["year"].astype(str),
                y=annual_df[col],
                name=_FUEL_LABEL[fuel],
                marker_color=_FUEL_COLOR[fuel],
                hovertemplate="%{x}<br>" + _FUEL_LABEL[fuel] + ": %{y:,.0f} kWh<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    return base_layout(fig, "Annual Consumption by Fuel", "kWh", "Year")
