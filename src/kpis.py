"""Period-over-period KPI comparisons for the Executive Summary.

Separate from ``src.statistics`` (which covers distributional statistics of
a single series) because these functions are specifically about comparing
two time periods -- trailing windows, complete calendar years, and
year-to-date -- against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import get_logger, pct_change, safe_divide

logger = get_logger(__name__)


@dataclass
class KPIComparison:
    label: str
    current: float
    previous: float | None
    pct_change: float | None
    unit: str = ""


def compute_kpis(df: pd.DataFrame, months: int = 12) -> list[KPIComparison]:
    """Compare the trailing ``months``-month period against the preceding one.

    Using a rolling trailing window (rather than calendar-year boundaries)
    means the comparison is meaningful regardless of where "today" falls in
    the calendar year. If fewer than ``2 * months`` rows of history exist,
    ``previous``/``pct_change`` are returned as ``None`` rather than
    comparing against a partial or nonexistent period.
    """
    df = df.sort_values("month_start")
    if len(df) < months:
        logger.warning(
            "Only %d month(s) of data available; requested a %d-month KPI window",
            len(df),
            months,
        )
        months = len(df)
    if months == 0:
        return []

    current = df.tail(months)
    has_previous = len(df) >= 2 * months
    previous = df.tail(2 * months).head(months) if has_previous else None

    def _weighted_unit_rate(frame: pd.DataFrame) -> float:
        return safe_divide(frame["cost_gbp"].sum(), frame["consumption_kwh"].sum())

    def _weighted_avg_daily(frame: pd.DataFrame, col: str) -> float:
        return safe_divide(frame[col].sum(), frame["days_in_month"].sum())

    metrics: list[tuple[str, str, float, float | None]] = [
        (
            "Total consumption",
            "kWh",
            current["consumption_kwh"].sum(),
            previous["consumption_kwh"].sum() if has_previous else None,
        ),
        (
            "Total cost",
            "£",
            current["cost_gbp"].sum(),
            previous["cost_gbp"].sum() if has_previous else None,
        ),
        (
            "Average monthly consumption",
            "kWh",
            current["consumption_kwh"].mean(),
            previous["consumption_kwh"].mean() if has_previous else None,
        ),
        (
            "Average daily consumption",
            "kWh",
            _weighted_avg_daily(current, "consumption_kwh"),
            _weighted_avg_daily(previous, "consumption_kwh") if has_previous else None,
        ),
        (
            "Average monthly cost",
            "£",
            current["cost_gbp"].mean(),
            previous["cost_gbp"].mean() if has_previous else None,
        ),
        (
            "Average cost per kWh",
            "£/kWh",
            _weighted_unit_rate(current),
            _weighted_unit_rate(previous) if has_previous else None,
        ),
    ]

    return [
        KPIComparison(
            label=label,
            current=curr,
            previous=prev,
            pct_change=pct_change(curr, prev) if prev is not None else None,
            unit=unit,
        )
        for label, unit, curr, prev in metrics
    ]


def full_year_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Total consumption/cost per *complete* calendar year, with year-on-year % change.

    Partial years (fewer than 12 months present -- typically the first and
    most recent years in a real export set) are excluded so the comparison
    is always like-for-like.
    """
    complete = df[~df["is_partial_year"]]
    if complete.empty:
        return pd.DataFrame(
            columns=["year", "total_kwh", "total_cost_gbp", "yoy_kwh_pct", "yoy_cost_pct"]
        )

    yearly = (
        complete.groupby("year")
        .agg(total_kwh=("consumption_kwh", "sum"), total_cost_gbp=("cost_gbp", "sum"))
        .reset_index()
        .sort_values("year")
    )
    yearly["yoy_kwh_pct"] = yearly["total_kwh"].pct_change() * 100
    yearly["yoy_cost_pct"] = yearly["total_cost_gbp"].pct_change() * 100
    return yearly


def year_to_date_comparison(df: pd.DataFrame) -> dict | None:
    """Compare the most recent partial year's months to the same months a year earlier.

    Returns None if the most recent year is complete (nothing "to date" to
    compare) or if the prior year doesn't have matching months available.
    """
    partial_years = sorted(df.loc[df["is_partial_year"], "year"].unique())
    if not partial_years:
        return None
    current_year = int(partial_years[-1])
    current_rows = df[df["year"] == current_year]
    months_present = sorted(current_rows["month_num"].unique())

    previous_year = current_year - 1
    previous_rows = df[(df["year"] == previous_year) & (df["month_num"].isin(months_present))]
    if previous_rows.empty:
        return None

    current_kwh, previous_kwh = (
        current_rows["consumption_kwh"].sum(),
        previous_rows["consumption_kwh"].sum(),
    )
    current_cost, previous_cost = (
        current_rows["cost_gbp"].sum(),
        previous_rows["cost_gbp"].sum(),
    )

    return {
        "current_year": current_year,
        "previous_year": previous_year,
        "months_compared": months_present,
        "current_kwh": current_kwh,
        "previous_kwh": previous_kwh,
        "kwh_pct_change": pct_change(current_kwh, previous_kwh),
        "current_cost_gbp": current_cost,
        "previous_cost_gbp": previous_cost,
        "cost_pct_change": pct_change(current_cost, previous_cost),
    }
