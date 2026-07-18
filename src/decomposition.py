"""STL decomposition of the monthly consumption series.

Splits ``consumption_kwh`` into trend, seasonal, and residual components via
statsmodels' STL (Seasonal-Trend decomposition using LOESS), and quantifies
how much of the variation each component explains using the standard
Hyndman & Athanasopoulos strength measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class STLResult:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    seasonal_strength: float
    trend_strength: float

    @property
    def deseasonalized(self) -> pd.Series:
        """Trend + residual -- the series with the regular seasonal cycle removed.

        Used as the input to change-point detection (``src.changepoints``)
        so that seasonal winter/summer swings aren't mistaken for
        behavioural shifts.
        """
        return self.trend + self.resid


def _strength(component: pd.Series, resid: pd.Series) -> float:
    """Hyndman & Athanasopoulos strength measure: 1 - Var(resid) / Var(component + resid)."""
    denom = np.var(component + resid)
    if denom == 0:
        return 0.0
    return max(0.0, 1.0 - np.var(resid) / denom)


def stl_decompose(monthly_df: pd.DataFrame, period: int = 12) -> STLResult:
    """Decompose ``consumption_kwh`` into trend/seasonal/residual via STL.

    Requires a contiguous monthly series (no gaps) indexed by
    ``month_start`` -- pass the output of ``src.preprocessing.run_pipeline``
    directly, which already reports any missing months separately (see the
    Data Quality tab).

    Raises:
        ValueError: if the months aren't contiguous, or if fewer than one
            full period plus one observation (13 months for the default
            ``period=12``) is available. STL can technically run below 2
            full periods (24 months), but results with under 24 months
            carry a "read with caution" note from :func:`interpret_decomposition`
            rather than being blocked outright.
    """
    df = monthly_df.sort_values("month_start")
    months = df["month_start"]
    expected = pd.date_range(months.iloc[0], months.iloc[-1], freq="MS")
    if len(months) != len(expected) or not (months.to_numpy() == expected.to_numpy()).all():
        raise ValueError(
            "stl_decompose requires a contiguous monthly series with no gaps -- "
            "check the Data Quality tab for missing months."
        )
    if len(df) < period + 1:
        raise ValueError(
            f"STL needs at least {period + 1} months (one full period plus one "
            f"observation) of data; got {len(df)}."
        )

    series = pd.Series(
        df["consumption_kwh"].to_numpy(),
        index=pd.DatetimeIndex(df["month_start"], freq="MS"),
        name="consumption_kwh",
    )

    fit = STL(series, period=period, robust=True).fit()
    seasonal_strength = _strength(fit.seasonal, fit.resid)
    trend_strength = _strength(fit.trend, fit.resid)

    logger.info(
        "STL decomposition: n=%d, seasonal_strength=%.2f, trend_strength=%.2f",
        len(series),
        seasonal_strength,
        trend_strength,
    )

    return STLResult(
        observed=series,
        trend=fit.trend,
        seasonal=fit.seasonal,
        resid=fit.resid,
        seasonal_strength=seasonal_strength,
        trend_strength=trend_strength,
    )


def interpret_decomposition(result: STLResult) -> str:
    """Plain-English narrative of the seasonal/trend strength.

    Thresholds used: strength < 0.3 = weak, 0.3-0.6 = moderate, > 0.6 = strong.
    """
    n = len(result.observed)
    caveat = (
        " With fewer than 24 months of history, edge effects in the trend "
        "estimate are more pronounced than usual -- read the first and last "
        "few points with extra caution."
        if n < 24
        else ""
    )

    def _band(x: float) -> str:
        if x < 0.3:
            return "weak"
        if x < 0.6:
            return "moderate"
        return "strong"

    return (
        f"Across {n} months, the seasonal (calendar-month) pattern explains a "
        f"{_band(result.seasonal_strength)} share of month-to-month variation "
        f"(seasonal strength = {result.seasonal_strength:.0%}), and the "
        f"underlying trend is {_band(result.trend_strength)} "
        f"(trend strength = {result.trend_strength:.0%}). Seasonal strength near "
        "100% means swings are almost entirely the regular calendar cycle (e.g. "
        "winter vs. summer); trend strength near 100% means there is a "
        "persistent underlying rise or fall beyond that cycle." + caveat
    )
