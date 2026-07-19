"""Plain-language layer over an already-computed STL decomposition.

Everything here is derived from ``STLResult`` and ``Anomaly`` objects that
``src.decomposition``/``src.anomalies`` have already produced -- no new
statistics are introduced, only calendar-averaging and thresholded
labelling of numbers that already exist, in service of the "How the
Seasons Affect Usage" tab's non-technical summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.anomalies import Anomaly
from src.confidence import Confidence
from src.decomposition import STLResult
from src.utils import pct_change

SeasonalBand = Literal["Low", "Moderate", "Strong"]
TrendLabel = Literal["Falling", "Stable", "Rising"]

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip


def _seasonal_band(strength: float) -> SeasonalBand:
    """Same 0.3/0.6 cut points as ``src.decomposition.interpret_decomposition`` and
    ``src.findings.finding_seasonality``, so "Strong" here means the same thing everywhere."""
    if strength < 0.3:
        return "Low"
    if strength < 0.6:
        return "Moderate"
    return "Strong"


def trend_direction(result: STLResult, threshold_pct_per_year: float = 5.0) -> tuple[TrendLabel, float]:
    """Classify the STL trend component's direction from its first-to-last change, annualized.

    Uses the already-smoothed ``result.trend`` (LOESS-smoothed by STL itself), not the raw
    series, so a single noisy month at either end can't flip the label. A move of less than
    ``threshold_pct_per_year`` (default 5%/year) reads as "Stable" -- small drift is normal and
    calling it a trend would overstate the evidence.

    Returns ``(label, annualized_pct_change)``; the percentage is signed (negative = falling).
    """
    trend = result.trend
    n_months = len(trend)
    if n_months < 2:
        return "Stable", 0.0

    total_pct = pct_change(float(trend.iloc[-1]), float(trend.iloc[0]))
    if total_pct is None:
        return "Stable", 0.0

    years = n_months / 12.0
    annualized_pct = total_pct / years

    if annualized_pct >= threshold_pct_per_year:
        return "Rising", annualized_pct
    if annualized_pct <= -threshold_pct_per_year:
        return "Falling", annualized_pct
    return "Stable", annualized_pct


def seasonal_profile(result: STLResult) -> pd.DataFrame:
    """Typical seasonal effect (kWh) for each calendar month, averaged across every year present.

    Positive values mean that calendar month typically sits above the household's overall level
    once trend is removed; negative means below. Returns one row per January-December, ordered.
    """
    seasonal = result.seasonal
    by_month = seasonal.groupby(seasonal.index.month).mean()
    return pd.DataFrame(
        {
            "month_num": range(1, 13),
            "month_name": _MONTH_NAMES,
            "typical_effect_kwh": [float(by_month.get(m, 0.0)) for m in range(1, 13)],
        }
    )


@dataclass
class SeasonalPatternSummary:
    seasonal_band: SeasonalBand
    seasonal_strength: float
    trend_label: TrendLabel
    trend_pct_per_year: float
    trend_strength: float
    main_conclusion: str
    why_it_matters: str
    confidence: Confidence
    confidence_reason: str
    unusual_month: Anomaly | None
    unusual_month_kwh: float | None
    unusual_month_context: str


_SEASONAL_CLAUSE: dict[SeasonalBand, str] = {
    "Strong": "Your energy use is driven mainly by the regular winter and summer cycle.",
    "Moderate": "Your energy use follows the winter and summer cycle to some extent, but not overwhelmingly.",
    "Low": "Your energy use is not strongly tied to the time of year.",
}


def summarize_seasonal_pattern(stl_result: STLResult, anomalies: list[Anomaly]) -> SeasonalPatternSummary:
    """Build the non-technical top-of-tab summary and interpretation-card data.

    ``anomalies`` should be the same cross-referenced list ``src.anomalies.detect_anomalies``
    already produced for this fuel -- the "unusual month" here is never a new threshold, it's
    whichever already-flagged month has the largest residual magnitude.
    """
    band = _seasonal_band(stl_result.seasonal_strength)
    trend_label, trend_pct = trend_direction(stl_result)
    n_months = len(stl_result.observed)

    trend_clause = {
        "Stable": "The underlying long-term trend is broadly stable.",
        "Rising": f"There is also a persistent underlying rise of roughly {abs(trend_pct):.0f}% a year.",
        "Falling": f"There is also a persistent underlying fall of roughly {abs(trend_pct):.0f}% a year.",
    }[trend_label]
    main_conclusion = f"{_SEASONAL_CLAUSE[band]} {trend_clause}"

    largest = max(anomalies, key=lambda a: abs(stl_result.resid.loc[a.date])) if anomalies else None
    if largest is not None:
        resid_val = float(stl_result.resid.loc[largest.date])
        direction_word = "higher" if largest.direction == "spike" else "lower"
        main_conclusion += (
            f" {largest.date.strftime('%B %Y')} was materially {direction_word} than the "
            "expected seasonal pattern."
        )
        unusual_month_context = (
            f"{largest.date.strftime('%B %Y')} was {abs(resid_val):,.0f} kWh {direction_word} than "
            f"trend and season alone predicted ({largest.rank_context})."
        )
        unusual_kwh: float | None = resid_val
    else:
        unusual_month_context = "No month departed materially from the expected seasonal pattern."
        unusual_kwh = None

    if band == "Strong" and trend_label == "Stable":
        why_it_matters = (
            "This is expected and predictable: usage should keep rising in winter and falling in "
            "summer, with no sign of an underlying increase to address."
        )
    elif trend_label != "Stable":
        why_it_matters = (
            f"Beyond the normal seasonal swing, a persistent {trend_label.lower()} trend suggests "
            "something other than weather is changing -- worth checking for a change in occupancy, "
            "equipment, or habits."
        )
    else:
        why_it_matters = (
            "Usage doesn't follow a strong predictable pattern, so month-to-month changes are "
            "harder to anticipate from the calendar alone."
        )

    confidence: Confidence = "High" if n_months >= 24 else "Medium"
    confidence_reason = (
        f"Based on {n_months} months of history."
        if n_months >= 24
        else f"Only {n_months} months of history -- under 24 months, edge effects in the trend "
        "estimate are more pronounced than usual."
    )

    return SeasonalPatternSummary(
        seasonal_band=band,
        seasonal_strength=stl_result.seasonal_strength,
        trend_label=trend_label,
        trend_pct_per_year=trend_pct,
        trend_strength=stl_result.trend_strength,
        main_conclusion=main_conclusion,
        why_it_matters=why_it_matters,
        confidence=confidence,
        confidence_reason=confidence_reason,
        unusual_month=largest,
        unusual_month_kwh=unusual_kwh,
        unusual_month_context=unusual_month_context,
    )
