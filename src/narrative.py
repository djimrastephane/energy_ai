"""Deterministic, plain-English narrative for a single month, built entirely from computed statistics.

Every sentence traces to a number already computed elsewhere (year-on-year
billing totals, the weather regression's already-fitted series, anomaly
detection results, forecast confidence) -- no free-form generation, no
hallucination. Callable for any month (the "for every month" capability);
the UI shows the current month prominently plus an expandable history
rather than 35 paragraphs inline -- see ``app/tabs_analyst.py``.
"""

from __future__ import annotations

import pandas as pd

from src.anomalies import Anomaly
from src.confidence import ConfidenceRating
from src.energy_signature import EnergySignatureResult
from src.utils import get_logger

logger = get_logger(__name__)

_MIN_MEANINGFUL_DELTA_KWH = 1.0


def _yoy_change(month: pd.Timestamp, clean_indexed: pd.DataFrame) -> tuple[float, float, float] | None:
    """(current_kwh, prior_kwh, pct_change) for ``month`` vs. the same month a year earlier, or None."""
    if month not in clean_indexed.index:
        return None
    prior_month = month - pd.DateOffset(years=1)
    if prior_month not in clean_indexed.index:
        return None
    current_kwh = float(clean_indexed.loc[month, "consumption_kwh"])
    prior_kwh = float(clean_indexed.loc[prior_month, "consumption_kwh"])
    if prior_kwh == 0:
        return None
    return current_kwh, prior_kwh, (current_kwh - prior_kwh) / prior_kwh * 100


def _weather_explained_fraction(
    month: pd.Timestamp,
    prior_month: pd.Timestamp,
    merged_indexed: pd.DataFrame,
    energy_result: EnergySignatureResult,
    actual_delta_kwh: float,
) -> float | None:
    """Fraction of the YoY kWh change attributable to weather, using the already-fitted series."""
    if month not in merged_indexed.index or prior_month not in merged_indexed.index:
        return None
    if month not in energy_result.fitted.index or prior_month not in energy_result.fitted.index:
        return None
    fitted_current_kwh = energy_result.fitted.loc[month] * merged_indexed.loc[month, "days_in_month"]
    fitted_prior_kwh = energy_result.fitted.loc[prior_month] * merged_indexed.loc[prior_month, "days_in_month"]
    return (fitted_current_kwh - fitted_prior_kwh) / actual_delta_kwh


def _weather_sentence(
    month: pd.Timestamp,
    prior_month: pd.Timestamp,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    actual_delta_kwh: float,
) -> str:
    if merged_df is None or energy_result is None:
        return "Enable weather adjustment in the sidebar for a fuller explanation."
    if abs(actual_delta_kwh) < _MIN_MEANINGFUL_DELTA_KWH:
        return ""

    merged_indexed = merged_df.sort_values("month_start").set_index("month_start")
    fraction = _weather_explained_fraction(month, prior_month, merged_indexed, energy_result, actual_delta_kwh)
    if fraction is None:
        return "Weather-adjusted comparison is unavailable for this month."
    if fraction < 0:
        return "Weather alone would have predicted the opposite direction, so this change isn't weather-related."
    if fraction > 1.5:
        return "Weather more than fully explains this change; other factors may be partly offsetting it."
    direction = "colder" if actual_delta_kwh > 0 else "milder"
    return f"Approximately {fraction:.0%} is explained by {direction} weather."


def monthly_narrative(
    month: pd.Timestamp,
    clean_df: pd.DataFrame,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    anomalies: list[Anomaly],
    forecast_rating: ConfidenceRating | None,
) -> str:
    """Build the narrative for one month. Returns a short paragraph, entirely evidence-grounded."""
    clean_indexed = clean_df.sort_values("month_start").set_index("month_start")
    sentences: list[str] = []

    yoy = _yoy_change(month, clean_indexed)
    if yoy is None:
        return (
            f"No year-on-year comparison is available for {month.strftime('%B %Y')} "
            "(no matching month a year earlier in the data)."
        )

    current_kwh, prior_kwh, pct_change = yoy
    direction = "increased" if pct_change >= 0 else "decreased"
    sentences.append(f"Electricity consumption {direction} {abs(pct_change):.0f}%.")

    weather_sentence = _weather_sentence(
        month, month - pd.DateOffset(years=1), merged_df, energy_result, current_kwh - prior_kwh
    )
    if weather_sentence:
        sentences.append(weather_sentence)

    month_anomaly = next((a for a in anomalies if a.date == month), None)
    if month_anomaly is not None and len(month_anomaly.methods) >= 2:
        sentences.append(f"{len(month_anomaly.methods)} independent anomaly detectors agree this month is unusual.")
    elif month_anomaly is not None:
        sentences.append("One anomaly detection method flagged this month, but it isn't independently confirmed.")
    else:
        sentences.append("No statistically significant anomalies were detected.")

    if forecast_rating is not None:
        sentences.append(f"Forecast confidence remains {forecast_rating.level.lower()}.")

    if month_anomaly is not None and len(month_anomaly.methods) >= 2:
        sentences.append("Recommendation: review heating settings and appliance usage around this date.")
    else:
        sentences.append("No action recommended.")

    return " ".join(sentences)
