"""Shared confidence taxonomy used across findings, recommendations, and the AI Analyst report.

Defined once so "High/Medium/Low" means the same thing everywhere in the
platform. Every rating carries a ``reason`` -- the point isn't just to
label something confident, it's to say why, so a reader can judge for
themselves whether they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult
from src.preprocessing import PreprocessingReport

Confidence = Literal["High", "Medium", "Low"]


@dataclass
class ConfidenceRating:
    level: Confidence
    reason: str


def rate_data_quality(report: PreprocessingReport, n_months: int) -> ConfidenceRating:
    """Rate confidence in the underlying data itself."""
    if report.missing_months or report.conflicts or n_months < 12:
        issues = []
        if report.missing_months:
            issues.append(f"{len(report.missing_months)} missing month(s)")
        if report.conflicts:
            issues.append(f"{len(report.conflicts)} source conflict(s)")
        if n_months < 12:
            issues.append(f"only {n_months} month(s) of history")
        return ConfidenceRating("Low", "Data gaps or short history: " + ", ".join(issues) + ".")
    if report.outlier_warnings or n_months < 24:
        reason = (
            f"{len(report.outlier_warnings)} plausibility warning(s) flagged for review."
            if report.outlier_warnings
            else f"Only {n_months} months of history -- under 2 full years."
        )
        return ConfidenceRating("Medium", reason)
    return ConfidenceRating("High", f"No missing months, no conflicts, {n_months} months of clean history.")


def rate_weather_model(result: EnergySignatureResult | None) -> ConfidenceRating:
    """Rate confidence in the weather (degree-day regression) model, if one was fitted."""
    if result is None:
        return ConfidenceRating("Low", "Weather adjustment is off or the model couldn't be fitted.")

    heating_significant = not np.isnan(result.heating_pvalue) and result.heating_pvalue < 0.05
    cooling_significant = not np.isnan(result.cooling_pvalue) and result.cooling_pvalue < 0.05
    has_driver = heating_significant or cooling_significant
    dw_in_range = 1.2 <= result.durbin_watson <= 2.8

    if result.r_squared < 0.3 or not has_driver:
        return ConfidenceRating(
            "Low",
            f"Weather explains only a small part of the swing in your usage (about "
            f"{result.r_squared:.0%}), and/or no clear weather effect could be pinned down here.",
        )
    if result.r_squared < 0.6 or not dw_in_range:
        reason = (
            f"Weather explains a moderate share of your usage (about {result.r_squared:.0%}) -- "
            "not enough to be highly confident."
            if result.r_squared < 0.6
            else "There's a leftover month-to-month pattern the model doesn't fully capture, so "
            "treat this with a little more caution."
        )
        return ConfidenceRating("Medium", reason)
    return ConfidenceRating(
        "High",
        f"The model explains most of the month-to-month swing in your usage (about "
        f"{result.r_squared:.0%}), with a clear weather effect and no leftover pattern it's missing.",
    )


def rate_forecast(result: ForecastResult | None, history_mean: float) -> ConfidenceRating:
    """Rate confidence in the chosen forecast model's near-term accuracy."""
    if result is None or history_mean <= 0:
        return ConfidenceRating("Low", "No forecast is available for this data.")

    best = result.comparison.iloc[0]
    relative_mae = best["mae"] / history_mean
    band_width_ratio = float(np.mean((result.p90 - result.p10) / np.where(result.p50 == 0, np.nan, result.p50)))
    band_width_ratio = 0.0 if np.isnan(band_width_ratio) else band_width_ratio

    if relative_mae > 0.30 or band_width_ratio > 1.2 or best["n_folds"] < 6:
        return ConfidenceRating(
            "Low",
            f"This model's typical error is large next to a normal month's usage (about "
            f"{relative_mae:.0%}), and/or it's only been checked against a few past months "
            f"({int(best['n_folds'])}) -- treat it as a rough guide.",
        )
    if relative_mae > 0.15 or band_width_ratio > 0.7:
        return ConfidenceRating(
            "Medium",
            f"This model's typical error is moderate -- about {relative_mae:.0%} of a normal "
            "month's usage.",
        )
    return ConfidenceRating(
        "High",
        f"This model has been accurate in past tests -- typical error of about {relative_mae:.0%} "
        f"of a normal month's usage, checked against {int(best['n_folds'])} past months.",
    )


def rate_anomaly(anomaly: Anomaly) -> ConfidenceRating:
    """Rate confidence in a detected anomaly based on how many independent methods agree."""
    n = len(anomaly.methods)
    if n >= 3:
        return ConfidenceRating("High", "All three detection methods (rolling z-score, ESD, Isolation Forest) agree.")
    if n == 2:
        return ConfidenceRating("Medium", f"Two independent methods agree ({', '.join(anomaly.methods)}).")
    return ConfidenceRating("Low", f"Only one method flagged this ({anomaly.methods[0]}) -- treat as a lead, not a conclusion.")


def rate_changepoint(cp: ChangePoint) -> ConfidenceRating:
    """Rate confidence in a detected change point based on method agreement."""
    if cp.method == "both":
        return ConfidenceRating("High", "Both PELT and CUSUM independently agree on this date.")
    return ConfidenceRating("Medium", f"Detected by a single method ({cp.method}) only.")
