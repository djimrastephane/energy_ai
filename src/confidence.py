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
            f"R-squared is only {result.r_squared:.0%} and/or no weather driver reached "
            "statistical significance -- weather doesn't reliably explain consumption here.",
        )
    if result.r_squared < 0.6 or not dw_in_range:
        reason = (
            f"R-squared is a moderate {result.r_squared:.0%}."
            if result.r_squared < 0.6
            else f"Durbin-Watson ({result.durbin_watson:.2f}) suggests nearby months' residuals "
            "are correlated, so significance should be read with some caution."
        )
        return ConfidenceRating("Medium", reason)
    return ConfidenceRating(
        "High",
        f"R-squared of {result.r_squared:.0%} with a statistically significant weather driver "
        f"and no strong residual autocorrelation (Durbin-Watson {result.durbin_watson:.2f}).",
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
            f"Cross-validated error is large relative to typical consumption "
            f"({relative_mae:.0%}) and/or based on few validation folds ({int(best['n_folds'])}).",
        )
    if relative_mae > 0.15 or band_width_ratio > 0.7:
        return ConfidenceRating(
            "Medium",
            f"Cross-validated error is moderate ({relative_mae:.0%} of typical monthly consumption).",
        )
    return ConfidenceRating(
        "High",
        f"Cross-validated error is small ({relative_mae:.0%} of typical monthly consumption) "
        f"across {int(best['n_folds'])} validation folds.",
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
