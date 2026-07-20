"""Translate computed statistics into plain-English findings with evidence and confidence.

Every Finding traces back to a number already computed elsewhere in the
platform (STL, the weather regression, change-point/anomaly detection,
forecast cross-validation) -- this module only narrates and rates
confidence, it never computes a new statistic. A builder returns ``None``
when there isn't enough evidence for a finding, rather than manufacturing
one from thin air.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.anomalies import Anomaly
from src.changepoints import ChangePoint
from src.confidence import (
    Confidence,
    rate_anomaly,
    rate_changepoint,
    rate_forecast,
    rate_weather_model,
)
from src.decomposition import STLResult, interpret_decomposition
from src.energy_signature import EnergySignatureResult
from src.forecast_evaluation import ForecastResult
from src.kpis import KPIComparison
from src.utils import get_logger

logger = get_logger(__name__)

_CONFIDENCE_RANK: dict[Confidence, int] = {"High": 3, "Medium": 2, "Low": 1}


@dataclass
class Finding:
    title: str
    narrative: str
    evidence: list[str]
    confidence: Confidence
    confidence_reason: str
    category: str


def _fraction_phrase(x: float) -> str:
    """Map a 0-1 fraction to approximate spoken English, e.g. 0.73 -> 'about three quarters'."""
    buckets = [
        (0.08, "almost none"),
        (0.20, "about a fifth"),
        (0.30, "about a third"),
        (0.45, "just under half"),
        (0.55, "about half"),
        (0.70, "roughly two thirds"),
        (0.80, "about three quarters"),
        (0.92, "the large majority"),
        (1.01, "almost all"),
    ]
    for threshold, phrase in buckets:
        if x < threshold:
            return phrase
    return "all"


def finding_yoy_trend(kpis: list[KPIComparison]) -> Finding | None:
    """Trailing 12-month vs. previous 12-month consumption -- a direct sum, not a model."""
    total = next((k for k in kpis if k.label == "Total consumption"), None)
    if total is None or total.pct_change is None:
        return None
    direction = "higher" if total.pct_change >= 0 else "lower"
    narrative = (
        f"Consumption over the last 12 months is {abs(total.pct_change):.1f}% {direction} than the "
        f"previous 12 months ({total.previous:,.0f} to {total.current:,.0f} kWh)."
    )
    return Finding(
        title="Year-on-year consumption trend",
        narrative=narrative,
        evidence=[
            f"Previous 12 months: {total.previous:,.0f} kWh",
            f"Current 12 months: {total.current:,.0f} kWh",
        ],
        confidence="High",
        confidence_reason="Directly summed from billing data, not a statistical model.",
        category="trend",
    )


def finding_seasonality(stl_result: STLResult | None) -> Finding | None:
    if stl_result is None:
        return None
    if stl_result.seasonal_strength >= 0.6:
        confidence: Confidence = "High"
        reason = f"Seasonal strength is {stl_result.seasonal_strength:.0%}, a clear, dominant calendar pattern."
    elif stl_result.seasonal_strength >= 0.3:
        confidence, reason = "Medium", f"Seasonal strength is a moderate {stl_result.seasonal_strength:.0%}."
    else:
        confidence, reason = "Low", f"Seasonal strength is only {stl_result.seasonal_strength:.0%} -- the calendar pattern is weak."
    return Finding(
        title="Seasonal pattern",
        narrative=interpret_decomposition(stl_result),
        evidence=[
            f"Seasonal strength: {stl_result.seasonal_strength:.0%}",
            f"Trend strength: {stl_result.trend_strength:.0%}",
        ],
        confidence=confidence,
        confidence_reason=reason,
        category="seasonality",
    )


_FUEL_NOUN = {"electricity": "electricity", "gas": "gas", "total": "energy"}
_HEATING_ATTRIBUTION = {
    "electricity": "consistent with some electric heating in this home",
    "gas": "consistent with gas heating in this home",
}
_TOTAL_HEATING_NOTE = (
    " This combines electricity and gas -- switch the sidebar's Fuel selector to Electricity or "
    "Gas only to see which fuel is driving it."
)


def finding_weather(
    energy_result: EnergySignatureResult | None, unit_rate: float, fuel: str = "total"
) -> Finding | None:
    if energy_result is None:
        return None
    rating = rate_weather_model(energy_result)
    fuel_noun = _FUEL_NOUN.get(fuel, "energy")
    narrative = f"Weather explains {_fraction_phrase(energy_result.r_squared)} of the variation in {fuel_noun} consumption."

    heating_significant = not np.isnan(energy_result.heating_pvalue) and energy_result.heating_pvalue < 0.05
    if heating_significant:
        heat_cost = energy_result.heating_slope * unit_rate
        heating_amount = (
            f" Each additional heating-degree-day is associated with an extra "
            f"{energy_result.heating_slope:.2f} kWh/day (about £{heat_cost:.2f}/day)"
        )
        if fuel in _HEATING_ATTRIBUTION:
            narrative += f"{heating_amount}, {_HEATING_ATTRIBUTION[fuel]}."
        else:
            narrative += f"{heating_amount}.{_TOTAL_HEATING_NOTE}"
    else:
        narrative += " No statistically significant heating sensitivity was detected."

    evidence = [f"R-squared: {energy_result.r_squared:.0%}"]
    evidence.append(
        f"Heating sensitivity p-value: {energy_result.heating_pvalue:.3f}"
        if not np.isnan(energy_result.heating_pvalue)
        else "Heating sensitivity: not estimable (no variation in heating-degree-days)"
    )
    return Finding(
        title="Weather sensitivity",
        narrative=narrative,
        evidence=evidence,
        confidence=rating.level,
        confidence_reason=rating.reason,
        category="weather",
    )


def finding_biggest_change(
    anomalies: list[Anomaly],
    changepoints: list[ChangePoint],
    energy_result: EnergySignatureResult | None,
) -> Finding | None:
    """Pick the single most important unusual event, preferring cross-validated anomalies.

    ``anomalies`` is expected pre-sorted by confidence then date (as
    ``src.anomalies.detect_anomalies`` already returns it), so the first
    entry is the strongest lead. Change points are used only as a fallback
    when no anomalies were detected at all.
    """
    if anomalies:
        top = anomalies[0]
        rating = rate_anomaly(top)
        weather_phrase = "typical seasonal patterns"
        if energy_result is not None and top.date in energy_result.resid.index:
            weather_resid = energy_result.resid.loc[top.date]
            resid_std = energy_result.resid.std()
            if resid_std > 0 and abs(weather_resid) > resid_std:
                weather_phrase = "weather"
        verb = "increase" if top.direction == "spike" else "decrease"
        narrative = (
            f"{top.date.strftime('%B %Y')} remains the strongest unexplained {verb} after adjusting for "
            f"{weather_phrase} -- {top.rank_context}."
        )
        return Finding(
            title="Biggest finding",
            narrative=narrative,
            evidence=[f"Detected by: {', '.join(top.methods)}", top.rank_context],
            confidence=rating.level,
            confidence_reason=rating.reason,
            category="anomaly",
        )

    if changepoints:
        both = [c for c in changepoints if c.method == "both"]
        top_cp = max(both, key=lambda c: abs(c.magnitude_kwh)) if both else max(changepoints, key=lambda c: abs(c.magnitude_kwh))
        rating = rate_changepoint(top_cp)
        narrative = (
            f"A sustained {top_cp.direction} in consumption began around {top_cp.date.strftime('%B %Y')} "
            f"(change of about {abs(top_cp.magnitude_kwh):.0f} kWh/month)."
        )
        return Finding(
            title="Biggest finding",
            narrative=narrative,
            evidence=[f"Detected by: {top_cp.method}", f"Magnitude: {top_cp.magnitude_kwh:+.0f} kWh"],
            confidence=rating.level,
            confidence_reason=rating.reason,
            category="changepoint",
        )

    return None


def finding_forecast_outlook(forecast_result: ForecastResult | None, history_mean: float) -> Finding | None:
    if forecast_result is None:
        return None
    rating = rate_forecast(forecast_result, history_mean)
    total_p50 = float(forecast_result.p50.sum())
    total_p10 = float(forecast_result.p10.sum())
    total_p90 = float(forecast_result.p90.sum())
    horizon = len(forecast_result.forecast_dates)
    narrative = (
        f"Over the next {horizon} month(s), consumption is most likely to total about {total_p50:,.0f} kWh "
        f"(plausible range {total_p10:,.0f}-{total_p90:,.0f} kWh), based on the {forecast_result.model_name} model."
    )
    return Finding(
        title="Forecast outlook",
        narrative=narrative,
        evidence=[
            f"Model: {forecast_result.model_name}",
            f"Cross-validated MAE: {forecast_result.comparison.iloc[0]['mae']:.1f} kWh",
        ],
        confidence=rating.level,
        confidence_reason=rating.reason,
        category="forecast",
    )


def generate_findings(
    kpis: list[KPIComparison],
    stl_result: STLResult | None,
    energy_result: EnergySignatureResult | None,
    unit_rate: float,
    anomalies: list[Anomaly],
    changepoints: list[ChangePoint],
    forecast_result: ForecastResult | None,
    history_mean: float,
    fuel: str = "total",
) -> list[Finding]:
    """Run every finding builder, drop the ones with insufficient evidence, sort by confidence.

    ``fuel`` is one of ``"total"``/``"electricity"``/``"gas"`` -- passed through to
    :func:`finding_weather` so its narrative names the fuel ``energy_result`` was actually fitted on.
    """
    candidates = [
        finding_yoy_trend(kpis),
        finding_biggest_change(anomalies, changepoints, energy_result),
        finding_weather(energy_result, unit_rate, fuel),
        finding_seasonality(stl_result),
        finding_forecast_outlook(forecast_result, history_mean),
    ]
    findings = [f for f in candidates if f is not None]
    findings.sort(key=lambda f: _CONFIDENCE_RANK[f.confidence], reverse=True)
    return findings
