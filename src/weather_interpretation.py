"""Combine weather context with energy evidence into a hedged, deterministic interpretation.

For an unusual month, this joins what :mod:`src.weather_context` observed
(snow/wind/rain facts) with what the energy analysis already computed (the
anomaly direction, per-fuel flags, and the temperature-model residual) into
one of a small set of interpretation patterns. The output always separates
observed facts from plausible interpretation, uses hedged wording ("may
have contributed", "is consistent with", "cannot be confirmed from monthly
data"), and never asserts home-working, road closures, or occupancy changes
as fact -- monthly bills cannot show those.

Pattern reference (high-usage direction; low-usage months get mirrored wording):

- A: severe weather + residual within expectation -> broadly consistent with weather.
- B: severe weather + residual still materially positive -> weather may have
  contributed, but doesn't fully explain it.
- C: electricity flagged while gas stable during severe weather -> not primarily
  heating-driven; time at home or appliance use may have contributed (unconfirmable).
- D: no severe weather + material residual -> weather doesn't explain it; review
  occupancy, heating settings, or appliance changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.anomalies import Anomaly
from src.confidence import Confidence
from src.energy_signature import EnergySignatureResult
from src.weather_context import WeatherContextClassification

# Same "materially exceeds weather-adjusted expectations" bar as src.cross_fuel_anomalies,
# so "material" means one thing across the app.
_RESIDUAL_MATERIAL_Z = 1.5

_LIMITATION = (
    "Monthly billing data cannot confirm behavioural causes -- for example whether anyone "
    "worked from home, stayed in, or changed heating settings. Weather facts here are "
    "context that can support such a hypothesis, never proof of it."
)


@dataclass
class UnusualMonthInterpretation:
    month_start: pd.Timestamp
    headline: str  # the one-paragraph plausible interpretation (hedged)
    weather_facts: list[str]  # observed facts, from the weather-context classification
    energy_evidence: list[str]  # numbers from the energy analysis
    residual_monthly_kwh: float | None  # actual minus temperature-model expectation, whole month
    confidence: Confidence
    limitation: str


def _monthly_residual_kwh(
    date: pd.Timestamp, energy_result: EnergySignatureResult | None, merged: pd.DataFrame | None
) -> tuple[float | None, bool]:
    """(residual in kWh for the whole month, is it material) from the fitted model's own
    residual series (kWh/day), scaled by that month's day count. (None, False) when weather
    adjustment wasn't run or the month isn't in the fitted series."""
    if energy_result is None or merged is None or date not in energy_result.resid.index:
        return None, False
    resid_per_day = float(energy_result.resid.loc[date])
    match = merged.loc[merged["month_start"] == date, "days_in_month"]
    days = int(match.iloc[0]) if not match.empty else date.days_in_month
    resid_std = float(energy_result.resid.std())
    material = resid_std > 0 and abs(resid_per_day) / resid_std > _RESIDUAL_MATERIAL_Z
    return resid_per_day * days, material


def interpret_unusual_month(
    anomaly: Anomaly,
    classification: WeatherContextClassification | None,
    energy_result: EnergySignatureResult | None,
    merged: pd.DataFrame | None,
    electricity_flagged: bool | None = None,
    gas_flagged: bool | None = None,
) -> UnusualMonthInterpretation:
    """Deterministic pattern A-D interpretation for one flagged month.

    ``electricity_flagged``/``gas_flagged`` say whether each fuel's own anomaly detection
    flagged this month (``None`` = fuel-level data unavailable). ``energy_result``/``merged``
    are the already-fitted temperature model for the *selected* fuel, or ``None`` with
    weather adjustment off.
    """
    date = anomaly.date
    month_label = date.strftime("%B %Y")
    higher = anomaly.direction == "spike"
    direction_word = "higher" if higher else "lower"

    residual_kwh, residual_material = _monthly_residual_kwh(date, energy_result, merged)
    severe = classification is not None and classification.is_notable
    weather_facts = list(classification.facts) if classification is not None else []

    energy_evidence = [
        f"Flagged as a {anomaly.direction} by {len(anomaly.methods)} of 3 detection methods "
        f"({', '.join(anomaly.methods)})",
        anomaly.rank_context,
    ]
    if residual_kwh is not None:
        energy_evidence.append(
            f"Consumption vs. the temperature model's expectation: {residual_kwh:+,.0f} kWh for the month"
        )

    confidence: Confidence = "Medium" if len(anomaly.methods) >= 2 else "Low"

    if classification is None:
        headline = (
            f"{month_label} was {direction_word} than expected, but weather context is "
            "unavailable (weather adjustment is off or no weather data covers this month), "
            "so weather's contribution can't be assessed."
        )
        return UnusualMonthInterpretation(date, headline, weather_facts, energy_evidence, residual_kwh, "Low", _LIMITATION)

    electricity_only = bool(electricity_flagged) and gas_flagged is False

    if severe and higher and electricity_only:
        # Pattern C: not heating-shaped, despite severe weather.
        headline = (
            f"In {month_label}, electricity rose while gas stayed in its normal range, so the "
            "increase does not look primarily heating-driven even though the month had "
            f"{classification.label.lower()}. More time spent at home or extra appliance use "
            "may have contributed, but this cannot be confirmed from monthly data."
        )
    elif severe and higher and not residual_material:
        # Pattern A: weather explains it well.
        headline = (
            f"The increase in {month_label} is broadly consistent with colder and severe "
            f"winter weather ({classification.label.lower()}): after accounting for "
            "temperature, consumption was close to what the model expected."
        )
        confidence = "High" if len(anomaly.methods) >= 2 else "Medium"
    elif severe and higher and residual_material:
        # Pattern B: weather contributed, but doesn't fully explain.
        extra = f" ({residual_kwh:+,.0f} kWh above the temperature-adjusted expectation)" if residual_kwh else ""
        headline = (
            f"{month_label} had {classification.label.lower()}, which may have contributed to "
            f"higher usage -- but consumption remained materially above what the temperature "
            f"model expected{extra}, so severe weather alone does not explain the full increase."
        )
    elif not severe and residual_material:
        # Pattern D: no weather story at all.
        headline = (
            f"No notable severe weather was recorded in {month_label}, and consumption was "
            f"materially {direction_word} than the temperature model expected -- weather does "
            "not explain this month well. Reviewing occupancy, heating settings, or appliance "
            "changes would be more productive than a weather explanation."
        )
    elif severe and not higher:
        headline = (
            f"{month_label} had {classification.label.lower()}, yet usage was *lower* than "
            "usual -- severe weather would normally push heating demand up, so the weather "
            "does not explain this month's drop. An absence from home may be consistent with "
            "this pattern, but cannot be confirmed from monthly data."
        )
    else:
        headline = (
            f"{month_label} was {direction_word} than usual with no notable severe weather; "
            "after temperature adjustment the deviation is within the model's normal spread, "
            "so no strong weather-based explanation applies."
        )

    return UnusualMonthInterpretation(
        date, headline, weather_facts, energy_evidence, residual_kwh, confidence, _LIMITATION
    )
