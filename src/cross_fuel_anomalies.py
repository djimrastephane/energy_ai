"""Task 6 -- cross-fuel anomaly attribution.

Split out of ``src.comparisons`` to stay under the ~300-line guideline (same
reactive-split pattern used for ``charts.py`` -> ``charts_forecast.py``). Cross-references each fuel's independently-run anomaly detection
(``src.anomalies.detect_anomalies``, called once per fuel elsewhere) to say
*which* fuel is responsible for a flagged month, not just that something
unusual happened. No new anomaly-detection logic here -- this only
classifies results that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.anomalies import Anomaly
from src.confidence import Confidence, rate_anomaly
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType

_RESIDUAL_MATERIAL_Z = 1.5  # "materially exceeds weather-adjusted expectations" threshold


@dataclass
class CrossFuelAnomalyInsight:
    month: pd.Timestamp
    pattern: str  # "electricity_only" | "gas_only" | "both" | "total_only"
    narrative: str
    confidence: Confidence
    evidence: list[str]


_PATTERN_LABELS = {
    "electricity_only": "Electricity flagged, gas unchanged",
    "gas_only": "Gas flagged, electricity unchanged",
    "both": "Both fuels flagged",
    "total_only": "Total flagged (fuel-level detail unavailable)",
}


def _residual_evidence(
    fuel: str, date: pd.Timestamp, energy_result: EnergySignatureResult | None
) -> tuple[str | None, bool]:
    """One evidence line plus whether the residual is 'material' (|z| > threshold) for ``fuel``
    at ``date``, using the already-fitted weather regression's residual series. Returns
    (None, False) if weather adjustment wasn't run or this month isn't in the fitted series."""
    if energy_result is None or date not in energy_result.resid.index:
        return None, False
    resid_kwh = float(energy_result.resid.loc[date])
    resid_std = float(energy_result.resid.std())
    if resid_std <= 0:
        return f"{fuel.capitalize()} residual after weather adjustment: {resid_kwh:+.1f} kWh/day", False
    z = resid_kwh / resid_std
    material = abs(z) > _RESIDUAL_MATERIAL_Z
    pct_of_spread = z * 100
    return (
        f"{fuel.capitalize()} residual after weather adjustment: {resid_kwh:+.2f} kWh/day "
        f"({pct_of_spread:+.0f}% of the typical month-to-month spread)",
        material,
    )


def _fuel_index(fuel_clean_dfs: dict[EnergyType, pd.DataFrame], fuel: EnergyType) -> pd.DataFrame:
    df = fuel_clean_dfs.get(fuel, pd.DataFrame())
    return df.set_index("month_start") if not df.empty else df


def cross_fuel_anomaly_insights(
    fuel_anomalies: dict[EnergyType, list[Anomaly]],
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame],
    fuel_energy_results: dict[EnergyType, EnergySignatureResult] | None = None,
) -> list[CrossFuelAnomalyInsight]:
    """Classify every month flagged in any fuel's anomaly list.

      - electricity flagged, gas not -> likely appliance/occupancy change.
      - gas flagged, electricity not -> likely a heating event.
      - both flagged -> if weather adjustment is available (``fuel_energy_results``) and
        both fuels' residuals are within the "material" threshold, classified as
        weather-driven (the raw spike is consistent with what colder weather alone would
        predict); if either residual still materially exceeds the weather-adjusted
        expectation, classified as partially unexplained rather than forced into a
        weather-driven conclusion; if weather adjustment wasn't run at all, classified as
        Medium confidence "consistent with a single external driver" without claiming to
        have confirmed it.
      - only Total flagged (no electricity/gas breakdown available) -> fuel-level
        attribution isn't possible from this data alone.
    """
    fuel_energy_results = fuel_energy_results or {}
    elec_by_date = {a.date: a for a in fuel_anomalies.get("electricity", [])}
    gas_by_date = {a.date: a for a in fuel_anomalies.get("gas", [])}
    total_by_date = {a.date: a for a in fuel_anomalies.get("total", [])}
    elec_df = _fuel_index(fuel_clean_dfs, "electricity")
    gas_df = _fuel_index(fuel_clean_dfs, "gas")

    all_dates = sorted(set(elec_by_date) | set(gas_by_date) | set(total_by_date))
    insights: list[CrossFuelAnomalyInsight] = []

    for date in all_dates:
        elec_flagged = date in elec_by_date
        gas_flagged = date in gas_by_date
        evidence: list[str] = []

        if elec_flagged and date in elec_df.index:
            evidence.append(
                f"Electricity: {elec_df.loc[date, 'consumption_kwh']:.0f} kWh "
                f"({elec_by_date[date].direction}), flagged by {', '.join(elec_by_date[date].methods)}"
            )
        if gas_flagged and date in gas_df.index:
            evidence.append(
                f"Gas: {gas_df.loc[date, 'consumption_kwh']:.0f} kWh "
                f"({gas_by_date[date].direction}), flagged by {', '.join(gas_by_date[date].methods)}"
            )

        if elec_flagged and gas_flagged:
            pattern = "both"
            elec_a, gas_a = elec_by_date[date], gas_by_date[date]

            if elec_a.direction != gas_a.direction:
                # Not a shared driver -- one fuel spiked while the other dropped. Attribute to
                # whichever signal is materially stronger (more methods agreeing) rather than
                # forcing a "weather-driven" conclusion onto a conflict.
                evidence.append(f"Directions conflict: electricity {elec_a.direction}, gas {gas_a.direction}")
                if len(gas_a.methods) != len(elec_a.methods):
                    stronger_fuel, stronger_a, weaker_a = (
                        ("gas", gas_a, elec_a) if len(gas_a.methods) > len(elec_a.methods) else ("electricity", elec_a, gas_a)
                    )
                    narrative_tail = (
                        f"electricity and gas moved in opposite directions ({elec_a.direction} vs "
                        f"{gas_a.direction} respectively) -- not a shared driver. The {stronger_fuel} signal "
                        f"is stronger ({len(stronger_a.methods)} method(s) vs {len(weaker_a.methods)}), so this "
                        f"looks more like a {stronger_fuel}-specific event than something affecting the whole "
                        "household."
                    )
                    confidence = rate_anomaly(stronger_a).level
                else:
                    narrative_tail = (
                        "electricity and gas moved in opposite directions with equally strong signals -- "
                        "conflicting evidence, so no single explanation is clearly supported here."
                    )
                    confidence = "Low"
            else:
                elec_evidence, elec_material = _residual_evidence(
                    "electricity", date, fuel_energy_results.get("electricity")
                )
                gas_evidence, gas_material = _residual_evidence("gas", date, fuel_energy_results.get("gas"))
                for e in (elec_evidence, gas_evidence):
                    if e:
                        evidence.append(e)

                if elec_evidence is None and gas_evidence is None:
                    narrative_tail = (
                        "both fuels moved together -- consistent with a single external driver such as "
                        "weather, though weather adjustment wasn't run so this can't be confirmed against "
                        "a weather-adjusted expectation."
                    )
                    confidence = "Medium"
                elif elec_material or gas_material:
                    narrative_tail = (
                        "both fuels moved together, and at least one still materially exceeds what the "
                        "weather-adjusted model would predict -- likely more than weather alone, worth "
                        "investigating further."
                    )
                    confidence = "Medium"
                else:
                    narrative_tail = (
                        "both fuels moved together and are within the range the weather-adjusted model "
                        "would predict -- likely weather-driven."
                    )
                    confidence = "High"
        elif elec_flagged:
            pattern = "electricity_only"
            narrative_tail = "likely an appliance or occupancy change, since gas was unaffected."
            confidence = rate_anomaly(elec_by_date[date]).level
        elif gas_flagged:
            pattern = "gas_only"
            narrative_tail = "likely a heating event, since electricity was unaffected."
            confidence = rate_anomaly(gas_by_date[date]).level
        else:
            pattern = "total_only"
            a = total_by_date[date]
            evidence.append(f"Total: {a.direction}, flagged by {', '.join(a.methods)}")
            narrative_tail = "electricity/gas-level data isn't available to attribute this to one fuel."
            confidence = rate_anomaly(a).level

        insights.append(
            CrossFuelAnomalyInsight(
                month=date,
                pattern=pattern,
                narrative=f"{date.strftime('%B %Y')}: {_PATTERN_LABELS[pattern]} -- {narrative_tail}",
                confidence=confidence,
                evidence=evidence,
            )
        )

    return insights
