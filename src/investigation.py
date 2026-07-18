"""Investigation checklists for anomalies and change points.

When something unusual is detected, this builds a "possible causes"
checklist and ticks items only when the available data actually supports
them. Most items legitimately stay unchecked: billing (plus, optionally,
weather) data alone can't see appliances, occupancy, or guests. "Unknown"
is an honest, expected answer here, not a failure of the system.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.energy_signature import EnergySignatureResult

_UNDETERMINABLE_REASON = "Not determinable from billing data alone."
_UNDETERMINABLE_LABELS = ["New appliance", "Holiday occupancy", "Guests", "Home working"]


@dataclass
class InvestigationItem:
    label: str
    checked: bool
    reason: str


@dataclass
class InvestigationChecklist:
    date: pd.Timestamp
    items: list[InvestigationItem]

    @property
    def checked_items(self) -> list[InvestigationItem]:
        return [i for i in self.items if i.checked]


def _check_colder_weather(
    date: pd.Timestamp, merged_indexed: pd.DataFrame | None, energy_result: EnergySignatureResult | None
) -> InvestigationItem:
    label = "Colder weather"
    if merged_indexed is None or energy_result is None:
        return InvestigationItem(label, False, "Weather adjustment is off, so this can't be checked.")
    if date not in merged_indexed.index:
        return InvestigationItem(label, False, "No weather data available for this month.")
    prior_year_date = date - pd.DateOffset(years=1)
    if prior_year_date not in merged_indexed.index:
        return InvestigationItem(label, False, "No prior-year weather data available for comparison.")

    current_hdd = merged_indexed.loc[date, "avg_daily_hdd"]
    prior_hdd = merged_indexed.loc[prior_year_date, "avg_daily_hdd"]
    heating_significant = not pd.isna(energy_result.heating_pvalue) and energy_result.heating_pvalue < 0.05

    if heating_significant and current_hdd > prior_hdd * 1.1:
        return InvestigationItem(
            label,
            True,
            f"Heating-degree-days this month ({current_hdd:.1f}/day) are notably higher than the same "
            f"month last year ({prior_hdd:.1f}/day), and heating sensitivity is statistically significant.",
        )
    return InvestigationItem(
        label,
        False,
        "Heating-degree-days aren't notably higher than the same month last year, or heating "
        "sensitivity isn't statistically significant.",
    )


def _check_tariff_change(date: pd.Timestamp, clean_indexed: pd.DataFrame) -> InvestigationItem:
    label = "Tariff change"
    if date not in clean_indexed.index:
        return InvestigationItem(label, False, "No billing data available for this month.")

    prior_months = clean_indexed.loc[:date].iloc[:-1].tail(6)
    if len(prior_months) < 3:
        return InvestigationItem(label, False, "Not enough prior months to establish a baseline unit rate.")

    baseline_rate = prior_months["unit_rate_gbp_per_kwh"].mean()
    current_rate = clean_indexed.loc[date, "unit_rate_gbp_per_kwh"]
    if baseline_rate <= 0:
        return InvestigationItem(label, False, "Baseline unit rate is not available.")

    deviation = abs(current_rate - baseline_rate) / baseline_rate
    if deviation > 0.15:
        return InvestigationItem(
            label,
            True,
            f"Unit rate this month (£{current_rate:.3f}/kWh) differs from the trailing 6-month "
            f"average (£{baseline_rate:.3f}/kWh) by {deviation:.0%}.",
        )
    return InvestigationItem(
        label, False, f"Unit rate is within {deviation:.0%} of the trailing average -- no evidence of a tariff change."
    )


_HEATING_ATTRIBUTION = {
    "electricity": "consistent with some electric heating",
    "gas": "consistent with gas heating",
    "total": "this combined (electricity + gas) view can't say which fuel -- check Electricity/Gas only",
}


def _check_electric_heating(
    energy_result: EnergySignatureResult | None, fuel: str = "total"
) -> InvestigationItem:
    label = "Heating sensitivity"
    if energy_result is None:
        return InvestigationItem(label, False, "Weather adjustment is off, so this can't be checked.")
    heating_significant = not pd.isna(energy_result.heating_pvalue) and energy_result.heating_pvalue < 0.05
    if heating_significant:
        attribution = _HEATING_ATTRIBUTION.get(fuel, _HEATING_ATTRIBUTION["total"])
        return InvestigationItem(
            label,
            True,
            f"This home shows statistically significant heating sensitivity "
            f"({energy_result.heating_slope:.2f} kWh/day per heating-degree-day, "
            f"p={energy_result.heating_pvalue:.3f}), {attribution}.",
        )
    return InvestigationItem(label, False, "No statistically significant heating sensitivity detected for this home.")


def build_investigation_checklist(
    date: pd.Timestamp,
    clean_df: pd.DataFrame,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    fuel: str = "total",
) -> InvestigationChecklist:
    """Build the "possible causes" checklist for one flagged month.

    ``clean_df`` is the standard monthly frame from
    ``src.preprocessing.run_pipeline``; ``merged_df``/``energy_result`` are
    the weather-merged frame and fitted model from ``src.weather``/
    ``src.energy_signature`` (pass ``None`` for both if weather adjustment
    is off -- every weather-dependent item then honestly reports itself as
    unchecked rather than guessing). ``fuel`` is one of
    ``"total"``/``"electricity"``/``"gas"`` -- which fuel ``energy_result``
    was actually fitted on, so the heating-sensitivity item names the right one.
    """
    clean_indexed = clean_df.sort_values("month_start").set_index("month_start")
    merged_indexed = (
        merged_df.sort_values("month_start").set_index("month_start") if merged_df is not None else None
    )

    items = [
        _check_colder_weather(date, merged_indexed, energy_result),
        _check_tariff_change(date, clean_indexed),
        InvestigationItem(_UNDETERMINABLE_LABELS[0], False, _UNDETERMINABLE_REASON),
        _check_electric_heating(energy_result, fuel),
        *[InvestigationItem(label, False, _UNDETERMINABLE_REASON) for label in _UNDETERMINABLE_LABELS[1:]],
    ]

    if any(item.checked for item in items):
        items.append(InvestigationItem("Unknown", False, "At least one other cause is supported by the data."))
    else:
        items.append(
            InvestigationItem("Unknown", True, "None of the checkable causes are supported by the available data.")
        )

    return InvestigationChecklist(date=date, items=items)
