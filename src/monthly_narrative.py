"""Deterministic plain-language narrative for one month comparison.

Assembled purely from values already computed in ``src.monthly_comparison``
-- every sentence traces to a number, none is generated freely. Wording is
direct household English ("You used 9% less energy than June 2025", "Gas
caused most of the decrease"), never analyst filler ("the analysis
suggests", "it is important to note").

Interpretation rules (documented thresholds in
``config.MonthComparisonThresholds``):

- Lower usage in colder weather, weather-adjusted also lower -> a
  meaningful improvement.
- Higher usage, unexplained gap within the fit's residual noise -> broadly
  explained by colder weather.
- Higher usage with a meaningful positive unexplained gap -> deserves
  investigation; the action names the dominant fuel.
- |change| below ``little_change_pct`` -> "little change", never narrated
  as meaningful, no action.

No claim of statistical significance is ever made -- a single month's
change isn't formally testable at this sample size, and the limitation
line says what monthly billing data can't show.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.confidence import Confidence
from src.monthly_comparison import FuelContributions, MonthlyComparison, weather_explains_most

_CONFIDENCE_RANK: dict[Confidence, int] = {"High": 3, "Medium": 2, "Low": 1}

_BILLING_LIMITATION = (
    "Monthly billing data cannot show which day, room, or appliance caused a change."
)
_NO_WEATHER_LIMITATION = (
    "Weather adjustment is off, so how much of this change the weather explains is unknown."
)


@dataclass(frozen=True)
class MonthlyComparisonNarrative:
    """The structured answer: what changed, why, is it unusual, what to do.

    ``narrative`` is the assembled primary paragraph (kept under 80 words by
    construction); the individual parts are exposed so surfaces can render
    them separately (cards, Consultant breakdowns) without re-deriving.
    """

    headline: str  # one sentence, under 25 words
    narrative: str  # the assembled primary paragraph
    what_changed: str
    which_fuel: str | None
    weather_effect: str | None
    is_unusual: str | None
    action: str | None
    confidence: Confidence
    confidence_reason: str
    limitation: str


def _fuel_noun(fuel: str) -> str:
    return {"total": "energy", "electricity": "electricity", "gas": "gas"}.get(fuel, "energy")


def _comparison_phrase(comparison: MonthlyComparison) -> str:
    if comparison.comparison_mode == "typical_month":
        return f"a typical {comparison.selected_month.strftime('%B')}"
    if comparison.comparison_month is not None:
        return comparison.comparison_month.strftime("%B %Y")
    return "the comparison period"


def _what_changed(comparison: MonthlyComparison) -> str:
    noun = _fuel_noun(comparison.fuel)
    pct = comparison.percentage_change
    if pct is None:
        return comparison.interpretation  # the honest-fallback sentence
    target = _comparison_phrase(comparison)
    if comparison.change_category == "little":
        return f"You used about the same {noun} as {target} ({pct:+.0f}%)."
    direction = "more" if pct >= 0 else "less"
    return f"You used {abs(pct):.0f}% {direction} {noun} than {target}."


def _which_fuel(contributions: FuelContributions | None, pct: float | None) -> str | None:
    if contributions is None or contributions.dominant_fuel is None or pct is None:
        return None
    dominant = contributions.dominant_fuel
    dominant_change = (
        contributions.gas_change_kwh if dominant == "gas" else contributions.electricity_change_kwh
    )
    other_change = (
        contributions.electricity_change_kwh if dominant == "gas" else contributions.gas_change_kwh
    )
    verb = "decrease" if contributions.total_change_kwh < 0 else "increase"
    # Fuels moving in opposite directions deserve saying so, not a share percentage
    # that would exceed 100%.
    if dominant_change * contributions.total_change_kwh < 0:
        return None
    share = (
        contributions.gas_share_pct if dominant == "gas" else contributions.electricity_share_pct
    )
    if share is None:
        return None
    other = "electricity" if dominant == "gas" else "gas"
    if other_change * contributions.total_change_kwh < 0 and abs(other_change) >= 0.15 * abs(dominant_change):
        other_verb = "fell" if other_change < 0 else "rose"
        return f"{dominant.capitalize()} drove the {verb}; {other} use actually {other_verb}."
    if abs(other_change) < 0.15 * abs(dominant_change):
        return f"{dominant.capitalize()} caused almost all of the {verb}; {other} barely moved."
    if share >= 60:
        return f"{dominant.capitalize()} caused most of the {verb}."
    return f"Gas and electricity contributed roughly equally to the {verb}."


def _weather_effect(comparison: MonthlyComparison) -> str | None:
    if comparison.weather_explained_change_kwh is None or comparison.percentage_change is None:
        return None
    hdd_cur, hdd_cmp = comparison.current_heating_degree_days, comparison.comparison_heating_degree_days
    colder = hdd_cur is not None and hdd_cmp is not None and hdd_cur > hdd_cmp
    milder = hdd_cur is not None and hdd_cmp is not None and hdd_cur < hdd_cmp
    if comparison.change_category == "little":
        return None
    if comparison.percentage_change < 0:
        if colder:
            return "The weather was colder, so the reduction is meaningful."
        if milder:
            return "Milder weather explains part of the reduction."
        return None
    if comparison.unexplained_change_is_meaningful:
        return "Usage stayed high after accounting for weather."
    if comparison.unexplained_change_is_meaningful is False:
        weather_explained = comparison.weather_explained_change_kwh
        if weather_explains_most(weather_explained, comparison.absolute_change_kwh, colder):
            return "The increase is broadly explained by colder weather."
        if colder and weather_explained is not None and weather_explained > 0:
            return "Colder weather explains part of the increase; the rest is within this home's normal variation."
        return "The increase is not weather-related, but it is within this home's normal variation."
    return None


def _is_unusual(comparison: MonthlyComparison) -> str | None:
    if comparison.same_month_median_kwh is None or comparison.same_month_low_kwh is None:
        return None
    month_name = comparison.selected_month.strftime("%B")
    kwh = comparison.current_consumption_kwh
    if kwh < comparison.same_month_low_kwh:
        return f"This is your lowest {month_name} on record."
    if comparison.same_month_high_kwh is not None and kwh > comparison.same_month_high_kwh:
        return f"This is your highest {month_name} on record."
    return f"This month was within the normal {month_name} range."


def _record_position(comparison: MonthlyComparison) -> str | None:
    """'low'/'high' when the month sits outside every previous same-calendar-month, else None."""
    if comparison.same_month_low_kwh is None or comparison.same_month_high_kwh is None:
        return None
    if comparison.current_consumption_kwh < comparison.same_month_low_kwh:
        return "low"
    if comparison.current_consumption_kwh > comparison.same_month_high_kwh:
        return "high"
    return None


def _action(comparison: MonthlyComparison, contributions: FuelContributions | None) -> str | None:
    """A recommendation only when the evidence threshold is met -- otherwise the honest
    'no action' line, or None when there's nothing to judge."""
    pct = comparison.percentage_change
    if pct is None:
        return None
    month_name = comparison.selected_month.strftime("%B")
    record = _record_position(comparison)
    if comparison.change_category == "little" or pct < 0:
        # Never claim "within the normal range" for a month that sets a record.
        if record == "low":
            return f"No action is needed. This sets a new low for {month_name}."
        if record == "high":
            return "No action is needed, but this is still the highest " f"{month_name} on record."
        return f"No action is needed. This month remains within the normal range for {month_name}."
    if comparison.unexplained_change_is_meaningful:
        dominant = contributions.dominant_fuel if contributions else None
        if dominant == "gas" or (dominant is None and comparison.fuel == "gas"):
            return "Review heating use. Gas consumption stayed high after accounting for weather."
        if dominant == "electricity" or (dominant is None and comparison.fuel == "electricity"):
            return "Review appliance use or time spent at home. The increase did not come from heating weather."
        return "Review this month's usage. It stayed high after accounting for weather."
    if comparison.unexplained_change_is_meaningful is False:
        hdd_cur = comparison.current_heating_degree_days
        hdd_cmp = comparison.comparison_heating_degree_days
        colder = hdd_cur is not None and hdd_cmp is not None and hdd_cur > hdd_cmp
        if weather_explains_most(comparison.weather_explained_change_kwh, comparison.absolute_change_kwh, colder):
            return f"No action is needed. The increase is consistent with the weather for {month_name}."
        if record == "high":
            # The noise threshold is dominated by winter variation; a record same-calendar
            # month still deserves a glance even when the gap sits inside that noise.
            return (
                f"Worth a quick look: this is your highest {month_name} on record, although the "
                "change is within this home's overall month-to-month variation."
            )
        return "No action is needed. The change is within this home's normal variation."
    # No weather model: an increase alone isn't evidence enough to prescribe anything.
    if comparison.change_category == "large":
        return "Turn on weather adjustment to see whether colder weather explains this increase."
    return None


def _headline(comparison: MonthlyComparison, contributions: FuelContributions | None) -> str:
    what = _what_changed(comparison)
    fuel_part = _which_fuel(contributions, comparison.percentage_change)
    if fuel_part is not None and comparison.change_category != "little":
        # "You used 11% less energy than June 2025, mainly because gas use fell."
        dominant = contributions.dominant_fuel if contributions else None
        if dominant is not None and comparison.percentage_change is not None:
            verb = "fell" if comparison.percentage_change < 0 else "rose"
            return what[:-1] + f", mainly because {dominant} use {verb}."
    return what


def build_monthly_comparison_narrative(
    comparison: MonthlyComparison,
    contributions: FuelContributions | None = None,
    weather_enabled: bool = True,
) -> MonthlyComparisonNarrative:
    """Assemble the deterministic narrative for one comparison.

    ``contributions`` (per-fuel split of a combined-energy change) is
    optional -- without it the fuel sentence is simply omitted, never
    guessed. ``weather_enabled`` only changes the limitation line.
    """
    what_changed = _what_changed(comparison)
    which_fuel = _which_fuel(contributions, comparison.percentage_change)
    weather_effect = _weather_effect(comparison)
    is_unusual = _is_unusual(comparison)
    action = _action(comparison, contributions)

    parts = [what_changed]
    if which_fuel:
        parts.append(which_fuel)
    if weather_effect:
        parts.append(weather_effect)
    # The action often restates the record line ("...highest June on record...") --
    # keep whichever is more specific, never both.
    if is_unusual and not (action and "on record" in action):
        parts.append(is_unusual)
    if action:
        # The action's justification sentence often restates the weather sentence
        # ("...within this home's normal variation.") -- keep only its first
        # sentence/clause in the flowing narrative when the reason is already stated.
        condensed = action.split(". ")[0].split(", although")[0].rstrip(".") + "."
        parts.append(condensed if weather_effect else action)
    narrative = " ".join(parts)

    limitation = _BILLING_LIMITATION if weather_enabled else (
        _NO_WEATHER_LIMITATION + " " + _BILLING_LIMITATION
    )

    return MonthlyComparisonNarrative(
        headline=_headline(comparison, contributions),
        narrative=narrative,
        what_changed=what_changed,
        which_fuel=which_fuel,
        weather_effect=weather_effect,
        is_unusual=is_unusual,
        action=action,
        confidence=comparison.confidence,
        confidence_reason=comparison.confidence_reason,
        limitation=limitation,
    )
