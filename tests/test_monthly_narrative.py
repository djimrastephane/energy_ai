"""Tests for the deterministic month-comparison narrative (src.monthly_narrative).

Each interpretation rule from the design brief gets a scenario; wording is
asserted on the phrases that carry the judgement, not full strings, so
copy-editing stays cheap but a rule regression still fails loudly.
"""

import dataclasses

import pandas as pd
import pytest

from src.monthly_comparison import FuelContributions, MonthlyComparison
from src.monthly_narrative import build_monthly_comparison_narrative

_BANNED_PHRASES = [
    "consumption is running at expected levels",
    "this indicates",
    "it is important to note",
    "based on the data",
    "the analysis suggests",
    "you may wish to consider",
]


def _comparison(**overrides) -> MonthlyComparison:
    """A baseline June-vs-June comparison; tests override just the fields they exercise."""
    base = dict(
        selected_month=pd.Timestamp("2026-06-01"),
        comparison_mode="same_month_last_year",
        comparison_month=pd.Timestamp("2025-06-01"),
        fuel="total",
        current_consumption_kwh=230.0,
        comparison_consumption_kwh=200.0,
        absolute_change_kwh=30.0,
        percentage_change=15.0,
        current_avg_daily_kwh=230.0 / 30,
        comparison_avg_daily_kwh=200.0 / 30,
        current_cost_gbp=46.0,
        comparison_cost_gbp=40.0,
        cost_change_gbp=6.0,
        cost_change_from_usage_gbp=6.0,
        cost_change_from_rate_gbp=0.0,
        current_weather_adjusted_kwh=None,
        comparison_weather_adjusted_kwh=None,
        weather_explained_change_kwh=None,
        unexplained_change_kwh=None,
        unexplained_change_is_meaningful=None,
        current_heating_degree_days=None,
        comparison_heating_degree_days=None,
        same_month_low_kwh=190.0,
        same_month_median_kwh=210.0,
        same_month_high_kwh=240.0,
        same_month_years=(2023, 2024, 2025),
        data_complete=True,
        change_category="moderate",
        judgement="Higher",
        confidence="High",
        confidence_reason="Both months are complete billing months.",
        interpretation="June 2026 used 15% more energy than June 2025.",
    )
    base.update(overrides)
    return MonthlyComparison(**base)


def _contributions(elec: float, gas: float) -> FuelContributions:
    total = elec + gas
    return FuelContributions(
        total_change_kwh=total,
        electricity_change_kwh=elec,
        gas_change_kwh=gas,
        electricity_share_pct=elec / total * 100 if total else None,
        gas_share_pct=gas / total * 100 if total else None,
        dominant_fuel="gas" if abs(gas) >= abs(elec) else "electricity",
    )


# --- interpretation rules -----------------------------------------------------------------


def test_meaningful_reduction_despite_colder_weather():
    comparison = _comparison(
        percentage_change=-12.0,
        absolute_change_kwh=-24.0,
        current_consumption_kwh=176.0,
        weather_explained_change_kwh=10.0,
        unexplained_change_kwh=-34.0,
        unexplained_change_is_meaningful=True,
        current_heating_degree_days=80.0,
        comparison_heating_degree_days=60.0,  # colder now
        current_weather_adjusted_kwh=150.0,
        comparison_weather_adjusted_kwh=180.0,  # adjusted also lower
        judgement="Better than comparable period",
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert "12% less" in narrative.what_changed
    assert "colder" in narrative.weather_effect
    assert "meaningful" in narrative.weather_effect
    assert narrative.action.startswith("No action is needed")


def test_increase_fully_explained_by_colder_weather():
    comparison = _comparison(
        weather_explained_change_kwh=25.0,
        unexplained_change_kwh=5.0,
        unexplained_change_is_meaningful=False,
        current_heating_degree_days=90.0,
        comparison_heating_degree_days=60.0,
        judgement="Higher but weather-explained",
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert "broadly explained by colder weather" in narrative.weather_effect
    assert "consistent with the weather" in narrative.action


def test_increase_not_explained_by_weather_deserves_investigation():
    comparison = _comparison(
        weather_explained_change_kwh=-5.0,
        unexplained_change_kwh=35.0,
        unexplained_change_is_meaningful=True,
        current_heating_degree_days=55.0,
        comparison_heating_degree_days=60.0,
        judgement="Higher and unexplained",
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert "stayed high after accounting for weather" in narrative.weather_effect
    assert "Review" in narrative.action


def test_increase_within_noise_but_not_colder_is_not_credited_to_weather():
    # The regression: milder weather + increase inside residual noise must NOT
    # read as "explained by colder weather".
    comparison = _comparison(
        weather_explained_change_kwh=-5.0,
        unexplained_change_kwh=35.0,
        unexplained_change_is_meaningful=False,
        current_heating_degree_days=55.0,
        comparison_heating_degree_days=60.0,
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert "colder" not in narrative.weather_effect
    assert "normal variation" in narrative.weather_effect


def test_electricity_only_increase_names_appliances_not_heating():
    comparison = _comparison(
        unexplained_change_is_meaningful=True,
        weather_explained_change_kwh=0.0,
        unexplained_change_kwh=30.0,
    )
    contributions = _contributions(elec=28.0, gas=2.0)
    narrative = build_monthly_comparison_narrative(comparison, contributions)
    assert "almost all" in narrative.which_fuel
    assert "appliance" in narrative.action.lower()
    assert "heating" not in narrative.action.lower() or "not" in narrative.action.lower()


def test_gas_only_increase_recommends_heating_review():
    comparison = _comparison(
        unexplained_change_is_meaningful=True,
        weather_explained_change_kwh=0.0,
        unexplained_change_kwh=30.0,
    )
    contributions = _contributions(elec=2.0, gas=28.0)
    narrative = build_monthly_comparison_narrative(comparison, contributions)
    assert "Gas" in narrative.which_fuel
    assert "heating" in narrative.action.lower()


def test_small_change_is_never_narrated_as_meaningful():
    comparison = _comparison(
        percentage_change=1.2,
        absolute_change_kwh=2.4,
        change_category="little",
        judgement="Similar",
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert "about the same" in narrative.what_changed
    assert narrative.weather_effect is None
    assert narrative.action.startswith("No action is needed")
    assert "meaningful" not in narrative.narrative.lower()


def test_opposite_fuel_movements_are_stated_not_hidden_in_shares():
    comparison = _comparison()
    contributions = FuelContributions(
        total_change_kwh=30.0,
        electricity_change_kwh=55.0,
        gas_change_kwh=-25.0,
        electricity_share_pct=183.3,
        gas_share_pct=-83.3,
        dominant_fuel="electricity",
    )
    narrative = build_monthly_comparison_narrative(comparison, contributions)
    assert "gas use actually fell" in narrative.which_fuel


def test_insufficient_evidence_fallback_narrative():
    comparison = _comparison(
        comparison_month=None,
        comparison_consumption_kwh=None,
        absolute_change_kwh=None,
        percentage_change=None,
        change_category=None,
        judgement="No comparison available",
        confidence="Low",
        confidence_reason="No June 2025 in the data.",
        interpretation="No June 2025 in the data, so a same-month year-on-year comparison isn't possible.",
        same_month_low_kwh=None,
        same_month_median_kwh=None,
        same_month_high_kwh=None,
        same_month_years=(),
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert "isn't possible" in narrative.what_changed
    assert narrative.action is None
    assert narrative.confidence == "Low"


def test_no_weather_model_large_increase_suggests_enabling_weather():
    comparison = _comparison(percentage_change=25.0, absolute_change_kwh=50.0, change_category="large")
    narrative = build_monthly_comparison_narrative(comparison, weather_enabled=False)
    assert "weather adjustment" in narrative.action.lower()
    assert "unknown" in narrative.limitation.lower()


def test_within_range_month_is_reported_as_normal():
    narrative = build_monthly_comparison_narrative(_comparison())
    assert narrative.is_unusual == "This month was within the normal June range."


def test_record_high_month_is_reported():
    comparison = _comparison(current_consumption_kwh=250.0, same_month_high_kwh=240.0)
    narrative = build_monthly_comparison_narrative(comparison)
    assert narrative.is_unusual == "This is your highest June on record."


def test_record_low_month_is_reported():
    comparison = _comparison(
        current_consumption_kwh=180.0,
        percentage_change=-10.0,
        absolute_change_kwh=-20.0,
        judgement="Better than comparable period",
    )
    narrative = build_monthly_comparison_narrative(comparison)
    assert narrative.is_unusual == "This is your lowest June on record."


# --- style constraints --------------------------------------------------------------------


@pytest.mark.parametrize("banned", _BANNED_PHRASES)
def test_banned_generic_phrases_never_appear(banned):
    scenarios = [
        build_monthly_comparison_narrative(_comparison()),
        build_monthly_comparison_narrative(
            _comparison(
                unexplained_change_is_meaningful=True,
                weather_explained_change_kwh=0.0,
                unexplained_change_kwh=30.0,
            ),
            _contributions(elec=2.0, gas=28.0),
        ),
        build_monthly_comparison_narrative(
            _comparison(percentage_change=-12.0, absolute_change_kwh=-24.0), weather_enabled=False
        ),
    ]
    for narrative in scenarios:
        assert banned not in narrative.narrative.lower()


def test_primary_narrative_stays_under_80_words():
    narrative = build_monthly_comparison_narrative(
        _comparison(
            unexplained_change_is_meaningful=True,
            weather_explained_change_kwh=-5.0,
            unexplained_change_kwh=35.0,
            current_heating_degree_days=55.0,
            comparison_heating_degree_days=60.0,
        ),
        _contributions(elec=10.0, gas=20.0),
    )
    assert len(narrative.narrative.split()) <= 80


def test_headline_stays_under_25_words():
    narrative = build_monthly_comparison_narrative(_comparison(), _contributions(elec=10.0, gas=20.0))
    assert len(narrative.headline.split()) <= 25


def test_narrative_fields_survive_frozen_dataclass_roundtrip():
    comparison = _comparison()
    assert dataclasses.is_dataclass(comparison)
    with pytest.raises(dataclasses.FrozenInstanceError):
        comparison.percentage_change = 99.0
