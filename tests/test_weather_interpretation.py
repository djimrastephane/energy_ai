import numpy as np
import pandas as pd
import pytest

from src.anomalies import Anomaly
from src.weather_context import WeatherContextClassification
from src.weather_interpretation import interpret_unusual_month

_DATE = pd.Timestamp("2024-12-01")


def _anomaly(direction="spike", n_methods=3) -> Anomaly:
    methods = ["rolling_zscore", "stl_esd", "isolation_forest"][:n_methods]
    return Anomaly(date=_DATE, methods=methods, direction=direction, rank_context="1st highest of 35 months")


def _classification(label="Mixed severe weather") -> WeatherContextClassification:
    return WeatherContextClassification(
        month_start=_DATE,
        label=label,
        confidence="High",
        facts=["6 snow day(s), 12 cm total snowfall", "Maximum gust 91 km/h"],
        limitation="Monthly billing data cannot prove behavioural causes.",
    )


class _FakeEnergyResult:
    """Only the attribute interpret_unusual_month reads: a resid Series (kWh/day)."""

    def __init__(self, resid_at_date: float, spread: float = 1.0):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=24, freq="MS")
        values = rng.normal(0, spread, 24)
        self.resid = pd.Series(values, index=idx)
        self.resid.loc[_DATE] = resid_at_date


def _merged() -> pd.DataFrame:
    return pd.DataFrame({"month_start": [_DATE], "days_in_month": [31]})


def test_pattern_a_severe_weather_explains_increase():
    # Residual near zero -> within expectation.
    result = interpret_unusual_month(_anomaly(), _classification(), _FakeEnergyResult(0.05), _merged())

    assert "broadly consistent with colder and severe winter weather" in result.headline
    assert result.confidence == "High"
    assert result.weather_facts  # observed facts carried through
    assert result.residual_monthly_kwh == pytest.approx(0.05 * 31)


def test_pattern_b_severe_weather_only_partly_explains():
    # Residual 10 kWh/day against ~1 kWh/day spread -> material.
    result = interpret_unusual_month(_anomaly(), _classification(), _FakeEnergyResult(10.0), _merged())

    assert "may have contributed" in result.headline
    assert "does not explain the full increase" in result.headline
    assert f"{10.0 * 31:+,.0f}" in result.headline  # the kWh figure is stated
    assert "cannot confirm" in result.limitation


def test_pattern_c_electricity_only_rise_during_snow():
    result = interpret_unusual_month(
        _anomaly(),
        _classification("Snowy period"),
        _FakeEnergyResult(10.0),
        _merged(),
        electricity_flagged=True,
        gas_flagged=False,
    )

    assert "does not look primarily heating-driven" in result.headline
    assert "cannot be confirmed from monthly data" in result.headline


def test_pattern_d_no_severe_weather_high_residual():
    quiet = _classification("No notable severe weather")
    result = interpret_unusual_month(_anomaly(), quiet, _FakeEnergyResult(10.0), _merged())

    assert "does not explain this month well" in result.headline
    assert "occupancy, heating settings, or appliance changes" in result.headline


def test_drop_during_severe_weather_gets_mirrored_wording():
    result = interpret_unusual_month(_anomaly(direction="drop"), _classification(), _FakeEnergyResult(-10.0), _merged())

    assert "lower" in result.headline
    assert "cannot be confirmed from monthly data" in result.headline
    # Never asserts the absence as fact:
    assert "was away" not in result.headline


def test_no_weather_context_available_is_honest():
    result = interpret_unusual_month(_anomaly(), None, None, None)

    assert "weather context is unavailable" in result.headline
    assert result.residual_monthly_kwh is None
    assert result.confidence == "Low"


def test_single_method_anomaly_gets_lower_confidence():
    quiet = _classification("No notable severe weather")
    result = interpret_unusual_month(_anomaly(n_methods=1), quiet, _FakeEnergyResult(0.05), _merged())

    assert result.confidence == "Low"


def test_never_asserts_behavioural_causes_as_fact():
    """Across every pattern, forbidden causal-assertion phrases must not appear."""
    cases = [
        interpret_unusual_month(_anomaly(), _classification(), _FakeEnergyResult(10.0), _merged()),
        interpret_unusual_month(_anomaly(), _classification(), _FakeEnergyResult(0.05), _merged()),
        interpret_unusual_month(
            _anomaly(), _classification("Snowy period"), _FakeEnergyResult(10.0), _merged(),
            electricity_flagged=True, gas_flagged=False,
        ),
        interpret_unusual_month(_anomaly(), None, None, None),
        interpret_unusual_month(_anomaly(direction="drop"), _classification(), _FakeEnergyResult(-10.0), _merged()),
    ]
    for result in cases:
        text = result.headline.lower()
        assert "caused working from home" not in text
        assert "roads were closed" not in text
        assert "you worked from home" not in text
        assert "occupancy increased" not in text


def test_energy_evidence_includes_methods_and_residual():
    result = interpret_unusual_month(_anomaly(), _classification(), _FakeEnergyResult(10.0), _merged())

    joined = " ".join(result.energy_evidence)
    assert "3 of 3 detection methods" in joined
    assert "temperature model" in joined
