"""Tests for the monthly comparison domain model (src.monthly_comparison).

Fixtures are synthetic and deterministic; ``today`` is always injected so
nothing depends on the wall clock. The weather-decomposition tests build an
``EnergySignatureResult`` by hand (it is a plain dataclass) rather than
fitting anything -- these tests cover the comparison arithmetic, not the
regression, which has its own suite.
"""

import numpy as np
import pandas as pd
import pytest

from config import MonthComparisonThresholds
from src.energy_signature import EnergySignatureResult
from src.monthly_comparison import (
    build_monthly_comparison,
    categorize_change,
    fuel_contributions,
    in_progress_month,
    is_month_complete,
    latest_complete_month,
    month_to_date_note,
    same_month_history,
    selectable_months,
)

TODAY = pd.Timestamp("2026-07-19")


def _clean_df(start: str, kwh_values: list[float], cost_values: list[float] | None = None) -> pd.DataFrame:
    months = pd.date_range(start, periods=len(kwh_values), freq="MS")
    costs = cost_values if cost_values is not None else [v * 0.2 for v in kwh_values]
    df = pd.DataFrame({"month_start": months, "consumption_kwh": kwh_values, "cost_gbp": costs})
    df["year"] = df["month_start"].dt.year
    df["month_num"] = df["month_start"].dt.month
    df["days_in_month"] = df["month_start"].dt.days_in_month
    months_per_year = df.groupby("year")["month_num"].transform("nunique")
    df["is_partial_year"] = months_per_year < 12
    return df


# Three full years 2023-2025 plus Jan-Jul 2026; July 2026 is the in-progress month.
def _three_year_df() -> pd.DataFrame:
    seasonal = [500, 450, 400, 300, 250, 200, 150, 160, 220, 300, 400, 480]
    values = (
        [v * 1.2 for v in seasonal]  # 2023
        + [v * 1.1 for v in seasonal]  # 2024
        + list(map(float, seasonal))  # 2025
        + [520.0, 430.0, 390.0, 310.0, 240.0, 230.0, 60.0]  # Jan-Jul 2026 (Jul partial)
    )
    return _clean_df("2023-01-01", values)


# --- month completeness -------------------------------------------------------------------


def test_is_month_complete_only_past_calendar_months():
    assert is_month_complete(pd.Timestamp("2026-06-01"), TODAY)
    assert not is_month_complete(pd.Timestamp("2026-07-01"), TODAY)
    assert not is_month_complete(pd.Timestamp("2026-08-01"), TODAY)


def test_latest_complete_month_skips_in_progress_month():
    df = _three_year_df()
    assert latest_complete_month(df, TODAY) == pd.Timestamp("2026-06-01")
    assert in_progress_month(df, TODAY) == pd.Timestamp("2026-07-01")


def test_latest_complete_month_when_no_partial_month():
    df = _clean_df("2026-01-01", [100.0] * 6)  # Jan-Jun 2026, all complete
    assert latest_complete_month(df, TODAY) == pd.Timestamp("2026-06-01")
    assert in_progress_month(df, TODAY) is None
    assert month_to_date_note(df, TODAY) is None


def test_month_to_date_note_names_partial_and_fallback_months():
    note = month_to_date_note(_three_year_df(), TODAY)
    assert note == "July 2026 is incomplete. The primary comparison uses June 2026."


def test_selectable_months_excludes_partial_and_sorts_newest_first():
    months = selectable_months(_three_year_df(), TODAY)
    assert months[0] == pd.Timestamp("2026-06-01")
    assert pd.Timestamp("2026-07-01") not in months
    assert months == sorted(months, reverse=True)


def test_empty_frame_helpers_return_none():
    empty = _clean_df("2026-01-01", [])
    assert latest_complete_month(empty, TODAY) is None
    assert in_progress_month(empty, TODAY) is None
    assert month_to_date_note(empty, TODAY) is None
    assert selectable_months(empty, TODAY) == []
    assert build_monthly_comparison(empty, "total", today=TODAY) is None


# --- same-month history -------------------------------------------------------------------


def test_same_month_history_only_same_calendar_month_earlier_years():
    df = _three_year_df()
    history = same_month_history(df, pd.Timestamp("2026-06-01"), TODAY)
    assert list(history["month_start"]) == [
        pd.Timestamp("2023-06-01"),
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2025-06-01"),
    ]


def test_same_month_history_never_contains_july_when_comparing_january():
    df = _three_year_df()
    history = same_month_history(df, pd.Timestamp("2026-01-01"), TODAY)
    assert set(history["month_start"].dt.month) == {1}


# --- comparison modes ---------------------------------------------------------------------


def test_same_month_last_year_default_selects_latest_complete_month():
    df = _three_year_df()
    result = build_monthly_comparison(df, "total", today=TODAY)
    assert result.selected_month == pd.Timestamp("2026-06-01")
    assert result.comparison_month == pd.Timestamp("2025-06-01")
    assert result.current_consumption_kwh == pytest.approx(230.0)
    assert result.comparison_consumption_kwh == pytest.approx(200.0)
    assert result.absolute_change_kwh == pytest.approx(30.0)
    assert result.percentage_change == pytest.approx(15.0)
    assert result.data_complete is True


def test_previous_month_mode_compares_adjacent_month():
    df = _three_year_df()
    result = build_monthly_comparison(df, "total", "previous_month", today=TODAY)
    assert result.comparison_month == pd.Timestamp("2026-05-01")
    assert result.comparison_consumption_kwh == pytest.approx(240.0)
    assert result.confidence == "Medium"
    assert "seasonality" in result.confidence_reason.lower()


def test_typical_month_mode_uses_median_of_same_month_history():
    df = _three_year_df()
    result = build_monthly_comparison(df, "total", "typical_month", today=TODAY)
    assert result.comparison_month is None
    assert result.comparison_consumption_kwh == pytest.approx(220.0)  # median of 240/220/200
    assert result.same_month_low_kwh == pytest.approx(200.0)
    assert result.same_month_high_kwh == pytest.approx(240.0)
    assert result.same_month_years == (2023, 2024, 2025)


def test_best_and_worst_month_modes_pick_extremes_of_same_calendar_month():
    df = _three_year_df()
    best = build_monthly_comparison(df, "total", "best_month", today=TODAY)
    worst = build_monthly_comparison(df, "total", "worst_month", today=TODAY)
    assert best.comparison_month == pd.Timestamp("2025-06-01")  # lowest June: 200
    assert worst.comparison_month == pd.Timestamp("2023-06-01")  # highest June: 240
    assert best.comparison_consumption_kwh == pytest.approx(200.0)
    assert worst.comparison_consumption_kwh == pytest.approx(240.0)


def test_best_month_never_compares_across_calendar_months():
    # June's best must come from Junes only, even though July/August are lower.
    df = _three_year_df()
    result = build_monthly_comparison(df, "total", "best_month", today=TODAY)
    assert result.comparison_month.month == 6


# --- honest fallbacks ---------------------------------------------------------------------


def test_missing_prior_year_month_returns_honest_fallback():
    df = _clean_df("2026-01-01", [100.0] * 6)
    result = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-03-01"), today=TODAY
    )
    assert result.comparison_month is None
    assert result.percentage_change is None
    assert result.judgement == "No comparison available"
    assert result.confidence == "Low"
    assert "March 2025" in result.interpretation


def test_single_year_of_data_cannot_support_typical_mode():
    df = _clean_df("2026-01-01", [100.0] * 6)
    result = build_monthly_comparison(
        df, "total", "typical_month", pd.Timestamp("2026-06-01"), today=TODAY
    )
    assert result.percentage_change is None
    assert "not enough history" in result.interpretation.lower()
    assert "June" in result.interpretation


def test_two_same_month_observations_is_the_minimum_for_typical():
    df = _clean_df("2024-06-01", [100.0] + [50.0] * 11 + [120.0] + [50.0] * 11 + [110.0])
    # Junes: 2024=100, 2025=120, selected 2026=110 -> 2 prior observations
    result = build_monthly_comparison(
        df, "total", "typical_month", pd.Timestamp("2026-06-01"), today=TODAY
    )
    assert result.comparison_consumption_kwh == pytest.approx(110.0)
    assert result.confidence == "Medium"  # below solid_same_month_observations


def test_first_month_of_history_has_no_previous_month():
    df = _clean_df("2026-01-01", [100.0] * 6)
    result = build_monthly_comparison(
        df, "total", "previous_month", pd.Timestamp("2026-01-01"), today=TODAY
    )
    assert result.comparison_month is None
    assert result.percentage_change is None


def test_zero_comparison_consumption_yields_none_percentage():
    df = _clean_df("2025-06-01", [0.0] + [50.0] * 11 + [100.0])
    result = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-06-01"), today=TODAY
    )
    assert result.comparison_consumption_kwh == pytest.approx(0.0)
    assert result.percentage_change is None
    assert result.absolute_change_kwh == pytest.approx(100.0)


def test_selected_month_not_in_data_returns_none():
    df = _three_year_df()
    assert (
        build_monthly_comparison(df, "total", selected_month=pd.Timestamp("2019-01-01"), today=TODAY)
        is None
    )


# --- partial months -----------------------------------------------------------------------


def test_in_progress_month_selected_manually_is_flagged_incomplete():
    df = _three_year_df()
    result = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-07-01"), today=TODAY
    )
    assert result.data_complete is False
    assert result.confidence == "Low"
    assert "month-to-date" in result.confidence_reason


# --- cost decomposition -------------------------------------------------------------------


def test_cost_change_splits_exactly_into_usage_and_rate_parts():
    kwh = [100.0] + [50.0] * 11 + [120.0]
    cost = [20.0] + [10.0] * 11 + [30.0]  # rate rises from 0.20 to 0.25
    df = _clean_df("2025-06-01", kwh, cost)
    result = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-06-01"), today=TODAY
    )
    assert result.cost_change_gbp == pytest.approx(10.0)
    # usage part: +20 kWh at last year's 0.20 = £4; rate part: +0.05 x 120 = £6
    assert result.cost_change_from_usage_gbp == pytest.approx(4.0)
    assert result.cost_change_from_rate_gbp == pytest.approx(6.0)
    assert result.cost_change_from_usage_gbp + result.cost_change_from_rate_gbp == pytest.approx(
        result.cost_change_gbp
    )


# --- leap year ----------------------------------------------------------------------------


def test_leap_february_uses_actual_days_for_daily_averages():
    df = _clean_df("2023-02-01", [280.0] + [50.0] * 11 + [290.0])  # Feb 2023 (28d), Feb 2024 (29d)
    result = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2024-02-01"), today=TODAY
    )
    assert result.current_avg_daily_kwh == pytest.approx(10.0)  # 290/29
    assert result.comparison_avg_daily_kwh == pytest.approx(10.0)  # 280/28


# --- change categories --------------------------------------------------------------------


def test_categorize_change_documented_thresholds():
    thresholds = MonthComparisonThresholds()
    assert categorize_change(1.0, thresholds) == "little"
    assert categorize_change(-4.9, thresholds) == "little"
    assert categorize_change(5.0, thresholds) == "moderate"
    assert categorize_change(-19.9, thresholds) == "moderate"
    assert categorize_change(20.0, thresholds) == "large"
    assert categorize_change(None, thresholds) is None


def test_little_change_is_judged_similar_not_meaningful():
    df = _clean_df("2025-06-01", [100.0] + [50.0] * 11 + [102.0])
    result = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-06-01"), today=TODAY
    )
    assert result.change_category == "little"
    assert result.judgement == "Similar"
    assert "about the same" in result.interpretation


# --- weather decomposition ----------------------------------------------------------------


def _weather_objects(df: pd.DataFrame, fitted_daily: dict[str, float], hdd: dict[str, float]):
    """Hand-built merged frame + EnergySignatureResult covering the months in ``fitted_daily``."""
    months = [pd.Timestamp(m) for m in fitted_daily]
    rows = df[df["month_start"].isin(months)].copy()
    rows["hdd"] = [hdd[m.strftime("%Y-%m-%d")] for m in rows["month_start"]]
    rows["avg_daily_hdd"] = rows["hdd"] / rows["days_in_month"]
    rows["avg_daily_cdd"] = 0.0
    fitted = pd.Series(
        [fitted_daily[m.strftime("%Y-%m-%d")] for m in rows["month_start"]],
        index=pd.DatetimeIndex(rows["month_start"]),
    )
    actual_daily = rows.set_index("month_start")["consumption_kwh"] / rows.set_index("month_start")["days_in_month"]
    resid = actual_daily - fitted
    result = EnergySignatureResult(
        intercept=2.0, intercept_se=0.1,
        heating_slope=0.5, heating_se=0.05, heating_pvalue=0.001,
        cooling_slope=0.0, cooling_se=np.nan, cooling_pvalue=np.nan,
        r_squared=0.9, adj_r_squared=0.89, durbin_watson=2.0, n_obs=len(rows),
        fitted=fitted, resid=resid,
    )
    return rows, result


def test_weather_decomposition_splits_change_into_explained_and_unexplained():
    # June 2025: 200 kWh (30 days); June 2026: 230 kWh. Model expects 6.0 kWh/day
    # in 2025 (180 kWh) and 6.5 kWh/day in 2026 (195 kWh) -> weather explains +15
    # of the +30 change; +15 is unexplained.
    df = _three_year_df()
    merged, result = _weather_objects(
        df,
        {"2025-06-01": 6.0, "2026-06-01": 6.5},
        {"2025-06-01": 60.0, "2026-06-01": 90.0},
    )
    comparison = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-06-01"), merged, result, today=TODAY
    )
    assert comparison.weather_explained_change_kwh == pytest.approx(15.0)
    assert comparison.unexplained_change_kwh == pytest.approx(15.0)
    assert comparison.current_heating_degree_days == pytest.approx(90.0)
    assert comparison.comparison_heating_degree_days == pytest.approx(60.0)
    # adjusted = actual - (expected - base): base is 2.0 kWh/day x 30 = 60
    assert comparison.current_weather_adjusted_kwh == pytest.approx(230.0 - (195.0 - 60.0))
    assert comparison.comparison_weather_adjusted_kwh == pytest.approx(200.0 - (180.0 - 60.0))


def test_weather_fields_none_when_month_not_covered_by_fit():
    df = _three_year_df()
    merged, result = _weather_objects(
        df, {"2026-06-01": 6.5}, {"2026-06-01": 90.0}
    )  # comparison month missing from the fit
    comparison = build_monthly_comparison(
        df, "total", "same_month_last_year", pd.Timestamp("2026-06-01"), merged, result, today=TODAY
    )
    assert comparison.weather_explained_change_kwh is None
    assert comparison.unexplained_change_kwh is None
    assert comparison.current_weather_adjusted_kwh is not None  # current month is covered


def test_weather_fields_none_without_weather_objects():
    df = _three_year_df()
    comparison = build_monthly_comparison(df, "total", today=TODAY)
    assert comparison.weather_explained_change_kwh is None
    assert comparison.unexplained_change_is_meaningful is None
    assert comparison.judgement == "Higher"  # 15% up, no weather model -> plain "Higher"


# --- fuel contributions -------------------------------------------------------------------


def test_fuel_contributions_split_and_dominant_fuel():
    df_total = _clean_df("2025-06-01", [300.0] + [50.0] * 11 + [400.0])
    df_elec = _clean_df("2025-06-01", [100.0] + [20.0] * 11 + [120.0])
    df_gas = _clean_df("2025-06-01", [200.0] + [30.0] * 11 + [280.0])
    month = pd.Timestamp("2026-06-01")
    total = build_monthly_comparison(df_total, "total", selected_month=month, today=TODAY)
    elec = build_monthly_comparison(df_elec, "electricity", selected_month=month, today=TODAY)
    gas = build_monthly_comparison(df_gas, "gas", selected_month=month, today=TODAY)

    contributions = fuel_contributions(total, elec, gas)
    assert contributions.total_change_kwh == pytest.approx(100.0)
    assert contributions.electricity_change_kwh == pytest.approx(20.0)
    assert contributions.gas_change_kwh == pytest.approx(80.0)
    assert contributions.electricity_share_pct == pytest.approx(20.0)
    assert contributions.gas_share_pct == pytest.approx(80.0)
    assert contributions.dominant_fuel == "gas"


def test_fuel_contributions_zero_total_change_has_no_shares():
    df_total = _clean_df("2025-06-01", [300.0] + [50.0] * 11 + [300.0])
    df_elec = _clean_df("2025-06-01", [100.0] + [20.0] * 11 + [150.0])
    df_gas = _clean_df("2025-06-01", [200.0] + [30.0] * 11 + [150.0])
    month = pd.Timestamp("2026-06-01")
    contributions = fuel_contributions(
        build_monthly_comparison(df_total, "total", selected_month=month, today=TODAY),
        build_monthly_comparison(df_elec, "electricity", selected_month=month, today=TODAY),
        build_monthly_comparison(df_gas, "gas", selected_month=month, today=TODAY),
    )
    assert contributions.electricity_share_pct is None
    assert contributions.dominant_fuel is None


def test_fuel_contributions_none_when_any_fuel_missing():
    df_total = _three_year_df()
    total = build_monthly_comparison(df_total, "total", today=TODAY)
    assert fuel_contributions(total, None, None) is None
