import pandas as pd
import pytest

from src.kpis import (
    compute_kpis,
    full_year_comparison,
    winter_over_winter_comparison,
    year_to_date_comparison,
)


def _monthly_df(n_months: int, kwh_values: list[float]) -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    df = pd.DataFrame(
        {
            "month_start": months,
            "consumption_kwh": kwh_values,
            "cost_gbp": [v * 0.3 for v in kwh_values],
        }
    )
    df["year"] = df["month_start"].dt.year
    df["month_num"] = df["month_start"].dt.month
    df["days_in_month"] = df["month_start"].dt.days_in_month
    months_per_year = df.groupby("year")["month_num"].transform("nunique")
    df["is_partial_year"] = months_per_year < 12
    return df


def test_compute_kpis_current_vs_previous_period():
    values = [100.0] * 12 + [200.0] * 12  # previous 12 months avg 100, current avg 200
    df = _monthly_df(24, values)

    kpis = {k.label: k for k in compute_kpis(df, months=12)}

    total = kpis["Total consumption"]
    assert total.current == pytest.approx(2400.0)
    assert total.previous == pytest.approx(1200.0)
    assert total.pct_change == pytest.approx(100.0)

    unit_rate = kpis["Average cost per kWh"]
    assert unit_rate.current == pytest.approx(0.3)
    assert unit_rate.previous == pytest.approx(0.3)


def test_compute_kpis_insufficient_history_returns_none_previous():
    df = _monthly_df(6, [100.0] * 6)

    kpis = compute_kpis(df, months=12)

    assert all(k.previous is None and k.pct_change is None for k in kpis)


def test_full_year_comparison_excludes_partial_years():
    df = _monthly_df(30, [100.0] * 30)  # 2.5 years: 2024 full, 2025 full, 2026 partial (6 months)

    result = full_year_comparison(df)

    assert set(result["year"]) == {2024, 2025}
    assert result[result["year"] == 2024]["total_kwh"].iloc[0] == pytest.approx(1200.0)


def test_year_to_date_comparison_matches_same_months():
    # 2024 full year (100/month), 2025-01..2025-03 partial with 150/month vs 2024 same months.
    months = list(pd.date_range("2024-01-01", periods=12, freq="MS")) + list(
        pd.date_range("2025-01-01", periods=3, freq="MS")
    )
    kwh = [100.0] * 12 + [150.0] * 3
    df = pd.DataFrame({"month_start": months, "consumption_kwh": kwh})
    df["cost_gbp"] = df["consumption_kwh"] * 0.3
    df["year"] = df["month_start"].dt.year
    df["month_num"] = df["month_start"].dt.month
    months_per_year = df.groupby("year")["month_num"].transform("nunique")
    df["is_partial_year"] = months_per_year < 12

    result = year_to_date_comparison(df)

    assert result["current_year"] == 2025
    assert result["previous_year"] == 2024
    assert result["months_compared"] == [1, 2, 3]
    assert result["current_kwh"] == pytest.approx(450.0)
    assert result["previous_kwh"] == pytest.approx(300.0)
    assert result["kwh_pct_change"] == pytest.approx(50.0)


def _simple_monthly_df(start: str, n: int, kwh_by_month) -> pd.DataFrame:
    """kwh_by_month: either a constant or a callable(month_start) -> float."""
    months = pd.date_range(start, periods=n, freq="MS")
    kwh = [kwh_by_month(m) if callable(kwh_by_month) else kwh_by_month for m in months]
    return pd.DataFrame({"month_start": months, "consumption_kwh": kwh, "cost_gbp": [v * 0.3 for v in kwh]})


def test_winter_over_winter_none_with_fewer_than_two_complete_winters():
    # Sep 2023 - Jul 2024: only one complete winter (Dec 2023/Jan 2024/Feb 2024).
    df = _simple_monthly_df("2023-09-01", 11, 100.0)
    assert winter_over_winter_comparison(df) is None


def test_winter_over_winter_none_when_no_winter_months_present():
    df = _simple_monthly_df("2024-03-01", 6, 100.0)  # Mar-Aug, no Dec/Jan/Feb at all
    assert winter_over_winter_comparison(df) is None


def test_winter_over_winter_compares_latest_two_complete_seasons():
    # Sep 2023 - Jan 2026 (29 months): winters 2024 (Dec23/Jan24/Feb24, complete), 2025
    # (Dec24/Jan25/Feb25, complete), and an incomplete 2026 season (Dec25+Jan26, no Feb26 yet).
    def kwh(month_start: pd.Timestamp) -> float:
        if month_start.year == 2024 and month_start.month in (1, 2):
            return 300.0
        if month_start == pd.Timestamp("2023-12-01"):
            return 300.0
        if month_start.year == 2025 and month_start.month in (1, 2):
            return 400.0
        if month_start == pd.Timestamp("2024-12-01"):
            return 400.0
        return 50.0  # non-winter months, irrelevant to the comparison

    df = _simple_monthly_df("2023-09-01", 29, kwh)
    result = winter_over_winter_comparison(df)

    assert result is not None
    assert result["latest_season"] == 2025
    assert result["previous_season"] == 2024
    assert result["latest_kwh"] == pytest.approx(1200.0)  # 400*3
    assert result["previous_kwh"] == pytest.approx(900.0)  # 300*3
    assert result["kwh_pct_change"] == pytest.approx((1200.0 - 900.0) / 900.0 * 100)
    assert result["latest_cost_gbp"] == pytest.approx(1200.0 * 0.3)


def test_winter_over_winter_excludes_incomplete_latest_season():
    # Complete winter 2024 and 2025, but the trailing data stops mid-way through winter 2026
    # (only Dec 2025 present, no Jan/Feb 2026 yet) -- 2026 must not be counted as complete.
    def kwh(month_start: pd.Timestamp) -> float:
        if month_start in (pd.Timestamp("2023-12-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")):
            return 300.0
        if month_start in (pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")):
            return 400.0
        return 50.0

    df = _simple_monthly_df("2023-09-01", 28, kwh)  # Sep 2023 - Dec 2025 (28 months)
    result = winter_over_winter_comparison(df)

    assert result is not None
    assert result["latest_season"] == 2025  # not 2026, since Jan/Feb 2026 aren't in the data
    assert result["previous_season"] == 2024
