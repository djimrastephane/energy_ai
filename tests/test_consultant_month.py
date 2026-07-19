"""Tests for the Consultant's month-comparison handlers (src.consultant_month).

Contexts are built with synthetic three-year fuel frames; ``selected_comparison_month``
is always set explicitly so nothing depends on the wall clock.
"""

import numpy as np
import pandas as pd

from src.consultant import ConsultantContext
from src.consultant_month import (
    answer_best_or_worst_month,
    answer_month_vs_last_year,
    answer_month_vs_typical,
    answer_should_i_be_concerned,
    answer_was_it_weather,
    answer_what_next,
    answer_which_fuel_caused_change,
    answer_why_expensive,
)
from src.energy_signature import EnergySignatureResult


def _clean_df(start: str, kwh_values: list[float]) -> pd.DataFrame:
    months = pd.date_range(start, periods=len(kwh_values), freq="MS")
    df = pd.DataFrame(
        {"month_start": months, "consumption_kwh": kwh_values, "cost_gbp": [v * 0.2 for v in kwh_values]}
    )
    df["year"] = df["month_start"].dt.year
    df["month_num"] = df["month_start"].dt.month
    df["days_in_month"] = df["month_start"].dt.days_in_month
    months_per_year = df.groupby("year")["month_num"].transform("nunique")
    df["is_partial_year"] = months_per_year < 12
    return df


def _fuel_frames() -> dict:
    seasonal = [500, 450, 400, 300, 250, 200, 150, 160, 220, 300, 400, 480]
    total = [v * 1.1 for v in seasonal] + list(map(float, seasonal)) + [520.0, 430.0, 390.0, 310.0, 240.0, 260.0]
    elec = [v * 0.4 for v in total]
    gas = [v * 0.6 for v in total]
    return {
        "total": _clean_df("2024-01-01", total),
        "electricity": _clean_df("2024-01-01", elec),
        "gas": _clean_df("2024-01-01", gas),
    }


def _ctx(**overrides) -> ConsultantContext:
    base = dict(
        analyst_report=None,
        clean=_fuel_frames()["total"],
        fuel_frames=_fuel_frames(),
        fuel_energy_results={},
        anomalies=[],
        forecast_12mo=None,
        multi_fuel_forecasts=None,
        weather_enabled=False,
        selected_comparison_month=pd.Timestamp("2026-06-01"),
        comparison_mode="same_month_last_year",
        comparison_fuel="total",
    )
    base.update(overrides)
    return ConsultantContext(**base)


def test_month_vs_last_year_answers_for_selected_month_with_breakdown():
    answer = answer_month_vs_last_year(_ctx())
    assert "June 2025" in answer.answer
    assert "Answering for June 2026, combined energy." in answer.answer
    assert any(e.startswith("Gas:") for e in answer.evidence)
    assert any(e.startswith("Electricity:") for e in answer.evidence)
    assert answer.related_tab == "How did this month compare?"


def test_month_vs_last_year_respects_selected_fuel():
    answer = answer_month_vs_last_year(_ctx(comparison_fuel="gas"))
    assert "gas only" in answer.answer


def test_month_vs_typical_reports_range_and_position():
    answer = answer_month_vs_typical(_ctx())
    assert "typical June range" in answer.answer
    assert "median" in answer.answer
    assert answer.confidence is not None


def test_which_fuel_needs_both_fuel_exports():
    frames = _fuel_frames()
    frames["electricity"] = frames["electricity"].iloc[0:0]
    answer = answer_which_fuel_caused_change(_ctx(fuel_frames=frames))
    assert "needs both Electricity and Gas" in answer.answer


def test_which_fuel_attributes_the_change():
    answer = answer_which_fuel_caused_change(_ctx())
    assert "gas" in answer.answer.lower() or "electricity" in answer.answer.lower()
    assert len(answer.evidence) >= 2


def test_was_it_weather_requires_weather_enabled():
    answer = answer_was_it_weather(_ctx(weather_enabled=False))
    assert "turn on" in answer.answer.lower()


def test_was_it_weather_reports_split_when_model_covers_both_months():
    frames = _fuel_frames()
    df = frames["total"]
    months = [pd.Timestamp("2025-06-01"), pd.Timestamp("2026-06-01")]
    rows = df[df["month_start"].isin(months)].copy()
    rows["hdd"] = [60.0, 90.0]
    rows["avg_daily_hdd"] = rows["hdd"] / rows["days_in_month"]
    rows["avg_daily_cdd"] = 0.0
    fitted = pd.Series([6.0, 7.0], index=pd.DatetimeIndex(months))
    actual_daily = rows.set_index("month_start")["consumption_kwh"] / rows.set_index("month_start")["days_in_month"]
    result = EnergySignatureResult(
        intercept=2.0, intercept_se=0.1, heating_slope=0.5, heating_se=0.05, heating_pvalue=0.001,
        cooling_slope=0.0, cooling_se=np.nan, cooling_pvalue=np.nan, r_squared=0.9,
        adj_r_squared=0.89, durbin_watson=2.0, n_obs=2, fitted=fitted,
        resid=actual_daily - fitted,
    )
    ctx = _ctx(
        weather_enabled=True,
        fuel_merged={"total": rows, "electricity": None, "gas": None},
        fuel_energy_results={"total": result},
    )
    answer = answer_was_it_weather(ctx)
    assert "kWh change" in answer.answer
    assert "temperature model expected" in answer.answer
    assert answer.confidence is not None


def test_best_or_worst_month_reports_both_extremes():
    answer = answer_best_or_worst_month(_ctx())
    assert "best recorded June" in answer.answer or "best June" in answer.answer
    assert any("Best recorded June" in e for e in answer.evidence)
    assert any("Worst recorded June" in e for e in answer.evidence)


def test_why_expensive_splits_usage_and_rate():
    answer = answer_why_expensive(_ctx())
    assert "cost" in answer.answer.lower()
    assert "Standing charges" in answer.answer
    assert any("From usage" in e for e in answer.evidence)


def test_should_i_be_concerned_answers_directly():
    answer = answer_should_i_be_concerned(_ctx())
    assert answer.answer.split()[0] in ("No.", "Probably", "This")
    assert "Anomaly detection:" in answer.answer


def test_what_next_gives_action_and_limitation():
    answer = answer_what_next(_ctx())
    assert answer.answer  # always something, never empty
    assert "Answering for June 2026" in answer.answer


def test_handlers_degrade_honestly_with_no_data():
    empty = {"total": _clean_df("2026-01-01", []), "electricity": _clean_df("2026-01-01", []), "gas": _clean_df("2026-01-01", [])}
    ctx = _ctx(fuel_frames=empty, selected_comparison_month=None)
    for handler in (
        answer_month_vs_last_year,
        answer_month_vs_typical,
        answer_which_fuel_caused_change,
        answer_best_or_worst_month,
        answer_why_expensive,
        answer_should_i_be_concerned,
        answer_what_next,
    ):
        answer = handler(ctx)
        assert answer.answer  # honest fallback text, never an exception
        assert answer.confidence is None or answer.confidence == "Low"


def test_long_term_mode_falls_back_to_same_month_last_year():
    answer = answer_month_vs_last_year(_ctx(comparison_mode="long_term"))
    assert "June 2025" in answer.answer
