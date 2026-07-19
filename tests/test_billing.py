"""Tests for full-bill estimates (src.billing): billing periods, standing charges, VAT,
and the billing-cycle-aware completeness rule."""

import pandas as pd
import pytest

from config import BillingConfig
from src.billing import (
    bill_breakdown,
    bill_breakdown_frame,
    billing_period,
    is_billing_month_complete,
    standing_charge_for_months,
    standing_charge_gbp,
)

CONFIG = BillingConfig()


# --- billing periods ----------------------------------------------------------------------


def test_billing_period_runs_6th_to_5th_of_next_month():
    start, end, days = billing_period(pd.Timestamp("2026-04-01"), CONFIG)
    assert start == pd.Timestamp("2026-04-06")
    assert end == pd.Timestamp("2026-05-05")
    assert days == 30  # April's day count: 25 April days + 5 May days


def test_billing_period_day_count_equals_labelled_months_days():
    for month, expected_days in [("2026-01-01", 31), ("2026-04-01", 30), ("2026-02-01", 28)]:
        _, _, days = billing_period(pd.Timestamp(month), CONFIG)
        assert days == expected_days


def test_billing_period_leap_february_has_29_days():
    start, end, days = billing_period(pd.Timestamp("2024-02-01"), CONFIG)
    assert days == 29
    assert start == pd.Timestamp("2024-02-06")
    assert end == pd.Timestamp("2024-03-05")


# --- completeness -------------------------------------------------------------------------


def test_month_complete_only_after_the_5th_of_the_next_month():
    june = pd.Timestamp("2026-06-01")  # bill covers 6 Jun - 5 Jul
    assert not is_billing_month_complete(june, pd.Timestamp("2026-07-01"))
    assert not is_billing_month_complete(june, pd.Timestamp("2026-07-05"))
    assert is_billing_month_complete(june, pd.Timestamp("2026-07-06"))
    assert is_billing_month_complete(june, pd.Timestamp("2026-08-01"))


def test_current_billing_month_is_incomplete():
    july = pd.Timestamp("2026-07-01")  # bill covers 6 Jul - 5 Aug
    assert not is_billing_month_complete(july, pd.Timestamp("2026-07-19"))


# --- standing charges ---------------------------------------------------------------------


def test_standing_charge_per_fuel_uses_daily_rate_times_period_days():
    june = pd.Timestamp("2026-06-01")  # 30-day period
    assert standing_charge_gbp(june, "electricity", CONFIG) == pytest.approx(0.6277 * 30)
    assert standing_charge_gbp(june, "gas", CONFIG) == pytest.approx(0.3497 * 30)


def test_combined_stream_pays_both_standing_charges():
    june = pd.Timestamp("2026-06-01")
    assert standing_charge_gbp(june, "total", CONFIG) == pytest.approx((0.6277 + 0.3497) * 30)


def test_standing_charge_for_months_sums_periods():
    months = [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")]  # 30 + 31 days
    assert standing_charge_for_months(months, "gas", CONFIG) == pytest.approx(0.3497 * 61)


# --- bill breakdown -----------------------------------------------------------------------


def test_bill_breakdown_components_sum_to_total():
    breakdown = bill_breakdown(pd.Timestamp("2026-06-01"), 41.37, "electricity", CONFIG)
    assert breakdown.consumption_cost_gbp == pytest.approx(41.37)
    assert breakdown.standing_charge_gbp == pytest.approx(0.6277 * 30)
    assert breakdown.vat_gbp == pytest.approx((41.37 + 0.6277 * 30) * 0.05)
    assert breakdown.total_bill_gbp == pytest.approx(
        breakdown.consumption_cost_gbp + breakdown.standing_charge_gbp + breakdown.vat_gbp
    )
    assert breakdown.days_in_period == 30


def test_bill_breakdown_backs_out_vat_when_export_includes_it():
    config = BillingConfig(export_cost_includes_vat=True)
    breakdown = bill_breakdown(pd.Timestamp("2026-06-01"), 42.0, "gas", config)
    assert breakdown.consumption_cost_gbp == pytest.approx(40.0)  # 42 / 1.05
    assert breakdown.vat_gbp == pytest.approx((40.0 + 0.3497 * 30) * 0.05)


def test_bill_breakdown_frame_one_row_per_month_with_period_labels():
    df = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")],
            "consumption_kwh": [149.52, 178.80],
            "cost_gbp": [37.62, 44.96],
        }
    )
    frame = bill_breakdown_frame(df, "electricity", CONFIG)
    assert len(frame) == 2
    assert frame.iloc[0]["billing_period"] == "06 Apr 2026 - 05 May 2026"
    assert frame.iloc[0]["total_bill_gbp"] == pytest.approx((37.62 + 0.6277 * 30) * 1.05)


def test_bill_breakdown_frame_empty_input_returns_typed_empty_frame():
    frame = bill_breakdown_frame(pd.DataFrame(columns=["month_start", "cost_gbp"]), "gas", CONFIG)
    assert frame.empty
    assert "total_bill_gbp" in frame.columns
