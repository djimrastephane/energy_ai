"""Full-bill estimates: billing periods, standing charges, and VAT.

The OVO exports carry consumption cost only. With the user-supplied tariff
facts in ``config.BillingConfig`` (standing charges, 5% VAT, and the
6th-to-5th billing cycle), a full bill can now be estimated as

    total = (consumption cost + standing charge) x (1 + VAT rate)

with each component reported separately, never blended. The billing cycle
matters twice: the standing charge for the month labelled M covers the 6th
of M to the 5th of M+1 (always exactly ``days_in_month(M)`` days), and a
month's bill is not *complete* until the 5th of the following month has
passed -- ``is_billing_month_complete`` is the single source of truth the
month-comparison journey uses for completeness.

Known limitation, documented rather than silently ignored: the weather
merge aligns consumption to calendar months, so each row's degree days are
offset ~5 days from its true billing window. Re-aligning would change the
validated weather models and is deliberately out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import SETTINGS, BillingConfig
from src.ingestion import EnergyType


def billing_period(
    month_start: pd.Timestamp, config: BillingConfig | None = None
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """(period_start, period_end, days) for the bill labelled ``month_start``.

    The April 2026 bill runs 6 Apr - 5 May 2026 inclusive: 25 April days +
    5 May days = 30 = April's day count. In general the period always
    contains exactly ``days_in_month`` days of the labelled month.
    """
    config = config or SETTINGS.billing
    start = month_start.replace(day=config.billing_cycle_start_day)
    end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    return start, end, int(month_start.days_in_month)


def is_billing_month_complete(month_start: pd.Timestamp, today: pd.Timestamp | None = None) -> bool:
    """A month's bill is complete once its billing period has fully elapsed.

    June 2026 (6 Jun - 5 Jul) is complete from 6 July onward -- *not* from
    1 July, which the previous calendar-month convention assumed.
    """
    today = today if today is not None else pd.Timestamp.now()
    _, end, _ = billing_period(month_start)
    return today.normalize() > end


def standing_charge_gbp(
    month_start: pd.Timestamp, fuel: EnergyType, config: BillingConfig | None = None
) -> float:
    """Standing charge for the month's billing period, excluding VAT.

    For the combined ``"total"`` stream both fuels' daily rates apply --
    a dual-fuel bill pays both standing charges.
    """
    config = config or SETTINGS.billing
    _, _, days = billing_period(month_start, config)
    daily = {
        "electricity": config.electricity_standing_gbp_per_day,
        "gas": config.gas_standing_gbp_per_day,
        "total": config.electricity_standing_gbp_per_day + config.gas_standing_gbp_per_day,
    }[fuel]
    return daily * days


@dataclass(frozen=True)
class MonthlyBillBreakdown:
    """One month's estimated full bill, component by component (all GBP)."""

    month_start: pd.Timestamp
    fuel: EnergyType
    billing_period_start: pd.Timestamp
    billing_period_end: pd.Timestamp
    days_in_period: int
    consumption_cost_gbp: float  # ex VAT (backed out if the export includes it)
    standing_charge_gbp: float  # ex VAT
    vat_gbp: float
    total_bill_gbp: float


def bill_breakdown(
    month_start: pd.Timestamp,
    exported_cost_gbp: float,
    fuel: EnergyType,
    config: BillingConfig | None = None,
) -> MonthlyBillBreakdown:
    """Break one month's bill into consumption cost, standing charge, and VAT.

    ``exported_cost_gbp`` is the export's ``Cost (£)`` value; whether it
    already includes VAT is governed by ``config.export_cost_includes_vat``
    (see the config docstring for the assumption and how to flip it).
    """
    config = config or SETTINGS.billing
    start, end, days = billing_period(month_start, config)
    consumption_ex_vat = (
        exported_cost_gbp / (1 + config.vat_rate)
        if config.export_cost_includes_vat
        else exported_cost_gbp
    )
    standing = standing_charge_gbp(month_start, fuel, config)
    vat = (consumption_ex_vat + standing) * config.vat_rate
    return MonthlyBillBreakdown(
        month_start=month_start,
        fuel=fuel,
        billing_period_start=start,
        billing_period_end=end,
        days_in_period=days,
        consumption_cost_gbp=consumption_ex_vat,
        standing_charge_gbp=standing,
        vat_gbp=vat,
        total_bill_gbp=consumption_ex_vat + standing + vat,
    )


def bill_breakdown_frame(
    clean_df: pd.DataFrame, fuel: EnergyType, config: BillingConfig | None = None
) -> pd.DataFrame:
    """Per-month bill components for a whole clean frame -- the Cost Intelligence table."""
    if clean_df.empty:
        return pd.DataFrame(
            columns=[
                "month_start", "billing_period", "consumption_cost_gbp",
                "standing_charge_gbp", "vat_gbp", "total_bill_gbp",
            ]
        )
    rows = []
    for _, row in clean_df.sort_values("month_start").iterrows():
        breakdown = bill_breakdown(row["month_start"], float(row["cost_gbp"]), fuel, config)
        rows.append(
            {
                "month_start": breakdown.month_start,
                "billing_period": (
                    f"{breakdown.billing_period_start.strftime('%d %b %Y')} - "
                    f"{breakdown.billing_period_end.strftime('%d %b %Y')}"
                ),
                "consumption_cost_gbp": breakdown.consumption_cost_gbp,
                "standing_charge_gbp": breakdown.standing_charge_gbp,
                "vat_gbp": breakdown.vat_gbp,
                "total_bill_gbp": breakdown.total_bill_gbp,
            }
        )
    return pd.DataFrame(rows)


def standing_charge_for_months(
    months: list[pd.Timestamp], fuel: EnergyType, config: BillingConfig | None = None
) -> float:
    """Total ex-VAT standing charge across several billing months (e.g. a forecast horizon)."""
    return sum(standing_charge_gbp(m, fuel, config) for m in months)
