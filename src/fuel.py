"""Electricity/gas fuel-level analysis: cross-checking, combining, and a fuel-mix finding.

Every other module in this platform is generic over "a monthly clean
DataFrame with consumption_kwh/cost_gbp" -- the app's sidebar fuel selector
picks which such frame (Total, Electricity, or Gas) flows through the
entire existing pipeline, so this module only needs to cover what's
genuinely new: cross-checking the three sources against each other, and
comparing electricity vs. gas directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.findings import Finding
from src.preprocessing import (
    PreprocessingReport,
    build_monthly_series,
    detect_missing_months,
    validate_units,
)
from src.utils import get_logger, safe_divide

logger = get_logger(__name__)

_MIN_MONTHS_FOR_FUEL_MIX_FINDING = 6


@dataclass
class FuelShares:
    """Structured gas/electricity split -- the numeric core of ``finding_fuel_mix``, exposed
    separately so other modules (e.g. ``src.recommendations.recommend_fuel_focus``) can consume
    the same numbers without re-deriving them or parsing the Finding's prose narrative."""

    gas_share_kwh_pct: float
    electricity_share_kwh_pct: float
    gas_share_cost_pct: float
    electricity_share_cost_pct: float
    total_electricity_kwh: float
    total_gas_kwh: float
    total_electricity_cost_gbp: float
    total_gas_cost_gbp: float
    n_months: int


def compute_fuel_shares(combined_df: pd.DataFrame) -> FuelShares | None:
    """The gas/electricity consumption and cost split, from already-combined billing data.

    Returns ``None`` under the same conditions ``finding_fuel_mix`` returns
    ``None`` for (too little overlapping history, or zero total consumption).
    """
    if len(combined_df) < _MIN_MONTHS_FOR_FUEL_MIX_FINDING:
        return None

    total_elec_kwh = float(combined_df["electricity_kwh"].sum())
    total_gas_kwh = float(combined_df["gas_kwh"].sum())
    total_kwh = total_elec_kwh + total_gas_kwh
    if total_kwh <= 0:
        return None
    gas_share_kwh = total_gas_kwh / total_kwh * 100

    total_elec_cost = float(combined_df["electricity_cost_gbp"].sum())
    total_gas_cost = float(combined_df["gas_cost_gbp"].sum())
    gas_share_cost = safe_divide(total_gas_cost, total_elec_cost + total_gas_cost, default=0.0) * 100

    return FuelShares(
        gas_share_kwh_pct=gas_share_kwh,
        electricity_share_kwh_pct=100 - gas_share_kwh,
        gas_share_cost_pct=gas_share_cost,
        electricity_share_cost_pct=100 - gas_share_cost,
        total_electricity_kwh=total_elec_kwh,
        total_gas_kwh=total_gas_kwh,
        total_electricity_cost_gbp=total_elec_cost,
        total_gas_cost_gbp=total_gas_cost,
        n_months=len(combined_df),
    )


def cross_check_fuel_totals(
    total_df: pd.DataFrame,
    electricity_df: pd.DataFrame,
    gas_df: pd.DataFrame,
    tolerance_kwh: float = 1.0,
    tolerance_gbp: float = 0.05,
) -> list[str]:
    """Verify Electricity + Gas == Total for every month present in all three frames.

    A genuine data-quality cross-check unique to having three correlated
    sources -- "never assume the data is clean" applied to the exports
    themselves, not just to parsing a single file. Small tolerances absorb
    the provider's own rounding, not a real discrepancy.
    """
    total_idx = total_df.set_index("month_start")
    elec_idx = electricity_df.set_index("month_start")
    gas_idx = gas_df.set_index("month_start")
    common_months = sorted(set(total_idx.index) & set(elec_idx.index) & set(gas_idx.index))

    warnings: list[str] = []
    for month in common_months:
        expected_kwh = elec_idx.loc[month, "consumption_kwh"] + gas_idx.loc[month, "consumption_kwh"]
        actual_kwh = total_idx.loc[month, "consumption_kwh"]
        expected_gbp = elec_idx.loc[month, "cost_gbp"] + gas_idx.loc[month, "cost_gbp"]
        actual_gbp = total_idx.loc[month, "cost_gbp"]

        if abs(actual_kwh - expected_kwh) > tolerance_kwh or abs(actual_gbp - expected_gbp) > tolerance_gbp:
            warnings.append(
                f"{month.strftime('%B %Y')}: Electricity + Gas ({expected_kwh:.2f} kWh, "
                f"£{expected_gbp:.2f}) doesn't match Total ({actual_kwh:.2f} kWh, £{actual_gbp:.2f})."
            )

    if warnings:
        logger.warning("%d month(s) failed the electricity+gas=total cross-check", len(warnings))
    return warnings


def combine_fuel_frames(electricity_df: pd.DataFrame, gas_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join electricity and gas on ``month_start``, adding each month's electricity share."""
    elec = electricity_df[["month_start", "consumption_kwh", "cost_gbp"]].rename(
        columns={"consumption_kwh": "electricity_kwh", "cost_gbp": "electricity_cost_gbp"}
    )
    gas = gas_df[["month_start", "consumption_kwh", "cost_gbp"]].rename(
        columns={"consumption_kwh": "gas_kwh", "cost_gbp": "gas_cost_gbp"}
    )
    combined = elec.merge(gas, on="month_start", how="inner").sort_values("month_start").reset_index(drop=True)

    total_kwh = combined["electricity_kwh"] + combined["gas_kwh"]
    combined["electricity_share_pct"] = np.where(total_kwh > 0, combined["electricity_kwh"] / total_kwh * 100, np.nan)
    return combined


def infer_total_from_electricity_and_gas(
    electricity_df: pd.DataFrame, gas_df: pd.DataFrame
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Synthesize a Total (Electricity + Gas) monthly series when no separate Total Use
    export exists, by summing the two already-cleaned per-fuel frames.

    A provider that only exports per-fuel breakdowns shouldn't lose the combined view --
    this is what lets ``app.sidebar`` offer "Total" even without a dedicated file.

    Inner join on ``month_start``: a month only appears in the result if both fuels have
    data for it, since a partial sum would understate the real total. Reuses
    :func:`src.preprocessing.build_monthly_series`/``validate_units``/``detect_missing_months``
    so the output is the exact same "clean monthly frame" shape and carries the same
    plausibility/missing-month checks every other module already expects -- a drop-in
    substitute for a real Total Use file wherever the app reads "total". Deliberately does
    not re-run :func:`src.preprocessing.deduplicate`: the inputs are already deduplicated
    individually, and summing them cannot introduce a new duplicate month.
    """
    if electricity_df.empty or gas_df.empty:
        empty = build_monthly_series(pd.DataFrame(columns=["month_start", "cost_gbp", "consumption_kwh"]))
        return empty, PreprocessingReport(
            n_files_loaded=0,
            source_files=[],
            date_range=None,
            n_months=0,
            missing_months=[],
            duplicates_removed=0,
        )

    elec = electricity_df[["month_start", "consumption_kwh", "cost_gbp"]]
    gas = gas_df[["month_start", "consumption_kwh", "cost_gbp"]]
    summed = (
        elec.merge(gas, on="month_start", suffixes=("_elec", "_gas"), how="inner")
        .assign(
            consumption_kwh=lambda d: d["consumption_kwh_elec"] + d["consumption_kwh_gas"],
            cost_gbp=lambda d: d["cost_gbp_elec"] + d["cost_gbp_gas"],
        )[["month_start", "cost_gbp", "consumption_kwh"]]
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    summed, outlier_warnings = validate_units(summed)
    missing_months = detect_missing_months(summed)
    clean = build_monthly_series(summed)

    date_range = (clean["month_start"].min(), clean["month_start"].max()) if not clean.empty else None
    report = PreprocessingReport(
        n_files_loaded=0,
        source_files=["inferred: Electricity Use + Gas Use"],
        date_range=date_range,
        n_months=len(clean),
        missing_months=missing_months,
        duplicates_removed=0,
        outlier_warnings=outlier_warnings,
    )
    return clean, report


def finding_fuel_mix(combined_df: pd.DataFrame) -> Finding | None:
    """Gas vs. electricity split and relative weather-sensitivity, from already-combined billing data.

    Returns None if fewer than 6 months have both fuels recorded -- too
    little to say anything about seasonality.
    """
    shares = compute_fuel_shares(combined_df)
    if shares is None:
        return None

    month_num = combined_df["month_start"].dt.month
    winter = combined_df[month_num.isin([12, 1, 2])]
    summer = combined_df[month_num.isin([6, 7, 8])]

    seasonality_sentence = ""
    if not winter.empty and not summer.empty:
        elec_ratio = safe_divide(winter["electricity_kwh"].mean(), summer["electricity_kwh"].mean(), default=float("nan"))
        gas_ratio = safe_divide(winter["gas_kwh"].mean(), summer["gas_kwh"].mean(), default=float("nan"))
        if not (pd.isna(elec_ratio) or pd.isna(gas_ratio)):
            more_seasonal_fuel = "gas" if gas_ratio > elec_ratio else "electricity"
            heating_type = "gas central heating" if more_seasonal_fuel == "gas" else "electric heating"
            seasonality_sentence = (
                f" Gas usage swings {gas_ratio:.1f}x between winter and summer, versus {elec_ratio:.1f}x "
                f"for electricity -- {more_seasonal_fuel} is the more weather-driven fuel here, consistent "
                f"with {heating_type}."
            )

    narrative = (
        f"Gas accounts for {shares.gas_share_kwh_pct:.0f}% of total energy consumption and "
        f"{shares.gas_share_cost_pct:.0f}% of total cost across the {shares.n_months} month(s) with both "
        "fuels recorded." + seasonality_sentence
    )

    return Finding(
        title="Fuel mix",
        narrative=narrative,
        evidence=[
            f"Electricity: {shares.total_electricity_kwh:,.0f} kWh, £{shares.total_electricity_cost_gbp:,.2f}",
            f"Gas: {shares.total_gas_kwh:,.0f} kWh, £{shares.total_gas_cost_gbp:,.2f}",
        ],
        confidence="High",
        confidence_reason="Directly summed from billing data across both fuels, not a statistical model.",
        category="fuel",
    )
