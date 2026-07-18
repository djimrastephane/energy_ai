"""Task 4 -- cost intelligence: per-fuel and combined billing breakdown, forecast bills.

Standing charges are deliberately omitted: the OVO exports only carry
Month/Cost/Consumption, with no standing-charge or tariff-rate column
anywhere in the raw data -- fabricating a split would violate the "never
invent a number" principle this platform is built on. Everything here
reuses existing per-period aggregation (``src.kpis``) and forecasting
(``src.forecast_evaluation``) rather than recomputing totals or refitting
anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.forecast_evaluation import ForecastResult
from src.ingestion import EnergyType
from src.kpis import full_year_comparison
from src.utils import safe_divide


@dataclass
class CostBreakdown:
    fuel: EnergyType
    annual_total_gbp: float
    monthly_avg_gbp: float
    effective_unit_rate_gbp_per_kwh: float
    trend_description: str


def compute_cost_breakdown(clean_df: pd.DataFrame, fuel: EnergyType) -> CostBreakdown | None:
    """One fuel's cost picture: total spend, monthly average, effective £/kWh, and a
    plain-English YoY trend description (reusing ``full_year_comparison`` -- no new
    aggregation). Returns ``None`` if ``clean_df`` is empty.
    """
    if clean_df.empty:
        return None

    yearly = full_year_comparison(clean_df)
    if len(yearly) >= 2 and pd.notna(yearly["yoy_cost_pct"].iloc[-1]):
        latest_change = float(yearly["yoy_cost_pct"].iloc[-1])
        direction = "up" if latest_change >= 0 else "down"
        trend = f"{abs(latest_change):.0f}% {direction} year-on-year (most recent complete year vs. previous)."
    else:
        trend = "Not enough complete calendar years yet for a year-on-year cost trend."

    return CostBreakdown(
        fuel=fuel,
        annual_total_gbp=float(clean_df["cost_gbp"].sum()),
        monthly_avg_gbp=float(clean_df["cost_gbp"].mean()),
        effective_unit_rate_gbp_per_kwh=safe_divide(clean_df["cost_gbp"].sum(), clean_df["consumption_kwh"].sum()),
        trend_description=trend,
    )


def compute_combined_cost_breakdown(fuel_clean_dfs: dict[EnergyType, pd.DataFrame]) -> pd.DataFrame:
    """All fuels' cost breakdowns as one table -- the Cost Intelligence tab's backbone."""
    rows = []
    for fuel, df in fuel_clean_dfs.items():
        breakdown = compute_cost_breakdown(df, fuel)
        if breakdown is None:
            continue
        rows.append(
            {
                "fuel": breakdown.fuel,
                "annual_total_gbp": breakdown.annual_total_gbp,
                "monthly_avg_gbp": breakdown.monthly_avg_gbp,
                "effective_unit_rate_gbp_per_kwh": breakdown.effective_unit_rate_gbp_per_kwh,
                "trend": breakdown.trend_description,
            }
        )
    return pd.DataFrame(
        rows, columns=["fuel", "annual_total_gbp", "monthly_avg_gbp", "effective_unit_rate_gbp_per_kwh", "trend"]
    )


@dataclass
class ForecastBillComparison:
    per_fuel: pd.DataFrame  # fuel, model, best_gbp, likely_gbp, worst_gbp
    summed_likely_gbp: float | None  # electricity + gas, if both present
    total_likely_gbp: float | None  # Total's own independent forecast
    divergence_pct: float | None  # (summed - total) / total * 100, if both present
    divergence_note: str | None  # only set if the two approaches diverge by >10%


def forecast_bill_by_fuel(
    fuel_forecast_results: dict[EnergyType, ForecastResult],
    fuel_unit_rates: dict[EnergyType, float],
) -> ForecastBillComparison:
    """Convert each fuel's *already-computed* forecast (kWh P10/P50/P90) to a £ bill using
    that fuel's own effective unit rate -- no new forecasting model, same conversion the
    Executive Briefing already does for Total alone. Also reports the combined bill both as
    Electricity-£ + Gas-£ and as Total's own independent forecast-£; a >10% divergence
    between the two is surfaced as an honest signal (different models were auto-selected
    per fuel), not hidden.
    """
    rows = [
        {
            "fuel": fuel,
            "model": result.model_name,
            "best_gbp": float(result.p10.sum()) * fuel_unit_rates.get(fuel, 0.0),
            "likely_gbp": float(result.p50.sum()) * fuel_unit_rates.get(fuel, 0.0),
            "worst_gbp": float(result.p90.sum()) * fuel_unit_rates.get(fuel, 0.0),
        }
        for fuel, result in fuel_forecast_results.items()
    ]
    per_fuel = pd.DataFrame(rows, columns=["fuel", "model", "best_gbp", "likely_gbp", "worst_gbp"])

    summed_likely = total_likely = divergence_pct = None
    divergence_note = None
    elec = per_fuel[per_fuel["fuel"] == "electricity"]
    gas = per_fuel[per_fuel["fuel"] == "gas"]
    total = per_fuel[per_fuel["fuel"] == "total"]
    if not elec.empty and not gas.empty:
        summed_likely = float(elec["likely_gbp"].iloc[0] + gas["likely_gbp"].iloc[0])
        if not total.empty:
            total_likely = float(total["likely_gbp"].iloc[0])
            if total_likely > 0:
                divergence_pct = (summed_likely - total_likely) / total_likely * 100
                if abs(divergence_pct) > 10:
                    direction = "higher" if divergence_pct > 0 else "lower"
                    divergence_note = (
                        f"Electricity + Gas forecasts sum to £{summed_likely:,.0f}, {abs(divergence_pct):.0f}% "
                        f"{direction} than Total's own independent forecast (£{total_likely:,.0f}) -- the two "
                        "forecasting approaches (summing two separately auto-selected models vs. one model on "
                        "the combined series) don't always agree exactly."
                    )

    return ForecastBillComparison(
        per_fuel=per_fuel,
        summed_likely_gbp=summed_likely,
        total_likely_gbp=total_likely,
        divergence_pct=divergence_pct,
        divergence_note=divergence_note,
    )
