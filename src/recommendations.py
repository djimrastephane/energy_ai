"""Rule-based recommendation engine: a recommendation is only produced when evidence supports it.

No solar/EV/battery rules exist in this module at all -- "never recommend
those without sufficient information" is enforced by the absence of a code
path, not a runtime check, since this dataset has no roof, vehicle, or
appliance data to evaluate them against. Every recommendation carries its
evidence so it's traceable back to the numbers that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.anomalies import Anomaly
from src.comparisons import WeatherSensitivityShares
from src.confidence import Confidence, ConfidenceRating
from src.energy_signature import EnergySignatureResult
from src.fuel import FuelShares
from src.investigation import InvestigationChecklist
from src.utils import format_gbp, get_logger, winter_season_label

logger = get_logger(__name__)

NO_RECOMMENDATIONS_MESSAGE = (
    "Unable to recommend additional actions at this time -- no findings currently meet the "
    "evidence threshold for a specific recommendation."
)

_WINTER_MONTHS = {12, 1, 2}
Z_ONE_SIDED_95 = 1.645  # standard one-sided 95% normal-approximation threshold


@dataclass
class Recommendation:
    title: str
    action: str
    estimated_saving_gbp: float | None
    evidence: list[str]
    confidence: Confidence
    confidence_reason: str
    rationale: str


def recommend_heating_review(
    clean_df: pd.DataFrame,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    reduction_pct: float = 0.10,
) -> Recommendation | None:
    """Gate: weather model available, heating significant, and the latest complete winter's actual
    consumption exceeds the weather-predicted amount by more than a typical month's residual spread.
    """
    if merged_df is None or energy_result is None:
        return None
    heating_significant = not pd.isna(energy_result.heating_pvalue) and energy_result.heating_pvalue < 0.05
    if not heating_significant:
        return None

    merged_indexed = merged_df.sort_values("month_start").set_index("month_start")
    winter_rows = merged_indexed[merged_indexed.index.month.isin(_WINTER_MONTHS)].copy()
    if winter_rows.empty:
        return None
    winter_rows["season"] = [winter_season_label(d) for d in winter_rows.index]
    complete_seasons = winter_rows.groupby("season").size()
    complete_seasons = complete_seasons[complete_seasons == 3].index
    if len(complete_seasons) == 0:
        return None
    latest_season = max(complete_seasons)
    season_rows = winter_rows[winter_rows["season"] == latest_season]

    # heating_slope is kWh/day per heating-degree-day; avg_daily_hdd * days_in_month gives each
    # month's total heating-degree-days, so heating_slope * that sum = total heating-attributable kWh.
    monthly_resid_kwh = energy_result.resid * merged_indexed["days_in_month"]
    winter_resid_kwh = monthly_resid_kwh.reindex(season_rows.index).sum()
    # Baseline std excludes the season being tested -- including it would let a genuine elevation
    # inflate its own baseline and mask itself, the same self-contamination bug already caught and
    # fixed once in this codebase for the rolling z-score anomaly detector (src/anomalies.py).
    baseline_std_kwh = monthly_resid_kwh.drop(season_rows.index, errors="ignore").std()
    # Summing 3 independent months' residuals scales their combined std by sqrt(3) -- comparing
    # against 1x that (a 1-sigma one-sided test) fires on pure noise ~16% of the time (measured
    # empirically), too loose for a specific £-savings claim. Z_ONE_SIDED_95 is the standard
    # one-sided 95% threshold (measured empirically at ~5-6% false-positive rate on pure noise).
    season_threshold_kwh = baseline_std_kwh * (len(season_rows) ** 0.5) * Z_ONE_SIDED_95
    if pd.isna(baseline_std_kwh) or baseline_std_kwh <= 0 or winter_resid_kwh <= season_threshold_kwh:
        return None

    unit_rate = clean_df["cost_gbp"].sum() / clean_df["consumption_kwh"].sum()
    winter_hdd_total = (season_rows["avg_daily_hdd"] * season_rows["days_in_month"]).sum()
    savings_kwh = reduction_pct * energy_result.heating_slope * winter_hdd_total
    savings_gbp = savings_kwh * unit_rate
    season_label = f"winter {latest_season - 1}/{latest_season}"

    return Recommendation(
        title="Review heating schedule",
        action=(
            f"Winter consumption ({season_label}) is running above what the weather model would predict. "
            f"Reducing winter heating demand by {reduction_pct:.0%} could reduce annual costs by "
            f"approximately {format_gbp(savings_gbp)}."
        ),
        estimated_saving_gbp=savings_gbp,
        evidence=[
            f"Heating sensitivity: {energy_result.heating_slope:.2f} kWh/day per heating-degree-day "
            f"(p={energy_result.heating_pvalue:.3f})",
            f"{season_label} actual vs. weather-predicted: {winter_resid_kwh:+.0f} kWh above expected",
        ],
        confidence="Medium",
        confidence_reason="Based on a single winter's deviation from the weather model -- more winters of data would firm this up.",
        rationale=(
            f"{season_label} consumption exceeded the weather-adjusted expectation by more than a "
            "typical month's residual variability, and heating sensitivity is statistically significant."
        ),
    )


def recommend_more_data(forecast_rating: ConfidenceRating | None, n_months: int) -> Recommendation | None:
    """Gate: forecast confidence is Low."""
    if forecast_rating is None or forecast_rating.level != "Low":
        return None
    return Recommendation(
        title="Collect more historical data",
        action=(
            f"With {n_months} months of history, forecast accuracy is limited. Continuing to track "
            "consumption for another year or more would meaningfully improve forecast reliability."
        ),
        estimated_saving_gbp=None,
        evidence=[forecast_rating.reason],
        confidence="High",
        confidence_reason="Forecast accuracy reliably improves with more historical data -- this is a "
        "data-collection recommendation, not a savings estimate.",
        rationale="The forecast's own cross-validation shows high uncertainty relative to typical consumption.",
    )


def recommend_investigate_anomaly(
    anomaly: Anomaly, checklist: InvestigationChecklist, anomaly_rating: ConfidenceRating
) -> Recommendation | None:
    """Gate: anomaly confidence is Medium or High (2+ independent methods agree)."""
    if anomaly_rating.level == "Low":
        return None

    checked_causes = [item for item in checklist.items if item.checked and item.label != "Unknown"]
    month_label = anomaly.date.strftime("%B %Y")
    if checked_causes:
        cause_labels = ", ".join(item.label.lower() for item in checked_causes)
        action = f"Review {cause_labels} for {month_label} -- the data is consistent with this explaining the unusual reading."
    else:
        action = (
            f"{month_label} remains unexplained by weather or tariff changes. Review occupancy, new "
            "appliances, or heating changes around this date."
        )

    return Recommendation(
        title=f"Investigate {month_label}",
        action=action,
        estimated_saving_gbp=None,
        evidence=[f"Detected by: {', '.join(anomaly.methods)}", anomaly.rank_context],
        confidence=anomaly_rating.level,
        confidence_reason=anomaly_rating.reason,
        rationale="Anomaly detection flagged this month as statistically unusual relative to the trend and season.",
    )


_DOMINANCE_THRESHOLD_PCT = 60.0


def recommend_fuel_focus(
    fuel_shares: FuelShares | None,
    weather_shares: WeatherSensitivityShares | None,
) -> Recommendation | None:
    """Gate: one fuel clearly dominates both consumption/cost share (``fuel_shares``, from
    ``src.fuel.compute_fuel_shares``) and weather-sensitivity share (``weather_shares``, from
    ``src.comparisons.compute_weather_sensitivity_shares``) -- both already-computed elsewhere,
    reused here rather than re-derived. Fires only when both agree on the same dominant fuel
    and each share exceeds a 60% "clearly dominant" threshold; otherwise returns None rather
    than picking a fuel to focus on when the evidence doesn't clearly point at one.
    """
    if fuel_shares is None or weather_shares is None:
        return None

    consumption_dominant = (
        "gas" if fuel_shares.gas_share_kwh_pct >= fuel_shares.electricity_share_kwh_pct else "electricity"
    )
    if consumption_dominant != weather_shares.dominant_fuel:
        return None

    consumption_share = (
        fuel_shares.gas_share_kwh_pct if consumption_dominant == "gas" else fuel_shares.electricity_share_kwh_pct
    )
    weather_share = (
        weather_shares.gas_share_pct if consumption_dominant == "gas" else weather_shares.electricity_share_pct
    )
    if consumption_share < _DOMINANCE_THRESHOLD_PCT or weather_share < _DOMINANCE_THRESHOLD_PCT:
        return None

    other_fuel = "electricity" if consumption_dominant == "gas" else "gas"
    both_significant = weather_shares.gas_significant and weather_shares.electricity_significant
    confidence: Confidence = "High" if both_significant else "Medium"
    confidence_reason = (
        "Both fuels show a statistically significant heating response, so the weather-sensitivity "
        "split is well-supported."
        if both_significant
        else "The weather-sensitivity split rests on only one fuel's statistically significant heating response."
    )

    return Recommendation(
        title=f"Focus on {consumption_dominant}",
        action=(
            f"{consumption_dominant.capitalize()} accounts for {consumption_share:.0f}% of annual energy "
            f"and drives {weather_share:.0f}% of heating sensitivity; reducing {consumption_dominant} "
            f"demand is likely to produce larger savings than reducing {other_fuel} use."
        ),
        estimated_saving_gbp=None,
        evidence=[
            f"Consumption share: {consumption_dominant} {consumption_share:.0f}%",
            f"Heating-sensitivity share: {consumption_dominant} {weather_share:.0f}%",
        ],
        confidence=confidence,
        confidence_reason=confidence_reason,
        rationale=(
            f"{consumption_dominant.capitalize()} dominates both total energy/cost share and the "
            "weather-driven (heating) portion of consumption, so it's the fuel where usage reduction "
            "would have the largest impact on the bill."
        ),
    )


def generate_recommendations(
    clean_df: pd.DataFrame,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
    forecast_rating: ConfidenceRating | None,
    n_months: int,
    top_anomaly: Anomaly | None = None,
    top_anomaly_checklist: InvestigationChecklist | None = None,
    top_anomaly_rating: ConfidenceRating | None = None,
    fuel_shares: FuelShares | None = None,
    weather_shares: WeatherSensitivityShares | None = None,
) -> list[Recommendation]:
    """Run every recommendation rule and return whichever ones actually fired."""
    candidates = [
        recommend_heating_review(clean_df, merged_df, energy_result),
        recommend_more_data(forecast_rating, n_months),
        recommend_fuel_focus(fuel_shares, weather_shares),
    ]
    if top_anomaly is not None and top_anomaly_checklist is not None and top_anomaly_rating is not None:
        candidates.append(recommend_investigate_anomaly(top_anomaly, top_anomaly_checklist, top_anomaly_rating))
    return [r for r in candidates if r is not None]
