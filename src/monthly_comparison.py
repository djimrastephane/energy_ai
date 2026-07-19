"""Monthly comparison domain model: one selected month against a comparable period.

This is the calculation layer behind the "How did this month compare?"
journey. The default comparison everywhere is the latest *complete* month
against the same calendar month one year earlier -- never the immediately
preceding month (January vs. December is a seasonality comparison, not a
performance one), and never a partial in-progress month presented as
complete.

Everything here reuses numbers already computed elsewhere: monthly totals
from the clean billing frame (``src.preprocessing``), weather expectations
from the already-fitted degree-day regression (``src.energy_signature`` --
nothing is refitted), and the shared High/Medium/Low confidence taxonomy
(``src.confidence``). No Streamlit imports; every function is pure and
unit-testable, per the project's src/app split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from config import SETTINGS, MonthComparisonThresholds
from src.billing import is_billing_month_complete
from src.confidence import Confidence
from src.energy_signature import EnergySignatureResult
from src.ingestion import EnergyType
from src.utils import pct_change, safe_divide

ComparisonMode = Literal[
    "same_month_last_year",  # June 2026 vs June 2025 -- the default
    "previous_month",  # June 2026 vs May 2026 -- short-term movement only
    "typical_month",  # June 2026 vs the median of previous Junes
    "best_month",  # June 2026 vs the lowest-consumption previous June
    "worst_month",  # June 2026 vs the highest-consumption previous June
]

COMPARISON_MODE_LABELS: dict[ComparisonMode, str] = {
    "same_month_last_year": "Same month last year",
    "previous_month": "Previous complete month",
    "typical_month": "Typical value for this calendar month",
    "best_month": "Best recorded value for this calendar month",
    "worst_month": "Worst recorded value for this calendar month",
}

ChangeCategory = Literal["little", "moderate", "large"]

# Judgement labels for the primary comparison card (colour is support, not the
# only signal -- these words carry the verdict; see docs for the rules).
JudgementLabel = Literal[
    "Better than comparable period",
    "Similar",
    "Higher but weather-explained",
    "Higher and unexplained",
    "Higher",
    "No comparison available",
]


@dataclass(frozen=True)
class MonthlyComparison:
    """One month compared against one comparable period, fully computed.

    ``comparison_month`` is ``None`` for distribution-style modes
    (``typical_month``) where the comparator is a summary of several months
    rather than a single one, and for any mode where no valid comparator
    exists (the honest-fallback path -- ``interpretation`` then says why).
    All ``*_kwh``/``*_gbp`` fields are whole-month totals for the stated
    month; ``*_weather_adjusted_kwh`` are actual consumption minus the
    fitted degree-day-driven component (base load + residual), i.e. what
    the month would have used with the weather effect removed.
    """

    selected_month: pd.Timestamp
    comparison_mode: ComparisonMode
    comparison_month: pd.Timestamp | None
    fuel: EnergyType
    current_consumption_kwh: float
    comparison_consumption_kwh: float | None
    absolute_change_kwh: float | None
    percentage_change: float | None
    current_avg_daily_kwh: float
    comparison_avg_daily_kwh: float | None
    current_cost_gbp: float | None
    comparison_cost_gbp: float | None
    cost_change_gbp: float | None
    # Exact split of the cost change: usage priced at the comparison month's
    # effective rate, plus the rate movement priced at the current month's usage.
    cost_change_from_usage_gbp: float | None
    cost_change_from_rate_gbp: float | None
    current_weather_adjusted_kwh: float | None
    comparison_weather_adjusted_kwh: float | None
    weather_explained_change_kwh: float | None
    unexplained_change_kwh: float | None
    unexplained_change_is_meaningful: bool | None  # vs the fit's residual spread; None = no model
    current_heating_degree_days: float | None
    comparison_heating_degree_days: float | None
    # Same-calendar-month history (previous years only, selected month excluded):
    same_month_low_kwh: float | None
    same_month_median_kwh: float | None
    same_month_high_kwh: float | None
    same_month_years: tuple[int, ...]
    data_complete: bool  # False only when the selected month is the in-progress calendar month
    change_category: ChangeCategory | None
    judgement: JudgementLabel
    confidence: Confidence
    confidence_reason: str
    interpretation: str


def is_month_complete(month: pd.Timestamp, today: pd.Timestamp | None = None) -> bool:
    """A month is complete once its *billing period* has fully elapsed.

    The exports follow a 6th-to-5th billing cycle (``config.BillingConfig``):
    the June 2026 row covers 6 Jun - 5 Jul, so June is only complete from
    6 July onward -- not from 1 July, as a calendar-month reading would
    assume. Delegates to ``src.billing.is_billing_month_complete``, the
    single source of truth ``src.preprocessing`` shares.
    """
    return is_billing_month_complete(month, today)


def latest_complete_month(clean_df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """The most recent month in the data that is a complete calendar month.

    This is the default focus of the whole journey -- never the in-progress
    month. Returns ``None`` if the data holds nothing but the current month.
    """
    if clean_df.empty:
        return None
    complete = clean_df[clean_df["month_start"].apply(lambda m: is_month_complete(m, today))]
    if complete.empty:
        return None
    return complete["month_start"].max()


def in_progress_month(clean_df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """The current in-progress calendar month, if the data contains it (else None)."""
    if clean_df.empty:
        return None
    latest = clean_df["month_start"].max()
    return latest if not is_month_complete(latest, today) else None


def selectable_months(clean_df: pd.DataFrame, today: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """Complete months only, newest first -- the month selector's option list.

    The in-progress month is deliberately not selectable as a primary month;
    it is shown separately as month-to-date (see ``month_to_date_note``).
    """
    if clean_df.empty:
        return []
    months = [m for m in clean_df["month_start"] if is_month_complete(m, today)]
    return sorted(months, reverse=True)


def month_to_date_note(clean_df: pd.DataFrame, today: pd.Timestamp | None = None) -> str | None:
    """Plain-language notice for an in-progress month, or None when every month is complete.

    Example: "July 2026 is incomplete. The primary comparison uses June 2026."
    Never annualizes or extrapolates the partial figure.
    """
    partial = in_progress_month(clean_df, today)
    if partial is None:
        return None
    latest_complete = latest_complete_month(clean_df, today)
    note = f"{partial.strftime('%B %Y')} is incomplete."
    if latest_complete is not None:
        note += f" The primary comparison uses {latest_complete.strftime('%B %Y')}."
    return note


def same_month_history(
    clean_df: pd.DataFrame, selected_month: pd.Timestamp, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Every *previous* complete observation of the selected month's calendar month.

    January is only ever compared with previous Januaries -- rows are earlier
    years of the same ``month_num``, excluding the selected month itself and
    excluding any in-progress month. Sorted oldest first.
    """
    if clean_df.empty:
        return clean_df
    mask = (
        (clean_df["month_start"].dt.month == selected_month.month)
        & (clean_df["month_start"] < selected_month)
        & clean_df["month_start"].apply(lambda m: is_month_complete(m, today))
    )
    return clean_df[mask].sort_values("month_start").reset_index(drop=True)


def categorize_change(
    percentage_change: float | None, thresholds: MonthComparisonThresholds | None = None
) -> ChangeCategory | None:
    """Map a % change to little/moderate/large using the documented practical thresholds.

    A 1% movement is never narrated as meaningful; thresholds live in
    ``config.MonthComparisonThresholds`` with their rationale.
    """
    if percentage_change is None:
        return None
    thresholds = thresholds or SETTINGS.month_comparison
    magnitude = abs(percentage_change)
    if magnitude < thresholds.little_change_pct:
        return "little"
    if magnitude < thresholds.large_change_pct:
        return "moderate"
    return "large"


def _row_for(clean_df: pd.DataFrame, month: pd.Timestamp) -> pd.Series | None:
    rows = clean_df[clean_df["month_start"] == month]
    return rows.iloc[0] if not rows.empty else None


def _weather_month_values(
    month: pd.Timestamp,
    merged_df: pd.DataFrame | None,
    energy_result: EnergySignatureResult | None,
) -> tuple[float, float, float] | None:
    """(weather_expected_kwh, weather_adjusted_kwh, heating_degree_days) for one month.

    Reads the *already-fitted* regression: expected = fitted kWh/day x days;
    adjusted = actual minus the degree-day-driven component (i.e. base load +
    residual). Returns None when the month isn't covered by the fit.
    """
    if merged_df is None or energy_result is None:
        return None
    rows = merged_df[merged_df["month_start"] == month]
    if rows.empty or month not in energy_result.fitted.index:
        return None
    row = rows.iloc[0]
    days = float(row["days_in_month"])
    expected_kwh = float(energy_result.fitted.loc[month]) * days
    base_kwh = energy_result.intercept * days
    actual_kwh = float(row["consumption_kwh"])
    adjusted_kwh = actual_kwh - (expected_kwh - base_kwh)
    hdd = float(row["hdd"]) if "hdd" in row.index else float(row["avg_daily_hdd"]) * days
    return expected_kwh, adjusted_kwh, hdd


def _monthly_resid_std_kwh(energy_result: EnergySignatureResult, merged_df: pd.DataFrame) -> float:
    """Std of the fit's residuals in whole-month kWh -- same construction as
    ``src.energy_signature.annual_weather_adjusted_comparison``."""
    days = merged_df.set_index("month_start")["days_in_month"]
    return float((energy_result.resid * days).std())


def weather_explains_most(
    weather_explained_change_kwh: float | None,
    absolute_change_kwh: float | None,
    weather_was_colder: bool | None,
) -> bool | None:
    """Does the weather genuinely account for the larger share of an increase?

    Requires the weather to have moved in the explaining direction (colder,
    positive model-expected change) *and* to cover at least half the actual
    change -- a gap merely sitting inside residual noise is not the same as
    the weather explaining it. Returns ``None`` when there's no model."""
    if weather_explained_change_kwh is None or absolute_change_kwh is None:
        return None
    if not weather_was_colder or weather_explained_change_kwh <= 0:
        return False
    if abs(absolute_change_kwh) < 1e-9:
        return False
    return weather_explained_change_kwh / absolute_change_kwh >= 0.5


def _judge(
    percentage_change: float | None,
    change_category: ChangeCategory | None,
    weather_explained_change_kwh: float | None,
    absolute_change_kwh: float | None,
    unexplained_is_meaningful: bool | None,
    weather_was_colder: bool | None,
) -> JudgementLabel:
    """Word-first verdict for the primary card. Higher winter use may be expected,
    so 'higher' is only 'unexplained' when the weather model says the gap is real --
    and only 'weather-explained' when the weather covers at least half the change
    (see :func:`weather_explains_most`)."""
    if percentage_change is None:
        return "No comparison available"
    if change_category == "little":
        return "Similar"
    if percentage_change < 0:
        return "Better than comparable period"
    if unexplained_is_meaningful is None:
        return "Higher"
    if unexplained_is_meaningful:
        return "Higher and unexplained"
    if weather_explains_most(weather_explained_change_kwh, absolute_change_kwh, weather_was_colder):
        return "Higher but weather-explained"
    return "Higher"


def _comparison_target(
    mode: ComparisonMode,
    clean_df: pd.DataFrame,
    selected_month: pd.Timestamp,
    history: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Series | None, str | None]:
    """Resolve the comparator for a mode: (comparison_month, its row, failure_reason)."""
    month_label = selected_month.strftime("%B")
    if mode == "same_month_last_year":
        target = selected_month - pd.DateOffset(years=1)
        row = _row_for(clean_df, target)
        if row is None:
            return None, None, (
                f"No {month_label} {selected_month.year - 1} in the data, so a same-month "
                "year-on-year comparison isn't possible."
            )
        return target, row, None
    if mode == "previous_month":
        earlier = clean_df[clean_df["month_start"] < selected_month]
        if earlier.empty:
            return None, None, "No earlier month in the data to compare against."
        row = earlier.sort_values("month_start").iloc[-1]
        return row["month_start"], row, None
    # Same-calendar-month distribution modes.
    if history.empty:
        return None, None, (
            f"There is not enough history to compare this {month_label} with previous "
            f"{month_label}s."
        )
    if mode == "typical_month":
        return None, None, None  # comparator is the median, not a single month
    if mode == "best_month":
        row = history.loc[history["consumption_kwh"].idxmin()]
        return row["month_start"], row, None
    if mode == "worst_month":
        row = history.loc[history["consumption_kwh"].idxmax()]
        return row["month_start"], row, None
    raise ValueError(f"Unknown comparison mode: {mode!r}")


def _confidence_for(
    mode: ComparisonMode,
    data_complete: bool,
    n_history: int,
    has_weather: bool,
    thresholds: MonthComparisonThresholds,
) -> tuple[Confidence, str]:
    """Rate the comparison itself (not the underlying models, which carry their own ratings)."""
    if not data_complete:
        return "Low", "The selected month is still in progress -- its figures are month-to-date."
    if mode in ("typical_month", "best_month", "worst_month"):
        if n_history < thresholds.solid_same_month_observations:
            return "Medium", (
                f"Only {n_history} previous observation(s) of this calendar month -- the "
                "typical/best/worst picture will firm up as more years accumulate."
            )
        return "High", f"{n_history} previous observations of this calendar month."
    if mode == "previous_month":
        return "Medium", (
            "Adjacent months can differ because of seasonality -- use this for short-term "
            "movement, not performance judgement."
        )
    if has_weather:
        return "High", (
            "Both months are complete billing months and the weather model covers both, so "
            "the change can be split into weather and non-weather parts."
        )
    return "High", "Both months are complete billing months summed directly from billing data."


def _fallback(
    selected_month: pd.Timestamp,
    mode: ComparisonMode,
    fuel: EnergyType,
    row: pd.Series,
    data_complete: bool,
    history: pd.DataFrame,
    reason: str,
) -> MonthlyComparison:
    """The honest no-comparison result: current month's own numbers, nothing invented."""
    years = tuple(int(y) for y in history["month_start"].dt.year) if not history.empty else ()
    return MonthlyComparison(
        selected_month=selected_month,
        comparison_mode=mode,
        comparison_month=None,
        fuel=fuel,
        current_consumption_kwh=float(row["consumption_kwh"]),
        comparison_consumption_kwh=None,
        absolute_change_kwh=None,
        percentage_change=None,
        current_avg_daily_kwh=float(row["consumption_kwh"]) / float(row["days_in_month"]),
        comparison_avg_daily_kwh=None,
        current_cost_gbp=float(row["cost_gbp"]) if pd.notna(row["cost_gbp"]) else None,
        comparison_cost_gbp=None,
        cost_change_gbp=None,
        cost_change_from_usage_gbp=None,
        cost_change_from_rate_gbp=None,
        current_weather_adjusted_kwh=None,
        comparison_weather_adjusted_kwh=None,
        weather_explained_change_kwh=None,
        unexplained_change_kwh=None,
        unexplained_change_is_meaningful=None,
        current_heating_degree_days=None,
        comparison_heating_degree_days=None,
        same_month_low_kwh=None,
        same_month_median_kwh=None,
        same_month_high_kwh=None,
        same_month_years=years,
        data_complete=data_complete,
        change_category=None,
        judgement="No comparison available",
        confidence="Low",
        confidence_reason=reason,
        interpretation=reason,
    )


def build_monthly_comparison(
    clean_df: pd.DataFrame,
    fuel: EnergyType,
    mode: ComparisonMode = "same_month_last_year",
    selected_month: pd.Timestamp | None = None,
    merged_df: pd.DataFrame | None = None,
    energy_result: EnergySignatureResult | None = None,
    today: pd.Timestamp | None = None,
    thresholds: MonthComparisonThresholds | None = None,
) -> MonthlyComparison | None:
    """Build the typed comparison for one fuel, one month, one mode.

    ``selected_month`` defaults to the latest complete month. ``merged_df``/
    ``energy_result`` are the *already-fitted* weather objects for the same
    fuel (optional -- without them the weather fields are ``None``, never
    guessed). Returns ``None`` only when ``clean_df`` is empty or the
    selected month isn't in the data at all.
    """
    thresholds = thresholds or SETTINGS.month_comparison
    if clean_df.empty:
        return None
    if selected_month is None:
        selected_month = latest_complete_month(clean_df, today)
        if selected_month is None:
            return None
    current_row = _row_for(clean_df, selected_month)
    if current_row is None:
        return None

    data_complete = is_month_complete(selected_month, today)
    history = same_month_history(clean_df, selected_month, today)
    n_history = len(history)
    month_label = selected_month.strftime("%B")

    if mode in ("typical_month", "best_month", "worst_month") and n_history < thresholds.min_same_month_observations:
        return _fallback(
            selected_month, mode, fuel, current_row, data_complete, history,
            f"There is not enough history to compare this {month_label} with previous "
            f"{month_label}s ({n_history} previous observation(s); at least "
            f"{thresholds.min_same_month_observations} needed).",
        )

    comparison_month, comparison_row, failure = _comparison_target(mode, clean_df, selected_month, history)
    if failure is not None:
        return _fallback(selected_month, mode, fuel, current_row, data_complete, history, failure)

    current_kwh = float(current_row["consumption_kwh"])
    current_cost = float(current_row["cost_gbp"]) if pd.notna(current_row["cost_gbp"]) else None
    current_days = float(current_row["days_in_month"])

    if mode == "typical_month":
        comparison_kwh: float | None = float(history["consumption_kwh"].median())
        comparison_cost = float(history["cost_gbp"].median()) if history["cost_gbp"].notna().all() else None
        comparison_days = float(history["days_in_month"].median())
    else:
        assert comparison_row is not None
        comparison_kwh = float(comparison_row["consumption_kwh"])
        comparison_cost = float(comparison_row["cost_gbp"]) if pd.notna(comparison_row["cost_gbp"]) else None
        comparison_days = float(comparison_row["days_in_month"])

    absolute_change = current_kwh - comparison_kwh
    percentage = pct_change(current_kwh, comparison_kwh)

    cost_change = cost_from_usage = cost_from_rate = None
    if current_cost is not None and comparison_cost is not None:
        cost_change = current_cost - comparison_cost
        comparison_rate = safe_divide(comparison_cost, comparison_kwh)
        current_rate = safe_divide(current_cost, current_kwh)
        if comparison_kwh > 0 and current_kwh > 0:
            # Exact decomposition: Δcost = Δkwh * rate_cmp + Δrate * kwh_cur
            cost_from_usage = absolute_change * comparison_rate
            cost_from_rate = (current_rate - comparison_rate) * current_kwh

    # Weather decomposition -- only for single-month comparators the fit covers.
    current_weather = _weather_month_values(selected_month, merged_df, energy_result)
    comparison_weather = (
        _weather_month_values(comparison_month, merged_df, energy_result)
        if comparison_month is not None
        else None
    )
    current_adjusted = comparison_adjusted = None
    weather_explained = unexplained = None
    unexplained_meaningful = None
    current_hdd = comparison_hdd = None
    if current_weather is not None:
        _, current_adjusted, current_hdd = current_weather
    if current_weather is not None and comparison_weather is not None:
        expected_cur, _, _ = current_weather
        expected_cmp, comparison_adjusted, comparison_hdd = comparison_weather
        weather_explained = expected_cur - expected_cmp
        unexplained = absolute_change - weather_explained
        if merged_df is not None and energy_result is not None:
            resid_std = _monthly_resid_std_kwh(energy_result, merged_df)
            if resid_std > 0:
                unexplained_meaningful = abs(unexplained) > thresholds.meaningful_residual_z * resid_std

    change_category = categorize_change(percentage, thresholds)
    weather_was_colder = (
        current_hdd > comparison_hdd
        if current_hdd is not None and comparison_hdd is not None
        else None
    )
    judgement = _judge(
        percentage, change_category, weather_explained, absolute_change,
        unexplained_meaningful, weather_was_colder,
    )
    confidence, confidence_reason = _confidence_for(
        mode, data_complete, n_history, weather_explained is not None, thresholds
    )

    history_low = float(history["consumption_kwh"].min()) if not history.empty else None
    history_median = float(history["consumption_kwh"].median()) if not history.empty else None
    history_high = float(history["consumption_kwh"].max()) if not history.empty else None

    interpretation = _interpret(
        selected_month, mode, comparison_month, percentage, change_category,
        absolute_change, weather_explained, unexplained, unexplained_meaningful,
        current_hdd, comparison_hdd, current_kwh, history_low, history_median, history_high,
    )
    return MonthlyComparison(
        selected_month=selected_month,
        comparison_mode=mode,
        comparison_month=comparison_month,
        fuel=fuel,
        current_consumption_kwh=current_kwh,
        comparison_consumption_kwh=comparison_kwh,
        absolute_change_kwh=absolute_change,
        percentage_change=percentage,
        current_avg_daily_kwh=current_kwh / current_days,
        comparison_avg_daily_kwh=comparison_kwh / comparison_days if comparison_days else None,
        current_cost_gbp=current_cost,
        comparison_cost_gbp=comparison_cost,
        cost_change_gbp=cost_change,
        cost_change_from_usage_gbp=cost_from_usage,
        cost_change_from_rate_gbp=cost_from_rate,
        current_weather_adjusted_kwh=current_adjusted,
        comparison_weather_adjusted_kwh=comparison_adjusted,
        weather_explained_change_kwh=weather_explained,
        unexplained_change_kwh=unexplained,
        unexplained_change_is_meaningful=unexplained_meaningful,
        current_heating_degree_days=current_hdd,
        comparison_heating_degree_days=comparison_hdd,
        same_month_low_kwh=history_low,
        same_month_median_kwh=history_median,
        same_month_high_kwh=history_high,
        same_month_years=tuple(int(y) for y in history["month_start"].dt.year),
        data_complete=data_complete,
        change_category=change_category,
        judgement=judgement,
        confidence=confidence,
        confidence_reason=confidence_reason,
        interpretation=interpretation,
    )


def _interpret(
    selected_month: pd.Timestamp,
    mode: ComparisonMode,
    comparison_month: pd.Timestamp | None,
    percentage: float | None,
    change_category: ChangeCategory | None,
    absolute_change_kwh: float | None,
    weather_explained: float | None,
    unexplained: float | None,
    unexplained_meaningful: bool | None,
    current_hdd: float | None,
    comparison_hdd: float | None,
    current_kwh: float,
    history_low: float | None,
    history_median: float | None,
    history_high: float | None,
) -> str:
    """One deterministic plain-language sentence -- the card-level takeaway.

    The longer structured narrative lives in ``src.monthly_narrative``; this
    is the single sentence the primary card and Consultant lead with.
    """
    month_label = selected_month.strftime("%B %Y")
    if percentage is None:
        return f"No comparison is available for {month_label}."

    if mode in ("typical_month", "best_month", "worst_month"):
        month_name = selected_month.strftime("%B")
        if mode == "typical_month":
            if history_low is not None and history_low <= current_kwh <= (history_high or current_kwh):
                position = f"within the normal {month_name} range"
            elif history_low is not None and current_kwh < history_low:
                position = f"below every previous {month_name} on record"
            else:
                position = f"above every previous {month_name} on record"
            direction = "more" if percentage >= 0 else "less"
            return (
                f"{month_label} used {abs(percentage):.0f}% {direction} than a typical "
                f"{month_name}, and sits {position}."
            )
        if mode == "best_month":
            if percentage < 0:
                return f"{month_label} is your best {month_name} on record."
            return (
                f"{month_label} used {abs(percentage):.0f}% more than your best recorded "
                f"{month_name} ({comparison_month.year})."
            )
        if percentage > 0:
            return f"{month_label} is your worst {month_name} on record."
        return (
            f"{month_label} used {abs(percentage):.0f}% less than your worst recorded "
            f"{month_name} ({comparison_month.year})."
        )

    comparison_label = comparison_month.strftime("%B %Y") if comparison_month is not None else "the comparison period"
    if change_category == "little":
        return f"{month_label} was about the same as {comparison_label} ({percentage:+.0f}%)."

    direction = "more" if percentage >= 0 else "less"
    base = f"{month_label} used {abs(percentage):.0f}% {direction} energy than {comparison_label}."

    if weather_explained is None or unexplained is None or unexplained_meaningful is None:
        return base

    colder = current_hdd is not None and comparison_hdd is not None and current_hdd > comparison_hdd
    if percentage < 0:
        if colder:
            return base + " The weather was colder, so this reduction is a meaningful improvement."
        return base + " Milder weather explains part of the reduction."
    if unexplained_meaningful:
        return base + " Usage stayed high after accounting for weather, so this month deserves a closer look."
    if weather_explains_most(weather_explained, absolute_change_kwh, colder):
        return base + " The increase is broadly explained by colder weather."
    if colder and weather_explained is not None and weather_explained > 0:
        return base + " Colder weather explains part of the increase; the rest is within this home's normal variation."
    return base + " The increase is not weather-related, but it is within this home's normal month-to-month variation."


@dataclass(frozen=True)
class FuelContributions:
    """How much of the combined-energy change each fuel accounts for, in kWh and % of the
    total change. Only built when electricity + gas cover the same months as the total."""

    total_change_kwh: float
    electricity_change_kwh: float
    gas_change_kwh: float
    electricity_share_pct: float | None  # None when the total change is ~zero
    gas_share_pct: float | None
    dominant_fuel: EnergyType | None


def fuel_contributions(
    total: MonthlyComparison | None,
    electricity: MonthlyComparison | None,
    gas: MonthlyComparison | None,
) -> FuelContributions | None:
    """Split the combined change into per-fuel contributions -- pure arithmetic over
    already-built comparisons; returns None unless all three carry a change."""
    if total is None or electricity is None or gas is None:
        return None
    if (
        total.absolute_change_kwh is None
        or electricity.absolute_change_kwh is None
        or gas.absolute_change_kwh is None
    ):
        return None
    total_change = total.absolute_change_kwh
    elec_change = electricity.absolute_change_kwh
    gas_change = gas.absolute_change_kwh
    if abs(total_change) < 1e-9:
        elec_share = gas_share = None
        dominant = None
    else:
        elec_share = elec_change / total_change * 100
        gas_share = gas_change / total_change * 100
        dominant = "gas" if abs(gas_change) >= abs(elec_change) else "electricity"
    return FuelContributions(
        total_change_kwh=total_change,
        electricity_change_kwh=elec_change,
        gas_change_kwh=gas_change,
        electricity_share_pct=elec_share,
        gas_share_pct=gas_share,
        dominant_fuel=dominant,
    )
