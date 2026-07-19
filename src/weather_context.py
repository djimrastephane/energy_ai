"""Weather Context Engine: monthly snow/wind/precipitation context for unusual energy months.

Aggregates the extended daily weather from :mod:`src.weather` into monthly
context (snow days, strong-wind days, severe-weather runs, ...), rates each
month's severity against the household's own history with robust statistics,
and classifies the month with a deterministic label + confidence + facts +
limitation.

Design principle (see ``docs/weather_context.md``): these variables are
**contextual evidence only**. The energy-signature regression stays
HDD/CDD-only; nothing here feeds a model. The output distinguishes observed
weather facts from plausible interpretation, and never claims snow/wind
*caused* a consumption change -- monthly billing data cannot prove that.

All thresholds live in ``config.WeatherContextThresholds`` -- none are
hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import SETTINGS, WeatherContextThresholds
from src.confidence import Confidence
from src.utils import get_logger

logger = get_logger(__name__)

# Metrics that get percentile ranks vs. the household's own history. Percentiles are
# computed across ALL available months, not same-calendar-month peers: with ~3 years of
# history a season-matched comparison would have n≈3, too few to rank against.
_SEVERITY_METRICS = ("snowfall_total_cm", "precipitation_total_mm", "max_wind_gust_kmh")

_MIN_HISTORY_MONTHS = 12

_LIMITATION = (
    "Monthly billing data cannot prove that severe weather caused higher consumption or "
    "changed behaviour (for example home working or being kept indoors) -- these are "
    "observed weather facts, offered as context only."
)

# No composite "severe_weather_score" is computed, deliberately: any weighting of
# snow vs. wind vs. rain into one number would be opaque, and the classification
# label below already serves as the interpretable summary of the components.


def aggregate_monthly_weather_context(
    daily_df: pd.DataFrame, thresholds: WeatherContextThresholds | None = None
) -> pd.DataFrame:
    """Aggregate the daily schema frame into one row of weather context per month.

    Day counts treat ``NaN`` as "doesn't qualify" (a day with unknown snowfall is not a
    snow day), and ``weather_context_complete`` is ``False`` for any month with missing
    days or missing values in any context field -- so a count from an incomplete month is
    visibly a lower bound, never silently wrong.
    """
    t = thresholds or SETTINGS.weather_context
    w = SETTINGS.weather
    df = daily_df.copy()
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["hdd"] = (w.base_heat_c - df["temp_mean_c"]).clip(lower=0)
    df["cdd"] = (df["temp_mean_c"] - w.base_cool_c).clip(lower=0)

    context_cols = ["temp_mean_c", "snowfall_cm", "snow_depth_m", "precipitation_mm", "wind_speed_max_kmh", "wind_gust_max_kmh"]
    df["is_snow_day"] = df["snowfall_cm"] > t.snow_day_cm
    df["is_heavy_snow_day"] = df["snowfall_cm"] >= t.heavy_snow_day_cm
    df["is_heavy_rain_day"] = df["precipitation_mm"] >= t.heavy_rain_day_mm
    df["is_strong_wind_day"] = df["wind_speed_max_kmh"] >= t.strong_wind_day_kmh
    df["is_severe_gust_day"] = df["wind_gust_max_kmh"] >= t.severe_gust_day_kmh
    df["is_severe_day"] = (
        df["is_heavy_snow_day"] | df["is_heavy_rain_day"] | df["is_strong_wind_day"] | df["is_severe_gust_day"]
    )

    def _longest_run(severe: pd.Series) -> int:
        longest = current = 0
        for flag in severe:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        return longest

    rows = []
    for month_start, month_df in df.sort_values("date").groupby("month_start"):
        days_in_month = month_start.days_in_month
        n_days_present = len(month_df)
        has_missing_values = bool(month_df[context_cols].isna().any().any())
        rows.append(
            {
                "month_start": month_start,
                "n_weather_days": n_days_present,
                "avg_temp_c": month_df["temp_mean_c"].mean(),
                "hdd": month_df["hdd"].sum(),
                "cdd": month_df["cdd"].sum(),
                "snowfall_total_cm": month_df["snowfall_cm"].sum(),
                "snow_days": int(month_df["is_snow_day"].sum()),
                "heavy_snow_days": int(month_df["is_heavy_snow_day"].sum()),
                "max_snow_depth_m": month_df["snow_depth_m"].max(),
                "precipitation_total_mm": month_df["precipitation_mm"].sum(),
                "heavy_rain_days": int(month_df["is_heavy_rain_day"].sum()),
                "strong_wind_days": int(month_df["is_strong_wind_day"].sum()),
                "severe_gust_days": int(month_df["is_severe_gust_day"].sum()),
                "max_wind_speed_kmh": month_df["wind_speed_max_kmh"].max(),
                "max_wind_gust_kmh": month_df["wind_gust_max_kmh"].max(),
                "longest_severe_weather_run_days": _longest_run(month_df["is_severe_day"]),
                "weather_context_complete": n_days_present == days_in_month and not has_missing_values,
            }
        )
    return pd.DataFrame(rows)


def severity_band(value: float, history: pd.Series) -> str:
    """Robust relative severity of one month's value against the household's history.

    Bands by *strict* percentile rank -- the share of months strictly below this value:
    under 0.75 is "Typical", 0.75-0.90 "Above usual", 0.90+ "Unusually high". Strict
    (not <=) matters: in a history where every month is essentially identical, every
    month would otherwise rank at the 100th percentile of itself and read as unusual
    (caught by this module's own tests via 31- vs 30-day months' precipitation totals).
    Ties therefore rank conservatively low. "Extreme" additionally requires the value to
    exceed median + 3x IQR -- and is never awarded when the IQR is zero (e.g. snowfall,
    where most months are 0), because a robust spread can't be established from a
    zero-variation history. No statistical significance is claimed at this sample size.
    """
    valid = history.dropna()
    if len(valid) < _MIN_HISTORY_MONTHS or pd.isna(value):
        return "Insufficient history"
    percentile = float((valid < value).mean())
    if percentile < 0.75:
        return "Typical"
    if percentile < 0.90:
        return "Above usual"
    q1, median, q3 = valid.quantile(0.25), valid.median(), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0 and value >= median + 3 * iqr:
        return "Extreme"
    return "Unusually high"


def add_relative_severity(context_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<metric>_percentile`` and ``<metric>_band`` for each severity metric.

    Percentile ranks and bands compare each month against every month in ``context_df``
    (the household's own available history) -- see ``_SEVERITY_METRICS`` for why not
    season-matched peers.
    """
    df = context_df.copy()
    for metric in _SEVERITY_METRICS:
        valid = df[metric].dropna()
        if len(valid) < _MIN_HISTORY_MONTHS:
            df[f"{metric}_percentile"] = float("nan")
            df[f"{metric}_band"] = "Insufficient history"
            continue
        # Same strict-rank definition as severity_band, so the percentile column and the
        # band column can never disagree about what "unusually high" means.
        df[f"{metric}_percentile"] = df[metric].apply(lambda v, s=df[metric]: float((s.dropna() < v).mean()))
        df[f"{metric}_band"] = df[metric].apply(lambda v, s=df[metric]: severity_band(v, s))
    return df


@dataclass
class WeatherContextClassification:
    month_start: pd.Timestamp
    label: str
    confidence: Confidence
    facts: list[str]
    limitation: str

    @property
    def is_notable(self) -> bool:
        return self.label not in (
            "No notable severe weather",
            "Incomplete weather coverage",
            "Insufficient history",
        )


def _month_facts(row: pd.Series, t: WeatherContextThresholds) -> list[str]:
    """Observed facts only -- numbers with their thresholds, no interpretation."""
    facts = []
    if row["snow_days"]:
        facts.append(f"{row['snow_days']} snow day(s), {row['snowfall_total_cm']:.0f} cm total snowfall")
    if row["heavy_snow_days"]:
        facts.append(f"{row['heavy_snow_days']} heavy-snow day(s) (>= {t.heavy_snow_day_cm:.0f} cm/day)")
    if not pd.isna(row["max_snow_depth_m"]) and row["max_snow_depth_m"] > 0:
        facts.append(f"Maximum snow depth {row['max_snow_depth_m'] * 100:.0f} cm")
    if row["strong_wind_days"]:
        facts.append(f"{row['strong_wind_days']} strong-wind day(s) (>= {t.strong_wind_day_kmh:.0f} km/h sustained)")
    if row["severe_gust_days"]:
        facts.append(f"{row['severe_gust_days']} severe-gust day(s) (>= {t.severe_gust_day_kmh:.0f} km/h)")
    if not pd.isna(row["max_wind_gust_kmh"]):
        facts.append(f"Maximum gust {row['max_wind_gust_kmh']:.0f} km/h")
    if row["heavy_rain_days"]:
        facts.append(f"{row['heavy_rain_days']} heavy-rain day(s) (>= {t.heavy_rain_day_mm:.0f} mm/day)")
    facts.append(f"Total precipitation {row['precipitation_total_mm']:.0f} mm")
    if row["longest_severe_weather_run_days"] >= 2:
        facts.append(f"Longest severe-weather run {row['longest_severe_weather_run_days']} consecutive day(s)")
    return facts


def classify_monthly_weather_context(
    context_df: pd.DataFrame, month_start: pd.Timestamp, thresholds: WeatherContextThresholds | None = None
) -> WeatherContextClassification:
    """Deterministic severity label for one month, judged against the household's history.

    ``context_df`` should be the full-history output of
    :func:`aggregate_monthly_weather_context` (severity columns from
    :func:`add_relative_severity` are used when present).
    """
    t = thresholds or SETTINGS.weather_context
    matches = context_df[context_df["month_start"] == month_start]
    if matches.empty:
        return WeatherContextClassification(
            month_start, "Incomplete weather coverage", "Low",
            ["No weather data is available for this month."], _LIMITATION,
        )
    row = matches.iloc[0]
    facts = _month_facts(row, t)

    if not row["weather_context_complete"]:
        facts.insert(0, f"Only {row['n_weather_days']} day(s) of complete weather data for this month.")
        return WeatherContextClassification(
            month_start, "Incomplete weather coverage", "Low", facts, _LIMITATION
        )
    if len(context_df) < _MIN_HISTORY_MONTHS:
        return WeatherContextClassification(
            month_start, "Insufficient history", "Low", facts,
            _LIMITATION + f" Under {_MIN_HISTORY_MONTHS} months of weather history, this month also "
            "can't be compared against what's usual for this household.",
        )

    snowy = row["snow_days"] >= 1
    prolonged_snow = row["snow_days"] >= 5 or row["heavy_snow_days"] >= 2
    windy = row["strong_wind_days"] >= 3 or row["severe_gust_days"] >= 2
    rain_band = row.get("precipitation_total_mm_band", "Typical")
    wet = row["heavy_rain_days"] >= 2 or rain_band in ("Unusually high", "Extreme")

    if (snowy and windy) or (snowy and wet) or (prolonged_snow and (windy or wet)):
        label = "Mixed severe weather"
    elif prolonged_snow:
        label = "Prolonged snow"
    elif snowy:
        label = "Snowy period"
    elif windy and wet:
        label = "Wet and windy period"
    elif windy:
        label = "Strong-wind period"
    elif wet:
        label = "Wet period"
    else:
        label = "No notable severe weather"

    confidence: Confidence = "High" if len(context_df) >= 24 else "Medium"
    return WeatherContextClassification(month_start, label, confidence, facts, _LIMITATION)


def cooling_variation_negligible(merged_df: pd.DataFrame) -> bool:
    """True when this household's cooling-degree-day variation is too small to interpret.

    Used by the UI to keep cooling charts/wording out of the main view for homes (like
    this Aberdeen household) where warm-weather effects are effectively absent -- CDD
    stays computed and available in advanced views. Criterion: total CDD under 1% of
    total HDD (or zero HDD with zero CDD).
    """
    cdd_total = float(merged_df["cdd"].sum())
    hdd_total = float(merged_df["hdd"].sum())
    if cdd_total == 0:
        return True
    return hdd_total > 0 and cdd_total / hdd_total < 0.01
