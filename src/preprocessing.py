"""Clean, validate, and enrich the raw ingested OVO data.

Never assumes the data is clean: duplicates are removed and reported,
conflicting values are resolved deterministically and reported, implausible
values are flagged (not silently dropped), and missing months are detected
and reported. See :func:`run_pipeline` for the single entry point that ties
these steps together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.utils import get_logger, safe_divide

logger = get_logger(__name__)


@dataclass
class PreprocessingReport:
    """Summary of data-quality findings, surfaced verbatim in the Data Quality tab."""

    n_files_loaded: int
    source_files: list[str]
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    n_months: int
    missing_months: list[pd.Timestamp]
    duplicates_removed: int
    conflicts: list[str] = field(default_factory=list)
    outlier_warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.missing_months or self.conflicts or self.outlier_warnings)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """Remove duplicate rows and resolve same-month conflicts.

    Exact duplicate rows (identical month, cost, and consumption -- e.g. the
    same export uploaded twice) are dropped silently and counted. If the
    same month appears with *different* values across files, the value from
    the most-recently-loaded file wins (later entries in ``df`` take
    precedence) and the conflict is reported rather than averaged away.

    Returns:
        A tuple of (deduplicated frame, count of exact duplicates removed,
        list of human-readable conflict descriptions).
    """
    if df.empty:
        return df, 0, []

    exact_dup_mask = df.duplicated(
        subset=["month_start", "cost_gbp", "consumption_kwh"], keep="first"
    )
    n_exact = int(exact_dup_mask.sum())
    df = df[~exact_dup_mask].copy()

    conflicts: list[str] = []
    conflicting_months = df.loc[df.duplicated(subset=["month_start"], keep=False), "month_start"]
    for month in sorted(conflicting_months.unique()):
        rows = df[df["month_start"] == month]
        kept = rows.iloc[-1]
        conflicts.append(
            f"{month.strftime('%B %Y')}: conflicting values across files "
            f"{rows['source_file'].tolist()} -- kept {kept['cost_gbp']:.2f} GBP / "
            f"{kept['consumption_kwh']:.2f} kWh from '{kept['source_file']}'"
        )

    df = df.drop_duplicates(subset=["month_start"], keep="last").reset_index(drop=True)

    if n_exact:
        logger.info("Removed %d exact duplicate row(s)", n_exact)
    if conflicts:
        logger.warning("Resolved %d conflicting month(s): %s", len(conflicts), conflicts)

    return df, n_exact, conflicts


def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by month and assert every month is unique.

    Raises:
        ValueError: if a duplicate month survives (indicates a bug upstream
            in :func:`deduplicate`, not a normal data-quality condition).
    """
    df = df.sort_values("month_start").reset_index(drop=True)
    n_dupes = int(df["month_start"].duplicated().sum())
    if n_dupes:
        raise ValueError(
            f"{n_dupes} duplicate month(s) remain after deduplication -- "
            "this indicates a bug, not a data-quality issue"
        )
    return df


def validate_units(
    df: pd.DataFrame, thresholds: object | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Flag (never drop) negative or implausible cost/consumption/unit-rate values."""
    from config import SETTINGS

    thresholds = thresholds or SETTINGS.validation
    warnings: list[str] = []

    for _, row in df.iterrows():
        label = row["month_start"].strftime("%B %Y")
        kwh, cost = row["consumption_kwh"], row["cost_gbp"]

        if kwh < thresholds.min_monthly_kwh:
            warnings.append(f"{label}: negative consumption ({kwh:.2f} kWh)")
        elif kwh > thresholds.max_monthly_kwh:
            warnings.append(
                f"{label}: unusually high consumption ({kwh:.0f} kWh) exceeds "
                f"plausibility threshold of {thresholds.max_monthly_kwh:.0f} kWh"
            )

        if cost < thresholds.min_monthly_cost_gbp:
            warnings.append(f"{label}: negative cost (£{cost:.2f})")
        elif cost > thresholds.max_monthly_cost_gbp:
            warnings.append(
                f"{label}: unusually high cost (£{cost:.2f}) exceeds "
                f"plausibility threshold of £{thresholds.max_monthly_cost_gbp:.2f}"
            )

        unit_rate = safe_divide(cost, kwh)
        if kwh > 0 and unit_rate > thresholds.max_plausible_unit_rate_gbp_per_kwh:
            warnings.append(f"{label}: implausible unit rate (£{unit_rate:.2f}/kWh)")

    if warnings:
        logger.warning("%d unit-plausibility warning(s): %s", len(warnings), warnings)
    return df, warnings


def flag_in_progress_month(df: pd.DataFrame, today: pd.Timestamp | None = None) -> list[str]:
    """Warn when the latest month's *billing period* has not fully elapsed.

    OVO exports made mid-period contain a period-to-date figure (verified on
    real data: July 2026 showed 107.87 kWh on July 18 vs. 150.59 kWh for the
    full July 2025), but every downstream analysis treats each row as a
    complete month -- ``avg_daily_kwh`` divides by the full ``days_in_month``,
    trailing-window KPIs sum it as a whole month, and anomaly detection
    scores it against complete months. The billing cycle runs 6th-to-5th
    (``config.BillingConfig``), so the month labelled June is still
    accumulating until 5 July inclusive -- ``src.billing`` owns that
    completeness rule. A flag, not a drop, since the value is real, just
    possibly still accumulating.

    ``today`` is injectable for tests; defaults to the current date.
    """
    from src.billing import billing_period, is_billing_month_complete

    if df.empty:
        return []
    latest = df["month_start"].max()
    if not is_billing_month_complete(latest, today):
        start, end, _ = billing_period(latest)
        return [
            f"{latest.strftime('%B %Y')}'s billing period ({start.strftime('%d %b')} - "
            f"{end.strftime('%d %b %Y')}) has not finished -- its figures may be "
            "period-to-date rather than a full month, and every analysis treats it as complete."
        ]
    return []


def detect_missing_months(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return every calendar month between the earliest and latest present month that is absent."""
    if df.empty:
        return []
    full_range = pd.date_range(df["month_start"].min(), df["month_start"].max(), freq="MS")
    present = set(df["month_start"])
    missing = [m for m in full_range if m not in present]
    if missing:
        logger.warning(
            "%d missing month(s) detected: %s",
            len(missing),
            [m.strftime("%Y-%m") for m in missing],
        )
    return missing


def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Attach derived columns needed by downstream analysis.

    Adds: ``year``, ``month_num``, ``month_name``, ``days_in_month``,
    ``avg_daily_kwh``, ``avg_daily_cost_gbp``, ``unit_rate_gbp_per_kwh``, and
    ``is_partial_year`` (True for any calendar year with fewer than 12
    months present in the data -- e.g. the first and current years in a
    typical export set).
    """
    df = df.sort_values("month_start").reset_index(drop=True).copy()
    if df.empty:
        for col in (
            "year",
            "month_num",
            "month_name",
            "days_in_month",
            "avg_daily_kwh",
            "avg_daily_cost_gbp",
            "unit_rate_gbp_per_kwh",
            "is_partial_year",
        ):
            df[col] = pd.Series(dtype="float64")
        return df

    df["year"] = df["month_start"].dt.year
    df["month_num"] = df["month_start"].dt.month
    df["month_name"] = df["month_start"].dt.strftime("%B")
    df["days_in_month"] = df["month_start"].dt.days_in_month
    df["avg_daily_kwh"] = df["consumption_kwh"] / df["days_in_month"]
    df["avg_daily_cost_gbp"] = df["cost_gbp"] / df["days_in_month"]
    df["unit_rate_gbp_per_kwh"] = df.apply(
        lambda r: safe_divide(r["cost_gbp"], r["consumption_kwh"]), axis=1
    )

    months_per_year = df.groupby("year")["month_num"].transform("nunique")
    df["is_partial_year"] = months_per_year < 12

    return df


def run_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Run the full clean -> validate -> enrich pipeline and produce a data-quality report.

    This is the single entry point the Streamlit app and integration tests
    should call after :func:`src.ingestion.load_all`.
    """
    source_files = sorted(raw_df["source_file"].unique().tolist()) if not raw_df.empty else []

    df, n_exact_dupes, conflicts = deduplicate(raw_df)
    df = validate_timestamps(df)
    df, outlier_warnings = validate_units(df)
    outlier_warnings.extend(flag_in_progress_month(df))
    missing_months = detect_missing_months(df)
    df = build_monthly_series(df)

    date_range = (df["month_start"].min(), df["month_start"].max()) if not df.empty else None

    report = PreprocessingReport(
        n_files_loaded=len(source_files),
        source_files=source_files,
        date_range=date_range,
        n_months=len(df),
        missing_months=missing_months,
        duplicates_removed=n_exact_dupes,
        conflicts=conflicts,
        outlier_warnings=outlier_warnings,
    )
    return df, report
