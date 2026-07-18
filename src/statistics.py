"""Descriptive statistics and confidence intervals for a single numeric series.

Operates on a plain ``pd.Series`` (e.g. a column pulled from the clean
monthly frame produced by ``src.preprocessing.run_pipeline``). Every number
here is computed directly from the data -- nothing is hard-coded or
estimated without a traceable source. Period-over-period KPI comparisons
live in :mod:`src.kpis`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils import get_logger, safe_divide

logger = get_logger(__name__)


@dataclass
class StatsSummary:
    """Descriptive statistics for a single numeric series."""

    n: int
    mean: float
    median: float
    std: float
    variance: float
    coefficient_of_variation: float
    iqr: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    skewness: float
    kurtosis: float
    ci95_lower: float
    ci95_upper: float
    bootstrap_ci95_lower: float
    bootstrap_ci95_upper: float


def _bootstrap_mean_ci(
    data: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for the mean, via resampling with replacement.

    Assumption: observations are treated as independent and identically
    distributed, which is a simplification for a time series with
    seasonality -- interpret this interval as a measure of sampling
    uncertainty in the mean, not a forecast interval.
    """
    n = len(data)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resamples = rng.choice(data, size=(n_boot, n), replace=True)
    boot_means = resamples.mean(axis=1)
    lower, upper = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def describe(series: pd.Series, n_boot: int = 5000, seed: int = 42) -> StatsSummary:
    """Compute a full descriptive-statistics summary for ``series``.

    Args:
        series: numeric data, e.g. monthly consumption in kWh.
        n_boot: number of bootstrap resamples for the bootstrap CI.
        seed: RNG seed, fixed by default so results are reproducible.

    Note:
        With a small number of monthly observations (typically well under
        100), skewness, kurtosis, and confidence intervals carry more
        sampling noise than they would in a large dataset -- they are still
        computed faithfully from the data, but should be read as indicative
        rather than precise.
    """
    clean = series.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n == 0:
        raise ValueError("Cannot compute statistics on an empty series")

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if n > 1 else float("nan")
    variance = float(np.var(clean, ddof=1)) if n > 1 else float("nan")
    p10, p25, p50, p75, p90 = (float(x) for x in np.percentile(clean, [10, 25, 50, 75, 90]))

    if n > 1 and std > 0:
        sem = std / np.sqrt(n)
        ci_lower, ci_upper = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    else:
        ci_lower, ci_upper = float("nan"), float("nan")

    boot_lower, boot_upper = _bootstrap_mean_ci(clean, n_boot=n_boot, seed=seed)

    return StatsSummary(
        n=n,
        mean=mean,
        median=float(np.median(clean)),
        std=std,
        variance=variance,
        coefficient_of_variation=safe_divide(std, mean, default=float("nan")),
        iqr=p75 - p25,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        skewness=float(scipy_stats.skew(clean)) if n > 2 else float("nan"),
        kurtosis=float(scipy_stats.kurtosis(clean)) if n > 3 else float("nan"),
        ci95_lower=float(ci_lower),
        ci95_upper=float(ci_upper),
        bootstrap_ci95_lower=boot_lower,
        bootstrap_ci95_upper=boot_upper,
    )


def interpret(summary: StatsSummary, label: str = "consumption") -> str:
    """Translate a ``StatsSummary`` into a short plain-English narrative.

    Thresholds used (documented here since they drive the wording):
    coefficient of variation < 15% = low variability, 15-35% = moderate,
    > 35% = high; skewness magnitude < 0.5 = roughly symmetric, >= 0.5 =
    notably skewed.
    """
    if summary.n < 8:
        return (
            f"Only {summary.n} observation(s) are available, which is too few for "
            "reliable higher-order statistics (skewness, kurtosis). Mean, median, "
            "and range below are still directly computed from the data."
        )

    cv = summary.coefficient_of_variation
    if cv < 0.15:
        variability = "low variability month-to-month"
    elif cv < 0.35:
        variability = "moderate month-to-month variability"
    else:
        variability = "high month-to-month variability"

    if abs(summary.skewness) < 0.5:
        shape = "roughly symmetric around the mean"
    elif summary.skewness >= 0.5:
        shape = "right-skewed (a handful of unusually high months pull the average up)"
    else:
        shape = "left-skewed (a handful of unusually low months pull the average down)"

    return (
        f"Across {summary.n} months, average {label} is {summary.mean:,.1f} "
        f"(median {summary.median:,.1f}), with {variability} "
        f"(coefficient of variation = {cv:.0%}). The distribution is {shape}. "
        f"The 95% confidence interval for the true mean is "
        f"[{summary.ci95_lower:,.1f}, {summary.ci95_upper:,.1f}]."
    )
