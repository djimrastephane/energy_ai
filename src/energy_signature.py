"""Fit and interpret the monthly "energy signature" degree-day regression.

The degree-day regression is the standard method utility analysts use on
monthly billing data to separate a fixed weather-independent base load from
heating/cooling-driven consumption. Operates on the merged
consumption+weather frame produced by
:func:`src.weather.merge_weather_with_consumption`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EnergySignatureResult:
    """OLS fit of ``avg_daily_kwh ~ avg_daily_hdd + avg_daily_cdd``."""

    intercept: float
    intercept_se: float
    heating_slope: float
    heating_se: float
    heating_pvalue: float
    cooling_slope: float
    cooling_se: float
    cooling_pvalue: float
    r_squared: float
    adj_r_squared: float
    durbin_watson: float
    n_obs: int
    fitted: pd.Series
    resid: pd.Series


def fit_energy_signature(merged_df: pd.DataFrame) -> EnergySignatureResult:
    """Fit the degree-day regression.

    Raises:
        ValueError: if fewer than 6 matched months are available -- below
            that, a 3-parameter regression (intercept + 2 slopes) has too
            few residual degrees of freedom to be meaningful.
    """
    if len(merged_df) < 6:
        raise ValueError(
            "Need at least 6 months of matched consumption+weather data for a meaningful "
            f"3-parameter regression; got {len(merged_df)}."
        )

    # A feature with (near) zero variance -- e.g. cooling degree days in a climate that
    # never crosses the cooling threshold -- is not estimable and is dropped from the
    # design matrix rather than left in, where it would produce a spurious NaN/singular
    # coefficient. Its slope is reported as exactly 0 with an explicit NaN p-value
    # (see `interpret_energy_signature`, which distinguishes "not estimable" from
    # "estimated but not statistically significant").
    features = ["avg_daily_hdd", "avg_daily_cdd"]
    usable = [f for f in features if merged_df[f].std(ddof=1) > 1e-9]
    dropped = [f for f in features if f not in usable]
    if dropped:
        logger.info("No variation in %s across this data; excluding from the fit", dropped)

    x = sm.add_constant(merged_df[usable])
    y = merged_df["avg_daily_kwh"]
    model = sm.OLS(y, x).fit()

    def _coef(name: str) -> tuple[float, float, float]:
        if name not in usable:
            return 0.0, float("nan"), float("nan")
        return float(model.params[name]), float(model.bse[name]), float(model.pvalues[name])

    heating_slope, heating_se, heating_pvalue = _coef("avg_daily_hdd")
    cooling_slope, cooling_se, cooling_pvalue = _coef("avg_daily_cdd")

    fitted = pd.Series(model.fittedvalues.to_numpy(), index=merged_df["month_start"], name="fitted")
    resid = pd.Series(model.resid.to_numpy(), index=merged_df["month_start"], name="resid")

    return EnergySignatureResult(
        intercept=float(model.params["const"]),
        intercept_se=float(model.bse["const"]),
        heating_slope=heating_slope,
        heating_se=heating_se,
        heating_pvalue=heating_pvalue,
        cooling_slope=cooling_slope,
        cooling_se=cooling_se,
        cooling_pvalue=cooling_pvalue,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        durbin_watson=float(durbin_watson(model.resid)),
        n_obs=int(model.nobs),
        fitted=fitted,
        resid=resid,
    )


def _heating_not_significant_reason(fuel: str) -> str:
    if fuel == "electricity":
        return (
            "consumption doesn't move detectably with colder weather, consistent with heating "
            "coming from gas or another non-electric source."
        )
    if fuel == "gas":
        return "consumption doesn't move detectably with colder weather in this gas data."
    return "consumption doesn't move detectably with colder weather in this combined (electricity + gas) view."


def _heating_significant_addendum(fuel: str) -> str:
    if fuel == "electricity":
        return " This is consistent with some electric heating or heating-linked electrical load in this home."
    if fuel == "gas":
        return " This is consistent with gas heating in this home."
    return (
        " This view combines electricity and gas -- switch the sidebar's Fuel selector to "
        "Electricity or Gas only to see which fuel is actually driving it."
    )


def _describe_sensitivity(
    kind: str,
    slope: float,
    pvalue: float,
    unit_rate_gbp_per_kwh: float,
    not_significant_reason: str,
) -> str:
    if np.isnan(pvalue):
        return (
            f"{kind.capitalize()} degree days showed no variation at all across this data, so "
            f"a {kind} sensitivity couldn't be estimated -- expected if every month's average "
            f"temperature stayed on the same side of the {kind} threshold."
        )
    if pvalue < 0.05:
        cost = slope * unit_rate_gbp_per_kwh
        return (
            f"Each additional {kind}-degree-day is associated with an extra {slope:.3f} kWh/day "
            f"(about £{cost:.3f}/day), a statistically significant effect (p={pvalue:.3f})."
        )
    return (
        f"No statistically significant {kind} sensitivity was detected (p={pvalue:.2f}) -- "
        f"{not_significant_reason}"
    )


def interpret_energy_signature(
    result: EnergySignatureResult, unit_rate_gbp_per_kwh: float, fuel: str = "total"
) -> str:
    """Plain-English narrative of the fitted energy signature.

    Significance threshold used throughout: p < 0.05. A feature with no
    variation in the data (e.g. cooling degree days in a climate that never
    crosses the cooling threshold) is reported as "couldn't be estimated",
    distinct from "estimated but not significant". ``fuel`` is one of
    ``"total"``/``"electricity"``/``"gas"`` (see ``src.ingestion.FUEL_FILE_PATTERNS``)
    -- it only changes which fuel the heating-attribution sentence names, since
    this same regression now runs against any of the three via the sidebar's
    Fuel selector, and "electric heating" would be a wrong claim when ``result``
    was fitted on gas-only data.
    """
    base_cost = result.intercept * unit_rate_gbp_per_kwh
    parts = [
        f"Across {result.n_obs} months, the estimated weather-independent base load is "
        f"{result.intercept:.2f} kWh/day (about £{base_cost:.2f}/day) -- consumption that "
        "doesn't track heating or cooling degree days.",
        _describe_sensitivity(
            "heating",
            result.heating_slope,
            result.heating_pvalue,
            unit_rate_gbp_per_kwh,
            _heating_not_significant_reason(fuel),
        )
        + (_heating_significant_addendum(fuel) if result.heating_pvalue < 0.05 else ""),
        _describe_sensitivity(
            "cooling",
            result.cooling_slope,
            result.cooling_pvalue,
            unit_rate_gbp_per_kwh,
            "unsurprising in a climate where cooling degree days are rare.",
        ),
    ]

    parts.append(
        f"This weather-only model explains {result.r_squared:.0%} of month-to-month variation "
        f"in daily consumption (R-squared = {result.r_squared:.2f}). Durbin-Watson = "
        f"{result.durbin_watson:.2f} (2.0 = no autocorrelation); values well below 2 mean "
        "nearby months' residuals are correlated, so the p-values above should be read as "
        "indicative rather than exact."
    )

    return " ".join(parts)


def partial_dependence(
    result: EnergySignatureResult, merged_df: pd.DataFrame, feature: str
) -> pd.DataFrame:
    """Predicted consumption across the observed range of ``feature``, other feature held at its mean.

    For a linear model this is exactly the fitted line -- computed
    explicitly (not via a black-box PDP routine) so it stays traceable to
    the fitted coefficients.
    """
    if feature not in ("avg_daily_hdd", "avg_daily_cdd"):
        raise ValueError(f"feature must be 'avg_daily_hdd' or 'avg_daily_cdd', got {feature!r}")

    other = "avg_daily_cdd" if feature == "avg_daily_hdd" else "avg_daily_hdd"
    slope = result.heating_slope if feature == "avg_daily_hdd" else result.cooling_slope
    other_slope = result.cooling_slope if feature == "avg_daily_hdd" else result.heating_slope
    other_mean = merged_df[other].mean()

    grid = np.linspace(merged_df[feature].min(), merged_df[feature].max(), 20)
    predicted = result.intercept + slope * grid + other_slope * other_mean
    return pd.DataFrame({feature: grid, "predicted_avg_daily_kwh": predicted})


def _interpret_annual_difference(difference_kwh: float, resid_std_kwh: float, n_months: int) -> str:
    if resid_std_kwh <= 0 or np.isnan(resid_std_kwh):
        return "Not enough residual variability to judge whether this difference is meaningful."
    # Same one-sided 95% normal-approximation convention as src.recommendations' heating-review
    # threshold: summing n independent months' residuals scales their combined std by sqrt(n).
    threshold = resid_std_kwh * (n_months**0.5) * 1.645
    if abs(difference_kwh) <= threshold:
        return "Consumption this year is well explained by weather."
    if difference_kwh > 0:
        return "Consumption exceeds what weather alone would predict, suggesting a behavioural change."
    return "Consumption is lower than weather would predict, suggesting reduced usage or improved efficiency."


def annual_weather_adjusted_comparison(merged_df: pd.DataFrame, result: EnergySignatureResult) -> pd.DataFrame:
    """Per complete calendar year: actual vs. weather-predicted consumption, and the gap between them.

    Answers "why did my bill change: weather or behaviour?" directly from
    the already-fitted regression -- no second normalization model.
    ``weather_predicted_kwh`` is what the degree-day model expects given
    that year's *actual* weather (not a long-run-average weather year); a
    near-zero difference means weather already explains the actual total
    well, a large positive difference means usage exceeded what weather
    alone would predict (a behavioural signal), and a large negative
    difference means usage was lower than weather would predict.

    Only years with all 12 months actually present in ``merged_df`` are
    included (computed from ``merged_df`` itself, not a completeness flag
    inherited from before the weather merge -- a month can be dropped here
    even in an otherwise-complete billing year if its weather coverage was
    incomplete; see ``src.weather.merge_weather_with_consumption``).
    """
    df = merged_df.copy()
    df["year"] = df["month_start"].dt.year
    fitted_avg_daily = result.fitted.reindex(pd.DatetimeIndex(df["month_start"])).to_numpy()
    df["fitted_kwh"] = fitted_avg_daily * df["days_in_month"].to_numpy()

    months_per_year = df.groupby("year")["month_start"].transform("size")
    complete = df[months_per_year == 12]
    if complete.empty:
        return pd.DataFrame(
            columns=["year", "actual_kwh", "weather_predicted_kwh", "difference_kwh", "interpretation"]
        )

    resid_std_kwh = (result.resid * merged_df.set_index("month_start")["days_in_month"]).std()

    rows = []
    for year, group in complete.groupby("year"):
        actual = float(group["consumption_kwh"].sum())
        predicted = float(group["fitted_kwh"].sum())
        difference = actual - predicted
        rows.append(
            {
                "year": int(year),
                "actual_kwh": actual,
                "weather_predicted_kwh": predicted,
                "difference_kwh": difference,
                "interpretation": _interpret_annual_difference(difference, resid_std_kwh, len(group)),
            }
        )
    return pd.DataFrame(rows)
