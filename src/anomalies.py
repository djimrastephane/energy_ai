"""Anomaly detection: rolling z-score, STL-residual ESD, Isolation Forest.

The first two run on the STL residual, not raw consumption -- same
reasoning as ``src.changepoints``: a normal winter shouldn't read as
anomalous once trend and seasonality are accounted for, only what's left
over (the residual) should. Isolation Forest is multivariate by design and
uses a small feature set that includes that same residual, per the spec's
framing of it as a secondary/exploratory check.

Every method here is imperfect at n around 35 (see ``src.esd``'s
calibration-caveat docstring for the ESD method specifically);
:func:`detect_anomalies` cross-references all three so a date flagged by
2+ methods reads as meaningfully more confident than one flagged by a
single method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.esd import generalized_esd_test
from src.utils import get_logger

logger = get_logger(__name__)


def detect_rolling_zscore(series: pd.Series, window: int = 6, threshold: float = 2.5) -> list[pd.Timestamp]:
    """Flag points more than ``threshold`` standard deviations from the *preceding* window's mean/std.

    The baseline window is shifted to exclude the point being tested --
    including a point in its own baseline lets a genuine spike inflate the
    very std it's being judged against, which can mask it entirely (a real
    self-contamination bug caught by this module's own tests: an injected
    spike went undetected until the window was shifted). Using each
    baseline's own local mean/std (rather than the whole series) also means
    a level shift partway through the data doesn't make everything before
    or after it look anomalous by comparison. The first ``window`` points
    have no full preceding window yet and are never flagged.
    """
    if len(series) < window + 1:
        return []
    baseline_mean = series.shift(1).rolling(window=window, min_periods=window).mean()
    baseline_std = series.shift(1).rolling(window=window, min_periods=window).std(ddof=1)
    z = (series - baseline_mean) / baseline_std
    flagged = z[(z.abs() > threshold) & baseline_std.gt(0)]
    return list(flagged.index)


def detect_stl_esd(
    resid: pd.Series, alpha: float = 0.01, max_anomalies_frac: float = 0.1
) -> list[pd.Timestamp]:
    """Generalized ESD test (``src.esd``) on the STL residual.

    ``alpha=0.01``, not the textbook-default 0.05, is deliberately
    conservative -- see ``src.esd``'s calibration-caveat docstring for why
    the median/MAD-substituted test needs a tighter alpha at this sample
    size to keep its false-positive rate reasonable.
    """
    max_anomalies = max(1, int(len(resid) * max_anomalies_frac))
    if max_anomalies >= len(resid) - 2:
        return []
    indices = generalized_esd_test(resid.to_numpy(), max_anomalies=max_anomalies, alpha=alpha)
    return [resid.index[i] for i in indices]


def detect_isolation_forest(
    feature_df: pd.DataFrame, contamination: float = 0.1, seed: int = 42
) -> list[pd.Timestamp]:
    """Multivariate anomaly check via sklearn's IsolationForest.

    ``feature_df`` must already be numeric, indexed by ``month_start``.
    Standardized internally since IsolationForest is scale-sensitive --
    unscaled kWh vs. £/kWh columns would otherwise let the largest-magnitude
    column dominate the isolation splits.
    """
    if len(feature_df) < 10:  # not enough points for a meaningful forest
        return []
    standardized = (feature_df - feature_df.mean()) / feature_df.std(ddof=1).replace(0, 1)
    model = IsolationForest(contamination=contamination, random_state=seed)
    labels = model.fit_predict(standardized.to_numpy())
    return list(feature_df.index[labels == -1])


@dataclass
class Anomaly:
    date: pd.Timestamp
    methods: list[str]
    direction: Literal["spike", "drop"]
    rank_context: str


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rank_context(date: pd.Timestamp, values: pd.Series) -> str:
    """Grounded description of how extreme this month's residual is -- never a fabricated cause."""
    n = len(values)
    rank_desc = int(values.rank(ascending=False, method="min").loc[date])
    rank_asc = int(values.rank(ascending=True, method="min").loc[date])
    if rank_desc <= rank_asc:
        return f"{_ordinal(rank_desc)} highest of {n} months (by how far it sits from the trend/season fit)"
    return f"{_ordinal(rank_asc)} lowest of {n} months (by how far it sits from the trend/season fit)"


def detect_anomalies(clean_df: pd.DataFrame, stl_result) -> list[Anomaly]:
    """Run all three methods, cross-reference by date, and rank by how many methods agree."""
    resid = stl_result.resid

    zscore_dates = set(detect_rolling_zscore(resid))
    esd_dates = set(detect_stl_esd(resid))

    feature_df = pd.DataFrame(
        {
            "consumption_kwh": clean_df["consumption_kwh"].to_numpy(),
            "cost_gbp": clean_df["cost_gbp"].to_numpy(),
            "unit_rate_gbp_per_kwh": clean_df["unit_rate_gbp_per_kwh"].to_numpy(),
            "stl_resid": resid.reindex(clean_df["month_start"]).to_numpy(),
        },
        index=pd.DatetimeIndex(clean_df["month_start"]),
    )
    iso_dates = set(detect_isolation_forest(feature_df))

    anomalies = []
    for date in sorted(zscore_dates | esd_dates | iso_dates):
        methods = [
            name
            for name, dates in (
                ("rolling_zscore", zscore_dates),
                ("stl_esd", esd_dates),
                ("isolation_forest", iso_dates),
            )
            if date in dates
        ]
        resid_value = resid.loc[date]
        anomalies.append(
            Anomaly(
                date=date,
                methods=methods,
                direction="spike" if resid_value >= 0 else "drop",
                rank_context=_rank_context(date, resid),
            )
        )

    anomalies.sort(key=lambda a: (-len(a.methods), a.date))
    return anomalies


def interpret_anomalies(anomalies: list[Anomaly]) -> str:
    """Plain-English summary; never speculates about a specific cause beyond what's in the data."""
    if not anomalies:
        return (
            "No months were flagged as unusual by any method once trend and seasonality are "
            "accounted for."
        )

    high_confidence = [a for a in anomalies if len(a.methods) >= 2]
    parts = [
        f"{len(anomalies)} month(s) flagged by at least one method; {len(high_confidence)} "
        "flagged by 2 or more methods (higher confidence)."
    ]
    for a in anomalies[:5]:
        parts.append(
            f"{a.date.strftime('%B %Y')}: {a.direction} ({', '.join(a.methods)}) -- {a.rank_context}."
        )
    if len(anomalies) > 5:
        parts.append(f"...and {len(anomalies) - 5} more below.")
    return " ".join(parts)
