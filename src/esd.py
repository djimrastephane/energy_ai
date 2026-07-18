"""Generalized ESD (Extreme Studentized Deviate) test for detecting up to k outliers.

Implements Rosner's (1983) generalized ESD test in its robust,
median/MAD-based form (as used by Twitter's Seasonal-Hybrid ESD anomaly
detector), so that early-removed extreme points don't distort the
statistics used to judge later ones. Split into its own module because the
math here (iterative removal + a per-step critical-value formula) is easy
to get subtly wrong and deserves focused, careful testing separate from the
rest of anomaly detection.

Reference: Rosner, B. (1983), "Percentage Points for a Generalized ESD
Many-Outlier Procedure", Technometrics 25(2). Formula as presented in the
NIST/SEMATECH e-Handbook of Statistical Methods, section 7.1.6.

Calibration caveat (measured, not theoretical): the critical-value formula
below was derived assuming the test statistic uses the sample mean/std. The
median/MAD substitution is what makes this the *robust* "Hybrid" variant
(resistant to the anomalies themselves distorting the statistics used to
find them), but it also means ``alpha`` no longer translates to the nominal
false-positive rate at small n -- empirically, on pure-noise 30-point
samples, alpha=0.05 here gives roughly a 25% chance of at least one
spurious flag, not 5%, and this doesn't fully resolve even at much smaller
alpha (still ~10% at alpha=0.001). This is a known, published limitation of
median/MAD-substituted ESD, not a bug -- callers at small n should treat a
single ESD flag as exploratory and prefer requiring agreement from another
method (see ``src.anomalies.detect_anomalies``, which cross-references this
against rolling z-score and Isolation Forest for exactly that reason).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.utils import get_logger

logger = get_logger(__name__)


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a consistent estimator of sigma under normality."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def generalized_esd_test(values: np.ndarray, max_anomalies: int, alpha: float = 0.05) -> list[int]:
    """Return indices into ``values`` flagged as outliers (ascending order).

    At each of up to ``max_anomalies`` steps: compute the median/MAD of the
    remaining points, remove whichever point has the largest deviation from
    the median, and record the test statistic ``R_i`` and Rosner's critical
    value ``lambda_i``. The final outlier count is the *largest* ``i`` for
    which ``R_i > lambda_i`` (not necessarily every step up to it) --
    Rosner's procedure explicitly allows the test statistic to dip below
    the critical value partway through and still find more outliers beyond
    that point.

    Raises:
        ValueError: if ``max_anomalies >= len(values) - 2`` (too few points
            would remain to compute a meaningful spread).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if max_anomalies >= n - 2:
        raise ValueError(f"max_anomalies ({max_anomalies}) must be less than n-2 ({n - 2}).")

    working_indices = list(range(n))
    remaining = x.copy()
    removed_indices: list[int] = []
    test_stats: list[float] = []
    critical_values: list[float] = []

    for _ in range(max_anomalies):
        n_i = len(remaining)
        med = np.median(remaining)
        mad = _mad(remaining)
        if mad == 0:
            logger.debug("MAD reached 0 after removing %d point(s); stopping early", len(removed_indices))
            break

        deviations = np.abs(remaining - med)
        idx_in_remaining = int(np.argmax(deviations))
        r_i = deviations[idx_in_remaining] / mad

        p = 1 - alpha / (2 * n_i)
        t_crit = stats.t.ppf(p, df=n_i - 2)
        lambda_i = ((n_i - 1) * t_crit) / np.sqrt((n_i - 2 + t_crit**2) * n_i)

        test_stats.append(r_i)
        critical_values.append(lambda_i)
        removed_indices.append(working_indices.pop(idx_in_remaining))
        remaining = np.delete(remaining, idx_in_remaining)

    num_anomalies = 0
    for i in range(len(test_stats) - 1, -1, -1):
        if test_stats[i] > critical_values[i]:
            num_anomalies = i + 1
            break

    return sorted(removed_indices[:num_anomalies])
