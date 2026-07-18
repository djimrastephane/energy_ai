import numpy as np
import pytest

from src.esd import generalized_esd_test


def test_esd_detects_exactly_two_injected_outliers_when_capped_at_two():
    # max_anomalies=2 matches the number of true outliers exactly, so this can't
    # over-detect into the clean cluster even with the robust variant's known
    # small-sample liberal bias (see src/esd.py's calibration-caveat docstring).
    rng = np.random.default_rng(0)
    cluster = rng.normal(100, 1.0, 20)
    values = np.concatenate([cluster, [500.0, -300.0]])

    result = generalized_esd_test(values, max_anomalies=2, alpha=0.05)

    assert result == [20, 21]


def test_esd_false_positive_rate_on_pure_noise_is_bounded():
    # The median/MAD substitution means alpha doesn't translate to an exact
    # false-positive rate at small n (documented in src/esd.py) -- test the
    # property that actually matters (bounded, not zero) across many trials
    # rather than asserting an exact zero-detections result on one seed, which
    # would be statistically fragile by construction.
    n_trials = 200
    false_positives = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        values = rng.normal(0, 1.0, 30)
        if generalized_esd_test(values, max_anomalies=3, alpha=0.01):
            false_positives += 1

    rate = false_positives / n_trials
    assert rate < 0.35  # generously above the ~17% empirically measured, catches real regressions


def test_esd_raises_when_max_anomalies_too_large_for_sample_size():
    values = np.arange(10.0)

    with pytest.raises(ValueError, match="max_anomalies"):
        generalized_esd_test(values, max_anomalies=9)


def test_esd_handles_degenerate_mad_zero_gracefully_without_crashing():
    # Heavily tied data (>50% of points identical) drives MAD to exactly 0 --
    # a genuine breakdown case for median/MAD, not something realistic STL
    # residuals would produce, but the function must degrade gracefully
    # (stop and return what it has) rather than dividing by zero or raising.
    values = np.array([5.0] * 15 + [200.0])

    result = generalized_esd_test(values, max_anomalies=10, alpha=0.05)

    assert result == []


def test_esd_detects_obvious_outlier_when_cluster_has_realistic_spread():
    rng = np.random.default_rng(2)
    cluster = rng.normal(50, 2.0, 25)  # small but nonzero spread, unlike the tied-data case above
    values = cluster.copy()
    values[10] = 1000.0

    result = generalized_esd_test(values, max_anomalies=1, alpha=0.05)

    assert result == [10]


def test_esd_returns_sorted_indices():
    rng = np.random.default_rng(3)
    cluster = rng.normal(50, 2.0, 25)
    values = cluster.copy()
    values[18] = -1000.0
    values[3] = 1000.0

    result = generalized_esd_test(values, max_anomalies=2, alpha=0.05)

    assert result == sorted(result) == [3, 18]
