import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from src.statistics import describe


def test_describe_matches_numpy_scipy_reference():
    data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    arr = data.to_numpy()

    summary = describe(data)

    assert summary.n == 10
    assert summary.mean == pytest.approx(np.mean(arr))
    assert summary.median == pytest.approx(np.median(arr))
    assert summary.std == pytest.approx(np.std(arr, ddof=1))
    assert summary.variance == pytest.approx(np.var(arr, ddof=1))
    assert summary.skewness == pytest.approx(scipy_stats.skew(arr))
    assert summary.kurtosis == pytest.approx(scipy_stats.kurtosis(arr))
    assert summary.iqr == pytest.approx(summary.p75 - summary.p25)


def test_describe_bootstrap_ci_brackets_sample_mean():
    rng = np.random.default_rng(0)
    data = pd.Series(rng.normal(loc=500, scale=50, size=40))

    summary = describe(data, n_boot=2000, seed=1)

    assert summary.bootstrap_ci95_lower < summary.mean < summary.bootstrap_ci95_upper


def test_describe_raises_on_empty_series():
    with pytest.raises(ValueError):
        describe(pd.Series([], dtype=float))


def test_describe_handles_single_observation_gracefully():
    summary = describe(pd.Series([42.0]))
    assert summary.n == 1
    assert summary.mean == 42.0
    assert np.isnan(summary.std)
