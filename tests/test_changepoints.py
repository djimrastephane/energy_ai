import numpy as np
import pandas as pd

from src.changepoints import (
    detect_changepoints,
    detect_changepoints_cusum,
    detect_changepoints_pelt,
)


def _mean_shift_series(n_before=12, n_after=12, shift=10.0, noise_std=0.5, seed=0):
    rng = np.random.default_rng(seed)
    values = np.concatenate(
        [
            rng.normal(0, noise_std, n_before),
            rng.normal(shift, noise_std, n_after),
        ]
    )
    index = pd.date_range("2020-01-01", periods=n_before + n_after, freq="MS")
    return pd.Series(values, index=index)


def test_pelt_detects_obvious_mean_shift():
    series = _mean_shift_series()
    dates = detect_changepoints_pelt(series)

    assert len(dates) >= 1
    expected = series.index[12]
    assert min(abs(d - expected) for d in dates) <= pd.Timedelta(days=93)


def test_cusum_detects_obvious_mean_shift():
    series = _mean_shift_series()
    dates = detect_changepoints_cusum(series)

    assert len(dates) >= 1
    expected = series.index[12]
    assert min(abs(d - expected) for d in dates) <= pd.Timedelta(days=93)


def test_flat_series_has_no_changepoints():
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-01", periods=24, freq="MS")
    series = pd.Series(rng.normal(0, 0.3, 24), index=index)

    assert detect_changepoints_pelt(series) == []
    assert detect_changepoints_cusum(series) == []


def test_detect_changepoints_merges_both_methods():
    series = _mean_shift_series()
    result = detect_changepoints(series)

    assert len(result) >= 1
    closest = min(result, key=lambda cp: abs(cp.date - series.index[12]))
    assert closest.method in ("pelt", "cusum", "both")
    assert closest.direction == "increase"
    assert closest.magnitude_kwh > 0


def test_detect_changepoints_empty_for_constant_series():
    index = pd.date_range("2020-01-01", periods=24, freq="MS")
    series = pd.Series([100.0] * 24, index=index)

    assert detect_changepoints(series) == []
