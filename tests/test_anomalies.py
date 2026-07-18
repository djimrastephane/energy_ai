from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.anomalies import (
    detect_anomalies,
    detect_isolation_forest,
    detect_rolling_zscore,
    detect_stl_esd,
    interpret_anomalies,
)


def _stable_series_with_spike(n=24, spike_idx=15, spike_value=500.0, base=100.0, noise_std=1.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="MS")
    values = base + rng.normal(0, noise_std, n)
    values[spike_idx] = spike_value
    return pd.Series(values, index=idx)


def test_detect_rolling_zscore_flags_injected_spike_only():
    series = _stable_series_with_spike()

    flagged = detect_rolling_zscore(series, window=6, threshold=2.5)

    assert series.index[15] in flagged
    assert len(flagged) <= 2  # the spike itself, plus possibly one neighbouring window effect


def test_detect_rolling_zscore_returns_empty_for_short_series():
    series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2022-01-01", periods=3, freq="MS"))

    assert detect_rolling_zscore(series, window=6) == []


def test_detect_stl_esd_wrapper_maps_indices_to_dates():
    series = _stable_series_with_spike(n=30, spike_idx=10, noise_std=2.0)

    flagged = detect_stl_esd(series, alpha=0.01, max_anomalies_frac=0.1)

    assert series.index[10] in flagged
    assert all(isinstance(d, pd.Timestamp) for d in flagged)


def test_detect_isolation_forest_flags_obvious_multivariate_outlier():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=24, freq="MS")
    df = pd.DataFrame(
        {
            "consumption_kwh": rng.normal(300, 10, 24),
            "cost_gbp": rng.normal(90, 3, 24),
            "unit_rate_gbp_per_kwh": rng.normal(0.3, 0.01, 24),
            "stl_resid": rng.normal(0, 1, 24),
        },
        index=idx,
    )
    df.iloc[12] = [1500.0, 5.0, 2.0, 300.0]  # wildly inconsistent across every feature

    flagged = detect_isolation_forest(df, contamination=0.1)

    assert idx[12] in flagged


def test_detect_isolation_forest_returns_empty_for_too_few_points():
    idx = pd.date_range("2022-01-01", periods=5, freq="MS")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

    assert detect_isolation_forest(df) == []


def test_detect_anomalies_ranks_multi_method_agreement_first():
    n = 30
    idx = pd.date_range("2022-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    consumption = 300 + rng.normal(0, 5, n)
    consumption[20] = 900.0  # obvious spike -> should trip rolling z-score, ESD, and isolation forest

    clean_df = pd.DataFrame(
        {
            "month_start": idx,
            "consumption_kwh": consumption,
            "cost_gbp": consumption * 0.3,
            "unit_rate_gbp_per_kwh": 0.3,
        }
    )
    resid = pd.Series(consumption - consumption.mean(), index=idx)
    stl_result = SimpleNamespace(resid=resid)

    anomalies = detect_anomalies(clean_df, stl_result)

    assert len(anomalies) > 0
    top = anomalies[0]
    assert top.date == idx[20]
    assert len(top.methods) >= 2
    assert top.direction == "spike"
    assert "of 30 months" in top.rank_context


def test_interpret_anomalies_handles_empty_list():
    assert "No months" in interpret_anomalies([])


def test_interpret_anomalies_mentions_high_confidence_count():
    from src.anomalies import Anomaly

    anomalies = [
        Anomaly(
            date=pd.Timestamp("2024-06-01"),
            methods=["rolling_zscore", "stl_esd"],
            direction="spike",
            rank_context="1st highest of 30 months",
        )
    ]

    text = interpret_anomalies(anomalies)

    assert "1 flagged by 2 or more methods" in text
    assert "June 2024" in text
