import numpy as np
import pandas as pd
import pytest

from config import SETTINGS
from src.decomposition import interpret_decomposition, stl_decompose
from src.ingestion import discover_csv_files, load_all
from src.preprocessing import run_pipeline


def _synthetic_monthly_df(n_months: int, seasonal_amplitude: float, trend_per_month: float):
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    month_num = months.month
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (month_num - 1) / 12)
    trend = trend_per_month * np.arange(n_months)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, n_months)
    values = 300 + trend + seasonal + noise
    return pd.DataFrame({"month_start": months, "consumption_kwh": values})


def test_stl_decompose_detects_strong_seasonality():
    df = _synthetic_monthly_df(36, seasonal_amplitude=100.0, trend_per_month=0.0)

    result = stl_decompose(df)

    assert result.seasonal_strength > 0.9
    assert len(result.observed) == 36
    assert len(result.deseasonalized) == 36


def test_stl_decompose_detects_strong_trend():
    df = _synthetic_monthly_df(36, seasonal_amplitude=1.0, trend_per_month=10.0)

    result = stl_decompose(df)

    assert result.trend_strength > 0.9


def test_stl_decompose_raises_on_gap():
    df = _synthetic_monthly_df(30, seasonal_amplitude=50.0, trend_per_month=0.0)
    df = df.drop(df.index[10])  # remove one month, creating a gap

    with pytest.raises(ValueError, match="contiguous"):
        stl_decompose(df)


def test_stl_decompose_raises_on_insufficient_history():
    df = _synthetic_monthly_df(10, seasonal_amplitude=50.0, trend_per_month=0.0)

    with pytest.raises(ValueError, match="at least 13"):
        stl_decompose(df)


def test_stl_decompose_runs_below_two_full_periods():
    df = _synthetic_monthly_df(18, seasonal_amplitude=50.0, trend_per_month=0.0)

    result = stl_decompose(df)

    assert len(result.observed) == 18


def test_interpret_decomposition_mentions_short_history_caveat():
    df = _synthetic_monthly_df(18, seasonal_amplitude=50.0, trend_per_month=0.0)
    result = stl_decompose(df)

    text = interpret_decomposition(result)

    assert "18 months" in text
    assert "extra caution" in text


def test_interpret_decomposition_omits_caveat_with_enough_history():
    df = _synthetic_monthly_df(36, seasonal_amplitude=50.0, trend_per_month=0.0)
    result = stl_decompose(df)

    text = interpret_decomposition(result)

    assert "extra caution" not in text


def test_stl_decompose_runs_on_synthetic_data_without_exceptions():
    files = discover_csv_files(SETTINGS.synthetic_data_dir)
    raw = load_all(files)
    clean, _ = run_pipeline(raw)

    result = stl_decompose(clean)

    assert len(result.observed) == len(clean)
    assert 0.0 <= result.seasonal_strength <= 1.0
    assert 0.0 <= result.trend_strength <= 1.0
