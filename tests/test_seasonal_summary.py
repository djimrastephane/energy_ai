import numpy as np
import pandas as pd

from src.anomalies import Anomaly, detect_anomalies
from src.decomposition import stl_decompose
from src.seasonal_summary import (
    seasonal_profile,
    summarize_seasonal_pattern,
    trend_direction,
)


def _synthetic_monthly_df(
    n_months: int,
    seasonal_amplitude: float = 0.0,
    trend_per_month: float = 0.0,
    start: str = "2020-01-01",
    spike_at: int | None = None,
    spike_size: float = 0.0,
    seed: int = 0,
):
    months = pd.date_range(start, periods=n_months, freq="MS")
    month_num = months.month
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (month_num - 1) / 12)
    trend = trend_per_month * np.arange(n_months)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, n_months)
    values = np.asarray(300 + trend + seasonal + noise, dtype=float)
    if spike_at is not None:
        values[spike_at] += spike_size
    return pd.DataFrame({"month_start": months, "consumption_kwh": values})


def _anomaly(date: pd.Timestamp, direction="spike", n_methods=2) -> Anomaly:
    methods = ["rolling_zscore", "stl_esd", "isolation_forest"][:n_methods]
    return Anomaly(date=date, methods=methods, direction=direction, rank_context="1st highest of 36 months")


def test_strong_seasonality_stable_trend_conclusion():
    df = _synthetic_monthly_df(36, seasonal_amplitude=100.0, trend_per_month=0.0)
    result = stl_decompose(df)

    summary = summarize_seasonal_pattern(result, anomalies=[])

    assert summary.seasonal_band == "Strong"
    assert summary.trend_label == "Stable"
    assert "winter and summer cycle" in summary.main_conclusion
    assert "broadly stable" in summary.main_conclusion
    assert summary.unusual_month is None
    assert "No month departed" in summary.unusual_month_context


def test_weak_seasonality_band():
    # No true seasonal signal (amplitude 0) and enough months that STL doesn't overfit noise
    # into a spurious seasonal shape -- verified empirically at n=48, seed=2 (strength 0.04).
    df = _synthetic_monthly_df(48, seasonal_amplitude=0.0, trend_per_month=0.0, seed=2)
    result = stl_decompose(df)

    summary = summarize_seasonal_pattern(result, anomalies=[])

    assert summary.seasonal_band == "Low"
    assert "not strongly tied to the time of year" in summary.main_conclusion


def test_rising_trend_detected():
    df = _synthetic_monthly_df(36, seasonal_amplitude=20.0, trend_per_month=5.0)
    result = stl_decompose(df)

    label, pct = trend_direction(result)

    assert label == "Rising"
    assert pct > 0
    summary = summarize_seasonal_pattern(result, anomalies=[])
    assert "rise" in summary.main_conclusion
    assert "something other than weather is changing" in summary.why_it_matters


def test_falling_trend_detected():
    df = _synthetic_monthly_df(36, seasonal_amplitude=20.0, trend_per_month=-5.0)
    result = stl_decompose(df)

    label, pct = trend_direction(result)

    assert label == "Falling"
    assert pct < 0
    summary = summarize_seasonal_pattern(result, anomalies=[])
    assert "fall" in summary.main_conclusion


def test_stable_trend_within_threshold_not_flagged_as_rising():
    # A tiny drift (well under 5%/year) must not be called "Rising".
    df = _synthetic_monthly_df(36, seasonal_amplitude=20.0, trend_per_month=0.05)
    result = stl_decompose(df)

    label, _ = trend_direction(result)

    assert label == "Stable"


def test_seasonal_profile_has_positive_and_negative_months():
    # cos() peaks at month 1 (positive) and troughs at month 7 (negative) with this phasing.
    df = _synthetic_monthly_df(36, seasonal_amplitude=100.0, trend_per_month=0.0)
    result = stl_decompose(df)

    profile = seasonal_profile(result)

    assert list(profile["month_name"]) == [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]  # fmt: skip
    assert len(profile) == 12
    jan = profile.loc[profile["month_name"] == "January", "typical_effect_kwh"].iloc[0]
    jul = profile.loc[profile["month_name"] == "July", "typical_effect_kwh"].iloc[0]
    assert jan > 0
    assert jul < 0


def test_material_residual_anomaly_surfaced_as_unusual_month():
    df = _synthetic_monthly_df(36, seasonal_amplitude=50.0, trend_per_month=0.0, spike_at=23, spike_size=400.0)
    result = stl_decompose(df)
    anomalies = detect_anomalies(
        df.assign(cost_gbp=df["consumption_kwh"] * 0.3, unit_rate_gbp_per_kwh=0.3), result
    )

    summary = summarize_seasonal_pattern(result, anomalies)

    assert summary.unusual_month is not None
    assert summary.unusual_month.date == df["month_start"].iloc[23]
    assert summary.unusual_month_kwh is not None
    assert summary.unusual_month_kwh > 0
    assert "materially higher" in summary.main_conclusion


def test_unusual_month_picks_largest_magnitude_not_just_first_flagged():
    date_a = pd.Timestamp("2021-06-01")
    date_b = pd.Timestamp("2021-09-01")
    df = _synthetic_monthly_df(36, seasonal_amplitude=50.0, trend_per_month=0.0)
    result = stl_decompose(df)
    # Manually construct residuals so date_b is clearly the larger deviation.
    result.resid.loc[date_a] = 10.0
    result.resid.loc[date_b] = 90.0

    anomalies = [_anomaly(date_a), _anomaly(date_b)]
    summary = summarize_seasonal_pattern(result, anomalies)

    assert summary.unusual_month.date == date_b


def test_confidence_medium_below_24_months_high_at_or_above():
    short = stl_decompose(_synthetic_monthly_df(13, seasonal_amplitude=20.0))
    long = stl_decompose(_synthetic_monthly_df(24, seasonal_amplitude=20.0))

    assert summarize_seasonal_pattern(short, []).confidence == "Medium"
    assert summarize_seasonal_pattern(long, []).confidence == "High"


def test_partial_year_data_does_not_crash_summary():
    # 3 months of a partial first year, followed by full years -- still contiguous for STL.
    df = _synthetic_monthly_df(27, seasonal_amplitude=40.0, trend_per_month=1.0, start="2022-10-01")
    result = stl_decompose(df)

    summary = summarize_seasonal_pattern(result, anomalies=[])
    profile = seasonal_profile(result)

    assert summary.seasonal_band in ("Low", "Moderate", "Strong")
    assert len(profile) == 12


def test_minimal_valid_history_thirteen_months():
    df = _synthetic_monthly_df(13, seasonal_amplitude=30.0, trend_per_month=0.0)
    result = stl_decompose(df)

    summary = summarize_seasonal_pattern(result, anomalies=[])

    assert summary.confidence == "Medium"
    assert "months of history" in summary.confidence_reason
