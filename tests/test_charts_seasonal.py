import numpy as np
import pandas as pd

from app.charts_seasonal import seasonal_calendar_profile_figure, seasonal_overview_figure
from src.anomalies import Anomaly
from src.decomposition import stl_decompose
from src.seasonal_summary import seasonal_profile


def _clean_df(n=36, start="2022-01-01", amplitude=80.0, trend_per_month=0.0, spike_at=None, spike_size=0.0):
    months = pd.date_range(start, periods=n, freq="MS")
    month_num = np.asarray(months.month)
    seasonal = amplitude * np.cos(2 * np.pi * (month_num - 1) / 12)
    trend = trend_per_month * np.arange(n)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, n)
    values = np.asarray(300 + trend + seasonal + noise, dtype=float)
    if spike_at is not None:
        values[spike_at] += spike_size
    df = pd.DataFrame({"month_start": months, "consumption_kwh": values})
    df["year"] = df["month_start"].dt.year
    months_per_year = df.groupby("year")["month_start"].transform("size")
    df["is_partial_year"] = months_per_year < 12
    return df


def test_overview_figure_builds_without_anomalies():
    df = _clean_df()
    result = stl_decompose(df)

    fig = seasonal_overview_figure(df, result, anomalies=[])

    trace_types = [type(t).__name__ for t in fig.data]
    assert "Bar" in trace_types
    assert "Scatter" in trace_types  # the trend line
    assert fig.layout.xaxis.title.text == "Month"
    assert fig.layout.yaxis.title.text == "kWh"


def test_overview_figure_marks_spike_and_drop_anomalies():
    df = _clean_df(spike_at=23, spike_size=400.0)
    result = stl_decompose(df)
    date = df["month_start"].iloc[23]
    anomalies = [Anomaly(date=date, methods=["stl_esd"], direction="spike", rank_context="1st highest of 36 months")]

    fig = seasonal_overview_figure(df, result, anomalies)

    marker_traces = [t for t in fig.data if getattr(t, "mode", "") and "markers" in t.mode]
    assert len(marker_traces) == 1
    assert marker_traces[0].marker.symbol == "triangle-up"
    assert marker_traces[0].text[0] == date.strftime("%b %Y")


def test_overview_figure_distinguishes_partial_year_bars_by_color():
    df = _clean_df(n=15, start="2023-10-01")  # first and last calendar years are partial
    result = stl_decompose(df)

    fig = seasonal_overview_figure(df, result, anomalies=[])

    bar_trace = next(t for t in fig.data if type(t).__name__ == "Bar")
    colors = set(bar_trace.marker.color)
    assert len(colors) == 2  # at least one partial-year and one full-year color present


def test_overview_figure_handles_empty_anomaly_list_and_short_history():
    df = _clean_df(n=13)
    result = stl_decompose(df)

    fig = seasonal_overview_figure(df, result, anomalies=[])

    assert len(fig.data) == 2  # bars + trend line only, no marker traces


def test_calendar_profile_figure_signs_bars_correctly():
    df = _clean_df(amplitude=100.0)
    result = stl_decompose(df)
    profile = seasonal_profile(result)

    fig = seasonal_calendar_profile_figure(profile)

    bar = fig.data[0]
    assert list(bar.x) == list(profile["month_name"])
    colors = list(bar.marker.color)
    patterns = list(bar.marker.pattern.shape)
    for value, color, pattern in zip(profile["typical_effect_kwh"], colors, patterns, strict=True):
        if value >= 0:
            assert pattern == ""
        else:
            assert pattern == "/"
        assert color in ("#eda100", "#e87ba4")
    assert "vs. typical month" in fig.layout.yaxis.title.text
