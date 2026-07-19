import numpy as np
import pandas as pd

from src.weather_context import (
    WeatherContextClassification,
    add_relative_severity,
    aggregate_monthly_weather_context,
    classify_monthly_weather_context,
    cooling_variation_negligible,
    severity_band,
)


def _daily(start: str, n_days: int, **overrides) -> pd.DataFrame:
    """Quiet default weather (5C, no snow, light rain, light wind); override per-column."""
    df = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=n_days, freq="D"),
            "temp_mean_c": 5.0,
            "snowfall_cm": 0.0,
            "snow_depth_m": 0.0,
            "precipitation_mm": 1.0,
            "wind_speed_max_kmh": 20.0,
            "wind_gust_max_kmh": 35.0,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def _full_year_plus(month_overrides: dict[str, dict] | None = None, n_months: int = 24) -> pd.DataFrame:
    """Contiguous daily history starting 2023-01-01; overrides keyed by 'YYYY-MM'."""
    frames = []
    current = pd.Timestamp("2023-01-01")
    for _ in range(n_months):
        key = current.strftime("%Y-%m")
        overrides = (month_overrides or {}).get(key, {})
        frames.append(_daily(current.strftime("%Y-%m-%d"), current.days_in_month, **overrides))
        current += pd.offsets.MonthBegin(1)
    return pd.concat(frames, ignore_index=True)


# --- aggregation --------------------------------------------------------------


def test_aggregate_no_snow_month():
    ctx = aggregate_monthly_weather_context(_daily("2024-01-01", 31))

    row = ctx.iloc[0]
    assert row["snow_days"] == 0
    assert row["heavy_snow_days"] == 0
    assert row["snowfall_total_cm"] == 0.0
    assert row["longest_severe_weather_run_days"] == 0
    assert bool(row["weather_context_complete"]) is True
    assert row["n_weather_days"] == 31


def test_aggregate_single_snow_event_and_prolonged_snow():
    snow = [0.0] * 31
    snow[10] = 3.0  # one snow day
    one_event = aggregate_monthly_weather_context(_daily("2024-01-01", 31, snowfall_cm=snow))
    assert one_event.iloc[0]["snow_days"] == 1
    assert one_event.iloc[0]["heavy_snow_days"] == 0

    prolonged = [0.0] * 31
    prolonged[5:11] = [2.0, 6.0, 8.0, 1.0, 2.0, 0.5]  # 6 snow days, 2 heavy
    result = aggregate_monthly_weather_context(_daily("2024-01-01", 31, snowfall_cm=prolonged))
    assert result.iloc[0]["snow_days"] == 6
    assert result.iloc[0]["heavy_snow_days"] == 2
    assert result.iloc[0]["snowfall_total_cm"] == sum(prolonged)


def test_aggregate_strong_wind_and_gust_days():
    wind = [20.0] * 31
    wind[0:5] = [70.0, 65.0, 63.0, 80.0, 90.0]
    gusts = [35.0] * 31
    gusts[0:5] = [95.0, 85.0, 79.9, 100.0, 120.0]
    ctx = aggregate_monthly_weather_context(_daily("2024-01-01", 31, wind_speed_max_kmh=wind, wind_gust_max_kmh=gusts))

    row = ctx.iloc[0]
    assert row["strong_wind_days"] == 5  # all >= 62
    assert row["severe_gust_days"] == 4  # 79.9 misses the 80 threshold
    assert row["max_wind_speed_kmh"] == 90.0
    assert row["max_wind_gust_kmh"] == 120.0
    assert row["longest_severe_weather_run_days"] == 5


def test_aggregate_heavy_rain_days():
    rain = [1.0] * 30
    rain[3] = 30.0
    rain[4] = 26.0
    rain[20] = 24.9  # just under the 25 mm threshold
    ctx = aggregate_monthly_weather_context(_daily("2024-06-01", 30, precipitation_mm=rain))

    assert ctx.iloc[0]["heavy_rain_days"] == 2
    assert ctx.iloc[0]["longest_severe_weather_run_days"] == 2


def test_aggregate_mixed_severe_weather_run_spans_categories():
    """A run of consecutive severe days should count even when the reason differs day to day
    (heavy snow, then a gust day, then heavy rain)."""
    snow = [0.0] * 31
    snow[10] = 6.0
    gusts = [35.0] * 31
    gusts[11] = 85.0
    rain = [1.0] * 31
    rain[12] = 30.0
    ctx = aggregate_monthly_weather_context(
        _daily("2024-01-01", 31, snowfall_cm=snow, wind_gust_max_kmh=gusts, precipitation_mm=rain)
    )

    assert ctx.iloc[0]["longest_severe_weather_run_days"] == 3


def test_aggregate_missing_values_mark_month_incomplete_and_do_not_count():
    snow = [0.0] * 31
    snow[5] = np.nan
    ctx = aggregate_monthly_weather_context(_daily("2024-01-01", 31, snowfall_cm=snow))

    row = ctx.iloc[0]
    assert bool(row["weather_context_complete"]) is False
    assert row["snow_days"] == 0  # NaN never counts as a snow day


def test_aggregate_partial_month_marked_incomplete():
    ctx = aggregate_monthly_weather_context(_daily("2024-01-01", 20))

    row = ctx.iloc[0]
    assert row["n_weather_days"] == 20
    assert bool(row["weather_context_complete"]) is False


def test_aggregate_leap_year_february_complete_at_29_days():
    ctx = aggregate_monthly_weather_context(_daily("2024-02-01", 29))
    assert bool(ctx.iloc[0]["weather_context_complete"]) is True

    non_leap = aggregate_monthly_weather_context(_daily("2023-02-01", 28))
    assert bool(non_leap.iloc[0]["weather_context_complete"]) is True


# --- relative severity --------------------------------------------------------


def test_severity_band_zero_variation_history_never_extreme():
    history = pd.Series([0.0] * 23 + [10.0])  # snowfall-like: almost always zero

    assert severity_band(10.0, history) == "Unusually high"  # never "Extreme" with IQR 0


def test_severity_band_typical_above_usual_extreme():
    rng = np.random.default_rng(0)
    history = pd.Series(np.concatenate([rng.normal(50, 10, 30), [200.0]]))

    assert severity_band(45.0, history) == "Typical"
    assert severity_band(history.quantile(0.8), history) == "Above usual"
    assert severity_band(200.0, history) == "Extreme"


def test_severity_band_insufficient_history():
    assert severity_band(5.0, pd.Series([1.0, 2.0, 3.0])) == "Insufficient history"


def test_add_relative_severity_columns_present():
    daily = _full_year_plus({"2024-01": {"snowfall_cm": [2.0] * 31}})
    ctx = add_relative_severity(aggregate_monthly_weather_context(daily))

    for metric in ("snowfall_total_cm", "precipitation_total_mm", "max_wind_gust_kmh"):
        assert f"{metric}_percentile" in ctx.columns
        assert f"{metric}_band" in ctx.columns
    jan24 = ctx[ctx["month_start"] == "2024-01-01"].iloc[0]
    assert jan24["snowfall_total_cm_band"] in ("Unusually high", "Above usual")


# --- classifier ---------------------------------------------------------------


def _classify(daily: pd.DataFrame, month: str) -> WeatherContextClassification:
    ctx = add_relative_severity(aggregate_monthly_weather_context(daily))
    return classify_monthly_weather_context(ctx, pd.Timestamp(month))


def test_classify_quiet_month():
    c = _classify(_full_year_plus(), "2023-06-01")

    assert c.label == "No notable severe weather"
    assert c.confidence == "High"  # 24 complete months of history
    assert not c.is_notable
    assert "cannot prove" in c.limitation


def test_classify_snowy_period():
    snow = [0.0] * 31
    snow[10] = 3.0
    c = _classify(_full_year_plus({"2024-01": {"snowfall_cm": snow}}), "2024-01-01")

    assert c.label == "Snowy period"
    assert any("1 snow day" in f for f in c.facts)


def test_classify_prolonged_snow():
    snow = [0.0] * 31
    snow[5:11] = [2.0, 6.0, 8.0, 1.0, 2.0, 0.5]
    c = _classify(_full_year_plus({"2024-01": {"snowfall_cm": snow}}), "2024-01-01")

    assert c.label == "Prolonged snow"
    assert any("heavy-snow" in f for f in c.facts)


def test_classify_strong_wind_period():
    wind = [20.0] * 31
    wind[0:4] = [70.0, 65.0, 63.0, 80.0]
    c = _classify(_full_year_plus({"2024-01": {"wind_speed_max_kmh": wind}}), "2024-01-01")

    assert c.label == "Strong-wind period"


def test_classify_wet_and_windy_period():
    wind = [20.0] * 31
    wind[0:4] = [70.0, 65.0, 63.0, 80.0]
    rain = [1.0] * 31
    rain[5] = 30.0
    rain[6] = 28.0
    c = _classify(_full_year_plus({"2024-01": {"wind_speed_max_kmh": wind, "precipitation_mm": rain}}), "2024-01-01")

    assert c.label == "Wet and windy period"


def test_classify_mixed_severe_weather():
    snow = [0.0] * 31
    snow[10] = 4.0
    wind = [20.0] * 31
    wind[0:4] = [70.0, 65.0, 63.0, 80.0]
    c = _classify(_full_year_plus({"2024-01": {"snowfall_cm": snow, "wind_speed_max_kmh": wind}}), "2024-01-01")

    assert c.label == "Mixed severe weather"
    assert c.is_notable


def test_classify_incomplete_coverage_month():
    daily = pd.concat([_full_year_plus(), _daily("2025-01-01", 10)], ignore_index=True)
    ctx = add_relative_severity(aggregate_monthly_weather_context(daily))
    c = classify_monthly_weather_context(ctx, pd.Timestamp("2025-01-01"))

    assert c.label == "Incomplete weather coverage"
    assert c.confidence == "Low"
    assert any("10 day(s)" in f for f in c.facts)


def test_classify_month_absent_from_context():
    ctx = add_relative_severity(aggregate_monthly_weather_context(_full_year_plus()))
    c = classify_monthly_weather_context(ctx, pd.Timestamp("2030-01-01"))

    assert c.label == "Incomplete weather coverage"


def test_classify_insufficient_history():
    snow = [0.0] * 31
    snow[3] = 4.0
    daily = _full_year_plus({"2023-01": {"snowfall_cm": snow}}, n_months=6)
    ctx = add_relative_severity(aggregate_monthly_weather_context(daily))
    c = classify_monthly_weather_context(ctx, pd.Timestamp("2023-01-01"))

    assert c.label == "Insufficient history"
    assert "can't be compared" in c.limitation


# --- cooling negligibility ----------------------------------------------------


def test_cooling_negligible_when_cdd_zero_or_tiny():
    merged = pd.DataFrame({"hdd": [300.0, 250.0], "cdd": [0.0, 0.0]})
    assert cooling_variation_negligible(merged) is True

    tiny = pd.DataFrame({"hdd": [300.0, 250.0], "cdd": [1.0, 0.5]})
    assert cooling_variation_negligible(tiny) is True


def test_cooling_not_negligible_for_warm_climate():
    merged = pd.DataFrame({"hdd": [100.0, 50.0], "cdd": [80.0, 120.0]})
    assert cooling_variation_negligible(merged) is False
