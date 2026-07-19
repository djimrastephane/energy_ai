import json

import numpy as np
import pandas as pd
import pytest
import requests

from src.weather import (
    DAILY_WEATHER_SCHEMA,
    WeatherFetchError,
    compute_monthly_degree_days,
    fetch_daily_weather,
    merge_weather_with_consumption,
    validate_daily_weather,
)

_TZ = "Europe/London"
_TZ_SLUG = "Europe-London"


class _FakeResponse:
    def __init__(self, payload, ok: bool = True, invalid_json: bool = False):
        self._payload = payload
        self._ok = ok
        self._invalid_json = invalid_json

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("simulated failure")

    def json(self):
        if self._invalid_json:
            raise json.JSONDecodeError("bad", "doc", 0)
        return self._payload


def _fake_payload(dates: list[str], temps: list[float], **overrides) -> dict:
    """Full six-field payload; pass e.g. snowfall_sum=[...] to override, or =None to omit."""
    n = len(dates)
    daily = {
        "time": dates,
        "temperature_2m_mean": temps,
        "snowfall_sum": [0.0] * n,
        "snow_depth_mean": [0.0] * n,
        "precipitation_sum": [1.0] * n,
        "wind_speed_10m_max": [20.0] * n,
        "wind_gusts_10m_max": [35.0] * n,
    }
    for field, values in overrides.items():
        if values is None:
            daily.pop(field, None)
        else:
            daily[field] = values
    return {"daily": daily}


def _cache_frame(dates, temps=None) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "temp_mean_c": temps if temps is not None else np.linspace(0, 10, n),
            "snowfall_cm": [0.0] * n,
            "snow_depth_m": [0.0] * n,
            "precipitation_mm": [1.0] * n,
            "wind_speed_max_kmh": [20.0] * n,
            "wind_gust_max_kmh": [35.0] * n,
        }
    )


def _v2_cache_path(tmp_path, lat, lon):
    return tmp_path / f"weather_daily_v2_{lat:.4f}_{lon:.4f}_{_TZ_SLUG}.csv"


# --- fetch_daily_weather (network mocked) ------------------------------------


def test_fetch_uses_cache_without_network_call_when_fully_covered(tmp_path, monkeypatch):
    lat, lon = 57.1436, -2.0981
    _cache_frame(pd.date_range("2024-01-01", periods=31, freq="D")).to_csv(
        _v2_cache_path(tmp_path, lat, lon), index=False
    )

    def _boom(*args, **kwargs):
        raise AssertionError("network should not be called on a full cache hit")

    monkeypatch.setattr("src.weather.requests.get", _boom)

    result = fetch_daily_weather(lat, lon, "2024-01-05", "2024-01-10", _TZ, tmp_path)

    assert len(result) == 6
    assert result["date"].min() == pd.Timestamp("2024-01-05")
    assert set(DAILY_WEATHER_SCHEMA).issubset(result.columns)


def test_fetch_ignores_old_v1_temperature_only_cache(tmp_path, monkeypatch):
    """A v1 (unversioned, temperature-only) cache must not satisfy a v2 request -- the
    schema differs. v2 fetches fresh once and writes its own file; v1 stays untouched."""
    lat, lon = 57.14, -2.10
    v1_path = tmp_path / f"weather_daily_{lat:.4f}_{lon:.4f}.csv"
    v1_original = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5, freq="D"), "temp_mean_c": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    v1_original.to_csv(v1_path, index=False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
    calls = {"n": 0}

    def _fake_get(*a, **kw):
        calls["n"] += 1
        return _FakeResponse(_fake_payload(dates, [9.0] * 5))

    monkeypatch.setattr("src.weather.requests.get", _fake_get)

    result = fetch_daily_weather(lat, lon, "2024-01-01", "2024-01-05", _TZ, tmp_path)

    assert calls["n"] == 1  # network was used despite the v1 file covering the range
    assert result["temp_mean_c"].tolist() == [9.0] * 5  # v2 data, not the v1 values
    assert _v2_cache_path(tmp_path, lat, lon).exists()
    pd.testing.assert_frame_equal(pd.read_csv(v1_path, parse_dates=["date"]), v1_original)  # untouched


def test_fetch_writes_v2_cache_then_hits_it_without_network(tmp_path, monkeypatch):
    lat, lon = 57.14, -2.10
    dates = pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
    monkeypatch.setattr(
        "src.weather.requests.get", lambda *a, **kw: _FakeResponse(_fake_payload(dates, [1.0, 2.0, 3.0, 4.0, 5.0]))
    )
    first = fetch_daily_weather(lat, lon, "2024-01-01", "2024-01-05", _TZ, tmp_path)
    assert len(first) == 5

    def _boom(*args, **kwargs):
        raise AssertionError("second call must be a cache hit")

    monkeypatch.setattr("src.weather.requests.get", _boom)
    second = fetch_daily_weather(lat, lon, "2024-01-01", "2024-01-05", _TZ, tmp_path)
    pd.testing.assert_frame_equal(first, second)


def test_fetch_falls_back_to_partial_cache_on_network_failure(tmp_path, monkeypatch):
    lat, lon = 57.14, -2.10
    _cache_frame(pd.date_range("2024-01-01", periods=5, freq="D"), temps=[1.0, 2.0, 3.0, 4.0, 5.0]).to_csv(
        _v2_cache_path(tmp_path, lat, lon), index=False
    )

    def _fail(*args, **kwargs):
        raise requests.ConnectionError("simulated network outage")

    monkeypatch.setattr("src.weather.requests.get", _fail)

    # Requested range extends beyond the cache, so a fetch is attempted, fails, and falls back.
    result = fetch_daily_weather(lat, lon, "2024-01-01", "2024-01-10", _TZ, tmp_path)

    assert len(result) == 5  # only the cached days are returned


def test_fetch_raises_when_network_fails_and_no_cache(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise requests.ConnectionError("simulated network outage")

    monkeypatch.setattr("src.weather.requests.get", _fail)

    with pytest.raises(WeatherFetchError):
        fetch_daily_weather(57.14, -2.10, "2024-01-01", "2024-01-05", _TZ, tmp_path)


def test_fetch_raises_on_timeout_with_no_cache(tmp_path, monkeypatch):
    def _timeout(*args, **kwargs):
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr("src.weather.requests.get", _timeout)

    with pytest.raises(WeatherFetchError, match="timeout"):
        fetch_daily_weather(57.14, -2.10, "2024-01-01", "2024-01-05", _TZ, tmp_path)


def test_fetch_raises_on_http_error_with_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.weather.requests.get", lambda *a, **kw: _FakeResponse({}, ok=False))

    with pytest.raises(WeatherFetchError):
        fetch_daily_weather(57.14, -2.10, "2024-01-01", "2024-01-05", _TZ, tmp_path)


# --- response parsing and validation -----------------------------------------


def test_parse_full_extended_response():
    dates = ["2024-01-01", "2024-01-02"]
    payload = _fake_payload(dates, [2.0, 3.0], snowfall_sum=[1.5, 0.0], wind_gusts_10m_max=[90.7, 40.0])

    df = _parse(payload)

    assert list(df.columns) == ["date", *DAILY_WEATHER_SCHEMA]
    assert df["snowfall_cm"].tolist() == [1.5, 0.0]
    assert df["wind_gust_max_kmh"].tolist() == [90.7, 40.0]


def _parse(payload, **kwargs):
    from src.weather import _parse_daily_response

    return _parse_daily_response(_FakeResponse(payload, **kwargs))


def test_parse_invalid_json_raises_weather_error():
    with pytest.raises(WeatherFetchError, match="invalid JSON"):
        _parse({}, invalid_json=True)


def test_parse_missing_daily_block_raises():
    with pytest.raises(WeatherFetchError, match="daily"):
        _parse({"latitude": 57.1})


def test_parse_api_error_payload_raises_with_reason():
    with pytest.raises(WeatherFetchError, match="out of range"):
        _parse({"error": True, "reason": "start_date out of range"})


def test_parse_missing_required_temperature_raises():
    payload = _fake_payload(["2024-01-01"], [1.0], temperature_2m_mean=None)
    with pytest.raises(WeatherFetchError, match="temperature_2m_mean"):
        _parse(payload)


def test_parse_missing_optional_severe_field_degrades_to_nan_not_zero():
    payload = _fake_payload(["2024-01-01", "2024-01-02"], [1.0, 2.0], snowfall_sum=None)

    df = _parse(payload)

    assert df["snowfall_cm"].isna().all()  # NaN = "not available", never silently 0
    assert df["temp_mean_c"].tolist() == [1.0, 2.0]


def test_parse_unequal_array_lengths_raises():
    payload = _fake_payload(["2024-01-01", "2024-01-02"], [1.0, 2.0], precipitation_sum=[5.0])
    with pytest.raises(WeatherFetchError, match="2 dates"):
        _parse(payload)


def test_parse_null_daily_values_become_nan():
    payload = _fake_payload(["2024-01-01", "2024-01-02"], [1.0, None], wind_speed_10m_max=[None, 30.0])

    df = _parse(payload)

    assert np.isnan(df["temp_mean_c"].iloc[1])
    assert np.isnan(df["wind_speed_max_kmh"].iloc[0])


def test_validate_nulls_out_implausible_values_with_warnings():
    df = _cache_frame(pd.date_range("2024-01-01", periods=3, freq="D"))
    df.loc[0, "snowfall_cm"] = -1.0
    df.loc[1, "wind_gust_max_kmh"] = 999.0
    df.loc[2, "snow_depth_m"] = 50.0

    warnings = validate_daily_weather(df)

    assert len(warnings) == 3
    assert np.isnan(df.loc[0, "snowfall_cm"])
    assert np.isnan(df.loc[1, "wind_gust_max_kmh"])
    assert np.isnan(df.loc[2, "snow_depth_m"])


def test_validate_raises_on_duplicate_dates():
    df = _cache_frame(["2024-01-01", "2024-01-01"])
    with pytest.raises(WeatherFetchError, match="duplicate"):
        validate_daily_weather(df)


# --- degree days and merge ---------------------------------------------------


def test_compute_monthly_degree_days_matches_hand_calc():
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "temp_mean_c": [10.0, 20.0, 25.0],
        }
    )

    result = compute_monthly_degree_days(daily_df, base_heat_c=15.5, base_cool_c=22.0)

    # HDD: max(0,15.5-10)=5.5, max(0,15.5-20)=0, max(0,15.5-25)=0 -> sum 5.5
    # CDD: max(0,10-22)=0, max(0,20-22)=0, max(0,25-22)=3 -> sum 3.0
    row = result.iloc[0]
    assert row["hdd"] == pytest.approx(5.5)
    assert row["cdd"] == pytest.approx(3.0)
    assert row["avg_temp_c"] == pytest.approx((10 + 20 + 25) / 3)


def test_compute_monthly_degree_days_excludes_null_temp_days_from_coverage():
    """A day with a null temperature contributes no degree days, so it must not count toward
    n_days either -- otherwise the month would pass the completeness check while silently
    under-counting HDD."""
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "temp_mean_c": [10.0, np.nan, 25.0],
        }
    )

    result = compute_monthly_degree_days(daily_df, base_heat_c=15.5, base_cool_c=22.0)

    assert result.iloc[0]["n_days"] == 2


def test_merge_weather_with_consumption_adds_daily_normalized_columns():
    monthly_df = pd.DataFrame(
        {"month_start": [pd.Timestamp("2024-02-01")], "days_in_month": [29]}
    )
    degree_days_df = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2024-02-01")],
            "hdd": [290.0],
            "cdd": [0.0],
            "avg_temp_c": [5.0],
            "n_days": [29],
        }
    )

    merged = merge_weather_with_consumption(monthly_df, degree_days_df)

    assert merged.loc[0, "avg_daily_hdd"] == pytest.approx(10.0)
    assert merged.loc[0, "avg_daily_cdd"] == pytest.approx(0.0)


def test_merge_weather_with_consumption_drops_incomplete_months():
    monthly_df = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")],
            "days_in_month": [29, 31],
        }
    )
    degree_days_df = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")],
            "hdd": [290.0, 100.0],
            "cdd": [0.0, 0.0],
            "avg_temp_c": [5.0, 8.0],
            "n_days": [29, 18],  # March only has 18 of 31 days covered
        }
    )

    merged = merge_weather_with_consumption(monthly_df, degree_days_df)

    assert list(merged["month_start"]) == [pd.Timestamp("2024-02-01")]
