import numpy as np
import pandas as pd
import pytest
import requests

from src.weather import (
    WeatherFetchError,
    compute_monthly_degree_days,
    fetch_daily_temperature,
    merge_weather_with_consumption,
)


class _FakeResponse:
    def __init__(self, payload: dict, ok: bool = True):
        self._payload = payload
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("simulated failure")

    def json(self):
        return self._payload


def _fake_payload(dates: list[str], temps: list[float]) -> dict:
    return {"daily": {"time": dates, "temperature_2m_mean": temps}}


# --- fetch_daily_temperature (network mocked) -------------------------------


def test_fetch_uses_cache_without_network_call_when_fully_covered(tmp_path, monkeypatch):
    cache_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=31, freq="D"),
            "temp_mean_c": np.linspace(0, 10, 31),
        }
    )
    lat, lon = 57.1436, -2.0981
    cache_path = tmp_path / f"weather_daily_{lat:.4f}_{lon:.4f}.csv"
    cache_df.to_csv(cache_path, index=False)

    def _boom(*args, **kwargs):
        raise AssertionError("network should not be called on a full cache hit")

    monkeypatch.setattr("src.weather.requests.get", _boom)

    result = fetch_daily_temperature(lat, lon, "2024-01-05", "2024-01-10", "Europe/London", tmp_path)

    assert len(result) == 6
    assert result["date"].min() == pd.Timestamp("2024-01-05")


def test_fetch_calls_network_on_cache_miss_and_writes_cache(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
    temps = [1.0, 2.0, 3.0, 4.0, 5.0]

    monkeypatch.setattr(
        "src.weather.requests.get", lambda *a, **kw: _FakeResponse(_fake_payload(dates, temps))
    )

    result = fetch_daily_temperature(57.14, -2.10, "2024-01-01", "2024-01-05", "Europe/London", tmp_path)

    assert len(result) == 5
    assert result["temp_mean_c"].tolist() == temps
    cached_files = list(tmp_path.glob("weather_daily_*.csv"))
    assert len(cached_files) == 1


def test_fetch_falls_back_to_cache_on_network_failure(tmp_path, monkeypatch):
    lat, lon = 57.14, -2.10
    cache_df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5, freq="D"), "temp_mean_c": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    cache_path = tmp_path / f"weather_daily_{lat:.4f}_{lon:.4f}.csv"
    cache_df.to_csv(cache_path, index=False)

    def _fail(*args, **kwargs):
        raise requests.ConnectionError("simulated network outage")

    monkeypatch.setattr("src.weather.requests.get", _fail)

    # Requested range extends beyond the cache, so a fetch is attempted, fails, and falls back.
    result = fetch_daily_temperature(lat, lon, "2024-01-01", "2024-01-10", "Europe/London", tmp_path)

    assert len(result) == 5  # only the cached days are returned


def test_fetch_raises_when_network_fails_and_no_cache(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise requests.ConnectionError("simulated network outage")

    monkeypatch.setattr("src.weather.requests.get", _fail)

    with pytest.raises(WeatherFetchError):
        fetch_daily_temperature(57.14, -2.10, "2024-01-01", "2024-01-05", "Europe/London", tmp_path)


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
