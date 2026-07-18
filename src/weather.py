
"""Fetch historical daily temperature and derive monthly heating/cooling degree days.

Weather data comes from the free Open-Meteo archive API (no key required)
and is cached to disk so the app works offline after the first successful
fetch. The resulting degree days feed the "energy signature" regression in
:mod:`src.energy_signature`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.utils import get_logger

logger = get_logger(__name__)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class WeatherFetchError(RuntimeError):
    """Raised when weather data can't be fetched and no usable cache exists."""


def _cache_path(cache_dir: Path, lat: float, lon: float) -> Path:
    return cache_dir / f"weather_daily_{lat:.4f}_{lon:.4f}.csv"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except (OSError, ValueError, KeyError):
        logger.warning("Could not read weather cache at %s; ignoring it", path)
        return None
    return df.sort_values("date").reset_index(drop=True)


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(path, index=False)


def _request_daily_temperature(lat: float, lon: float, start: str, end: str, tz: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": tz,
    }
    response = requests.get(_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]
    return pd.DataFrame(
        {"date": pd.to_datetime(daily["time"]), "temp_mean_c": daily["temperature_2m_mean"]}
    )


def fetch_daily_temperature(
    lat: float,
    lon: float,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    timezone: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Return daily mean temperature for ``[start, end]``, using and updating a disk cache.

    Behaviour: a cache that already fully covers the requested range is used
    without any network call. Otherwise Open-Meteo is queried for the full
    range and the result is merged into the cache. If the request fails
    (network error, non-2xx response), falls back to whatever cache exists
    (logging a warning that it may be incomplete/stale); with no cache at
    all, raises :class:`WeatherFetchError` rather than crashing the caller.

    The archive API has no data for future dates and only a ~1 day lag on
    recent ones, so ``end`` is silently clipped to yesterday if it's later
    than that -- relevant when the latest month in the consumption data is
    still in progress. Callers that need to know a month's weather coverage
    is incomplete should check the ``n_days`` column from
    :func:`compute_monthly_degree_days` rather than relying on ``end`` alone.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    if end > yesterday:
        logger.info("Clipping requested end date %s to %s (no future weather data)", end.date(), yesterday.date())
        end = yesterday
    if start > end:
        raise WeatherFetchError(
            f"Requested start date {start.date()} is after the latest available weather date "
            f"{end.date()}."
        )
    cache_path = _cache_path(cache_dir, lat, lon)
    cached = _load_cache(cache_path)

    if cached is not None and cached["date"].min() <= start and cached["date"].max() >= end:
        logger.info("Weather cache hit for (%.4f, %.4f), %s to %s", lat, lon, start.date(), end.date())
        return cached[(cached["date"] >= start) & (cached["date"] <= end)].reset_index(drop=True)

    try:
        fetched = _request_daily_temperature(
            lat, lon, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), timezone
        )
    except requests.RequestException as exc:
        logger.warning("Open-Meteo request failed (%s); falling back to cache if available", exc)
        if cached is not None:
            return cached[(cached["date"] >= start) & (cached["date"] <= end)].reset_index(drop=True)
        raise WeatherFetchError(f"Could not fetch weather data and no cache exists: {exc}") from exc

    combined = fetched if cached is None else pd.concat([cached, fetched]).drop_duplicates("date")
    combined = combined.sort_values("date").reset_index(drop=True)
    _save_cache(cache_path, combined)

    return combined[(combined["date"] >= start) & (combined["date"] <= end)].reset_index(drop=True)


def compute_monthly_degree_days(
    daily_df: pd.DataFrame, base_heat_c: float, base_cool_c: float
) -> pd.DataFrame:
    """Aggregate daily temperature into monthly heating/cooling degree days.

    HDD(day) = max(0, base_heat_c - temp_mean_c); CDD(day) = max(0, temp_mean_c - base_cool_c).
    Includes an ``n_days`` count of how many days of temperature data went
    into each month, so callers can detect and exclude months where weather
    coverage is incomplete (e.g. the current in-progress month) rather than
    silently under-counting their degree days.
    """
    df = daily_df.copy()
    df["hdd"] = (base_heat_c - df["temp_mean_c"]).clip(lower=0)
    df["cdd"] = (df["temp_mean_c"] - base_cool_c).clip(lower=0)
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month_start")
        .agg(
            hdd=("hdd", "sum"),
            cdd=("cdd", "sum"),
            avg_temp_c=("temp_mean_c", "mean"),
            n_days=("date", "count"),
        )
        .reset_index()
    )
    return monthly


def merge_weather_with_consumption(monthly_df: pd.DataFrame, degree_days_df: pd.DataFrame) -> pd.DataFrame:
    """Join consumption with degree days on ``month_start``, adding daily-normalized weather columns.

    Months where weather coverage doesn't span the full month (``n_days <
    days_in_month`` -- typically the current in-progress month, where the
    archive API has no future data) are dropped with a warning rather than
    silently producing an under-counted ``avg_daily_hdd``/``avg_daily_cdd``.
    """
    merged = monthly_df.merge(degree_days_df, on="month_start", how="inner")

    missing = set(monthly_df["month_start"]) - set(merged["month_start"])
    if missing:
        logger.warning(
            "%d month(s) have consumption data but no matching weather data: %s",
            len(missing),
            sorted(m.strftime("%Y-%m") for m in missing),
        )

    incomplete = merged["n_days"] < merged["days_in_month"]
    if incomplete.any():
        logger.warning(
            "Dropping %d month(s) with incomplete weather coverage: %s",
            int(incomplete.sum()),
            sorted(m.strftime("%Y-%m") for m in merged.loc[incomplete, "month_start"]),
        )
        merged = merged.loc[~incomplete].reset_index(drop=True)

    merged["avg_daily_hdd"] = merged["hdd"] / merged["days_in_month"]
    merged["avg_daily_cdd"] = merged["cdd"] / merged["days_in_month"]
    return merged
