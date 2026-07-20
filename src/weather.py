"""Fetch historical daily weather and derive monthly heating/cooling degree days.

Weather data comes from the free Open-Meteo archive API (no key required)
and is cached to disk so the app works offline after the first successful
fetch. Daily mean temperature feeds the "energy signature" regression in
:mod:`src.energy_signature`; the additional snow/precipitation/wind fields
feed the Weather Context Engine (:mod:`src.weather_context`) as *contextual
evidence only* -- they are deliberately not regression predictors (see
``docs/weather_context.md``).

Data contract for the daily frame returned by :func:`fetch_daily_weather`
(exact Open-Meteo variable, unit, and missing-value policy per column) is
``DAILY_WEATHER_SCHEMA`` below. Missing *temperature* makes a day not count
toward monthly weather coverage; missing context fields (snow/wind/rain)
stay ``NaN`` -- they are never silently converted to zero, because for
these variables zero means "none observed", not "not available".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import SETTINGS
from src.utils import get_logger

logger = get_logger(__name__)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# column name -> (Open-Meteo daily variable, unit, required, description)
# "required": the fetch fails without it. Non-required fields degrade to an all-NaN
# column with a logged warning, and src.weather_context marks affected months
# ``weather_context_complete = False``.
DAILY_WEATHER_SCHEMA: dict[str, tuple[str, str, bool, str]] = {
    "temp_mean_c": ("temperature_2m_mean", "°C", True, "Daily mean 2 m air temperature"),
    "snowfall_cm": ("snowfall_sum", "cm", False, "Total daily snowfall; 0 = none observed"),
    "snow_depth_m": ("snow_depth_mean", "m", False, "Mean snow depth on the ground"),
    "precipitation_mm": ("precipitation_sum", "mm", False, "Total daily precipitation (rain + snow water equivalent)"),
    "wind_speed_max_kmh": ("wind_speed_10m_max", "km/h", False, "Maximum sustained 10 m wind speed"),
    "wind_gust_max_kmh": ("wind_gusts_10m_max", "km/h", False, "Maximum 10 m wind gust"),
}

# Bump when the fetched schema changes so older caches are never mistaken for new ones.
# v1 (implicit, unversioned filename): temperature only. v2: the six-field schema above.
_CACHE_SCHEMA_VERSION = 2


class WeatherFetchError(RuntimeError):
    """Raised when weather data can't be fetched and no usable cache exists."""


def _cache_path(cache_dir: Path, lat: float, lon: float, timezone: str) -> Path:
    """v2 cache identity includes schema version and timezone.

    Timezone matters because Open-Meteo aggregates daily values in the requested
    timezone -- the same coordinates fetched under a different timezone would have
    slightly different daily boundaries. v1 caches (``weather_daily_<lat>_<lon>.csv``,
    temperature-only) use a different filename and are left untouched: v2 simply
    fetches fresh once and uses its own file from then on.
    """
    tz_slug = timezone.replace("/", "-")
    return cache_dir / f"weather_daily_v{_CACHE_SCHEMA_VERSION}_{lat:.4f}_{lon:.4f}_{tz_slug}.csv"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except (OSError, ValueError, KeyError):
        logger.warning("Could not read weather cache at %s; ignoring it", path)
        return None
    if missing := [c for c in DAILY_WEATHER_SCHEMA if c not in df.columns]:
        logger.warning("Weather cache at %s lacks columns %s; ignoring it", path, missing)
        return None
    return df.sort_values("date").reset_index(drop=True)


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(path, index=False)


def _parse_daily_response(response: requests.Response) -> pd.DataFrame:
    """Validate an Open-Meteo daily response and convert it to the schema frame.

    Never trusts ``response.json()["daily"]`` blindly: handles invalid JSON,
    API error payloads, a missing/short ``daily`` block, missing fields, and
    unequal array lengths, converting each into :class:`WeatherFetchError`
    with the original exception preserved as the cause where there is one.
    """
    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherFetchError(f"Open-Meteo returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise WeatherFetchError(f"Open-Meteo returned an unexpected payload type: {type(payload).__name__}")
    if payload.get("error"):
        raise WeatherFetchError(f"Open-Meteo returned an error payload: {payload.get('reason', 'no reason given')}")

    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise WeatherFetchError("Open-Meteo response has no usable 'daily' block.")

    try:
        dates = pd.to_datetime(daily["time"])
    except (ValueError, TypeError) as exc:
        raise WeatherFetchError(f"Open-Meteo returned unparseable dates: {exc}") from exc

    frame: dict[str, object] = {"date": dates}
    for column, (api_field, _unit, required, _desc) in DAILY_WEATHER_SCHEMA.items():
        values = daily.get(api_field)
        if values is None:
            if required:
                raise WeatherFetchError(f"Open-Meteo response is missing required field '{api_field}'.")
            logger.warning(
                "Open-Meteo response is missing optional field '%s'; %s will be NaN "
                "(weather context for affected months will be marked incomplete)",
                api_field,
                column,
            )
            frame[column] = np.full(len(dates), np.nan)
            continue
        if len(values) != len(dates):
            raise WeatherFetchError(
                f"Open-Meteo field '{api_field}' has {len(values)} values for {len(dates)} dates."
            )
        # JSON nulls arrive as None -> NaN. Deliberately NOT filled with 0: for snow/wind/
        # rain, zero means "none observed" while NaN means "not available".
        frame[column] = pd.array(values, dtype="float64")

    return pd.DataFrame(frame)


def validate_daily_weather(df: pd.DataFrame) -> list[str]:
    """Check the daily frame against the schema's plausibility rules, in place.

    Hard violations (missing required columns, duplicate dates) raise
    :class:`WeatherFetchError`. Implausible values (negative snowfall/
    precipitation, snow depth or wind beyond the configured bounds) are set
    to ``NaN`` -- never silently kept or clipped to a plausible-looking value
    -- and reported in the returned warnings list.
    """
    if missing := [c for c in ("date", *DAILY_WEATHER_SCHEMA) if c not in df.columns]:
        raise WeatherFetchError(f"Daily weather frame is missing columns: {missing}")
    if df["date"].duplicated().any():
        raise WeatherFetchError("Daily weather frame contains duplicate dates.")

    bounds = SETTINGS.weather_context
    warnings: list[str] = []
    checks = [
        ("snowfall_cm", lambda s: s < 0, "negative snowfall"),
        ("precipitation_mm", lambda s: s < 0, "negative precipitation"),
        ("snow_depth_m", lambda s: (s < 0) | (s > bounds.max_plausible_snow_depth_m), "implausible snow depth"),
        ("wind_speed_max_kmh", lambda s: (s < 0) | (s > bounds.max_plausible_wind_kmh), "implausible wind speed"),
        ("wind_gust_max_kmh", lambda s: (s < 0) | (s > bounds.max_plausible_wind_kmh), "implausible wind gust"),
    ]
    for column, is_bad, label in checks:
        bad = is_bad(df[column]).fillna(False)
        if bad.any():
            warnings.append(f"{int(bad.sum())} day(s) with {label} set to NaN.")
            df.loc[bad, column] = np.nan
    n_missing_temp = int(df["temp_mean_c"].isna().sum())
    if n_missing_temp:
        warnings.append(
            f"{n_missing_temp} day(s) have no mean temperature; they don't count toward monthly weather coverage."
        )
    for w in warnings:
        logger.warning("Daily weather validation: %s", w)
    return warnings


def _request_daily_weather(lat: float, lon: float, start: str, end: str, tz: str) -> pd.DataFrame:
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(api_field for api_field, *_ in DAILY_WEATHER_SCHEMA.values()),
        "timezone": tz,
    }
    response = requests.get(_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    df = _parse_daily_response(response)
    validate_daily_weather(df)
    return df


def fetch_daily_weather(
    lat: float,
    lon: float,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    timezone: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Return daily weather (see ``DAILY_WEATHER_SCHEMA``) for ``[start, end]``, disk-cached.

    Behaviour: a cache that already fully covers the requested range is used
    without any network call. Otherwise Open-Meteo is queried for the full
    range and the result is merged into the cache. If the request fails
    (network error, non-2xx response, malformed response), falls back to
    whatever cache exists (logging a warning that it may be incomplete/
    stale); with no cache at all, raises :class:`WeatherFetchError` rather
    than crashing the caller.

    The archive API has no data for future dates and only a ~1 day lag on
    recent ones, so ``end`` is silently clipped to yesterday if it's later
    than that -- relevant when the latest month in the consumption data is
    still in progress. Callers that need to know a month's weather coverage
    is incomplete should check the ``n_days`` column from
    :func:`compute_monthly_degree_days` rather than relying on ``end`` alone.

    A full-range refetch (not a missing-range delta fetch) happens at most
    once per new consumption month, so the added transfer is small; delta
    fetching is a documented deferral in ``docs/weather_context.md``.
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
    cache_path = _cache_path(cache_dir, lat, lon, timezone)
    cached = _load_cache(cache_path)

    if cached is not None and cached["date"].min() <= start and cached["date"].max() >= end:
        logger.info("Weather cache hit for (%.4f, %.4f), %s to %s", lat, lon, start.date(), end.date())
        return cached[(cached["date"] >= start) & (cached["date"] <= end)].reset_index(drop=True)

    try:
        fetched = _request_daily_weather(
            lat, lon, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), timezone
        )
    except (requests.RequestException, WeatherFetchError) as exc:
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
    Includes an ``n_days`` count of how many days of *non-null* temperature
    went into each month, so callers can detect and exclude months where
    weather coverage is incomplete (e.g. the current in-progress month, or a
    month with null temperatures in the API response) rather than silently
    under-counting their degree days.
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
            n_days=("temp_mean_c", "count"),  # count() skips NaN: null-temp days aren't coverage
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
