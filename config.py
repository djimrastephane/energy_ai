"""Central configuration for the AI Home Energy Intelligence Platform.

All paths are resolved relative to this file so the application works
regardless of the current working directory it is launched from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

for _directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ValidationThresholds:
    """Plausibility bounds used to flag (not silently drop) suspect rows.

    Bounds reflect a typical UK residential household on a standard
    electricity-only tariff. Values outside this range are not evidence of
    an error by themselves -- they are surfaced to the user as warnings so a
    human can judge whether they reflect real events (e.g. an EV charger
    installed mid-year) or a data entry problem.
    """

    min_monthly_kwh: float = 0.0
    max_monthly_kwh: float = 3000.0
    min_monthly_cost_gbp: float = 0.0
    max_monthly_cost_gbp: float = 600.0
    max_plausible_unit_rate_gbp_per_kwh: float = 1.00


@dataclass(frozen=True)
class LoggingConfig:
    level: int = logging.INFO
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class WeatherConfig:
    """Location and degree-day settings for weather-adjusted analysis.

    The location was provided by the user (Aberdeen, Scotland / postcode
    area AB21) and geocoded via Open-Meteo's free geocoding API to city-
    centre coordinates -- close enough to AB21 (a few km) that the
    difference is immaterial for monthly degree-day regression in a
    maritime Scottish climate.

    Degree-day base temperatures follow the standard UK convention (15.5C
    heating base; see e.g. degreedays.net's documentation of UK practice).
    22C for cooling is a conservative UK default -- Aberdeen in particular
    is expected to show near-zero cooling degree days, which is a real
    finding, not a bug.
    """

    latitude: float = 57.14369
    longitude: float = -2.09814
    location_label: str = "Aberdeen, Scotland (AB21 area)"
    timezone: str = "Europe/London"
    base_heat_c: float = 15.5
    base_cool_c: float = 22.0


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    weather_cache_dir: Path = PROCESSED_DATA_DIR
    validation: ValidationThresholds = field(default_factory=ValidationThresholds)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)


SETTINGS = Settings()
