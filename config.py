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
class WeatherContextThresholds:
    """Day-classification thresholds for the Weather Context Engine (``src.weather_context``).

    These classify *days* into snow/heavy-snow/heavy-rain/strong-wind/severe-gust categories
    for contextual explanation of unusual energy months. They deliberately do NOT feed the
    energy-signature regression -- severe weather is contextual evidence only (see
    ``docs/weather_context.md`` for the design rationale).

    Sources are approximate UK conventions, chosen conservatively so a "severe" label
    understates rather than overstates:

    - ``heavy_snow_day_cm``: 5 cm/day. UK Met Office snow warnings commonly reference
      accumulations of 2-5+ cm; 5 cm/day is the conservative upper end.
    - ``heavy_rain_day_mm``: 25 mm/day. Met Office guidance describes "heavy rain" warnings
      around 25 mm within a few hours; 25 mm over a whole day is therefore conservative.
    - ``strong_wind_day_kmh``: 62 km/h max sustained wind -- the Beaufort scale's gale
      (force 8) threshold (62-74 km/h).
    - ``severe_gust_day_kmh``: 80 km/h max gust (~50 mph) -- the typical trigger level for
      a Met Office yellow wind warning's headline gust figures.
    - ``snow_day_cm``: any measurable snowfall (> 0 cm) counts as a snow day.

    Plausibility bounds (used by validation, not classification): UK 10 m gust record is
    ~228 km/h (Fraserburgh 1989, low-altitude); 300 km/h allows margin without accepting
    nonsense. Snow depth of 6 m exceeds any UK lowland record by a wide margin.
    """

    snow_day_cm: float = 0.0  # snowfall strictly greater than this = a snow day
    heavy_snow_day_cm: float = 5.0
    heavy_rain_day_mm: float = 25.0
    strong_wind_day_kmh: float = 62.0
    severe_gust_day_kmh: float = 80.0
    max_plausible_snow_depth_m: float = 6.0
    max_plausible_wind_kmh: float = 300.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Published UK/Scotland average annual consumption, for the Cost Intelligence tab's
    "below/average/above average" benchmarking (never a single "Energy Score" -- see
    ``src.benchmarking``).

    UK figures: Ofgem "Review of typical domestic consumption values" (TDCV), decision
    published 2026, effective from July 2026 -- medium-usage household. Supersedes the
    older, still widely-cited 2023 TDCV figures (2,700 kWh electricity / 11,500 kWh gas).

    Scotland electricity figure: DESNZ/ONS sub-national electricity consumption
    statistics (England 3,462 / Scotland 3,429 / Wales 3,213 kWh/year -- most recent
    available at time of writing). No Scotland-specific *gas* consumption figure was
    found during research -- ``src.benchmarking`` and the UI say so explicitly rather
    than silently reusing the UK-wide gas figure under a "Scotland" label.
    """

    uk_electricity_kwh_per_year: float = 2500.0
    uk_gas_kwh_per_year: float = 9500.0
    scotland_electricity_kwh_per_year: float = 3429.0
    scotland_gas_kwh_per_year: float | None = None  # deliberately unavailable -- see docstring
    band_tolerance_pct: float = 15.0  # +/- this % of the reference = "Average"


@dataclass(frozen=True)
class CarbonConfig:
    """UK grid/gas emission factors for the Carbon tab's estimates (``src.carbon``).

    Static, documented, offline constants -- not a live API -- consistent with this
    project's "not live data" approach to benchmarking elsewhere in Phase 4.

    electricity_kg_co2e_per_kwh: UK Government (DESNZ/DEFRA) GHG Conversion Factors for
    Company Reporting, 2024 edition, location-based UK grid electricity. The grid factor
    has been declining year over year with continued decarbonisation (a further ~26%
    reduction was reported for more recent editions during research, but could not be
    confirmed to a precise, citable figure at time of writing) -- this should be treated
    as a periodically-refreshed estimate, not a permanent constant.

    gas_kg_co2e_per_kwh: DEFRA GHG Conversion Factors, natural gas combustion (gross
    calorific value basis) -- stable at ~0.182-0.184 kgCO2e/kWh across recent editions,
    since it reflects the chemistry of combustion rather than the grid mix.
    """

    electricity_kg_co2e_per_kwh: float = 0.207
    gas_kg_co2e_per_kwh: float = 0.183


@dataclass(frozen=True)
class MonthComparisonThresholds:
    """Practical-significance thresholds for the month-comparison journey
    (``src.monthly_comparison`` / ``src.monthly_narrative``).

    These are *practical* materiality bands, not statistical significance --
    monthly billing data at n~35 doesn't support a formal test of a single
    month's change, and the narratives never claim one. Rationale:

    - ``little_change_pct``: below 5% a monthly billing figure is within
      ordinary meter-read/billing-calendar noise (OVO bills whole months but
      read dates wobble); calling it "little change" avoids narrating noise.
    - ``large_change_pct``: 20%+ of a month's usage is unmistakably material
      on a household bill; between the two bounds is "moderate".
    - ``min_same_month_observations``: with only one prior observation of a
      calendar month there is no distribution to call "typical"; two or more
      prior observations are required before typical/best/worst modes claim
      anything, and even then confidence is capped at Medium below four.
    - ``meaningful_residual_z``: a single month's weather-unexplained gap is
      called meaningful only beyond 1.645 residual standard deviations
      (one-sided 95% normal approximation) -- the same convention
      ``src.energy_signature`` and ``src.recommendations`` already use for
      annual and winter residuals.
    """

    little_change_pct: float = 5.0
    large_change_pct: float = 20.0
    min_same_month_observations: int = 2
    solid_same_month_observations: int = 4
    meaningful_residual_z: float = 1.645


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
    weather_context: WeatherContextThresholds = field(default_factory=WeatherContextThresholds)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    carbon: CarbonConfig = field(default_factory=CarbonConfig)
    month_comparison: MonthComparisonThresholds = field(default_factory=MonthComparisonThresholds)


SETTINGS = Settings()
