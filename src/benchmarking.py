"""Task 12 -- benchmarking against published UK/Scotland average annual consumption.

No single "Energy Score": reports a categorical band (Below average /
Average / Above average) with a confidence level, per the spec's explicit
constraint. Reference values are static, cited config constants
(``config.BenchmarkConfig``), not live data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import SETTINGS
from src.confidence import Confidence
from src.ingestion import EnergyType

Region = Literal["uk", "scotland"]
Band = Literal["Below average", "Average", "Above average"]


@dataclass
class BenchmarkResult:
    fuel: EnergyType
    region: Region
    band: Band
    annual_kwh: float
    reference_kwh: float
    confidence: Confidence
    source: str


def _reference_kwh(fuel: EnergyType, region: Region) -> tuple[float | None, str]:
    b = SETTINGS.benchmark
    if fuel == "electricity":
        if region == "scotland":
            return (
                b.scotland_electricity_kwh_per_year,
                "DESNZ/ONS sub-national electricity consumption statistics, Scotland average",
            )
        return b.uk_electricity_kwh_per_year, "Ofgem Typical Domestic Consumption Values (TDCV), 2026, medium usage"
    if fuel == "gas":
        if region == "scotland":
            return (
                b.scotland_gas_kwh_per_year,
                "No Scotland-specific gas consumption benchmark is available -- not estimated",
            )
        return b.uk_gas_kwh_per_year, "Ofgem Typical Domestic Consumption Values (TDCV), 2026, medium usage"
    return (
        None,
        "Benchmarking isn't meaningful for 'total' -- it mixes two fuels with very different "
        "typical scales; compare Electricity and Gas separately instead.",
    )


def compare_to_benchmark(
    annual_kwh: float,
    fuel: EnergyType,
    region: Region,
    n_months: int,
    weather_adjusted: bool,
) -> BenchmarkResult | None:
    """Compare one fuel's annual consumption to a published reference.

    Returns ``None`` when no reference value exists for this fuel/region
    combination (``fuel="total"``, or Scotland gas -- see
    ``config.BenchmarkConfig``'s docstring) rather than silently falling
    back to a misleading substitute.
    """
    reference_kwh, source = _reference_kwh(fuel, region)
    if reference_kwh is None:
        return None

    tolerance = SETTINGS.benchmark.band_tolerance_pct / 100
    lower, upper = reference_kwh * (1 - tolerance), reference_kwh * (1 + tolerance)
    band: Band
    if annual_kwh < lower:
        band = "Below average"
    elif annual_kwh > upper:
        band = "Above average"
    else:
        band = "Average"

    confidence: Confidence = "High" if (n_months >= 12 and weather_adjusted) else "Medium"

    return BenchmarkResult(
        fuel=fuel,
        region=region,
        band=band,
        annual_kwh=annual_kwh,
        reference_kwh=reference_kwh,
        confidence=confidence,
        source=source,
    )
