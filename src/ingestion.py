"""Load raw OVO Energy CSV exports into a single tidy DataFrame.

Handles multiple yearly files (past or future), files supplied as
filesystem paths, and files uploaded through the Streamlit sidebar
(``UploadedFile`` objects, which are file-like and expose a ``.name``
attribute). Never assumes the data is clean: malformed schemas raise a
clear :class:`IngestionError` instead of being silently coerced.

OVO exports come in three flavours that share the same schema: "Total Use"
(electricity + gas combined), "Electricity Use", and "Gas Use". By default
only "Total Use" is discovered (see :data:`FUEL_FILE_PATTERNS` and
:func:`filter_sources_by_fuel` for fuel-level analysis) -- mixing all three
for the same months would make every month look like a source conflict.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd

from src.utils import get_logger, parse_month_year

logger = get_logger(__name__)

# Canonical column names used throughout the rest of the pipeline.
_MONTH_COL = "Month"
_COST_COL = "Cost (£)"
_CONSUMPTION_COL = "Consumption (kWh)"
_REQUIRED_COLUMNS = (_MONTH_COL, _COST_COL, _CONSUMPTION_COL)

FUEL_FILE_PATTERNS: dict[EnergyType, str] = {
    "total": "*Total Use*.csv",
    "electricity": "*Electricity Use*.csv",
    "gas": "*Gas Use*.csv",
}

# The one generic "energy stream" type every Phase 4 module is written against
# (see FUEL_FILE_PATTERNS above) -- there is deliberately no separate wrapper
# class per fuel; every analysis function already operates on "a monthly clean
# DataFrame with consumption_kwh/cost_gbp" regardless of which fuel produced it.
EnergyType = Literal["total", "electricity", "gas"]


class IngestionError(ValueError):
    """Raised when a source file does not match the expected OVO export schema."""


class _NamedBuffer(Protocol):
    name: str

    def read(self, size: int = ...) -> Any: ...


def discover_csv_files(raw_dir: Path, pattern: str = FUEL_FILE_PATTERNS["total"]) -> list[Path]:
    """Return every file matching ``pattern`` in ``raw_dir``, sorted for deterministic ordering.

    Defaults to "Total Use" exports only -- ingesting Total, Electricity,
    and Gas together for the same months would make every month look like
    a source conflict (see the module docstring). Pass
    ``FUEL_FILE_PATTERNS["electricity"]``/``["gas"]`` for fuel-level
    analysis.
    """
    if not raw_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_dir)
        return []
    return sorted(raw_dir.glob(pattern))


def _source_name(source: Path | _NamedBuffer) -> str:
    if isinstance(source, Path):
        return source.name
    name = getattr(source, "name", None)
    return name if name else "uploaded_file"


def filter_sources_by_fuel(sources: list, fuel: EnergyType) -> list:
    """Filter an already-known list of sources (paths or uploaded files) down to one fuel.

    Works uniformly on filesystem ``Path``s and Streamlit ``UploadedFile``
    objects by matching :func:`_source_name` against
    ``FUEL_FILE_PATTERNS[fuel]``. This is what lets the sidebar uploader
    accept a mixed batch of Total/Electricity/Gas files and have each
    bucketed correctly, rather than assuming every upload is a Total Use
    export.

    Raises:
        KeyError: if ``fuel`` isn't one of ``FUEL_FILE_PATTERNS``.
    """
    pattern = FUEL_FILE_PATTERNS[fuel]
    return [s for s in sources if fnmatch.fnmatch(_source_name(s), pattern)]


def load_single_csv(source: Path | _NamedBuffer) -> pd.DataFrame:
    """Parse one OVO export into a tidy frame with columns:

    ``month_start`` (Timestamp), ``cost_gbp`` (float), ``consumption_kwh`` (float),
    ``source_file`` (str).

    Raises:
        IngestionError: if required columns are missing, or if the ``Month``
            values cannot be parsed as ``"<MonthName> <Year>"``.
    """
    source_name = _source_name(source)
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise IngestionError(f"{source_name}: file is empty") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in _REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise IngestionError(
            f"{source_name}: missing expected column(s) {missing}; found {list(raw.columns)}"
        )

    if raw.empty:
        logger.warning("%s: contains headers but no data rows", source_name)

    try:
        month_start = raw[_MONTH_COL].apply(lambda text: parse_month_year(str(text)))
    except ValueError as exc:
        raise IngestionError(f"{source_name}: {exc}") from exc

    tidy = pd.DataFrame(
        {
            "month_start": pd.to_datetime(month_start),
            "cost_gbp": pd.to_numeric(raw[_COST_COL], errors="coerce"),
            "consumption_kwh": pd.to_numeric(raw[_CONSUMPTION_COL], errors="coerce"),
            "source_file": source_name,
        }
    )

    bad_numeric = tidy[tidy["cost_gbp"].isna() | tidy["consumption_kwh"].isna()]
    if not bad_numeric.empty:
        raise IngestionError(
            f"{source_name}: non-numeric cost/consumption value(s) in row(s) "
            f"{bad_numeric.index.tolist()}"
        )

    logger.info("Loaded %d row(s) from %s", len(tidy), source_name)
    return tidy


def load_all(sources: list[Path] | list[_NamedBuffer]) -> pd.DataFrame:
    """Load and concatenate every source file into one raw (not yet deduplicated) frame.

    Returns an empty, correctly-typed frame if ``sources`` is empty, so
    downstream code never has to special-case "no files".
    """
    if not sources:
        logger.warning("No source files provided to load_all()")
        return pd.DataFrame(
            columns=["month_start", "cost_gbp", "consumption_kwh", "source_file"]
        )

    frames = [load_single_csv(source) for source in sources]
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined %d file(s) into %d total row(s) before deduplication",
        len(sources),
        len(combined),
    )
    return combined
