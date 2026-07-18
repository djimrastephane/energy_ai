"""Shared, dependency-free helpers used across the platform."""

from __future__ import annotations

import logging
from datetime import datetime

_CONFIGURED_LOGGERS: set[str] = set()

_MONTH_NAME_TO_NUM = {
    name: i
    for i, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured once with a stream handler.

    Safe to call repeatedly (e.g. once per module import) -- handlers are
    only attached the first time a given logger name is requested, so
    messages are never duplicated.
    """
    from config import SETTINGS

    logger = logging.getLogger(name)
    if name not in _CONFIGURED_LOGGERS:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(SETTINGS.logging.fmt, datefmt=SETTINGS.logging.datefmt)
        )
        logger.addHandler(handler)
        logger.setLevel(SETTINGS.logging.level)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)
    return logger


def parse_month_year(text: str) -> datetime:
    """Parse an OVO-style ``"September 2023"`` string into a month-start datetime.

    Raises:
        ValueError: if the text does not match a recognised ``"<Month> <Year>"``
            format.
    """
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Expected '<Month> <Year>', got: {text!r}")
    month_name, year_text = parts
    month_name = month_name.capitalize()
    if month_name not in _MONTH_NAME_TO_NUM:
        raise ValueError(f"Unrecognised month name: {month_name!r} in {text!r}")
    if not year_text.isdigit() or len(year_text) != 4:
        raise ValueError(f"Unrecognised year: {year_text!r} in {text!r}")
    return datetime(int(year_text), _MONTH_NAME_TO_NUM[month_name], 1)


def format_gbp(value: float) -> str:
    """Format a number as a GBP currency string, e.g. ``1234.5 -> "£1,234.50"``."""
    return f"£{value:,.2f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` instead of raising on a zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def pct_change(current: float, previous: float) -> float | None:
    """Percentage change from ``previous`` to ``current``.

    Returns ``None`` (rather than inf/nan) when ``previous`` is zero, since a
    percentage change is undefined in that case.
    """
    if previous == 0:
        return None
    return (current - previous) / previous * 100.0
