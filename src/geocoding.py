"""Resolve a free-text place name to coordinates via Open-Meteo's free geocoding API.

Deliberately separate from :mod:`src.weather` (which fetches historical
weather for already-known coordinates): geocoding is a one-off text-to-
coordinates lookup, not part of the degree-day pipeline itself. Common
place names are ambiguous (there are Manchesters in England, New
Hampshire, and Tennessee) -- callers must treat the result list as
candidates for the user to confirm, never the single answer to apply
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from src.utils import get_logger

logger = get_logger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class GeocodingError(RuntimeError):
    """Raised when the location search itself fails (network/API error).

    A query that simply matches no place is not an error -- it returns an
    empty list, same convention as an empty search result anywhere else.
    """


@dataclass(frozen=True)
class LocationCandidate:
    """One geocoding match, ready to display for user confirmation."""

    label: str  # e.g. "Manchester, England, United Kingdom"
    latitude: float
    longitude: float
    timezone: str


def search_location(query: str, count: int = 5) -> list[LocationCandidate]:
    """Return up to ``count`` candidate places matching ``query``, best match first.

    An empty/whitespace-only ``query`` returns an empty list without a network
    call. Raises :class:`GeocodingError` only on a genuine request failure
    (network error, non-2xx response, malformed JSON) -- callers should let
    the user retry rather than silently falling back to a default location.
    """
    query = query.strip()
    if not query:
        return []

    try:
        params: dict[str, str | int] = {"name": query, "count": count, "language": "en", "format": "json"}
        response = requests.get(_GEOCODING_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise GeocodingError(f"Could not search for {query!r}: {exc}") from exc

    candidates = []
    for result in payload.get("results") or []:
        parts = [result["name"]]
        admin1 = result.get("admin1")
        if admin1 and admin1 != result["name"]:
            parts.append(admin1)
        country = result.get("country")
        if country:
            parts.append(country)
        candidates.append(
            LocationCandidate(
                label=", ".join(parts),
                latitude=result["latitude"],
                longitude=result["longitude"],
                timezone=result.get("timezone", "auto"),
            )
        )
    logger.info("Location search %r returned %d candidate(s)", query, len(candidates))
    return candidates
