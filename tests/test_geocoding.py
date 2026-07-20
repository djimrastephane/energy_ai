import json

import pytest
import requests

from src.geocoding import GeocodingError, LocationCandidate, search_location


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


def test_empty_query_returns_empty_list_without_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("network should not be called for an empty query")

    monkeypatch.setattr("src.geocoding.requests.get", _boom)
    assert search_location("") == []
    assert search_location("   ") == []


def test_no_results_returns_empty_list(monkeypatch):
    payload = {"generationtime_ms": 0.5}  # no "results" key -- the real API's shape for zero matches
    monkeypatch.setattr("src.geocoding.requests.get", lambda *a, **kw: _FakeResponse(payload))

    assert search_location("zzzznotarealplacexyz") == []


def test_parses_candidates_with_admin1_and_country(monkeypatch):
    payload = {
        "results": [
            {
                "name": "Manchester",
                "latitude": 53.48095,
                "longitude": -2.23743,
                "timezone": "Europe/London",
                "admin1": "England",
                "country": "United Kingdom",
            },
            {
                "name": "Manchester",
                "latitude": 42.99564,
                "longitude": -71.45479,
                "timezone": "America/New_York",
                "admin1": "New Hampshire",
                "country": "United States",
            },
        ]
    }
    monkeypatch.setattr("src.geocoding.requests.get", lambda *a, **kw: _FakeResponse(payload))

    candidates = search_location("Manchester")

    assert candidates == [
        LocationCandidate(
            label="Manchester, England, United Kingdom",
            latitude=53.48095,
            longitude=-2.23743,
            timezone="Europe/London",
        ),
        LocationCandidate(
            label="Manchester, New Hampshire, United States",
            latitude=42.99564,
            longitude=-71.45479,
            timezone="America/New_York",
        ),
    ]


def test_omits_admin1_when_redundant_with_name(monkeypatch):
    """City-states / single-name matches (admin1 == name) shouldn't duplicate in the label."""
    payload = {
        "results": [
            {
                "name": "Singapore",
                "latitude": 1.28967,
                "longitude": 103.85007,
                "timezone": "Asia/Singapore",
                "admin1": "Singapore",
                "country": "Singapore",
            }
        ]
    }
    monkeypatch.setattr("src.geocoding.requests.get", lambda *a, **kw: _FakeResponse(payload))

    candidates = search_location("Singapore")

    assert candidates[0].label == "Singapore, Singapore"


def test_missing_timezone_defaults_to_auto(monkeypatch):
    payload = {"results": [{"name": "Nowhere", "latitude": 0.0, "longitude": 0.0, "country": "Testland"}]}
    monkeypatch.setattr("src.geocoding.requests.get", lambda *a, **kw: _FakeResponse(payload))

    candidates = search_location("Nowhere")

    assert candidates[0].timezone == "auto"


def test_raises_geocoding_error_on_network_failure(monkeypatch):
    def _fail(*args, **kwargs):
        raise requests.ConnectionError("simulated network outage")

    monkeypatch.setattr("src.geocoding.requests.get", _fail)

    with pytest.raises(GeocodingError, match="Aberdeen"):
        search_location("Aberdeen")


def test_raises_geocoding_error_on_http_error(monkeypatch):
    monkeypatch.setattr("src.geocoding.requests.get", lambda *a, **kw: _FakeResponse({}, ok=False))

    with pytest.raises(GeocodingError):
        search_location("Aberdeen")


def test_raises_geocoding_error_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "src.geocoding.requests.get", lambda *a, **kw: _FakeResponse(None, invalid_json=True)
    )

    with pytest.raises(GeocodingError):
        search_location("Aberdeen")


def test_count_is_forwarded_as_a_request_param(monkeypatch):
    seen = {}

    def _fake_get(url, params=None, timeout=None):
        seen.update(params or {})
        return _FakeResponse({"results": []})

    monkeypatch.setattr("src.geocoding.requests.get", _fake_get)
    search_location("Aberdeen", count=3)

    assert seen["count"] == 3
    assert seen["name"] == "Aberdeen"
