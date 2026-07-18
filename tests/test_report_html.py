import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from report_html import build_household_report_html  # noqa: E402

from src.confidence import ConfidenceRating  # noqa: E402
from src.findings import Finding  # noqa: E402
from src.preprocessing import PreprocessingReport  # noqa: E402
from src.recommendations import NO_RECOMMENDATIONS_MESSAGE, Recommendation  # noqa: E402
from src.report import AnalystReport  # noqa: E402

_SECTION_HEADINGS = [
    "Household Energy Review",
    "Executive summary",
    "Key findings",
    "Fuel mix",
    "Weather analysis",
    "Consumption history",
    "Forecast",
    "Recommendations",
    "Carbon",
    "Benchmark",
    "Methodology",
    "Limitations and data quality",
]


def _clean_df(n=24, start="2023-01-01", kwh=300.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    df = pd.DataFrame({"month_start": idx, "consumption_kwh": np.full(n, kwh), "cost_gbp": np.full(n, kwh * 0.3)})
    df["year"] = df["month_start"].dt.year
    df["month_num"] = df["month_start"].dt.month
    df["month_name"] = df["month_start"].dt.strftime("%B")
    df["days_in_month"] = df["month_start"].dt.days_in_month
    df["is_partial_year"] = df.groupby("year")["month_start"].transform("size") < 12
    return df


def _confidence_dict():
    ok = ConfidenceRating("High", "clean")
    return {"data_quality": ok, "weather_model": ok, "forecast": ok, "anomaly_detection": ok}


def _analyst_report(**overrides) -> AnalystReport:
    defaults = dict(
        executive_summary="Overall things look stable.",
        overall_assessment="Consumption is running at normal, expected levels.",
        findings=[],
        biggest_finding=None,
        recommendations=[],
        largest_saving=None,
        confidence=_confidence_dict(),
        limitations=["Only 24 months of history."],
        monitoring_priorities=["Continue routine monitoring."],
    )
    defaults.update(overrides)
    return AnalystReport(**defaults)


def _preprocessing_report(warnings=None) -> PreprocessingReport:
    return PreprocessingReport(
        n_files_loaded=1,
        source_files=["a.csv"],
        date_range=(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-01")),
        n_months=24,
        missing_months=[],
        duplicates_removed=0,
        conflicts=[],
        outlier_warnings=warnings or [],
    )


def _build(analyst_report=None, fuel_frames=None, weather_enabled=False, prep_report=None) -> str:
    return build_household_report_html(
        analyst_report=analyst_report or _analyst_report(),
        fuel_frames=fuel_frames if fuel_frames is not None else {"total": _clean_df()},
        fuel_merged={},
        fuel_energy_results={},
        forecast_12mo=None,
        weather_enabled=weather_enabled,
        preprocessing_report=prep_report or _preprocessing_report(),
    )


def test_report_is_full_html_document_with_all_sections():
    html = _build()
    assert html.startswith("<!DOCTYPE html>")
    for heading in _SECTION_HEADINGS:
        assert heading in html, f"missing section: {heading}"


def test_report_includes_finding_narratives_and_confidence():
    finding = Finding(
        title="Seasonal pattern",
        narrative="Winter usage is roughly 8x summer usage.",
        evidence=["Seasonal strength: 93%"],
        confidence="High",
        confidence_reason="clear pattern",
        category="seasonality",
    )
    html = _build(analyst_report=_analyst_report(findings=[finding]))
    assert "Winter usage is roughly 8x summer usage." in html
    assert "Seasonal strength: 93%" in html
    assert "High confidence" in html


def test_report_weather_off_shows_honest_note():
    html = _build(weather_enabled=False)
    assert "Weather adjustment was not enabled" in html


def test_report_no_recommendations_uses_existing_message():
    html = _build()
    assert NO_RECOMMENDATIONS_MESSAGE in html


def test_report_includes_recommendations_when_present():
    rec = Recommendation(
        title="Focus on gas",
        action="Reducing gas demand is likely to produce larger savings.",
        estimated_saving_gbp=None,
        evidence=["Consumption share: gas 65%"],
        confidence="High",
        confidence_reason="both signals agree",
        rationale="Gas dominates both consumption and heating sensitivity.",
    )
    html = _build(analyst_report=_analyst_report(recommendations=[rec]))
    assert "Reducing gas demand is likely to produce larger savings." in html
    assert "Consumption share: gas 65%" in html


def test_report_missing_fuels_degrade_honestly():
    html = _build(fuel_frames={"total": _clean_df()})
    assert "needs both Electricity and Gas exports" in html  # fuel mix section
    assert "Carbon estimates need both Electricity and Gas" in html
    assert "Benchmarking needs fuel-level data" in html


def test_report_surfaces_data_warnings():
    prep = _preprocessing_report(warnings=["July 2026 is the current calendar month -- month-to-date."])
    html = _build(prep_report=prep)
    assert "month-to-date" in html


def test_report_escapes_html_in_narratives():
    finding = Finding(
        title="Injected",
        narrative='<script>alert("x")</script>',
        evidence=[],
        confidence="Low",
        confidence_reason="r",
        category="trend",
    )
    html = _build(analyst_report=_analyst_report(findings=[finding]))
    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;" in html


def test_report_raises_on_empty_data():
    with pytest.raises(ValueError, match="No data"):
        _build(fuel_frames={"total": pd.DataFrame()})


def test_report_standing_charge_qualifier_present():
    html = _build()
    assert "exclude standing charges" in html or "excl. standing charges" in html
