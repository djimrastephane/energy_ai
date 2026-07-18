import warnings

import pandas as pd

from config import SETTINGS
from src.anomalies import detect_anomalies
from src.changepoints import detect_changepoints
from src.confidence import ConfidenceRating
from src.decomposition import stl_decompose
from src.forecast_evaluation import generate_forecast
from src.ingestion import discover_csv_files, load_all
from src.preprocessing import PreprocessingReport, run_pipeline
from src.report import (
    NO_SAVINGS_MESSAGE,
    _limitations,
    _monitoring_priorities,
    _overall_assessment,
    build_analyst_report,
)


def _report(missing=None) -> PreprocessingReport:
    return PreprocessingReport(
        n_files_loaded=1,
        source_files=["a.csv"],
        date_range=(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-01")),
        n_months=36,
        missing_months=missing or [],
        duplicates_removed=0,
        conflicts=[],
        outlier_warnings=[],
    )


def test_overall_assessment_reports_high_uncertainty_on_low_data_quality():
    text = _overall_assessment(10.0, ConfidenceRating("Low", "big gaps"))
    assert "High uncertainty" in text


def test_overall_assessment_normal_for_small_change():
    text = _overall_assessment(2.0, ConfidenceRating("High", "clean"))
    assert "normal, expected levels" in text


def test_overall_assessment_higher_than_expected():
    text = _overall_assessment(12.0, ConfidenceRating("High", "clean"))
    assert "higher than expected" in text
    assert "12%" in text


def test_overall_assessment_lower_than_expected():
    text = _overall_assessment(-8.0, ConfidenceRating("High", "clean"))
    assert "lower than expected" in text


def test_overall_assessment_none_when_no_yoy_data():
    text = _overall_assessment(None, ConfidenceRating("High", "clean"))
    assert "Not enough history" in text


def test_limitations_flags_short_history():
    items = _limitations(n_months=20, report=_report(), energy_result=None, anomalies=[])
    assert any("Only 20 months" in item for item in items)
    assert any("Weather adjustment is off" in item for item in items)


def test_limitations_flags_missing_months():
    items = _limitations(n_months=36, report=_report(missing=[pd.Timestamp("2024-05-01")]), energy_result=None, anomalies=[])
    assert any("missing from the billing data" in item for item in items)


def test_monitoring_priorities_fallback_when_everything_high():
    items = _monitoring_priorities(
        ConfidenceRating("High", "a"), ConfidenceRating("High", "b"), ConfidenceRating("High", "c"), []
    )
    assert items == ["No specific concerns identified -- continue routine monitoring."]


def test_monitoring_priorities_lists_non_high_domains():
    items = _monitoring_priorities(
        ConfidenceRating("Low", "gaps"), ConfidenceRating("High", "b"), ConfidenceRating("Medium", "wide bands"), []
    )
    assert any("Data quality" in i for i in items)
    assert any("Forecast reliability" in i for i in items)
    assert not any("Weather model" in i for i in items)


def test_build_analyst_report_end_to_end_on_real_data():
    warnings.filterwarnings("ignore")
    files = discover_csv_files(SETTINGS.raw_data_dir)
    raw = load_all(files)
    clean, report = run_pipeline(raw)

    stl_result = stl_decompose(clean)
    changepoints = detect_changepoints(stl_result.deseasonalized)
    anomalies = detect_anomalies(clean, stl_result)
    series = clean.set_index("month_start")["consumption_kwh"]
    forecast_result = generate_forecast(series, horizon=12, model_name="auto")

    analyst_report = build_analyst_report(
        clean=clean,
        report=report,
        stl_result=stl_result,
        merged_df=None,
        energy_result=None,
        changepoints=changepoints,
        anomalies=anomalies,
        forecast_result=forecast_result,
    )

    assert analyst_report.executive_summary
    assert analyst_report.overall_assessment
    assert len(analyst_report.findings) > 0
    # December 2024 is the known, previously-verified strongest anomaly in this real dataset.
    assert analyst_report.biggest_finding is not None
    assert "December 2024" in analyst_report.biggest_finding.narrative
    assert analyst_report.confidence["data_quality"].level == "High"  # 35 clean months, no gaps
    assert set(analyst_report.confidence) == {"data_quality", "weather_model", "forecast", "anomaly_detection"}
    # December 2024 was flagged by all 3 anomaly methods -> High confidence.
    assert analyst_report.confidence["anomaly_detection"].level == "High"
    assert analyst_report.largest_saving is None  # no weather model supplied in this test
    assert NO_SAVINGS_MESSAGE in analyst_report.executive_summary
    assert len(analyst_report.limitations) > 0
    assert len(analyst_report.monitoring_priorities) > 0
    # Every finding and recommendation carries its evidence -- traceability, not just conclusions.
    for finding in analyst_report.findings:
        assert finding.evidence
    for rec in analyst_report.recommendations:
        assert rec.evidence
