import warnings

import pandas as pd

from config import SETTINGS
from src.anomalies import detect_anomalies
from src.changepoints import detect_changepoints
from src.confidence import ConfidenceRating
from src.decomposition import stl_decompose
from src.energy_signature import fit_energy_signature
from src.forecast_evaluation import generate_forecast
from src.ingestion import FUEL_FILE_PATTERNS, discover_csv_files, load_all
from src.preprocessing import PreprocessingReport, run_pipeline
from src.report import (
    NO_SAVINGS_MESSAGE,
    _limitations,
    _monitoring_priorities,
    _overall_assessment,
    build_analyst_report,
)
from src.weather import (
    compute_monthly_degree_days,
    fetch_daily_temperature,
    merge_weather_with_consumption,
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
    # Data quality is Medium (not High) on the real data because the latest month (the
    # current calendar month) is flagged as possibly month-to-date -- an intentional,
    # honest downgrade added by the audit (finding F1): trailing-window KPIs include a
    # month whose figures may still be accumulating.
    assert analyst_report.confidence["data_quality"].level == "Medium"
    assert "warning" in analyst_report.confidence["data_quality"].reason.lower()
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


def test_build_analyst_report_household_sections_on_real_data():
    """Real-data end-to-end check for Task 10/11's household-level sections: fuel mix,
    weather-sensitivity attribution, largest cost driver, and per-fuel findings -- all built
    from the three real fuel exports, using the existing (unmodified) analysis functions."""
    warnings.filterwarnings("ignore")
    w = SETTINGS.weather

    fuel_clean, fuel_stl, fuel_anomalies, fuel_energy = {}, {}, {}, {}
    for fuel_key, pattern in FUEL_FILE_PATTERNS.items():
        files = discover_csv_files(SETTINGS.raw_data_dir, pattern=pattern)
        clean, _ = run_pipeline(load_all(files))
        fuel_clean[fuel_key] = clean
        stl_result = stl_decompose(clean)
        fuel_stl[fuel_key] = stl_result
        fuel_anomalies[fuel_key] = detect_anomalies(clean, stl_result)

        start = clean["month_start"].min()
        end = clean["month_start"].max() + pd.offsets.MonthEnd(1)
        daily = fetch_daily_temperature(w.latitude, w.longitude, start, end, w.timezone, SETTINGS.weather_cache_dir)
        monthly_dd = compute_monthly_degree_days(daily, w.base_heat_c, w.base_cool_c)
        merged = merge_weather_with_consumption(clean, monthly_dd)
        fuel_energy[fuel_key] = fit_energy_signature(merged)

    clean = fuel_clean["total"]
    report = _report()
    stl_result = fuel_stl["total"]
    changepoints = detect_changepoints(stl_result.deseasonalized)

    analyst_report = build_analyst_report(
        clean=clean,
        report=report,
        stl_result=stl_result,
        merged_df=None,
        energy_result=fuel_energy["total"],
        changepoints=changepoints,
        anomalies=fuel_anomalies["total"],
        forecast_result=None,
        fuel="total",
        fuel_clean_dfs=fuel_clean,
        fuel_stl_results=fuel_stl,
        fuel_energy_results=fuel_energy,
        fuel_anomalies=fuel_anomalies,
    )

    assert analyst_report.fuel_mix_finding is not None
    assert "Gas accounts for 65%" in analyst_report.fuel_mix_finding.narrative

    assert analyst_report.weather_sensitivity_finding is not None
    assert "gas accounts for 89%" in analyst_report.weather_sensitivity_finding.narrative
    assert "electricity accounts for 11%" in analyst_report.weather_sensitivity_finding.narrative

    assert analyst_report.largest_cost_driver is not None
    assert "Electricity" in analyst_report.largest_cost_driver  # gas is cheaper per kWh, so costs less overall

    assert set(analyst_report.per_fuel_findings) == {"electricity", "gas"}
    assert len(analyst_report.per_fuel_findings["electricity"]) > 0
    assert len(analyst_report.per_fuel_findings["gas"]) > 0

    # The fuel-focus recommendation should fire: gas dominates both consumption and weather
    # sensitivity well past the 60% threshold.
    fuel_focus = next((r for r in analyst_report.recommendations if r.title == "Focus on gas"), None)
    assert fuel_focus is not None
    assert fuel_focus.confidence == "High"


def test_build_analyst_report_without_fuel_dicts_leaves_household_sections_none():
    """Omitting the fuel_* dicts (the pre-existing call signature) must produce the same
    single-fuel report as before -- these Task 10/11 additions are opt-in, not required."""
    warnings.filterwarnings("ignore")
    files = discover_csv_files(SETTINGS.raw_data_dir)
    clean, report = run_pipeline(load_all(files))
    stl_result = stl_decompose(clean)

    analyst_report = build_analyst_report(
        clean=clean,
        report=report,
        stl_result=stl_result,
        merged_df=None,
        energy_result=None,
        changepoints=[],
        anomalies=[],
        forecast_result=None,
    )

    assert analyst_report.fuel_mix_finding is None
    assert analyst_report.weather_sensitivity_finding is None
    assert analyst_report.largest_cost_driver is None
    assert analyst_report.weather_vs_behavioural_summary is None
    assert analyst_report.per_fuel_findings == {}
