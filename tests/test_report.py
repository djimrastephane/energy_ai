import warnings

import pandas as pd

from config import SETTINGS
from src.anomalies import detect_anomalies
from src.changepoints import detect_changepoints
from src.confidence import ConfidenceRating
from src.decomposition import stl_decompose
from src.energy_signature import EnergySignatureResult, fit_energy_signature
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
    fetch_daily_weather,
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
    assert "about the same energy" in text


def test_overall_assessment_states_percentage_increase():
    text = _overall_assessment(12.0, ConfidenceRating("High", "clean"))
    assert "12% more energy" in text
    assert "12%" in text


def test_overall_assessment_states_percentage_decrease():
    text = _overall_assessment(-8.0, ConfidenceRating("High", "clean"))
    assert "less energy" in text


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


def _energy_result(cooling_pvalue) -> EnergySignatureResult:
    return EnergySignatureResult(
        intercept=1.5,
        intercept_se=0.2,
        heating_slope=1.6,
        heating_se=0.1,
        heating_pvalue=0.001,
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=cooling_pvalue,
        r_squared=0.73,
        adj_r_squared=0.72,
        durbin_watson=2.0,
        n_obs=34,
        fitted=pd.Series(dtype=float),
        resid=pd.Series(dtype=float),
    )


def test_limitations_notes_no_cooling_load_when_cdd_inestimable():
    """Zero-variance CDD (a no-air-conditioning home in a cool climate, e.g. Aberdeen) makes
    the regression report cooling as inestimable (NaN p-value) -- the Limitations section must
    state that explicitly rather than leaving the absence of cooling analysis unexplained."""
    items = _limitations(
        n_months=36, report=_report(), energy_result=_energy_result(float("nan")), anomalies=[]
    )
    assert any("without air conditioning" in item for item in items)


def test_limitations_no_cooling_note_when_cooling_was_estimable():
    items = _limitations(
        n_months=36, report=_report(), energy_result=_energy_result(0.02), anomalies=[]
    )
    assert not any("air conditioning" in item for item in items)


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


def test_build_analyst_report_end_to_end_on_synthetic_data():
    warnings.filterwarnings("ignore")
    files = discover_csv_files(SETTINGS.synthetic_data_dir)
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
    # January 2023 (the generator's deliberate cold-snap gas spike) is the known,
    # previously-verified strongest anomaly in this synthetic dataset.
    assert analyst_report.biggest_finding is not None
    assert "January 2023" in analyst_report.biggest_finding.narrative
    # Data quality is High here: unlike the real household export this suite used to run
    # against, the bundled synthetic dataset's last row is already a complete month, so the
    # in-progress-month downgrade (audit finding F1) never fires in this specific check --
    # that behavior has its own dedicated unit tests (test_monthly_comparison.py,
    # test_preprocessing.py), so it isn't lost, just not re-exercised by this end-to-end test.
    assert analyst_report.confidence["data_quality"].level == "High"
    assert "clean history" in analyst_report.confidence["data_quality"].reason.lower()
    assert set(analyst_report.confidence) == {"data_quality", "weather_model", "forecast", "anomaly_detection"}
    # January 2023 was flagged by all 3 anomaly methods -> High confidence.
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


def test_build_analyst_report_household_sections_on_synthetic_data():
    """Synthetic-data end-to-end check for Task 10/11's household-level sections: fuel mix,
    weather-sensitivity attribution, largest cost driver, and per-fuel findings -- all built
    from the three bundled demo fuel exports, using the existing (unmodified) analysis functions."""
    warnings.filterwarnings("ignore")
    w = SETTINGS.weather

    fuel_clean, fuel_stl, fuel_anomalies, fuel_energy = {}, {}, {}, {}
    for fuel_key, pattern in FUEL_FILE_PATTERNS.items():
        files = discover_csv_files(SETTINGS.synthetic_data_dir, pattern=pattern)
        clean, _ = run_pipeline(load_all(files))
        fuel_clean[fuel_key] = clean
        stl_result = stl_decompose(clean)
        fuel_stl[fuel_key] = stl_result
        fuel_anomalies[fuel_key] = detect_anomalies(clean, stl_result)

        start = clean["month_start"].min()
        end = clean["month_start"].max() + pd.offsets.MonthEnd(1)
        daily = fetch_daily_weather(w.latitude, w.longitude, start, end, w.timezone, SETTINGS.weather_cache_dir)
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
    assert "Gas accounts for 84%" in analyst_report.fuel_mix_finding.narrative

    assert analyst_report.weather_sensitivity_finding is not None
    assert "gas accounts for 95%" in analyst_report.weather_sensitivity_finding.narrative
    assert "electricity accounts for 5%" in analyst_report.weather_sensitivity_finding.narrative

    assert analyst_report.largest_cost_driver is not None
    assert "Gas" in analyst_report.largest_cost_driver  # gas dominates both consumption and cost here

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
    files = discover_csv_files(SETTINGS.synthetic_data_dir)
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
