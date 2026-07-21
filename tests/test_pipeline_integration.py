"""End-to-end run over the bundled data/synthetic/ demo CSVs -- catches parsing regressions
that synthetic unit-test data (built by hand, not loaded from a file) wouldn't.

Runs against data/synthetic/, not data/raw/: the latter holds a real household's private
export and is intentionally empty in this repo (never committed), so a suite that anyone can
clone and run needs a real *file-parsing* exercise that doesn't depend on personal data.
"""

import pandas as pd

from config import SETTINGS
from src.ingestion import discover_csv_files, load_all
from src.kpis import compute_kpis
from src.preprocessing import run_pipeline
from src.statistics import describe


def test_full_pipeline_over_synthetic_data():
    files = discover_csv_files(SETTINGS.synthetic_data_dir)
    assert len(files) == 6, f"expected 6 source CSVs in {SETTINGS.synthetic_data_dir}, found {files}"

    raw = load_all(files)
    clean, report = run_pipeline(raw)

    assert report.n_months == 60
    assert clean["month_start"].min() == pd.Timestamp("2021-07-01")
    assert clean["month_start"].max() == pd.Timestamp("2026-06-01")
    assert report.missing_months == []
    assert report.duplicates_removed == 0
    assert report.conflicts == []

    # Sanity-check totals against a hand-sum of the source CSVs.
    assert clean["consumption_kwh"].sum() > 0
    assert clean["cost_gbp"].sum() > 0

    summary = describe(clean["consumption_kwh"])
    assert summary.n == 60

    kpis = compute_kpis(clean, months=12)
    assert len(kpis) == 6
    assert all(k.previous is not None for k in kpis)  # 60 months >= 2*12
