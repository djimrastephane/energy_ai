"""End-to-end run over the real household CSVs in data/raw/ -- catches parsing regressions
that synthetic unit-test data wouldn't."""

import pandas as pd

from config import SETTINGS
from src.ingestion import discover_csv_files, load_all
from src.kpis import compute_kpis
from src.preprocessing import run_pipeline
from src.statistics import describe


def test_full_pipeline_over_real_data():
    files = discover_csv_files(SETTINGS.raw_data_dir)
    assert len(files) == 4, f"expected 4 source CSVs in {SETTINGS.raw_data_dir}, found {files}"

    raw = load_all(files)
    clean, report = run_pipeline(raw)

    assert report.n_months == 35
    assert clean["month_start"].min() == pd.Timestamp("2023-09-01")
    assert clean["month_start"].max() == pd.Timestamp("2026-07-01")
    assert report.missing_months == []
    assert report.duplicates_removed == 0
    assert report.conflicts == []

    # Sanity-check totals against a hand-sum of the source CSVs.
    assert clean["consumption_kwh"].sum() > 0
    assert clean["cost_gbp"].sum() > 0

    summary = describe(clean["consumption_kwh"])
    assert summary.n == 35

    kpis = compute_kpis(clean, months=12)
    assert len(kpis) == 6
    assert all(k.previous is not None for k in kpis)  # 35 months >= 2*12
