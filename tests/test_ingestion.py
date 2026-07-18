import pandas as pd
import pytest

from src.ingestion import (
    FUEL_FILE_PATTERNS,
    IngestionError,
    discover_csv_files,
    filter_sources_by_fuel,
    load_all,
    load_single_csv,
)


def _write_csv(path, rows: str) -> None:
    path.write_text("Month,Cost (£),Consumption (kWh)\n" + rows)


def test_load_single_csv_parses_month_and_values(tmp_path):
    path = tmp_path / "OVO Total Use 2024.csv"
    _write_csv(path, "January 2024,103.90,959.40\nFebruary 2024,56.15,360.06\n")

    df = load_single_csv(path)

    assert list(df.columns) == ["month_start", "cost_gbp", "consumption_kwh", "source_file"]
    assert len(df) == 2
    assert df.loc[0, "month_start"] == pd.Timestamp("2024-01-01")
    assert df.loc[0, "cost_gbp"] == pytest.approx(103.90)
    assert df.loc[0, "consumption_kwh"] == pytest.approx(959.40)
    assert df.loc[0, "source_file"] == "OVO Total Use 2024.csv"


def test_load_single_csv_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Month,Consumption (kWh)\nJanuary 2024,959.40\n")

    with pytest.raises(IngestionError, match="missing expected column"):
        load_single_csv(path)


def test_load_single_csv_malformed_month_raises(tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(path, "Somemonth 2024,10.0,5.0\n")

    with pytest.raises(IngestionError):
        load_single_csv(path)


def test_load_single_csv_non_numeric_value_raises(tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(path, "January 2024,oops,5.0\n")

    with pytest.raises(IngestionError, match="non-numeric"):
        load_single_csv(path)


def test_discover_csv_files_only_returns_csvs_sorted(tmp_path):
    (tmp_path / "OVO Total Use 2024.csv").write_text("x")
    (tmp_path / "OVO Total Use 2023.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    files = discover_csv_files(tmp_path)

    assert [f.name for f in files] == ["OVO Total Use 2023.csv", "OVO Total Use 2024.csv"]


def test_discover_csv_files_missing_dir_returns_empty(tmp_path):
    assert discover_csv_files(tmp_path / "does_not_exist") == []


def test_discover_csv_files_ignores_electricity_and_gas_breakdowns(tmp_path):
    # OVO also offers separate per-fuel "Electricity Use"/"Gas Use" exports that report the
    # same months as the "Total Use" files at finer granularity -- ingesting all three would
    # make every month look like a source conflict. Only "Total Use" files should be picked up.
    (tmp_path / "OVO Total Use 2023.csv").write_text("x")
    (tmp_path / "OVO Electricity Use 2023.csv").write_text("x")
    (tmp_path / "OVO Gas Use 2023.csv").write_text("x")

    files = discover_csv_files(tmp_path)

    assert [f.name for f in files] == ["OVO Total Use 2023.csv"]


def test_discover_csv_files_with_electricity_pattern(tmp_path):
    (tmp_path / "OVO Total Use 2023.csv").write_text("x")
    (tmp_path / "OVO Electricity Use 2023.csv").write_text("x")
    (tmp_path / "OVO Gas Use 2023.csv").write_text("x")

    files = discover_csv_files(tmp_path, pattern=FUEL_FILE_PATTERNS["electricity"])

    assert [f.name for f in files] == ["OVO Electricity Use 2023.csv"]


def test_filter_sources_by_fuel_on_paths(tmp_path):
    total = tmp_path / "OVO Total Use 2023.csv"
    elec = tmp_path / "OVO Electricity Use 2023.csv"
    gas = tmp_path / "OVO Gas Use 2023.csv"
    for p in (total, elec, gas):
        p.write_text("x")
    sources = [total, elec, gas]

    assert filter_sources_by_fuel(sources, "total") == [total]
    assert filter_sources_by_fuel(sources, "electricity") == [elec]
    assert filter_sources_by_fuel(sources, "gas") == [gas]


def test_filter_sources_by_fuel_on_uploaded_file_like_objects():
    class _FakeUpload:
        def __init__(self, name):
            self.name = name

    sources = [_FakeUpload("OVO Total Use 2024.csv"), _FakeUpload("OVO Gas Use 2024.csv")]

    filtered = filter_sources_by_fuel(sources, "gas")

    assert [s.name for s in filtered] == ["OVO Gas Use 2024.csv"]


def test_filter_sources_by_fuel_raises_on_unknown_fuel():
    with pytest.raises(KeyError):
        filter_sources_by_fuel([], "oil")


def test_load_all_combines_multiple_files_with_source_tagging(tmp_path):
    path_2023 = tmp_path / "OVO Total Use 2023.csv"
    path_2024 = tmp_path / "OVO Total Use 2024.csv"
    _write_csv(path_2023, "September 2023,34.66,180.25\n")
    _write_csv(path_2024, "January 2024,103.90,959.40\n")

    df = load_all([path_2023, path_2024])

    assert len(df) == 2
    assert set(df["source_file"]) == {"OVO Total Use 2023.csv", "OVO Total Use 2024.csv"}


def test_load_all_empty_sources_returns_empty_typed_frame():
    df = load_all([])
    assert df.empty
    assert list(df.columns) == ["month_start", "cost_gbp", "consumption_kwh", "source_file"]
