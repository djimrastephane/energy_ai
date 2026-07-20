"""Direct unit tests for app.sidebar's data-loading logic -- no full AppTest render needed."""

import pandas as pd

from app import sidebar
from src.preprocessing import PreprocessingReport


def _report(n_files, source_files, idx) -> PreprocessingReport:
    date_range = (idx.min(), idx.max()) if len(idx) else None
    return PreprocessingReport(
        n_files_loaded=n_files,
        source_files=source_files,
        date_range=date_range,
        n_months=len(idx),
        missing_months=[],
        duplicates_removed=0,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["month_start", "cost_gbp", "consumption_kwh"])


def test_load_all_fuels_infers_total_when_only_electricity_and_gas_exist(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    elec = pd.DataFrame({"month_start": idx, "consumption_kwh": [100.0, 110.0, 120.0], "cost_gbp": [30.0, 33.0, 36.0]})
    gas = pd.DataFrame({"month_start": idx, "consumption_kwh": [500.0, 300.0, 100.0], "cost_gbp": [50.0, 30.0, 10.0]})

    def _fake_load_default(pattern, raw_dir=None):
        if "Total" in pattern:
            return _empty_frame(), _report(0, [], pd.DatetimeIndex([]))
        if "Electricity" in pattern:
            return elec, _report(1, ["Electricity Use 2024.csv"], idx)
        if "Gas" in pattern:
            return gas, _report(1, ["Gas Use 2024.csv"], idx)
        raise AssertionError(f"unexpected pattern {pattern!r}")

    monkeypatch.setattr(sidebar, "_load_default_data", _fake_load_default)

    frames, reports = sidebar._load_all_fuels(uploaded=None)

    assert not frames["total"].empty
    assert list(frames["total"]["consumption_kwh"]) == [600.0, 410.0, 220.0]
    assert reports["total"].n_files_loaded == 0
    assert reports["total"].source_files == ["inferred: Electricity Use + Gas Use"]


def test_load_all_fuels_leaves_total_empty_when_only_electricity_exists(monkeypatch):
    """No inference with only one fuel -- summing needs both sides."""
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    elec = pd.DataFrame({"month_start": idx, "consumption_kwh": [100.0, 110.0, 120.0], "cost_gbp": [30.0, 33.0, 36.0]})

    def _fake_load_default(pattern, raw_dir=None):
        if "Electricity" in pattern:
            return elec, _report(1, ["Electricity Use 2024.csv"], idx)
        return _empty_frame(), _report(0, [], pd.DatetimeIndex([]))

    monkeypatch.setattr(sidebar, "_load_default_data", _fake_load_default)

    frames, _reports = sidebar._load_all_fuels(uploaded=None)

    assert frames["total"].empty


def test_load_all_fuels_does_not_override_a_real_total_file(monkeypatch):
    """A real Total Use export always wins -- inference only fills a genuine gap."""
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    elec = pd.DataFrame({"month_start": idx, "consumption_kwh": [100.0, 110.0, 120.0], "cost_gbp": [30.0, 33.0, 36.0]})
    gas = pd.DataFrame({"month_start": idx, "consumption_kwh": [500.0, 300.0, 100.0], "cost_gbp": [50.0, 30.0, 10.0]})
    real_total = pd.DataFrame({"month_start": idx, "consumption_kwh": [601.0, 411.0, 221.0], "cost_gbp": [80.5, 63.5, 46.5]})

    def _fake_load_default(pattern, raw_dir=None):
        if "Total" in pattern:
            return real_total, _report(1, ["Total Use 2024.csv"], idx)
        if "Electricity" in pattern:
            return elec, _report(1, ["Electricity Use 2024.csv"], idx)
        return gas, _report(1, ["Gas Use 2024.csv"], idx)

    monkeypatch.setattr(sidebar, "_load_default_data", _fake_load_default)

    frames, reports = sidebar._load_all_fuels(uploaded=None)

    assert list(frames["total"]["consumption_kwh"]) == [601.0, 411.0, 221.0]
    assert reports["total"].source_files == ["Total Use 2024.csv"]
