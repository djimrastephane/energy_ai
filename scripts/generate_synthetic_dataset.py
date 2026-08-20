"""Generate a synthetic energy-provider-style dataset for public deployment.

The user's real household data in ``data/raw/`` stays there for local
testing; this script produces a *separate*, non-personal dataset in
``data/synthetic/`` with the exact same file naming and CSV schema, so it
can be dropped into ``data/raw/`` (or pointed at via ``RAW_DATA_DIR``) for
a public-facing deployment without exposing anyone's real usage.

Consumption is synthesized from a household "energy signature" model
(fixed base load + heating-degree-day slope + noise) rather than pure
noise, and driven by *real* Aberdeen weather (the location already fixed
in ``config.WeatherConfig``) fetched from Open-Meteo -- this keeps the
Weather Adjustment tab's regression meaningful in the demo (a strong,
realistic R-squared) while the consumption/cost numbers themselves are
entirely synthetic. Annual totals are tuned to sit close to the UK average
benchmarks already in ``config.BenchmarkConfig`` (2500 kWh electricity /
9500 kWh gas), so the Cost Intelligence tab's benchmarking shows a
sensible "about average" household by default.

A few deliberate, documented features make the demo worth exploring:
a permanent step-change in electricity (WFH/appliance story, for change-
point detection), a cold-snap gas spike and a holiday electricity dip
(for anomaly detection + the investigation checklist), a gentle multi-year
gas decline (loft insulation story, for the long-term trend view), and a
single tariff rise partway through (for the cost rate-vs-usage split).

Usage:  ./.venv/bin/python scripts/generate_synthetic_dataset.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SETTINGS
from src.energy_signature import fit_energy_signature
from src.fuel import cross_check_fuel_totals
from src.ingestion import load_all
from src.preprocessing import run_pipeline
from src.weather import (
    compute_monthly_degree_days,
    fetch_daily_weather,
    merge_weather_with_consumption,
)

SEED = 42
START = pd.Timestamp("2021-07-01")
END = pd.Timestamp("2026-07-01")  # inclusive; 61 complete months
OUT_DIR = SETTINGS.project_root / "data" / "synthetic"

# The story beats, chosen to fall well inside [START, END] with margin at
# both ends (STL/change-point detection need context on either side).
RATE_CHANGE_MONTH = pd.Timestamp("2023-10-01")  # UK energy price cap review month
STEP_CHANGE_MONTH = pd.Timestamp("2024-10-01")  # permanent +15% electricity step
GAS_ANOMALY_MONTH = pd.Timestamp("2023-01-01")  # cold-snap spike beyond weather-explained
ELEC_ANOMALY_MONTH = pd.Timestamp("2024-08-01")  # 3-week holiday, usage dip

# Annual targets close to config.BenchmarkConfig's UK averages.
ANNUAL_ELEC_KWH = 2500.0
ANNUAL_GAS_KWH = 9500.0
BASE_ELEC_KWH_PER_MONTH = 120.0  # weather-independent (lighting, appliances, standby)
BASE_GAS_KWH_PER_MONTH = 60.0  # weather-independent (hot water)
GAS_ANNUAL_DECLINE_PCT = 1.5  # loft-insulation-style gentle efficiency trend

ELEC_RATE_BEFORE, ELEC_RATE_AFTER = 0.24, 0.28  # GBP/kWh, ex VAT
GAS_RATE_BEFORE, GAS_RATE_AFTER = 0.055, 0.065  # GBP/kWh, ex VAT


def _fetch_hdd() -> pd.DataFrame:
    w = SETTINGS.weather
    daily = fetch_daily_weather(
        w.latitude, w.longitude, START, END + pd.offsets.MonthEnd(1), w.timezone, SETTINGS.weather_cache_dir
    )
    return compute_monthly_degree_days(daily, w.base_heat_c, w.base_cool_c)[["month_start", "hdd"]]


def _solve_slope(monthly_hdd: pd.Series, base_kwh: float, annual_target_kwh: float) -> float:
    """Least-squares-free direct solve: annual_target = 12*base + slope*sum(hdd)."""
    n_years = len(monthly_hdd) / 12
    return (annual_target_kwh * n_years - base_kwh * len(monthly_hdd)) / monthly_hdd.sum()


def build_household() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    hdd_df = _fetch_hdd()
    months = hdd_df["month_start"]
    hdd = hdd_df["hdd"]

    elec_slope = _solve_slope(hdd, BASE_ELEC_KWH_PER_MONTH, ANNUAL_ELEC_KWH) * 0.35  # weak: most of
    # electricity's variation is non-weather, matching a household with no air conditioning and
    # gas heating -- the remaining 65% of the target is absorbed into a slightly higher base load.
    elec_base = BASE_ELEC_KWH_PER_MONTH + (ANNUAL_ELEC_KWH - 12 * BASE_ELEC_KWH_PER_MONTH - elec_slope * hdd.sum()) / len(hdd)
    gas_slope = _solve_slope(hdd, BASE_GAS_KWH_PER_MONTH, ANNUAL_GAS_KWH)

    years_elapsed = (months - START).dt.days / 365.25
    gas_trend = (1 - GAS_ANNUAL_DECLINE_PCT / 100) ** years_elapsed
    elec_step = np.where(months >= STEP_CHANGE_MONTH, 1.15, 1.0)

    elec_kwh = (elec_base + elec_slope * hdd) * elec_step
    elec_kwh *= 1 + rng.normal(0, 0.02, len(months))
    elec_kwh = np.where(months == ELEC_ANOMALY_MONTH, elec_kwh * 0.62, elec_kwh)  # holiday dip

    gas_kwh = (BASE_GAS_KWH_PER_MONTH + gas_slope * hdd) * gas_trend
    gas_kwh *= 1 + rng.normal(0, 0.025, len(months))
    gas_kwh = np.where(months == GAS_ANOMALY_MONTH, gas_kwh * 1.45, gas_kwh)  # cold-snap spike

    elec_kwh = np.clip(elec_kwh, 40.0, None).round(2)
    gas_kwh = np.clip(gas_kwh, 20.0, None).round(2)

    elec_rate = np.where(months >= RATE_CHANGE_MONTH, ELEC_RATE_AFTER, ELEC_RATE_BEFORE)
    gas_rate = np.where(months >= RATE_CHANGE_MONTH, GAS_RATE_AFTER, GAS_RATE_BEFORE)
    elec_cost = (elec_kwh * elec_rate * (1 + rng.normal(0, 0.01, len(months)))).round(2)
    gas_cost = (gas_kwh * gas_rate * (1 + rng.normal(0, 0.01, len(months)))).round(2)

    return pd.DataFrame(
        {
            "month_start": months,
            "electricity_kwh": elec_kwh,
            "electricity_cost": elec_cost,
            "gas_kwh": gas_kwh,
            "gas_cost": gas_cost,
            "total_kwh": (elec_kwh + gas_kwh).round(2),
            "total_cost": (elec_cost + gas_cost).round(2),
        }
    )


def write_csvs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flavors = {
        "Total": ("total_kwh", "total_cost"),
        "Electricity": ("electricity_kwh", "electricity_cost"),
        "Gas": ("gas_kwh", "gas_cost"),
    }
    for label, (kwh_col, cost_col) in flavors.items():
        for year, year_df in df.groupby(df["month_start"].dt.year):
            path = OUT_DIR / f"{label} Use {year}.csv"
            lines = ["Month,Cost (£),Consumption (kWh)"]
            for _, row in year_df.iterrows():
                lines.append(f"{row['month_start']:%B %Y},{row[cost_col]:.2f},{row[kwh_col]:.2f}")
            path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {3 * df['month_start'].dt.year.nunique()} CSV file(s) to {OUT_DIR}")


def validate(df: pd.DataFrame) -> None:
    """Run the synthetic data through the real ingestion/preprocessing/weather pipeline --
    the same checks the deployed app itself performs -- and print a pass/fail summary."""
    from src.ingestion import FUEL_FILE_PATTERNS, discover_csv_files

    fuel_clean = {}
    for fuel, pattern in FUEL_FILE_PATTERNS.items():
        files = discover_csv_files(OUT_DIR, pattern=pattern)
        clean, report = run_pipeline(load_all(files))
        fuel_clean[fuel] = clean
        status = "clean" if report.is_clean else "HAS WARNINGS"
        print(f"  {fuel:<12} {report.n_months} months, {status}: {report.outlier_warnings or report.conflicts or report.missing_months}")

    warnings = cross_check_fuel_totals(fuel_clean["total"], fuel_clean["electricity"], fuel_clean["gas"])
    print(f"  cross-fuel check: {warnings if warnings else 'OK'}")

    w = SETTINGS.weather
    daily = fetch_daily_weather(w.latitude, w.longitude, START, END + pd.offsets.MonthEnd(1), w.timezone, SETTINGS.weather_cache_dir)
    dd = compute_monthly_degree_days(daily, w.base_heat_c, w.base_cool_c)
    for fuel in ("electricity", "gas"):
        merged = merge_weather_with_consumption(fuel_clean[fuel], dd)
        result = fit_energy_signature(merged)
        print(
            f"  {fuel} weather fit: R2={result.r_squared:.2f}, heating p={result.heating_pvalue:.4f}, "
            f"n={result.n_obs}"
        )


if __name__ == "__main__":
    household = build_household()
    write_csvs(household)
    print("\nValidating against the real ingestion/preprocessing/weather pipeline:")
    validate(household)
    print(
        "\nTo use for a public deployment: point config.RAW_DATA_DIR (or copy the files) at "
        "data/synthetic/ instead of data/raw/ -- the real data in data/raw/ is untouched."
    )
