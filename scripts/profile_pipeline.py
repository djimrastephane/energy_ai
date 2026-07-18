"""Repeatable performance profiling for the energy_ai pipeline (audit area 6).

Run from the repo root: .venv/bin/python scripts/profile_pipeline.py

Imports are deliberately interleaved with timing blocks (they are themselves
measurements), so import-order lint rules are disabled for this file.
"""

# ruff: noqa: E402, I001

import resource
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

TIMINGS: dict[str, float] = {}


def timed(label):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()

        def __exit__(self, *a):
            TIMINGS[label] = time.perf_counter() - self.t0

    return _T()


with timed("import: pandas+numpy"):
    import pandas as pd

with timed("import: statsmodels (STL)"):
    from src.decomposition import stl_decompose
with timed("import: ruptures (changepoints)"):
    from src.changepoints import detect_changepoints
with timed("import: sklearn (anomalies)"):
    from src.anomalies import detect_anomalies
with timed("import: prophet+xgboost+lightgbm (forecast registry)"):
    from src.forecast_evaluation import generate_forecast

from config import SETTINGS
from src.consultant import ConsultantContext
from src.consultant_router import QUESTIONS, route_question
from src.energy_signature import fit_energy_signature
from src.ingestion import FUEL_FILE_PATTERNS, discover_csv_files, load_all
from src.preprocessing import run_pipeline
from src.report import build_analyst_report
from src.weather import compute_monthly_degree_days, fetch_daily_temperature, merge_weather_with_consumption

# --- data load, per fuel ---
fuel_clean, fuel_report = {}, {}
with timed("load+validate all 3 fuels (12 CSVs)"):
    for fuel, pattern in FUEL_FILE_PATTERNS.items():
        files = discover_csv_files(SETTINGS.raw_data_dir, pattern=pattern)
        fuel_clean[fuel], fuel_report[fuel] = run_pipeline(load_all(files))

# --- per-rerun uncached work (what every widget interaction pays today) ---
fuel_stl, fuel_anom = {}, {}
with timed("stl_decompose x3 fuels"):
    for fuel, df in fuel_clean.items():
        fuel_stl[fuel] = stl_decompose(df)
with timed("detect_anomalies x3 fuels"):
    for fuel, df in fuel_clean.items():
        fuel_anom[fuel] = detect_anomalies(df, fuel_stl[fuel])
with timed("stl_decompose selected fuel (duplicate of above in main())"):
    _stl_sel = stl_decompose(fuel_clean["total"])
with timed("detect_changepoints (selected fuel)"):
    changepoints = detect_changepoints(_stl_sel.deseasonalized)
with timed("detect_anomalies selected fuel (duplicate)"):
    _anom_sel = detect_anomalies(fuel_clean["total"], _stl_sel)

# --- weather (disk cache present) ---
w = SETTINGS.weather
fuel_merged, fuel_energy = {}, {}
with timed("weather fetch(disk-cached)+degree-days+merge+fit x3 fuels"):
    for fuel, df in fuel_clean.items():
        start = df["month_start"].min()
        end = df["month_start"].max() + pd.offsets.MonthEnd(1)
        daily = fetch_daily_temperature(w.latitude, w.longitude, start, end, w.timezone, SETTINGS.weather_cache_dir)
        dd = compute_monthly_degree_days(daily, w.base_heat_c, w.base_cool_c)
        fuel_merged[fuel] = merge_weather_with_consumption(df, dd)
        fuel_energy[fuel] = fit_energy_signature(fuel_merged[fuel])

# --- forecast: full 8-model walk-forward CV (cold; st.cache_data hides this after 1st run) ---
series = fuel_clean["total"].set_index("month_start")["consumption_kwh"]
with timed("generate_forecast COLD (8-model walk-forward CV, h=12)"):
    forecast_12mo = generate_forecast(series, horizon=12, model_name="auto")

# --- report assembly (uncached by design) ---
with timed("build_analyst_report (full, all fuel dicts)"):
    analyst_report = build_analyst_report(
        fuel_clean["total"], fuel_report["total"], _stl_sel, fuel_merged["total"], fuel_energy["total"],
        changepoints, _anom_sel, forecast_12mo, "total", fuel_clean, fuel_stl, fuel_energy, fuel_anom,
    )

# --- consultant ---
ctx = ConsultantContext(
    analyst_report=analyst_report, clean=fuel_clean["total"], fuel_frames=fuel_clean,
    fuel_energy_results=fuel_energy, anomalies=_anom_sel, forecast_12mo=forecast_12mo,
    multi_fuel_forecasts=None, weather_enabled=True,
)
with timed("consultant: all 8 handlers"):
    for _q, handler in QUESTIONS:
        handler(ctx)
with timed("consultant: route_question x13 phrasings"):
    for p in ["why did my bill go up", "winter", "gas or electric", "forecast", "benchmark", "carbon",
              "last month", "save money", "gibberish", "", "co2", "typical household", "lower my bill"]:
        route_question(p, ctx)

# --- chart builders (all main figures, built once each) ---
sys.path.insert(0, "app")
from charts import annual_totals_bar, monthly_consumption_bar, rolling_average_line, year_over_year_overlay  # noqa: E402
from charts_fuel import fuel_comparison_bar, fuel_mix_annual_stacked_bar, fuel_share_area  # noqa: E402
from src.fuel import combine_fuel_frames  # noqa: E402

combined = combine_fuel_frames(fuel_clean["electricity"], fuel_clean["gas"])
with timed("7 core plotly figures built once"):
    monthly_consumption_bar(fuel_clean["total"])
    annual_totals_bar(fuel_clean["total"])
    year_over_year_overlay(fuel_clean["total"])
    rolling_average_line(fuel_clean["total"])
    fuel_comparison_bar(combined)
    fuel_share_area(combined)
    fuel_mix_annual_stacked_bar(combined)

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

print("\n=== BASELINE TIMINGS (seconds) ===")
for label, secs in TIMINGS.items():
    print(f"{secs:8.3f}  {label}")
rerun_cost = (
    TIMINGS["stl_decompose x3 fuels"] + TIMINGS["detect_anomalies x3 fuels"]
    + TIMINGS["stl_decompose selected fuel (duplicate of above in main())"]
    + TIMINGS["detect_changepoints (selected fuel)"]
    + TIMINGS["detect_anomalies selected fuel (duplicate)"]
    + TIMINGS["build_analyst_report (full, all fuel dicts)"]
)
print(f"\nEstimated UNCACHED work repeated on EVERY Streamlit rerun: {rerun_cost:.3f}s")
print(f"Peak RSS: {peak_mb:.0f} MB")
