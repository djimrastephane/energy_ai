"""Real-data verification of the month-comparison journey.

Runs the exact objects the app builds (per-fuel comparisons, contributions,
narrative) against the real exports in data/raw/ for the scenario list from
the month-comparison design brief, and prints every value a human needs to
sanity-check the wording. Weather comes from the on-disk cache (no network
needed after the app's first fetch).

Usage:  ./.venv/bin/python scripts/verify_month_comparison.py
"""

from __future__ import annotations

import pandas as pd

from config import SETTINGS
from src.anomalies import detect_anomalies
from src.decomposition import stl_decompose
from src.ingestion import FUEL_FILE_PATTERNS, discover_csv_files, load_all
from src.monthly_comparison import (
    build_monthly_comparison,
    fuel_contributions,
    in_progress_month,
    latest_complete_month,
    month_to_date_note,
)
from src.monthly_narrative import build_monthly_comparison_narrative
from src.preprocessing import run_pipeline
from src.weather import (
    compute_monthly_degree_days,
    fetch_daily_weather,
    merge_weather_with_consumption,
)


def _fmt(value: float | None, unit: str = "", signed: bool = False) -> str:
    if value is None:
        return "n/a"
    spec = "+,.1f" if signed else ",.1f"
    return f"{value:{spec}}{unit}"


def main() -> None:
    frames = {}
    for fuel, pattern in FUEL_FILE_PATTERNS.items():
        files = discover_csv_files(SETTINGS.raw_data_dir, pattern=pattern)
        frames[fuel], _ = run_pipeline(load_all(files))
    clean = frames["total"]

    daily = fetch_daily_weather(
        SETTINGS.weather.latitude,
        SETTINGS.weather.longitude,
        clean["month_start"].min().strftime("%Y-%m-%d"),
        (clean["month_start"].max() + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
        SETTINGS.weather.timezone,
        SETTINGS.processed_data_dir,
    )
    degree_days = compute_monthly_degree_days(
        daily, SETTINGS.weather.base_heat_c, SETTINGS.weather.base_cool_c
    )
    merged = {f: merge_weather_with_consumption(frames[f], degree_days) for f in frames}

    from src.energy_signature import fit_energy_signature

    energy = {f: fit_energy_signature(merged[f]) for f in frames}
    stl = {f: stl_decompose(frames[f]) for f in frames}
    anomalies = {f: detect_anomalies(frames[f], stl[f]) for f in frames}

    latest = latest_complete_month(clean)
    partial = in_progress_month(clean)

    scenarios: list[tuple[str, pd.Timestamp]] = [
        ("Latest complete month", latest),
        ("December 2024 (known anomaly)", pd.Timestamp("2024-12-01")),
        ("Normal winter month", pd.Timestamp("2025-01-01")),
        ("Summer month", pd.Timestamp("2025-07-01")),
        ("Lower year-on-year usage", pd.Timestamp("2026-01-01")),
        ("Higher year-on-year usage", pd.Timestamp("2026-04-01")),
    ]

    for title, month in scenarios:
        comparisons = {
            f: build_monthly_comparison(
                frames[f], f, "same_month_last_year", month, merged[f], energy[f]
            )
            for f in ("total", "electricity", "gas")
        }
        total = comparisons["total"]
        if total is None:
            print("=" * 78)
            print(f"{title}: no comparison available for {month.strftime('%B %Y')}")
            continue
        contributions = fuel_contributions(
            comparisons["total"], comparisons["electricity"], comparisons["gas"]
        )
        narrative = build_monthly_comparison_narrative(total, contributions, weather_enabled=True)
        month_anomaly = next((a for a in anomalies["total"] if a.date == month), None)

        print("=" * 78)
        print(f"{title}: {month.strftime('%B %Y')}")
        print(
            f"  Comparison month:        "
            f"{total.comparison_month.strftime('%B %Y') if total.comparison_month is not None else 'n/a'}"
        )
        for fuel_name, comparison in (("Gas", comparisons["gas"]), ("Electricity", comparisons["electricity"]), ("Combined", total)):
            if comparison is None:
                print(f"  {fuel_name + ' usage:':<24} n/a")
                continue
            pct = f" ({comparison.percentage_change:+.1f}%)" if comparison.percentage_change is not None else ""
            print(
                f"  {fuel_name + ' usage:':<24} {_fmt(comparison.current_consumption_kwh, ' kWh')} "
                f"vs {_fmt(comparison.comparison_consumption_kwh, ' kWh')}{pct}"
            )
        print(
            f"  HDD (weather):           {_fmt(total.current_heating_degree_days)} vs "
            f"{_fmt(total.comparison_heating_degree_days)}"
        )
        print(f"  Weather-explained:       {_fmt(total.weather_explained_change_kwh, ' kWh', signed=True)}")
        print(
            f"  Unexplained:             {_fmt(total.unexplained_change_kwh, ' kWh', signed=True)}"
            f" (meaningful: {total.unexplained_change_is_meaningful})"
        )
        print(f"  Cost difference:         {_fmt(total.cost_change_gbp, ' GBP', signed=True)}")
        if month_anomaly is not None:
            print(
                f"  Anomaly status:          {month_anomaly.direction}, "
                f"{len(month_anomaly.methods)}/3 methods ({', '.join(month_anomaly.methods)})"
            )
        else:
            print("  Anomaly status:          not flagged")
        print(f"  Judgement:               {total.judgement} ({total.change_category})")
        print(f"  Narrative:               {narrative.narrative}")
        print(f"  Recommendation:          {narrative.action}")
        print(f"  Confidence:              {narrative.confidence} -- {narrative.confidence_reason}")
        print(f"  Limitation:              {narrative.limitation}")

    print("=" * 78)
    print(f"Latest partial month: {partial.strftime('%B %Y') if partial is not None else 'none'}")
    print(f"  Notice: {month_to_date_note(clean)}")
    if partial is not None:
        row = clean[clean["month_start"] == partial].iloc[0]
        print(f"  Month-to-date: {row['consumption_kwh']:,.1f} kWh, £{row['cost_gbp']:,.2f}")
        mtd = build_monthly_comparison(clean, "total", "same_month_last_year", partial)
        if mtd is None:
            print("  No month-to-date comparison available.")
        else:
            print(f"  data_complete flag: {mtd.data_complete} | confidence: {mtd.confidence}")
            print(f"  Confidence reason: {mtd.confidence_reason}")


if __name__ == "__main__":
    main()
