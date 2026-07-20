"""Build the downloadable "Household Energy Review" as one self-contained HTML file.

Pure rendering: every number comes from objects ``main()`` already computed
(``AnalystReport``, the per-fuel frames, the forecast) plus the same cited
constants the tabs use -- this module computes no new statistics. No
Streamlit imports, so it's unit-testable without ``AppTest``. Jinja2
autoescaping is on: all narrative/filename text is escaped, and only the
Plotly fragments this module itself generates are marked safe.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from plotly.graph_objects import Figure

from app.charts import energy_signature_scatter, monthly_consumption_bar, year_over_year_overlay
from app.charts_forecast import forecast_fan_chart
from app.charts_fuel import fuel_comparison_bar, fuel_mix_annual_stacked_bar
from config import SETTINGS
from src.benchmarking import compare_to_benchmark
from src.carbon import combined_annual_emissions
from src.energy_signature import EnergySignatureResult, partial_dependence
from src.forecast_evaluation import ForecastResult
from src.fuel import combine_fuel_frames
from src.ingestion import EnergyType
from src.kpis import compute_kpis
from src.preprocessing import PreprocessingReport
from src.recommendations import NO_RECOMMENDATIONS_MESSAGE
from src.report import AnalystReport
from src.utils import safe_divide

_TEMPLATE_DIR = Path(__file__).resolve().parent
_ENV = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=True)


def _chart_fragment(fig: Figure, include_js: bool) -> Markup:
    """Render a Plotly figure as an embeddable HTML fragment.

    The first chart in the document inlines plotly.js (~4 MB) so the file
    works fully offline; every later chart reuses that copy.
    """
    html = fig.to_html(full_html=False, include_plotlyjs="inline" if include_js else False)
    return Markup(f'<div class="chart">{html}</div>')  # noqa: S704 -- app-generated Plotly output, not user input


def _household_frame(fuel_frames: dict[EnergyType, pd.DataFrame]) -> pd.DataFrame:
    """The frame consumption charts/KPIs are drawn from: Total when present, else the
    largest available fuel frame (a Total-only or single-fuel dataset still gets a report)."""
    total = fuel_frames.get("total")
    if total is not None and not total.empty:
        return total
    non_empty = [df for df in fuel_frames.values() if not df.empty]
    return max(non_empty, key=len) if non_empty else pd.DataFrame()


def _headline_kpis(household_df: pd.DataFrame) -> list[dict[str, str]]:
    kpis = compute_kpis(household_df, months=12)
    out = []
    for kpi in kpis:
        if kpi.label == "Total consumption":
            out.append({"label": "Consumption (trailing 12 mo)", "value": f"{kpi.current:,.0f} kWh"})
        elif kpi.label == "Total cost":
            out.append({"label": "Energy cost (trailing 12 mo, excl. standing charges)", "value": f"£{kpi.current:,.0f}"})
    return out


def _forecast_narrative(forecast: ForecastResult | None, household_df: pd.DataFrame) -> str | None:
    if forecast is None or household_df.empty:
        return None
    unit_rate = safe_divide(float(household_df["cost_gbp"].sum()), float(household_df["consumption_kwh"].sum()))
    likely = float(forecast.p50.sum()) * unit_rate
    best = float(forecast.p10.sum()) * unit_rate
    worst = float(forecast.p90.sum()) * unit_rate
    return (
        f"Over the next 12 months, the {forecast.model_name} model (auto-selected by walk-forward "
        f"cross-validation) predicts a most-likely energy cost of £{likely:,.0f}, with a plausible "
        f"range of £{best:,.0f}-£{worst:,.0f} -- excluding standing charges, which aren't present "
        "in the billing exports."
    )


def _carbon_rows(fuel_frames: dict[EnergyType, pd.DataFrame]) -> list[dict[str, str]]:
    annual = combined_annual_emissions(fuel_frames)
    if annual.empty:
        return []
    return [
        {
            "year": f"{int(row['year'])}",
            "electricity": f"{row['electricity_kg_co2e']:,.0f}" if pd.notna(row["electricity_kg_co2e"]) else "n/a",
            "gas": f"{row['gas_kg_co2e']:,.0f}" if pd.notna(row["gas_kg_co2e"]) else "n/a",
            "combined_tonnes": f"{row['combined_tonnes_co2e']:.2f}",
        }
        for _, row in annual.iterrows()
    ]


def _benchmark_rows(
    fuel_frames: dict[EnergyType, pd.DataFrame], weather_enabled: bool
) -> list[dict[str, str]]:
    rows = []
    for fuel in ("electricity", "gas"):
        df = fuel_frames.get(fuel)
        if df is None or df.empty:
            continue
        annualized_kwh = float(df["consumption_kwh"].sum()) / len(df) * 12
        result = compare_to_benchmark(annualized_kwh, fuel, "uk", n_months=len(df), weather_adjusted=weather_enabled)
        if result is None:
            continue
        rows.append(
            {
                "fuel": fuel.capitalize(),
                "annual_kwh": f"{result.annual_kwh:,.0f} kWh/yr",
                "reference_kwh": f"{result.reference_kwh:,.0f} kWh/yr",
                "band": result.band,
                "source": result.source,
            }
        )
    return rows


def build_household_report_html(
    analyst_report: AnalystReport,
    fuel_frames: dict[EnergyType, pd.DataFrame],
    fuel_merged: dict[EnergyType, pd.DataFrame],
    fuel_energy_results: dict[EnergyType, EnergySignatureResult],
    forecast_12mo: ForecastResult | None,
    weather_enabled: bool,
    preprocessing_report: PreprocessingReport,
) -> str:
    """Assemble the full report. Returns the complete HTML document as a string."""
    household_df = _household_frame(fuel_frames)
    if household_df.empty:
        raise ValueError("No data available to build a report from.")

    start, end = household_df["month_start"].min(), household_df["month_start"].max()
    period = f"{start.year}–{end.year}" if start.year != end.year else f"{start.year}"
    coverage = f"{start.strftime('%B %Y')} to {end.strftime('%B %Y')}"
    fuels_present = ", ".join(
        f.capitalize() for f in ("total", "electricity", "gas") if not fuel_frames.get(f, pd.DataFrame()).empty
    )

    # Charts -- first fragment carries the inline plotly.js payload.
    monthly_chart = _chart_fragment(monthly_consumption_bar(household_df), include_js=True)
    yoy_chart = _chart_fragment(year_over_year_overlay(household_df), include_js=False)

    fuel_mix_chart = fuel_annual_chart = Markup("")
    fuel_mix_narrative = None
    elec_df, gas_df = fuel_frames.get("electricity"), fuel_frames.get("gas")
    if (
        analyst_report.fuel_mix_finding is not None
        and elec_df is not None
        and gas_df is not None
        and not elec_df.empty
        and not gas_df.empty
    ):
        combined = combine_fuel_frames(elec_df, gas_df)
        fuel_mix_narrative = analyst_report.fuel_mix_finding.narrative
        fuel_mix_chart = _chart_fragment(fuel_comparison_bar(combined), include_js=False)
        fuel_annual_chart = _chart_fragment(fuel_mix_annual_stacked_bar(combined), include_js=False)

    weather_narrative = None
    weather_chart = Markup("")
    if weather_enabled:
        weather_finding = analyst_report.weather_sensitivity_finding or next(
            (f for f in analyst_report.findings if f.category == "weather"), None
        )
        if weather_finding is not None:
            weather_narrative = weather_finding.narrative
        # Energy-signature scatter for the household frame's fuel, when its fit exists.
        for chart_fuel in ("total", "electricity", "gas"):
            merged = fuel_merged.get(chart_fuel)
            result = fuel_energy_results.get(chart_fuel)
            if merged is not None and result is not None and not merged.empty:
                pdp = partial_dependence(result, merged, "avg_daily_hdd")
                weather_chart = _chart_fragment(
                    energy_signature_scatter(merged, pdp, "avg_daily_hdd"), include_js=False
                )
                break

    forecast_narrative = _forecast_narrative(forecast_12mo, household_df)
    forecast_chart = (
        _chart_fragment(forecast_fan_chart(household_df, forecast_12mo), include_js=False)
        if forecast_12mo is not None
        else Markup("")
    )

    data_warnings = list(preprocessing_report.outlier_warnings)
    if preprocessing_report.missing_months:
        data_warnings.append(
            f"{len(preprocessing_report.missing_months)} month(s) are missing from the billing data."
        )

    template = _ENV.get_template("report_template.html")
    return template.render(
        period=period,
        generated_date=date.today().strftime("%d %B %Y"),
        n_months=len(household_df),
        coverage=coverage,
        fuels_present=fuels_present,
        headline_kpis=_headline_kpis(household_df),
        executive_summary=analyst_report.executive_summary,
        largest_cost_driver=analyst_report.largest_cost_driver,
        data_warnings=data_warnings,
        findings=analyst_report.findings,
        fuel_mix_narrative=fuel_mix_narrative,
        fuel_mix_chart=fuel_mix_chart,
        fuel_annual_chart=fuel_annual_chart,
        weather_narrative=weather_narrative,
        weather_vs_behavioural=analyst_report.weather_vs_behavioural_summary,
        weather_chart=weather_chart,
        monthly_chart=monthly_chart,
        yoy_chart=yoy_chart,
        forecast_narrative=forecast_narrative,
        forecast_chart=forecast_chart,
        recommendations=analyst_report.recommendations,
        no_recommendations_message=NO_RECOMMENDATIONS_MESSAGE,
        carbon_rows=_carbon_rows(fuel_frames),
        carbon_factors_note=(
            f"Factors: electricity {SETTINGS.carbon.electricity_kg_co2e_per_kwh} kgCO₂e/kWh, gas "
            f"{SETTINGS.carbon.gas_kg_co2e_per_kwh} kgCO₂e/kWh (UK Government DESNZ/DEFRA GHG "
            "Conversion Factors; the electricity factor declines as the grid decarbonises, so "
            "recent years' electricity emissions are, if anything, modestly overstated)."
        ),
        benchmark_rows=_benchmark_rows(fuel_frames, weather_enabled),
        limitations=analyst_report.limitations,
        monitoring_priorities=analyst_report.monitoring_priorities,
        confidence_rows=[
            {"area": key.replace("_", " ").capitalize(), "level": rating.level, "reason": rating.reason}
            for key, rating in analyst_report.confidence.items()
        ],
    )
