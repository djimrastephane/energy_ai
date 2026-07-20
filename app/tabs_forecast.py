"""Forecasting tab renderer plus the cached forecast generators.

Split out to match the ``tabs_core.py``/``tabs_drivers.py`` pattern; the
Unusual Months renderer lives in ``tabs_anomalies.py``. The cached
generators here are shared by the Forecasting tab, the Executive Briefing,
the Cost Intelligence tab, and the Comparisons tab.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.charts_forecast import forecast_fan_chart, model_comparison_bar
from config import SETTINGS, BillingConfig
from src.billing import standing_charge_for_months
from src.confidence import rate_forecast
from src.forecast_evaluation import ForecastResult, generate_forecast
from src.ingestion import EnergyType
from src.monthly_comparison import is_month_complete


@st.cache_data(show_spinner="Cross-validating forecasting models...")
def generate_forecast_cached(clean: pd.DataFrame, horizon: int, model_name: str) -> ForecastResult:
    """Cached wrapper around ``src.forecast_evaluation.generate_forecast``.

    No network call is involved (unlike ``tabs_drivers``'s weather fetch) -- this is
    purely CPU-bound and fast enough (benchmarked well under 10s for the
    full 8-model comparison on the real data) to run automatically, but is
    still cached since Streamlit reruns the whole script on every widget
    interaction.
    """
    series = clean.set_index("month_start")["consumption_kwh"]
    return generate_forecast(series, horizon=horizon, model_name=model_name)


@st.cache_data(show_spinner="Running the full forecast model suite for Electricity, Gas, and Total...")
def generate_multi_fuel_forecast_cached(
    fuel_clean_dfs: dict[EnergyType, pd.DataFrame], horizon: int = 12
) -> dict[EnergyType, ForecastResult]:
    """Run the same cross-validated forecast once per fuel -- opt-in (behind a button in the
    Cost Intelligence and Comparisons tabs) since this triples the walk-forward-CV cost of the
    single-fuel Forecasting tab. A fuel with insufficient history simply doesn't appear in the
    result rather than failing the whole comparison.
    """
    results: dict[EnergyType, ForecastResult] = {}
    for fuel, df in fuel_clean_dfs.items():
        if df.empty:
            continue
        try:
            results[fuel] = generate_forecast_cached(df, horizon, "auto")
        except ValueError:
            continue
    return results


def _trailing_actual_kwh(clean: pd.DataFrame, months: int) -> float | None:
    """Sum of the last ``months`` *complete* months of actuals -- the in-progress month is
    excluded so the forecast isn't compared against a period with a month-to-date stub."""
    complete = clean[clean["month_start"].apply(is_month_complete)].sort_values("month_start")
    if len(complete) < months:
        return None
    return float(complete.tail(months)["consumption_kwh"].sum())


def _seasonal_expectation(result: ForecastResult, month_nums: set[int]) -> float | None:
    """Expected (P50) total over the forecast months whose calendar month is in ``month_nums``,
    or None if the horizon doesn't include all of them."""
    dates = list(result.forecast_dates)
    matching = [i for i, d in enumerate(dates) if d.month in month_nums]
    if len({dates[i].month for i in matching}) < len(month_nums):
        return None
    return float(sum(result.p50[i] for i in matching))


def render_forecasting(
    clean: pd.DataFrame,
    horizon: int,
    model_choice: str,
    fuel: EnergyType = "total",
    billing_config: BillingConfig | None = None,
) -> None:
    billing_config = billing_config or SETTINGS.billing
    model_name = "auto" if model_choice == "Auto (best by CV)" else model_choice

    try:
        result = generate_forecast_cached(clean, horizon, model_name)
    except ValueError as exc:
        st.warning(str(exc))
        return

    # Answer first: what to expect, how it compares with the last year, and how much to
    # trust it -- the model that produced it is a detail, kept in the expander below.
    expected_kwh = float(result.p50.sum())
    unit_rate = clean["cost_gbp"].sum() / clean["consumption_kwh"].sum()
    rating = rate_forecast(result, float(clean["consumption_kwh"].mean()))
    trailing_kwh = _trailing_actual_kwh(clean, horizon)

    st.subheader(f"What should I expect over the next {horizon} months?")
    delta = None
    if trailing_kwh is not None and trailing_kwh > 0:
        change_pct = (expected_kwh - trailing_kwh) / trailing_kwh * 100
        delta = f"{change_pct:+.0f}% vs the last {horizon} complete months ({trailing_kwh:,.0f} kWh)"
    c1, c2 = st.columns([1.4, 1])
    c1.metric(
        "Expected energy use",
        f"≈ {expected_kwh:,.0f} kWh",
        delta,
        delta_color="off",
        help=(
            "The forecast's central (P50) estimate, produced by the best-performing model in "
            "cross-validation -- see Model details below."
        ),
    )
    c2.metric(
        "Forecast confidence",
        rating.level,
        help=rating.reason,
    )
    standing = standing_charge_for_months(list(result.forecast_dates), fuel, billing_config)
    vat_rate = billing_config.vat_rate
    total_bill = (expected_kwh * unit_rate + standing) * (1 + vat_rate)
    st.caption(
        f"≈ £{expected_kwh * unit_rate:,.0f} consumption at your average rate, plus "
        f"≈ £{standing:,.0f} standing charges and {vat_rate:.0%} VAT ⇒ estimated total bill "
        f"≈ £{total_bill:,.0f}. Confidence is {rating.level.lower()}: {rating.reason}"
    )

    # Seasonal expectations communicate most of the practical value without leaning on
    # the wide 12-month interval.
    winter_kwh = _seasonal_expectation(result, {12, 1, 2})
    summer_kwh = _seasonal_expectation(result, {6, 7, 8})
    if winter_kwh is not None or summer_kwh is not None:
        cols = st.columns(2)
        if winter_kwh is not None:
            cols[0].metric(
                "Expected winter months (Dec-Feb)",
                f"≈ {winter_kwh:,.0f} kWh",
                help="Sum of the expected (P50) values for the forecast's December-February months.",
            )
        if summer_kwh is not None:
            cols[1].metric(
                "Expected summer months (Jun-Aug)",
                f"≈ {summer_kwh:,.0f} kWh",
                help="Sum of the expected (P50) values for the forecast's June-August months.",
            )

    st.plotly_chart(forecast_fan_chart(clean, result), width="stretch")
    st.caption(
        "The forecast assumes your future use resembles previous years and that nothing major "
        "changes -- no heat pump, no change in occupancy or tariff, no new large appliances. "
        "It cannot anticipate one-off events."
    )

    # The range: bounds on plausibility, not three equally likely outcomes.
    total_lower = float(result.p10.sum())
    total_upper = float(result.p90.sum())
    with st.expander(f"Plausible range for the {horizon}-month total"):
        tooltip = (
            "From resampling the model's own past forecast errors, scaled up the further ahead "
            "the month is. These are bounds on what's plausible -- not equally likely outcomes; "
            "values near the expected estimate are more likely than values near either bound. "
            "£ figures are consumption cost only -- the headline caption above adds the "
            "standing-charge and VAT estimate."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Lower estimate", f"{total_lower:,.0f} kWh", f"£{total_lower * unit_rate:,.0f}", help=tooltip)
        c2.metric("Expected estimate", f"{expected_kwh:,.0f} kWh", f"£{expected_kwh * unit_rate:,.0f}", help=tooltip)
        c3.metric("Upper estimate", f"{total_upper:,.0f} kWh", f"£{total_upper * unit_rate:,.0f}", help=tooltip)
        st.caption(
            "The range is wide because ~35 monthly observations can't pin down a 12-month total "
            "tightly -- that's the honest width, not a hedge."
        )

    with st.expander("Model details"):
        selection_note = "auto-selected by cross-validation" if model_choice == "Auto (best by CV)" else "forced via the sidebar"
        st.markdown(f"**Selected model: {result.model_name}** ({selection_note})")
        st.caption(
            "Uses full history regardless of the sidebar year filter. Cross-validation is "
            "1-step-ahead walk-forward, not multi-step -- ~35 months of data doesn't support "
            "reliable multi-step CV at longer horizons, so this only validates 1-month-ahead "
            "accuracy. The chart projects the chosen model to the full horizon; that projection "
            "isn't itself cross-validated at longer lead times."
        )
        st.plotly_chart(model_comparison_bar(result.comparison, result.model_name), width="stretch")
        display = result.comparison.rename(
            columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)", "n_folds": "CV folds"}
        ).round(1)
        st.dataframe(display, hide_index=True, width="stretch")
