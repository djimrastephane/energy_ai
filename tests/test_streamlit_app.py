"""Smoke and journey tests: the Streamlit app must render with real data and no uncaught
exceptions, with the month-comparison journey as the primary surface.

Navigation is organized around user questions; ``st.tabs`` nests one level,
and AppTest flattens nested tabs into ``at.tabs`` in creation order -- so
tabs are looked up by label here, never by index.

"Real data" above means the bundled ``data/synthetic/`` demo dataset, not ``data/raw/``: the
latter holds a real household's private export and is intentionally empty/absent in this repo,
so a suite anyone can clone and run can't depend on it. The autouse fixture below points the
app's default (no-upload, no-demo-button) data source at ``data/synthetic/`` for every test in
this file; the one test that needs to see the true empty-``data/raw/`` state (the demo-data
button itself) layers its own further override on top.
"""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


@pytest.fixture(autouse=True)
def _default_to_synthetic_data(monkeypatch):
    from app import sidebar

    monkeypatch.setattr(
        sidebar, "SETTINGS", dataclasses.replace(sidebar.SETTINGS, raw_data_dir=sidebar.SETTINGS.synthetic_data_dir)
    )

TOP_LEVEL_TABS = [
    "Home",
    "How did this month compare?",
    "Ask the Energy Consultant",
    "Why did this happen?",
    "Costs and carbon",
    "What should I expect next?",
    "Advanced",
]

N_CONSULTANT_QUESTIONS = 17


def _tab(at: AppTest, label: str):
    return next(t for t in at.tabs if t.label == label)


def test_app_renders_without_exceptions():
    """Default render (weather toggle off) must stay network-free and exception-free."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    assert list(at.exception) == []
    labels = [t.label for t in at.tabs]
    # Top-level question-oriented navigation, in order (sub-tabs interleave in at.tabs).
    assert [label for label in labels if label in TOP_LEVEL_TABS] == TOP_LEVEL_TABS
    # Technical/whole-period surfaces are demoted to sub-tabs, never removed.
    for demoted in (
        "Fuel breakdown",
        "Seasons",
        "Weather impact",
        "Costs",
        "Carbon",
        "Unusual months",
        "Usage shifts",
        "Consumption over time",
        "Fuel comparisons (all years)",
        "Full report (AI Analyst)",
        "Statistical analysis",
        "Data quality",
    ):
        assert demoted in labels
    assert len(at.get("metric")) > 0

    # Home: leads with the month comparison, then the trimmed briefing (biggest finding,
    # largest saving, one forecast figure) -- overall assessment/household profile/confidence
    # grid were removed as duplicates of the Full Report, not moved to a new location here.
    home = _tab(at, "Home")
    home_subheaders = [s.value.lower() for s in home.get("subheader")]
    assert home_subheaders[0] == "this month compared with last year"
    assert "biggest finding" in home_subheaders
    assert "overall assessment" not in home_subheaders
    assert list(home.exception) == []

    # Consultant: renders the question list and states which fuel/month it answers for.
    consultant = _tab(at, "Ask the Energy Consultant")
    assert list(consultant.exception) == []
    assert any("questions i can answer" in md.value.lower() for md in consultant.get("markdown"))
    assert len(consultant.get("button")) == N_CONSULTANT_QUESTIONS
    assert any("answering for" in c.value.lower() for c in consultant.get("caption"))
    assert any("month questions answer for" in c.value.lower() for c in consultant.get("caption"))

    # Full report (AI Analyst): every section present, one level down under Advanced.
    analyst = _tab(at, "Full report (AI Analyst)")
    assert list(analyst.exception) == []
    analyst_headers = " ".join(md.value.lower() for md in analyst.get("header"))
    for section in (
        "executive summary",
        "key findings",
        "household energy profile",
        "recommendations",
        "confidence",
        "limitations",
    ):
        assert section in analyst_headers
    analyst_subheaders = " ".join(md.value.lower() for md in analyst.get("subheader"))
    assert "electricity findings" in analyst_subheaders
    assert "gas findings" in analyst_subheaders

    # Fuel breakdown: cross-check status and a real fuel-mix finding.
    fuel_tab = _tab(at, "Fuel breakdown")
    assert list(fuel_tab.exception) == []
    assert any("matches total" in s.value.lower() for s in fuel_tab.get("success"))
    assert any("fuel mix" in s.value.lower() for s in fuel_tab.get("subheader"))

    # Whole-period comparisons: retained under Advanced; weather prompt inert.
    comparisons = _tab(at, "Fuel comparisons (all years)")
    assert list(comparisons.exception) == []
    assert any("turn on" in info.value.lower() for info in comparisons.get("info"))

    # Costs: combined cost breakdown, the consumption+standing+VAT bill table, and
    # benchmark bands all render without any button click.
    costs = _tab(at, "Costs")
    assert list(costs.exception) == []
    assert len(costs.get("dataframe")) >= 2  # cost breakdown + bill breakdown tables
    assert any("bill breakdown" in s.value.lower() for s in costs.get("subheader"))
    bill_tables = [df for df in costs.get("dataframe") if "Total bill (£)" in df.value.columns]
    assert len(bill_tables) == 1
    assert {"Consumption (£)", "Standing (£)", "VAT 5% (£)", "Billing period"} <= set(
        bill_tables[0].value.columns
    )
    assert any("vs." in m.label for m in costs.get("metric"))

    # Carbon: real data has both fuels, so real emissions numbers.
    carbon = _tab(at, "Carbon")
    assert list(carbon.exception) == []
    assert any("emissions" in m.label.lower() for m in carbon.get("metric"))

    # Seasons: plain-language summary with the advanced STL panel retained.
    seasons = _tab(at, "Seasons")
    assert list(seasons.exception) == []
    assert "winter and summer cycle" in " ".join(md.value.lower() for md in seasons.get("markdown"))
    assert any("advanced statistical decomposition" in e.label.lower() for e in seasons.get("expander"))

    # Weather impact: toggle defaults off -> inert prompt with an inline enable button, no fetch.
    weather = _tab(at, "Weather impact")
    assert any("turn on weather adjustment" in b.label.lower() for b in weather.get("button"))

    # Usage shifts and Unusual months render whatever they find, without error.
    assert list(_tab(at, "Usage shifts").exception) == []
    assert list(_tab(at, "Unusual months").exception) == []

    # Forecasting: answer-first layout -- expected value + confidence lead, the model is a
    # detail inside the Model details expander (still present, still traceable).
    forecast = _tab(at, "What should I expect next?")
    assert any("what should i expect" in md.value.lower() for md in forecast.get("subheader"))
    forecast_metrics = [m.label for m in forecast.get("metric")]
    assert "Expected energy use" in forecast_metrics
    assert "Forecast confidence" in forecast_metrics
    assert any("expected" in m.label.lower() and "estimate" in m.label.lower() for m in forecast.get("metric"))
    expander_labels = [e.label.lower() for e in forecast.get("expander")]
    assert any("model details" in label for label in expander_labels)
    assert any("plausible range" in label for label in expander_labels)
    assert any("selected model" in md.value.lower() for md in forecast.get("markdown"))
    assert any("assumes your future use resembles previous years" in c.value for c in forecast.get("caption"))


# --- month-comparison journey -------------------------------------------------------------


def test_month_tab_defaults_to_latest_complete_month_vs_last_year():
    """The primary journey: latest complete month, combined energy, same month last year.

    The bundled synthetic dataset's last row is already a complete month (unlike the real
    household export this suite used to run against), so the in-progress-month notice never
    fires in this particular check -- that behavior (flagging/excluding a still-open billing
    period) has its own dedicated, date-controlled unit tests in test_monthly_comparison.py,
    so it isn't lost, just not re-exercised end-to-end here.
    """
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    month_tab = _tab(at, "How did this month compare?")
    assert list(month_tab.exception) == []

    month_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Month")
    # Bills run 6th-to-5th, so last month's bill is only complete from the 6th of this
    # month -- on the 1st-5th the default falls back one month further.
    today = pd.Timestamp.now()
    months_back = 1 if today.day >= 6 else 2
    latest_complete = (today.to_period("M") - months_back).to_timestamp()
    assert month_select.value == latest_complete
    # A month whose billing period is still open must not be selectable.
    assert today.to_period("M").to_timestamp() not in month_select.options

    mode_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Compare against")
    assert mode_select.value == "same_month_last_year"

    # Headline answer present, plus the same-month history chart (2 charts total).
    headline = next(md.value for md in month_tab.get("markdown") if md.value.startswith("####"))
    assert "%" in headline
    assert len(month_tab.get("plotly_chart")) == 2

    # Explanation structure renders.
    md_text = " ".join(md.value for md in month_tab.get("markdown"))
    assert "**What changed:**" in md_text
    assert "**Which fuel caused it:**" in md_text

    # Cost section: the full bill breakdown (consumption + standing + VAT = total).
    cost_metric_labels = [m.label for m in month_tab.get("metric")]
    for label in ("Consumption cost", "Standing charge", "VAT (5%)", "Estimated total bill"):
        assert label in cost_metric_labels
    assert any("bills run 6th to 5th" in c.value.lower() for c in month_tab.get("caption"))


def test_month_tab_mode_switching_updates_content_and_state():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    month_tab = _tab(at, "How did this month compare?")
    mode_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Compare against")
    mode_select.set_value("previous_month").run(timeout=60)
    assert list(at.exception) == []
    month_tab = _tab(at, "How did this month compare?")
    captions = " ".join(c.value for c in month_tab.get("caption"))
    assert "Adjacent months can differ because of seasonality" in captions

    mode_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Compare against")
    mode_select.set_value("typical_month").run(timeout=60)
    assert list(at.exception) == []
    month_tab = _tab(at, "How did this month compare?")
    captions = " ".join(c.value for c in month_tab.get("caption"))
    assert "Typical" in captions

    mode_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Compare against")
    mode_select.set_value("long_term").run(timeout=60)
    assert list(at.exception) == []
    month_tab = _tab(at, "How did this month compare?")
    assert any("trailing 12 months" in m.label.lower() for m in month_tab.get("metric"))


def test_month_tab_month_switching_is_exception_free_and_updates_headline():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    month_tab = _tab(at, "How did this month compare?")
    month_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Month")
    month_select.set_value(pd.Timestamp("2024-12-01")).run(timeout=60)
    assert list(at.exception) == []

    month_tab = _tab(at, "How did this month compare?")
    headline = next(md.value for md in month_tab.get("markdown") if md.value.startswith("####"))
    assert "December 2023" in headline  # December 2024 vs December 2023


def test_month_tab_fuel_switching_via_segmented_control():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    month_tab = _tab(at, "How did this month compare?")
    fuel_control = month_tab.get("button_group")[0]  # st.segmented_control in AppTest
    fuel_control.set_value("gas").run(timeout=60)
    assert list(at.exception) == []

    month_tab = _tab(at, "How did this month compare?")
    headline = next(md.value for md in month_tab.get("markdown") if md.value.startswith("####"))
    assert "gas" in headline.lower()


def test_home_month_cards_show_all_three_fuels_with_yoy_deltas():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    home = _tab(at, "Home")
    metric_labels = [m.label for m in home.get("metric")]
    assert any(label.startswith("Total energy") for label in metric_labels)
    assert any(label.startswith("Gas") for label in metric_labels)
    assert any(label.startswith("Electricity") for label in metric_labels)
    month_metrics = [m for m in home.get("metric") if "--" in m.label]
    assert all("vs" in (m.delta or "") for m in month_metrics)
    # Partial-month honesty on the homepage too -- not exercised here since the synthetic
    # dataset's last row is already complete (see test_month_tab_defaults_to_latest_complete_
    # month_vs_last_year's docstring); covered directly by test_monthly_comparison.py instead.


def test_consultant_month_answer_uses_currently_selected_month():
    """Selecting a different month on the comparison page must change what the Consultant
    answers -- no stale state (the audit-F2 rule applied to the month journey)."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    month_tab = _tab(at, "How did this month compare?")
    month_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Month")
    month_select.set_value(pd.Timestamp("2024-12-01")).run(timeout=60)

    consultant = _tab(at, "Ask the Energy Consultant")
    assert any("December 2024" in c.value for c in consultant.get("caption"))
    button = next(
        b for b in consultant.get("button") if b.label == "How did this month compare with last year?"
    )
    button.click().run(timeout=60)
    assert list(at.exception) == []
    consultant = _tab(at, "Ask the Energy Consultant")
    answer_text = " ".join(md.value for md in consultant.get("markdown"))
    assert "December 2023" in answer_text  # answers for Dec 2024 vs Dec 2023, not the default month


def test_consultant_free_text_routes_to_month_comparison():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant = _tab(at, "Ask the Energy Consultant")
    text_input = consultant.get("text_input")[0]
    text_input.set_value("did I use more than usual this month?").run(timeout=60)
    assert list(at.exception) == []
    consultant = _tab(at, "Ask the Energy Consultant")
    answer_text = " ".join(md.value.lower() for md in consultant.get("markdown"))
    assert "did i use more energy than usual this month?" in answer_text


def test_tariff_inputs_drive_bill_estimates():
    """Changing the sidebar's Tariff inputs must flow through to the bill breakdown --
    the defaults are this household's default rates, but another provider's rates work too."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    inputs = {n.label: n for n in at.sidebar.get("number_input")}
    assert inputs["Electricity standing charge (p/day)"].value == 62.77
    assert inputs["Gas standing charge (p/day)"].value == 34.97
    assert inputs["VAT rate (%)"].value == 5.0

    month_tab = _tab(at, "How did this month compare?")
    standing_before = next(m for m in month_tab.get("metric") if m.label == "Standing charge").value

    inputs["Electricity standing charge (p/day)"].set_value(100.0)
    inputs["VAT rate (%)"].set_value(20.0).run(timeout=60)
    assert list(at.exception) == []

    month_tab = _tab(at, "How did this month compare?")
    metric_labels = [m.label for m in month_tab.get("metric")]
    assert "VAT (20%)" in metric_labels  # label follows the input
    standing_after = next(m for m in month_tab.get("metric") if m.label == "Standing charge").value
    assert standing_after != standing_before
    # Default fuel is combined, so both daily rates apply across the billing period's days
    # (which always equal the labelled month's day count).
    selected = next(sb for sb in month_tab.get("selectbox") if sb.label == "Month").value
    expected_standing = (1.0 + 0.3497) * selected.days_in_month
    assert standing_after == f"£{expected_standing:,.2f}"

    costs = _tab(at, "Costs")
    bill_tables = [df for df in costs.get("dataframe") if "Total bill (£)" in df.value.columns]
    assert any("VAT 20% (£)" in df.value.columns for df in bill_tables)


# --- consultant regression tests (pre-existing behaviour) ---------------------------------


def test_consultant_preset_question_button_renders_real_answer():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant = _tab(at, "Ask the Energy Consultant")
    button = next(
        b for b in consultant.get("button") if "focus on reducing gas or electricity" in b.label.lower()
    )
    button.click().run(timeout=60)

    assert list(at.exception) == []
    consultant = _tab(at, "Ask the Energy Consultant")
    markdown_text = " ".join(md.value.lower() for md in consultant.get("markdown"))
    assert "gas" in markdown_text or "electricity" in markdown_text
    assert len(consultant.get("expander")) > 0  # the evidence expander rendered


def test_consultant_answer_recomputed_after_fuel_switch():
    """Regression (audit F2): the Consultant stores the selected *question*, not the computed
    answer, so switching fuel recomputes the answer from the new context."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant = _tab(at, "Ask the Energy Consultant")
    button = next(b for b in consultant.get("button") if "compare to average" in b.label.lower())
    button.click().run(timeout=60)
    assert at.session_state["consultant_selected_question"] == "How does my usage compare to average?"

    fuel_select = next(sb for sb in at.sidebar.selectbox if sb.label == "Fuel to analyze")
    fuel_select.set_value("gas").run(timeout=60)

    assert list(at.exception) == []
    assert at.session_state["consultant_selected_question"] == "How does my usage compare to average?"
    markdown_text = " ".join(md.value.lower() for md in _tab(at, "Ask the Energy Consultant").get("markdown"))
    assert "uk household" in markdown_text


def test_consultant_unmatched_free_text_falls_back_to_question_list():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant = _tab(at, "Ask the Energy Consultant")
    text_input = consultant.get("text_input")[0]
    text_input.set_value("asdkjaslkdj random gibberish text").run(timeout=60)

    assert list(at.exception) == []
    consultant = _tab(at, "Ask the Energy Consultant")
    assert any("couldn't confidently match" in info.value.lower() for info in consultant.get("info"))
    assert len(consultant.get("button")) == N_CONSULTANT_QUESTIONS


def test_multi_fuel_forecast_button_populates_comparison_across_tabs():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    costs = _tab(at, "Costs")
    button = next(b for b in costs.get("button") if "multi-fuel forecast" in b.label.lower())
    button.click().run(timeout=120)

    assert list(at.exception) == []
    assert len(_tab(at, "Costs").get("dataframe")) > 0
    comparisons = _tab(at, "Fuel comparisons (all years)")
    assert list(comparisons.exception) == []
    assert any(
        "likely (kwh)" in df.value.columns.str.lower().tolist() for df in comparisons.get("dataframe")
    )


def test_report_generate_download_and_fingerprint_invalidation():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    gen = next(b for b in at.sidebar.button if "generate household energy review" in b.label.lower())
    gen.click().run(timeout=120)

    assert list(at.exception) == []
    stored = at.session_state["household_report"]
    assert stored is not None
    _fingerprint, html = stored
    assert html.startswith("<!DOCTYPE html>")
    assert len(html) > 1_000_000
    assert "Household Energy Review" in html
    download_labels = [d.label.lower() for d in at.sidebar.get("download_button")]
    assert any("download household energy review" in label for label in download_labels)

    fuel_select = next(sb for sb in at.sidebar.selectbox if sb.label == "Fuel to analyze")
    fuel_select.set_value("gas").run(timeout=120)
    assert list(at.exception) == []
    assert "household_report" not in at.session_state
    assert any("generate household energy review" in b.label.lower() for b in at.sidebar.button)


def test_fuel_selectbox_offers_all_three_fuels_and_switching_is_exception_free():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    fuel_select = next(sb for sb in at.sidebar.selectbox if sb.label == "Fuel to analyze")
    assert fuel_select.options == ["Total (Electricity + Gas)", "Electricity only", "Gas only"]
    assert fuel_select.value == "total"

    fuel_select.set_value("electricity").run(timeout=60)
    assert list(at.exception) == []
    assert "Fuel: Electricity only" in " ".join(md.value for md in at.get("caption"))

    fuel_select.set_value("gas").run(timeout=60)
    assert list(at.exception) == []
    assert "Fuel: Gas only" in " ".join(md.value for md in at.get("caption"))


def test_seasonal_tab_plain_language_summary_matches_synthetic_data():
    """On the bundled 61-month synthetic dataset: strong seasonality, a stable trend, January
    2023 as the largest unexplained deviation (+575 kWh)."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    seasons = _tab(at, "Seasons")
    conclusion = next(md.value for md in seasons.get("markdown") if md.value.startswith("#####"))
    assert "winter and summer cycle" in conclusion
    assert "broadly stable" in conclusion
    assert "January 2023 was materially higher" in conclusion

    metrics = {m.label: (m.value, m.delta) for m in seasons.get("metric")}
    assert metrics["Seasonal influence"][0] == "Strong"
    assert metrics["Long-term trend"][0] == "Stable"
    assert metrics["Largest unexplained deviation"][0] == "Jan 2023"
    assert metrics["Largest unexplained deviation"][1] == "+575 kWh"


def _synthetic_daily_weather(start: str, end: str):
    """Deterministic fake daily weather covering [start, end]: seasonal temperatures, a
    snowy + windy January 2023, quiet otherwise. Full coverage, all schema columns.

    January 2023 (not December 2024) so the injected severe weather lands on the bundled
    data/synthetic/ dataset's actual largest flagged anomaly month -- see
    test_seasonal_tab_plain_language_summary_matches_synthetic_data.
    """
    import numpy as np

    dates = pd.date_range(start, end, freq="D")
    day_of_year = dates.day_of_year.to_numpy()
    temp = 9.0 - 8.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    df = pd.DataFrame(
        {
            "date": dates,
            "temp_mean_c": temp,
            "snowfall_cm": 0.0,
            "snow_depth_m": 0.0,
            "precipitation_mm": 1.0,
            "wind_speed_max_kmh": 25.0,
            "wind_gust_max_kmh": 40.0,
        }
    )
    event_month = (df["date"] >= "2023-01-01") & (df["date"] <= "2023-01-31")
    snow_days = df["date"].between("2023-01-18", "2023-01-23")
    df.loc[snow_days, "snowfall_cm"] = [2.0, 6.0, 8.0, 1.0, 2.0, 0.5]
    df.loc[snow_days, "snow_depth_m"] = 0.08
    wind_days = df["date"].between("2023-01-04", "2023-01-08")
    df.loc[wind_days, "wind_speed_max_kmh"] = [70.0, 65.0, 75.0, 68.0, 63.0]
    df.loc[wind_days, "wind_gust_max_kmh"] = [95.0, 88.0, 110.0, 92.0, 85.0]
    df.loc[event_month, "temp_mean_c"] = df.loc[event_month, "temp_mean_c"] - 3.0  # a cold January
    return df


def test_weather_on_renders_severe_weather_context_without_network(monkeypatch):
    """Turning the weather toggle on must exercise the full Weather Context Engine path,
    plus the month tab's weather-explained breakdown, with the fetch mocked out."""
    from app import tabs_drivers, tabs_weather_context

    def _fake_fetch(lat, lon, start, end, timezone, cache_dir):
        return _synthetic_daily_weather(start, end)

    monkeypatch.setattr(tabs_drivers, "fetch_daily_weather", _fake_fetch)
    monkeypatch.setattr(tabs_weather_context, "fetch_daily_weather", _fake_fetch)
    tabs_drivers.load_weather_analysis.clear()
    tabs_weather_context.load_weather_context.clear()

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)
    toggle = next(t for t in at.sidebar.toggle if t.label == "Weather adjustment")
    toggle.set_value(True).run(timeout=120)

    assert list(at.exception) == []

    weather = _tab(at, "Weather impact")
    assert list(weather.exception) == []
    subheaders = " ".join(s.value.lower() for s in weather.get("subheader"))
    assert "severe weather in your history" in subheaders

    anomalies_tab = _tab(at, "Unusual months")
    assert list(anomalies_tab.exception) == []
    anom_markdown = " ".join(md.value.lower() for md in anomalies_tab.get("markdown"))
    assert "what the weather was like" in anom_markdown
    assert "snow day" in anom_markdown

    # Month tab: with weather on, the explanation includes the weather split. The app's
    # *default* month (latest complete vs. same month last year) happens to be a "little"
    # change for the synthetic dataset, which intentionally suppresses the weather-effect
    # line (src.monthly_narrative._weather_effect) -- select a month with a real swing instead,
    # so this test still checks the line renders when it's supposed to.
    month_tab = _tab(at, "How did this month compare?")
    month_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Month")
    month_select.set_value(pd.Timestamp("2024-12-01")).run(timeout=60)
    month_tab = _tab(at, "How did this month compare?")
    assert list(month_tab.exception) == []
    month_markdown = " ".join(md.value for md in month_tab.get("markdown"))
    assert "**How weather affected it:**" in month_markdown

    # Home: "What explains the change?" breakdown appears only when the model covers it.
    home = _tab(at, "Home")
    home_markdown = " ".join(md.value for md in home.get("markdown"))
    assert "What explains the change?" in home_markdown
    assert "Weather-related" in home_markdown

    # Consultant: the weather-attribution month question now has a real split to report.
    consultant = _tab(at, "Ask the Energy Consultant")
    button = next(b for b in consultant.get("button") if b.label == "Was the difference caused by weather?")
    button.click().run(timeout=120)
    assert list(at.exception) == []
    answer_text = " ".join(md.value.lower() for md in _tab(at, "Ask the Energy Consultant").get("markdown"))
    assert "kwh change" in answer_text or "matches what the temperature model expected" in answer_text


def test_weather_impact_inline_button_enables_weather_adjustment(monkeypatch):
    """The Weather Impact tab's own 'Turn on weather adjustment' button (UX audit finding: the
    app's most persuasive analysis was opt-in with no in-page way to enable it) must actually
    flip the sidebar toggle, not just be decorative."""
    from app import tabs_drivers, tabs_weather_context

    def _fake_fetch(lat, lon, start, end, timezone, cache_dir):
        return _synthetic_daily_weather(start, end)

    monkeypatch.setattr(tabs_drivers, "fetch_daily_weather", _fake_fetch)
    monkeypatch.setattr(tabs_weather_context, "fetch_daily_weather", _fake_fetch)
    tabs_drivers.load_weather_analysis.clear()
    tabs_weather_context.load_weather_context.clear()

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    toggle = next(t for t in at.sidebar.toggle if t.label == "Weather adjustment")
    assert toggle.value is False

    weather = _tab(at, "Weather impact")
    button = next(b for b in weather.get("button") if "turn on weather adjustment" in b.label.lower())
    button.click().run(timeout=120)

    assert list(at.exception) == []
    toggle = next(t for t in at.sidebar.toggle if t.label == "Weather adjustment")
    assert toggle.value is True
    weather = _tab(at, "Weather impact")
    assert list(weather.exception) == []
    assert any("severe weather in your history" in s.value.lower() for s in weather.get("subheader"))


def test_weather_location_search_requires_explicit_confirmation(monkeypatch):
    """Typing a new location must not change what weather adjustment uses until a specific
    candidate is confirmed -- ambiguous names (multiple real Manchesters) are exactly why."""
    from app import sidebar, tabs_drivers, tabs_weather_context
    from src.geocoding import LocationCandidate

    manchester_uk = LocationCandidate(
        label="Manchester, England, United Kingdom",
        latitude=53.48, longitude=-2.24, timezone="Europe/London",
    )
    manchester_us = LocationCandidate(
        label="Manchester, New Hampshire, United States",
        latitude=42.99, longitude=-71.45, timezone="America/New_York",
    )
    monkeypatch.setattr(sidebar, "search_location", lambda query, count=5: [manchester_uk, manchester_us])
    sidebar._search_location_cached.clear()

    def _fake_fetch(lat, lon, start, end, timezone, cache_dir):
        return _synthetic_daily_weather(start, end)

    monkeypatch.setattr(tabs_drivers, "fetch_daily_weather", _fake_fetch)
    monkeypatch.setattr(tabs_weather_context, "fetch_daily_weather", _fake_fetch)
    tabs_drivers.load_weather_analysis.clear()
    tabs_weather_context.load_weather_context.clear()

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    location_input = next(t for t in at.sidebar.text_input if "Location" in t.label)
    assert location_input.value == "Aberdeen, Scotland (AB21 area)"  # the default, unconfirmed

    location_input.set_value("Manchester").run(timeout=60)
    assert list(at.exception) == []

    match_radio = next(r for r in at.sidebar.radio if r.label == "Confirm the match")
    assert match_radio.options == [manchester_uk.label, manchester_us.label]

    # Typing alone must not have changed the location weather adjustment would use.
    caption_text = " ".join(c.value for c in at.sidebar.caption)
    assert "Location: Aberdeen" in caption_text
    assert "pending confirmation" in caption_text

    match_radio.set_value(manchester_us.label).run(timeout=60)
    confirm_button = next(b for b in at.sidebar.button if b.label == "Confirm location")
    confirm_button.click().run(timeout=60)
    assert list(at.exception) == []

    location_input = next(t for t in at.sidebar.text_input if "Location" in t.label)
    assert location_input.value == manchester_us.label
    caption_text = " ".join(c.value for c in at.sidebar.caption)
    assert "Location: Manchester, New Hampshire" in caption_text
    assert "pending confirmation" not in caption_text

    toggle = next(t for t in at.sidebar.toggle if t.label == "Weather adjustment")
    toggle.set_value(True).run(timeout=120)
    assert list(at.exception) == []

    weather_tab = _tab(at, "Weather impact")
    assert list(weather_tab.exception) == []
    weather_caption = " ".join(c.value for c in weather_tab.get("caption"))
    assert "Location: Manchester, New Hampshire, United States" in weather_caption


def test_no_data_offers_demo_button_and_loads_synthetic_dataset(tmp_path, monkeypatch):
    """UX audit finding: a first-time visitor with no CSV in hand and an empty data/raw/ had no
    path forward but a bare upload box. The sidebar's 'Try the demo data' button must point the
    same loading pipeline at the real, bundled data/synthetic/ dataset and actually render."""
    import dataclasses

    from app import sidebar

    empty_raw_dir = tmp_path / "raw"
    empty_raw_dir.mkdir()
    monkeypatch.setattr(sidebar, "SETTINGS", dataclasses.replace(sidebar.SETTINGS, raw_data_dir=empty_raw_dir))

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    assert list(at.exception) == []
    assert any("no data found in data/raw/" in w.value.lower() for w in at.sidebar.get("warning"))
    button = next(b for b in at.sidebar.get("button") if b.label == "Try the demo data")
    button.click().run(timeout=60)

    assert list(at.exception) == []
    assert any("using the bundled demo dataset" in i.value.lower() for i in at.sidebar.get("info"))
    home = _tab(at, "Home")
    assert list(home.exception) == []
    assert len(home.get("metric")) > 0  # real KPI metrics rendered, not an empty state

    # Uploading real data must still win over demo mode, even with the flag left set.
    switch_button = next(b for b in at.sidebar.get("button") if b.label == "Use my own data instead")
    switch_button.click().run(timeout=60)
    assert list(at.exception) == []
    assert any("no data found in data/raw/" in w.value.lower() for w in at.sidebar.get("warning"))
