"""Smoke and journey tests: the Streamlit app must render with real data and no uncaught
exceptions, with the month-comparison journey as the primary surface.

Navigation is organized around user questions; ``st.tabs`` nests one level,
and AppTest flattens nested tabs into ``at.tabs`` in creation order -- so
tabs are looked up by label here, never by index.
"""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"

TOP_LEVEL_TABS = [
    "Home",
    "How did this month compare?",
    "What drives my usage?",
    "Costs and carbon",
    "Did anything unusual happen?",
    "What should I expect next?",
    "Ask the Energy Consultant",
    "Long-term trends",
    "Data and methods",
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

    # Home: leads with the month comparison, then the evidence-based briefing.
    home = _tab(at, "Home")
    home_subheaders = [s.value.lower() for s in home.get("subheader")]
    assert home_subheaders[0] == "this month compared with last year"
    assert "overall assessment" in home_subheaders
    assert "biggest finding" in home_subheaders
    assert list(home.exception) == []

    # Consultant: renders the question list and states which fuel/month it answers for.
    consultant = _tab(at, "Ask the Energy Consultant")
    assert list(consultant.exception) == []
    assert any("questions i can answer" in md.value.lower() for md in consultant.get("markdown"))
    assert len(consultant.get("button")) == N_CONSULTANT_QUESTIONS
    assert any("answering for" in c.value.lower() for c in consultant.get("caption"))
    assert any("month questions answer for" in c.value.lower() for c in consultant.get("caption"))

    # Full report (AI Analyst): every section present, one level down under Data and methods.
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

    # Whole-period comparisons: retained under Long-term trends; weather prompt inert.
    comparisons = _tab(at, "Fuel comparisons (all years)")
    assert list(comparisons.exception) == []
    assert any("turn on" in info.value.lower() for info in comparisons.get("info"))

    # Costs: combined cost breakdown and benchmark bands render without any button click.
    costs = _tab(at, "Costs")
    assert list(costs.exception) == []
    assert len(costs.get("dataframe")) > 0
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

    # Weather impact: toggle defaults off -> inert prompt, no fetch.
    weather = _tab(at, "Weather impact")
    assert any("turn on" in info.value.lower() for info in weather.get("info"))

    # Usage shifts and Unusual months render whatever they find, without error.
    assert list(_tab(at, "Usage shifts").exception) == []
    assert list(_tab(at, "Unusual months").exception) == []

    # Forecasting: CV ran and picked a model.
    forecast = _tab(at, "What should I expect next?")
    assert any("selected model" in md.value.lower() for md in forecast.get("subheader"))


# --- month-comparison journey -------------------------------------------------------------


def test_month_tab_defaults_to_latest_complete_month_vs_last_year():
    """The primary journey: latest complete month, combined energy, same month last year --
    with the in-progress month flagged and excluded, and exactly one primary chart."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    month_tab = _tab(at, "How did this month compare?")
    assert list(month_tab.exception) == []

    month_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Month")
    latest_complete = (pd.Timestamp.now().to_period("M") - 1).to_timestamp()
    assert month_select.value == latest_complete
    # The in-progress current month must not be selectable.
    assert pd.Timestamp.now().to_period("M").to_timestamp() not in month_select.options

    mode_select = next(sb for sb in month_tab.get("selectbox") if sb.label == "Compare against")
    assert mode_select.value == "same_month_last_year"

    # Partial-month notice names both months.
    infos = " ".join(i.value for i in month_tab.get("info"))
    assert "is incomplete" in infos
    assert "primary comparison uses" in infos

    # Headline answer present, plus the same-month history chart (2 charts total).
    headline = next(md.value for md in month_tab.get("markdown") if md.value.startswith("####"))
    assert "%" in headline
    assert len(month_tab.get("plotly_chart")) == 2

    # Explanation structure renders.
    md_text = " ".join(md.value for md in month_tab.get("markdown"))
    assert "**What changed:**" in md_text
    assert "**Which fuel caused it:**" in md_text


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
    # Partial-month honesty on the homepage too.
    assert any("is incomplete" in i.value for i in home.get("info"))


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


def test_seasonal_tab_plain_language_summary_matches_real_data():
    """On the real 35-month dataset: strong seasonality, a stable trend, December 2024 as
    the largest unexplained deviation (+253 kWh)."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    seasons = _tab(at, "Seasons")
    conclusion = next(md.value for md in seasons.get("markdown") if md.value.startswith("#####"))
    assert "winter and summer cycle" in conclusion
    assert "broadly stable" in conclusion
    assert "December 2024 was materially higher" in conclusion

    metrics = {m.label: (m.value, m.delta) for m in seasons.get("metric")}
    assert metrics["Seasonal influence"][0] == "Strong"
    assert metrics["Long-term trend"][0] == "Stable"
    assert metrics["Largest unexplained deviation"][0] == "Dec 2024"
    assert metrics["Largest unexplained deviation"][1] == "+253 kWh"


def _synthetic_daily_weather(start: str, end: str):
    """Deterministic fake daily weather covering [start, end]: seasonal temperatures, a
    snowy + windy December 2024, quiet otherwise. Full coverage, all schema columns."""
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
    dec24 = (df["date"] >= "2024-12-01") & (df["date"] <= "2024-12-31")
    snow_days = df["date"].between("2024-12-18", "2024-12-23")
    df.loc[snow_days, "snowfall_cm"] = [2.0, 6.0, 8.0, 1.0, 2.0, 0.5]
    df.loc[snow_days, "snow_depth_m"] = 0.08
    wind_days = df["date"].between("2024-12-04", "2024-12-08")
    df.loc[wind_days, "wind_speed_max_kmh"] = [70.0, 65.0, 75.0, 68.0, 63.0]
    df.loc[wind_days, "wind_gust_max_kmh"] = [95.0, 88.0, 110.0, 92.0, 85.0]
    df.loc[dec24, "temp_mean_c"] = df.loc[dec24, "temp_mean_c"] - 3.0  # a cold December
    return df


def test_weather_on_renders_severe_weather_context_without_network(monkeypatch):
    """Turning the weather toggle on must exercise the full Weather Context Engine path,
    plus the month tab's weather-explained breakdown, with the fetch mocked out."""
    import sys

    sys.path.insert(0, str(APP_PATH.parent))
    import tabs_phase2
    import tabs_weather_context

    def _fake_fetch(lat, lon, start, end, timezone, cache_dir):
        return _synthetic_daily_weather(start, end)

    monkeypatch.setattr(tabs_phase2, "fetch_daily_weather", _fake_fetch)
    monkeypatch.setattr(tabs_weather_context, "fetch_daily_weather", _fake_fetch)
    tabs_phase2.load_weather_analysis.clear()
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

    # Month tab: with weather on, the explanation includes the weather split.
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
