"""Smoke test: the Streamlit app must render with real data and no uncaught exceptions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_renders_without_exceptions():
    """Default render (weather toggle off) must stay network-free and exception-free.

    Forecasting and the decision-support report (findings/recommendations/
    confidence) aren't behind an opt-in toggle like weather -- they're pure
    CPU-bound, benchmarked well under 10s on the real data -- so this render
    includes that cost. Timeout raised accordingly.
    """
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    assert list(at.exception) == []
    assert [t.label for t in at.tabs] == [
        "Executive Summary",
        "AI Consultant",
        "AI Analyst",
        "Consumption Analysis",
        "Fuel Breakdown",
        "Comparisons",
        "Cost Intelligence",
        "Carbon",
        "Statistical Analysis",
        "How the Seasons Affect Usage",
        "Weather Impact",
        "Usage Shifts",
        "Forecasting",
        "Unusual Months",
        "Data Quality",
    ]
    assert len(at.get("metric")) > 0

    # Executive Summary: real briefing content, not placeholders.
    assert any("overall assessment" in md.value.lower() for md in at.tabs[0].get("subheader"))
    assert any("biggest finding" in md.value.lower() for md in at.tabs[0].get("subheader"))
    assert list(at.tabs[0].exception) == []
    # Household energy profile section (Task 10) -- real data has all 3 fuels available.
    assert any("household energy profile" in s.value.lower() for s in at.tabs[0].get("subheader"))

    # AI Consultant tab: renders the question list without error, and states which fuel
    # its answers describe (audit finding F6).
    assert list(at.tabs[1].exception) == []
    assert any("questions i can answer" in md.value.lower() for md in at.tabs[1].get("markdown"))
    assert len(at.tabs[1].get("button")) == 9
    assert any("answering for" in c.value.lower() for c in at.tabs[1].get("caption"))

    # AI Analyst: full deterministic report renders with every section present.
    assert list(at.tabs[2].exception) == []
    analyst_headers = " ".join(md.value.lower() for md in at.tabs[2].get("header"))
    for section in (
        "executive summary",
        "key findings",
        "household energy profile",
        "recommendations",
        "confidence",
        "limitations",
    ):
        assert section in analyst_headers
    # Electricity/Gas Findings render as subheaders, not headers -- a different AppTest element type.
    analyst_subheaders = " ".join(md.value.lower() for md in at.tabs[2].get("subheader"))
    assert "electricity findings" in analyst_subheaders
    assert "gas findings" in analyst_subheaders

    # Fuel Breakdown tab: real data has all three fuel exports, so this should show the
    # cross-check status and a real (not placeholder) fuel-mix finding.
    assert list(at.tabs[4].exception) == []
    assert any("matches total" in s.value.lower() for s in at.tabs[4].get("success"))
    assert any("fuel mix" in s.value.lower() for s in at.tabs[4].get("subheader"))
    assert any("%" in md.value for md in at.tabs[4].get("markdown"))

    # Comparisons tab (Task 2/5/6/7): renders without error; weather sensitivity needs the
    # weather toggle (off by default here), so it should show the inert prompt.
    assert list(at.tabs[5].exception) == []
    assert any("turn on" in info.value.lower() for info in at.tabs[5].get("info"))

    # Cost Intelligence tab (Task 4/12): renders the combined cost breakdown table and
    # benchmark bands without needing the forecast button clicked.
    assert list(at.tabs[6].exception) == []
    assert len(at.tabs[6].get("dataframe")) > 0
    assert any("vs." in m.label for m in at.tabs[6].get("metric"))

    # Carbon tab (Task 8): real data has both electricity and gas, so this should show real
    # emissions numbers, not the unavailable-data message.
    assert list(at.tabs[7].exception) == []
    assert any("emissions" in m.label.lower() for m in at.tabs[7].get("metric"))
    assert any("co2e" in m.value.lower() for m in at.tabs[7].get("metric"))

    # How the Seasons Affect Usage tab: plain-language summary, interpretation cards, and the
    # collapsed advanced expander (which still retains the original STL chart/strengths) must
    # all render from real data without exceptions.
    seasonal_tab = at.tabs[9]
    assert list(seasonal_tab.exception) == []
    seasonal_markdown = " ".join(md.value.lower() for md in seasonal_tab.get("markdown"))
    assert "winter and summer cycle" in seasonal_markdown
    metric_labels = [m.label for m in seasonal_tab.get("metric")]
    assert "Seasonal influence" in metric_labels
    assert "Long-term trend" in metric_labels
    assert "Largest unexplained deviation" in metric_labels
    assert "Seasonal strength" in metric_labels  # retained inside the advanced expander
    assert "Trend strength" in metric_labels
    assert any("advanced statistical decomposition" in e.label.lower() for e in seasonal_tab.get("expander"))
    assert len(seasonal_tab.get("plotly_chart")) == 3  # overview + calendar profile + advanced STL panel

    # Weather tab: toggle defaults off, so this must be the inert prompt, not a fetch attempt.
    assert any("turn on" in info.value.lower() for info in at.tabs[10].get("info"))

    # Usage Shifts tab: renders (either a detected-points table or the "stable" message).
    assert list(at.tabs[11].exception) == []

    # Forecasting tab: CV ran and picked a model, shown as a subheader.
    assert any("selected model" in md.value.lower() for md in at.tabs[12].get("subheader"))

    # Unusual Months tab: renders without error, whatever it finds.
    assert list(at.tabs[13].exception) == []


def test_consultant_preset_question_button_renders_real_answer():
    """Clicking a preset question button on the AI Consultant tab must render a real answer
    (evidence-backed, not a placeholder) without raising."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant_tab = at.tabs[1]
    button = next(b for b in consultant_tab.get("button") if "focus on reducing gas or electricity" in b.label.lower())
    button.click().run(timeout=60)

    assert list(at.exception) == []
    assert list(at.tabs[1].exception) == []
    markdown_text = " ".join(md.value.lower() for md in at.tabs[1].get("markdown"))
    assert "gas" in markdown_text or "electricity" in markdown_text
    assert len(at.tabs[1].get("expander")) > 0  # the evidence expander rendered


def test_consultant_answer_recomputed_after_fuel_switch():
    """Regression (audit F2): the Consultant stores the selected *question*, not the computed
    answer, so switching fuel recomputes the answer from the new context instead of replaying
    a stale one built for the previous fuel."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    button = next(b for b in at.tabs[1].get("button") if "compare to average" in b.label.lower())
    button.click().run(timeout=60)
    assert at.session_state["consultant_selected_question"] == "How does my usage compare to average?"

    fuel_select = next(sb for sb in at.sidebar.selectbox if sb.label == "Fuel to analyze")
    fuel_select.set_value("gas").run(timeout=60)

    assert list(at.exception) == []
    # The question survives the fuel switch and the answer is re-rendered from current context.
    assert at.session_state["consultant_selected_question"] == "How does my usage compare to average?"
    markdown_text = " ".join(md.value.lower() for md in at.tabs[1].get("markdown"))
    assert "uk household" in markdown_text


def test_consultant_free_text_question_routes_correctly():
    """Typing a natural-language phrasing of a supported question must route to the right
    handler and render an answer, without needing the exact preset wording."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant_tab = at.tabs[1]
    text_input = consultant_tab.get("text_input")[0]
    text_input.set_value("why did my bill increase this year").run(timeout=60)

    assert list(at.exception) == []
    assert list(at.tabs[1].exception) == []
    markdown_text = " ".join(md.value.lower() for md in at.tabs[1].get("markdown"))
    assert "why did my bill change" in markdown_text


def test_consultant_unmatched_free_text_falls_back_to_question_list():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    consultant_tab = at.tabs[1]
    text_input = consultant_tab.get("text_input")[0]
    text_input.set_value("asdkjaslkdj random gibberish text").run(timeout=60)

    assert list(at.exception) == []
    assert any("couldn't confidently match" in info.value.lower() for info in at.tabs[1].get("info"))
    assert len(at.tabs[1].get("button")) == 9  # the fallback question list still renders


def test_multi_fuel_forecast_button_populates_comparison_across_tabs():
    """Clicking the opt-in forecast-comparison button on Cost Intelligence must not raise, and
    the result should also become visible on the Comparisons tab (shared via session state)."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    cost_tab = at.tabs[6]
    button = next(b for b in cost_tab.get("button") if "multi-fuel forecast" in b.label.lower())
    button.click().run(timeout=120)

    assert list(at.exception) == []
    assert list(at.tabs[6].exception) == []
    assert len(at.tabs[6].get("dataframe")) > 0

    comparisons_tab = at.tabs[5]
    assert list(comparisons_tab.exception) == []
    assert any("likely (kwh)" in df.value.columns.str.lower().tolist() for df in comparisons_tab.get("dataframe"))


def test_report_generate_download_and_fingerprint_invalidation():
    """The sidebar's two-step report flow: Generate builds and stores the HTML, Download
    appears with a non-trivial payload, and switching fuel invalidates the stored report
    (fingerprint mismatch) so a stale report is never served -- the audit F2/F7 lesson."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    gen = next(b for b in at.sidebar.button if "generate household energy review" in b.label.lower())
    gen.click().run(timeout=120)

    assert list(at.exception) == []
    stored = at.session_state["household_report"]
    assert stored is not None
    fingerprint, html = stored
    assert html.startswith("<!DOCTYPE html>")
    assert len(html) > 1_000_000  # plotly.js inlined -- a real self-contained report
    assert "Household Energy Review" in html
    # Download button now present (a distinct AppTest element type), Generate gone.
    download_labels = [d.label.lower() for d in at.sidebar.get("download_button")]
    assert any("download household energy review" in label for label in download_labels)
    assert not any("generate household energy review" in b.label.lower() for b in at.sidebar.button)

    # Fuel switch -> fingerprint mismatch -> stored report discarded, Generate returns.
    fuel_select = next(sb for sb in at.sidebar.selectbox if sb.label == "Fuel to analyze")
    fuel_select.set_value("gas").run(timeout=120)
    assert list(at.exception) == []
    assert "household_report" not in at.session_state
    assert any("generate household energy review" in b.label.lower() for b in at.sidebar.button)
    assert at.sidebar.get("download_button") == []


def test_fuel_selectbox_offers_all_three_fuels_and_switching_is_exception_free():
    """The real data/raw/ has Total/Electricity/Gas exports, so all three options should appear,
    and switching the selection must re-run the entire pipeline against that fuel's data cleanly.
    """
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
    """On the real 35-month dataset: strong seasonality, a stable trend, and December 2024 as
    the largest unexplained deviation (+253 kWh) -- verified against src.seasonal_summary
    directly, not just "renders something"."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    seasonal_tab = at.tabs[9]
    conclusion = next(
        md.value for md in seasonal_tab.get("markdown") if md.value.startswith("#####")
    )
    assert "winter and summer cycle" in conclusion
    assert "broadly stable" in conclusion
    assert "December 2024 was materially higher" in conclusion

    metrics = {m.label: (m.value, m.delta) for m in seasonal_tab.get("metric")}
    assert metrics["Seasonal influence"][0] == "Strong"
    assert metrics["Long-term trend"][0] == "Stable"
    assert metrics["Largest unexplained deviation"][0] == "Dec 2024"
    assert metrics["Largest unexplained deviation"][1] == "+253 kWh"

    assert any("december 2024 was 253 kwh higher" in i.value.lower() for i in seasonal_tab.get("info"))


def test_seasonal_tab_renders_without_exceptions_regardless_of_container_width():
    """Layout responsiveness at 390px/768px/desktop relies entirely on Streamlit's built-in
    responsive columns/containers (st.columns stacks vertically below ~640px CSS breakpoints
    by framework default) -- there is no width-conditional Python branching in
    ``render_seasonality`` for a narrow viewport to break. AppTest has no viewport/pixel-width
    concept, so this can only assert the exception-free render that width-independent code
    guarantees; real responsive layout was checked manually in a browser (see PR notes)."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    assert list(at.tabs[9].exception) == []
    assert len(at.tabs[9].get("column")) >= 3  # the 3-card interpretation row uses st.columns(3)


def _synthetic_daily_weather(start: str, end: str):
    """Deterministic fake daily weather covering [start, end]: seasonal temperatures, a
    snowy + windy December 2024, quiet otherwise. Full coverage, all schema columns."""
    import numpy as np
    import pandas as pd

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
    """Turning the weather toggle on must exercise the full Weather Context Engine path --
    energy signature, monthly context, per-anomaly interpretation, Consultant question --
    with the fetch mocked out, so the suite stays network-free."""
    import sys

    sys.path.insert(0, str(APP_PATH.parent))
    import tabs_phase2
    import tabs_weather_context

    def _fake_fetch(lat, lon, start, end, timezone, cache_dir):
        return _synthetic_daily_weather(start, end)

    monkeypatch.setattr(tabs_phase2, "fetch_daily_weather", _fake_fetch)
    monkeypatch.setattr(tabs_weather_context, "fetch_daily_weather", _fake_fetch)
    # The cached loaders would replay results computed with the real fetch (or a previous
    # test's fake) -- clear them so this test's fake is actually exercised.
    tabs_phase2.load_weather_analysis.clear()
    tabs_weather_context.load_weather_context.clear()

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)
    toggle = next(t for t in at.sidebar.toggle if t.label == "Weather adjustment")
    toggle.set_value(True).run(timeout=120)

    assert list(at.exception) == []

    # Weather Impact tab: severe-weather history summary renders, and cooling stays out of
    # the main view for this (synthetically cold-climate) household.
    weather_tab = at.tabs[10]
    assert list(weather_tab.exception) == []
    subheaders = " ".join(s.value.lower() for s in weather_tab.get("subheader"))
    assert "severe weather in your history" in subheaders
    captions = " ".join(c.value.lower() for c in weather_tab.get("caption"))
    assert "too small to detect" in captions  # cooling hidden from main view

    # Unusual Months tab: December 2024 (the real data's known anomaly) gets a weather
    # context block with facts, meaning, and the honesty limitation -- no hover required.
    anomalies_tab = at.tabs[13]
    assert list(anomalies_tab.exception) == []
    anom_subheaders = " ".join(s.value.lower() for s in anomalies_tab.get("subheader"))
    assert "severe weather context" in anom_subheaders
    anom_markdown = " ".join(md.value.lower() for md in anomalies_tab.get("markdown"))
    assert "what the weather was like" in anom_markdown
    assert "what this may mean" in anom_markdown
    assert "snow day" in anom_markdown
    expander_labels = [e.label.lower() for e in anomalies_tab.get("expander")]
    assert any("advanced weather details" in label for label in expander_labels)

    # Consultant: the new severe-weather question renders a hedged, deterministic answer.
    button = next(b for b in at.tabs[1].get("button") if "severe weather" in b.label.lower())
    button.click().run(timeout=120)
    assert list(at.exception) == []
    consultant_markdown = " ".join(md.value.lower() for md in at.tabs[1].get("markdown"))
    assert "december 2024" in consultant_markdown
    assert "cannot confirm" in consultant_markdown
