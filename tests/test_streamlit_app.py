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
        "Seasonality & Trend",
        "Weather Adjustment",
        "Change Points",
        "Forecasting",
        "Anomaly Detection",
        "Data Quality",
    ]
    assert len(at.get("metric")) > 0

    # Executive Summary: real briefing content, not placeholders.
    assert any("overall assessment" in md.value.lower() for md in at.tabs[0].get("subheader"))
    assert any("biggest finding" in md.value.lower() for md in at.tabs[0].get("subheader"))
    assert list(at.tabs[0].exception) == []
    # Household energy profile section (Task 10) -- real data has all 3 fuels available.
    assert any("household energy profile" in s.value.lower() for s in at.tabs[0].get("subheader"))

    # AI Consultant tab: renders the question list without error.
    assert list(at.tabs[1].exception) == []
    assert any("questions i can answer" in md.value.lower() for md in at.tabs[1].get("markdown"))
    assert len(at.tabs[1].get("button")) == 8

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

    # Seasonality tab: STL runs on the real (contiguous, 35-month) dataset without error.
    assert any("seasonal" in md.value.lower() for md in at.tabs[9].get("markdown"))

    # Weather tab: toggle defaults off, so this must be the inert prompt, not a fetch attempt.
    assert any("turn on" in info.value.lower() for info in at.tabs[10].get("info"))

    # Change Points tab: renders (either a detected-points table or the "stable" message).
    assert list(at.tabs[11].exception) == []

    # Forecasting tab: CV ran and picked a model, shown as a subheader.
    assert any("selected model" in md.value.lower() for md in at.tabs[12].get("subheader"))

    # Anomaly Detection tab: renders without error, whatever it finds.
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
    assert len(at.tabs[1].get("button")) == 8  # the fallback question list still renders


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
