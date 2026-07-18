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
        "AI Analyst",
        "Consumption Analysis",
        "Fuel Breakdown",
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

    # AI Analyst: full deterministic report renders with every section present.
    assert list(at.tabs[1].exception) == []
    analyst_headers = " ".join(md.value.lower() for md in at.tabs[1].get("header"))
    for section in ("executive summary", "key findings", "recommendations", "confidence", "limitations"):
        assert section in analyst_headers

    # Fuel Breakdown tab: real data has all three fuel exports, so this should show the
    # cross-check status and a real (not placeholder) fuel-mix finding.
    assert list(at.tabs[3].exception) == []
    assert any("matches total" in s.value.lower() for s in at.tabs[3].get("success"))
    assert any("fuel mix" in s.value.lower() for s in at.tabs[3].get("subheader"))
    assert any("%" in md.value for md in at.tabs[3].get("markdown"))

    # Seasonality tab: STL runs on the real (contiguous, 35-month) dataset without error.
    assert any("seasonal" in md.value.lower() for md in at.tabs[5].get("markdown"))

    # Weather tab: toggle defaults off, so this must be the inert prompt, not a fetch attempt.
    assert any("turn on" in info.value.lower() for info in at.tabs[6].get("info"))

    # Change Points tab: renders (either a detected-points table or the "stable" message).
    assert list(at.tabs[7].exception) == []

    # Forecasting tab: CV ran and picked a model, shown as a subheader.
    assert any("selected model" in md.value.lower() for md in at.tabs[8].get("subheader"))

    # Anomaly Detection tab: renders without error, whatever it finds.
    assert list(at.tabs[9].exception) == []


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
