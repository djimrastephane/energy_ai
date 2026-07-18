import pandas as pd

from src.confidence import ConfidenceRating
from src.consultant import ConsultantContext
from src.consultant_router import QUESTIONS, route_question
from src.report import AnalystReport


def _confidence_dict():
    ok = ConfidenceRating("High", "clean")
    return {"data_quality": ok, "weather_model": ok, "forecast": ok, "anomaly_detection": ok}


def _ctx() -> ConsultantContext:
    idx = pd.date_range("2024-01-01", periods=6, freq="MS")
    clean = pd.DataFrame({"month_start": idx, "consumption_kwh": [100.0] * 6, "cost_gbp": [30.0] * 6})
    report = AnalystReport(
        executive_summary="",
        overall_assessment="Consumption is running at normal, expected levels compared to a year ago.",
        findings=[],
        biggest_finding=None,
        recommendations=[],
        largest_saving=None,
        confidence=_confidence_dict(),
        limitations=[],
        monitoring_priorities=[],
    )
    return ConsultantContext(
        analyst_report=report,
        clean=clean,
        fuel_frames={},
        fuel_energy_results={},
        anomalies=[],
        forecast_12mo=None,
        multi_fuel_forecasts=None,
        weather_enabled=False,
    )


def test_questions_list_has_eight_unique_entries():
    assert len(QUESTIONS) == 8
    assert len(set(q for q, _ in QUESTIONS)) == 8


def test_route_question_returns_none_for_empty_or_unrelated_text():
    ctx = _ctx()
    assert route_question("", ctx) is None
    assert route_question("   ", ctx) is None
    assert route_question("asdkjaslkdj random gibberish", ctx) is None


def test_route_question_matches_exact_question_text_for_every_question():
    ctx = _ctx()
    for question, _ in QUESTIONS:
        result = route_question(question, ctx)
        assert result is not None, f"'{question}' did not route to any handler"
        assert result.question == question


def test_route_question_matches_natural_phrasings():
    ctx = _ctx()
    cases = [
        ("Why did my bill increase?", "Why did my bill change?"),
        ("why is my bill so high", "Why did my bill change?"),
        ("how does this winter compare", "What changed compared with last winter?"),
        ("gas or electricity, which should I cut", "Should I focus on reducing gas or electricity?"),
        ("whats my forecast for next year", "What's my forecast for next year?"),
        ("how does my usage compare to average", "How does my usage compare to average?"),
        ("whats my carbon footprint", "What's my carbon footprint?"),
        ("was last month normal", "Was last month's usage normal, or an anomaly?"),
        ("where can I save money", "Where can I realistically save money?"),
    ]
    for phrase, expected_question in cases:
        result = route_question(phrase, ctx)
        assert result is not None, f"'{phrase}' did not match any question"
        assert result.question == expected_question, f"'{phrase}' matched {result.question!r}, expected {expected_question!r}"


def test_route_question_lower_my_bill_routes_to_savings_not_bill_change():
    """Regression guard: 'lower my bill' contains 'bill' but is a savings question, not a
    bill-change explanation -- the bill-change keyword list must not use a bare 'bill' trigger."""
    ctx = _ctx()
    result = route_question("how can I lower my bill", ctx)
    assert result is not None
    assert result.question == "Where can I realistically save money?"
