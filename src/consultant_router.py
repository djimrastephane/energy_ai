"""Question routing for the AI Energy Consultant -- split out of ``src.consultant`` to stay
under the ~300-line guideline (same reactive-split pattern used elsewhere in this project, e.g.
``charts.py`` -> ``charts_phase3.py``, ``comparisons.py`` -> ``cross_fuel_anomalies.py``).

``QUESTIONS`` is the single source of truth for both the clickable question list (the app UI
iterates it directly) and ``route_question``'s keyword matching, so they can never drift apart.
"""

from __future__ import annotations

from collections.abc import Callable

from src.consultant import (
    ConsultantAnswer,
    ConsultantContext,
    answer_benchmark,
    answer_bill_change,
    answer_carbon,
    answer_forecast,
    answer_fuel_focus,
    answer_last_month_anomaly,
    answer_savings,
    answer_winter_comparison,
)

ConsultantHandler = Callable[[ConsultantContext], ConsultantAnswer]

QUESTIONS: list[tuple[str, ConsultantHandler]] = [
    ("Why did my bill change?", answer_bill_change),
    ("What changed compared with last winter?", answer_winter_comparison),
    ("Should I focus on reducing gas or electricity?", answer_fuel_focus),
    ("What's my forecast for next year?", answer_forecast),
    ("How does my usage compare to average?", answer_benchmark),
    ("What's my carbon footprint?", answer_carbon),
    ("Was last month's usage normal, or an anomaly?", answer_last_month_anomaly),
    ("Where can I realistically save money?", answer_savings),
]

_KEYWORDS: dict[str, list[str]] = {
    "Why did my bill change?": [
        "why did my bill",
        "why is my bill",
        "bill increase",
        "bill went up",
        "bill go up",
        "cost more than",
        "why did it cost",
    ],
    "What changed compared with last winter?": ["winter"],
    "Should I focus on reducing gas or electricity?": ["gas or electric", "electric or gas", "which fuel", "focus on"],
    "What's my forecast for next year?": ["forecast", "predict", "next year", "future bill"],
    "How does my usage compare to average?": [
        "average household",
        "compare to average",
        "typical household",
        "benchmark",
        "vs average",
        "compared to average",
    ],
    "What's my carbon footprint?": ["carbon", "co2", "emissions", "footprint"],
    "Was last month's usage normal, or an anomaly?": ["last month", "anomaly", "unusual", "was it normal"],
    "Where can I realistically save money?": ["save money", "saving", "reduce my bill", "lower my bill", "cut cost"],
}


def route_question(text: str, ctx: ConsultantContext) -> ConsultantAnswer | None:
    """Simple keyword-overlap match against ``_KEYWORDS``. Returns ``None`` (not a guess) when
    nothing matches, so the UI falls back to the visible question list rather than pretending to
    understand a question this deterministic router can't actually parse.
    """
    normalized = text.lower().strip()
    if not normalized:
        return None
    handlers = dict(QUESTIONS)
    for question, keywords in _KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            return handlers[question](ctx)
    return None
