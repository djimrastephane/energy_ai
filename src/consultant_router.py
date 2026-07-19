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
    answer_severe_weather,
    answer_winter_comparison,
)
from src.consultant_month import (
    answer_best_or_worst_month,
    answer_month_vs_last_year,
    answer_month_vs_typical,
    answer_should_i_be_concerned,
    answer_was_it_weather,
    answer_what_next,
    answer_which_fuel_caused_change,
    answer_why_expensive,
)

ConsultantHandler = Callable[[ConsultantContext], ConsultantAnswer]

# Month-comparison questions first -- they answer for the currently selected
# month/fuel/comparison mode (see src.consultant_month) and are the primary journey.
QUESTIONS: list[tuple[str, ConsultantHandler]] = [
    ("How did this month compare with last year?", answer_month_vs_last_year),
    ("Did I use more energy than usual this month?", answer_month_vs_typical),
    ("Which fuel caused the change?", answer_which_fuel_caused_change),
    ("Was the difference caused by weather?", answer_was_it_weather),
    ("Is this my best month on record?", answer_best_or_worst_month),
    ("Why was this month expensive?", answer_why_expensive),
    ("Should I be concerned?", answer_should_i_be_concerned),
    ("What should I do next?", answer_what_next),
    ("Why did my bill change?", answer_bill_change),
    ("What changed compared with last winter?", answer_winter_comparison),
    ("Should I focus on reducing gas or electricity?", answer_fuel_focus),
    ("What's my forecast for next year?", answer_forecast),
    ("How does my usage compare to average?", answer_benchmark),
    ("What's my carbon footprint?", answer_carbon),
    ("Was last month's usage normal, or an anomaly?", answer_last_month_anomaly),
    ("Could severe weather explain my unusual months?", answer_severe_weather),
    ("Where can I realistically save money?", answer_savings),
]

_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

_KEYWORDS: dict[str, list[str]] = {
    # Month-comparison questions first: their phrasings are more specific than the
    # older whole-period questions below, so they must win the keyword scan.
    "How did this month compare with last year?": [
        "this month compare",
        "month compare with last year",
        "compared with last year",
        "compare to last year",
        "than last year",
        "vs last year",
        "same month last year",
    ],
    "Did I use more energy than usual this month?": [
        "than usual",
        "more than usual",
        "less than usual",
        "usual this month",
        "normal for this month",
        "typical for this month",
    ],
    "Which fuel caused the change?": [
        "which fuel caused",
        "fuel caused",
        "caused the change",
        "what caused the change",
        "gas or electricity caused",
    ],
    "Was the difference caused by weather?": [
        "caused by weather",
        "because of weather",
        "because of the weather",
        "due to weather",
        "down to weather",
        "weather explain the change",
        "weather explain the difference",
        "explained by weather",
        "difference caused by",
    ],
    "Is this my best month on record?": [
        "on record",
        "my best",
        "my worst",
        "best month",
        "worst month",
        *[f"best {m}" for m in _MONTH_NAMES],
        *[f"worst {m}" for m in _MONTH_NAMES],
    ],
    "Why was this month expensive?": [
        "expensive",
        "cost so much",
        "month cost more",
        "why was this month",
    ],
    "Should I be concerned?": [
        "concerned",
        "worried",
        "should i worry",
        "worry about",
        "be alarmed",
    ],
    "What should I do next?": [
        "what should i do",
        "do next",
        "next step",
        "what now",
        "what action",
    ],
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
    # Must be checked before the last-month question: "Was the month unusually windy?"
    # contains "unusual", which would otherwise route to the anomaly handler first.
    "Could severe weather explain my unusual months?": [
        "severe weather",
        "bad weather",
        "snow",
        "blizzard",
        "storm",
        "windy",
        "wind",
        "gust",
        "kept me at home",
        "stuck at home",
        "stayed home because",
        "heavy rain",
        "heating use higher",
        "heating higher than expected",
        "higher than expected",
    ],
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
