# AI Home Energy Intelligence Platform

[![Tests](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml/badge.svg)](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

An evidence-based energy analyst for your home -- not a statistics
dashboard. Runs entirely on your machine from OVO Energy monthly billing
CSV exports: no cloud, no LLM, no API keys. Every conclusion is
deterministic and traceable to a specific computed number, carries a
confidence rating with its reason, and says "not enough evidence" instead
of guessing.

## What it can tell you

Real answers from the author's real 35-month household dataset:

| You ask | It answers |
|---|---|
| Should I focus on reducing gas or electricity? | *"Gas accounts for 65% of annual energy and drives 89% of heating sensitivity; reducing gas demand is likely to produce larger savings than reducing electricity use."* |
| Why was December 2024 unusual? | *"December 2024 had mixed severe weather (5 severe-gust days, max 91 km/h), which may have contributed to higher usage -- but consumption remained +219 kWh above what the temperature model expected, so severe weather alone does not explain the full increase."* |
| What changed compared with last winter? | *"Winter 2025/2026 used 21% less energy than winter 2024/2025 (1,720 kWh vs 2,187 kWh)."* |
| How does my usage compare to average? | *"Electricity usage is below average for a UK household (1,555 kWh/year vs a 2,500 kWh/year reference). Gas usage is below average (2,863 vs 9,500)."* |

## Features

- **AI Consultant & AI Analyst** -- 9 canonical questions with free-text
  routing, plus a full written report. Deterministic (no LLM): every answer
  cites its evidence, and unmatched questions get the question list back,
  never a guess.
- **Weather-adjusted analysis** -- the standard degree-day "energy
  signature" regression (Open-Meteo, free, cached to disk for offline use)
  separates weather from behaviour; snow/wind/severe-weather context
  explains unusual months without ever claiming causation
  ([docs/weather_context.md](docs/weather_context.md)). Cooling terms stay
  in the model for reuse in warmer climates, but for a home without air
  conditioning -- like this Aberdeen dataset, where cooling degree days
  are zero across the entire history -- cooling is reported as
  undetectable and kept out of the main view, never fabricated.
- **Forecasting** -- 8 models (naive through SARIMA, Prophet, XGBoost,
  LightGBM) compared by walk-forward cross-validation, with P10/P50/P90
  bands. On the real data, Seasonal Naive wins -- an honest finding that
  complexity doesn't help at ~35 monthly observations.
- **Unusual-month detection** -- three cross-referenced anomaly methods
  plus change-point detection, each flagged month getting a "possible
  causes" checklist that only ticks what the data actually supports.
- **Multi-fuel intelligence** -- a sidebar Fuel selector drives the whole
  app as Total, Electricity-only, or Gas-only; cross-fuel anomaly
  attribution, cost breakdowns, CO2e estimates (cited factors), and
  UK/Scotland benchmarking.
- **Household Energy Review** -- a downloadable, self-contained HTML report
  (interactive charts, print stylesheet for Save-as-PDF).

## Quick start

```bash
cd energy_ai
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/streamlit run app/streamlit_app.py
```

Tests and linting (the suite is network-free -- weather calls are mocked
or served from the disk cache):

```bash
./.venv/bin/pytest -v
./.venv/bin/ruff check src tests app config.py scripts
```

## Your data

Drop OVO's "Total Use" (and optionally "Electricity Use"/"Gas Use") CSV
exports into `data/raw/`, or upload them from the sidebar. These are
*monthly* billing summaries (`Month, Cost (£), Consumption (kWh)`), not
smart-meter readings -- every analysis here is scoped to what ~35 monthly
observations honestly support, rather than faking daily-data resolution.
Nothing assumes the input is clean: duplicates, conflicts, missing months,
and implausible values are detected and reported on the Data Quality tab.

## Design principles

- **Never invent a number.** Standing charges aren't in the exports, so no
  £ figure pretends to include them. No solar/EV/battery recommendation
  exists because no roof/vehicle/appliance data exists -- enforced by the
  absence of a code path, not a runtime check.
- **Evidence and confidence on everything.** Findings and recommendations
  carry their evidence lines, a High/Medium/Low rating, and the reason for
  that rating; builders return nothing rather than fabricate.
- **Hedged, not causal.** Monthly bills can't prove behaviour: wording is
  "may have contributed" / "cannot be confirmed from monthly data", tested
  to never assert home-working or occupancy as fact.
- **Prefer simple models**, and say so when they win.

## Project layout

```
energy_ai/
├── app/          # Streamlit UI: 15 tabs, sidebar, charts, HTML report builder
├── src/          # all analysis logic -- no Streamlit imports, unit-testable directly
├── data/raw/     # your OVO CSV exports (data/processed/ holds the weather cache)
├── docs/         # roadmap, weather-context data dictionary, audit reports
├── tests/        # pytest suite: unit + integration + UI (AppTest), network-free
└── config.py     # paths, thresholds, weather/benchmark/carbon constants -- all in one place
```

Each module carries a docstring explaining what it does and why -- the
layout above is deliberately shallow; start at `app/streamlit_app.py` or
`src/report.py` and follow the imports.

## More documentation

- [docs/roadmap.md](docs/roadmap.md) -- the full build history, phase by
  phase, including every real bug that manual verification against real
  data caught along the way.
- [docs/weather_context.md](docs/weather_context.md) -- weather data
  dictionary, severe-weather thresholds and sources, cache versioning,
  interpretation rules.
- [docs/audits/](docs/audits/) -- a full engineering/security/performance/
  UX audit with before/after measurements.
