# AI Home Energy Intelligence Platform

[![Tests](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml/badge.svg)](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

An evidence-based energy analyst for your home -- not a statistics
dashboard. Runs entirely on your machine from OVO Energy monthly billing
CSV exports: no cloud, no LLM, no API keys. Every conclusion is
deterministic and traceable to a specific computed number, carries a
confidence rating with its reason, and says "not enough evidence" instead
of guessing.

![Home: this month compared with last year](docs/screenshots/home.jpg)

## What it can tell you

Real answers from the author's real 35-month household dataset:

| You ask | It answers |
|---|---|
| How did this month compare with last year? | *"You used 14% more energy than June 2025, mainly because electricity use rose. Gas use actually fell. Worth a quick look: this is your highest June on record."* |
| Should I focus on reducing gas or electricity? | *"Gas accounts for 65% of annual energy and drives 89% of heating sensitivity; reducing gas demand is likely to produce larger savings than reducing electricity use."* |
| Why was December 2024 unusual? | *"December 2024 had mixed severe weather (5 severe-gust days, max 91 km/h), which may have contributed to higher usage -- but consumption remained +219 kWh above what the temperature model expected, so severe weather alone does not explain the full increase."* |
| How does my usage compare to average? | *"Electricity usage is below average for a UK household (1,555 kWh/year vs a 2,500 kWh/year reference). Gas usage is below average (2,863 vs 9,500)."* |

## Features

- **Month-first journey** -- the default view compares the latest
  *complete* month with the same calendar month last year (never January
  vs December: that mostly measures the seasons, and never a partial
  month presented as complete). Alternative modes: previous month,
  typical/best/worst same-calendar month, long-term trend. Changes are
  split into weather-explained and unexplained parts, attributed to gas
  vs electricity, and judged in words ("Higher but weather-explained"),
  with documented practical thresholds -- see
  [docs/month_comparison.md](docs/month_comparison.md).

- **AI Consultant & AI Analyst** -- 17 canonical questions with free-text
  routing (month-comparison questions answer for the currently selected
  month, fuel, and comparison mode -- never stale state), plus a full
  written report. Deterministic (no LLM): every answer cites its
  evidence, and unmatched questions get the question list back, never a
  guess.
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

## Screenshots

**How did this month compare?** -- the primary journey: latest complete
month vs the same calendar month last year, with fuel and comparison-mode
controls and the two-period chart.

![Month comparison: June 2026 vs June 2025](docs/screenshots/month_comparison.jpg)

**What should I expect next?** -- answer-first forecasting: expected use
and confidence lead; the model and the plausible range live in collapsed
detail sections, and the uncertainty band visibly fades with the horizon.

![Forecast: expected use, confidence, and the fading plausible range](docs/screenshots/forecast.jpg)

## Quick start

```bash
cd energy_ai
python3 -m venv .venv
./.venv/bin/pip install -r requirements.lock && ./.venv/bin/pip install -e . --no-deps
./.venv/bin/streamlit run app/streamlit_app.py
```

Tests and linting (the suite is network-free -- weather calls are mocked
or served from the disk cache):

```bash
./.venv/bin/pytest -v
./.venv/bin/ruff check src tests app config.py scripts
./.venv/bin/mypy
```

Dependencies are declared in `pyproject.toml` (floor versions); CI and the
quick start install from `requirements.lock`, the exact known-good set. To
upgrade: bump the floor in `pyproject.toml` if needed, then regenerate the
lock with `pip install -e ".[dev]" && pip freeze --exclude-editable > requirements.lock`.

## Your data

Drop OVO's "Total Use" (and optionally "Electricity Use"/"Gas Use") CSV
exports into `data/raw/`, or upload them from the sidebar. These are
*monthly* billing summaries (`Month, Cost (£), Consumption (kWh)`), not
smart-meter readings -- every analysis here is scoped to what ~35 monthly
observations honestly support, rather than faking daily-data resolution.
Nothing assumes the input is clean: duplicates, conflicts, missing months,
and implausible values are detected and reported on the Data Quality tab.

## Design principles

- **Never invent a number.** Standing charges and VAT aren't in the
  exports, so bill totals only appeared once the user supplied the actual
  tariff facts (62.77p/34.97p per day standing, 5% VAT, 6th-to-5th billing
  cycle -- `config.BillingConfig`); every bill figure is labelled as the
  estimate it is, component by component. No solar/EV/battery
  recommendation exists because no roof/vehicle/appliance data exists --
  enforced by the absence of a code path, not a runtime check.
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
├── app/          # Streamlit UI: 9 question-oriented tabs, sidebar, charts, HTML report
├── src/          # all analysis logic -- no Streamlit imports, unit-testable directly
├── data/raw/     # your OVO CSV exports (data/processed/ holds the weather cache)
├── docs/         # roadmap, month-comparison + weather data dictionaries, audit reports
├── tests/        # pytest suite: unit + integration + UI (AppTest), network-free
└── config.py     # paths, thresholds, weather/benchmark/carbon constants -- all in one place
```

Navigation is organized around user questions -- Home, "How did this
month compare?", "What drives my usage?", "Costs and carbon", "Did
anything unusual happen?", "What should I expect next?", "Ask the Energy
Consultant" -- with whole-period views under **Long-term trends** and the
statistical machinery (full AI Analyst report, diagnostics, data
quality) under **Data and methods**: demoted, never deleted.

Each module carries a docstring explaining what it does and why -- the
layout above is deliberately shallow; start at `app/streamlit_app.py` or
`src/report.py` and follow the imports.

## More documentation

- [docs/roadmap.md](docs/roadmap.md) -- the full build history, phase by
  phase, including every real bug that manual verification against real
  data caught along the way.
- [docs/month_comparison.md](docs/month_comparison.md) -- why
  same-month-last-year is the default, how partial months are handled,
  the weather decomposition, change-category thresholds, judgement
  labels, and edge-case fallbacks.
- [docs/weather_context.md](docs/weather_context.md) -- weather data
  dictionary, severe-weather thresholds and sources, cache versioning,
  interpretation rules.
- [docs/audits/](docs/audits/) -- a full engineering/security/performance/
  UX audit with before/after measurements.
