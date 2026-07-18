# AI Home Energy Intelligence Platform

[![Tests](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml/badge.svg)](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

A local, no-cloud analytics platform for residential energy consumption,
built on OVO Energy "Total Use" CSV exports (and, when available,
fuel-level "Electricity Use"/"Gas Use" exports -- see
[Fuel-level analysis](#fuel-level-analysis-electricity-vs-gas),
[Phase 4: household energy intelligence](#phase-4-household-energy-intelligence),
and the [AI Energy Consultant](#ai-energy-consultant) below).

## Status: Phase 4 of 6, plus a decision-support layer and an AI Consultant

This is an intentionally broad, phased build. **Phase 1** covers data
ingestion, validation, descriptive statistics, and core KPIs. **Phase 2**
adds STL seasonality/trend decomposition, weather-adjusted "energy
signature" regression (via the free Open-Meteo API), and change-point
detection. **Phase 3** adds an 8-model forecasting suite (compared by
walk-forward cross-validation, auto-selected, with P10/P50/P90 bands) and
anomaly detection (rolling z-score, STL-residual ESD, Isolation Forest,
cross-referenced). A **decision-support layer** turns all of the above from
a statistics dashboard into an evidence-based briefing: plain-English
findings, evidence-gated recommendations (never fabricated, never solar/EV/
battery without the data to support it), an investigation checklist for
anomalies/change points, and a deterministic "AI Analyst" report -- no LLM,
no free-form generation, every conclusion traceable to a specific number
computed elsewhere in the app. **Phase 4** compares Electricity/Gas/Total
directly, adds cost intelligence, carbon estimates, and UK/Scotland
benchmarking. The **AI Energy Consultant** (see below) is a bounded,
deterministic Q&A interface over everything above -- still no LLM. See
`docs/roadmap.md` for what Phase 5 (SHAP/explainability) and Phase 6 add,
and for the real bugs the manual real-data verification step caught along
the way.

**A note on the data:** the OVO exports in `data/raw/` are *monthly* billing
summaries (`Month, Cost (£), Consumption (kWh)`), not daily or half-hourly
meter readings. Every module here is scoped to what's honestly supportable
from ~35 monthly observations -- e.g. the "Consumption Analysis" tab omits
calendar heatmaps and duration curves (which need daily data) rather than
faking that resolution. Monthly billing-data analysis is a real discipline
in its own right (utility "energy signature" regression is standard
practice), just a different one than daily smart-meter analytics.

## Setup

```bash
cd energy_ai
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Running the app

```bash
./.venv/bin/streamlit run app/streamlit_app.py
```

By default it loads every matching `*.csv` in `data/raw/`. You can also
upload files directly from the sidebar (they don't need to be saved to disk
first). A sidebar **Fuel** selector picks whether "Total", "Electricity
only", or "Gas only" data drives the whole app -- "Electricity only"/"Gas
only" only appear once matching exports exist.

## Running tests and linting

```bash
./.venv/bin/pytest -v
./.venv/bin/ruff check src tests app config.py
```

## Project layout

```
energy_ai/
├── app/
│   ├── streamlit_app.py   # top-level orchestration, kept under ~300 lines
│   ├── sidebar.py         # sidebar controls + fuel-aware data loading
│   ├── multi_fuel.py      # eagerly computes STL/anomalies/weather for all 3 fuels
│   ├── tabs_briefing.py   # Executive Summary: evidence-based briefing, not a KPI dump
│   ├── tabs_consultant.py # AI Consultant: bounded Q&A, free text + preset question list
│   ├── tabs_analyst.py    # AI Analyst: the full deterministic report
│   ├── tabs_core.py       # Consumption Analysis, Statistical Analysis, Data Quality
│   ├── tabs_fuel.py       # Fuel Breakdown: electricity vs. gas comparison
│   ├── tabs_comparisons.py # Comparisons: Electricity vs. Gas vs. Total, side by side
│   ├── tabs_cost.py       # Cost Intelligence: billing breakdown, forecast bills, benchmarking
│   ├── tabs_carbon.py     # Carbon: estimated CO2e emissions
│   ├── tabs_phase2.py     # Seasonal Patterns, Weather Impact, Usage Shifts
│   ├── tabs_phase3.py     # Forecasting, Unusual Months (anomaly detection)
│   ├── charts.py          # reusable Plotly chart builders (Phase 1-2)
│   ├── charts_phase3.py   # forecast fan chart, model comparison, anomaly scatter
│   ├── charts_fuel.py     # electricity-vs-gas comparison charts
│   ├── charts_comparisons.py # annual comparison chart across fuels
│   ├── report_html.py     # Household Energy Review: self-contained HTML report builder
│   └── report_template.html # Jinja2 template for the report (screen + print CSS)
├── data/
│   ├── raw/                # source CSVs (OVO exports)
│   └── processed/           # weather cache (disk-cached Open-Meteo responses)
├── src/
│   ├── ingestion.py         # CSV parsing/loading, schema validation, EnergyType
│   ├── preprocessing.py     # dedup, validation, missing-month detection, enrichment
│   ├── statistics.py        # descriptive stats (mean/median/CV/skew/CI/bootstrap)
│   ├── kpis.py               # period-over-period KPI comparisons
│   ├── decomposition.py      # STL trend/seasonal/residual + strength metrics
│   ├── weather.py            # Open-Meteo fetch + degree days
│   ├── energy_signature.py   # degree-day regression + weather-adjusted annual comparison
│   ├── changepoints.py       # PELT + CUSUM change-point detection
│   ├── forecast_models.py    # naive, seasonal-naive, linear trend, Holt-Winters, SARIMA
│   ├── forecast_ml.py        # Prophet, XGBoost, LightGBM
│   ├── forecast_evaluation.py # walk-forward CV, model registry, auto-selection
│   ├── forecast_uncertainty.py # P10/P50/P90 via residual bootstrap
│   ├── esd.py                 # generalized ESD (Rosner) outlier test
│   ├── anomalies.py           # rolling z-score + STL-ESD + Isolation Forest, cross-referenced
│   ├── fuel.py                 # electricity/gas cross-check, combining, fuel-mix finding
│   ├── comparisons.py          # annual/monthly/seasonality/weather-sensitivity comparisons
│   ├── cross_fuel_anomalies.py # cross-references per-fuel anomalies to attribute a cause
│   ├── cost_engine.py          # per-fuel/combined cost breakdown, forecast bill by fuel
│   ├── benchmarking.py         # Below/Average/Above-average bands vs. UK/Scotland
│   ├── carbon.py                # CO2e emissions estimates, static cited factors
│   ├── confidence.py          # shared High/Medium/Low taxonomy, used everywhere below
│   ├── findings.py            # translates stats into plain-English findings + evidence
│   ├── investigation.py       # "possible causes" checklist for anomalies/change points
│   ├── recommendations.py     # evidence-gated recommendation rules (never fabricated)
│   ├── narrative.py           # deterministic per-month plain-English narrative
│   ├── report.py              # AnalystReport: single source of truth for both report tabs
│   ├── consultant.py           # AI Consultant: 8 canonical question handlers, no LLM
│   ├── consultant_router.py    # keyword-match routing for the free-text question box
│   └── utils.py               # logging, formatting, shared helpers
├── tests/                   # pytest suite (unit + integration + UI smoke test, network-free)
├── config.py                # paths, thresholds, weather/benchmark/carbon config -- no hard-coded paths
└── requirements.txt
```

## Data quality philosophy

Nothing here assumes the input is clean. `src/preprocessing.py` explicitly:

- removes exact duplicate rows and reports how many
- resolves same-month conflicts across files deterministically (most
  recently loaded file wins) and reports every conflict
- flags (never silently drops) implausible cost/consumption values
- detects and reports missing months within the available date range

The results are surfaced verbatim in the app's **Data Quality** tab.

## Forecasting and anomaly detection

The **Forecasting** tab runs 8 models (naive, seasonal-naive, linear trend,
Holt-Winters, SARIMA, Prophet, XGBoost, LightGBM), compares them by
1-step-ahead walk-forward cross-validation, and auto-selects the lowest-MAE
model (overridable from the sidebar). On the real data, **Seasonal Naive
wins** -- a genuine finding that model complexity doesn't help at ~35
months of history, not a shortcoming to paper over. Uncertainty bands
(P10/P50/P90) come from bootstrapping the chosen model's own cross-validation
residuals, not each library's own interval machinery, and are floored at
zero (kWh can't be negative).

The **Unusual Months** tab (anomaly detection) cross-references three methods (rolling
z-score, STL-residual generalized ESD, Isolation Forest) so a month flagged
by 2+ methods reads as meaningfully more confident than a single-method
flag -- important because the median/MAD "Hybrid" ESD variant has a real,
measured false-positive rate well above its nominal alpha at this sample
size (documented in `src/esd.py`).

Manual verification against the real data (not just synthetic unit tests)
caught three genuine bugs during this phase, each now covered by a
regression test: Prophet's optimizer could take 60x longer on unusually
regular data (capped `iter`), the residual bootstrap could forecast
negative kWh (floored at zero), and a naive rolling z-score's baseline
window included the very point being tested, which could mask an obvious
spike (fixed by shifting the baseline to exclude it).

## Decision-support layer: findings, recommendations, and the AI Analyst

Everything from Phases 1-3 answers "what happened statistically." This
layer (`src/confidence.py`, `src/findings.py`, `src/investigation.py`,
`src/recommendations.py`, `src/narrative.py`, `src/report.py`) turns that
into "what should I do about it" -- entirely by synthesizing outputs that
already exist, with **no new statistical computation**:

- **Confidence** (`src/confidence.py`) is rated High/Medium/Low for data
  quality, the weather model, the forecast, and anomaly detection, each
  with a one-line reason -- the same taxonomy everywhere so "High" always
  means the same thing.
- **Findings** (`src/findings.py`) translate numbers into sentences (R² of
  0.73 becomes "weather explains about three quarters of the variation")
  and return `None` -- not a fabricated finding -- when there isn't enough
  evidence.
- **Recommendations** (`src/recommendations.py`) fire only when a specific,
  checkable condition is met (e.g. the latest winter's actual consumption
  exceeds the weather model's prediction by more than a statistically
  calibrated threshold -- verified empirically at a ~5-6% false-positive
  rate on pure noise, not just asserted). There is no code path that can
  recommend solar, EV charging, or a battery, because this dataset has no
  roof, vehicle, or appliance data to evaluate them against -- "never
  recommend without sufficient information" is enforced by what code
  exists, not a runtime check.
- **Investigation checklists** (`src/investigation.py`) list possible causes
  for a flagged anomaly or change point (colder weather, tariff change,
  new appliance, ...) and tick only the ones the data actually supports;
  most stay honestly unchecked, since billing data alone can't see
  occupancy or appliances.
- **The AI Analyst tab** renders `src/report.py`'s `AnalystReport` in full
  (findings, recommendations, confidence, limitations, monitoring
  priorities, and a per-month narrative for the whole history); the
  **Executive Summary** tab renders a condensed version of the exact same
  object, so the two can never contradict each other.

Manual verification against the real data caught a genuine calibration bug
here too: an early version of the heating-review threshold compared a
3-month residual *sum* against a single month's standard deviation, which
fires on pure noise ~15% of the time. Fixed with proper variance
propagation (`sqrt(3) * std`) plus a standard one-sided 95% threshold,
re-verified empirically at ~5-6%.

## Weather adjustment

The **Weather Impact** tab is opt-in (sidebar toggle, off by default)
since it fetches historical daily temperature from the free Open-Meteo
archive API for the configured location (`config.py`'s `WeatherConfig`,
currently Aberdeen/AB21). Results are cached to `data/processed/` so
subsequent runs work offline. If the fetch fails and no cache exists, the
tab shows a clear warning instead of crashing the app. The regression
(monthly avg. daily kWh ~ heating-degree-days + cooling-degree-days) is the
standard "energy signature" method utility analysts use on billing data; a
feature with zero variance in the data (e.g. cooling degree days in a
climate that never crosses the cooling threshold) is reported as "couldn't
be estimated" rather than a misleading p-value.

## Fuel-level analysis: electricity vs. gas

OVO also exports "Electricity Use" and "Gas Use" breakdowns alongside the
combined "Total Use" file, for the same months. `src/ingestion.py`
discovers all three kinds (`FUEL_FILE_PATTERNS`) but keeps them separate --
mixing all three for the same month would make every month look like a
source conflict. The sidebar's **Fuel** selector picks which of the three
cleaned DataFrames flows into the *entire rest of the app*: every existing
tab, finding, recommendation, and the AI Analyst report becomes fuel-aware
for free, with zero duplicated analysis logic, because none of Phases 1-3
or the decision-support layer is Total-specific by name.

`src/fuel.py` adds only what's genuinely new for having three correlated
sources:

- **`cross_check_fuel_totals`** verifies Electricity + Gas == Total for
  every overlapping month (within a small rounding tolerance) -- "never
  assume the data is clean" applied to the new capability itself, surfaced
  in both the **Fuel Breakdown** and **Data Quality** tabs. On the real
  data: zero mismatches across all 35 months.
- **`finding_fuel_mix`** compares gas vs. electricity consumption, cost,
  and winter/summer seasonality directly from the combined billing data
  (`None` below 6 overlapping months) -- no new modeling.

**A genuinely useful finding this unlocked**, found during manual
verification: fitting the *existing, unmodified* weather-adjusted "energy
signature" regression separately on Electricity-only and Gas-only data
(instead of only ever seeing it fit on the combined Total) shows heating in
this house is overwhelmingly **gas-driven**:

| Fuel | Heating slope (kWh/day per HDD) | R² | p-value |
|---|---|---|---|
| Total | 1.661 | 0.73 | <0.0001 |
| Gas | 1.472 (89% of Total's) | 0.69 | <0.0001 |
| Electricity | 0.189 (11% of Total's) | 0.34 | 0.0003 |

Electricity does have a small, statistically real weather sensitivity (not
noise) -- plausibly an immersion heater or more device/lighting use on cold
days -- but gas carries the large majority of the heating signal. On the
real billing data overall, gas is ~65% of consumption but only ~31% of
cost (it's the cheaper fuel per kWh), and swings ~8x between winter and
summer versus ~1.6x for electricity. Before this feature, the Total-only
Weather Impact tab could only say "consistent with *some* electric
heating" -- switching the Fuel selector to "Electricity only" now shows
directly how small that contribution actually is, rather than leaving it
to be inferred from a combined fit.

## Phase 4: household energy intelligence

Requested directly by the user: turn the app from "electricity analytics"
into genuine household energy intelligence -- explain *why* energy changed,
*which fuel* caused it, add cost/carbon estimation, and benchmark against
published averages, reusing the existing statistical engine rather than
duplicating it. No `EnergyStream` wrapper class was built: the `fuel: str`
/ sidebar-selector pattern from the fuel-level-analysis phase already made
every earlier module generic over Electricity/Gas/Total, so this phase only
adds what's genuinely new -- three new tabs (**Comparisons**, **Cost
Intelligence**, **Carbon**) and household-level sections on the Executive
Summary and AI Analyst tabs.

**`src/comparisons.py` + `src/cross_fuel_anomalies.py`** (Comparisons tab):
annual/monthly/seasonality tables across fuels; `compare_weather_sensitivity`
formalizes the fuel-level-analysis payoff check into a reusable `Finding`
(**gas explains 89%** of the two fuels' combined heating-driven response,
electricity 11%); `cross_fuel_anomaly_insights` cross-references each
fuel's independently-detected anomalies to say *which* fuel is responsible
for a flagged month:

| Pattern | Interpretation |
|---|---|
| Electricity flagged, gas unchanged | Likely appliance/occupancy change |
| Gas flagged, electricity unchanged | Likely a heating event |
| Both flagged, same direction, within weather-adjusted expectation | Likely weather-driven |
| Both flagged, same direction, still exceeds weather-adjusted expectation | Partially unexplained -- worth investigating |
| Both flagged, **opposite** directions | Not a shared driver -- attributed to the stronger signal, conflict stated explicitly |

A real bug this design caught: December 2024 (the known 738 kWh outlier
from Phase 3) has electricity *dropping* 149 kWh (1 method, weather-adjusted
z≈-0.01 -- essentially exactly at its predicted level) while gas *spiked*
589 kWh (all 3 methods, weather-adjusted z=+1.96 -- still elevated even
after accounting for that month's cold weather). An early version of the
classifier only checked "were both fuels flagged," which called this "both
moved together" -- simply false when the directions disagree. Fixed to
detect direction conflicts explicitly; the real classification is now
*"the gas signal is stronger (3 methods vs 1), so this looks more like a
gas-specific event than something affecting the whole household"* -- High
confidence, with the conflicting electricity signal stated as evidence, not
hidden.

**`src/cost_engine.py`** (Cost Intelligence tab): per-fuel and combined
billing breakdown, and `forecast_bill_by_fuel` (converts each fuel's
already-computed forecast to £, opt-in via a button since it triples the
walk-forward-CV cost of the single-fuel Forecasting tab). **Standing
charges are deliberately omitted** -- the OVO exports have no standing-
charge or tariff-rate column at all (only `Month, Cost (£), Consumption
(kWh)`), so fabricating a fixed/variable split would violate this
platform's "never invent a number" rule. On the real data, cost reconciles
exactly: Electricity (£1,200.13) + Gas (£535.70) = Total (£1,735.83), to
the penny.

**`src/benchmarking.py`** (also in Cost Intelligence): Below/Average/
Above-average bands (±15% tolerance, deliberately never a single "Energy
Score") against Ofgem's Typical Domestic Consumption Values (medium usage,
2,500 kWh electricity / 9,500 kWh gas per year, 2026) and Scotland-specific
average electricity consumption (3,429 kWh/year, DESNZ/ONS sub-national
statistics). No Scotland-specific *gas* consumption benchmark was found
during research -- the UI says so explicitly rather than silently reusing
the UK-wide gas figure under a "Scotland" label.

**`src/carbon.py`** (Carbon tab): monthly/annual/weather-adjusted/forecast
CO2e emissions from static, cited factors (`config.CarbonConfig`:
electricity 0.207, gas 0.183 kgCO2e/kWh -- DESNZ/DEFRA GHG Conversion
Factors). Only supported for Electricity/Gas individually -- "Total" can't
be honestly split into a blended factor without knowing the mix. On the
real data (2024, the most recent complete calendar year): 303 kg CO2e from
electricity + 477 kg CO2e from gas = **0.78 tonnes CO2e combined**,
independently cross-checked by manual recomputation from raw annual kWh ×
factor (matches to the gram). Gas is 64% of combined kWh but only 61% of
combined emissions, since it has a lower emission factor per kWh than grid
electricity.

**`src/recommendations.py`** gained `recommend_fuel_focus`: fires only when
one fuel dominates *both* consumption/cost share and weather-sensitivity
share (>60% each), matching the spec's own worked example almost exactly.
On the real data it fires: *"Gas accounts for 65% of annual energy and
drives 89% of heating sensitivity; reducing gas demand is likely to produce
larger savings than reducing electricity use."* -- High confidence, since
both fuels' heating sensitivity is statistically significant.

The AI Analyst tab gained **Household Energy Profile**, **Electricity
Findings**, and **Gas Findings** sections (from `AnalystReport`'s new
`fuel_mix_finding`/`weather_sensitivity_finding`/`per_fuel_findings`
fields); the Executive Summary gained a condensed version. Both stay fully
deterministic and traceable, per the project's existing pattern.

Coverage: 97% on `src/`, 258 tests total (up from 200). Deferred, not built
in this phase: tariff-switch/solar/battery/heat-pump/EV cost-optimisation
calculators (no roof/vehicle/appliance data exists to evaluate them
against), live `carbonintensity.org.uk` integration (carbon stays
offline-first with static factors), and Monte Carlo simulation -- see
`docs/roadmap.md`.

## AI Energy Consultant

Requested directly by the user, after Phase 4: *"the analytics engine is
now mature"* -- the next investment should be an interface over it, not
more analysis. This is this project's original Phase 5 sketch, built early:
*"A grounded, deterministic Q&A engine over the computed statistics (no
LLM/paid API) -- answers only from data already computed, always citing the
underlying numbers."*

**The consultant explains analysis that's already been run -- it never runs
new analysis.** Every answer comes from objects `main()` already computes
(`AnalystReport`, the per-fuel result dicts, the forecast); visiting the
tab triggers zero new computation. Without an LLM, "understands any
question" wouldn't be honest, so it supports a bounded set of **8
canonical questions**, each with its own deterministic handler in
`src/consultant.py`. A free-text box (`app/tabs_consultant.py`) routes via
simple keyword matching (`src/consultant_router.py`) to the same 8
handlers, falling back to a visible clickable list -- not a guess -- when
nothing matches confidently.

Real answers this gives on the household's actual data (Total fuel, 35
months, weather adjustment on):

| Question | Answer |
|---|---|
| Why did my bill change? | *"Consumption is running at normal, expected levels compared to a year ago. Largest cost driver: Electricity (69% of combined cost). 2025: actual 4,418 kWh vs. weather-adjusted 4,157 kWh (+261 kWh) -- well explained by weather."* |
| What changed compared with last winter? | *"Winter 2025/2026 used 21% less energy than winter 2024/2025 (1,720 kWh vs 2,187 kWh), costing £204.85 vs £219.11."* |
| Should I focus on reducing gas or electricity? | *"Gas accounts for 65% of annual energy and drives 89% of heating sensitivity; reducing gas demand is likely to produce larger savings than reducing electricity use."* |
| How does my usage compare to average? | *"Electricity usage is below average for a UK household (1,555 kWh/year vs a 2,500 kWh/year reference). Gas usage is below average (2,863 kWh/year vs a 9,500 kWh/year reference)."* |
| What's my carbon footprint? | *"In 2025, estimated emissions were 0.85 tonnes CO2e -- 326 kg from electricity and 520 kg from gas."* |

The one genuinely new piece of logic this needed:
`src.kpis.winter_over_winter_comparison` (pure aggregation, sums kWh/cost
per winter season -- not a new statistic); `winter_season_label` was
promoted from `src/recommendations.py` to `src/utils.py` so both modules
share the same December-groups-with-following-Jan/Feb convention.

A real bug caught while building this: *"Where can I realistically save
money?"* initially picked whichever recommendation happened to be first in
the list -- which could be "Collect more historical data" (real advice,
but not itself a savings action) even when a more directly relevant
recommendation like "Focus on gas" had also fired. Fixed to prefer any
actionable recommendation over that one.

Coverage: 97% on `src/`, 291 tests total (up from 258).

## Household Energy Review (downloadable report)

The sidebar's **Report** section generates a self-contained HTML
"Household Energy Review" -- executive summary, key findings, fuel mix,
weather analysis, consumption history, forecast, recommendations, carbon,
benchmark, methodology, and limitations -- rendered entirely from analysis
the app has already computed (`app/report_html.py`, no new statistics).
Charts are interactive in a browser; the built-in print stylesheet means
the browser's Print -> "Save as PDF" produces a print-quality PDF. The file
embeds plotly.js inline (~5 MB) so it works fully offline, and Jinja2
autoescaping is on. Generation is a deliberate two-step (Generate ->
Download) with a context fingerprint, so a report generated for one
fuel/weather/data state is discarded rather than served stale after the
context changes. Zero new dependencies: jinja2 ships with Streamlit.
In-app PDF (reportlab + kaleido) and an Excel appendix were considered and
consciously deferred.
