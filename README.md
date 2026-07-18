# AI Home Energy Intelligence Platform

[![Tests](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml/badge.svg)](https://github.com/djimrastephane/energy_ai/actions/workflows/tests.yml)
![coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)

A local, no-cloud analytics platform for residential energy consumption,
built on OVO Energy "Total Use" CSV exports (and, when available,
fuel-level "Electricity Use"/"Gas Use" exports -- see
[Fuel-level analysis](#fuel-level-analysis-electricity-vs-gas) below).

## Status: Phase 3 of 6, plus a decision-support layer

This is an intentionally broad, phased build. **Phase 1** covers data
ingestion, validation, descriptive statistics, and core KPIs. **Phase 2**
adds STL seasonality/trend decomposition, weather-adjusted "energy
signature" regression (via the free Open-Meteo API), and change-point
detection. **Phase 3** adds an 8-model forecasting suite (compared by
walk-forward cross-validation, auto-selected, with P10/P50/P90 bands) and
anomaly detection (rolling z-score, STL-residual ESD, Isolation Forest,
cross-referenced). On top of that, a **decision-support layer** turns all
of the above from a statistics dashboard into an evidence-based briefing:
plain-English findings, evidence-gated recommendations (never fabricated,
never solar/EV/battery without the data to support it), an investigation
checklist for anomalies/change points, and a deterministic "AI Analyst"
report -- no LLM, no free-form generation, every conclusion traceable to a
specific number computed elsewhere in the app. See `docs/roadmap.md` for
what Phases 4-6 still add, and for the real bugs the manual real-data
verification step caught along the way.

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
│   ├── streamlit_app.py   # sidebar, data loading, orchestration
│   ├── tabs_briefing.py   # Executive Summary: evidence-based briefing, not a KPI dump
│   ├── tabs_analyst.py    # AI Analyst: the full deterministic report
│   ├── tabs_core.py       # Consumption Analysis, Statistical Analysis, Data Quality
│   ├── tabs_fuel.py       # Fuel Breakdown: electricity vs. gas comparison
│   ├── tabs_phase2.py     # Seasonality, Weather Adjustment, Change Points
│   ├── tabs_phase3.py     # Forecasting, Anomaly Detection
│   ├── charts.py          # reusable Plotly chart builders (Phase 1-2)
│   ├── charts_phase3.py   # forecast fan chart, model comparison, anomaly scatter
│   └── charts_fuel.py     # electricity-vs-gas comparison charts
├── data/
│   ├── raw/                # source CSVs (OVO exports)
│   └── processed/           # weather cache (disk-cached Open-Meteo responses)
├── src/
│   ├── ingestion.py         # CSV parsing/loading, schema validation
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
│   ├── confidence.py          # shared High/Medium/Low taxonomy, used everywhere below
│   ├── findings.py            # translates stats into plain-English findings + evidence
│   ├── investigation.py       # "possible causes" checklist for anomalies/change points
│   ├── recommendations.py     # evidence-gated recommendation rules (never fabricated)
│   ├── narrative.py           # deterministic per-month plain-English narrative
│   ├── report.py              # AnalystReport: single source of truth for both report tabs
│   └── utils.py               # logging, formatting, shared helpers
├── tests/                   # pytest suite (unit + integration + UI smoke test, network-free)
├── config.py                # paths, validation thresholds, weather location -- no hard-coded paths
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

The **Anomaly Detection** tab cross-references three methods (rolling
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

The **Weather Adjustment** tab is opt-in (sidebar toggle, off by default)
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
Weather Adjustment tab could only say "consistent with *some* electric
heating" -- switching the Fuel selector to "Electricity only" now shows
directly how small that contribution actually is, rather than leaving it
to be inferred from a combined fit.
