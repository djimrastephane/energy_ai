# Month-Comparison UX Audit (before implementation)

Date: 2026-07-19. Scope: refactor the primary user journey around
"latest complete month vs. the same calendar month last year", per the
same-month-comparison UX brief. This audit records the current state
before any code changes, per Phase 1 of that brief.

## Baseline measurements

| Metric | Value |
|---|---|
| Test count | 378 passed, 0 failed |
| Test suite wall time | ~61 s |
| Coverage (src + app + config) | 95% (3,698 stmts, 183 missed) |
| Cold first render (AppTest, weather off) | 5.51 s |
| Warm rerun | 0.19 s |
| Fuel-switch rerun | 3.52 s |
| Top-level tabs | 15 |
| AppTest scenarios | 11 (in `tests/test_streamlit_app.py`) |
| Data | 35 months/fuel (Sep 2023 – Jul 2026), no gaps, all 3 fuels |

Latest month in the real data (July 2026) is **partial** — 107.87 kWh
month-to-date on 2026-07-19 vs. 150.59 kWh for the full July 2025. The
latest **complete** month is June 2026: 284.09 kWh vs. June 2025's
249.47 kWh = **+13.9%** — the natural headline the app currently never
states anywhere.

## Current navigation (15 tabs)

Executive Summary · AI Consultant · AI Analyst · Consumption Analysis ·
Fuel Breakdown · Comparisons · Cost Intelligence · Carbon · Statistical
Analysis · How the Seasons Affect Usage · Weather Impact · Usage Shifts ·
Forecasting · Unusual Months · Data Quality

Observations:

1. **No surface answers "how did this month compare?"** The closest
   things are buried: `src/narrative.py`'s per-month YoY paragraph
   (inside AI Analyst → "Monthly Narrative History", collapsed), and the
   Comparisons tab's whole-period annual/monthly tables.
2. **Whole-period statistics dominate.** The Executive Summary opens
   with *trailing-12-month* totals (`compute_kpis(months=12)`), then an
   overall assessment, biggest finding, and annual forecast — all
   whole-period or 12-month-window framings. Consumption Analysis,
   Statistical Analysis, Comparisons, Cost Intelligence, and Carbon all
   lead with full-history charts/annual tables.
3. **Method-named / technical tabs at top level:** Statistical Analysis,
   Usage Shifts (change points), Forecasting's model-comparison table,
   Unusual Months' method columns. An earlier audit already renamed the
   worst offenders, but they still occupy primary navigation.
4. **The partial current month is flagged but not excluded.** `
   flag_in_progress_month` warns (Data Quality tab + Consultant's
   "last month" answer), but the KPI strip, charts, anomaly scoring and
   narratives still treat July 2026 as a full month.

## Existing comparison logic traced

| Concern | Where computed | Notes for reuse |
|---|---|---|
| Monthly consumption/cost | `src/preprocessing.build_monthly_series` | canonical `clean` frame: `month_start, consumption_kwh, cost_gbp, days_in_month, year, month_num, is_partial_year` |
| Same-month-last-year % | `src/narrative._yoy_change` | exactly the default mode's arithmetic; single fuel, single month, no typed result |
| Trailing 12-mo KPIs | `src/kpis.compute_kpis` | stays as the Long-term trends view |
| Complete-calendar-year YoY | `src/kpis.full_year_comparison` | complete-period-only convention to copy |
| Year-to-date | `src/kpis.year_to_date_comparison` | partial-year guard precedent |
| Winter vs winter | `src/kpis.winter_over_winter_comparison` | "complete season only" precedent |
| Gas/elec split | `src/fuel.py`, sidebar `fuel_frames` dict | per-fuel clean frames already loaded for all 3 fuels every run |
| Weather adjustment | `src/energy_signature` (`fitted`/`resid` per month, kWh/day) | monthly weather-explained kWh = `fitted[month] * days_in_month`; the YoY weather fraction already exists in `src/narrative._weather_explained_fraction` |
| Anomalies | `src/anomalies.detect_anomalies` (3 methods, cross-referenced) | reuse flags per month; do not refit |
| Forecasts | `src/forecast_evaluation` + `tabs_phase3.generate_forecast_cached` | untouched |
| Recommendations | `src/recommendations` | untouched; new comparison page carries its own action sentence from interpretation rules |
| Confidence | `src/confidence` (`High/Medium/Low` + reason) | reuse the taxonomy |

**Gap:** nothing computes "this January vs. typical/best/worst January"
(same-calendar-month history). Nothing represents a month comparison as
a typed object. Nothing decomposes a month's YoY change into
weather-explained vs. unexplained kWh as a reusable value (only as a
one-off sentence in `src/narrative.py`).

## Repetition inventory

- Trailing-12-month totals appear on: Executive Summary KPI strip, AI
  Analyst executive summary, "Why did my bill change?" Consultant
  answer.
- Full-history monthly chart appears on: Consumption Analysis,
  Comparisons (Monthly), Cost Intelligence, Carbon, Seasonality
  overview.
- Annual totals appear on: Comparisons, Cost Intelligence, Carbon,
  AI Analyst.

None of these tells the user how the latest complete month compares to
the same month last year.

## Plan consequences (what will change)

1. New `src/monthly_comparison.py` — typed `MonthlyComparison` result +
   6 comparison modes, complete-month detection, weather decomposition
   reusing the fitted energy signature. Pure, Streamlit-free, unit-tested.
2. New `src/monthly_narrative.py` (or same module) — deterministic
   narrative + interpretation rules with documented thresholds.
3. Executive Summary leads with "latest complete month vs. same month
   last year" cards; trailing-12-month view moves below under
   Long-term trends.
4. New "How did this month compare?" tab (replacing the whole-period
   "Comparisons" tab position; that content moves down/relabelled).
5. Navigation regrouped around user questions; technical surfaces
   consolidated (details in final report).
6. Consultant gains same-month comparison questions wired to the live
   selected month/fuel/mode state.

## After (implementation results, same day)

| Metric | Before | After |
|---|---|---|
| Test count | 378 | 448 |
| Coverage (src + app + config) | 95% | 94% (new UI code) |
| Cold first render | 5.51 s | 5.06 s |
| Warm rerun | 0.19 s | 0.20 s |
| Sidebar fuel-switch rerun | 3.52 s | 3.47 s |
| Month-selection response | n/a | 0.24 s |
| Comparison-mode response | n/a | 0.20 s |
| Page fuel-switch response | n/a | 0.20 s |
| Top-level tabs | 15 | 9 (12 sub-tabs; nothing removed) |
| AppTest scenarios | 11 | 16 |
| Consultant questions | 9 | 17 |

All warm comparison interactions are ~0.2 s, well under the 500 ms
target; no forecast, weather, or anomaly model is refitted when the
selection changes. Real-data verification
(`scripts/verify_month_comparison.py`) covers the brief's 7 scenarios;
three wording defects were caught and fixed during that review (weather
wrongly credited for a noise-band increase under milder weather;
"broadly explained by weather" at a 43% weather share; a record-low
month called "within the normal range"). Regression: STL, energy
signature, change points, anomaly detection, forecast CV, cost and
carbon totals are untouched -- their pinned tests (e.g. December 2024 =
+253 kWh deviation, 3/3 anomaly methods) all still pass. One deliberate
wording change outside the new modules: `src/report.py`'s overall
assessment now states the change directly instead of "running at
normal, expected levels" (tests updated).

## Explicitly out of scope (per the brief)

- No changes to validated statistical models (STL, energy signature,
  ESD, PELT/CUSUM, Isolation Forest, forecast CV) — they are reused,
  not refitted.
- No new ML methods.
- Long-term/whole-period analysis retained, demoted to supporting
  context.
