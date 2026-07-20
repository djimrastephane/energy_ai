# Roadmap

This platform is being built in phases, each verified (lint + tests + a real
app launch) before the next begins.

## Phase 1 -- done

Project scaffolding, `src/ingestion.py`, `src/preprocessing.py`,
`src/statistics.py`, `src/kpis.py`, a Streamlit app with Executive Summary /
Consumption Analysis / Statistical Analysis / Data Quality tabs, and a
pytest suite (unit + integration + UI smoke test).

## Phase 2 -- done

- `src/decomposition.py`: STL decomposition of the monthly series (trend /
  seasonal / residual, with seasonal-strength and trend-strength metrics).
- `src/weather.py`: historical daily temperature for the user's location
  (Aberdeen, AB21) from the free Open-Meteo API, cached to disk; aggregated
  to monthly heating/cooling degree days; fit as a degree-day "energy
  signature" regression (base load, heating sensitivity, cooling
  sensitivity, R², Durbin-Watson) -- the standard method utility analysts
  use on monthly billing data. On the real data: base load ~1.57 kWh/day,
  heating sensitivity ~1.66 kWh/day per HDD (p<0.001, R²=0.73), no
  detectable cooling sensitivity (expected in this climate).
- `src/changepoints.py`: PELT + CUSUM change-point detection, run on the
  STL-deseasonalized series so the regular winter/summer swing isn't
  mistaken for a behavioural shift.
- New Streamlit tabs: Seasonality & Trend, Weather Adjustment (opt-in via a
  sidebar toggle, off by default so the app stays network-free until asked),
  Change Points. `app/streamlit_app.py` was split into
  `tabs_core.py`/`tabs_drivers.py` (named `tabs_phase2.py` until the 2026-07
  packaging cleanup) to stay under the ~300-line guideline.

## Phase 3 -- done

- 8-model forecasting suite: `src/forecast_models.py` (naive, seasonal-naive,
  linear trend, Holt-Winters, SARIMA) + `src/forecast_ml.py` (Prophet,
  XGBoost, LightGBM). Compared via `src/forecast_evaluation.py`'s
  1-step-ahead expanding-window walk-forward cross-validation (MAE/RMSE/
  safe-MAPE), auto-selecting the lowest-MAE model. `src/forecast_uncertainty.py`
  builds P10/P50/P90 bands by bootstrapping each model's own CV residuals,
  scaled by sqrt(horizon) -- uniform across every model rather than trusting
  each library's own (differently-valid-at-n≈35) interval machinery.
  On the real data: **Seasonal Naive wins** (MAE 84.5, narrowly ahead of
  LightGBM 85.6 and Holt-Winters 86.3) -- a genuine, honest finding that
  model complexity doesn't help with 3 years of history, exactly per the
  "prefer simple models" principle.
- Anomaly detection: `src/anomalies.py` (rolling z-score, STL-residual
  generalized ESD via `src/esd.py`, Isolation Forest as a secondary
  multivariate check), cross-referenced so a month flagged by 2+ methods
  reads as higher-confidence. On the real data, correctly re-identifies
  December 2024 (the known 738 kWh outlier) as the #1 anomaly by all three
  methods.
- Three real bugs caught and fixed via manual verification against real
  (not just synthetic) data, each with a regression test: Prophet's
  optimizer could take 60x longer to converge on unusually clean/regular
  data (fixed with a capped `iter`); the residual bootstrap could forecast
  negative kWh (fixed with a physical floor at 0); a naive rolling z-score
  included the point being tested in its own baseline window, which could
  mask the exact spike it was meant to catch (fixed with a shifted
  baseline). Also documented, not silently patched over: the median/MAD
  "Hybrid" ESD variant has a real, measured, published-precedent liberal
  bias at small n (`src/esd.py`'s calibration-caveat docstring) -- mitigated
  by cross-referencing multiple methods rather than trusting ESD alone.
- New Streamlit tabs: Forecasting (model comparison + fan chart), Anomaly
  Detection. Executive Summary's "Forecast annual bill" placeholder is now
  a real KPI. `app/charts.py` and `app/streamlit_app.py` were further split
  (now `charts_forecast.py`, `tabs_forecast.py`/`tabs_anomalies.py`) to stay under ~300 lines/file.

## Decision-support layer -- done

Requested directly by the user after Phase 3: "technically strong" but
still a dashboard, not decision support. No new statistics were added --
this is a synthesis/presentation layer over everything Phases 1-3 already
compute.

- `src/confidence.py`: shared High/Medium/Low taxonomy (data quality,
  weather model, forecast, anomaly detection), each rating carrying a
  one-line reason.
- `src/findings.py`: translates numbers into plain-English findings
  (`Finding`: title, narrative, evidence, confidence) -- returns `None`
  rather than fabricating a finding when evidence is insufficient.
- `src/investigation.py`: "possible causes" checklist for a flagged
  anomaly/change point (colder weather, tariff change, new appliance,
  electric heating, holiday occupancy, guests, home working, unknown) --
  only ticks what the data actually supports; most items honestly stay
  unchecked given what billing data alone can see.
- `src/recommendations.py`: evidence-gated recommendation rules. No
  solar/EV/battery rule exists at all -- enforced by the absence of a code
  path, not a runtime check, since this dataset has no roof/vehicle/
  appliance data. `NO_RECOMMENDATIONS_MESSAGE` shown verbatim when nothing
  fires.
- `src/narrative.py`: deterministic per-month narrative (YoY change, %
  explained by weather using the already-fitted regression, anomaly
  status, forecast confidence, recommendation pointer).
- `src/energy_signature.py` gained `annual_weather_adjusted_comparison`
  (actual vs. weather-predicted vs. difference per year -- "was this
  weather or behaviour?") using the existing fitted/residual series, no
  second model.
- `src/report.py`: `AnalystReport`, the single source of truth consumed by
  both the **Executive Summary** (condensed) and **AI Analyst** (full)
  tabs, so the two can't drift out of sync.
- A real calibration bug caught by manual verification against real data:
  the heating-review recommendation's threshold compared a 3-month
  residual sum against a 1-month standard deviation, which fires on pure
  noise ~15% of the time. Fixed with proper variance propagation
  (`sqrt(3) * std`) and a standard one-sided 95% threshold, re-verified
  empirically at ~5-6%.
- Coverage: 96% on `src/`, 180 tests total (up from 97 after Phase 3).

## Fuel-level analysis (electricity vs. gas) -- done

Requested directly by the user after noticing OVO also exports
"Electricity Use"/"Gas Use" breakdowns alongside "Total Use". Rather than
building fuel-aware duplicates of every existing analysis, a sidebar
**Fuel** selector swaps which cleaned DataFrame (Total/Electricity/Gas)
flows through the *entire* existing pipeline -- every tab, finding,
recommendation, and the AI Analyst report becomes fuel-aware for free,
since none of it was ever Total-specific by name.

- `src/ingestion.py`: `discover_csv_files` gained an optional `pattern`
  parameter (`FUEL_FILE_PATTERNS`) and `filter_sources_by_fuel` buckets a
  mixed batch of uploaded/local files by fuel; the default behavior
  (Total Use only) is unchanged for backward compatibility.
- `src/fuel.py`: `cross_check_fuel_totals` (Electricity + Gas == Total,
  zero mismatches on the real 35-month data), `combine_fuel_frames`, and
  `finding_fuel_mix` (gas vs. electricity split and seasonality, `None`
  below 6 overlapping months) -- reuses `src.findings.Finding`, no new
  statistics.
- New **Fuel Breakdown** tab (`app/tabs_fuel.py`, `app/charts_fuel.py`).
- A genuine finding from manual verification: running the *unmodified*
  weather-adjusted energy-signature regression separately on
  Electricity-only vs. Gas-only data (rather than only ever the combined
  Total) shows heating here is overwhelmingly gas-driven -- electricity's
  weather sensitivity is small but statistically real (p=0.0003, not
  noise), plausibly an immersion heater or extra device/lighting use on
  cold days:

  | Fuel | Heating slope (kWh/day per HDD) | R² | p-value |
  |---|---|---|---|
  | Total | 1.661 | 0.73 | <0.0001 |
  | Gas | 1.472 (89% of Total's) | 0.69 | <0.0001 |
  | Electricity | 0.189 (11% of Total's) | 0.34 | 0.0003 |

  On the combined billing data, gas is ~65% of consumption but only ~31%
  of cost (the cheaper fuel per kWh), and swings ~8x between winter and
  summer versus ~1.6x for electricity.
- Coverage: 96% on `src/`, 193 tests total (up from 180).

## Phase 4 -- household energy intelligence -- done

Requested directly by the user: turn the app from "electricity analytics"
into genuine household energy intelligence -- compare fuels directly,
explain *why* energy changed and *which fuel* caused it, add cost/carbon
estimation, and benchmark against published averages. Most of this reuses
Phases 1-3 and the fuel-level analysis exactly as-is (no `EnergyStream`
wrapper class was built -- the existing `fuel: str` / sidebar-selector
pattern already made every analysis generic over Electricity/Gas/Total);
only genuinely new comparisons/cost/carbon/benchmarking logic was added.

- `src/comparisons.py` + `src/cross_fuel_anomalies.py`: annual/monthly/
  seasonality comparison tables across fuels; `compare_weather_sensitivity`
  (which fuel's usage responds more to weather -- confirms the earlier
  fuel-level-analysis finding as a reusable `Finding`: **gas explains 89%**
  of the two fuels' combined heating-driven response, electricity 11%);
  `compare_weather_adjusted_annual` (thin wrapper over the existing
  `annual_weather_adjusted_comparison`, run per fuel); `cross_fuel_anomaly_insights`
  cross-references each fuel's independently-detected anomalies to
  attribute a flagged month to one fuel (electricity-only -> appliance/
  occupancy, gas-only -> heating event, both -> weather-driven *unless* the
  weather-adjusted residual says otherwise, opposite-direction conflicts ->
  attributed to the stronger signal with the conflict stated explicitly,
  never silently resolved).
- `src/cost_engine.py`: per-fuel and combined cost breakdown (annual total,
  monthly average, effective £/kWh, YoY trend) and `forecast_bill_by_fuel`
  (converts each fuel's already-computed forecast to £, opt-in via a button
  since it triples the walk-forward-CV cost). **Standing charges are
  deliberately omitted** -- the OVO exports have no standing-charge/tariff-
  rate column at all, so fabricating a split would violate this platform's
  core "never invent a number" rule.
- `src/benchmarking.py`: Below/Average/Above-average bands (±15% tolerance,
  never a single "Energy Score") against Ofgem TDCV (2,500 kWh electricity
  / 9,500 kWh gas/year, medium usage, 2026) and Scotland-specific
  electricity consumption (3,429 kWh/year, DESNZ/ONS sub-national
  statistics). No Scotland-specific *gas* benchmark was found during
  research -- the UI says so explicitly rather than silently reusing the
  UK-wide figure under a "Scotland" label.
- `src/carbon.py`: monthly/annual/weather-adjusted/forecast CO2e emissions,
  static cited factors (`config.CarbonConfig`: electricity 0.207, gas 0.183
  kgCO2e/kWh) -- only supported for Electricity/Gas individually, since a
  blended factor for "Total" can't be honestly computed without knowing the
  split. On the real data (2024, the most recent complete year): 303 kg
  CO2e electricity + 477 kg CO2e gas = **0.78 tonnes CO2e combined**,
  cross-checked by independent manual recomputation.
- `src/recommendations.py` gained `recommend_fuel_focus`: fires only when
  one fuel dominates *both* consumption/cost share and weather-sensitivity
  share (>60% each). Fires on the real data: **"Focus on gas"**, High
  confidence (gas is 65% of consumption and 89% of heating sensitivity).
- `src/report.py`'s `AnalystReport` gained `fuel_mix_finding`,
  `weather_sensitivity_finding`, `largest_cost_driver`,
  `weather_vs_behavioural_summary`, `per_fuel_findings` -- new "Household
  Energy Profile"/"Electricity Findings"/"Gas Findings" sections on the AI
  Analyst tab, still fully deterministic, still traceable to a `Finding`/
  `Recommendation` object.
- Three new tabs: **Comparisons**, **Cost Intelligence** (includes
  benchmarking), **Carbon**.
- A real logic bug caught by manual verification against the real December
  2024 anomaly: electricity *dropped* (149 kWh, 1 method) while gas
  *spiked* (589 kWh, all 3 methods, weather-adjusted z=+1.96) the same
  month -- the initial cross-fuel classification called this "both moved
  together" without checking direction agreement, which is simply false
  when one fuel drops and the other spikes. Fixed to detect direction
  conflicts and attribute to the stronger signal (gas here) rather than
  forcing a shared-driver conclusion, with the conflict stated in the
  evidence rather than hidden.
- Coverage: 97% on `src/`, 258 tests total (up from 200).

Deferred, not built in this phase (kept honest rather than rushed):
tariff-switch/solar/battery/heat-pump/EV cost-optimisation calculators (no
roof/vehicle/appliance data exists to evaluate them against -- same
"absence of a code path" principle as the no-solar-recommendation rule);
live `carbonintensity.org.uk` integration (carbon estimates stay
offline-first with static factors instead); Monte Carlo simulation.

## AI Energy Consultant -- done

Requested directly by the user: "the analytics engine is now mature" --
invest in an *interface* over it, not more analysis. Matches this
roadmap's original Phase 5 sketch almost exactly (a grounded, deterministic
Q&A engine, no LLM, always citing the underlying numbers). The consultant
explains analysis that's already been run -- it never runs new analysis;
every answer is built from objects `main()` already computes (`AnalystReport`,
the per-fuel result dicts, the forecast), so visiting the tab triggers zero
new computation.

- `src/consultant.py` + `src/consultant_router.py`: 8 canonical questions
  ("Why did my bill change?", "What changed compared with last winter?",
  "Should I focus on reducing gas or electricity?", "What's my forecast for
  next year?", "How does my usage compare to average?", "What's my carbon
  footprint?", "Was last month's usage normal, or an anomaly?", "Where can
  I realistically save money?"), each a deterministic handler pulling from
  already-computed `Finding`/`Recommendation`/comparison/benchmark/carbon
  objects. Without an LLM, "understands any question" isn't honest, so a
  free-text box routes via keyword matching to the same 8 handlers and
  falls back to a visible clickable list (not a guess) when nothing matches.
- `src/kpis.py` gained `winter_over_winter_comparison` (the one genuinely
  new piece of logic -- pure aggregation, sums kWh/cost per winter season,
  not a new statistic); `winter_season_label` was promoted from
  `src/recommendations.py` to `src/utils.py` so both modules share it.
- New **AI Consultant** tab, positioned second (right after Executive
  Summary) as the primary entry point for quick questions before the full
  AI Analyst report.
- A real bug caught during build: "Where can I realistically save money?"
  initially picked whichever recommendation happened to be first in the
  list, which could be "Collect more historical data" (real advice, but
  not itself a savings action) even when a more directly relevant
  recommendation like "Focus on gas" had also fired. Fixed to prefer any
  actionable recommendation over that one, falling back to it only when
  it's the sole recommendation available.
- Coverage: 97% on `src/`, 291 tests total (up from 258).

## Household Energy Review report -- done

The sidebar's long-disabled "Download report" placeholder is now a real
feature: a **self-contained HTML "Household Energy Review"** (executive
summary, key findings, fuel mix, weather analysis, consumption history,
forecast, recommendations, carbon, benchmark, methodology, limitations &
data quality), generated on demand from the sidebar.

- `app/report_html.py` + `app/report_template.html`: pure rendering over
  the already-computed `AnalystReport`/fuel frames/forecast -- no new
  statistics, no Streamlit imports in the builder (unit-testable directly).
  Jinja2 autoescaping is on (escaping regression-tested); only the
  app-generated Plotly fragments are marked safe. Charts stay interactive
  in a browser; print CSS means Print -> "Save as PDF" yields the polished
  PDF. plotly.js is inlined once (~4.9 MB file, builds in 0.17 s on the
  real data).
- Two-step Generate -> Download flow with a context fingerprint (fuel,
  weather, data extent): a stored report is discarded rather than served
  stale after fuel/weather/data changes -- the audit F2/F7 session-state
  lesson applied preemptively, regression-tested via AppTest.
- Every honesty rule carries into the document: "excl. standing charges"
  on all £ figures, the in-progress-month warning, verbatim
  no-recommendations fallback, confidence + reason on every finding, and
  a footer noting results are household-specific.
- Format decision (recorded): self-contained HTML chosen over in-app PDF
  (reportlab + kaleido would add two heavy dependencies for static charts)
  and over an Excel appendix (declined for now) -- zero new dependencies,
  since jinja2 already ships with Streamlit.
- Coverage: 97% on `src/`, 306 tests total (up from 295).

## Weather Context Engine (snow, wind, severe weather) -- done

Contextual explanation of unusual months from extended Open-Meteo daily
fields (snowfall, snow depth, precipitation, max wind/gusts) -- kept
strictly out of the energy-signature regression (HDD/CDD only, unchanged).
See `docs/weather_context.md` for the full data dictionary, thresholds
(config-owned, conservative UK conventions), cache v2 migration, robust
relative-severity bands, and the A-D interpretation patterns.

- `src/weather.py` extended: 6-field fetch with hard response validation
  (invalid JSON, error payloads, missing fields, unequal lengths -> typed
  `WeatherFetchError` with cause preserved), v2 cache identity including
  schema version + timezone (v1 files ignored, untouched). One latent
  defect found and fixed while extending: a null-temperature day used to
  count toward monthly weather coverage while contributing zero degree
  days -- coverage now counts non-null temperature days only.
- `src/weather_context.py` + `src/weather_interpretation.py`: monthly
  aggregation, strict-percentile severity bands (a uniform history reads
  as Typical -- caught by tests via 31- vs 30-day precipitation totals),
  deterministic classification with facts + confidence + limitation, and
  hedged pattern A-D interpretations that never assert behavioural causes.
- Surfaced in: Unusual Months (per-anomaly context, top month inline,
  advanced details expander), Weather Impact (history summary; cooling
  hidden from the main view -- negligible for this household), and a 9th
  Consultant question routed before the "unusual" keyword to avoid a
  collision.
- Real-data: December 2024's 3-method spike = Storm Darragh-period gusts
  (5 severe-gust days, max 91 km/h) *plus* +219 kWh residual -> "may have
  contributed, does not explain the full increase". December 2023 (Extreme
  snowfall band, residual -22 kWh) correctly stays a non-event.
- Performance: cold fetch 4.4 s once; disk cache hit 2 ms with zero
  network (tested); weather-on warm rerun 0.30 s; context aggregation +
  classification ~30 ms total. Cache grew 16 KB -> 39 KB.
- Coverage: 97% on `src/`, 376 tests total (up from 324; the app smoke
  suite now also exercises the weather-ON path via a mocked fetch).
- Deferred: delta-fetching only missing date ranges (full-range refetch is
  at most once per new month at trivial size); weather context in the HTML
  report; adding severe-weather variables to the regression (requires a
  formal model-selection study first, per the design principle).

## Month-comparison UX refactor (2026-07)

The primary journey became "latest complete month vs the same calendar
month last year" (see `docs/month_comparison.md` and
`docs/audits/month_comparison_ux_audit.md` for the before-state audit).

- New `src/monthly_comparison.py`: typed frozen `MonthlyComparison`, five
  single-month modes (same-month-last-year default, previous-month with a
  seasonality warning, typical/best/worst same-calendar-month) plus a
  long-term mode; complete-month detection; exact cost usage/rate split;
  weather-explained vs unexplained decomposition reusing the already-
  fitted energy signature (nothing refitted on selection changes).
- New `src/monthly_narrative.py`: deterministic household-language
  narratives (headline <= 25 words, primary paragraph <= 80) built from
  documented rules -- practical change bands (5%/20%, in
  `config.MonthComparisonThresholds`), "explained by colder weather" only
  when weather covers >= half the change, record months never called
  "within the normal range". Banned generic-analyst phrasings are tested
  against. `src/report.py`'s overall assessment reworded the same way
  ("you used 14% more energy than the year before", not "running at
  expected levels").
- Navigation reorganized around user questions: 15 flat tabs became 9
  question-oriented top-level tabs (Home / How did this month compare? /
  What drives my usage? / Costs and carbon / Did anything unusual happen?
  / What should I expect next? / Ask the Energy Consultant / Long-term
  trends / Data and methods) with the technical and whole-period surfaces
  demoted to sub-tabs -- all retained, none removed. The trailing-12-month
  KPI strip moved from the homepage to Long-term trends.
- Home now opens with "This month compared with last year": headline
  sentence, three cards (Total/Gas/Electricity with contributions), a
  weather/unexplained breakdown when the model covers both months, one
  "what this means" line, and an evidence-gated action.
- New "How did this month compare?" page: month/fuel/mode controls
  (shared session state resolved once in `main()` so Home, the page, and
  the Consultant always describe the same comparison -- the audit-F2
  lesson), one grouped two-month bar chart above the fold, same-calendar-
  month history chart, structured explanation, costs/carbon for the
  selected month, technical-details expander with formulas and
  thresholds. Partial months are excluded from selection, flagged
  ("July 2026 is incomplete. The primary comparison uses June 2026."),
  and shown only as labelled month-to-date -- never extrapolated.
- Consultant: 8 new month questions (17 total), all answering from the
  currently selected month/fuel/mode via `src/consultant_month.py`.
- Real-data verification (`scripts/verify_month_comparison.py`) across
  the brief's 7 scenarios caught three wording defects before ship:
  an increase inside residual noise being credited to "colder weather"
  when the weather was actually milder; "broadly explained by weather"
  when weather covered only 43% of the change; and a record-low January
  described as "within the normal range".
- Performance: warm month/mode/fuel-switch interactions ~0.2 s (target
  < 500 ms); cold render 5.1 s and sidebar fuel switch 3.5 s, both
  marginally better than baseline. 448 tests (from 378), coverage 94%.

## Billing facts: standing charges, VAT, 6th-to-5th cycle (2026-07)

User-supplied tariff facts (`config.BillingConfig`) finally allow honest
full-bill estimates, ending the "consumption cost only" limitation:

- `src/billing.py`: billing periods (the April bill covers 6 Apr - 5 May,
  always exactly `days_in_month` days), per-fuel standing charges
  (62.77p/34.97p per day; the combined view pays both), 5% VAT on
  consumption + standing, and a per-month `bill_breakdown` /
  `bill_breakdown_frame`. The exported cost is treated as ex-VAT
  consumption charge -- a documented assumption with an
  `export_cost_includes_vat` flag to flip it.
- Completeness became billing-aware: a month's bill is only complete once
  the 5th of the *next* month has passed. `is_month_complete` (month
  journey) and `flag_in_progress_month` (data quality) now share
  `src.billing.is_billing_month_complete`; on the 1st-5th of a month the
  previous month correctly stays "period-to-date" where the old
  calendar-month rule called it closed.
- Surfaced: month-comparison page cost section (Consumption + Standing +
  VAT = Estimated total bill, with the billing-period dates), a per-fuel
  bill-breakdown table on Cost Intelligence, total-bill estimates on the
  forecast tab and Home forecast caption, and the Consultant's "Why was
  this month expensive?" now itemizes the full bill.
- Documented misalignment (not silently ignored): the weather merge is
  calendar-month based, ~5 days offset from the billing window;
  re-aligning would change the validated weather models and is deferred.
- Real-data check: June 2026 electricity £41.37 + £18.83 + £3.01 =
  £63.21; elec + gas component bills sum exactly to the combined bill
  (£81.38). 461 tests (from 448).
- Follow-up: the tariff rates became editable in-app (sidebar "Tariff"
  section -- standing charges in p/day, VAT in %) with the supplied OVO
  figures as defaults, threaded as a runtime ``BillingConfig`` through
  every bill-estimating surface (month page, Cost Intelligence, forecast
  captions, Consultant) so another provider's rates work without code
  changes. 462 tests.

## Phase 5 -- explainability

- Feature importance / SHAP for the regression and ML forecast models.

## Phase 6 -- remaining reporting and docs

- ~~HTML report generation~~ done (see "Household Energy Review report").
  In-app PDF and Excel export consciously deferred.
- Architecture diagram, user guide, API documentation, final polish.
