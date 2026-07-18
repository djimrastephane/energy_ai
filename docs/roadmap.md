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
  `tabs_core.py`/`tabs_phase2.py` to stay under the ~300-line guideline.

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
  (`charts_phase3.py`, `tabs_phase3.py`) to stay under ~300 lines/file.

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
  Total) shows heating here is overwhelmingly gas-driven -- gas's heating
  slope is 1.472 kWh/day/HDD (89% of the Total fit's 1.661) vs.
  electricity's 0.189 (11%, but still statistically significant at
  p=0.0003, not noise). Full detail in the README's
  [Fuel-level analysis](../README.md#fuel-level-analysis-electricity-vs-gas)
  section.
- Coverage: 96% on `src/`, 193 tests total (up from 180).

## Phase 4 -- benchmarking, optimisation, carbon, Monte Carlo

- Benchmarking against documented public UK/Scotland averages (Ofgem/DESNZ
  figures, cited with source and year, stored as config constants -- not
  live data).
- Cost-optimisation calculators (tariff switch, standby reduction, solar,
  battery, heat pump, EV/off-peak charging) as engineering estimates with
  stated, user-adjustable assumptions.
- Carbon analysis, optionally using the free carbonintensity.org.uk API for
  UK grid intensity.
- Monte Carlo simulation over inflation/temperature/occupancy/solar/battery
  scenarios.

## Phase 5 -- explainability and AI assistant

- Feature importance / SHAP for the regression and ML forecast models.
- A grounded, deterministic Q&A engine over the computed statistics (no
  LLM/paid API) -- answers only from data already computed, always citing
  the underlying numbers.

## Phase 6 -- reporting and docs

- PDF / HTML / Excel report generation (management summary + technical
  report).
- Architecture diagram, user guide, API documentation, final polish.
