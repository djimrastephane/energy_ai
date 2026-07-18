# Senior Engineering, Security, Performance, and End-User UX Audit

Audit date: 2026-07-18. Baseline commit: `68ad97b`. Auditors: senior Python
engineer / application security engineer / performance engineer /
non-technical homeowner perspectives, applied to the full codebase and the
live application running on the real household data (35 months, Sep 2023 -
Jul 2026, three fuel exports).

Method: full repository read; `ruff`, `bandit`, `pip-audit`, `pytest
--cov`; a timed profiling script covering every pipeline stage; `AppTest`
cold/warm/fuel-switch latency measurement; live headless launch. No code
was modified before this document was produced.

---

## 1. Executive verdict

**Is the application ready for public portfolio use?** As *code*, yes — the
engineering discipline (evidence-gated findings, empirical calibration of
thresholds, 291 meaningful tests, clean layering) is genuinely
portfolio-grade. As a *repository*, not yet: `data/raw/` contains the
owner's real household billing data. Before the repo is made public, that
data must be replaced with synthetic sample data. Nothing else blocks it.

**Is it safe for local use?** Yes. No secrets, no eval/exec/pickle, no
subprocess of user input, Bandit and pip-audit both clean, the single
outbound API call (Open-Meteo) is opt-in, timeout-bounded, TLS-verified,
and built from config constants, not user input. Two hygiene gaps: Streamlit
telemetry is on by default (an outbound call the user never consented to)
and the upload limit is the 200 MB default when real inputs are <1 KB.

**Is it safe for public deployment?** Mostly, with the same two fixes plus
one architectural caveat: `st.cache_data` on the default loader is shared
process-wide, which is fine single-user but means a multi-user deployment
shares one household's data — this app is single-household by design and
should be deployed per-user or not at all.

**Is performance acceptable?** Yes, measured: warm interactions complete in
0.40 s (target: <500 ms), cold first load is 5.2 s dominated by the
spinnered 8-model forecast CV, fuel switch is 3.7 s cold / 0.33 s warm.
~170 ms of statistical work per rerun is unnecessarily uncached — worth
fixing, not urgent.

**Can a normal homeowner understand it?** The Executive Summary and AI
Consultant, yes — they lead with conclusions, confidence, and evidence in
plain English. The full 15-tab surface, no — four tabs are named after
methods rather than questions, and a homeowner must know what "Change
Points" means before clicking it. The product is roughly 70% decision
tool, 30% report catalogue; the catalogue part is honest and well-separated
but over-weighted in primary navigation.

**Three largest risks:** (1) real household data in the repo if it goes
public; (2) the in-progress current month (July 2026 is month-to-date:
107.87 kWh on July 18) silently treated as complete throughout the billing
pipeline, biasing KPIs and the "was last month normal?" answer; (3) "annual
bill" labels understate real bills because OVO's Cost column excludes
standing charges (evidenced by unit rates matching published per-kWh-only
rates) — the omission is documented in one tab but the word "bill" is used
unqualified elsewhere.

**Three highest-value improvements:** (1) honesty fixes — partial-month
flagging, "excluding standing charges" labels, and the hardcoded
"Electricity consumption…" wording in `narrative.py` shown for gas data;
(2) two real Streamlit state bugs — the Consultant serves a stale answer
after fuel/weather changes, and the multi-fuel forecast result can go stale
after data changes; (3) renaming method-named tabs to user questions.

## 2. Scorecard

| Dimension | Score | Justification |
|---|---|---|
| Correctness | 8/10 | Statistics verified against real data repeatedly during development; remaining defects are presentation-layer (partial month, fuel wording), not model math. |
| Maintainability | 8/10 | Consistent ~300-line modules, one concern each, src/ has zero Streamlit imports; the `fuel_*` dict plumbing through `main()` is the weakest seam. |
| Naming clarity | 6/10 | Domain results are well-named (`heating_slope`, `kg_co2e`); the ubiquitous `clean` and `merged` DataFrames hide fuel context and unit contracts behind convention. |
| Data-contract clarity | 6/10 | Contracts are real and consistently honored but live in docstrings and convention, not one referenceable place (§4 of this audit now provides it). |
| Security (local) | 9/10 | Bandit/pip-audit clean, no dangerous constructs, bounded network; telemetry-on-by-default is the only ding. |
| Security (public deployment) | 6/10 | Real personal data in repo; process-wide data cache is single-user by design; upload cap at default 200 MB. |
| Performance | 8/10 | Measured 0.40 s warm rerun already meets the target; ~170 ms/rerun avoidable; cold paths are spinnered and cached. |
| UI clarity | 7/10 | Individual pages explain themselves well (captions everywhere); confidence always paired with a reason; some £ values show false precision. |
| Navigation | 5/10 | 15 flat tabs; four named after algorithms; overlapping content between Fuel Breakdown and Comparisons; no grouping. |
| Accessibility | 6/10 | Confidence icons always paired with text, CVD-validated palette; inherits Streamlit's limits; not audited on real assistive tech. |
| Decision usefulness | 8/10 | Executive Summary answers "so what" in one screen; recommendations are evidence-gated with £ impact where supportable. |
| Trust and explainability | 9/10 | The strongest dimension: every conclusion traces to evidence, `None` instead of fabrication, limitations stated unprompted. |
| Test quality | 8/10 | Tests assert behavior (thresholds, narratives, honest-fallback paths) not implementation; gaps: state-transition and routing-ambiguity coverage. |
| Documentation | 8/10 | README/roadmap document real findings and real bugs caught; no architecture diagram; data contracts previously undocumented. |

## 3. Findings register

Severity: Critical / High / Medium / Low / Info. Confidence: High = confirmed in code or by measurement; Medium = strong inference.

| ID | Category | Sev | Conf | Finding, evidence, impact, fix |
|---|---|---|---|---|
| F1 | Correctness / honesty | **High** | High | **In-progress current month treated as complete.** `data/raw/OVO Total Use 2026.csv` contains `July 2026,21.96,107.87` while the audit date is July 18, 2026 (full July 2025 was 150.59 kWh). `src/preprocessing.py` has no current-month guard, so `avg_daily_kwh` divides by 31, trailing-12-mo KPIs, benchmark annualization, and the Consultant's "Was last month normal?" (`src/consultant.py:229`) all treat a month-to-date figure as a full month. The weather module already guards its own side (`src/weather.py:170-177`) — the billing side must match. Fix: warning in `run_pipeline` when the latest month is the current calendar month + caveat in the Consultant answer. Effort: small. Test: required. |
| F2 | State bug | **Medium** | High | **Stale Consultant answer across context changes.** `app/tabs_consultant.py:52-57` stores the *computed* `ConsultantAnswer` in `st.session_state`; after switching fuel or toggling weather, the stored answer (built from the old context) still renders. Fix: store the selected question, recompute the answer each rerun (handlers are ~0.5 ms, measured). Effort: small. Test: required. |
| F3 | Privacy / hardening | Medium | High | **No `.streamlit/config.toml`:** telemetry `browser.gatherUsageStats=True` (verified via config API) sends usage stats — an outbound call without user action, contradicting the offline-first posture; `server.maxUploadSize=200` MB while real inputs are <1 KB (memory-DoS headroom for a public deployment). Fix: config file with `gatherUsageStats=false`, `maxUploadSize=10`. Effort: trivial. |
| F4 | Honesty / language | **Medium** | High | **"Annual bill" excludes standing charges but is labeled as the bill.** Effective rates from the data (electricity £0.2646/kWh, gas £0.0642/kWh) match published *unit-only* rates, so OVO's Cost column almost certainly excludes standing charges (~£150-300/yr). The Cost Intelligence tab says so; the Executive Summary ("Forecast: expected annual bill"), Forecasting tab, and Consultant forecast answer do not. Fix: qualify labels ("energy cost, excl. standing charges"). Effort: small. Test: update strings. |
| F5 | Correctness / language | Medium | High | **`src/narrative.py:101` hardcodes "Electricity consumption {direction}…"** but renders for whatever fuel is selected (gas/total) — the same fuel-wording bug class fixed in `findings.py`/`energy_signature.py` earlier, missed here. Fix: neutral "Consumption" or thread the fuel noun. Effort: trivial. Test: update/add. |
| F6 | UX context | Medium | High | **Consultant answers never state which fuel they describe.** `ConsultantContext` is built from the selected fuel; with "Gas only" selected, "What's my forecast?" returns a gas-only £ figure presented as "your bill". Fix: add `fuel_label` to the context, display "Answering for: X" and include it in the forecast answer. Effort: small. Test: required. |
| F7 | State consistency | Medium | High | **`st.session_state["multi_fuel_forecasts"]` can go stale and creates a hidden cross-tab ordering dependency.** `app/tabs_cost.py:108` stores results; `app/streamlit_app.py:202-213` must render Cost before Comparisons for same-run visibility (comment-documented but fragile), and the stored value survives data changes. Fix: store only the request flag; Comparisons calls the `st.cache_data`-backed generator itself (cache hit ≈ free, always consistent). Effort: small. Test: existing cross-tab test must keep passing. |
| F8 | Performance | Low | High | **~171 ms of statistical work recomputed on every rerun** (measured): `detect_anomalies` ×4 (3 fuels + selected-fuel duplicate, 158 ms), STL ×4, changepoints, report build. The selected fuel's STL/anomalies are computed twice per rerun (`app/streamlit_app.py:106-111` and again inside `compute_all_fuel_analysis`). Warm rerun is 0.40 s, so this is ~40% of interaction latency. Fix: `st.cache_data` on the all-fuels computation; reuse its output for the selected fuel. Effort: small. Test: behavior-preserving; suite must stay green. |
| F9 | Security hygiene | Low | High | Uploaded **filenames render as markdown** in the Data Quality tab (`app/tabs_core.py:71`): a file named `**bold**.csv` alters page formatting (Streamlit escapes HTML, so no script execution — formatting injection only). Fix: wrap names in backticks. Effort: trivial. |
| F10 | Language / precision | Low | High | **False precision in forecast £** ("£169.47–£1,017.56" from a Low-confidence bootstrap band) in Consultant and Forecasting outputs. Fix: whole pounds for forecast figures. Effort: trivial. |
| F11 | Dead code | Info | High | `src/kpis.py:134` `year_to_date_comparison` is implemented and tested but never called by the app (verified by grep). Keep or wire in; flagged for a future decision, not removed in this pass. |
| F12 | Reproducibility | Info | High | `requirements.txt` uses `>=` everywhere and CI installs latest — a supply-chain/reproducibility tradeoff. Recommend a constraints/lock file. Deferred (environment-changing). |
| F13 | Privacy | **High (public only)** | High | **Real household billing data committed** (`data/raw/*.csv`, 12 files). Correct for a private repo; must be replaced with synthetic data before any public release. Not an issue for local use. Owner decision — documented, not changed. |
| F14 | Navigation | Medium | High | **15 flat tabs, four named after methods** ("Change Points", "Anomaly Detection", "Weather Adjustment", "Statistical Analysis"); "Fuel Breakdown" and "Comparisons" overlap. Minimal fix implemented: question-oriented renames (§5). Full grouping (§8, deferred) is a larger redesign. |
| F15 | Test robustness | Info | High | `tests/test_pipeline_integration.py` and several `AppTest` assertions couple to the exact current data snapshot (n=35, specific months). Stable until the owner adds new exports, then several tests need updating together. Acceptable for a personal project; noted. |
| F16 | Data contract | Info | Medium | Benchmark annualization (`sum/len×12` over full history) uses the 3-year average, not current-year usage; carbon factors are the 2024 edition applied to 2024–2026 (grid factor declining → recent electricity emissions modestly overstated). Both already caveated in-app/config; kept, documented here. |

**Classification summary — Must fix:** F1, F2, F3. **Should fix (implementing):** F4, F5, F6, F7, F8, F9, F10, F14-minimal. **Could improve (deferred):** F11, F12, F16, full F14. **Keep as is / owner decision:** F13, F15.

## 4. Variable naming and data-contract register

Only names whose meaning depends on hidden context. Mass renames are
deliberately not proposed (the prompt's own constraint against
unreviewable repo-wide renames); the contract table below is the fix for
most of these.

| Current name | Where | Actually contains | Problem | Proposed | Migration risk |
|---|---|---|---|---|---|
| `clean` | `main()`, `build_analyst_report`, consultant, narrative — ubiquitous | The *selected fuel's* cleaned monthly billing DataFrame | Hides that every downstream number is fuel-scoped | `selected_fuel_monthly_df` (or keep + contract doc) | High — touches ~10 modules and their tests; **deferred**, contract documented below instead |
| `merged` | phase2/weather paths | Consumption ⋈ degree-days monthly frame | "Merged with what" is convention | `consumption_weather_df` | Medium — deferred, documented |
| `unit_rate` | briefing, cost, consultant | Effective blended £/kWh = cost÷kWh **excluding standing charges** | The exclusion is the trap | keep name; fix the *labels* (F4) | none |
| `total_kwh` | `src/fuel.py:71,87` | electricity+gas *reconciled sum*, not the reported Total file | Two "totals" exist in this domain | `reconciled_sum_kwh` | Low — local; worth doing opportunistically |
| `result` | `app/tabs_phase2.py:66-120` | `EnergySignatureResult` | Generic in a 50-line scope | `signature_result` | Low — cosmetic, skipped |
| `consultant_selected_answer` | session state | A *stale computed* answer (F2) | Name is fine; the *content* is wrong — fix stores the question | `consultant_selected_question` | none (part of F2) |

**Data contracts (the single reference the codebase lacked):**

*Cleaned monthly billing frame* (from `run_pipeline`, per fuel): columns
`month_start` (Timestamp, month-start, unique, sorted), `cost_gbp` (float £,
consumption cost only — standing charges not present in source),
`consumption_kwh` (float ≥0 expected, flagged not dropped), `source_file`
(str), plus derived `year`, `month_num`, `month_name`, `days_in_month`,
`avg_daily_kwh`, `avg_daily_cost_gbp`, `unit_rate_gbp_per_kwh`,
`is_partial_year` (bool: <12 months present in that calendar year).
Monthly frequency; missing months *reported*, not filled. The latest month
may be month-to-date (F1). Fuels: `total` is OVO-reported; electricity+gas
reconcile to it exactly (verified: 0 mismatches, `cross_check_fuel_totals`).

*Weather-merged frame* (from `merge_weather_with_consumption`): billing
frame ⋈ `hdd`/`cdd` (degree-days, base 15.5 °C/22 °C), `avg_temp_c`,
`n_days`, `avg_daily_hdd/cdd`; months with incomplete weather coverage are
dropped with a warning (never under-counted).

*Forecast* (`ForecastResult`): monthly kWh arrays `point`, `p10 ≤ p50 ≤ p90`
(order guaranteed by percentile construction; floored at 0), `forecast_dates`
month-starts, `comparison` CV table (MAE in kWh). £ conversions always
happen at display time via the fuel's own effective rate.

*Carbon*: kg CO₂e throughout; tonnes only as a derived display column
(÷1000, tested); electricity/gas factors only — never a blended factor for
"total" (enforced by `ValueError`).

## 5. Tab-by-tab UX review

| Tab | User question | Verdict & action |
|---|---|---|
| Executive Summary | "How am I doing; what should I do?" | Strong: assessment → biggest finding → saving → bill forecast → confidence in one screen. **Keep.** Bill label needs F4 qualifier. |
| AI Consultant | "Just answer my question" | Strong concept, honest fallback. **Keep**; fix stale answer (F2) + fuel context (F6). |
| AI Analyst | "Give me the full report" | Good progressive disclosure via expanders. **Keep.** |
| Consumption Analysis | "How has usage moved?" | Clear charts, honest caption about monthly-data limits. **Keep.** |
| Fuel Breakdown | "Gas vs electricity?" | Good, but overlaps Comparisons. **Keep now; merge candidate** (deferred F14). |
| Comparisons | "Which fuel changed?" | Dense but each block is captioned. **Keep.** |
| Cost Intelligence | "What does it cost; am I typical?" | Good; the only tab that already says "excl. standing charges". **Keep.** |
| Carbon | "What's my footprint?" | Clear, factors cited inline. **Keep.** |
| Statistical Analysis | (methodology) | Homeowner value ≈ 0; analyst value real. **Keep as advanced** — rename candidate ("Statistics (Advanced)") deferred. |
| Seasonality & Trend | "Is there a pattern?" | Rename → **"Seasonal Patterns"** (implemented). |
| Weather Adjustment | "Was it the weather?" | Rename → **"Weather Impact"** (implemented). |
| Change Points | "Did my usage shift?" | Jargon title. Rename → **"Usage Shifts"** (implemented). |
| Forecasting | "What's next?" | Good fan chart + honest CV caveats. **Keep.** F10 precision fix. |
| Anomaly Detection | "Any weird months?" | Jargon title. Rename → **"Unusual Months"** (implemented). |
| Data Quality | "Can I trust the data?" | Excellent honesty surface. **Keep.** F9 filename fix. |

30-second test (homeowner, Executive Summary): main finding and next action
are visible without scrolling — **passes** today; the Consultant makes it
faster still.

## 6. Performance baseline (measured, not impressions)

Hardware: Apple Silicon macOS; Python 3.12.7; warm disk cache.

| Measurement | Value |
|---|---|
| Full test suite (291 tests, with coverage) | 54.4 s |
| Headless server start → HTTP 200 | 0.38 s |
| Cold first page (empty caches; incl. 8-model CV) | 5.16 s |
| **Warm full rerun (any widget interaction)** | **0.40 s** |
| Fuel switch, cold (new fuel's 8-model CV) | 3.66 s |
| Fuel switch back, warm | 0.33 s |
| `generate_forecast` cold (8-model walk-forward CV) | 3.51 s |
| Uncached per-rerun statistical work (F8) | 0.171 s |
| — of which `detect_anomalies` ×4 | 0.158 s |
| Weather (disk-cached) + fit, ×3 fuels | 0.016 s |
| `build_analyst_report` (full household) | 0.007 s |
| Consultant: all 8 handlers + 13 routings | 0.010 s |
| 7 core Plotly figures | 0.082 s |
| Heavy imports (statsmodels/sklearn/prophet/xgb/lgbm) | 1.08 s total |
| Peak RSS (full pipeline, one process) | 340 MB |

Against the provisional targets: warm interaction <500 ms ✅ (0.40 s);
cached weather <2 s ✅ (0.016 s); explicit multi-model comparison shows a
spinner and is cached ✅; no operation >1 s runs without a spinner ✅
(forecast CV, weather fetch, multi-fuel comparison all have
`show_spinner`). One structural note: `st.tabs` renders *all 15 tabs'
content every rerun* — the 0.40 s warm figure already includes building
every chart on every interaction. Acceptable at current data size;
would need lazy navigation (F14 full version) before it ever grows.

## 7. Security threat model

**Local single-user (the actual deployment):**

| Threat | Path | Assessment |
|---|---|---|
| Malicious CSV upload | pandas parse → strict schema check | Mitigated: `IngestionError` on wrong columns/months/non-numerics (`src/ingestion.py:96-135`); month values must parse as "Month YYYY" so content injection into narratives fails. Residual: filename markdown (F9, low). |
| Memory exhaustion via upload | 200 MB default cap | F3: cap to 10 MB. |
| Outbound traffic | Open-Meteo only, opt-in toggle, `timeout=30`, TLS default, constant URL, params from frozen config (`src/weather.py:48-58`) | Sound. No SSRF path (no user-controlled URL parts). Residual: Streamlit telemetry (F3). |
| Secrets / dangerous constructs | — | None: no secrets, no eval/exec/pickle/yaml.load/subprocess-of-input. Bandit: 0 findings. pip-audit: 0 known vulns (2026-07-18). |
| Personal data in logs | Loggers print filenames + kWh to stdout | Local console only; no persistent log files; acceptable. |

**Public multi-user deployment (hypothetical) — additional:**

| Threat | Assessment |
|---|---|
| Real household data in repo (F13) | **Blocker** until replaced with synthetic data. |
| Cross-user data leakage | `_load_default_data` cache is process-wide; the whole design is single-household. Deploy per-user instance or don't. |
| DoS | Forecast CV (3.5 s CPU) is user-triggerable per fuel; cache limits repeats, but a public deployment needs rate limiting at the proxy. |
| Location privacy | Config exposes city-level coordinates (Aberdeen centre), not an address — acceptable coarseness, already documented. |

## 8. Proposed implementation plan (impact ÷ effort ordering)

Implemented in this audit cycle, small commits grouped by concern:

1. **Security & data handling** — F3 (`.streamlit/config.toml`), F9 (filename backticks).
2. **Correctness & honesty** — F1 (partial-month warning + Consultant caveat), F5 (`narrative.py` fuel wording).
3. **State bugs** — F2 (store question, recompute answer), F7 (drop cross-tab session-state dependency).
4. **Performance** — F8 (cache all-fuels analysis, deduplicate selected-fuel work) with before/after measurement.
5. **Language** — F4 ("excl. standing charges" labels), F10 (whole-£ forecasts), F6 (Consultant fuel context).
6. **Navigation** — F14-minimal (4 tab renames + Consultant `related_tab` strings + test updates).
7. **Tests & docs** — repeatable profiling script into `scripts/`, after-document.

Deferred with reasons: full navigation regrouping (major redesign; needs
owner input), `clean`/`merged` renames (high churn, contract table added
instead), F11 dead code (owner decision), F12 lockfile (environment
change), F13 (owner's data, owner's call).
