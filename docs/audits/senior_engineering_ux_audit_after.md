# Audit implementation report (after)

Companion to `senior_engineering_ux_audit.md`. Baseline commit `68ad97b`;
implementation commits `d27cf1f`..`e1cb27e` plus this document. Seven small
commits, each grouped by concern per the audit's implementation plan.

## What changed

| Commit | Concern | Findings |
|---|---|---|
| `d27cf1f` | Security & data handling | F3 (telemetry off, 10 MB upload cap via `.streamlit/config.toml`), F9 (filenames backtick-escaped in Data Quality) |
| `4dd27c6` | Correctness & honesty | F1 (`flag_in_progress_month`: the current calendar month is flagged as possibly month-to-date; Consultant "last month" answer caveats it; data-quality confidence honestly drops to Medium while present), F5 (`narrative.py` fuel-neutral wording) |
| `194475d` | State bugs | F2 (Consultant stores the *question*, recomputes the answer per rerun — stale-context bug gone, regression test added), F7 (multi-fuel forecast owned by `main()` from a request flag; session-state stash and cross-tab render-order dependency removed) |
| `9abeee1` | Performance | F8 (`compute_all_fuel_analysis` cached; selected-fuel STL/anomalies no longer computed twice; per-fuel error messages surfaced instead of swallowed) |
| `12e0071` | Language & honesty | F4 ("expected annual energy cost (excl. standing charges)" wherever forecasts were called "bills"), F10 (whole-pound forecast figures), F6 (Consultant shows "Answering for: {fuel}") |
| `5a36733` | Navigation | F14-minimal (four tab renames: Seasonal Patterns, Weather Impact, Usage Shifts, Unusual Months; Consultant pointers, README, tests updated together) |
| `e1cb27e` | Tests & docs | `scripts/profile_pipeline.py` committed as the repeatable profiling harness |

## Before / after measurements

| Metric | Before | After |
|---|---|---|
| Tests | 291 | **295** (3 partial-month tests, 1 stale-answer regression test) |
| Coverage (src/) | 97% | 97% |
| Suite runtime (with coverage) | 54.4 s | 52.1 s |
| Warm full rerun (any interaction) | 0.40 s | **0.23 s** (−42%) |
| Cold first load | 5.16 s | ~5.1 s (unchanged; dominated by the spinnered, cached 8-model CV) |
| Uncached per-rerun statistical work | 0.171 s | ~0 (cached; changepoints at 0.4 ms deliberately left uncached) |
| Ruff / Bandit / pip-audit | clean / clean / clean | clean / clean / clean |
| Headless launch | HTTP 200 | HTTP 200 |

Verification sweep (post-change): all 15 tabs render with zero exceptions;
weather on × electricity/gas/total all clean; Consultant behaves correctly
for canonical, paraphrased, single-word, and unsupported questions (honest
fallback, no false match); the Data Quality tab surfaces the July 2026
month-to-date warning; no new outbound network calls (telemetry now
disabled — strictly fewer than before); no data files added to git.

## Intentionally retained

- **Statistical behavior everywhere** — no model, threshold, or seed was
  touched. The only numeric-output change is display precision (whole
  pounds for forecast figures).
- **`year_to_date_comparison` (F11)** — dead but tested and harmless;
  removal or wiring-in is an owner decision.
- **Real household data in the repo (F13)** — the owner's data in the
  owner's private repo; flagged as a hard blocker for any public release,
  not unilaterally deleted.
- **`clean`/`merged` variable names (F15)** — repo-wide renames would be
  exactly the unreviewable churn the audit brief warns against; the data
  contracts are now documented in the audit (§4) instead.
- **Unpinned requirements (F12)** — environment-changing; recommended, not
  imposed.
- **The 15-tab flat layout beyond the four renames (F14 full)** — a
  question-grouped navigation (Home / Understand / Investigate / Plan /
  Ask / Data & methods) remains the right eventual shape but is a redesign
  needing owner input, not an audit patch.

## Recommendations rejected: none

Everything found was either implemented or explicitly deferred with a
reason above; no finding was judged wrong after investigation. Two
prompt-suggested tools were not adopted: mypy/pyright (not configured in
the project; adding a type-checker cold would blend hundreds of
stylistic errors into an audit diff — recommended as its own future task)
and property-based testing (would add a dependency; noted as a test-quality
gap, not a defect).

## Remaining risks

1. **Public release requires removing real data** (F13) — unchanged, by design.
2. **Current-month bias is now *flagged*, not corrected** — trailing-window
   KPIs still include the month-to-date value; the honest fix chosen was
   disclosure (warning + Medium confidence + Consultant caveat) rather than
   silently excluding real data. Excluding it is a defensible alternative
   the owner may prefer.
3. **Data-snapshot-coupled tests** (F15) — adding the August 2026 export
   will require updating several assertions in one sitting.
4. **Multi-user deployment remains out of scope** — process-wide data cache
   is single-household by design.
