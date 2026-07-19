# Month comparison: design, thresholds, and data dictionary

The primary user journey answers: *how did this month compare?* This
document explains the defaults, the arithmetic, the interpretation
thresholds, and how edge cases are handled. The implementation lives in
`src/monthly_comparison.py` (calculations), `src/monthly_narrative.py`
(wording rules), `app/tabs_month.py` (the page and shared state), and
`src/consultant_month.py` (Consultant answers).

## Why same-month-last-year is the default

The default comparison is the **latest complete month against the same
calendar month one year earlier** (June 2026 vs June 2025). Adjacent
months are *not* compared by default because seasonal differences make
that misleading: January vs December mostly measures winter, not
behaviour. The previous-month mode exists for short-term movement and is
labelled with exactly that warning.

## Comparison modes

| Mode | Comparator | Purpose |
|---|---|---|
| Same month last year (default) | The same calendar month, one year back | Fairest like-for-like performance comparison |
| Previous complete month | The month immediately before | Short-term movement only -- never a performance verdict |
| Typical value for this month | Median of all previous same-calendar months | "Is this January ordinary, low, or high?" |
| Best recorded value | Lowest-consumption previous same-calendar month | "Is this my best January?" |
| Worst recorded value | Highest-consumption previous same-calendar month | "Is this my worst January?" |
| Long-term trend | Trailing 12-month totals, multi-year overlay | Context; full detail under Long-term trends |

Best/worst/typical only ever compare a calendar month with itself across
years -- January is never compared with July. The median (not the mean)
defines "typical" because the history is short and anomalous months
(December 2024) would drag a mean.

## Partial months

A month is *complete* once the calendar has moved past it (the same
convention as `flag_in_progress_month`). The in-progress month is:

- excluded from the month selector and from same-month history,
- announced with a notice ("July 2026 is incomplete. The primary
  comparison uses June 2026."),
- inspectable only inside a clearly-labelled month-to-date expander,
- never annualized, extrapolated, or compared as if complete.

## Weather decomposition

Reuses the already-fitted degree-day regression (`src.energy_signature`)
-- nothing is refitted when a month is selected:

- weather-expected kWh for a month = fitted kWh/day x days in month
- **weather-explained change** = expected(selected) - expected(comparison)
- **unexplained change** = actual change - weather-explained change
- **weather-adjusted kWh** = actual - (expected - base load), i.e. base
  load plus residual

An unexplained gap is called **meaningful** only beyond 1.645 residual
standard deviations of the fit (one-sided 95% normal approximation) --
the same convention `src.energy_signature` uses for annual residuals.
"Explained by colder weather" is only claimed when the weather moved in
the explaining direction (more heating degree days, positive expected
change) **and** covers at least half of the actual change
(`weather_explains_most`). A gap merely sitting inside residual noise is
described as "within this home's normal variation", never credited to
weather.

## Change categories (practical, not statistical)

From `config.MonthComparisonThresholds`, shown in the page's Technical
details expander:

| Band | Range | Narrated as |
|---|---|---|
| little | < 5% | "about the same"; never called meaningful |
| moderate | 5-20% | stated plainly with the driver |
| large | > 20% | stated plainly with the driver |

These are materiality bands for a household bill, **not** statistical
significance -- a single month's change is not formally testable at ~35
observations, and the wording never claims otherwise.

## Judgement labels

Words carry the verdict; colour never does (an increase is not
automatically red -- higher winter use may be expected):

- *Better than comparable period* -- usage fell
- *Similar* -- inside the little-change band
- *Higher but weather-explained* -- weather covers >= half the increase
- *Higher and unexplained* -- meaningful residual gap; investigate
- *Higher* -- up, but neither weather-driven nor beyond residual noise

## Fuel attribution

`fuel_contributions` splits the combined change into per-fuel kWh
changes (electricity + gas = total by construction). When the fuels move
in opposite directions the narrative says so ("Electricity drove the
increase; gas use actually fell") instead of quoting a >100% share.

## Cost and carbon

- Cost change is split exactly: change x comparison-month rate (usage
  part) + rate change x current usage (price part). Standing charges are
  not in the OVO exports, so they are never estimated.
- Carbon compares the same months per fuel using the static cited
  factors in `config.CarbonConfig`; estimates are labelled rough.

## Edge cases and honest fallbacks

- No same month last year -> "No March 2025 in the data, so a same-month
  year-on-year comparison isn't possible."
- Fewer than 2 previous same-calendar months -> typical/best/worst modes
  say there isn't enough history (confidence capped Medium below 4).
- Prior-year value of zero -> percentage change is undefined and omitted
  (absolute kWh change still shown).
- Leap-year February -> daily averages use actual days in month.
- Missing fuel exports -> that fuel's card says "no data"; attribution
  requires both fuels.
- A record-setting month is never described as "within the normal
  range"; a record high inside residual noise still gets a "worth a
  quick look" action because the noise threshold is winter-dominated.

## Shared state

The selected month / fuel / mode live in `st.session_state`
(`month_cmp_*` keys), resolved once per rerun in `main()` via
`build_month_context`. Home, the comparison page, and the Consultant all
read the same objects, so the Consultant can never answer from stale
state. Warm interactions (month/mode/fuel switches) measured at ~0.2 s
on the real dataset -- no model is refitted on selection changes.

## Verification

`scripts/verify_month_comparison.py` re-runs the exact journey objects
against the real exports for the brief's seven scenarios (latest
complete month, December 2024, a normal winter month, a summer month,
lower-YoY, higher-YoY, latest partial month) and prints every computed
value plus the final narrative for manual wording review.
