# Weather Context Engine: snow, wind, and severe weather

Contextual explanation for unusual energy months -- **not** causal proof, and
**not** part of the temperature model.

## Design principle

The energy-signature regression stays exactly as it was: monthly average
daily kWh regressed on heating and cooling degree days only. Snow, wind, and
precipitation are **contextual evidence** attached to already-detected
unusual months. They were deliberately kept out of the regression because:

- ~35 monthly observations cannot support more predictors without overfitting
  (the existing 3-parameter model is already at the sensible limit -- see
  `src/energy_signature.py`).
- Severe weather correlates heavily with cold weather; adding both would make
  coefficients uninterpretable without a formal model-selection study.
- The honest question severe weather answers is "could this plausibly have
  contributed?", not "how many kWh did it cause?".

Adding these variables to the regression is allowed **only** after a separate
model-selection study shows the larger model is stable and better.

## What the system can and cannot say

Can say: "December 2024 had 5 severe-gust days (max 91 km/h); consumption
remained +219 kWh above the temperature-adjusted expectation, so severe
weather alone does not explain the full increase."

Cannot say (and is tested to never say): snow *caused* working from home;
roads were closed; occupancy increased. Monthly billing data cannot see
behaviour -- wording is always "may have contributed", "is consistent with",
"cannot be confirmed from monthly data".

## Data dictionary: daily weather (`src/weather.py`, `DAILY_WEATHER_SCHEMA`)

Fetched from the Open-Meteo archive API (exact variable names, no key, 30 s
timeout, TLS verified):

| Column | Open-Meteo variable | Unit | Required | Missing-value policy |
|---|---|---|---|---|
| `temp_mean_c` | `temperature_2m_mean` | °C | yes -- fetch fails without it | NaN day doesn't count toward monthly coverage |
| `snowfall_cm` | `snowfall_sum` | cm | no | NaN (never silently 0 -- 0 means "none observed") |
| `snow_depth_m` | `snow_depth_mean` | m | no | NaN, month marked incomplete |
| `precipitation_mm` | `precipitation_sum` | mm | no | NaN, month marked incomplete |
| `wind_speed_max_kmh` | `wind_speed_10m_max` | km/h | no | NaN, month marked incomplete |
| `wind_gust_max_kmh` | `wind_gusts_10m_max` | km/h | no | NaN, month marked incomplete |

Validation (`validate_daily_weather`): duplicate dates raise; negative
snowfall/precipitation, snow depth > 6 m, wind > 300 km/h are set to NaN with
a logged warning -- implausible values are never kept or silently clipped.

## Cache versioning

- v1 (pre-engine): `weather_daily_<lat>_<lon>.csv`, temperature only.
- v2: `weather_daily_v2_<lat>_<lon>_<timezone>.csv`, the six-field schema.
  Timezone is part of the identity because Open-Meteo aggregates daily values
  in the requested timezone.

Migration: v1 files are left untouched and simply ignored; the first run
after upgrading fetches the full range once (~4.4 s measured for 1,052 days)
and writes the v2 file (~39 KB); every later run is a disk cache hit (~2 ms,
zero network). A cache lacking any schema column is ignored, never
misread. Deferred: delta-fetching only missing date ranges -- a full-range
refetch happens at most once per new consumption month, which keeps the
added transfer trivial at this data size.

## Day-classification thresholds (`config.WeatherContextThresholds`)

All thresholds live in config, none in business logic. Conservative UK
conventions (chosen to understate severity):

| Threshold | Value | Source rationale |
|---|---|---|
| Snow day | snowfall > 0 cm | any measurable snowfall |
| Heavy-snow day | >= 5 cm/day | upper end of typical Met Office snow-warning accumulations |
| Heavy-rain day | >= 25 mm/day | Met Office heavy-rain guidance (~25 mm in a few hours) applied to a whole day |
| Strong-wind day | >= 62 km/h sustained | Beaufort gale (force 8) threshold |
| Severe-gust day | >= 80 km/h gust | typical Met Office yellow wind-warning gust level (~50 mph) |

A "severe day" (for run lengths) is any of: heavy snow, heavy rain, strong
wind, or severe gust.

## Relative severity (`src/weather_context.py`)

Each month's snowfall total, precipitation total, and max gust get a
percentile rank against the household's **own full history** (not
same-calendar-month peers: with ~3 years of data, season-matched comparison
would rank against n≈3). The rank is **strict** (share of months strictly
below) so a uniform history reads as Typical, not 100th-percentile-of-itself.

Bands: Typical (< 0.75), Above usual (0.75-0.90), Unusually high (>= 0.90),
Extreme (>= 0.90 **and** value > median + 3×IQR; never awarded when IQR = 0,
e.g. snowfall). No statistical significance is claimed at this sample size,
and no composite "severe weather score" exists -- any weighting of snow vs.
wind vs. rain would be opaque, so the classification label is the summary.

## Classification labels

`No notable severe weather / Snowy period / Prolonged snow / Strong-wind
period / Wet period / Wet and windy period / Mixed severe weather /
Incomplete weather coverage / Insufficient history` -- each with confidence,
observed facts, and a limitation statement. Labels use absolute day-count
criteria; the relative bands say whether that's *unusual for this location*.
In a windy location like Aberdeen, wind-based labels are common while the
gust band stays "Typical" -- both facts are shown, deliberately.

## Interpretation patterns (`src/weather_interpretation.py`)

"Material" residual = |z| > 1.5 of the fitted model's own residual spread --
the same bar as `src/cross_fuel_anomalies.py`.

| Pattern | Conditions | Interpretation shape |
|---|---|---|
| A | severe weather, residual within expectation | "broadly consistent with colder and severe winter weather" |
| B | severe weather, residual still materially positive | "may have contributed, but doesn't explain the full increase (+N kWh)" |
| C | electricity up, gas stable, severe weather | "not primarily heating-driven; time at home / appliances may have contributed -- unconfirmable" |
| D | no severe weather, material residual | "weather does not explain this month well; review occupancy/heating/appliances" |
| -- | severe weather but usage *dropped* | mirrored wording; absence "cannot be confirmed from monthly data" |

## Network & privacy behaviour

One HTTPS request to `archive-api.open-meteo.com` per uncovered date range
(coordinates and dates only -- no energy data ever leaves the machine),
30-second timeout, TLS verified, no API key. After the first successful
fetch the app is fully offline for weather. On a valid cache hit **no
network request occurs** (tested).

## Real-data verification highlights (July 2026)

- **December 2024** (the known 3-method spike): 2 snow days, 5 severe-gust
  days (max 91 km/h -- the Storm Darragh period), residual +219 kWh →
  Pattern B. The engine does *not* let severe weather absorb the whole
  anomaly.
- **December 2023**: the history's *Extreme* snowfall month (8 snow days),
  residual -22 kWh → not flagged, no explanation invented. Severe weather
  without an energy anomaly stays a non-event.
- **July 2026** (in-progress month): "Incomplete weather coverage", Low
  confidence -- honest degradation.
