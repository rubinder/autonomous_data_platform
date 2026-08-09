# ADR-0007: Persisted metric history, robust z-scores, and throttled alerts

**Status:** Accepted
**Date:** 2026-08-09

## Context

Row-level data quality checks — nulls, duplicates, ranges — all share a blind
spot: **an empty delta has no nulls, no duplicates, and nothing out of range.**
Every one of them passes when nothing arrives. Detecting "the number moved" or
"nothing arrived" requires comparing today's measurement against previous
measurements, which requires having kept them.

Three sub-decisions follow from that, and each had a wrong answer that looked
right first.

**Where does history live?** Deriving it from the Iceberg snapshot log is
tempting and mostly wrong. PyIceberg 0.11.1 commits an `overwrite()` as a
`delete` + plain `append` pair, and **neither half is ever tagged
`operation="overwrite"`**, so a monitor built on added-records reads every Silver
rebuild's entire row count as a spike.

**How is "unusual" defined?** Mean and standard deviation are the default and
they are self-defeating: one prior outlier inflates σ enough to hide the next
genuine shift. Measured on a baseline of twenty ~100 values plus one outlier at
1000, a real shift to 130 gives **stddev z = 0.065 (missed)** versus **MAD
robust z = 20.2 (caught)**.

**What stops an alert storm?** A breach that persists across scheduled runs
re-fires every run, and a channel that pages every six hours about the same
condition gets muted.

## Decision

**Persist metrics to `ops.monitor_results` and read baselines back from it.**
Seventeen checks defined in `monitors/{bronze,silver,gold}.yaml` write
`monitor_name`, `metric`, `baseline_median`, `status`, and `as_of` on every run;
baselines are loaded from that table. First run records with
`no baseline yet — recorded for future comparison` and cannot breach.

**Use median / median-absolute-deviation, not mean / stddev.** With a fallback
to *mean* absolute deviation when the MAD is exactly zero (a near-constant
baseline can have MAD 0 while still varying), and a `math.isclose` comparison —
not `==` — for the truly-constant case, because a Parquet double round-trip
changes the last couple of digits (`171.4137026799999` vs `171.41370268000063`)
and that must not read as a change.

**Choose the volume signal per write pattern.** Bronze is append-only and uses
per-snapshot `added_records`; Silver and Gold are full rebuilds and use
`count(*)`. This is not a stylistic split — measured: five 1,000-row Bronze
batches followed by a 1-row batch reports `5001 vs median 3000 → ok` under
cumulative `count(*)` (a near-total outage, invisible) and `1 vs median 1000 →
BREACH` under added-records.

**Add arrival SLAs on top of row-level checks** (`make arrival`, 9 checks over 3
tables): lag behind `AS_OF_DATE`, specific missing periods against the NYSE
trading calendar, and partial-arrival thinness.

**Throttle alerts on a `hashlib`-derived key** that is stable across processes,
logged to `ops.alert_log`. Consecutive identical alerts produce one delivery and
one throttle; distinct alerts never throttle each other.

## Consequences

**Positive.**

- The "nothing arrived" case is detectable at all, which it is not from row-level
  checks alone. `AS_OF_DATE=2026-07-31 make arrival` breaches 6 of 9 checks and
  names **23 specific missing trading dates** rather than reporting a lag number.
- Weekends and NYSE holidays are correctly not gaps (verified: Memorial Day
  2026-05-25 and Saturdays).
- Every one of the 17 monitors produces a real varying metric — none is stuck at
  a constant. This was enforced by review: an earlier
  `silver_txn_quarantine_rate` was literally `SELECT 0.0`, which could never fire
  while displaying as a passing check every run. It now reads the real quarantine
  table: 7 injected EUR rows produce metric 0.003488 and a breach.
- Alert keys are `hashlib`-based, so throttling survives process restarts —
  Python's builtin `hash()` is salted per process and would silently stop
  throttling (see `docs/ai-sdlc/decisions/0002-stable-merchant-hash.md` for the
  same bug caught elsewhere).

**Negative, and accepted.**

- **The monitors run in-process against the local warehouse, not as an
  independent scheduled service.** A failure of the pipeline is therefore also a
  failure of the thing meant to observe it. This is the most significant
  limitation of the ops layer and is called out in the README.
- Baseline history lives in the same warehouse it monitors. `make clean` erases
  the monitors' memory along with the data.
- Thresholds (`breach_ratio: 0.5`, `warn_ratio: 0.8`) are untuned starting
  policy. With a short history, a robust z-score over a handful of points is
  itself a noisy statistic.
- The throttle compares against the most recent few `ops.alert_log` entries, not
  a time window, and there is no escalation path. A persistent breach neither
  escalates nor re-notifies on a predictable schedule.
