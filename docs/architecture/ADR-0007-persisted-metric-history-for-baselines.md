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
re-fires every run, and a channel that pages on every scheduled run about the same
condition gets muted.

## Decision

**Persist metrics to `ops.monitor_results` and read baselines back from it.**
A `make monitor` run produces seventeen results, and it is worth being precise
about where they come from and which of them carry a number:

| | count | declared where | metric |
|---|---|---|---|
| SQL monitors | **10** (2 bronze / 6 silver / 2 gold) | `monitors/{bronze,silver,gold}.yaml` | a real, varying number |
| column-type checks | **7** | generated in Python by `monitors.check_column_types`, one per field of the two typed contracts | `metric=None` — a type either matches or it does not |

All seventeen write `monitor_name`, `metric`, `baseline_median`, `status`, and
`as_of` on every run; baselines are loaded from that table, filtered on
`metric IS NOT NULL`, which is exactly why the seven column-type checks never
contribute to one. First run records with
`no baseline yet — recorded for future comparison` and cannot breach.

So "adding a check is a data change, not a code change" is true of **10 of the
17** — the SQL ones. Adding a column-type check means adding a field to a
contract; adding a *table* to the type checks means editing `runner.TYPED_TABLES`.

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

**Add arrival SLAs on top of row-level checks** (`make arrival`, 8 checks over 3
tables): lag behind `AS_OF_DATE` and specific missing periods against the NYSE
trading calendar on all three, plus partial-arrival thinness on the two tables
that have a *structural* rows-per-period floor (one row per company per trading
day). `silver.transactions` has no such floor — transactions per day is a volume,
not a shape — and declares none, so it emits no thinness check. It used to
declare `min_rows_per_period: 1`, which against a `count(*)` GROUP BY is
unsatisfiable: a period that appears at all appears with at least one row. That
is 8 checks rather than 9 because two of the nine could never fire, and a check
that cannot fire must not be displayed as a passing check.

**Throttle alerts on a `hashlib`-derived key** that is stable across processes,
logged to `ops.alert_log`. Consecutive identical alerts produce one delivery and
one throttle; distinct alerts never throttle each other. `make monitor` is the
production caller: every run routes its `warn`/`breach` results through
`runner.route_alerts`, dry-run unless `--execute`.

## Consequences

**Positive.**

- The "nothing arrived" case is detectable at all, which it is not from row-level
  checks alone. `AS_OF_DATE=2026-07-31 make arrival` breaches 6 of 8 checks and
  names **22 specific missing trading dates** rather than reporting a lag number.
- Weekends and NYSE holidays are correctly not gaps — and this is now pinned by
  tests rather than cited: Saturdays, Memorial Day 2026-05-25, and holidays
  *past the end of the data*. That last one was a live defect. The holiday set
  was scoped to `START_DATE..END_DATE` and ended 2026-06-19, but the gap window
  runs to `as_of`, so at the documented `AS_OF_DATE=2026-07-31` the calendar had
  already run out and 2026-07-03 — Independence Day observed, since 4 July 2026
  is a Saturday — was reported as a missing trading day. The headline was 23
  dates; the correct number is 22. The calendar now covers all of 2026.
- All 10 of the SQL monitors produce a real varying metric — none is stuck at a
  constant. This was enforced by review: an earlier `silver_txn_quarantine_rate`
  was literally `SELECT 0.0`, which could never fire while displaying as a
  passing check every run. It now reads the real quarantine table: 7 injected
  EUR rows produce metric 0.003488 and a breach. The other 7 checks are the
  column-type ones, which are pass/fail by nature and carry `metric=None`; they
  are excluded from baselines and are not, and cannot be, "varying metrics".
- An unrecognised `kind` in a monitor's YAML now raises instead of returning
  `ok`. One typo used to mint a monitor that was counted, displayed green, and
  evaluated nothing — the same defect class as `SELECT 0.0`, one layer up.
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
