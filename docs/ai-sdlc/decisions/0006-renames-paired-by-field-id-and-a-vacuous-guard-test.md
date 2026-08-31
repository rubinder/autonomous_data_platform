# 0006 — Renames paired by field ID, and the guard test that was vacuous

**Type:** feature ([#1](https://github.com/rubinder/autonomous_data_platform/issues/1)),
plus a self-inflicted vacuous test caught by mutation
**Made:** during implementation, by planting a position-matching mutant
**Shipped:** `sensors.build_rename_map()` groups historical column names by
Iceberg field ID; `classify()` gains `renaming`, `widening` and `enum_drift`

## What changed

The agent used to report `checkNumber → check_reference` as two findings — a
`breaking` drop and an `additive` add — on the one table whose headline property
is that a rename is metadata-only because the field ID does not move. The
evidence to do better was already being fetched and then discarded:
`pyiceberg_engine.schema_history()` returns `field_ids`, and `observe()` read
only `["columns"]`.

Measured, at full scale, on a warehouse with no `ops.monitor_results` history:

| Sequence | Before | After |
|---|---|---|
| `all → agent` | 0 findings | 0 findings |
| `all → drift-demo → agent` | 5 findings, 2 incidents | **4 findings, 2 incidents** |
| `all → monitor → drift-demo → agent` | 6 findings, 3 incidents | **5 findings, 3 incidents** |

`renaming` is in `ACTIONABLE`: no data moved, but the published contract now
names a column that does not exist, and that needs a recorded decision. The
severity split only earns its keep if it changes what the agent *does*, so
there is a test asserting exactly which severities act.

## The vacuous guard test

The first draft included a test called
`test_rename_pairing_does_not_capture_an_unrelated_added_column`, written
specifically to prove the pairing was by field ID and not by position or name.
It constructed an `ObservedState` with `renames=` supplied as a **fixture** —
so it exercised `detect()`'s use of the map and never `build_rename_map` itself.

A planted mutant that replaced field-ID grouping with ordinal position matching
**passed all 19 tests.**

The reason it survives is worth recording, because it is not obvious: *a rename
does not move a column*. Ordinal matching therefore gets a plain rename right,
and gets the rename-plus-add case right too, since PyIceberg appends new columns
at the end. It breaks only when a column is deleted from the middle and every
later column shifts up by one — at which point it invents a rename that never
happened and simultaneously hides a real drop.

So the discriminating case had to be built deliberately:

```
history:      {a:1, b:2, c:3}  ->  {a:1, c_new:3}
by position:  [a,b,c] vs [a,c_new]  ->  pairs b -> c_new     (wrong, twice)
by field id:  id 2 is gone (a real drop); id 3 changed name  (the rename)
```

Three tests now cover it: a unit test on `build_rename_map` with that history, an
end-to-end `detect()` test asserting `b` is still `breaking`, and a real-Iceberg
test that deletes `detailCategoryId` and renames `checkNumber` in one step. The
position mutant and a name-similarity mutant both fail against them.

**The generalisable point:** a test that supplies the thing under test as a
fixture is testing the consumer, not the producer, however precisely its name
describes the producer's contract. This is the third instance of the same shape
in this repository (Task 8's dedup test, Task 16's widening test), and the only
reliable way any of the three were found was planting the mutant and watching the
suite stay green.

## Enum drift is a separate contract key, not an expectation

`enum_watch` is a new top-level key in the contract YAML rather than a new
`expectations` entry. Everything in `expectations` is a fail-closed gate:
`assert_valid` raises on failure, and `_evaluate` raises on any type it does not
recognise — which is precisely what stops a typo'd check from reading green
forever. An enum watch must never block a build, so putting it in `expectations`
would have required a branch that always passes: a check that displays green
while measuring nothing, which is the exact defect
`silver_txn_quarantine_rate: SELECT 0.0` already was once.

`baseType` is deliberately *not* watched. It already has a fail-closed
`accepted_values` expectation because the pipeline's sign convention branches on
it, and an unrecognised value there would silently mis-sign spend. One column,
one semantics: a gate or a watch, never both.

A watch whose column produces no values classifies `breaking`, not silent-pass —
a watch that cannot see its column occupies the slot where the alert would have
been. A watch on a column with more than `ENUM_CARDINALITY_LIMIT` (200) distinct
values reports itself misconfigured rather than emitting a finding carrying the
whole cardinality.

## Review round: two defects the tests did not reach

Both were found by probing the shipped code adversarially rather than by
re-reading the diff, and both are the same shape as bugs this repository has
already fixed once.

**1. A watch on a struct column crashed the sensor.**
`sorted({...})` over a struct column's values raises
`TypeError: unhashable type: 'dict'`, out of `observe()` *and* out of
`detect()`. Bronze has three struct columns (`amount`, `runningBalance`,
`merchant`), so this is a config mistake a human could plausibly make — and the
consequence was not a bad reading on one column but a dead sensor for the whole
table, exactly the failure the unparseable-date guard exists to prevent:

> a monitor that crashes on bad data stops monitoring at exactly the moment
> something has gone wrong

Guarded at the source, and again in `detect()` for a hand-built `ObservedState`.
An unevaluable watch now reports `not_comparable` and classifies `benign`
("remove the watch"), and the reason lives in `enum_errors`, deliberately
*separate* from `enum_values`, so "could not evaluate" and "no drift" can never
share a representation.

**2. An entirely-NULL watched column reported clean.**
Zero values means zero *new* values, so the naive diff reads green precisely when
a categorical column has stopped being populated. `validator._evaluate` had
already settled this question in the other direction for both `range` ("a column
with nothing in it is not a column that is in range") and `freshness`; the enum
watch now agrees with them and classifies `no_values` as `breaking`.

Neither fires on real data: the clean run is still 0 findings with all seven
watches active.

## Known gap, stated rather than hidden

`drift-demo`'s `settlementDays int → long` step reports as `additive`, not
`widening`, because `drift-demo` adds that column before widening it — the
shipped contract never declared it, so there is no narrower declared type to
compare against. `widening` fires when a *declared* column gets wider, covered
against a real table by `test_real_widening_classifies_as_widening`. Adding
`settlementDays` to the contract to make the demo read better would make the
clean run report a phantom dropped column, which is a worse trade.
