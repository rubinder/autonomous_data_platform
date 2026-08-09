# 0003 — Cross-version reads align by field ID, not by column name

**Type:** documented deviation from the plan; the plan's version was silently wrong
**Made:** during implementation (Task 16), confirmed by re-running the plan's code
**Plan said:** `_align_to()` — match each target field's *name* against
`rows.column_names`
**Shipped:** `_align_by_field_id()` — map each current field's `field_id`
through the historical snapshot's own schema

## The bug in the plan

`TableScan.projection()` in PyIceberg resolves a historical read against **that
snapshot's schema**, not the table's current one. So after the
`checkNumber → check_reference` rename, a scan of a pre-rename snapshot hands
back a column literally named `checkNumber`.

Measured directly:

```
old snapshot cols contains checkNumber: True   check_reference: False
current scan  check_reference null_count: 0 / 50 rows
```

Name-based alignment therefore looks for `check_reference` in a snapshot that has
no such name, finds nothing, and emits a typed NULL column. Every pre-rename row
loses that value — **in the one function whose entire purpose is to prove that
columns are not lost across a schema change.**

The failure is silent. The query returns the right number of rows with the right
column names and one column quietly all-NULL.

## Why it was caught

The implementer flagged it while writing the function; the reviewer did not take
that on trust and ran the plan's `_align_to()` verbatim against a real evolved
table:

| Implementation | `check_reference` null count on pre-rename rows |
|---|---|
| plan's name-based `_align_to()` | **800 / 800** |
| shipped field-ID version | **0 / 800** |

## The fix

`_align_by_field_id()` looks up each current field's `field_id` in the snapshot's
own schema to discover what that column was called at the time, and reads it
under the old name. Columns that genuinely postdate a snapshot still become
typed nulls, which is correct — they did not exist.

This is pinned by
`test_cross_version_query_reads_pre_rename_rows_under_the_new_name`, which
filters the union down to the oldest snapshot's rows and asserts
`check_reference` is populated there. Under name-based alignment it fails.

## Why this one matters beyond the bug

Field-ID resolution is the *reason* ADR-0001 chose Iceberg. Iceberg had done its
job perfectly — `checkNumber` and `check_reference` are both field id 20, and no
data file was rewritten. The plan's code would have thrown that away at the read
side and produced output that *looked* like a successful cross-version query.

A demonstration that quietly stops demonstrating its own mechanism is worse than
no demonstration, because it is presented as evidence.

## The generalisable rule

When a system's guarantee is expressed in one identifier space (field IDs), do
not reconcile in a different one (names) because the names usually match. "Usually
matches" is exactly the condition under which the failure is rare, silent, and
trusted.
