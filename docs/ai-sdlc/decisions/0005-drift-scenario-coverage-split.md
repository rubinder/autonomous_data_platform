# 0005 — Two of the five drift scenarios are covered by unit test, not by mutating a real table

**Type:** deliberate coverage split, recorded so it is not mistaken for an omission
**Decided:** plan self-review, confirmed in implementation (Tasks 14 and 16)

## The spec's five scenarios

Design spec §8 lists five drift scenarios the agent must detect:

| # | Scenario | How it is exercised |
|---|---|---|
| 1 | Additive column appears in Bronze | **real table** — `make drift-demo` adds `merchantCategoryCode` and `settlementDays` |
| 2 | Column dropped | **real table** — `checkNumber → check_reference` rename; the contract keys by name, so the rename presents as a drop |
| 3 | Type narrowing (long → int) | **unit test only** |
| 4 | Volume anomaly | **real table** — a 5-row batch appended against a trailing median of 250,000 |
| 5 | Staleness | **real table** — `AS_OF_DATE` advanced past the data |

## Why 3 is not demonstrated against a real table

PyIceberg's `update_schema` will not narrow a column type. Narrowing is not a
metadata operation in Iceberg — it can invalidate already-written values — so
producing the scenario against a live table would require either rewriting data
files or hand-editing table metadata to fabricate a state Iceberg refuses to
create.

Both options make the demo lie about what the storage layer does. The classifier
rule (`narrowed type → breaking`) is real code and is tested directly against a
constructed `ObservedState`; what is not tested is Iceberg producing that state,
because Iceberg does not produce that state.

The *widening* direction is demonstrated for real, and properly: `settlementDays`
is promoted int → long with 50 real int32 values (`7 * i`) written **before** the
promotion, then asserted identical afterwards, with the Arrow type checked on
both sides. An earlier version of that test ran against an all-NULL column and
would have passed even if widening had blanked the data — caught at review.

## Why 2 counts as a real-table demonstration anyway

A rename is not a drop at the storage layer: field id 20 is unchanged and no file
is rewritten. But `silver.transactions` reads Bronze **by name**, and the
contract declares columns **by name**, so from every consumer's point of view the
column is gone. The agent classifying it `breaking` is the correct answer, and it
is a more interesting one than a literal drop would be: the storage is fine and
the consumers are broken, which is exactly the situation a human should review.

## What this means for the coverage claim

The honest statement is: **the classifier handles all five scenarios; four of the
five are reproduced end-to-end against a real Iceberg table.** Anywhere the
five-scenario coverage is summarised, that qualification belongs with it.
