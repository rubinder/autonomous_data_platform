# 0007 — Finding history lives in Iceberg, not in the incident files

**Type:** design decision for [#6](https://github.com/rubinder/autonomous_data_platform/issues/6)
**Alternative considered:** an append-only event log inside each `docs/incidents/*.md`
**Shipped:** two append-only Iceberg tables, `ops.finding_log` and `ops.agent_runs`

## The problem

The platform was accumulating a narrative and discarding it.

`ops.monitor_results` has carried `run_at` (partitioned by month) since Task 18,
and **nothing ever read it back**. `make monitor` wrote; no code queried.

Worse, the one artifact that looked like a history was actively destroying one.
`actions.incident_slug()` is deliberately stable so a recurring finding rewrites
a single file rather than spawning one per run — the right call for bounded tree
churn, made deliberately by a human, and the reason `docs/incidents/` is
committable at all. But it means `write_incident` **overwrites**. Yesterday's
state is gone. Nothing in the repository could answer "when did this start?" or
"did yesterday's problem clear?"

## Why the table, not the file

The rejected alternative was to append a dated event line inside each incident
file, keeping everything readable in the repo with no warehouse.

Against it:

- **Diffing means parsing prose.** "What changed since the previous run" becomes
  a Markdown-scraping exercise over files whose format exists for humans. Any
  format change silently breaks the diff.
- **It reintroduces unbounded growth** — the exact thing the stable slug was
  chosen to avoid. A finding open for six months is a file with 180 event lines,
  and every agent run dirties the git tree.
- **Findings that never file an incident would have nowhere to live.** Only
  `ACTIONABLE` severities write incident files. `additive`, `widening` and
  `enum_drift` findings would be absent from a file-based history entirely —
  and those are precisely the ones whose *accumulation over time* is the story
  the report is supposed to tell.
- The ops tables already exist, already carry `run_at`, and are already
  partitioned by month. The table answer is consistent with how this codebase
  already records operational facts.

The incident files keep their existing job: the current state of each open
actionable finding, readable in the repo, one file per finding.

## Why `ops.agent_runs` is a second table and not a flag

A clean run produces no findings, so it writes no rows to `ops.finding_log`. A
report reading only that table cannot distinguish **"the agent ran and found
nothing"** from **"the agent never ran"** — and it would render the second as a
clean bill of health.

That is the same empty-delta blind spot this codebase has now closed four times:
`silver_txn_quarantine_rate: SELECT 0.0`, the `range` check passing on an
all-NULL column, `freshness` on no values, and (in
[0006](0006-renames-paired-by-field-id-and-a-vacuous-guard-test.md)) an enum
watch on a column holding nothing. One row per run, findings or not, closes it
structurally rather than by remembering to check.

## The report takes no engine

`build_report(snapshot: ReportSnapshot) -> str` is handed a dataclass, not a
warehouse. It therefore **cannot** compute a metric even by accident, which is
what makes "the report recomputes nothing" a property rather than a comment.
`load_snapshot()` is the only function in the module that touches an engine, and
a test asserts the module never references `monitors.evaluate`, `sensors.detect`
or `validator.validate`.

The reason is not tidiness. A report that recomputes a number can disagree with
the monitor that raised the alert — and when those two disagree, the report is
the one people believe.

## A bug found on the way in

`incident_slug()` hashed `table | kind | column | change`, with no monitor name.
Every `monitor_breach` on a given table therefore hashed **identically**:

```
bronze.yodlee_transactions_raw|monitor_breach|None|
```

With one monitor breaching, invisible. With two, the second incident file
silently overwrote the first — and the day-over-day diff would have reported a
finding as "cleared" while it was still breaching, which is the worst possible
lie for this feature to tell.

Fixed by extracting `sensors.finding_key()` as the single definition of finding
identity, used by both the incident filename and the `ops.finding_log` row. They
have to agree, or "first seen" describes something different from the file on
disk. The filename also stopped rendering `None` for breaches that carry no
column.

## Review round: two defects the tests did not reach

**1. Running `make agent` twice in one day doubled the findings.**
`ops.finding_log` is append-only, and re-running the agent on the same date is
an entirely ordinary thing for a human to do. The report counted *rows*, not
findings, so a second run turned one open finding into "2 still open". A
day-over-day diff whose counts are wrong is worse than no diff at all, because
the counts are read as a measurement. `_latest_per_key` now collapses to one row
per `finding_key` per run; the same applies to `make monitor`.

**2. The report was not byte-stable across regenerations.**
The diff sections were built from set differences and rendered in set-iteration
order, so simply regenerating a report reshuffled its lines. These are committed
artifacts: an unstable render produces a git diff on every regeneration that has
nothing to do with what changed, and a diff that is usually noise is a diff
nobody reads. Now sorted by finding key, with a test that renders twice and
compares.

Both were found by probing the shipped code, not by re-reading the diff. Neither
would have failed a test that only asked "does the report render?"

## Scope note

`drift-demo --day N` was added because the report needed something to narrate.
`evolve_all()` applies four schema changes in one shot, which produces one before
and one after and no story in between. Bare `drift-demo` is unchanged.
