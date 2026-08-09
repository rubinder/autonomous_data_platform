# The AI-driven SDLC actually used to build this

This is not a description of how one might build software with an AI. It is a
record of how *this repository* was built, including the parts that did not work
on the first attempt.

## The loop

```
spec ──▶ plan ──▶ ┌──────────────────────────────────────────────┐ ──▶ merge
                  │ for each of 20 tasks:                        │
                  │                                              │
                  │   implementer subagent (fresh context)       │
                  │        │                                     │
                  │        ▼                                     │
                  │   independent reviewer subagent              │
                  │        │                                     │
                  │        ├── clean ────────────────────────┐   │
                  │        │                                 │   │
                  │        ▼                                 │   │
                  │   fix round (original implementer,       │   │
                  │   resumed with its context)              │   │
                  │        │                                 │   │
                  │        ▼                                 │   │
                  │   scoped re-review of the fix ───────────┘   │
                  └──────────────────────────────────────────────┘
```

**Spec** (`docs/superpowers/specs/2026-08-09-autonomous-lakehouse-design.md`,
438 lines). Domain model, layer contracts, the planted signal, the Iceberg
feature list, the agent state machine, the test list, and the success criteria —
agreed before any code existed.

**Plan** (`docs/superpowers/plans/2026-08-09-autonomous-lakehouse.md`, 6,101
lines). Twenty tasks, each with its file list, its interfaces, its tests written
*before* its implementation, and often draft code. Interfaces were checked for
consistency across tasks in a plan self-review before execution started.

**Per-task implementer.** A subagent with a fresh context window and one task
brief. Fresh context is the point: it cannot be "sure" a thing works because it
watched it work three tasks ago.

**Independent review.** A *different* subagent, given the diff and the brief.
Its instruction is adversarial: assume the report is wrong, re-run the
commands, re-measure the numbers.

**Fix round.** The original implementer, resumed with its context, addressing
findings. Then a scoped re-review of the fix alone — because fixes introduce
bugs, and one did (Task 14, below).

**Ledger.** `.superpowers/sdd/2026-08-09-autonomous-lakehouse/progress.md`
records every dispatch, review verdict, measured number, deferral, and deviation
in one append-only file. It is the audit trail this document summarises.

## Model selection was a per-task decision

Not every task needs the same capability. Foundational tasks with API risk and
many downstream dependents got the strongest model; small, fully-specified
modules got a cheap one.

| Task | Model | Why |
|---|---|---|
| 5 (Iceberg catalog/engine) | opus | Foundational; 19 tasks depend on it; PyIceberg API risk |
| 10 (leakage-safe training set) | opus | The no-lookahead guarantee is the project's central correctness claim |
| 13 (metrics and report) | opus review | The honesty surface — where overclaiming would land |
| 16 (schema evolution) | opus | Field-ID semantics, the least forgiving Iceberg detail |
| 11 (purged CV splits) | haiku | Small, self-contained, complete code in the brief |
| most others | sonnet | Well-specified implementation work |

## Reviews were adversarial and empirical

The rule was: **a reviewer that only reads the diff is a linter.** Reviewers
re-ran the pipeline, re-derived the numbers with their own code, and tried to
make the tests fail.

Concretely, that produced:

- **A third independent implementation.** Task 10's reviewer wrote a *third*
  version of the feature computation to cross-check the pipeline's SQL against
  the test's recomputation: 3,678/3,678 rows agreed, max difference 7.1e-15.
- **Mutation testing to detect vacuous tests.** Task 8's
  `test_silver_dedup_late_arriving` passed — and could not have failed, because
  Bronze is deterministic so re-ingested duplicates are byte-identical and the
  `lastUpdated` tie-break was never exercised. Reverting the sort order
  (`DESC → ASC`) made only the *new* test fail while the old one stayed green,
  proving the old one was vacuous. The same technique caught Task 16's
  `test_widening_int_to_long_preserves_existing_values`, which tested an
  all-NULL column and would have passed had widening blanked the data.
- **Proving a test has teeth before trusting it.** Task 17's reviewer disabled
  the `TZ=UTC` fix and confirmed the cross-engine parity test then failed on a
  real ~8-hour timestamp divergence, caught at value level.
- **Re-running the brief's own draft code to check a claimed defect.** Task 16's
  implementer said the plan's name-based column alignment was wrong; the
  reviewer ran the plan's version verbatim and measured 800/800 nulls where the
  shipped version had 0/800.

## Several defects were in the *plan*, not the code

This is the finding worth generalising. An implementer following a plan faithfully
produces working code that is wrong in ways only re-measurement reveals.

| Where | The plan said | Reality |
|---|---|---|
| Task 13 | `verdict()` calls a win at ≥2% RMSE improvement | **Unreachable.** Persistence RMSE ≈ σ_y, so the best achievable ratio at correlation ρ is √(1−ρ²); a 2% gain needs \|ρ\| ≥ 0.199, but the best feature correlation is 0.156 and the best out-of-sample IC is 0.112. The clause could never pass — and since it bound alone, the "if all six beat the baseline, stop, you have a leak" canary was **dead code**. |
| Task 16 | Align cross-version columns by name | **Silently zeroes the renamed column.** PyIceberg scans a historical snapshot with *that snapshot's* schema, so pre-rename snapshots return `checkNumber`. Name alignment files those values under a column that no longer exists. Fixed to align by field ID — the exact mechanism the feature exists to demonstrate. |
| Task 18 | `silver_txn_quarantine_rate: SELECT 0.0` | A monitor that can never fire while displaying as a passing check on every run. |
| Task 14 | Bronze contract with no `freshness` expectation | Staleness detection was silently dead on the primary monitored table — the entire reason `transactionDate` is threaded through the pipeline. |
| Task 11 | Guard `usable < n_folds * 2` prevents degenerate folds | n=134 raises but n=135 silently yields five 2-row test blocks: metrics that look like results and are not. |
| Task 19 | Gap window ends at `min(newest, as_of)` | Always evaluates to `newest` in the stale case, making post-newest gaps *structurally* unreachable. Fixed to `max(as_of, newest)`; the stale feed now lists 23 specific missing dates. |

Three further defects were caught in a **plan pre-flight pass** before any task
ran (commit `30b824d`), including a volume-anomaly check defined against a
cumulative count that cannot collapse on an append-only table.

## Fixes introduce bugs, which is why re-review is scoped and separate

Task 14's fix round is the clearest case. The fix — reading column types from
metadata instead of a full scan — was correct, and it introduced a new failure:
a malformed but non-null `transactionDate` (`"zzz-not-a-date"`, which sorts as
SQL `MAX`) crashed the sensor with an uncaught `ValueError`. That is the worst
possible failure mode for a monitor: it dies exactly when the data goes bad, and
surfaces as a broken agent rather than a data-quality alert. Round 2 made
`newest_date` the max *parseable* value and added `unparseable_date_count` as its
own breaking finding, covered by six adversarial cases.

Task 17 needed two fix rounds for the same reason: round 1 protected string
literals in the SQL dialect shim but left SQL comments and quoted identifiers
silently corrupted, which the scoped re-review probed and caught.

## Where the human overrode the process

- **Deferrals were overturned.** Task 8's reviewer deferred the non-USD
  quarantine test; it was reinstated on the reasoning that a path which has never
  executed under test is exactly where silent row loss hides.
- **Tuning was explicitly forbidden.** Task 13's fix instruction said, in as many
  words: *do not tune toward beating the baseline; target calibration, not a win.
  If all six now read "beats", STOP and investigate — the canary is live again.*
  The result after recalibration was still **0 of 6**, and that is what the
  report says.
- **A change was reverted for being too flattering.** A 4× beta multiplier in the
  generator drove the planted correlation to ≈0.8 and trivialised the modelling
  task. Reverted; revert upheld at review.
- **Committing `docs/incidents/` was a human call**, on the grounds that agent
  output is a reviewable portfolio artifact and `write_incident` is idempotent
  per slug, so tree churn stays bounded to real drift.

## What this cost, and what it bought

Nineteen implementation tasks produced 33 commits: feature commits interleaved
with 13 review-driven follow-ups (11 `fix:`, 2 `test:`) whose messages say what
was found and why the fix is right. Eleven of the nineteen tasks needed at least
one fix round; two (14 and 17) needed two.

The bought outcome is that every quantitative claim in this repository was
measured by at least two independent parties, and the claims that did not survive
that — a model that beats its baseline, a full-pipeline engine parity claim, a
working leakage canary — were removed or narrowed rather than softened.

## Related records

- `decisions/` — implementation deviations from the spec or plan, with reasoning.
- `prompts/` — the dispatch and review prompt structure.
- `../architecture/` — ADR-0001 … ADR-0007, the durable design decisions.
- `../../.superpowers/sdd/2026-08-09-autonomous-lakehouse/` — briefs, per-task
  reports, review diffs, and the full ledger.
