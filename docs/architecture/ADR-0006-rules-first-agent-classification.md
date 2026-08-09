# ADR-0006: Rules-first agent classification, LLM reserved for unmatched cases

**Status:** Accepted
**Date:** 2026-08-09

## Context

The ops agent classifies each finding as `breaking`, `additive`, or `benign`,
and that classification decides whether an incident gets filed. Two designs were
available:

**LLM-first** — hand the observed state and the contract to a model and let it
reason. Flexible, and it makes the demo look impressive.

**Rules-first** — deterministic classification for every case the design
anticipates, with a model call reserved for genuinely unmatched findings.

LLM-first fails a hard requirement of this project: **CI must never depend on a
model call.** A test suite that needs an API key is a test suite that is red for
anyone without one, non-deterministic for everyone who has one, and untestable
offline. It also means the classification of a dropped column — a case with an
unambiguous correct answer — could vary between runs.

There is a subtler reason. A drift classifier is a *policy*, and policy should be
reviewable in a diff. "Row count below 50% of the trailing median is breaking"
belongs in version control where it can be argued with, not in a prompt where it
is rediscovered on each invocation.

## Decision

Deterministic rules first:

| Observation | Severity |
|---|---|
| Column added, nullable | `additive` |
| Column declared in contract but absent from table | `breaking` |
| Column type narrowed | `breaking` |
| Latest snapshot's added-records < 50% of trailing median | `breaking` |
| Newest data older than `max_lag_days` behind `AS_OF_DATE` | `breaking` |
| Unparseable values in the date column | `breaking` (`data_corruption`) |
| Monitor breach (`src/ops`) | `breaking` |

An optional Claude API call handles findings no rule matched. Absent
`ANTHROPIC_API_KEY`, the agent runs rules-only and every test passes. This is a
hard requirement, not a fallback of convenience.

The LangGraph state machine is:

```
sense ──▶ monitor ──▶ classify ──┬──▶ act ──▶ END
                                 └──▶ END (no action)
```

`act` writes `docs/incidents/<slug>.md` and is **dry-run by default**;
`--execute` is required for `gh issue create` to actually run. Slugs are derived
with `hashlib`, so they are stable across processes and re-running the agent
rewrites the same file instead of accumulating duplicates.

## Consequences

**Positive.**

- The whole suite runs offline. Verified with a dummy `ANTHROPIC_API_KEY` and an
  unreachable proxy: 180 passed, same runtime.
- Behaviour is reproducible: a clean lakehouse produces **0 findings**; after
  `make drift-demo` it produces exactly **5** — one breaking dropped column
  (`checkNumber`, renamed away), three additive new columns
  (`check_reference`, `merchantCategoryCode`, `settlementDays`), and one breaking
  volume collapse (5 rows against a trailing median of 250,000) — and writes
  exactly 2 incident files, for the breaking findings only.
- The rename producing a `breaking` finding is the right answer, not noise:
  Silver reads Bronze by name, so a rename *is* a drop as far as a name-keyed
  contract is concerned, even though Iceberg lost nothing. That distinction —
  the storage layer is fine, the consumers are not — is precisely what a human
  should review.
- `monitor_breach` has an explicit classifier branch, so it cannot fall through
  to the (offline-disabled) LLM path and get silently swallowed as benign.

**Negative, and accepted.**

- Rules only cover anticipated drift. A novel failure mode reaches the LLM path,
  which is disabled by default, and therefore goes unclassified.
- The 50%-of-trailing-median threshold is a judgement call with no tuning behind
  it. It is a starting policy, not a calibrated one.
- The optional LLM path is the least-exercised code in the repository precisely
  because CI never runs it.
- `act` has no cross-run diffing beyond the slug, so a finding that changes its
  detail text but keeps its slug overwrites the previous incident rather than
  appending a history.
