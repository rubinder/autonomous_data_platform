# ADR-0005: YAML data contracts instead of Great Expectations

**Status:** Accepted
**Date:** 2026-08-09

## Context

The platform needs declared expectations about each table — schema, nullability,
accepted values, ranges, freshness, row counts — enforced at build time and
**also** readable by the ops agent, which diffs declared state against live
Iceberg metadata.

Great Expectations is the obvious off-the-shelf answer. It was rejected for
three reasons specific to this design:

1. **Two consumers, one source of truth.** The pipeline validates against the
   contract; the agent *diffs* against the contract. GE's model is
   suite → checkpoint → validation result, oriented at producing a data-docs
   report. Its results are not a convenient diff target for a program that wants
   to ask "which declared columns are missing from the live table, and which
   live columns are undeclared?"
2. **Machinery weight.** GE brings a context directory, datasource
   configuration, store backends, and a plugin surface. At four contracts and
   seven expectation types, that is more configuration than the thing being
   configured.
3. **Freshness semantics.** This project requires freshness evaluated against a
   *logical* as-of date rather than wall-clock (see below), which means a custom
   expectation regardless.

## Decision

Four YAML contracts in `contracts/`, evaluated by ~200 lines in
`src/contracts/validator.py`, returning a `ValidationReport` with per-expectation
pass/fail, observed values, and affected row counts.

```yaml
table: silver.transactions
version: 2
owner: data-platform
schema:
  - {name: id, type: long, nullable: false}
  - {name: signed_amount, type: "decimal(18,2)", nullable: false}
expectations:
  - {type: unique, column: id}
  - {type: accepted_values, column: base_type, values: [CREDIT, DEBIT]}
  - {type: conditional_sign, column: signed_amount, when: {base_type: DEBIT}, sign: negative}
  - {type: freshness, column: txn_date, max_lag_days: 3}
  - {type: row_count, min: 1000}
```

Two rules make it load-bearing:

**Silver and Gold fail closed.** A violation raises `ContractViolation` *before*
any write, so the target table is left absent rather than half-correct. Verified
against real data: an injected sign inversion aborts the build over 19,290 rows
and `silver.transactions` does not exist afterward.

**Freshness is evaluated against a logical as-of date, never wall-clock time.**
The dataset ends 2026-06-30, so a wall-clock freshness check would fail
permanently and get muted — which is exactly how freshness monitoring dies in
real systems. `AS_OF_DATE` (default: max `transactionDate` in Bronze) is threaded
through the pipeline, the contracts, and the agent, so `max_lag_days` means "lag
behind the pipeline's logical now". The staleness scenario is reproduced by
advancing `AS_OF_DATE`, not by waiting.

No expectation type silently no-ops; an unknown `type` raises rather than
passing.

## Consequences

**Positive.**

- One artifact, two consumers. The agent's schema-drift detection is a diff
  against the same YAML the build validates against, so a contract update
  changes both behaviours at once and they cannot disagree.
- `conditional_sign` is mutation-tested by two people independently: patching
  the branch to report zero violations makes `test_sign_inversion_is_caught`
  fail. The fail-closed guarantee is demonstrated, not asserted.
- No clock reads anywhere in the validator module, which is what makes the
  freshness tests deterministic.

**Negative, and accepted.**

- Seven expectation types is all there is. Anything else needs code.
- **The contract must be complete or the agent gets noisy.** An early version
  declared 6 of the 24 non-underscore Bronze columns, so the agent emitted ~19
  spurious "added column" findings on every run — not actionable, but exactly
  the alert noise the additive/breaking split exists to prevent. The Bronze
  contract now matches `schemas.BRONZE_TRANSACTIONS` exactly: no missing columns
  (no noise) and no extra columns (no false "dropped" breaking findings).
  Completeness is a maintenance obligation this design creates.
- A `range` expectation on an all-null column passes silently rather than
  reporting "no values", unlike `freshness`. Known, deferred.
- Reimplementing validation means reimplementing its bugs. Three were found
  during development and fixed; a mature library would have had them fixed
  already.
