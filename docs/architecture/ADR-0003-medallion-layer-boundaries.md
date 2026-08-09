# ADR-0003: Medallion layer boundaries — what may and may not happen in each layer

**Status:** Accepted
**Date:** 2026-08-09

## Context

"Bronze / Silver / Gold" is a naming convention, not an architecture. Most of the
value comes from stating what is *forbidden* in each layer, because that is what
makes a violation reviewable. Without explicit boundaries, cleanup logic
migrates into ingestion, feature logic migrates into conforming, and the
lineage story quietly stops being true.

The specific failure this decision guards against: a transform that reads Bronze
and writes Gold. It works, it is faster, and it destroys the ability to explain
where any number came from.

## Decision

Three layers with hard rules.

**Bronze — append-only, source-faithful, never modified.**

- Preserves the source shape. Nested structs (`amount`, `runningBalance`,
  `merchant.address`) are **kept as structs**; flattening is Silver's job.
- Dates stay as source strings. No coercion, no cleaning, no dedup.
- Every row carries lineage: `_ingested_at`, `_source_file`, `_payload_hash`
  (sha256, order-independent), `_raw_payload` (the full original JSON).
- Write mode is `append`. Re-ingest produces duplicates **by design** — that is
  what an append-only raw layer means, and Silver's dedup is what resolves it.

Because Bronze does not coerce, malformed values are an *expected* input here.
The ops sensor is written to survive them: an unparseable `transactionDate` does
not crash the monitor, it becomes a `data_corruption` finding.

**Silver — conformed, deduplicated, contract-enforced.**

In order:

1. Type coercion — dates to `date`, timestamps to `timestamptz`, amounts to
   `decimal(18,2)`.
2. Deduplication on `id`, keeping the greatest `lastUpdated`, ties broken by
   greatest `_ingested_at`. Late-arriving corrections win.
3. `signed_amount = -amount WHEN baseType='DEBIT' ELSE amount`. Source amounts
   are unsigned and direction lives in `baseType`; this is the single most
   common correctness trap in this dataset, and it has a dedicated contract
   expectation (`conditional_sign`) plus a mutation-tested check.
4. Merchant normalization → `merchant_normalized`.
5. Currency assertion. Non-USD rows are routed to
   `silver.transactions_quarantine` with a reason — **never silently converted
   and never dropped**. Quarantine depth is itself monitored, so a rising reject
   rate surfaces rather than hiding.
6. Contract validation, **fail-closed**: a violation aborts the build before any
   write, leaving the target table absent rather than half-correct.

The merchant→ticker mapping lives in a **table** (`silver.merchant_ticker_map`),
not in transform code, so it is data: auditable, evolvable, and observable by the
ops agent.

**Gold — feature marts only.**

Reads Silver, never Bronze. Every feature at date *t* is computed from data with
timestamp ≤ *t*; rolling windows are trailing and right-closed; the target
`fwd_ret_5d` is the only forward-looking value in the layer.

**The rule that ties it together: no transform may skip a layer.** Gold reads
Silver. Silver reads Bronze. Nothing reads across two boundaries.

## Consequences

**Positive.**

- The no-lookahead guarantee is enforceable because feature logic lives in
  exactly one place. Every feature is recomputed by a second, independent
  implementation (`src/forecast/features.py::recompute_row`, sharing no code
  with the Gold SQL) across all 3,756 rows × 14 features = 52,584 comparisons:
  max discrepancy 1.42e-14, zero null-presence mismatches.
- Quarantine-instead-of-drop means row counts reconcile: kept + quarantined ==
  distinct Bronze rows, asserted in a test.
- Because Bronze keeps `_raw_payload`, any Silver bug is re-derivable from data
  already in the lakehouse. Nothing has to be re-fetched from the source.

**Negative, and accepted.**

- **Silver and Gold are full overwrites, rebuilt from Bronze on every run.** This
  is deterministic and easy to reason about and it does not survive real data
  volume. See the README's production-scale section.
- Keeping structs unflattened in Bronze means Silver carries the entire
  flattening burden, and a source schema change lands on Silver as a code change.
- Storing `_raw_payload` roughly doubles Bronze's footprint. At this scale that
  is free; at production scale it is a real decision with a real bill.
- Bronze re-ingest duplicating rows is correct but surprising, and it means the
  Bronze row count alone is not a data-volume signal — hence the per-snapshot
  added-records monitor in ADR-0007.
