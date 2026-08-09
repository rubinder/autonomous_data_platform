# ADR-0001: Apache Iceberg over Delta Lake and Apache Hudi

**Status:** Accepted
**Date:** 2026-08-09

## Context

The platform needs an open table format over object storage with ACID commits,
snapshot isolation, time travel, and schema evolution. Three candidates were
considered: Apache Iceberg, Delta Lake, and Apache Hudi.

Two constraints shaped the choice more than feature checklists did:

1. **The repository must run offline from a clean clone, on a laptop, with no
   JVM required.** That makes the quality of each format's *pure-Python* path a
   first-class criterion, not a nice-to-have.
2. **The engine must be swappable.** A design that only works because Spark is
   doing the hard parts teaches nothing about the table format.

Delta Lake's Python story (`delta-rs`) is real but its full feature surface —
and most of its documentation, tooling, and community answers — assumes
`delta-spark`. Hudi is more tightly coupled still: its write path, its table
services (compaction, clustering, cleaning), and its record-level index are
designed around Spark and Flink jobs, and the Python surface is thin.

Iceberg's specification is explicitly engine-neutral, and `pyiceberg` implements
catalog operations, commits, snapshot listing, schema evolution, and scan
planning as a first-class client rather than a binding.

## Decision

Use **Apache Iceberg**, with `pyiceberg` as the default client and a `SqlCatalog`
backed by SQLite at `./warehouse/catalog.db`.

The properties actually exercised in this repository are:

| Property | Where it is used | Proven by |
|---|---|---|
| Hidden partitioning | `day(_ingested_at)` on Bronze, `month(txn_date)` on Silver | partition specs resolve for 12/12 tables |
| Snapshot isolation | every layer write | re-run creates a new snapshot with no duplicate rows |
| Time travel | `make timetravel` | reads at snapshot N−1 vs N |
| Schema evolution | `make drift-demo` (5 schema versions) | old snapshots still readable, no rewrite |
| **Field-ID resolution** | `make cross-version` | a `checkNumber → check_reference` rename keeps field id 20; v1 files read under v5 |
| Snapshot expiry | `make maintenance` | 5 snapshots → 2, expired ids raise on scan |
| Metadata introspection | agent `sense` node | schema, snapshot log, per-snapshot added-records without scanning data |

Field-ID resolution is the property that most justifies the choice. It is what
makes a column rename a metadata-only operation, and it is the mechanism behind
the cross-version query described in the README.

## Consequences

**Positive.**

- `git clone && make all` works with no external services, no JVM, and no
  network. Every Iceberg claim in this repo is tested on that path.
- Adding the Spark engine (ADR-0002) required implementing a protocol, not
  rewriting the pipeline. The parity test proves this rather than asserting it.
- Metadata-only operations are genuinely metadata-only: the ops agent reads
  schema history and per-snapshot added-records without touching data files.

**Negative, and accepted.**

- `pyiceberg` 0.11.1 does not implement `MERGE INTO`. Silver and Gold are
  therefore full overwrites rebuilt from Bronze each run (ADR-0003), which is
  deterministic but does not survive real data volume. `MERGE INTO` exists only
  on the Spark path.
- `pyiceberg`'s `overwrite()` commits as a `delete` + `append` pair, and **no
  snapshot is ever tagged `operation="overwrite"`**. Anything reasoning about
  write semantics from the snapshot log has to derive it (`is_full_rebuild`),
  and the volume monitors had to be designed around this (ADR-0007).
- The API moves. `Table.expire_snapshots()` does not exist in 0.11.1; the real
  surface is `table.maintenance.expire_snapshots()` with no `retain_last`.
  Version pinning matters more than the docs suggest.
- Nested struct field ids are reassigned at `create_table` even though top-level
  ids survive. This does not affect the rename story (which is top-level) but it
  is a real asymmetry, visible in the `make schema-history` output where ids
  29–39 are consumed by nested fields.
