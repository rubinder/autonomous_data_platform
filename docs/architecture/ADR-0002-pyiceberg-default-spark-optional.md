# ADR-0002: PyIceberg + DuckDB as the default engine, Spark opt-in behind one protocol

**Status:** Accepted
**Date:** 2026-08-09

## Context

Spark is the default assumption for a lakehouse, and it is the wrong default for
this repository. A Spark-only build needs a JVM, a Maven Central round-trip for
`iceberg-spark-runtime-3.5_2.12`, several minutes of session startup across a
test suite, and a CI runner willing to pay for all of it. That cost buys nothing
for a 250,000-row dataset.

But dropping Spark entirely would leave the "engine-neutral table format" claim
in ADR-0001 untested. An interface with one implementation is a guess about what
would be portable, not a demonstration.

## Decision

Define a small `LakehouseEngine` protocol (`src/lakehouse/engines/base.py`) and
implement it twice.

```python
class LakehouseEngine(Protocol):
    def create_table(self, table_def: TableDef) -> None: ...
    def table_exists(self, ident: str) -> bool: ...
    def arrow_schema(self, ident: str) -> pa.Schema: ...
    def append(self, ident: str, data: pa.Table) -> None: ...
    def overwrite(self, ident: str, data: pa.Table) -> None: ...
    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table: ...
    def snapshots(self, ident: str) -> list[int]: ...
    def snapshot_row_counts(self, ident: str) -> list[int]: ...
    def snapshot_details(self, ident: str) -> list[dict]: ...
    def schema_history(self, ident: str) -> list[dict]: ...
    def sql(self, query: str, tables: dict[str, str]) -> pa.Table: ...
```

- **Default (`ENGINE=pyiceberg`)** — PyIceberg for catalog and commits, DuckDB
  for SQL over the scanned Arrow tables. Pure Python, offline, fast.
- **Opt-in (`ENGINE=spark`)** — `pyspark==3.5.1` with the Iceberg runtime, plus
  two extensions the PyIceberg path cannot offer: `merge_upsert()` and
  `compact()`.

The abstraction is proven by `tests/test_engine_parity.py`, which builds
`silver.transactions` through both engines and diffs the results row-for-row and
value-for-value.

`spark_engine.py` is never imported on the default path — `get_engine()` imports
it lazily only when `ENGINE=spark` is selected — so the pure-Python install has
no PySpark dependency at all.

## Consequences

**Positive.**

- The parity test **actually ran**: 9 tests pass on PySpark 3.5.1 / Java 17 /
  Iceberg runtime 1.5.2, and Silver built through both engines is identical.
  This is a checked claim, not an asserted one.
- The test has demonstrated teeth. Neutralising the `TZ=UTC` fix on an
  EDT host produces a real ~8-hour `_ingested_at` divergence
  (`1782885600000000` vs `1782856800000000`), caught at value level — so the
  diff genuinely compares values, not just shapes.
- CI stays green and fast on the pure-Python path; the parity tests
  `skipif` out cleanly when PySpark is absent.

**Negative, and accepted.**

- **Parity covers `silver.transactions` only** — not Bronze, not Gold. The
  abstraction is proven where it was tested and nowhere else. The README says
  this explicitly rather than claiming full-pipeline parity.
- The Spark path needs Maven Central for the Iceberg JAR, so it is **not
  offline** the way the PyIceberg path is.
- Supporting two engines surfaced three environment bugs that pure-Python never
  would have: `toPandas()` breaks on Python ≥ 3.12 (removed `distutils`), Py4J
  shifts collected timestamps by host timezone, and Spark rejects DuckDB's
  `TIMESTAMP WITH TIME ZONE` cast spelling. Each needed a real fix. The dialect
  shim that rewrites that cast is a string substitution, not a SQL parser: it
  protects string literals and comments and **raises** if the DuckDB spelling
  survives outside a literal, rather than silently corrupting a query.
- The protocol has grown once already (`arrow_schema` was added in Task 16). Any
  third engine inherits every method, including the ones added for one caller.
