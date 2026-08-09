"""Read Iceberg metadata and diff it against declared contracts.

Row counts, snapshot counts, and column types all come from table metadata,
not a data scan: `snapshot_row_counts` / `snapshot_details` read manifest
summaries, and `schema_history` reads the table's schema definitions --
already in Iceberg-native vocabulary (`long`, `string`, `struct<...>`, ...),
the same vocabulary contracts are written in, so no data read and no Arrow
bridging is needed for schema drift. The agent is meant to run cheaply and
often, and a sensor that scans the whole lake on every run is one that gets
switched off.

The one thing Iceberg metadata doesn't carry is the freshness column's
newest *value* -- that requires reading data. `observe()` pushes that read
through `engine.sql` as a `MAX(...)` aggregate (the same pattern
`bronze.max_txn_date` uses), so only the aggregate result crosses back into
Python rather than every value in the column. The engine protocol has no
column-projected scan, so this still reads the underlying table once; the
narrowing is in what gets materialised in Python, not in bytes read off disk.

Volume is compared per snapshot, not on the table's cumulative row count.
Bronze is append-only, so its total row count only ever grows -- comparing
totals against a trailing median can never fire. `rows_in_latest_snapshot`
exists precisely so an incident shows up as a dip in the batch just landed,
not diluted into a total that has been climbing for years.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

VOLUME_COLLAPSE_RATIO = 0.5  # below half the trailing median is an incident


@dataclass(frozen=True)
class ObservedState:
    table: str
    columns: dict[str, str]
    row_count: int
    rows_in_latest_snapshot: int
    snapshot_count: int
    newest_date: date | None


@dataclass(frozen=True)
class Finding:
    kind: str          # schema_drift | volume_anomaly | staleness
    table: str
    detail: str
    evidence: dict = field(default_factory=dict)


def observe(engine, table_def, date_column: str | None = None) -> ObservedState:
    """Build an `ObservedState` for `table_def`.

    Columns, row count, and snapshot count are metadata-only: `columns`
    comes from `schema_history` (the table's schema definitions), and the
    counts come from `snapshot_details` / `snapshot_row_counts` (manifest
    summaries). `date_column`, if given, names the column to check for
    freshness; the caller (which already has the contract) is responsible
    for supplying it, keeping this function decoupled from contract parsing.
    Its newest value is fetched with a `MAX(...)` pushed through `engine.sql`
    rather than pulled into a Python list -- see the module docstring for why
    that still isn't free, and what it does save.
    """
    ident = table_def.name

    schema_versions = engine.schema_history(ident)
    columns = schema_versions[-1]["columns"] if schema_versions else {}

    details = engine.snapshot_details(ident)
    per_snapshot = engine.snapshot_row_counts(ident)
    row_count = details[-1]["total_records"] if details else 0

    newest = None
    if date_column and date_column in columns:
        # date_column is supplied by trusted caller code (a contract's
        # freshness expectation), never external input.
        result = engine.sql(f"SELECT max({date_column}) AS d FROM t", tables={"t": ident})
        if result.num_rows and result.column("d")[0].as_py() is not None:
            newest = _to_date(result.column("d")[0].as_py())

    return ObservedState(
        table=ident,
        columns=columns,
        row_count=row_count,
        rows_in_latest_snapshot=per_snapshot[-1] if per_snapshot else 0,
        snapshot_count=len(per_snapshot),
        newest_date=newest,
    )


def _to_date(value) -> date:
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value.date() if hasattr(value, "date") else value


def detect(state: ObservedState, contract, as_of: date,
           history: list[int]) -> list[Finding]:
    findings: list[Finding] = []

    declared = {f.name: f.type for f in contract.schema_fields}
    observed = state.columns

    for name, declared_type in declared.items():
        if name not in observed:
            findings.append(Finding(
                "schema_drift", state.table,
                f"column '{name}' declared in contract but absent from table",
                {"change": "dropped", "column": name, "declared_type": declared_type}))
        elif not _types_compatible(declared_type, observed[name]):
            findings.append(Finding(
                "schema_drift", state.table,
                f"column '{name}' type changed: {declared_type} -> {observed[name]}",
                {"change": "type_changed", "column": name,
                 "declared_type": declared_type, "observed_type": observed[name]}))

    for name in observed:
        if name not in declared and not name.startswith("_"):
            findings.append(Finding(
                "schema_drift", state.table,
                f"column '{name}' present in table but not declared in contract",
                {"change": "added", "column": name,
                 "observed_type": observed[name]}))

    # Per-snapshot, not cumulative: an append-only table's total row count
    # only ever grows, so a total-vs-median check can never fire.
    if history:
        median = statistics.median(history)
        latest = state.rows_in_latest_snapshot
        if median > 0 and latest < median * VOLUME_COLLAPSE_RATIO:
            findings.append(Finding(
                "volume_anomaly", state.table,
                f"latest snapshot added {latest:,} rows, below "
                f"{VOLUME_COLLAPSE_RATIO:.0%} of trailing median {median:,.0f}",
                {"rows_in_latest_snapshot": latest, "median": median}))

    freshness = next((e for e in contract.expectations
                      if e.get("type") == "freshness"), None)
    if freshness and state.newest_date is not None:
        lag = (as_of - state.newest_date).days
        if lag > freshness["max_lag_days"]:
            findings.append(Finding(
                "staleness", state.table,
                f"newest data is {lag}d behind AS_OF {as_of} "
                f"(limit {freshness['max_lag_days']}d)",
                {"lag_days": lag, "as_of": as_of.isoformat()}))

    return findings


# `observe()` reads columns from `schema_history`, so in the normal path
# `declared == observed` already covers scalar types -- both sides use
# Iceberg vocabulary. This set is a defensive bridge for `ObservedState`
# built some other way (an Arrow schema, a hand-built test state) where
# `observed` might be in Arrow's vocabulary instead. Kept because that gap
# is exactly how bugs like this one hide: PyIceberg's `StringType()`
# round-trips through Arrow as `large_string`, not `string` -- omitting that
# pair would make every Arrow-sourced string column read as a breaking
# type change.
_COMPATIBLE = {("long", "int64"), ("string", "string"), ("string", "large_string"),
               ("struct", "struct"), ("double", "double"), ("date", "date32[day]"),
               ("timestamptz", "timestamp[us, tz=UTC]"), ("boolean", "bool")}


def _types_compatible(declared: str, observed: str) -> bool:
    if declared == observed:
        return True
    if (declared, observed) in _COMPATIBLE:
        return True
    return observed.startswith(declared)
