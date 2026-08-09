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
newest *value* -- that requires reading data, and the engine protocol has no
column-projected scan, so `observe()` genuinely reads the whole table via
`scan_arrow` to get it. There is no metadata-cheap path available today
through this engine interface; a column-projected read added to
`LakehouseEngine` would be the real fix, and is out of scope here.

Because Bronze does no type coercion, that column can hold values that
don't parse as dates. Computing "newest" with SQL `MAX()` over the raw
strings is actively dangerous: a garbage value that happens to sort
lexicographically above every real date (`"zzz-not-a-date"` sorts after any
`"20XX-..."` string) becomes the `MAX()` result, and parsing it crashes the
agent -- the worst failure mode here, since a monitor that crashes on bad
data stops monitoring at exactly the moment something has gone wrong.
`observe()` instead parses every value, keeps the max of only the ones that
parse, and counts the rest as `unparseable_date_count`; `detect()` turns a
nonzero count into its own `Finding` rather than silently dropping it.

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
    unparseable_date_count: int = 0


@dataclass(frozen=True)
class Finding:
    kind: str          # schema_drift | volume_anomaly | staleness | data_corruption
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

    Reading `date_column`'s newest value requires an actual scan -- see the
    module docstring for why, and why it deliberately does *not* ask SQL for
    `MAX(date_column)` over the raw values: Bronze does no type coercion, so
    a non-null but unparseable value is expected input, not an edge case,
    and letting the database sort raw strings risks handing back garbage
    that then crashes on parse. Every value is parsed defensively instead;
    `newest_date` is the max of the ones that parse, and the rest are
    counted in `unparseable_date_count` for `detect()` to raise on.
    """
    ident = table_def.name

    schema_versions = engine.schema_history(ident)
    columns = schema_versions[-1]["columns"] if schema_versions else {}

    details = engine.snapshot_details(ident)
    per_snapshot = engine.snapshot_row_counts(ident)
    row_count = details[-1]["total_records"] if details else 0

    newest = None
    unparseable = 0
    if date_column and date_column in columns:
        arrow = engine.scan_arrow(ident)
        if date_column in arrow.column_names:
            parsed: list[date] = []
            for value in arrow.column(date_column).to_pylist():
                if value is None:
                    continue
                try:
                    parsed.append(_to_date(value))
                except (ValueError, TypeError):
                    unparseable += 1
            if parsed:
                newest = max(parsed)

    return ObservedState(
        table=ident,
        columns=columns,
        row_count=row_count,
        rows_in_latest_snapshot=per_snapshot[-1] if per_snapshot else 0,
        snapshot_count=len(per_snapshot),
        newest_date=newest,
        unparseable_date_count=unparseable,
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

    # Independent of whether a freshness expectation is declared: a date
    # column that won't parse breaks every downstream time-based join, not
    # just this table's own staleness check.
    if state.unparseable_date_count > 0:
        column = freshness["column"] if freshness else None
        note = f" in '{column}'" if column else ""
        findings.append(Finding(
            "data_corruption", state.table,
            f"{state.unparseable_date_count} value(s){note} did not parse as a date",
            {"column": column, "unparseable_count": state.unparseable_date_count}))

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
