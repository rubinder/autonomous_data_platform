"""Read Iceberg metadata and diff it against declared contracts.

Row counts and schemas come from table metadata rather than full scans where
the engine allows it: the agent is meant to run cheaply and often, and a
sensor that scans the whole lake on every run is one that gets switched off.
`snapshot_row_counts` / `snapshot_details` read manifest summaries only.

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


def observe(engine, table_def, contract) -> ObservedState:
    """Build an `ObservedState` for `table_def`, checked against `contract`.

    Row count and snapshot count come entirely from manifest metadata
    (`snapshot_details`, `snapshot_row_counts`) -- no data scan. Column types
    and the freshness column's newest value are read from `scan_arrow`
    because the engine exposes no metadata-only accessor for either; that
    scan also means observed types are Arrow's names (`large_string`,
    `int64`, ...), not Iceberg's (`string`, `long`, ...), which is why
    `_types_compatible` below has to bridge the two vocabularies.
    """
    ident = table_def.name
    arrow = engine.scan_arrow(ident)
    columns = {f.name: str(f.type) for f in arrow.schema}

    details = engine.snapshot_details(ident)
    per_snapshot = engine.snapshot_row_counts(ident)
    row_count = details[-1]["total_records"] if details else arrow.num_rows

    newest = None
    date_column = _freshness_column(contract)
    if date_column and date_column in arrow.column_names:
        values = [v for v in arrow.column(date_column).to_pylist() if v is not None]
        if values:
            newest = _to_date(max(values))

    return ObservedState(
        table=ident,
        columns=columns,
        row_count=row_count,
        rows_in_latest_snapshot=per_snapshot[-1] if per_snapshot else 0,
        snapshot_count=len(per_snapshot),
        newest_date=newest,
    )


def _freshness_column(contract) -> str | None:
    exp = next((e for e in contract.expectations if e.get("type") == "freshness"), None)
    return exp.get("column") if exp else None


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


# Declared-contract vocabulary (Iceberg: "string", "long", ...) paired with
# what an Arrow scan actually reports for it. PyIceberg's `StringType()`
# round-trips through Arrow as `large_string`, not `string` -- omitting that
# pair would make every string column read as a breaking type change.
_COMPATIBLE = {("long", "int64"), ("string", "string"), ("string", "large_string"),
               ("struct", "struct"), ("double", "double"), ("date", "date32[day]"),
               ("timestamptz", "timestamp[us, tz=UTC]"), ("boolean", "bool")}


def _types_compatible(declared: str, observed: str) -> bool:
    if declared == observed:
        return True
    if (declared, observed) in _COMPATIBLE:
        return True
    return observed.startswith(declared)
