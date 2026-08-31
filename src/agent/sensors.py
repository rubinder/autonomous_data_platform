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
nonzero count into its own `Finding` rather than silently dropping it. That
pattern lives in `src.dates.max_parseable_date` as a single shared
implementation -- `bronze.max_txn_date`, which resolves AS_OF_DATE before the
agent graph starts, uses the same one, because a second hand-rolled copy of
this logic is exactly how the crash came back on a path the sensor could not
protect.

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

from src import dates

VOLUME_COLLAPSE_RATIO = 0.5  # below half the trailing median is an incident

# Above this many distinct values a column is not an enum, and an enum watch
# pointed at it is a misconfiguration. Reporting that is better than either
# staying silent or emitting a finding carrying thousands of values.
ENUM_CARDINALITY_LIMIT = 200


@dataclass(frozen=True)
class ObservedState:
    table: str
    columns: dict[str, str]
    row_count: int
    rows_in_latest_snapshot: int
    snapshot_count: int
    newest_date: date | None
    unparseable_date_count: int = 0
    # Field IDs are the only evidence that distinguishes a rename from a drop
    # plus an unrelated add. Iceberg keeps the ID stable across a rename --
    # that is precisely why no data file is rewritten -- so an agent that
    # reads names only cannot see the difference the storage layer is built
    # around. `renames` maps every historical name to the column's CURRENT
    # name, so a contract stuck at any past version resolves in one hop.
    field_ids: dict[str, int] = field(default_factory=dict)
    renames: dict[str, tuple[str, int]] = field(default_factory=dict)
    enum_values: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Finding:
    kind: str          # schema_drift | volume_anomaly | staleness | data_corruption
    table: str
    detail: str
    evidence: dict = field(default_factory=dict)


def build_rename_map(schema_versions: list[dict]) -> dict[str, tuple[str, int]]:
    """Every past name of a still-present column -> (current name, field id).

    Built by grouping names by field ID across the table's whole schema
    history. Transitive renames (a -> b -> c) fall out for free: `a` and `b`
    share field ID with `c`, so both map to `c` rather than to an
    intermediate name that no longer exists either.

    A field ID absent from the latest schema is a genuine drop and is
    deliberately not in the map -- `detect()` must still call that breaking.
    """
    if not schema_versions:
        return {}

    current_by_id = {fid: name
                     for name, fid in schema_versions[-1].get("field_ids", {}).items()}

    names_by_id: dict[int, set[str]] = {}
    for version in schema_versions:
        for name, fid in version.get("field_ids", {}).items():
            names_by_id.setdefault(fid, set()).add(name)

    return {old: (current_by_id[fid], fid)
            for fid, names in names_by_id.items() if fid in current_by_id
            for old in names if old != current_by_id[fid]}


def observe(engine, table_def, date_column: str | None = None,
            enum_columns: tuple[str, ...] = ()) -> ObservedState:
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
    latest = schema_versions[-1] if schema_versions else {}
    columns = latest.get("columns", {})
    field_ids = latest.get("field_ids", {})
    renames = build_rename_map(schema_versions)

    details = engine.snapshot_details(ident)
    per_snapshot = engine.snapshot_row_counts(ident)
    row_count = details[-1]["total_records"] if details else 0

    # One scan serves both the freshness read and the enum reads. Two would
    # double the cost of the sensor the module docstring already apologises
    # for; the scan happens only when something actually needs values.
    wants_date = bool(date_column and date_column in columns)
    arrow = engine.scan_arrow(ident) if wants_date or enum_columns else None

    newest = None
    unparseable = 0
    if arrow is not None and wants_date and date_column in arrow.column_names:
        newest, unparseable = dates.max_parseable_date(
            arrow.column(date_column).to_pylist())

    enum_values: dict[str, list[str]] = {}
    if arrow is not None:
        for column in enum_columns:
            if column not in arrow.column_names:
                # Left out of the dict on purpose. `detect()` turns a watched
                # column that produced no values into its own finding -- a
                # watch on a column that isn't there must not read as "no
                # drift", which is the empty-delta blind spot in miniature.
                continue
            distinct = sorted({v for v in arrow.column(column).to_pylist()
                               if v is not None})
            # Truncated to one past the limit: enough for `detect()` to see
            # the limit was exceeded, bounded enough that a watch mistakenly
            # aimed at a merchant column cannot carry the whole cardinality
            # into a finding, an incident file, and a report.
            enum_values[column] = distinct[:ENUM_CARDINALITY_LIMIT + 1]

    return ObservedState(
        table=ident,
        columns=columns,
        row_count=row_count,
        rows_in_latest_snapshot=per_snapshot[-1] if per_snapshot else 0,
        snapshot_count=len(per_snapshot),
        newest_date=newest,
        unparseable_date_count=unparseable,
        field_ids=field_ids,
        renames=renames,
        enum_values=enum_values,
    )


def detect(state: ObservedState, contract, as_of: date,
           history: list[int]) -> list[Finding]:
    findings: list[Finding] = []

    declared = {f.name: f.type for f in contract.schema_fields}
    observed = state.columns

    # New names reached by a rename. Reported once, from the old name's side,
    # and suppressed on the added-columns pass below -- otherwise the single
    # event "this column was renamed" surfaces as two findings that look
    # unrelated: a breaking drop and a benign add.
    renamed_to: set[str] = set()

    for name, declared_type in declared.items():
        if name not in observed:
            target = state.renames.get(name)
            if target and target[0] in observed:
                new_name, field_id = target
                observed_type = observed[new_name]
                compatible = _types_compatible(declared_type, observed_type)
                renamed_to.add(new_name)
                findings.append(Finding(
                    "schema_drift", state.table,
                    f"column '{name}' was renamed to '{new_name}' (field id "
                    f"{field_id}); the contract still declares the old name",
                    {"change": "renamed", "column": name,
                     "renamed_to": new_name, "field_id": field_id,
                     "declared_type": declared_type,
                     "observed_type": observed_type,
                     "type_compatible": compatible}))
                continue
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
        if name not in declared and not name.startswith("_") and name not in renamed_to:
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

    findings += _detect_enum_drift(state, contract)

    return findings


def _detect_enum_drift(state: ObservedState, contract) -> list[Finding]:
    """New values in a watched categorical column.

    Deliberately separate from the contract's `accepted_values` expectation.
    That one is a fail-closed gate: an unregistered value raises
    `ContractViolation` and stops the build, which is right for a column the
    pipeline's logic branches on (`baseType`). Most categorical columns are
    not like that -- upstream adds a category and nothing downstream breaks,
    but somebody needs to know. Blocking on those trains people to widen the
    allowed list without looking; staying silent means the first anyone hears
    of a new category is a mart that quietly under-counts.

    So this reports and records, and never blocks.
    """
    findings: list[Finding] = []

    for watch in getattr(contract, "enum_watch", ()):
        column = watch["column"]
        known = set(watch.get("known_values", ()))
        observed = state.enum_values.get(column)

        if observed is None:
            # A watch that measured nothing is not a watch that found nothing.
            findings.append(Finding(
                "enum_drift", state.table,
                f"enum watch declared on '{column}', but no values were read "
                f"from that column -- it is absent from the table or was not "
                f"collected, so this column is NOT being monitored",
                {"column": column, "error": "column_absent"}))
            continue

        new_values = sorted(set(observed) - known)

        if len(observed) > ENUM_CARDINALITY_LIMIT:
            findings.append(Finding(
                "enum_drift", state.table,
                f"'{column}' holds more than {ENUM_CARDINALITY_LIMIT} distinct "
                f"values, so it is not an enum; the watch is misconfigured and "
                f"its drift result is meaningless",
                {"column": column, "error": "cardinality_exceeded",
                 "new_values": new_values[:ENUM_CARDINALITY_LIMIT],
                 "distinct_at_least": len(observed)}))
            continue

        if new_values:
            findings.append(Finding(
                "enum_drift", state.table,
                f"{len(new_values)} new value(s) in '{column}': "
                f"{', '.join(new_values)}",
                {"column": column, "new_values": new_values,
                 "known_count": len(known), "distinct_observed": len(observed)}))

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
