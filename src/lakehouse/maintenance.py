"""Schema evolution, cross-version queries, and Iceberg table maintenance.

Iceberg resolves columns by field ID, not by name or position. A file written
under schema v1 still reads correctly under v5 after columns have been added,
widened, and renamed -- no rewrite, no backfill. The functions here exercise
that property, and the tests check it rather than trusting it.

The drift injectors exist so the ops agent has real table mutations to find.
An agent proven only against hand-written fixtures proves nothing about
whether it reads actual metadata correctly.
"""
from __future__ import annotations

import sys

import pyarrow as pa
from pyiceberg.types import IntegerType, LongType, StringType

from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine

IDENT = schemas.BRONZE_TRANSACTIONS.name

# Commands that reach past `LakehouseEngine` to `engine.catalog` for PyIceberg
# APIs the protocol does not expose (`update_schema`, `table.schemas()`,
# `table.maintenance.expire_snapshots`). Dispatching them through `get_engine()`
# without checking would raise a bare `AttributeError` under `ENGINE=spark`.
#
# Chosen fix, of the two available: **fail fast with a clear message**, rather
# than widening the protocol with three Iceberg-specific methods and
# implementing them twice. These are demo/maintenance commands, not pipeline
# steps -- nothing in `make all`, `make monitor`, `make arrival`, or the agent
# touches them -- so paying a permanent abstraction cost to make a demo
# engine-portable buys less than it costs. The pipeline itself stays fully
# dual-engine.
#
# `timetravel` and `schema-history` are NOT listed: they go through protocol
# methods only and work on either engine.
_CATALOG_COMMANDS = frozenset({"drift-demo", "cross-version", "expire"})


def _require_catalog(engine, command: str) -> None:
    if not hasattr(engine, "catalog"):
        raise SystemExit(
            f"maintenance: `{command}` needs direct PyIceberg catalog access "
            f"(schema evolution, snapshot expiry, per-snapshot schema "
            f"resolution), which {type(engine).__name__} does not expose. "
            f"Re-run with ENGINE=pyiceberg.")


def _fields(engine, ident: str) -> dict[str, object]:
    """Top-level fields of the table's current schema, keyed by name."""
    table = engine.catalog.load_table(ident)
    return {f.name: f for f in table.schema().fields}


def evolve_add_column(engine, name: str, iceberg_type, ident: str = IDENT) -> bool:
    """Additive evolution. Existing files gain the column as NULL, unrewritten."""
    if name in _fields(engine, ident):
        return False
    table = engine.catalog.load_table(ident)
    with table.update_schema() as update:
        update.add_column(name, iceberg_type)
    return True


def evolve_widen_column(engine, name: str, iceberg_type, ident: str = IDENT) -> bool:
    """Type promotion (int -> long). Allowed because every existing value fits.

    Returns False when the column is already that type, so `evolve_all` stays
    genuinely idempotent instead of re-issuing a no-op schema update.
    """
    field = _fields(engine, ident).get(name)
    if field is None or str(field.field_type) == str(iceberg_type):
        return False
    table = engine.catalog.load_table(ident)
    with table.update_schema() as update:
        update.update_column(name, field_type=iceberg_type)
    return True


def evolve_rename_column(engine, old: str, new: str, ident: str = IDENT) -> bool:
    """Rename. The field ID is unchanged, so old files need no rewrite."""
    fields = _fields(engine, ident)
    if old not in fields or new in fields:
        return False
    table = engine.catalog.load_table(ident)
    with table.update_schema() as update:
        update.rename_column(old, new)
    return True


def evolve_all(engine, ident: str = IDENT) -> list[str]:
    """The v2 -> v5 sequence. Idempotent: re-running applies nothing."""
    applied = []
    if evolve_add_column(engine, "merchantCategoryCode", StringType(), ident):
        applied.append("v2: +merchantCategoryCode (string)")
    if evolve_add_column(engine, "settlementDays", IntegerType(), ident):
        applied.append("v3: +settlementDays (int)")
    if evolve_widen_column(engine, "settlementDays", LongType(), ident):
        applied.append("v4: settlementDays int -> long")
    if evolve_rename_column(engine, "checkNumber", "check_reference", ident):
        applied.append("v5: checkNumber -> check_reference")
    return applied


def schema_versions(engine, ident: str = IDENT) -> list[dict]:
    return engine.schema_history(ident)


def query_across_versions(engine, ident: str = IDENT) -> pa.Table:
    """Union every snapshot's contents, reconciled to the current schema.

    Reconciliation is **by field ID**, which is the whole point. PyIceberg
    scans a historical snapshot using *that snapshot's* schema, so a snapshot
    older than the `checkNumber -> check_reference` rename hands back a column
    called `checkNumber`. Matching on names would file those values under a
    column that no longer exists and report `check_reference` as entirely
    NULL -- a cross-version query that quietly loses a column is worse than
    one that fails. Field IDs survive the rename, so the values come back
    under the new name with no file rewritten.

    Columns genuinely absent from an older snapshot -- ones added after it --
    are filled with typed NULLs rather than dropped, so a query can span a
    schema change without the caller knowing one happened. Each row is tagged
    with the snapshot and schema it came from.
    """
    table = engine.catalog.load_table(ident)
    current = table.schema()
    target = current.as_arrow()
    schemas_by_id = table.schemas()
    frames: list[pa.Table] = []

    for snapshot in table.metadata.snapshots:
        rows = engine.scan_arrow(ident, snapshot_id=snapshot.snapshot_id)
        source = schemas_by_id.get(snapshot.schema_id, current)
        aligned = _align_by_field_id(rows, current, source, target)
        n = aligned.num_rows
        aligned = aligned.append_column(
            "_snapshot_id", pa.array([snapshot.snapshot_id] * n, pa.int64()))
        aligned = aligned.append_column(
            "_schema_id", pa.array([snapshot.schema_id] * n, pa.int32()))
        frames.append(aligned)

    return pa.concat_tables(frames) if frames else target.empty_table()


def _align_by_field_id(rows: pa.Table, current, source, target: pa.Schema) -> pa.Table:
    """Project `rows` onto the current schema, matching columns by field ID.

    `source` is the schema the rows were actually read under; a column whose
    ID is missing from it postdates this snapshot and becomes typed NULLs.
    """
    names_by_id = {f.field_id: f.name for f in source.fields}
    arrays, names = [], []
    for field in current.fields:
        was_called = names_by_id.get(field.field_id)
        arrow_type = target.field(field.name).type
        if was_called is not None and was_called in rows.column_names:
            arrays.append(rows.column(was_called).cast(arrow_type))
        else:
            arrays.append(pa.nulls(rows.num_rows, arrow_type))
        names.append(field.name)
    return pa.Table.from_arrays(arrays, names=names)


def incremental_rows(engine, ident: str, from_snapshot: int,
                     to_snapshot: int) -> int:
    """Rows added between two snapshots, from metadata summaries only.

    `from_snapshot` is exclusive and `to_snapshot` inclusive -- the delta a
    consumer that has already read up to `from_snapshot` still owes itself.
    Identical ids therefore mean zero: nothing has happened since.

    Every other bad input raises instead of returning a number. An unknown id
    or a reversed range answered with a bare `0` is the worst possible output
    for a monitoring helper, because `0` is indistinguishable from the real
    and alarming answer "no rows arrived" -- a caller watching for a stalled
    feed would read a typo'd snapshot id as an outage, or a stalled feed as
    fine, with nothing to tell the two apart.
    """
    details = engine.snapshot_details(ident)
    positions = {d["snapshot_id"]: i for i, d in enumerate(details)}

    for label, snapshot_id in (("from_snapshot", from_snapshot),
                               ("to_snapshot", to_snapshot)):
        if snapshot_id not in positions:
            raise ValueError(
                f"{label}={snapshot_id} is not a snapshot of {ident}")

    start, end = positions[from_snapshot], positions[to_snapshot]
    if start == end:
        return 0
    if start > end:
        raise ValueError(
            f"from_snapshot={from_snapshot} is newer than "
            f"to_snapshot={to_snapshot} on {ident}; the range is reversed")

    return sum(d["added_records"] for d in details[start + 1:end + 1])


def time_travel_demo(engine, ident: str = IDENT) -> dict[str, int]:
    snaps = engine.snapshots(ident)
    if len(snaps) < 2:
        bronze.ingest_all(engine, n_transactions=500)
        snaps = engine.snapshots(ident)
    return {
        "snapshot_count": len(snaps),
        "rows_at_first_snapshot": engine.scan_arrow(
            ident, snapshot_id=snaps[0]).num_rows,
        "rows_at_latest_snapshot": engine.scan_arrow(ident).num_rows,
    }


def inject_volume_collapse(engine, ident: str = IDENT, rows: int = 5) -> int:
    """Append a near-empty batch so the per-snapshot volume check fires.

    An APPEND, not an overwrite: Bronze stays append-only (Global
    Constraints), and a collapsed batch is what a real upstream outage
    actually looks like -- a handful of rows landing where thousands should,
    not a table that truncates itself. The re-appended rows duplicate ids,
    which is fine here: Bronze does no dedupe by design and Silver collapses
    duplicates downstream.

    The slice comes from the live table rather than the authoring schema, so
    this still works after `evolve_all` has changed the column set.
    """
    batch = engine.scan_arrow(ident).slice(0, rows)
    engine.append(ident, batch)
    return batch.num_rows


def expire(engine, retain_last: int = 5) -> int:
    """Drop all but the newest `retain_last` snapshots of every table.

    PyIceberg 0.11.1 has no `retain_last` builder -- the API is
    `table.maintenance.expire_snapshots().by_ids([...]).commit()` -- so the
    ids to drop are computed here. Branch and tag heads are protected by
    PyIceberg and silently skipped, so the count returned is measured after
    the fact rather than assumed.
    """
    removed = 0
    for table_def in schemas.ALL_TABLES:
        if not engine.table_exists(table_def.name):
            continue
        table = engine.catalog.load_table(table_def.name)
        snapshot_ids = [s.snapshot_id for s in table.metadata.snapshots]
        stale = snapshot_ids[:-retain_last] if retain_last else snapshot_ids
        if not stale:
            continue
        table.maintenance.expire_snapshots().by_ids(stale).commit()
        removed += len(snapshot_ids) - len(engine.snapshots(table_def.name))
    return removed


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "timetravel"
    engine = get_engine()
    if command in _CATALOG_COMMANDS:
        _require_catalog(engine, command)

    if command == "timetravel":
        for key, value in time_travel_demo(engine).items():
            print(f"timetravel: {key} = {value:,}")

    elif command == "drift-demo":
        applied = evolve_all(engine)
        for step in applied:
            print(f"drift-demo: applied {step}")
        if not applied:
            print("drift-demo: schema already evolved, nothing to apply")
        added = inject_volume_collapse(engine)
        print(f"drift-demo: appended a collapsed batch of {added} rows")
        print("drift-demo: run `make agent` to see these detected")

    elif command == "schema-history":
        for version in schema_versions(engine):
            print(f"schema {version['schema_id']}: "
                  f"{len(version['columns'])} columns")
            for name, dtype in version["columns"].items():
                print(f"    [{version['field_ids'][name]:>3}] {name}: {dtype}")

    elif command == "cross-version":
        combined = query_across_versions(engine)
        snapshot_ids = combined.column("_snapshot_id").to_pylist()
        schema_ids = combined.column("_schema_id").to_pylist()
        print(f"cross-version: {combined.num_rows:,} rows across "
              f"{len(set(snapshot_ids))} snapshot(s)")
        for schema_id in sorted(set(schema_ids)):
            n = sum(1 for v in schema_ids if v == schema_id)
            print(f"    schema {schema_id}: {n:,} rows")

    elif command == "expire":
        print(f"maintenance: expired {expire(engine)} snapshot(s)")

    else:
        print(f"unknown command: {command}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
