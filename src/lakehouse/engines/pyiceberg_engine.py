"""Default engine: PyIceberg owns catalog and commits, DuckDB does the SQL."""
from __future__ import annotations

import duckdb
import pyarrow as pa

from src.lakehouse.schemas import TableDef


class PyIcebergEngine:
    def __init__(self, catalog):
        self.catalog = catalog

    def create_table(self, table_def: TableDef) -> None:
        if self.table_exists(table_def.name):
            return
        self.catalog.create_table(
            identifier=table_def.name,
            schema=table_def.schema,
            partition_spec=table_def.spec,
        )

    def table_exists(self, ident: str) -> bool:
        return bool(self.catalog.table_exists(ident))

    def append(self, ident: str, data: pa.Table) -> None:
        table = self.catalog.load_table(ident)
        table.append(_conform(data, table.schema().as_arrow()))

    def overwrite(self, ident: str, data: pa.Table) -> None:
        table = self.catalog.load_table(ident)
        table.overwrite(_conform(data, table.schema().as_arrow()))

    def arrow_schema(self, ident: str) -> pa.Schema:
        """The table's *live* Arrow schema. Metadata only, no data scan.

        Writers must build batches against this rather than against the
        authoring `TableDef`: once a table has been evolved the two differ,
        and `append` rejects any column-set mismatch (deliberately -- see
        `_conform`). Reading it from a scan would work but costs a full read.
        """
        return self.catalog.load_table(ident).schema().as_arrow()

    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table:
        table = self.catalog.load_table(ident)
        scan = table.scan() if snapshot_id is None else table.scan(snapshot_id=snapshot_id)
        return scan.to_arrow()

    def snapshots(self, ident: str) -> list[int]:
        """Snapshot ids oldest-first, so `[-1]` is the current snapshot."""
        table = self.catalog.load_table(ident)
        return [s.snapshot_id for s in table.metadata.snapshots]

    def snapshot_row_counts(self, ident: str) -> list[int]:
        """Rows added per snapshot, from metadata summaries. No data scan.

        Counts `added-records` for *every* operation, including `overwrite` and
        `replace`. A full Silver/Gold rebuild therefore reports its whole row
        count as "added", which is not comparable with an incremental append.
        Volume monitors that need that distinction should use
        `snapshot_details` and filter on `operation`.
        """
        return [d["added_records"] for d in self.snapshot_details(ident)]

    def snapshot_details(self, ident: str) -> list[dict]:
        """Per-snapshot metadata, oldest-first. Metadata only, no data scan.

        Each entry: `snapshot_id`, `parent_snapshot_id`, `operation`,
        `added_records`, `deleted_records`, `total_records`, `is_full_rebuild`.

        `is_full_rebuild` exists because `operation` alone is not enough.
        Measured on PyIceberg 0.11.1: `Table.overwrite()` commits **two**
        snapshots, a `delete` that clears the table followed by a plain
        `append` -- there is no snapshot whose operation reads "overwrite".
        So a Gold rebuild's append is, by operation, identical to an
        incremental append, and a volume monitor comparing `added_records`
        against a trailing median would read every rebuild as a spike.

        Filter on `is_full_rebuild` to compare like with like: it is True for
        both halves of an overwrite and False for incremental appends.
        """
        table = self.catalog.load_table(ident)
        raw = []
        for snapshot in table.metadata.snapshots:
            summary = getattr(snapshot, "summary", None) or {}
            operation = getattr(summary, "operation", None)
            raw.append({
                "snapshot_id": snapshot.snapshot_id,
                "parent_snapshot_id": snapshot.parent_snapshot_id,
                "operation": getattr(operation, "value", operation),
                "added_records": _summary_int(summary, "added-records"),
                "deleted_records": _summary_int(summary, "deleted-records"),
                "total_records": _summary_int(summary, "total-records"),
            })
        by_id = {d["snapshot_id"]: d for d in raw}
        for detail in raw:
            detail["is_full_rebuild"] = _is_full_rebuild(detail, by_id)
        return raw

    def schema_history(self, ident: str) -> list[dict]:
        """One entry per schema version this table has had."""
        table = self.catalog.load_table(ident)
        return [
            {"schema_id": s.schema_id,
             "columns": {f.name: str(f.field_type) for f in s.fields},
             "field_ids": {f.name: f.field_id for f in s.fields}}
            for s in table.metadata.schemas
        ]

    def sql(self, query: str, tables: dict[str, str]) -> pa.Table:
        con = duckdb.connect()
        try:
            for alias, ident in tables.items():
                con.register(alias, self.scan_arrow(ident))
            # `.arrow()` returns a RecordBatchReader on duckdb >= 1.5 and
            # `fetch_arrow_table()` is deprecated; the protocol promises a
            # materialised pa.Table.
            return con.execute(query).to_arrow_table()
        finally:
            con.close()


def _clears_table(detail: dict | None) -> bool:
    """A delete that removed rows and left the table empty."""
    return bool(
        detail
        and detail["operation"] == "delete"
        and detail["deleted_records"] > 0
        and detail["total_records"] == 0
    )


def _is_full_rebuild(detail: dict, by_id: dict) -> bool:
    """Both halves of the delete+append pair that `Table.overwrite()` commits."""
    if detail["operation"] in ("overwrite", "replace"):
        return True
    if _clears_table(detail):
        return True
    if detail["operation"] == "append":
        return _clears_table(by_id.get(detail["parent_snapshot_id"]))
    return False


def _summary_int(summary, key: str) -> int:
    """Read a numeric key from a snapshot summary. Iceberg stores them as str."""
    raw = summary.get(key) or summary.get(key.replace("-", "_")) or 0
    return int(raw)


def _conform(data: pa.Table, target: pa.Schema) -> pa.Table:
    """Match `data`'s columns to the table schema by name, then cast.

    `pa.Table.cast` compares names positionally and rejects any mismatch --
    `ValueError: Target schema's field names are not matching the table's
    field names` -- so it refuses a correct-but-differently-ordered table.
    This is purely a loosening of that: callers may build columns in any
    order and we reorder by name.

    The loosening is deliberately one-way. Column *sets* still have to match
    exactly, because `select()` would otherwise drop an unexpected column
    silently -- a stale or typo'd name in a downstream writer would vanish
    instead of failing. Missing and extra columns both raise.
    """
    missing = [name for name in target.names if name not in data.column_names]
    extra = [name for name in data.column_names if name not in target.names]
    if missing or extra:
        raise KeyError(
            f"column set does not match target schema: missing={missing}, extra={extra}"
        )
    return data.select(list(target.names)).cast(target)
