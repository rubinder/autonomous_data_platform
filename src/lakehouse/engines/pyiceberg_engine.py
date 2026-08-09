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

    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table:
        table = self.catalog.load_table(ident)
        scan = table.scan() if snapshot_id is None else table.scan(snapshot_id=snapshot_id)
        return scan.to_arrow()

    def snapshots(self, ident: str) -> list[int]:
        """Snapshot ids oldest-first, so `[-1]` is the current snapshot."""
        table = self.catalog.load_table(ident)
        return [s.snapshot_id for s in table.metadata.snapshots]

    def snapshot_row_counts(self, ident: str) -> list[int]:
        """Rows added per snapshot, from metadata summaries. No data scan."""
        table = self.catalog.load_table(ident)
        counts = []
        for snapshot in table.metadata.snapshots:
            summary = getattr(snapshot, "summary", None) or {}
            added = summary.get("added-records") or summary.get("added_records") or 0
            counts.append(int(added))
        return counts

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


def _conform(data: pa.Table, target: pa.Schema) -> pa.Table:
    """Reorder `data`'s columns to the table schema, then cast.

    `pa.Table.cast` is positional: it silently depends on the caller building
    columns in schema order. Selecting by name first makes a column-order
    mistake in a downstream job a loud KeyError instead of a type error or,
    worse, two same-typed columns swapping values.
    """
    missing = [name for name in target.names if name not in data.column_names]
    if missing:
        raise KeyError(f"missing columns for target schema: {missing}")
    return data.select(list(target.names)).cast(target)
