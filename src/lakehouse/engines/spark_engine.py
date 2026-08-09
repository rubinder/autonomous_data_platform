"""Opt-in Spark engine: same protocol, plus MERGE INTO and compaction.

Exists to prove the LakehouseEngine abstraction is real. An interface with
one implementation is a guess about what would be portable; the parity test
(`tests/test_engine_parity.py`) turns it into a claim that is checked --
running the same Silver build through both engines and diffing the output.

Requires the `spark` extra (`uv sync --extra spark`) and a local JVM
(Spark 3.5 needs Java 8/11/17; it will not start on newer JDKs). Importing
this module never happens on the default `ENGINE=pyiceberg` path --
`engines/__init__.get_engine` only imports it when `ENGINE=spark` is chosen,
and the parity test guards its own import with `pytest.importorskip`.

Three environment quirks shaped this file, each documented at its use site:

1. Py4J converts Iceberg TIMESTAMP values to Python `datetime.datetime`
   using the *JVM's default timezone*, not `spark.sql.session.timeZone`
   (that config only affects SQL-level parsing/formatting). The JVM
   timezone is fixed at process start, so `TZ` has to be set before any
   SparkSession exists -- see the module-level block below.
2. `DataFrame.toPandas()` -- and therefore `pa.Table.from_pandas` -- goes
   through `pyspark.sql.pandas.conversion`, which imports
   `distutils.version.LooseVersion`. `distutils` was removed in Python
   3.12, so `toPandas()` raises `ModuleNotFoundError` on this interpreter
   (3.13). Every Arrow conversion here instead goes through
   `DataFrame.collect()` + `Row.asDict(recursive=True)`, which never
   touches that code path.
3. PyIceberg's `StringType()` maps to Arrow `large_string`; Spark's
   `StringType` maps to plain `string`. Both directions are accepted on
   the way in (`_arrow_type_to_spark`), and the parity test normalises the
   difference by comparing `.to_pandas()` frames (where both collapse to
   the same pandas `object` dtype) rather than raw Arrow schemas -- it
   does not silently coerce one type into the other.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pyarrow as pa

from src import config
from src.lakehouse.engines.pyiceberg_engine import _conform, _is_full_rebuild, _summary_int
from src.lakehouse.schemas import TableDef

ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2"

# See docstring point 1. Must run before any SparkSession (and therefore
# before the JVM) is created. `setdefault` respects an operator's explicit
# TZ rather than clobbering it; the default keeps runs reproducible
# regardless of the host machine's local timezone.
os.environ.setdefault("TZ", "UTC")
if hasattr(time, "tzset"):
    time.tzset()

# DuckDB accepts `TIMESTAMP WITH TIME ZONE` as a CAST target; Spark SQL's
# parser rejects it outright (`PARSE_SYNTAX_ERROR`). Both engines' TIMESTAMP
# type is an instant (UTC-based) under the hood, so this is a syntax
# translation, not a semantic one -- the SQL strings in `silver.py` are
# authored once and run, unmodified, through whichever engine's `sql()` is
# called; this is the one spelling Spark cannot parse.
_TSTZ = re.compile(r"TIMESTAMP\s+WITH\s+TIME\s+ZONE", re.IGNORECASE)


class SparkEngine:
    def __init__(self, warehouse: Path | None = None):
        from pyspark.sql import SparkSession

        wh = Path(warehouse or config.WAREHOUSE_PATH / "spark")
        wh.mkdir(parents=True, exist_ok=True)
        self.spark = (
            SparkSession.builder.appName("autonomous-data-platform")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.jars.packages", ICEBERG_RUNTIME)
            .config("spark.sql.extensions",
                    "org.apache.iceberg.spark.extensions."
                    "IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.cat", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.cat.type", "hadoop")
            .config("spark.sql.catalog.cat.warehouse", str(wh))
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        # NOTE: SparkSession.getOrCreate() reuses a JVM-wide singleton. A
        # second `SparkEngine(...)` built with a different `warehouse` in
        # the same process gets the FIRST session's catalog config, not its
        # own -- Spark ignores config changes on an already-running
        # session. Tests must use one SparkEngine per process (the parity
        # test's `spark_engine` fixture is module-scoped for this reason).
        for namespace in ("bronze", "silver", "gold"):
            self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS cat.{namespace}")

    def _q(self, ident: str) -> str:
        return f"cat.{ident}"

    def _jtable(self, ident: str):
        """The underlying `org.apache.iceberg.Table` Java object.

        Spark SQL's Iceberg metadata tables (`snapshots`, `history`, ...)
        don't expose per-schema-version detail, and collecting them also
        means running everything through `.collect()`'s Row conversion.
        Reaching through to the same Java `Table` object PyIceberg wraps
        gives direct, typed access to snapshots and schema history without
        another SQL dialect to worry about.
        """
        return self.spark._jvm.org.apache.iceberg.spark.Spark3Util.loadIcebergTable(
            self.spark._jsparkSession, self._q(ident))

    # -- schema / lifecycle ------------------------------------------------

    def create_table(self, table_def: TableDef) -> None:
        if self.table_exists(table_def.name):
            return
        # DDL, not `writeTo(...).create()` on an empty DataFrame: the latter
        # was tried first and commits a genuine (if empty) "append" snapshot
        # at creation time, which PyIceberg's `catalog.create_table` does
        # not -- its `metadata.snapshots` starts empty. That mismatch would
        # throw off `snapshots`/`snapshot_details` by one entry on every
        # table relative to PyIcebergEngine, so DDL it is.
        spark_schema = _arrow_schema_to_spark(table_def.schema.as_arrow())
        columns_ddl = ", ".join(
            f"`{f.name}` {f.dataType.simpleString()}" for f in spark_schema.fields)
        partition_exprs = []
        for field in table_def.spec.fields:
            source = table_def.schema.find_field(field.source_id).name
            transform = str(field.transform)
            if transform in ("day", "month", "year", "hour"):
                partition_exprs.append(f"{transform}s(`{source}`)")
            else:  # identity (and anything else) is a plain column
                partition_exprs.append(f"`{source}`")
        partition_clause = (
            f" PARTITIONED BY ({', '.join(partition_exprs)})" if partition_exprs else "")
        self.spark.sql(
            f"CREATE TABLE {self._q(table_def.name)} ({columns_ddl}) "
            f"USING iceberg{partition_clause}")

    def table_exists(self, ident: str) -> bool:
        return self.spark.catalog.tableExists(self._q(ident))

    def arrow_schema(self, ident: str) -> pa.Schema:
        """The table's *live* Arrow schema, mirroring PyIcebergEngine's.

        Built from Spark's own catalog schema (`spark.table(...).schema`)
        rather than the authoring `TableDef`, so it follows schema
        evolution the same way PyIcebergEngine's does.
        """
        return _spark_schema_to_arrow(self.spark.table(self._q(ident)).schema)

    def schema_history(self, ident: str) -> list[dict]:
        """One entry per schema version this table has had, oldest-first."""
        jtable = self._jtable(ident)
        versions = sorted(jtable.schemas().keySet())
        history = []
        for schema_id in versions:
            jschema = jtable.schemas().get(schema_id)
            columns, field_ids = {}, {}
            for jfield in jschema.columns():
                columns[jfield.name()] = str(jfield.type().toString())
                field_ids[jfield.name()] = jfield.fieldId()
            history.append({
                "schema_id": schema_id, "columns": columns, "field_ids": field_ids,
            })
        return history

    # -- writes --------------------------------------------------------

    def append(self, ident: str, data: pa.Table) -> None:
        target = self.arrow_schema(ident)
        conformed = _conform(data, target)
        self._to_spark(conformed, target).writeTo(self._q(ident)).append()

    def overwrite(self, ident: str, data: pa.Table) -> None:
        from pyspark.sql.functions import lit

        target = self.arrow_schema(ident)
        conformed = _conform(data, target)
        # `.overwrite(lit(True))` is a row-level overwrite matching every
        # existing row -- the Spark analogue of PyIceberg's `Table.overwrite()`
        # default filter (ALWAYS_TRUE). Deliberately not
        # `.overwritePartitions()`, which is a *dynamic* partition overwrite:
        # it only replaces partitions present in the incoming data and would
        # leave stale partitions behind on a genuine full rebuild.
        #
        # The two engines diverge in how many snapshots this costs: PyIceberg's
        # `Table.overwrite()` commits a `delete` then an `append` (two
        # snapshots, neither with operation "overwrite" -- see
        # `pyiceberg_engine._is_full_rebuild`); Spark commits ONE snapshot
        # whose operation genuinely is "overwrite". `snapshot_details` below
        # normalises both into the same `is_full_rebuild` flag rather than
        # trying to make the snapshot counts match.
        self._to_spark(conformed, target).writeTo(self._q(ident)).overwrite(lit(True))

    def merge_upsert(self, ident: str, data: pa.Table, key: str) -> None:
        """Not on `LakehouseEngine` -- Spark-only, via Iceberg's MERGE INTO.

        Conforms `data` the same way `append`/`overwrite` do, so a
        typo'd or stale column fails loudly here too rather than being
        silently dropped by `UPDATE SET *` / `INSERT *`.
        """
        target = self.arrow_schema(ident)
        conformed = _conform(data, target)
        self._to_spark(conformed, target).createOrReplaceTempView("_updates")
        self.spark.sql(f"""
            MERGE INTO {self._q(ident)} t
            USING _updates s ON t.{key} = s.{key}
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
        self.spark.catalog.dropTempView("_updates")

    def compact(self, ident: str) -> None:
        """Not on `LakehouseEngine` -- Spark-only; PyIceberg has no built-in
        equivalent to Iceberg's `rewrite_data_files` stored procedure."""
        self.spark.sql(f"CALL cat.system.rewrite_data_files(table => '{ident}')")

    # -- reads -----------------------------------------------------------

    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table:
        query = f"SELECT * FROM {self._q(ident)}"
        if snapshot_id is not None:
            query += f" VERSION AS OF {snapshot_id}"
        return _df_to_arrow(self.spark.sql(query))

    def sql(self, query: str, tables: dict[str, str]) -> pa.Table:
        for alias, ident in tables.items():
            self.spark.sql(f"SELECT * FROM {self._q(ident)}").createOrReplaceTempView(alias)
        translated = _TSTZ.sub("TIMESTAMP", query)
        try:
            return _df_to_arrow(self.spark.sql(translated))
        finally:
            for alias in tables:
                self.spark.catalog.dropTempView(alias)

    # -- snapshot metadata -------------------------------------------------

    def snapshots(self, ident: str) -> list[int]:
        """Snapshot ids oldest-first, so `[-1]` is the current snapshot."""
        jsnaps = sorted(self._jtable(ident).snapshots(), key=lambda s: s.timestampMillis())
        return [int(s.snapshotId()) for s in jsnaps]

    def snapshot_row_counts(self, ident: str) -> list[int]:
        """Rows added per snapshot. See PyIcebergEngine's docstring: this
        counts `added-records` for every operation including overwrites, so
        it is not directly comparable across an overwrite boundary -- use
        `snapshot_details` and filter on `is_full_rebuild` for that."""
        return [d["added_records"] for d in self.snapshot_details(ident)]

    def snapshot_details(self, ident: str) -> list[dict]:
        """Per-snapshot metadata, oldest-first. Same shape and same
        `is_full_rebuild` semantics as PyIcebergEngine.snapshot_details --
        reuses its `_is_full_rebuild`/`_summary_int` helpers directly, since
        that logic (operation == overwrite/replace, or a delete that clears
        the table, or an append whose parent was such a delete) already
        covers Spark's single-snapshot "overwrite" op as well as PyIceberg's
        two-snapshot delete+append pair.
        """
        jsnaps = sorted(self._jtable(ident).snapshots(), key=lambda s: s.timestampMillis())
        raw = []
        for snap in jsnaps:
            summary = dict(snap.summary()) if snap.summary() is not None else {}
            parent = snap.parentId()
            raw.append({
                "snapshot_id": int(snap.snapshotId()),
                "parent_snapshot_id": int(parent) if parent is not None else None,
                "operation": str(snap.operation()),
                "added_records": _summary_int(summary, "added-records"),
                "deleted_records": _summary_int(summary, "deleted-records"),
                "total_records": _summary_int(summary, "total-records"),
            })
        by_id = {d["snapshot_id"]: d for d in raw}
        for detail in raw:
            detail["is_full_rebuild"] = _is_full_rebuild(detail, by_id)
        return raw

    # -- internal ----------------------------------------------------------

    def _to_spark(self, data: pa.Table, target: pa.Schema):
        """`data` must already be conformed to `target` (see `_conform`)."""
        spark_schema = _arrow_schema_to_spark(target)
        rows = data.to_pylist()
        return self.spark.createDataFrame(rows, schema=spark_schema)


def _df_to_arrow(df) -> pa.Table:
    """Materialise a Spark DataFrame as a `pa.Table`.

    Deliberately not `df.toPandas()` -- see the module docstring's point 2:
    that path imports `distutils`, which does not exist on Python >= 3.12.
    `collect()` + `Row.asDict(recursive=True)` sidesteps it entirely and
    also happens to handle nested structs (Bronze's `amount`, `merchant`,
    ...) for free, since `asDict(recursive=True)` unwraps nested `Row`
    objects into plain dicts.
    """
    schema = _spark_schema_to_arrow(df.schema)
    rows = [r.asDict(recursive=True) for r in df.collect()]
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _spark_type_to_arrow(dt) -> pa.DataType:
    import pyspark.sql.types as T

    if isinstance(dt, T.StructType):
        return pa.struct([
            pa.field(f.name, _spark_type_to_arrow(f.dataType), nullable=f.nullable)
            for f in dt.fields
        ])
    if isinstance(dt, T.StringType):
        return pa.string()
    if isinstance(dt, T.LongType):
        return pa.int64()
    if isinstance(dt, T.IntegerType):
        return pa.int32()
    if isinstance(dt, T.DoubleType):
        return pa.float64()
    if isinstance(dt, T.FloatType):
        return pa.float32()
    if isinstance(dt, T.BooleanType):
        return pa.bool_()
    if isinstance(dt, T.DateType):
        return pa.date32()
    if isinstance(dt, T.TimestampType):
        return pa.timestamp("us", tz="UTC")
    raise TypeError(f"unmapped Spark type for Arrow conversion: {dt}")


def _spark_schema_to_arrow(struct) -> pa.Schema:
    return pa.schema([
        pa.field(f.name, _spark_type_to_arrow(f.dataType), nullable=f.nullable)
        for f in struct.fields
    ])


def _arrow_type_to_spark(t: pa.DataType):
    import pyspark.sql.types as T

    if pa.types.is_struct(t):
        return T.StructType([
            T.StructField(f.name, _arrow_type_to_spark(f.type), nullable=f.nullable)
            for f in t
        ])
    # PyIceberg's StringType() -> Arrow `large_string`; Spark's own scans
    # produce plain `string`. Both map to Spark StringType on the way in.
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return T.StringType()
    if pa.types.is_int64(t):
        return T.LongType()
    if pa.types.is_int32(t):
        return T.IntegerType()
    if pa.types.is_float64(t):
        return T.DoubleType()
    if pa.types.is_float32(t):
        return T.FloatType()
    if pa.types.is_boolean(t):
        return T.BooleanType()
    if pa.types.is_date32(t):
        return T.DateType()
    if pa.types.is_timestamp(t):
        return T.TimestampType()
    raise TypeError(f"unmapped Arrow type for Spark conversion: {t}")


def _arrow_schema_to_spark(schema: pa.Schema):
    import pyspark.sql.types as T

    return T.StructType([
        T.StructField(f.name, _arrow_type_to_spark(f.type), nullable=f.nullable)
        for f in schema
    ])
