"""Cross-engine parity: the point of `LakehouseEngine` being a Protocol.

An interface with one implementation is a guess about what would be
portable. This file is what turns that guess into a checked claim: the same
Silver build runs through PyIceberg and Spark, and the outputs are diffed.

Everything here is `@pytest.mark.spark` and skips outright when PySpark is
absent (`pytest.importorskip` below) -- that is the expected, offline CI
outcome. `ENGINE=pyiceberg` stays the default and does not import this
module or `spark_engine.py` at all.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

pyspark = pytest.importorskip("pyspark", reason="Spark engine is opt-in")

from src.lakehouse import bronze, schemas, silver


@pytest.fixture(scope="module")
def spark_engine(tmp_path_factory):
    from src.lakehouse.engines.spark_engine import SparkEngine

    engine = SparkEngine(warehouse=tmp_path_factory.mktemp("spark_wh"))
    yield engine
    engine.spark.stop()


@pytest.fixture(scope="module")
def pyiceberg_engine(tmp_path_factory):
    from src.lakehouse import catalog as catalog_mod
    from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine

    return PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("py_wh")))


@pytest.mark.spark
def test_both_engines_produce_identical_silver(spark_engine, pyiceberg_engine):
    """Build Bronze + Silver through two independent engines, diff Silver.

    Bronze (nested structs: `amount`, `runningBalance`, `merchant.address`)
    is ingested through both engines as setup -- that alone proves the
    nested-struct write path works on Spark -- but only Silver's flat
    `transactions` table is diffed here. Scoping the *diff* to Silver, not
    Bronze/Gold, is a deliberate choice: Bronze's `_raw_payload` column
    round-trips through Spark's struct machinery, and Gold isn't built by
    either engine in this test.
    """
    for engine in (pyiceberg_engine, spark_engine):
        bronze.ingest_all(engine, n_transactions=5000)
        silver.build_all(engine)

    def normalized(engine):
        df = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).to_pandas()
        # PyIceberg's StringType() -> Arrow `large_string`; Spark's own
        # scans produce plain `string` (see spark_engine.py's module
        # docstring, point 3). `.to_pandas()` collapses both to the same
        # `object` dtype, so this comparison does not silently coerce that
        # difference away -- it normalises past a representation detail
        # that both engines' *string values* agree on.
        return (df.sort_values("id").reset_index(drop=True)
                  .reindex(sorted(df.columns), axis=1))

    left, right = normalized(pyiceberg_engine), normalized(spark_engine)
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    import pandas.testing as pdt
    pdt.assert_frame_equal(left, right, check_dtype=False, atol=1e-9)


@pytest.mark.spark
def test_spark_merge_upsert_replaces_matching_rows(spark_engine):
    td = schemas.SILVER_MERCHANT_MAP
    spark_engine.create_table(td)
    spark_engine.overwrite(td.name, pa.table({
        "merchant_normalized": ["A"], "ticker": ["OLD"],
        "company_name": ["a"], "match_type": ["exact"]}))
    spark_engine.merge_upsert(td.name, pa.table({
        "merchant_normalized": ["A"], "ticker": ["NEW"],
        "company_name": ["a"], "match_type": ["exact"]}), key="merchant_normalized")
    out = spark_engine.scan_arrow(td.name)
    assert out.num_rows == 1
    assert out.column("ticker")[0].as_py() == "NEW"


@pytest.mark.spark
def test_spark_engine_conforms_columns_by_name_like_pyiceberg(spark_engine):
    """`append`/`overwrite` must raise on missing OR extra columns, same as
    PyIcebergEngine's `_conform` -- otherwise the two engines have different
    write contracts and the parity test above would be comparing apples to
    a looser interpretation of oranges."""
    td = schemas.SILVER_MERCHANT_MAP
    spark_engine.create_table(td)
    with pytest.raises(KeyError, match="match_type"):
        spark_engine.append(td.name, pa.table({
            "merchant_normalized": ["A"], "ticker": ["A"], "company_name": ["a"]}))
    with pytest.raises(KeyError, match="tickr"):
        spark_engine.append(td.name, pa.table({
            "merchant_normalized": ["A"], "ticker": ["A"], "company_name": ["a"],
            "match_type": ["exact"], "tickr": ["A"]}))


@pytest.mark.spark
def test_spark_engine_snapshot_details_mark_overwrite_as_full_rebuild(spark_engine):
    """Spark's `overwrite()` commits ONE snapshot with operation "overwrite"
    -- unlike PyIceberg's delete+append pair -- but `is_full_rebuild` must
    read the same way on both engines so a Task 18 volume monitor doesn't
    need engine-specific logic. Uses a Gold table nothing else in this
    module touches, so its snapshot history is exactly what this test
    writes regardless of test order.
    """
    td = schemas.GOLD_SPEND

    def row(ticker: str) -> pa.Table:
        return pa.table({
            "ticker": [ticker], "spend_date": [pa.scalar("2024-01-01").cast(pa.date32())],
            "gross_spend": [10.0], "txn_count": [1], "unique_accounts": [1],
            "avg_ticket": [10.0], "median_ticket": [10.0],
        })

    spark_engine.create_table(td)
    spark_engine.append(td.name, row("SBUX"))
    spark_engine.overwrite(td.name, row("CMG"))
    details = spark_engine.snapshot_details(td.name)
    assert [d["operation"] for d in details] == ["append", "overwrite"]
    assert [d["is_full_rebuild"] for d in details] == [False, True]
    assert details[-1]["total_records"] == 1
    assert [d["snapshot_id"] for d in details] == spark_engine.snapshots(td.name)


@pytest.mark.spark
def test_spark_engine_schema_history_and_arrow_schema(spark_engine):
    """A Gold table nothing else touches, so `schema_history` has exactly
    the one version this test creates."""
    td = schemas.GOLD_STOCK_FEATURES
    spark_engine.create_table(td)
    assert spark_engine.table_exists(td.name)
    history = spark_engine.schema_history(td.name)
    assert len(history) == 1
    assert set(history[0]["columns"]) == set(td.schema.as_arrow().names)
    assert set(spark_engine.arrow_schema(td.name).names) == set(td.schema.as_arrow().names)


@pytest.mark.spark
def test_tstz_cast_translation_leaves_string_literals_alone():
    """`sql()`'s dialect shim rewrites `AS TIMESTAMP WITH TIME ZONE` (a CAST
    target Spark's parser rejects) to `AS TIMESTAMP`. It must not touch the
    same text when it appears as *data* inside a string literal -- a blind,
    unscoped substitution would silently corrupt a query like this rather
    than fail loudly, which is exactly the failure mode this guards
    against.
    """
    from src.lakehouse.engines.spark_engine import _translate_tstz_cast

    query = ("SELECT 'CAST(x AS TIMESTAMP WITH TIME ZONE)' AS note, "
             "CAST(y AS TIMESTAMP WITH TIME ZONE) AS ts")
    translated = _translate_tstz_cast(query)
    assert "'CAST(x AS TIMESTAMP WITH TIME ZONE)'" in translated  # literal: untouched
    assert "CAST(y AS TIMESTAMP)" in translated  # real cast target: translated
    assert "TIME ZONE" not in translated.split("note,")[1]  # only the literal half keeps it


@pytest.mark.spark
def test_spark_engine_sql_does_not_rewrite_the_phrase_inside_a_string_literal(spark_engine):
    """End-to-end: the query actually runs on Spark and the literal value
    comes back byte-for-byte, proving the shim didn't corrupt it."""
    out = spark_engine.sql(
        "SELECT 'CAST(x AS TIMESTAMP WITH TIME ZONE)' AS note", tables={})
    assert out.column("note")[0].as_py() == "CAST(x AS TIMESTAMP WITH TIME ZONE)"
