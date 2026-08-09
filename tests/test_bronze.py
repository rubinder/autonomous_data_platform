import pyarrow as pa
import pytest

from src.lakehouse import bronze, schemas
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    return PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))


def test_ingest_writes_all_three_bronze_tables(engine):
    counts = bronze.ingest_all(engine, n_transactions=3000)
    assert counts["bronze.yodlee_transactions_raw"] == 3000
    assert counts["bronze.yodlee_accounts_raw"] > 0
    assert counts["bronze.stock_prices_raw"] > 0


def test_lineage_columns_present_and_populated(engine):
    bronze.ingest_all(engine, n_transactions=500)
    tbl = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name)
    for col in ("_ingested_at", "_source_file", "_payload_hash", "_raw_payload"):
        assert col in tbl.column_names
        assert tbl.column(col).null_count == 0


def test_nested_structs_preserved_not_flattened(engine):
    bronze.ingest_all(engine, n_transactions=200)
    tbl = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name)
    assert pa.types.is_struct(tbl.schema.field("amount").type)
    assert pa.types.is_struct(tbl.schema.field("merchant").type)
    # Direction still lives in baseType -- Bronze must not pre-sign the amount.
    amounts = tbl.column("amount").combine_chunks().field("amount").to_pylist()
    assert all(a > 0 for a in amounts)


def test_reingest_creates_new_snapshot_and_duplicates_rows(engine):
    """Bronze is append-only: re-ingest duplicates by design. Silver dedupes."""
    bronze.ingest_all(engine, n_transactions=1000)
    snaps_before = len(engine.snapshots(schemas.BRONZE_TRANSACTIONS.name))
    bronze.ingest_all(engine, n_transactions=1000)
    assert len(engine.snapshots(schemas.BRONZE_TRANSACTIONS.name)) == snaps_before + 1
    assert engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows == 2000


def test_payload_hash_is_stable_for_identical_record(engine):
    rec = {"id": 1, "amount": {"amount": 5.0, "currency": "USD"}}
    a = bronze.payload_hash(rec)
    b = bronze.payload_hash(dict(reversed(list(rec.items()))))
    assert a == b  # key order must not change the hash


def test_max_txn_date_feeds_as_of_resolution(engine):
    bronze.ingest_all(engine, n_transactions=2000)
    from src import config
    assert bronze.max_txn_date(engine) <= config.END_DATE
