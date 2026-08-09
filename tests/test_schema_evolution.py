import pytest
from pyiceberg.types import IntegerType, LongType, StringType

from src.lakehouse import bronze, maintenance, schemas
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine

IDENT = schemas.BRONZE_TRANSACTIONS.name


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=1000)   # written under schema v1
    return eng


def test_added_column_is_null_for_rows_written_before_it(engine):
    rows_before = engine.scan_arrow(IDENT).num_rows
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    current = engine.scan_arrow(IDENT)
    assert "merchantCategoryCode" in current.column_names
    assert current.num_rows == rows_before
    # Old files carry no such column; Iceberg fills it rather than failing.
    assert current.column("merchantCategoryCode").null_count == rows_before


def test_old_snapshot_still_readable_after_evolution(engine):
    before = engine.snapshots(IDENT)[-1]
    maintenance.evolve_all(engine)
    historical = engine.scan_arrow(IDENT, snapshot_id=before)
    assert historical.num_rows > 0


def test_widening_int_to_long_preserves_existing_values(engine):
    maintenance.evolve_add_column(engine, "settlementDays", IntegerType())
    bronze.ingest_all(engine, n_transactions=200)
    widened = maintenance.evolve_widen_column(engine, "settlementDays", LongType())
    assert widened
    out = engine.scan_arrow(IDENT)
    assert "settlementDays" in out.column_names
    assert out.num_rows == 1200


def test_rename_resolves_by_field_id_not_name(engine):
    """The crux: rows written before the rename read back under the NEW name."""
    rows_before = engine.scan_arrow(IDENT).num_rows
    field_id_before = engine.schema_history(IDENT)[-1]["field_ids"]["checkNumber"]

    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")

    out = engine.scan_arrow(IDENT)
    assert "check_reference" in out.column_names
    assert "checkNumber" not in out.column_names
    assert out.num_rows == rows_before          # no rewrite, no data loss
    field_id_after = engine.schema_history(IDENT)[-1]["field_ids"]["check_reference"]
    assert field_id_after == field_id_before    # same field, new name


def test_rename_does_not_lose_the_values_written_under_the_old_name(engine):
    """The point of a field-ID rename: the data comes back, not just the column.

    A rename that resolved by name would return an all-null column here, which
    is exactly the failure `test_rename_resolves_by_field_id_not_name` is
    guarding against -- but a null column would still satisfy that test's
    column-name assertions, so assert on the values too.
    """
    before = engine.scan_arrow(IDENT).column("checkNumber").to_pylist()
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")
    after = engine.scan_arrow(IDENT).column("check_reference").to_pylist()
    assert after == before
    assert any(v is not None for v in after)


def test_schema_history_records_every_version(engine):
    maintenance.evolve_all(engine)
    history = maintenance.schema_versions(engine, IDENT)
    assert len(history) >= 4
    assert len({h["schema_id"] for h in history}) == len(history)
    assert "merchantCategoryCode" in history[-1]["columns"]


def test_cross_version_query_spans_every_snapshot(engine):
    """Ingest under v1, evolve, ingest under v5, query the whole history."""
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    bronze.ingest_all(engine, n_transactions=500)
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")
    bronze.ingest_all(engine, n_transactions=300)

    combined = maintenance.query_across_versions(engine, IDENT)
    assert "_snapshot_id" in combined.column_names
    assert "_schema_id" in combined.column_names
    assert "check_reference" in combined.column_names
    # Rows appear once per snapshot in which the table contained them.
    assert combined.num_rows > 1800
    assert len(set(combined.column("_snapshot_id").to_pylist())) == 3


def test_cross_version_query_reads_pre_rename_rows_under_the_new_name(engine):
    """Snapshots older than the rename are aligned by field ID, not by name.

    PyIceberg scans a historical snapshot with *that snapshot's* schema, so the
    oldest snapshot hands back `checkNumber`. Reconciling by name would drop
    those values on the floor and report an all-null `check_reference`; that
    would make the cross-version query a demonstration of nothing.
    """
    oldest = engine.snapshots(IDENT)[0]
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")

    combined = maintenance.query_across_versions(engine, IDENT)
    mask = [s == oldest for s in combined.column("_snapshot_id").to_pylist()]
    values = [v for v, keep in
              zip(combined.column("check_reference").to_pylist(), mask) if keep]
    assert values
    assert any(v is not None for v in values)


def test_cross_version_query_reconciles_missing_columns_to_null(engine):
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    bronze.ingest_all(engine, n_transactions=200)
    combined = maintenance.query_across_versions(engine, IDENT)
    assert "merchantCategoryCode" in combined.column_names
    # The first snapshot predates the column entirely.
    assert combined.column("merchantCategoryCode").null_count > 0


def test_incremental_read_returns_only_the_delta(engine):
    first = engine.snapshots(IDENT)[-1]
    bronze.ingest_all(engine, n_transactions=400)
    second = engine.snapshots(IDENT)[-1]
    assert maintenance.incremental_rows(engine, IDENT, first, second) == 400


def test_evolution_is_idempotent(engine):
    maintenance.evolve_all(engine)
    schema_count = len(maintenance.schema_versions(engine, IDENT))
    assert maintenance.evolve_all(engine) == []   # re-running must be a no-op
    assert len(maintenance.schema_versions(engine, IDENT)) == schema_count


def test_bronze_still_ingests_against_an_evolved_table(engine):
    """The source feed does not know about the new column; Bronze must not care.

    `engine.append` rejects a column set that does not match the table exactly,
    so a writer built from the *authoring* schema would raise KeyError the
    moment the live table gained a column. Bronze conforms to the live schema
    and leaves the unknown column NULL.
    """
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    bronze.ingest_all(engine, n_transactions=250)
    out = engine.scan_arrow(IDENT)
    assert out.num_rows == 1250
    assert out.column("merchantCategoryCode").null_count == 1250
