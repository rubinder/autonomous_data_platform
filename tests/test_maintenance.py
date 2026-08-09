from datetime import date

import pytest

from src.lakehouse import bronze, maintenance, schemas
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=2000)
    return eng


def test_time_travel_shows_history_growing(engine):
    bronze.ingest_all(engine, n_transactions=500)
    result = maintenance.time_travel_demo(engine)
    assert result["rows_at_first_snapshot"] < result["rows_at_latest_snapshot"]


def test_evolved_column_detected_as_additive_drift_by_the_agent(engine):
    from pyiceberg.types import StringType

    from src.agent import classifier, sensors
    from src.contracts import validator

    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    findings = sensors.detect(state, contract, date(2026, 6, 30),
                              history=[state.rows_in_latest_snapshot])
    added = [f for f in findings
             if f.evidence.get("column") == "merchantCategoryCode"]
    assert added
    assert classifier.classify(added[0])[0] == "additive"


def test_volume_collapse_is_detected_without_violating_append_only(engine):
    from src.agent import sensors
    from src.contracts import validator

    rows_before = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows
    history = engine.snapshot_row_counts(schemas.BRONZE_TRANSACTIONS.name)
    added = maintenance.inject_volume_collapse(engine)

    # Bronze stays append-only: the collapse is a tiny APPEND, not a rewrite.
    assert engine.scan_arrow(
        schemas.BRONZE_TRANSACTIONS.name).num_rows == rows_before + added
    details = engine.snapshot_details(schemas.BRONZE_TRANSACTIONS.name)
    assert details[-1]["operation"] == "append"
    assert not any(d["is_full_rebuild"] for d in details)

    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    findings = sensors.detect(state, contract, date(2026, 6, 30), history=history)
    assert any(f.kind == "volume_anomaly" for f in findings)


def test_volume_collapse_appends_against_an_evolved_table(engine):
    """drift-demo evolves the schema first, so the injector must follow it."""
    from pyiceberg.types import StringType

    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")
    rows_before = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows
    added = maintenance.inject_volume_collapse(engine)
    assert engine.scan_arrow(
        schemas.BRONZE_TRANSACTIONS.name).num_rows == rows_before + added


def test_expire_snapshots_retains_requested_count(engine):
    """Exactly the requested count, and exactly the newest ones."""
    ident = schemas.BRONZE_TRANSACTIONS.name
    for _ in range(4):
        bronze.ingest_all(engine, n_transactions=300)
    before = engine.snapshots(ident)
    assert len(before) == 5

    removed = maintenance.expire(engine, retain_last=2)

    after = engine.snapshots(ident)
    assert len(after) == 2
    assert after == before[-2:]     # the newest two, not any two
    assert removed >= 3             # bronze's 3, plus any other table's

    with pytest.raises(Exception):  # noqa: B017 -- PyIceberg raises ValueError
        engine.scan_arrow(ident, snapshot_id=before[0])  # genuinely gone


def test_expire_is_a_no_op_when_history_is_short(engine):
    before = engine.snapshots(schemas.BRONZE_TRANSACTIONS.name)
    assert maintenance.expire(engine, retain_last=5) == 0
    assert engine.snapshots(schemas.BRONZE_TRANSACTIONS.name) == before


def test_catalog_only_commands_fail_with_a_clear_message(monkeypatch):
    """`main()` dispatches through `get_engine()`, but drift-demo,
    cross-version and expire reach past the protocol to `engine.catalog` for
    PyIceberg APIs the protocol does not expose. Under an engine without one
    that used to be a bare `AttributeError` from deep inside a helper.
    """
    class NoCatalogEngine:
        pass

    engine = NoCatalogEngine()
    for command in sorted(maintenance._CATALOG_COMMANDS):
        with pytest.raises(SystemExit) as exc:
            maintenance._require_catalog(engine, command)
        message = str(exc.value)
        assert command in message
        assert "ENGINE=pyiceberg" in message

    # Protocol-only commands must not be gated.
    assert "timetravel" not in maintenance._CATALOG_COMMANDS
    assert "schema-history" not in maintenance._CATALOG_COMMANDS


def test_catalog_backed_engine_passes_the_guard(engine):
    for command in maintenance._CATALOG_COMMANDS:
        maintenance._require_catalog(engine, command)   # must not raise
