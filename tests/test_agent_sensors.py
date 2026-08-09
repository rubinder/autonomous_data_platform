from datetime import date

import pytest

from src.agent import classifier, sensors
from src.contracts import validator
from src.lakehouse import bronze, schemas
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


def _state(**kw):
    base = {"table": "bronze.yodlee_transactions_raw",
            "columns": {"id": "long", "baseType": "string", "amount": "struct"},
            "row_count": 250_000, "rows_in_latest_snapshot": 250_000,
            "snapshot_count": 3, "newest_date": date(2026, 6, 30)}
    base.update(kw)
    return sensors.ObservedState(**base)


def _contract(fields):
    from src.contracts.validator import Contract, SchemaField
    return Contract(
        table="bronze.yodlee_transactions_raw", version=1, owner="x",
        schema_fields=tuple(SchemaField(n, t) for n, t in fields),
        expectations=({"type": "freshness", "column": "txn_date", "max_lag_days": 3},))


def test_no_findings_when_state_matches_contract():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    findings = sensors.detect(_state(), contract, as_of=date(2026, 6, 30),
                              history=[250_000, 249_000, 251_000])
    assert findings == []


def test_added_column_is_detected_and_classified_additive():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    state = _state(columns={"id": "long", "baseType": "string", "amount": "struct",
                            "merchantCategoryCode": "string"})
    findings = sensors.detect(state, contract, as_of=date(2026, 6, 30),
                              history=[250_000] * 3)
    drift = [f for f in findings if f.kind == "schema_drift"]
    assert len(drift) == 1
    severity, reasoning = classifier.classify(drift[0])
    assert severity == "additive"
    assert reasoning


def test_dropped_column_is_breaking():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    state = _state(columns={"id": "long", "baseType": "string"})
    findings = sensors.detect(state, contract, as_of=date(2026, 6, 30),
                              history=[250_000] * 3)
    drift = next(f for f in findings if f.kind == "schema_drift")
    assert classifier.classify(drift)[0] == "breaking"


def test_type_narrowing_is_breaking():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    state = _state(columns={"id": "int", "baseType": "string", "amount": "struct"})
    findings = sensors.detect(state, contract, as_of=date(2026, 6, 30),
                              history=[250_000] * 3)
    drift = next(f for f in findings if f.kind == "schema_drift")
    assert classifier.classify(drift)[0] == "breaking"


def test_volume_collapse_is_detected():
    contract = _contract([("id", "long")])
    findings = sensors.detect(
        _state(rows_in_latest_snapshot=1000, columns={"id": "long"}),
        contract, as_of=date(2026, 6, 30),
        history=[250_000, 249_000, 251_000, 248_000])
    vol = [f for f in findings if f.kind == "volume_anomaly"]
    assert len(vol) == 1
    assert classifier.classify(vol[0])[0] == "breaking"


def test_modest_volume_change_is_not_flagged():
    contract = _contract([("id", "long")])
    findings = sensors.detect(
        _state(rows_in_latest_snapshot=230_000, columns={"id": "long"}),
        contract, as_of=date(2026, 6, 30),
        history=[250_000, 249_000, 251_000])
    assert not [f for f in findings if f.kind == "volume_anomaly"]


def test_staleness_measured_against_as_of_not_wall_clock():
    contract = _contract([("id", "long")])
    fresh = sensors.detect(_state(columns={"id": "long"}), contract,
                           as_of=date(2026, 6, 30), history=[250_000] * 3)
    assert not [f for f in fresh if f.kind == "staleness"]
    stale = sensors.detect(_state(columns={"id": "long"}), contract,
                           as_of=date(2026, 8, 9), history=[250_000] * 3)
    assert [f for f in stale if f.kind == "staleness"]


@pytest.fixture
def engine(tmp_path):
    return PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))


def test_observe_reads_real_bronze_table_from_metadata(engine):
    bronze.ingest_all(engine, n_transactions=1000)
    ident = schemas.BRONZE_TRANSACTIONS.name
    expected_last_batch = engine.snapshot_row_counts(ident)[-1]
    expected_columns = engine.schema_history(ident)[-1]["columns"]

    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")

    assert state.columns == expected_columns
    assert state.rows_in_latest_snapshot == expected_last_batch
    assert state.snapshot_count == len(engine.snapshot_row_counts(ident))
    assert isinstance(state.newest_date, date)
    from src import config
    assert state.newest_date <= config.END_DATE


def test_staleness_fires_end_to_end_against_real_bronze_contract(engine):
    bronze.ingest_all(engine, n_transactions=1000)
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    history = engine.snapshot_row_counts(schemas.BRONZE_TRANSACTIONS.name)

    fresh = sensors.detect(state, contract, as_of=date(2026, 6, 30), history=history)
    assert not [f for f in fresh if f.kind == "staleness"]

    stale = sensors.detect(state, contract, as_of=date(2026, 8, 9), history=history)
    stale_findings = [f for f in stale if f.kind == "staleness"]
    assert len(stale_findings) == 1
    assert classifier.classify(stale_findings[0])[0] == "breaking"


def test_llm_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib
    importlib.reload(classifier)
    assert classifier.USE_LLM is False
