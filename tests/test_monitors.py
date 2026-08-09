from datetime import date

import pytest

from src.lakehouse import bronze, gold, schemas, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import monitors, runner


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=20000)
    silver.build_all(eng)
    gold.build_all(eng)
    return eng


def test_monitor_definitions_load_from_yaml():
    defs = monitors.load_monitors()
    assert len(defs) >= 6
    kinds = {d.kind for d in defs}
    assert {"row_count", "null_rate", "distribution_shift",
            "duplicate_rate", "cardinality"} <= kinds
    assert all(d.query.strip() for d in defs)


def test_evaluate_flags_a_collapse_against_baseline():
    status, detail = monitors.evaluate(
        metric=10.0, baseline=[1000.0, 1010.0, 990.0, 1005.0],
        params={"kind": "row_count", "breach_ratio": 0.5})
    assert status == "breach"
    assert detail


def test_evaluate_passes_a_normal_value():
    status, _ = monitors.evaluate(
        metric=995.0, baseline=[1000.0, 1010.0, 990.0, 1005.0],
        params={"kind": "row_count", "breach_ratio": 0.5})
    assert status == "ok"


def test_evaluate_is_ok_with_no_baseline_rather_than_alarming():
    """First run has no history. Alerting on that trains people to ignore it."""
    status, detail = monitors.evaluate(
        metric=10.0, baseline=[], params={"kind": "row_count", "breach_ratio": 0.5})
    assert status == "ok"
    assert "no baseline" in detail.lower()


def test_distribution_shift_uses_robust_z_score():
    baseline = [100.0] * 10 + [101.0, 99.0]
    breach, _ = monitors.evaluate(
        metric=500.0, baseline=baseline,
        params={"kind": "distribution_shift", "breach_z": 4.0})
    ok, _ = monitors.evaluate(
        metric=100.5, baseline=baseline,
        params={"kind": "distribution_shift", "breach_z": 4.0})
    assert breach == "breach"
    assert ok == "ok"


def test_constant_baseline_does_not_divide_by_zero():
    status, _ = monitors.evaluate(
        metric=100.0, baseline=[100.0] * 8,
        params={"kind": "distribution_shift", "breach_z": 4.0})
    assert status == "ok"


def test_column_type_conformance_passes_on_clean_silver(engine):
    from src.contracts import validator
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "silver_transactions.yaml")
    results = monitors.check_column_types(
        engine, schemas.SILVER_TRANSACTIONS, contract)
    assert results
    assert all(r.status == "ok" for r in results), [
        r.detail for r in results if r.status != "ok"]


def test_column_type_conformance_detects_a_declared_type_mismatch(engine):
    from src.contracts.validator import Contract, SchemaField
    wrong = Contract(
        table="silver.transactions", version=1, owner="x",
        schema_fields=(SchemaField("signed_amount", "string", False),),
        expectations=())
    results = monitors.check_column_types(
        engine, schemas.SILVER_TRANSACTIONS, wrong)
    assert any(r.status == "breach" for r in results)


def test_run_monitors_returns_results_for_every_definition(engine):
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30))
    assert len(results) >= 6
    assert all(r.status in {"ok", "warn", "breach"} for r in results)


def test_results_persist_and_are_readable_as_baselines(engine):
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30))
    written = runner.persist(engine, results)
    assert written == len(results)
    assert engine.table_exists(schemas.OPS_MONITOR_RESULTS.name)

    target = next(r for r in results if r.metric is not None)
    baselines = runner.load_baselines(engine, target.monitor, target.column)
    assert baselines


def test_persisted_history_accumulates_across_runs(engine):
    before = engine.scan_arrow(schemas.OPS_MONITOR_RESULTS.name).num_rows
    runner.persist(engine, runner.run_monitors(engine, as_of=date(2026, 6, 29)))
    after = engine.scan_arrow(schemas.OPS_MONITOR_RESULTS.name).num_rows
    assert after > before


def test_duplicate_rate_monitor_is_zero_on_deduped_silver(engine):
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30))
    dupes = [r for r in results if r.monitor == "silver_txn_duplicate_ids"]
    assert dupes and dupes[0].metric == 0
