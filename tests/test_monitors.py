from datetime import date

import pyarrow as pa
import pytest

from src.lakehouse import bronze, gold, schemas, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import monitors, runner


def _txn_record(txn_id: int, last_updated: str, amount: float,
                 base_type: str = "DEBIT", currency: str = "USD") -> dict:
    """A minimal but schema-complete Bronze transaction record. Mirrors
    tests/test_silver.py's helper of the same name -- kept local rather than
    imported so this module does not depend on test collection order."""
    return {
        "id": txn_id, "accountId": 100_000, "date": "2024-06-01",
        "transactionDate": "2024-06-01", "postDate": "2024-06-02",
        "amount": {"amount": amount, "currency": currency},
        "runningBalance": {"amount": 1000.0, "currency": currency},
        "merchant": {
            "id": "M1", "source": "TEST MERCHANT", "categoryLabel": "Test",
            "address": {"city": "SEATTLE", "state": "WA", "country": "USA"},
        },
        "status": "POSTED", "baseType": base_type,
        "subType": "PAYMENT" if base_type == "DEBIT" else "CREDIT",
        "category": "Other Expenses",
        "categoryType": "EXPENSE" if base_type == "DEBIT" else "INCOME",
        "categoryId": 40, "detailCategoryId": 4000,
        "detailCategory": "Uncategorized", "highLevelCategoryId": 30,
        "categorySource": "SYSTEM", "sourceType": "AGGREGATED",
        "checkNumber": "", "isManual": False, "container": None,
        "createdDate": last_updated, "lastUpdated": last_updated,
    }


def _append_synthetic_txns(engine, records: list[dict]) -> None:
    enriched = bronze.add_lineage(records, "synthetic_test.json")
    table = pa.Table.from_pylist(enriched, schema=schemas.BRONZE_TRANSACTIONS.schema.as_arrow())
    engine.append(schemas.BRONZE_TRANSACTIONS.name, table)


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


def test_unknown_kind_raises_instead_of_reporting_ok():
    """One typo in a YAML `kind:` used to mint a permanently-passing monitor.

    Falling through to "ok" is the worst available outcome: the check is
    counted, displayed green, and evaluates nothing.
    """
    with pytest.raises(ValueError, match="unknown monitor kind"):
        monitors.evaluate(metric=1.0, baseline=[1.0, 2.0],
                          params={"kind": "row_cont"})  # typo


def test_a_typo_in_kind_surfaces_as_a_breach_not_a_pass(engine, tmp_path):
    """`runner` must convert that raise into a visible breach, not a crash."""
    (tmp_path / "typo.yaml").write_text(
        "- name: typo_monitor\n"
        "  table: silver.transactions\n"
        "  kind: row_cont\n"
        "  query: SELECT count(*) AS metric FROM t\n")
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30),
                                  monitor_dir=tmp_path)
    typo = next(r for r in results if r.monitor == "typo_monitor")
    assert typo.status == "breach"
    assert "unknown monitor kind" in typo.detail


def test_column_type_results_are_stamped_with_the_runs_as_of(engine):
    """These seven results used to re-derive `run_at` from the environment.

    `resolve_as_of_date(None)` ignores the caller's as_of and falls back to
    END_DATE, so one monitor run landed in `ops.monitor_results` under two
    different `run_at` values.
    """
    as_of = date(2024, 3, 15)
    results = runner.run_monitors(engine, as_of=as_of)
    typed = [r for r in results if r.kind == "column_type"]
    assert typed
    assert {r.run_at for r in typed} == {as_of}
    assert {r.run_at for r in results} == {as_of}


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


def test_bronze_volume_monitor_catches_a_collapsed_batch(tmp_path_factory):
    """Cumulative count(*) barely moves when a near-total outage hits an
    append-only feed -- five 1,000-row batches then a single row still reads
    as ~5,001 vs a baseline of ~3,000, comfortably above breach_ratio. The
    monitor instead measures rows added per snapshot (source: snapshot_added),
    which sees the collapse directly."""
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    results = []
    for n in (1000, 1000, 1000, 1000, 1000, 1):
        bronze.ingest_all(eng, n_transactions=n)
        results = runner.run_monitors(eng, as_of=date(2026, 6, 30))
        runner.persist(eng, results)

    volume = next(r for r in results if r.monitor == "bronze_txn_row_count")
    assert volume.metric == 1
    assert volume.status == "breach"


def test_quarantine_rate_monitor_fires_on_injected_non_usd_rows(tmp_path_factory):
    """The old query (`SELECT 0.0 ... LIMIT 1`) could never fire. The real
    one joins the quarantine table against the clean one and must breach
    when a batch of non-USD rows lands."""
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=200)
    silver.build_all(eng)

    eur_rows = [
        _txn_record(900_000 + i, "2024-06-01T00:00:00+00:00", 50.0, currency="EUR")
        for i in range(10)
    ]
    _append_synthetic_txns(eng, eur_rows)
    silver.build_all(eng)

    results = runner.run_monitors(eng, as_of=date(2026, 6, 30))
    quarantine = next(r for r in results if r.monitor == "silver_txn_quarantine_rate")
    assert quarantine.metric > 0
    assert quarantine.status == "breach"
