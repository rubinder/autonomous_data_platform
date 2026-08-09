"""Engine contract tests: create/append/overwrite/scan, snapshots, time travel."""
from __future__ import annotations

import importlib

import pyarrow as pa
import pytest

from src import config
from src.lakehouse import catalog as catalog_mod
from src.lakehouse import schemas
from src.lakehouse.engines import get_engine


@pytest.fixture
def engine(tmp_path):
    cat = catalog_mod.get_catalog(tmp_path / "wh")
    from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
    return PyIcebergEngine(cat)


def _row(marker: str) -> pa.Table:
    return pa.table({
        "merchant_normalized": [marker],
        "ticker": [marker],
        "company_name": [marker.lower()],
        "match_type": ["exact"],
    })


def test_create_append_and_scan_roundtrip(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    data = pa.table({
        "merchant_normalized": ["STARBUCKS"],
        "ticker": ["SBUX"],
        "company_name": ["Starbucks"],
        "match_type": ["exact"],
    })
    engine.append(td.name, data)
    out = engine.scan_arrow(td.name)
    assert out.num_rows == 1
    assert out.column("ticker")[0].as_py() == "SBUX"


def test_append_creates_a_new_snapshot_each_time(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    row = _row("X")
    engine.append(td.name, row)
    first = engine.snapshots(td.name)
    engine.append(td.name, row)
    second = engine.snapshots(td.name)
    assert len(second) == len(first) + 1


def test_time_travel_reads_prior_snapshot(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    row = _row("X")
    engine.append(td.name, row)
    snap_one = engine.snapshots(td.name)[-1]
    engine.append(td.name, row)
    assert engine.scan_arrow(td.name).num_rows == 2
    assert engine.scan_arrow(td.name, snapshot_id=snap_one).num_rows == 1


def test_overwrite_replaces_contents(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    engine.append(td.name, _row("A"))
    engine.overwrite(td.name, _row("B"))
    out = engine.scan_arrow(td.name)
    assert out.num_rows == 1
    assert out.column("ticker")[0].as_py() == "B"


def test_sql_runs_duckdb_over_registered_tables(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    engine.append(td.name, pa.table({
        "merchant_normalized": ["A", "B"], "ticker": ["A", "B"],
        "company_name": ["a", "b"], "match_type": ["exact", "exact"]}))
    out = engine.sql("SELECT count(*) AS n FROM silver_merchant_ticker_map",
                     tables={"silver_merchant_ticker_map": td.name})
    assert out.column("n")[0].as_py() == 2


def test_snapshot_row_counts_are_per_batch_not_cumulative(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    engine.append(td.name, _row("A"))
    engine.append(td.name, pa.table({
        "merchant_normalized": ["B", "C"], "ticker": ["B", "C"],
        "company_name": ["b", "c"], "match_type": ["exact", "exact"]}))
    assert engine.snapshot_row_counts(td.name) == [1, 2]


def test_every_table_definition_creates(engine):
    """All 12 schemas and partition specs must survive a real catalog commit."""
    for td in schemas.ALL_TABLES:
        engine.create_table(td)
        assert engine.table_exists(td.name)
        engine.create_table(td)  # idempotent


def test_schema_history_reports_current_columns(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    history = engine.schema_history(td.name)
    assert len(history) == 1
    assert set(history[0]["columns"]) == {
        "merchant_normalized", "ticker", "company_name", "match_type"}


def test_get_engine_defaults_to_pyiceberg(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGINE", raising=False)
    monkeypatch.setenv("WAREHOUSE_PATH", str(tmp_path / "wh2"))
    # config.WAREHOUSE_PATH is bound at import time; reload so the env var lands.
    importlib.reload(config)
    try:
        eng = get_engine()
        assert eng.__class__.__name__ == "PyIcebergEngine"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
