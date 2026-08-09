from datetime import date

import pytest

from src.lakehouse import bronze, schemas, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=20000)
    return eng


@pytest.mark.parametrize("raw,expected", [
    ("STARBUCKS STORE #1234", "STARBUCKS STORE"),
    ("Starbucks", "STARBUCKS"),
    ("CHIPOTLE 0455", "CHIPOTLE"),
    ("Domino's Pizza", "DOMINOS PIZZA"),
    ("TARGET T-1088", "TARGET T"),
    ("lululemon athletica", "LULULEMON ATHLETICA"),
])
def test_merchant_normalization(raw, expected):
    assert silver.normalize_merchant(raw) == expected


def test_signed_amount_direction_comes_from_base_type(engine):
    silver.build_all(engine)
    out = engine.sql(
        """SELECT base_type, min(signed_amount) AS lo, max(signed_amount) AS hi
           FROM s GROUP BY base_type""",
        tables={"s": schemas.SILVER_TRANSACTIONS.name})
    rows = {r["base_type"]: r for r in out.to_pylist()}
    assert rows["DEBIT"]["hi"] < 0
    assert rows["CREDIT"]["lo"] > 0


def test_dedup_keeps_latest_last_updated(engine):
    """Re-ingest duplicates Bronze rows; Silver must collapse them."""
    bronze_rows = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows
    bronze.ingest_all(engine, n_transactions=20000)
    assert engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows == bronze_rows * 2
    silver.build_all(engine)
    n = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).num_rows
    assert n == bronze_rows


def test_ids_unique_in_silver(engine):
    silver.build_all(engine)
    out = engine.sql("SELECT count(*) AS n, count(DISTINCT id) AS d FROM s",
                     tables={"s": schemas.SILVER_TRANSACTIONS.name})
    row = out.to_pylist()[0]
    assert row["n"] == row["d"]


def test_merchant_map_covers_every_configured_merchant_string(engine):
    from src import config
    silver.build_all(engine)
    mapped = engine.sql("SELECT merchant_normalized, ticker FROM m",
                        tables={"m": schemas.SILVER_MERCHANT_MAP.name}).to_pylist()
    lookup = {r["merchant_normalized"]: r["ticker"] for r in mapped}
    for company in config.COMPANIES:
        for raw in company.merchant_strings:
            assert lookup[silver.normalize_merchant(raw)] == company.ticker


def test_non_usd_rows_are_quarantined_not_dropped(engine):
    silver.build_all(engine)
    assert engine.table_exists(schemas.SILVER_QUARANTINE.name)
    kept = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).num_rows
    quarantined = engine.scan_arrow(schemas.SILVER_QUARANTINE.name).num_rows
    bronze_distinct = engine.sql(
        "SELECT count(DISTINCT id) AS d FROM b",
        tables={"b": schemas.BRONZE_TRANSACTIONS.name}).to_pylist()[0]["d"]
    assert kept + quarantined == bronze_distinct


def test_prices_typed_to_date_and_aligned_to_calendar(engine):
    silver.build_all(engine)
    tbl = engine.scan_arrow(schemas.SILVER_PRICES.name)
    dates = tbl.column("trade_date").to_pylist()
    assert all(isinstance(d, date) for d in dates)
    assert all(d.weekday() < 5 for d in dates)


def test_contract_violation_aborts_the_build(engine, monkeypatch):
    monkeypatch.setattr(silver, "_SIGN_DEBITS", False)  # inject the classic bug
    with pytest.raises(Exception):  # noqa: B017 -- any failure must abort; type isn't the point
        silver.build_all(engine)
