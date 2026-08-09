from datetime import date

import pyarrow as pa
import pytest

from src.lakehouse import bronze, schemas, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=20000)
    return eng


def _txn_record(txn_id: int, last_updated: str, amount: float,
                 base_type: str = "DEBIT", currency: str = "USD",
                 txn_date: str = "2024-06-01", post_date: str = "2024-06-02") -> dict:
    """A minimal but schema-complete Bronze transaction record, hand-built so
    tests can control `id`/`lastUpdated`/`amount`/`currency` directly instead
    of relying on the generator's deterministic (and therefore duplicate-free
    across the fields that matter) output."""
    return {
        "id": txn_id,
        "accountId": 100_000,
        "date": txn_date,
        "transactionDate": txn_date,
        "postDate": post_date,
        "amount": {"amount": amount, "currency": currency},
        "runningBalance": {"amount": 1000.0, "currency": currency},
        "merchant": {
            "id": "M1",
            "source": "TEST MERCHANT",
            "categoryLabel": "Test",
            "address": {"city": "SEATTLE", "state": "WA", "country": "USA"},
        },
        "status": "POSTED",
        "baseType": base_type,
        "subType": "PAYMENT" if base_type == "DEBIT" else "CREDIT",
        "category": "Other Expenses",
        "categoryType": "EXPENSE" if base_type == "DEBIT" else "INCOME",
        "categoryId": 40,
        "detailCategoryId": 4000,
        "detailCategory": "Uncategorized",
        "highLevelCategoryId": 30,
        "categorySource": "SYSTEM",
        "sourceType": "AGGREGATED",
        "checkNumber": "",
        "isManual": False,
        "container": None,
        "createdDate": last_updated,
        "lastUpdated": last_updated,
    }


def _append_synthetic_txns(engine, records: list[dict]) -> None:
    enriched = bronze.add_lineage(records, "synthetic_test.json")
    table = pa.Table.from_pylist(enriched, schema=schemas.BRONZE_TRANSACTIONS.schema.as_arrow())
    engine.append(schemas.BRONZE_TRANSACTIONS.name, table)


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


def test_dedup_tie_break_prefers_greater_last_updated(engine):
    """The generator's re-ingest duplicates are byte-identical, so
    `test_dedup_keeps_latest_last_updated` collapsing them proves nothing
    about which row the ORDER BY tie-break actually keeps. This test injects
    a same-id pair that differs in BOTH `lastUpdated` and `amount`, so only
    the correct tie-break produces the expected surviving amount."""
    txn_id = 999_999_001
    older = _txn_record(txn_id, "2020-01-01T00:00:00+00:00", 111.11)
    newer = _txn_record(txn_id, "2030-01-01T00:00:00+00:00", 222.22)
    _append_synthetic_txns(engine, [older, newer])

    silver.build_all(engine)

    out = engine.sql(
        f"SELECT signed_amount FROM s WHERE id = {txn_id}",
        tables={"s": schemas.SILVER_TRANSACTIONS.name})
    rows = out.to_pylist()
    assert len(rows) == 1
    # DEBIT signs negative; 2030 lastUpdated (222.22) must win over 2020 (111.11).
    assert rows[0]["signed_amount"] == pytest.approx(-222.22)


def test_synthetic_non_usd_row_is_quarantined_with_reason_and_currency(engine):
    """The generator emits USD only, so the quarantine path has never run
    against a real reject before. Inject one EUR row and assert it is
    excluded from silver.transactions and lands in quarantine with the
    expected reason and its original currency preserved."""
    txn_id = 999_999_002
    eur_row = _txn_record(txn_id, "2024-06-01T00:00:00+00:00", 50.0, currency="EUR")
    _append_synthetic_txns(engine, [eur_row])

    silver.build_all(engine)

    kept = engine.sql(
        f"SELECT * FROM s WHERE id = {txn_id}",
        tables={"s": schemas.SILVER_TRANSACTIONS.name})
    assert kept.num_rows == 0

    quarantined = engine.sql(
        f"SELECT currency, quarantine_reason FROM q WHERE id = {txn_id}",
        tables={"q": schemas.SILVER_QUARANTINE.name}).to_pylist()
    assert len(quarantined) == 1
    assert quarantined[0]["currency"] == "EUR"
    assert quarantined[0]["quarantine_reason"] == "non_usd_currency"
