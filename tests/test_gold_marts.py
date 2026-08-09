from itertools import pairwise

import pytest

from src import config
from src.lakehouse import bronze, gold, schemas, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=40000)
    silver.build_all(eng)
    gold.build_all(eng)
    return eng


def test_spend_grain_is_ticker_by_date(engine):
    out = engine.sql("SELECT count(*) AS n, count(DISTINCT (ticker, spend_date)) AS d FROM g",
                     tables={"g": schemas.GOLD_SPEND.name}).to_pylist()[0]
    assert out["n"] == out["d"]


def test_spend_is_positive_despite_signed_amounts(engine):
    """signed_amount is negative for DEBIT; gross_spend must be reported positive."""
    out = engine.sql("SELECT min(gross_spend) AS lo FROM g",
                     tables={"g": schemas.GOLD_SPEND.name}).to_pylist()[0]
    assert out["lo"] > 0


def test_only_tracked_tickers_appear(engine):
    tickers = {r["ticker"] for r in engine.sql(
        "SELECT DISTINCT ticker FROM g", tables={"g": schemas.GOLD_SPEND.name}).to_pylist()}
    assert tickers == {c.ticker for c in config.COMPANIES}


def test_noise_merchants_excluded_from_spend(engine):
    """~60% of transactions are noise; spend must not include them."""
    spend_txns = engine.sql("SELECT sum(txn_count) AS n FROM g",
                            tables={"g": schemas.GOLD_SPEND.name}).to_pylist()[0]["n"]
    all_txns = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).num_rows
    assert 0.30 < spend_txns / all_txns < 0.50


def test_stock_features_lags_are_backward_looking(engine):
    rows = engine.sql("""
        SELECT trade_date, ret_1d, ret_lag_1 FROM f
        WHERE ticker = 'SBUX' ORDER BY trade_date
    """, tables={"f": schemas.GOLD_STOCK_FEATURES.name}).to_pylist()
    for prev, cur in pairwise(rows[1:]):
        if prev["ret_1d"] is not None and cur["ret_lag_1"] is not None:
            assert abs(cur["ret_lag_1"] - prev["ret_1d"]) < 1e-9


def test_rsi_bounded(engine):
    out = engine.sql("SELECT min(rsi_14) AS lo, max(rsi_14) AS hi FROM f WHERE rsi_14 IS NOT NULL",
                     tables={"f": schemas.GOLD_STOCK_FEATURES.name}).to_pylist()[0]
    assert 0 <= out["lo"] and out["hi"] <= 100


def test_realized_vol_non_negative(engine):
    out = engine.sql("SELECT min(realized_vol_21d) AS lo FROM f WHERE realized_vol_21d IS NOT NULL",
                     tables={"f": schemas.GOLD_STOCK_FEATURES.name}).to_pylist()[0]
    assert out["lo"] >= 0
