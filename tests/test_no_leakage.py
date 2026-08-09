"""The load-bearing tests: every feature at t must be computable from data <= t.

`features.recompute_row` is a second, independent implementation that only ever
sees a truncated history. Agreement between it and the Gold SQL to 1e-6 is the
evidence that no future information reaches a feature column.
"""
import pandas as pd
import pytest

from src import config
from src.forecast import features
from src.lakehouse import bronze, gold, schemas, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=60000)
    silver.build_all(eng)
    gold.build_all(eng)
    training = eng.scan_arrow(schemas.GOLD_TRAINING.name).to_pandas()
    spend = eng.scan_arrow(schemas.GOLD_SPEND.name).to_pandas()
    prices = eng.scan_arrow(schemas.SILVER_PRICES.name).to_pandas()
    return training, spend, prices


def test_every_feature_recomputes_from_data_at_or_before_t(built):
    """The core guarantee. Recompute each feature using a truncated history."""
    training, spend, prices = built
    sample = training.dropna(subset=list(features.FEATURE_COLUMNS)).sample(
        30, random_state=0)

    for _, row in sample.iterrows():
        as_of = row["trade_date"]
        ticker = row["ticker"]
        spend_hist = spend[(spend["ticker"] == ticker) & (spend["spend_date"] <= as_of)]
        price_hist = prices[(prices["ticker"] == ticker) & (prices["trade_date"] <= as_of)]
        recomputed = features.recompute_row(spend_hist, price_hist, ticker, as_of)
        for col in features.FEATURE_COLUMNS:
            expected, actual = recomputed[col], row[col]
            if pd.isna(expected) and pd.isna(actual):
                continue
            assert abs(expected - actual) < 1e-6, (
                f"{ticker} {as_of} {col}: stored {actual} != recomputed-from-past {expected}")


def test_target_looks_forward_exactly_five_trading_days(built):
    training, _, prices = built
    px = prices[prices["ticker"] == "CMG"].sort_values("trade_date").reset_index(drop=True)
    tr = training[training["ticker"] == "CMG"].sort_values("trade_date").reset_index(drop=True)
    merged = tr.merge(px[["trade_date", "adj_close"]], on="trade_date")
    for i in range(len(merged) - 6):
        if pd.isna(merged.loc[i, "fwd_ret_5d"]):
            continue
        expected = merged.loc[i + 5, "adj_close"] / merged.loc[i, "adj_close"] - 1
        assert abs(merged.loc[i, "fwd_ret_5d"] - expected) < 1e-9
        break
    else:
        pytest.fail("no comparable row found")


def test_last_five_rows_per_ticker_have_null_target(built):
    training, _, _ = built
    for _ticker, grp in training.groupby("ticker"):
        tail = grp.sort_values("trade_date").tail(5)
        assert tail["fwd_ret_5d"].isna().all()


def test_no_feature_correlates_perfectly_with_target(built):
    """A near-1.0 correlation means the target leaked into a feature."""
    training, _, _ = built
    clean = training.dropna(subset=[features.TARGET])
    for col in features.FEATURE_COLUMNS:
        series = clean[[col, features.TARGET]].dropna()
        if len(series) < 50:
            continue
        corr = abs(series[col].corr(series[features.TARGET]))
        assert corr < 0.90, f"{col} correlates {corr:.3f} with target -- leak"


def test_training_set_has_all_tickers_and_enough_rows(built):
    training, _, _ = built
    assert set(training["ticker"]) == {c.ticker for c in config.COMPANIES}
    assert len(training) > 3000
