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
    """The core guarantee: EVERY row, EVERY feature, BOTH directions.

    Not a sample, and not the dropna'd frame. Sampling 30 rows out of
    `dropna(subset=FEATURE_COLUMNS)` is doubly blind: it misses most rows, and
    it structurally cannot see a row where the SQL stored a value the
    recomputation says is undefined -- which is the exact shape of a
    partial-window warm-up bug. So null-ness is asserted as an equality too:
    where one side is null the other must be null, and where both are present
    they must agree to 1e-6.
    """
    training, spend, prices = built
    spend_by = {t: g for t, g in spend.groupby("ticker")}
    prices_by = {t: g for t, g in prices.groupby("ticker")}
    checked = 0
    worst = 0.0

    for _, row in training.iterrows():
        as_of = row["trade_date"]
        ticker = row["ticker"]
        sh = spend_by[ticker]
        ph = prices_by[ticker]
        spend_hist = sh[sh["spend_date"] <= as_of]
        price_hist = ph[ph["trade_date"] <= as_of]
        recomputed = features.recompute_row(spend_hist, price_hist, ticker, as_of)
        for col in features.FEATURE_COLUMNS:
            expected, actual = recomputed[col], row[col]
            checked += 1
            assert pd.isna(expected) == pd.isna(actual), (
                f"{ticker} {as_of} {col}: stored {actual} is null-mismatched against "
                f"recomputed-from-past {expected}")
            if pd.isna(expected):
                continue
            worst = max(worst, abs(expected - actual))
            assert abs(expected - actual) < 1e-6, (
                f"{ticker} {as_of} {col}: stored {actual} != recomputed-from-past {expected}")

    assert checked == len(training) * len(features.FEATURE_COLUMNS)
    assert worst < 1e-6, f"largest discrepancy {worst:.3e}"


def test_recompute_row_rejects_history_reaching_past_as_of(built):
    """The truncation contract is the caller's job, so recompute_row polices it."""
    _, spend, prices = built
    as_of = sorted(prices[prices["ticker"] == "CMG"]["trade_date"])[100]
    sh = spend[(spend["ticker"] == "CMG") & (spend["spend_date"] <= as_of)]
    ph = prices[prices["ticker"] == "CMG"]  # NOT truncated: reaches into the future

    with pytest.raises(ValueError, match="looks past as_of"):
        features.recompute_row(sh, ph, "CMG", as_of)


def test_recompute_row_rejects_another_tickers_history(built):
    _, spend, prices = built
    as_of = sorted(prices[prices["ticker"] == "CMG"]["trade_date"])[100]
    sh = spend[spend["spend_date"] <= as_of]  # every ticker, not just CMG
    ph = prices[(prices["ticker"] == "CMG") & (prices["trade_date"] <= as_of)]

    with pytest.raises(ValueError, match="contains other tickers"):
        features.recompute_row(sh, ph, "CMG", as_of)


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
