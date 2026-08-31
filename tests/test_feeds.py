"""Per-feed config: it loads, it refuses bad input by name, and it preserves
exactly what the hardcoded Python it replaced used to do.

The last of those is the point. This was a migration, and the risk in a
migration is not that it fails loudly -- it is that it succeeds while quietly
changing a threshold, dropping a monitor, or widening an SLA. So the previous
values are pinned here as literals rather than derived from the same config
they are meant to check.
"""
import pytest

from src import feeds
from src.agent import graph
from src.ops import arrival


def _write(tmp_path, name, text):
    (tmp_path / f"{name}.yaml").write_text(text)
    return tmp_path


MINIMAL = """
name: demo
layers:
  silver: {table: silver.transactions}
monitors: []
"""


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------

def test_the_shipped_feeds_load():
    loaded = feeds.load_feeds()
    assert {f.name for f in loaded} == {
        "yodlee_transactions", "stock_prices", "forecast_training_set"}


def test_declared_monitor_count_is_unchanged_by_the_migration():
    """Ten declared monitors, exactly as monitors/{bronze,silver,gold}.yaml
    held: 2 bronze + 6 silver + 2 gold. The other 7 of the documented 17 are
    generated per contract field on `typed` layers."""
    assert len(feeds.all_monitors()) == 10
    names = {m.name for m in feeds.all_monitors()}
    assert names == {
        "bronze_txn_row_count", "bronze_txn_null_id_rate",
        "silver_txn_row_count", "silver_txn_duplicate_ids",
        "silver_txn_null_merchant_rate", "silver_txn_mean_abs_amount",
        "silver_txn_account_cardinality", "silver_txn_quarantine_rate",
        "gold_training_row_count", "gold_training_null_target_rate"}


def test_monitor_params_survive_the_move_intact():
    """A migration that silently loosened a threshold would still load."""
    by_name = {m.name: m for m in feeds.all_monitors()}
    assert by_name["bronze_txn_row_count"].source == "snapshot_added"
    assert by_name["bronze_txn_row_count"].params == {
        "breach_ratio": 0.5, "warn_ratio": 0.8}
    assert by_name["bronze_txn_null_id_rate"].params == {"max_absolute": 0}
    assert by_name["silver_txn_quarantine_rate"].params == {"max_absolute": 0.01}
    assert by_name["silver_txn_quarantine_rate"].tables == {
        "q": "silver.transactions_quarantine"}
    assert by_name["silver_txn_mean_abs_amount"].params == {
        "breach_z": 4.0, "warn_z": 3.0}
    assert by_name["gold_training_null_target_rate"].params == {
        "max_absolute": 0.05, "breach_z": 4.0}


def test_arrival_slas_match_the_hardcoded_tuple_they_replaced():
    """Pinned as literals, deliberately not derived from feeds/*.yaml -- a
    check that reads its expectation from the thing under test checks nothing.
    `min_rows_per_period` must never be 1 (it would be unsatisfiable) and must
    be None where no structural floor exists."""
    actual = {s.table: s for s in arrival.ARRIVAL_SLAS}
    assert set(actual) == {"silver.transactions", "silver.stock_prices",
                            "gold.forecast_training_set"}

    txn = actual["silver.transactions"]
    assert (txn.date_column, txn.calendar, txn.max_lag_days,
            txn.min_rows_per_period) == ("txn_date", "trading", 3, None)

    prices = actual["silver.stock_prices"]
    assert (prices.date_column, prices.calendar, prices.max_lag_days) == (
        "trade_date", "trading", 3)
    assert prices.min_rows_per_period == 6      # len(config.COMPANIES)

    training = actual["gold.forecast_training_set"]
    assert (training.date_column, training.calendar, training.max_lag_days) == (
        "trade_date", "trading", 5)
    assert training.min_rows_per_period == 6


def test_agent_watch_matches_the_hardcoded_tuple_it_replaced():
    watched = graph.watched_layers()
    assert len(watched) == 1
    table_def, contract, column = watched[0]
    assert table_def.name == "bronze.yodlee_transactions_raw"
    assert contract == "bronze_yodlee_transactions.yaml"
    assert column == "transactionDate"


def test_typed_layers_match_the_replaced_TYPED_TABLES():
    typed = {layer.table: layer.contract
             for feed in feeds.load_feeds() for layer in feed.typed_layers}
    assert typed == {"silver.transactions": "silver_transactions.yaml",
                     "gold.forecast_training_set": "gold_forecast_training_set.yaml"}


# --------------------------------------------------------------------------
# It refuses bad config, and says which key is wrong
# --------------------------------------------------------------------------

def test_a_monitor_on_an_undeclared_layer_is_rejected(tmp_path):
    _write(tmp_path, "bad", """
name: demo
layers:
  silver: {table: silver.transactions}
monitors:
  - {name: m, layer: bronze, kind: row_count, query: SELECT 1 AS metric}
""")
    with pytest.raises(feeds.FeedConfigError, match="layer: bronze"):
        feeds.load_feeds(tmp_path)


def test_a_layer_naming_an_unknown_table_is_rejected(tmp_path):
    _write(tmp_path, "bad", """
name: demo
layers:
  silver: {table: silver.does_not_exist}
""")
    with pytest.raises(feeds.FeedConfigError) as exc:
        feeds.load_feeds(tmp_path)
    assert "silver.does_not_exist" in str(exc.value)
    assert "ALL_TABLES" in str(exc.value)


def test_a_missing_contract_file_is_rejected(tmp_path):
    _write(tmp_path, "bad", """
name: demo
layers:
  silver: {table: silver.transactions, contract: nope.yaml}
""")
    with pytest.raises(feeds.FeedConfigError, match="nope.yaml"):
        feeds.load_feeds(tmp_path)


def test_typed_without_a_contract_is_rejected(tmp_path):
    """The vacuous case: column-type monitors are generated per contract
    field, so `typed: true` with no contract produces zero checks and reads as
    coverage. Refused rather than quietly generating nothing."""
    _write(tmp_path, "bad", """
name: demo
layers:
  silver: {table: silver.transactions, typed: true}
""")
    with pytest.raises(feeds.FeedConfigError) as exc:
        feeds.load_feeds(tmp_path)
    assert "typed" in str(exc.value)
    assert "no type checks at all" in str(exc.value)


def test_an_unknown_calendar_is_rejected(tmp_path):
    _write(tmp_path, "bad", """
name: demo
layers:
  silver: {table: silver.transactions}
arrival:
  - {layer: silver, column: txn_date, calendar: lunar, max_lag_periods: 3}
""")
    with pytest.raises(feeds.FeedConfigError, match="lunar"):
        feeds.load_feeds(tmp_path)


def test_an_unknown_ingest_mode_is_rejected(tmp_path):
    _write(tmp_path, "bad", """
name: demo
ingest: {mode: carrier_pigeon}
layers:
  silver: {table: silver.transactions}
""")
    with pytest.raises(feeds.FeedConfigError, match="carrier_pigeon"):
        feeds.load_feeds(tmp_path)


def test_an_unknown_symbolic_row_floor_is_rejected(tmp_path):
    _write(tmp_path, "bad", """
name: demo
layers:
  silver: {table: silver.transactions}
arrival:
  - {layer: silver, column: txn_date, calendar: trading, max_lag_periods: 3,
     min_rows_per_period: tickers}
""")
    with pytest.raises(feeds.FeedConfigError, match="tickers"):
        feeds.load_feeds(tmp_path)


def test_a_missing_required_key_names_the_key(tmp_path):
    _write(tmp_path, "bad", "name: demo\n")
    with pytest.raises(feeds.FeedConfigError, match="'layers'"):
        feeds.load_feeds(tmp_path)


def test_duplicate_monitor_names_across_feeds_are_rejected(tmp_path):
    """Monitor names are the join key between a definition and its persisted
    baseline history (`load_baselines` filters on `monitor` alone). Two feeds
    sharing one would interleave two different measurements into one baseline
    series -- an anomaly that would never reproduce."""
    body = """
name: {name}
layers:
  silver: {{table: silver.transactions}}
monitors:
  - {{name: shared, layer: silver, kind: row_count, query: SELECT 1 AS metric}}
"""
    _write(tmp_path, "a", body.format(name="feed_a"))
    _write(tmp_path, "b", body.format(name="feed_b"))
    with pytest.raises(feeds.FeedConfigError) as exc:
        feeds.load_feeds(tmp_path)
    assert "shared" in str(exc.value)
    assert "feed_a" in str(exc.value) and "feed_b" in str(exc.value)


# --------------------------------------------------------------------------
# The actual claim: one file, no Python
# --------------------------------------------------------------------------

def test_a_new_feed_is_one_yaml_file_and_zero_python_edits(tmp_path):
    """The whole point of the migration. A feed that declares a monitor, an
    arrival SLA, an agent watch and a typed layer is picked up entirely from
    its file -- nothing in `graph.py`, `arrival.py` or `runner.py` names it."""
    _write(tmp_path, "newcomer", """
name: newcomer
domain: financial
feed_type: event
ingest: {mode: stream}
layers:
  bronze:
    table: bronze.yodlee_accounts_raw
    contract: bronze_yodlee_transactions.yaml
    freshness_column: transactionDate
    agent_watch: true
  silver:
    table: silver.accounts
    contract: silver_transactions.yaml
    typed: true
monitors:
  - name: newcomer_row_count
    layer: silver
    kind: row_count
    params: {breach_ratio: 0.5}
    query: SELECT count(*) AS metric FROM t
arrival:
  - {layer: silver, column: txn_date, calendar: daily, max_lag_periods: 2}
""")
    loaded = feeds.load_feeds(tmp_path)
    assert len(loaded) == 1
    feed = loaded[0]

    assert feed.name == "newcomer"
    assert feed.feed_type == "event"
    assert feed.ingest_mode == "stream"
    assert [m.name for m in feed.monitors] == ["newcomer_row_count"]
    assert feed.monitors[0].table == "silver.accounts"
    assert [layer.table for layer in feed.watched_layers] == [
        "bronze.yodlee_accounts_raw"]
    assert [layer.table for layer in feed.typed_layers] == ["silver.accounts"]
    assert feed.arrival[0].table == "silver.accounts"
    assert feed.arrival[0].calendar == "daily"


def test_a_stream_feed_is_accepted_today_even_though_nothing_streams_yet():
    """`ingest.mode: stream` validates now so the streaming work (#4) adds a
    reader, not a config schema change."""
    assert "stream" in feeds.VALID_INGEST_MODES
