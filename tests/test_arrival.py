from datetime import date

import pytest

from src import config
from src.lakehouse import bronze, silver
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import arrival


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=30000)
    silver.build_all(eng)
    return eng


def test_missing_periods_finds_gaps_in_the_middle():
    expected = [date(2026, 6, d) for d in (1, 2, 3, 4, 5)]
    observed = {date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 5)}
    assert arrival.missing_periods(observed, expected) == [
        date(2026, 6, 3), date(2026, 6, 4)]


def test_missing_periods_empty_when_complete():
    expected = [date(2026, 6, d) for d in (1, 2, 3)]
    assert arrival.missing_periods(set(expected), expected) == []


def test_weekend_is_not_reported_missing_for_a_trading_calendar():
    """Saturday absence is normal. Reporting it is how alerting loses trust."""
    expected = config.trading_days(date(2026, 6, 1), date(2026, 6, 12))
    assert date(2026, 6, 6) not in expected      # Saturday
    assert arrival.missing_periods(set(expected), expected) == []


def test_arrival_check_passes_on_a_complete_feed(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 6, 30))
    assert results
    assert all(r.status == "ok" for r in results), [
        r.detail for r in results if r.status != "ok"]


def test_arrival_check_breaches_when_as_of_runs_ahead_of_the_data(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 7, 31))
    assert any(r.status == "breach" and r.kind == "arrival_lag" for r in results)


def test_arrival_reports_the_specific_missing_dates(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 7, 31))
    gap = next(r for r in results if r.kind == "arrival_gap")
    assert gap.metric is not None
    assert gap.metric > 0
    assert "2026-07" in gap.detail
