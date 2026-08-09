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


def test_memorial_day_is_not_reported_missing():
    """Cited as verified in both README and ADR-0007; nothing pinned it."""
    expected = config.trading_days(date(2026, 5, 18), date(2026, 5, 29))
    assert date(2026, 5, 25) not in expected
    assert arrival.missing_periods(set(expected), expected) == []


def test_holiday_past_end_date_is_not_reported_as_a_gap(engine):
    """The bug: the holiday calendar ended before the as-of window did.

    At the documented AS_OF_DATE=2026-07-31 the gap window runs to 2026-07-31,
    past END_DATE (2026-06-30). With the calendar stopping at END_DATE,
    2026-07-03 -- NYSE-observed Independence Day, because 4 July 2026 is a
    Saturday -- was listed as a missing trading day, three lines above the
    docs' claim that holidays are correctly not gaps.
    """
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 7, 31))
    gap = next(r for r in results if r.kind == "arrival_gap")

    window = arrival.expected_periods("trading", date(2026, 7, 1), date(2026, 7, 31))
    assert date(2026, 7, 3) not in window          # holiday
    assert date(2026, 7, 4) not in window          # Saturday
    assert date(2026, 7, 2) in window and date(2026, 7, 6) in window
    assert "2026-07-03" not in gap.detail


def test_partial_check_is_absent_where_it_could_never_fire():
    """`min_rows_per_period=1` against a count(*) GROUP BY is unsatisfiable.

    A check that cannot fire must not be displayed as a passing check. Where
    there is no structural floor the SLA declares None and emits nothing.
    """
    for sla in arrival.ARRIVAL_SLAS:
        assert sla.min_rows_per_period is None or sla.min_rows_per_period > 1


def test_no_partial_result_for_an_sla_without_a_floor(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.transactions")
    assert sla.min_rows_per_period is None
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 6, 30))
    assert not any(r.kind == "arrival_partial" for r in results)


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
