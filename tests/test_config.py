from datetime import date

from src import config


def test_six_companies_with_distinct_planted_parameters():
    tickers = [c.ticker for c in config.COMPANIES]
    assert tickers == ["SBUX", "CMG", "TGT", "LULU", "DPZ", "ULTA"]
    # Heterogeneous lags are deliberate: a single global rule must not fit all six.
    assert len({c.lag_days for c in config.COMPANIES}) >= 4


def test_trading_days_excludes_weekends_and_market_holidays():
    days = config.trading_days(date(2024, 1, 1), date(2024, 1, 31))
    assert date(2024, 1, 1) not in days      # New Year's Day
    assert date(2024, 1, 15) not in days     # MLK Day
    assert date(2024, 1, 6) not in days      # Saturday
    assert date(2024, 1, 2) in days
    assert days == sorted(days)


def test_full_window_has_enough_days_for_purged_cv():
    days = config.trading_days(config.START_DATE, config.END_DATE)
    assert 600 <= len(days) <= 650


def test_as_of_date_prefers_bronze_max_over_wall_clock(monkeypatch):
    monkeypatch.delenv("AS_OF_DATE", raising=False)
    assert config.resolve_as_of_date(date(2026, 6, 30)) == date(2026, 6, 30)


def test_as_of_date_env_override_wins():
    import os
    os.environ["AS_OF_DATE"] = "2026-07-15"
    try:
        assert config.resolve_as_of_date(date(2026, 6, 30)) == date(2026, 7, 15)
    finally:
        del os.environ["AS_OF_DATE"]
