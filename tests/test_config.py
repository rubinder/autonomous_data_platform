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


def test_memorial_day_2026_is_not_a_trading_day():
    """Both README and ADR-0007 cite this case as verified. Now it is."""
    days = config.trading_days(date(2026, 5, 18), date(2026, 5, 29))
    assert date(2026, 5, 25) not in days     # Memorial Day, a Monday
    assert date(2026, 5, 26) in days


def test_calendar_covers_holidays_past_end_date():
    """The calendar is consulted beyond the data, so it must extend beyond it.

    `AS_OF_DATE` can be advanced past END_DATE (2026-06-30) to reproduce a
    stale feed, and the arrival checks then expect trading days up to `as_of`.
    A holiday set that stops at END_DATE turns every later holiday into a
    reported missing trading day -- at the documented AS_OF_DATE=2026-07-31
    that misreported 2026-07-03 and inflated the headline gap count.
    """
    assert config.END_DATE == date(2026, 6, 30)
    beyond = {
        date(2026, 7, 3): "Independence Day observed (4 July is a Saturday)",
        date(2026, 9, 7): "Labor Day",
        date(2026, 11, 26): "Thanksgiving",
        date(2026, 12, 25): "Christmas Day",
    }
    for day, label in beyond.items():
        assert day > config.END_DATE
        assert day.weekday() < 5, f"{label} must be a weekday to matter"
        assert day in config.US_MARKET_HOLIDAYS, label
        assert day not in config.trading_days(date(2026, 7, 1), date(2026, 12, 31)), label


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
