import pandas as pd

from src import config
from src.generators import stock_prices
from src.generators.demand import build_demand_panel


def test_covers_all_tickers_and_trading_days():
    rows = stock_prices.generate_prices()
    df = pd.DataFrame(rows)
    days = config.trading_days(config.START_DATE, config.END_DATE)
    assert set(df["ticker"]) == {c.ticker for c in config.COMPANIES}
    assert len(df) == len(days) * len(config.COMPANIES)


def test_ohlc_invariants_hold():
    df = pd.DataFrame(stock_prices.generate_prices())
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()


def test_planted_signal_is_recoverable_at_the_configured_lag():
    """The whole project rests on this. If it fails, nothing downstream means anything."""
    df = pd.DataFrame(stock_prices.generate_prices())
    panel = build_demand_panel()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    for company in config.COMPANIES:
        px = df[df["ticker"] == company.ticker].sort_values("date").reset_index(drop=True)
        dm = panel[panel["ticker"] == company.ticker].sort_values("day").reset_index(drop=True)
        ret = px["close"].pct_change()
        lagged_shock = dm["shock"].shift(company.lag_days)
        joined = pd.DataFrame({"ret": ret, "shock": lagged_shock}).dropna()
        corr = joined["ret"].corr(joined["shock"])
        assert corr > 0.15, f"{company.ticker} planted signal not recoverable: {corr:.3f}"


def test_no_signal_at_a_wrong_lag_for_most_tickers():
    """Guards against a generator bug that smears signal across all lags."""
    df = pd.DataFrame(stock_prices.generate_prices())
    panel = build_demand_panel()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    weak = 0
    for company in config.COMPANIES:
        px = df[df["ticker"] == company.ticker].sort_values("date").reset_index(drop=True)
        dm = panel[panel["ticker"] == company.ticker].sort_values("day").reset_index(drop=True)
        ret = px["close"].pct_change()
        wrong = dm["shock"].shift(company.lag_days + 9)
        joined = pd.DataFrame({"ret": ret, "shock": wrong}).dropna()
        if abs(joined["ret"].corr(joined["shock"])) < 0.10:
            weak += 1
    assert weak >= 4


def test_deterministic():
    assert stock_prices.generate_prices(seed=3) == stock_prices.generate_prices(seed=3)
