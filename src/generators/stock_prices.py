"""Daily OHLCV whose returns respond to lagged demand shocks.

Synthetic by design (ADR-0004): pairing real market prices with synthetic
spend would guarantee zero correlation and a meaningless model. Here the
signal is planted at a known per-ticker lag, so "did the pipeline preserve
it" becomes a testable question.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.generators.demand import build_demand_panel


def generate_prices(seed: int = config.SEED) -> list[dict]:
    days = config.trading_days(config.START_DATE, config.END_DATE)
    n = len(days)
    panel = build_demand_panel(seed=seed)
    rows: list[dict] = []

    for idx, company in enumerate(config.COMPANIES):
        rng = np.random.default_rng(seed * 7919 + idx)
        shock = (panel.loc[panel["ticker"] == company.ticker]
                 .sort_values("day")["shock"].to_numpy(dtype=float))

        lagged = np.zeros(n)
        if company.lag_days < n:
            lagged[company.lag_days:] = shock[: n - company.lag_days]

        daily_drift = company.annual_drift / 252.0
        daily_vol = company.annual_vol / np.sqrt(252.0)

        # Scale idiosyncratic noise so the planted beta stays recoverable.
        noise = rng.standard_normal(n) * daily_vol
        returns = daily_drift + company.beta * daily_vol * lagged + noise

        close = company.start_price * np.exp(np.cumsum(returns - 0.5 * daily_vol**2))

        prev_close = np.concatenate(([company.start_price], close[:-1]))
        open_ = prev_close * (1 + rng.normal(0, daily_vol * 0.3, n))
        span = np.abs(rng.normal(0, daily_vol * 0.6, n))
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)
        volume = rng.integers(1_500_000, 12_000_000, n)

        for t, day in enumerate(days):
            rows.append({
                "ticker": company.ticker,
                "date": day.isoformat(),
                "open": round(float(open_[t]), 4),
                "high": round(float(high[t]), 4),
                "low": round(float(low[t]), 4),
                "close": round(float(close[t]), 4),
                "adj_close": round(float(close[t]), 4),
                "volume": int(volume[t]),
            })

    rows.sort(key=lambda r: (r["date"], r["ticker"]))
    return rows
