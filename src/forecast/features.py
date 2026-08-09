"""Feature contract, plus an independent recomputation used only by tests.

recompute_row deliberately does NOT share code with the Gold SQL. Two
implementations that agree are evidence; one implementation checked against
itself is a tautology.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "spend_mom_7d", "spend_z_28d", "txn_count_z_28d", "avg_ticket_delta_7d",
    "unique_acct_growth_7d", "spend_surprise",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
    "realized_vol_21d", "rsi_14", "volume_z_21d",
)
TARGET = "fwd_ret_5d"


def recompute_row(spend_hist: pd.DataFrame, price_hist: pd.DataFrame,
                  ticker: str, as_of: date) -> dict[str, float]:
    """Recompute every feature for (ticker, as_of) from history <= as_of only."""
    s = spend_hist.sort_values("spend_date").reset_index(drop=True)
    p = price_hist.sort_values("trade_date").reset_index(drop=True)
    out: dict[str, float] = {}

    gs = s["gross_spend"]
    out["spend_mom_7d"] = _safe(gs.tail(7).mean(), gs.tail(28).mean(), ratio=True)
    out["spend_z_28d"] = _z(gs, 28)
    out["txn_count_z_28d"] = _z(s["txn_count"], 28)
    out["avg_ticket_delta_7d"] = _safe(
        s["avg_ticket"].tail(7).mean(), s["avg_ticket"].tail(28).mean(), ratio=True)

    acct_ma7 = s["unique_accounts"].rolling(7).mean()
    out["unique_acct_growth_7d"] = _safe(
        acct_ma7.iloc[-1] if len(acct_ma7) else np.nan,
        acct_ma7.iloc[-8] if len(acct_ma7) > 7 else np.nan, ratio=True)

    same_dow = s[s["spend_date"].apply(lambda d: d.weekday()) == as_of.weekday()]
    prior = same_dow[same_dow["spend_date"] < as_of]["gross_spend"].tail(4)
    sigma = gs.tail(28).std(ddof=1)
    out["spend_surprise"] = _safe(
        (gs.iloc[-1] - prior.mean()) if len(prior) else np.nan, sigma)

    # ret.iloc[-1] is the return *of* as_of -- lag 0 -- so lag k is iloc[-(k+1)].
    # gold.STOCK_FEATURES_SQL defines ret_lag_k as lag(ret_1d, k), the return of
    # day t-k, and keeps the unlagged one in its own column, `ret_1d`. Indexing
    # from -k instead would make ret_lag_1 today's return, colliding with that
    # column and shifting the whole family one day fresher. Verified against the
    # mart: stored ret_lag_k equals ret_1d(t-k) to 1e-16 for every sampled row.
    ret = p["close"].pct_change()
    for k in range(1, 6):
        out[f"ret_lag_{k}"] = ret.iloc[-(k + 1)] if len(ret) > k + 1 else np.nan
    out["realized_vol_21d"] = ret.tail(21).std(ddof=1) * np.sqrt(252)

    gain = ret.clip(lower=0).tail(14).mean()
    loss = (-ret.clip(upper=0)).tail(14).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss) if loss and loss > 0 else np.nan

    vol = p["volume"].astype(float)
    out["volume_z_21d"] = _z(vol, 21)
    return out


def _z(series: pd.Series, window: int) -> float:
    tail = series.tail(window)
    std = tail.std(ddof=1)
    if not std or np.isnan(std) or std == 0:
        return np.nan
    return (series.iloc[-1] - tail.mean()) / std


def _safe(num, den, ratio: bool = False) -> float:
    if den is None or np.isnan(den) or den == 0 or num is None or np.isnan(num):
        return np.nan
    return num / den - 1 if ratio else num / den
