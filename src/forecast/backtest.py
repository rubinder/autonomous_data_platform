"""Metrics and a deliberately plain long/short backtest.

No transaction costs, no slippage, no capacity model. The Sharpe reported
here is an upper bound on a frictionless strategy, and the report labels it
as such rather than implying it is achievable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A win must clear a margin, not just come in lower by a rounding error.
RMSE_MARGIN = 0.02   # 2% relative improvement
IC_MARGIN = 0.02


def metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    data = df[["y_true", pred_col]].dropna()
    err = data["y_true"] - data[pred_col]

    # A constant prediction carries no directional information. Scoring the
    # zero baseline as 100% correct on flat days would flatter it enormously.
    moved = data[data[pred_col] != 0]
    direction = (float((np.sign(moved[pred_col]) == np.sign(moved["y_true"])).mean())
                 if len(moved) else float("nan"))

    ic = (float(data[pred_col].corr(data["y_true"], method="spearman"))
          if data[pred_col].nunique() > 1 else float("nan"))

    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "directional_accuracy": direction,
        "ic": 0.0 if np.isnan(ic) else ic,
    }


def long_short_sharpe(df: pd.DataFrame, pred_col: str, top_n: int = 2) -> float:
    """Long the top_n predicted tickers, short the bottom_n, daily rebalance."""
    daily = []
    for _, day in df.groupby("trade_date"):
        if len(day) < top_n * 2:
            continue
        ranked = day.sort_values(pred_col, ascending=False)
        longs = ranked.head(top_n)["y_true"].mean()
        shorts = ranked.tail(top_n)["y_true"].mean()
        daily.append((longs - shorts) / 2)
    if len(daily) < 20:
        return float("nan")
    series = pd.Series(daily)
    if series.std(ddof=1) == 0:
        return float("nan")
    # Overlapping 5-day targets sampled daily -> scale by trading days / horizon.
    return float(series.mean() / series.std(ddof=1) * np.sqrt(252 / 5))


def verdict(model: dict[str, float], baseline: dict[str, float]) -> str:
    rmse_gain = (baseline["rmse"] - model["rmse"]) / baseline["rmse"]
    ic_gain = model["ic"] - baseline["ic"]
    if rmse_gain > RMSE_MARGIN and ic_gain > IC_MARGIN:
        return "beats"
    if rmse_gain < -RMSE_MARGIN or ic_gain < -IC_MARGIN:
        return "loses"
    return "inconclusive"
