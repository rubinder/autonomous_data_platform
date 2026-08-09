"""Metrics and a deliberately plain long/short backtest.

No transaction costs, no slippage, no capacity model. The Sharpe reported
here is an upper bound on a frictionless strategy, and the report labels it
as such rather than implying it is achievable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# IC must clear this margin over baseline, and the model's own IC must be
# statistically significant, to call a "beat". RMSE is a non-degradation
# guard, not a win condition -- see verdict()'s docstring for why: at this
# feature set's measured correlation (Task 10: max |rho| ~0.156), the
# achievable RMSE ratio is sqrt(1-rho^2) ~ 0.988, so a 2% *required* RMSE
# improvement is an unreachable bar that would silence the leakage tripwire
# (a clean sweep of "beats" could never fire, so it could never warn).
IC_MARGIN = 0.02
RMSE_TOLERANCE = 0.02   # max relative RMSE degradation tolerated
SIGNIFICANCE_T = 2.0    # ~95% two-tailed threshold for the IC t-test


def metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    data = df[["y_true", pred_col]].dropna()
    err = data["y_true"] - data[pred_col]

    # A constant column -- all-zero, or any other single repeated value, such
    # as the ARIMA fallback's training-mean constant -- carries no
    # directional information. Scoring a constant predictor by how often the
    # market happened to move the way the constant points would credit it
    # with skill it doesn't have. Same nunique() test as the IC branch below,
    # deliberately: both statistics are undefined for a non-varying signal.
    if data[pred_col].nunique() > 1:
        direction = float((np.sign(data[pred_col]) == np.sign(data["y_true"])).mean())
    else:
        direction = float("nan")

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


def _ic_significant(ic: float, n: int, horizon: int = config.FORECAST_HORIZON_DAYS) -> bool:
    """t-test on the model's own IC, using overlap-adjusted effective sample size.

    Consecutive rows share up to horizon-1 days of their forward-return
    window, so they are not independent draws -- treating n raw rows as n
    independent observations would understate the standard error and call
    noise "significant". Dividing by horizon is a conservative correction:
    n_eff observations, each drawn roughly horizon days apart, approximates
    independence.
    """
    n_eff = n / horizon
    if n_eff <= 2 or abs(ic) >= 1:
        return False
    t = ic * np.sqrt(n_eff - 2) / np.sqrt(1 - ic ** 2)
    return bool(abs(t) > SIGNIFICANCE_T)


def verdict(model: dict[str, float], baseline: dict[str, float], n: int) -> str:
    """Compare model to baseline under a margin AND a significance test.

    RMSE is deliberately NOT a win condition. Persistence RMSE approximates
    sigma_y, so the best achievable ratio at correlation rho is
    sqrt(1-rho^2) -- at this project's measured feature correlations
    (Task 10: max |rho| ~0.156), that ratio is ~0.988, well short of a 2%
    *required* improvement. Demanding it anyway would make "beats"
    unreachable regardless of real skill, which would silence the leakage
    tripwire this function exists to serve (Step 6: a clean sweep of
    "beats" is the signal to stop and check for a leak -- a bar nothing can
    clear can never fire it). RMSE is instead a non-degradation guard: a
    model can beat on IC while its RMSE is roughly flat, but not while its
    RMSE is meaningfully worse than the baseline's.

    A win also requires the model's own IC to be positive AND statistically
    significant, not merely an improvement over the baseline's IC -- an
    anti-skilled baseline (e.g. a diverging ARIMA fit) can have a very
    negative IC, and "less negative than a broken baseline" is not a model
    that adds value.
    """
    ic_gain = model["ic"] - baseline["ic"]
    if baseline["rmse"] > 0:
        rmse_gain = (baseline["rmse"] - model["rmse"]) / baseline["rmse"]
    else:
        # A zero-RMSE baseline predicted every target exactly -- there is no
        # relative improvement to measure. Treat "model also perfect" as
        # neutral and anything short of that as a (large) degradation,
        # rather than dividing by zero.
        rmse_gain = 0.0 if model["rmse"] == 0 else -1.0

    if (ic_gain > IC_MARGIN and model["ic"] > 0
            and _ic_significant(model["ic"], n) and rmse_gain > -RMSE_TOLERANCE):
        return "beats"
    if ic_gain < -IC_MARGIN or rmse_gain < -RMSE_TOLERANCE:
        return "loses"
    return "inconclusive"
