from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.forecast import backtest


def _frame(n=200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n) * 0.02
    return pd.DataFrame({
        "ticker": ["AAA"] * n,
        "trade_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "y_true": y,
        "y_pred_good": y + rng.standard_normal(n) * noise * 0.02,
        "y_pred_bad": rng.standard_normal(n) * 0.02,
        "y_pred_zero": np.zeros(n),
    })


def test_metrics_keys_and_ranges():
    m = backtest.metrics(_frame(), "y_pred_good")
    assert set(m) == {"rmse", "mae", "directional_accuracy", "ic"}
    assert m["rmse"] >= 0
    assert 0 <= m["directional_accuracy"] <= 1
    assert -1 <= m["ic"] <= 1


def test_good_predictions_score_better_than_random():
    good = backtest.metrics(_frame(), "y_pred_good")
    bad = backtest.metrics(_frame(), "y_pred_bad")
    assert good["rmse"] < bad["rmse"]
    assert good["ic"] > bad["ic"]


def test_directional_accuracy_ignores_zero_predictions():
    """An all-zero baseline has no direction; scoring it 100% would be a lie."""
    m = backtest.metrics(_frame(), "y_pred_zero")
    assert np.isnan(m["directional_accuracy"])


def test_directional_accuracy_ignores_any_constant_predictor():
    """A constant *nonzero* predictor (e.g. ARIMA's training-mean fallback)
    carries the same zero directional information as an all-zero one -- it
    always points the same way, so its "accuracy" is just the base rate of
    up days, not skill.
    """
    f = _frame()
    f["y_pred_constant"] = 0.0123
    m = backtest.metrics(f, "y_pred_constant")
    assert np.isnan(m["directional_accuracy"])


def test_sharpe_positive_for_informative_predictions():
    frames = []
    for i, ticker in enumerate("ABCDEF"):
        f = _frame(seed=i)
        f["ticker"] = ticker
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    assert backtest.long_short_sharpe(df, "y_pred_good") > 0


def test_verdict_requires_a_margin_not_just_a_smaller_number():
    base = {"rmse": 0.0200, "ic": 0.00, "directional_accuracy": 0.50, "mae": 0.01}
    tie = {"rmse": 0.0199, "ic": 0.01, "directional_accuracy": 0.50, "mae": 0.01}
    win = {"rmse": 0.0150, "ic": 0.20, "directional_accuracy": 0.58, "mae": 0.01}
    loss = {"rmse": 0.0300, "ic": -0.05, "directional_accuracy": 0.47, "mae": 0.02}
    # n=600 at the default 5-day horizon -> n_eff=120, comfortably enough to
    # make win's IC=0.20 statistically significant (t ~ 2.2).
    assert backtest.verdict(tie, base, n=600) == "inconclusive"
    assert backtest.verdict(win, base, n=600) == "beats"
    assert backtest.verdict(loss, base, n=600) == "loses"


def test_verdict_requires_statistical_significance_not_just_margin():
    """The same IC that wins with plenty of data is noise with too little."""
    base = {"rmse": 0.0200, "ic": 0.00, "directional_accuracy": 0.50, "mae": 0.01}
    win = {"rmse": 0.0150, "ic": 0.20, "directional_accuracy": 0.58, "mae": 0.01}
    assert backtest.verdict(win, base, n=40) == "inconclusive"


def test_verdict_does_not_beat_a_worse_baseline_with_negative_ic():
    """A model with negative IC cannot 'beat' merely because the baseline
    (e.g. a diverging ARIMA fit) is even more negative -- see Task 13 review
    finding 4 (ULTA model IC -0.007 vs ARIMA IC -0.118).
    """
    broken_baseline = {"rmse": 0.0800, "ic": -0.30, "directional_accuracy": 0.40, "mae": 0.05}
    weak_model = {"rmse": 0.0600, "ic": -0.02, "directional_accuracy": 0.48, "mae": 0.04}
    assert backtest.verdict(weak_model, broken_baseline, n=600) != "beats"


def test_verdict_handles_zero_baseline_rmse_without_crashing():
    perfect_baseline = {"rmse": 0.0, "ic": 0.5, "directional_accuracy": 1.0, "mae": 0.0}
    imperfect_model = {"rmse": 0.01, "ic": 0.3, "directional_accuracy": 0.7, "mae": 0.01}
    assert backtest.verdict(imperfect_model, perfect_baseline, n=600) == "loses"
