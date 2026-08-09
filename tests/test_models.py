from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.forecast import features, models


def _synthetic(n: int = 500, ticker: str = "AAA", signal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({"ticker": ticker, "trade_date": dates})
    for col in features.FEATURE_COLUMNS:
        df[col] = rng.standard_normal(n)
    base = 0.6 * df["spend_mom_7d"] if signal else 0.0
    df[features.TARGET] = base + rng.standard_normal(n) * 0.4
    return df


def test_returns_one_row_per_test_observation():
    out = models.fit_predict_ticker(_synthetic(), n_folds=5)
    assert len(out) > 0
    assert set(out.columns) >= {
        "ticker", "trade_date", "fold", "y_true",
        "y_pred_model", "y_pred_persistence", "y_pred_arima"}
    assert out["fold"].nunique() == 5


def test_predictions_are_finite():
    out = models.fit_predict_ticker(_synthetic())
    for col in ("y_pred_model", "y_pred_persistence", "y_pred_arima"):
        assert np.isfinite(out[col]).all()


def test_beats_persistence_when_signal_is_present():
    out = models.fit_predict_ticker(_synthetic(signal=True))
    model_rmse = np.sqrt(((out["y_true"] - out["y_pred_model"]) ** 2).mean())
    base_rmse = np.sqrt(((out["y_true"] - out["y_pred_persistence"]) ** 2).mean())
    assert model_rmse < base_rmse


def test_does_not_beat_persistence_on_pure_noise():
    """Guards against a leak: with no signal the model must not win."""
    out = models.fit_predict_ticker(_synthetic(signal=False))
    model_rmse = np.sqrt(((out["y_true"] - out["y_pred_model"]) ** 2).mean())
    base_rmse = np.sqrt(((out["y_true"] - out["y_pred_persistence"]) ** 2).mean())
    assert model_rmse > base_rmse * 0.95


def test_rows_with_null_target_are_excluded():
    df = _synthetic()
    df.loc[df.index[-5:], features.TARGET] = np.nan
    out = models.fit_predict_ticker(df)
    assert out["y_true"].notna().all()


def test_run_all_handles_multiple_tickers():
    df = pd.concat([_synthetic(ticker="AAA"), _synthetic(ticker="BBB")],
                   ignore_index=True)
    out = models.run_all(df)
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_short_series_is_skipped_not_crashed():
    out = models.run_all(_synthetic(n=60))
    assert out.empty


def test_arima_fallback_survives_unexpected_exception(monkeypatch):
    """Pins the "must never abort the run" guarantee for _arima_forecast.

    statsmodels' exception surface is not enumerable -- MissingDataError
    inherits from Exception, not ValueError, so a narrowed
    `except (ValueError, LinAlgError)` would let it (and other statsmodels
    exceptions) propagate and crash the whole run. Injects exactly such an
    exception -- outside that narrower set -- and asserts the function still
    falls back to the training mean rather than raising.
    """
    from statsmodels.tools.sm_exceptions import MissingDataError

    class _ExplodingARIMA:
        def __init__(self, *args, **kwargs):
            raise MissingDataError("injected for test: not a ValueError")

    monkeypatch.setattr("statsmodels.tsa.arima.model.ARIMA", _ExplodingARIMA)

    out = models.fit_predict_ticker(_synthetic())
    assert len(out) > 0
    assert np.isfinite(out["y_pred_arima"]).all()
