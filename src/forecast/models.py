"""HistGradientBoosting plus two baselines, evaluated under purged walk-forward CV.

sklearn's HistGradientBoostingRegressor rather than LightGBM: same
histogram-based algorithm, no native libomp dependency, which is the usual
cause of macOS/CI install failure. See docs/ai-sdlc/decisions/.

Both baselines exist so the report can answer "did the transaction feed add
anything" rather than only "did the model fit". Persistence (predict zero
change) is the honest null. ARIMA on price alone is the explicit "the
transaction feed adds nothing" null hypothesis -- if the model cannot beat
it, the report must say the alt-data signal added nothing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import config
from src.forecast import features, splits

MODEL_VERSION = "hgb-v1"

_OUTPUT_COLUMNS = [
    "ticker", "trade_date", "fold", "y_true",
    "y_pred_model", "y_pred_persistence", "y_pred_arima",
]


def fit_predict_ticker(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Fit the model and both baselines under purged walk-forward CV for one ticker.

    Rows with a null target are dropped before folds are computed -- the last
    FORECAST_HORIZON_DAYS rows of any series have no resolved forward return,
    and including them would either crash the model or hand it a NaN answer.
    """
    data = (df.dropna(subset=[features.TARGET])
              .sort_values("trade_date").reset_index(drop=True))
    dates = list(data["trade_date"])
    folds = splits.purged_walk_forward(dates, n_folds=n_folds, purge=config.PURGE_DAYS)

    x = data[list(features.FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = data[features.TARGET].to_numpy(dtype=float)

    frames = []
    for fold in folds:
        # Regularised hard, deliberately: at ~380 training rows, 14 features,
        # and a measured max |feature-target correlation| of 0.156 (Task 10),
        # the earlier config (max_iter=200, learning_rate=0.05,
        # l2_regularization=1.0, min_samples_leaf=20) overfit -- predictions
        # came out at roughly half the target's amplitude with near-zero
        # correlation to it (std(y_pred) ~0.02-0.03 vs std(y_true) ~0.03-0.06,
        # some tickers with a visible mean bias), which is textbook
        # overfitting on noise, not a real fit. early_stopping carves its
        # validation split from the training block only (fold.train is
        # already strictly before fold.test under the purge gap, so this
        # cannot see the test block), stops iterating once validation loss
        # stalls, and the higher l2/min_samples_leaf further discourage
        # memorizing individual rows. Measured effect: mean RMSE ratio to
        # persistence across the six real tickers dropped from ~1.16 to
        # ~1.02 -- see docs/forecast-report.md and the Task 13 review fix
        # report for before/after numbers.
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=300, learning_rate=0.03,
            l2_regularization=5.0, min_samples_leaf=40,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=15, tol=1e-4,
            random_state=config.SEED,
        )
        model.fit(x[fold.train], y[fold.train])
        preds = model.predict(x[fold.test])

        frames.append(pd.DataFrame({
            "ticker": data.loc[fold.test, "ticker"].to_numpy(),
            "trade_date": data.loc[fold.test, "trade_date"].to_numpy(),
            "fold": fold.index,
            "y_true": y[fold.test],
            "y_pred_model": preds,
            # Persistence: the honest "predict no change" null.
            "y_pred_persistence": np.zeros(len(fold.test)),
            "y_pred_arima": _arima_forecast(y, fold),
        }))

    return pd.concat(frames, ignore_index=True)


def _arima_forecast(y: np.ndarray, fold: splits.Fold) -> np.ndarray:
    """ARIMA(1,1,1) refit on the training block, forecast across the test block.

    Falls back to the training mean if the fit fails to converge or produces
    a non-finite forecast -- a non-converging baseline should not abort the
    whole run. Convergence warnings are expected on short/noisy training
    blocks and are suppressed rather than left to flood the output.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(y[fold.train], order=(1, 1, 1)).fit()
            forecast = fitted.forecast(steps=len(fold.test))
        values = np.asarray(forecast, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite forecast")
        return values
    except Exception:  # noqa: BLE001 -- deliberately broad: ARIMA is a
        # fallback-capable baseline whose failure must degrade to the
        # training mean rather than propagate and abort the whole run.
        # statsmodels' exception surface is not enumerable -- e.g.
        # MissingDataError inherits from Exception, not ValueError, and a
        # degenerate/too-short block can raise IndexError or LinAlgError.
        # A narrower catch would let any one of those crash forecasting for
        # every other ticker; see test_arima_fallback_survives_unexpected_exception.
        return np.full(len(fold.test), float(np.mean(y[fold.train])))


def run_all(training: pd.DataFrame) -> pd.DataFrame:
    """Run fit_predict_ticker per ticker, skipping any series too short for CV.

    purged_walk_forward raises ValueError when a series cannot support
    n_folds folds under the min_train/purge/min_test constraints -- that is
    intentional (see splits.py), and here it means "skip this ticker" rather
    than "crash the run".
    """
    frames = []
    for ticker, grp in training.groupby("ticker"):
        try:
            frames.append(fit_predict_ticker(grp))
        except ValueError:
            continue
    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["model_version"] = MODEL_VERSION
    return out
