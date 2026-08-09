# 0001 — `HistGradientBoostingRegressor` instead of `LGBMRegressor`

**Type:** documented deviation from the design spec
**Made:** during implementation (Task 12)
**Spec said:** §7 Models — "`LGBMRegressor` — modest depth and strong
regularization; n≈630/ticker is small."
**Shipped:** `sklearn.ensemble.HistGradientBoostingRegressor`

## What changed

The spec named LightGBM. The implementation uses scikit-learn's
`HistGradientBoostingRegressor`, which is already a dependency
(`scikit-learn>=1.5.0`), and dropped `lightgbm` from the dependency list.

## Why

`lightgbm`'s wheel links against OpenMP. On macOS that means `libomp`, which is
not present by default and is not installed by pip. The failure mode is a
successful `pip install lightgbm` followed by an `OSError` at *import* time —
`dlopen(... libomp.dylib) ... image not found` — which is the single most common
install failure for this library and the least obvious to diagnose, because the
package installed cleanly. The fix is `brew install libomp`, i.e. a system
package manager step that a `git clone && make all` promise cannot make.

This project's first success criterion is that the repository runs offline from a
clean clone with no external services. A dependency whose import can fail on a
supported platform, for reasons pip cannot fix, is incompatible with that.

## Why it does not change the model's character

`HistGradientBoostingRegressor` is the same algorithm family: histogram-binned
gradient boosting with leaf-wise-equivalent tree growth, explicitly modelled on
LightGBM (scikit-learn's own documentation says so). It supports the properties
the spec actually cared about:

- shallow trees / limited leaves for a small sample,
- L2 regularisation and learning-rate control,
- native handling of missing values, which matters here because the spend
  features are legitimately NULL on days with no transactions,
- early stopping on an internal validation split.

The spec's requirement was "a regularised histogram gradient booster suitable for
n≈630 per ticker", and that is what shipped. Nothing downstream — the purged CV,
the baselines, the metrics, the verdict logic — depends on which library provides
it.

The model is versioned as `hgb-v1` in `gold.forecast_predictions` and named in
`docs/forecast-report.md`, so the report does not claim to be LightGBM.

## What it does not excuse

The substitution changes the library, not the result. After recalibration in
Task 13 the model still beats the persistence baseline on **0 of 6** tickers, and
that is reported as measured. Choosing a different gradient booster would not
have changed that, and this deviation is not offered as an explanation for it.

## Verification

- Early stopping uses an internal split of the *training* array only. Confirmed
  at review: `fit()` receives `x[fold.train]` and nothing else, so the split
  cannot reach across a fold boundary.
- No `libomp`, no `brew`, no post-install step. `uv sync` is sufficient on a
  clean machine.
