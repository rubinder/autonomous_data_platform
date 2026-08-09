"""Run the forecast end to end and publish an honest report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from src import config
from src.forecast import backtest, features, models
from src.lakehouse import schemas
from src.lakehouse.engines import get_engine

REPORT_PATH = config.REPO_ROOT / "docs" / "forecast-report.md"

# Above this fraction of training rows with every spend feature NULL, spend
# coverage is too thin for the model to learn anything from alt-data (e.g.
# the N_TRANSACTIONS=5000 smoke dataset, where 59% of rows have no matched
# transactions at all). That is a data-volume caveat, not a build failure.
SPEND_NULL_CAVEAT_THRESHOLD = 0.30


def build_report(preds: pd.DataFrame, training: pd.DataFrame | None = None) -> str:
    lines = [
        "# Forecast Report",
        "",
        f"Model: `{models.MODEL_VERSION}` — HistGradientBoostingRegressor",
        f"Target: forward {config.FORECAST_HORIZON_DAYS}-day return",
        f"Validation: purged walk-forward CV, {config.PURGE_DAYS}-day purge/embargo",
        "",
        ("Prices in this project are synthetic with a planted per-ticker "
         "spend→return relationship (ADR-0004). These results measure whether the "
         "pipeline preserves a known signal, **not** real-world alpha."),
        "",
        "## Per-ticker results",
        "",
        "| Ticker | RMSE model | RMSE persist | RMSE ARIMA | Dir. acc | IC | vs persistence | vs ARIMA |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for ticker, grp in preds.groupby("ticker"):
        n = int(grp["y_true"].notna().sum())
        m = backtest.metrics(grp, "y_pred_model")
        p = backtest.metrics(grp, "y_pred_persistence")
        a = backtest.metrics(grp, "y_pred_arima")
        lines.append(
            f"| {ticker} | {m['rmse']:.5f} | {p['rmse']:.5f} | {a['rmse']:.5f} | "
            f"{m['directional_accuracy']:.3f} | {m['ic']:.3f} | "
            f"**{backtest.verdict(m, p, n)}** | **{backtest.verdict(m, a, n)}** |")

    sharpe = backtest.long_short_sharpe(preds, "y_pred_model")
    sharpe_text = (f"{sharpe:.2f}" if not np.isnan(sharpe)
                   else "n/a (insufficient trading days)")

    def _beats_persistence(g: pd.DataFrame) -> bool:
        n = int(g["y_true"].notna().sum())
        return backtest.verdict(backtest.metrics(g, "y_pred_model"),
                                 backtest.metrics(g, "y_pred_persistence"), n) == "beats"

    beats = sum(1 for _, g in preds.groupby("ticker") if _beats_persistence(g))
    total = preds["ticker"].nunique()

    lines += [
        "",
        "## Portfolio",
        "",
        f"Long/short Sharpe (top-2 / bottom-2, daily rebalance): **{sharpe_text}**",
        "",
        ("This Sharpe figure is frictionless: no transaction costs, no slippage, "
         "no capacity constraints. It is an upper bound on a strategy that cannot "
         "actually be traded at this cost, not an achievable return."),
        "",
        "## Honest summary",
        "",
        f"The model beats the persistence baseline on **{beats} of {total}** tickers.",
        "",
    ]

    if total and beats == total:
        lines.append(
            "> Every ticker beating baseline is a warning sign, not a success. "
            "With planted signal at differing lags and betas, uniform wins suggest "
            "leakage. Re-check `tests/test_no_leakage.py` before trusting this.")
    else:
        losers = [t for t, g in preds.groupby("ticker") if not _beats_persistence(g)]
        if losers:
            # Deliberately no lag/beta narrative here: measured against this
            # run's results, ticker ranking by IC does not track planted lag
            # or beta (e.g. CMG has the shortest lag and highest beta but is
            # not the best performer) -- inventing a causal story the data
            # doesn't support is exactly the overclaiming this report exists
            # to avoid. State only what was measured.
            loser_ics = [backtest.metrics(preds[preds["ticker"] == t],
                                           "y_pred_model")["ic"] for t in losers]
            lines.append(
                f"Tickers where the model does **not** beat persistence: "
                f"{', '.join(losers)}. Measured IC on these tickers ranges from "
                f"{min(loser_ics):.3f} to {max(loser_ics):.3f} -- consistent with the "
                "modest planted signal this dataset carries overall (Task 10: max "
                "|feature-target correlation| 0.156), not with any pattern by "
                "planted lag or beta.")

    if training is not None and len(training):
        spend_cols = list(features.SPEND_FEATURES)
        null_frac = float(training[spend_cols].isna().all(axis=1).mean())
        if null_frac > SPEND_NULL_CAVEAT_THRESHOLD:
            lines += [
                "",
                (f"> **Small-sample caveat:** {null_frac:.0%} of training rows have "
                 "every spend feature NULL (no matched transactions that day), "
                 "typical of a reduced-transaction smoke run rather than the full "
                 "dataset. The alt-data signal is largely unavailable at this "
                 "scale, so the results above should be read as a pipeline "
                 "correctness check, not a skill result."),
            ]

    return "\n".join(lines) + "\n"


def main() -> int:
    engine = get_engine()
    training = engine.scan_arrow(schemas.GOLD_TRAINING.name).to_pandas()
    preds = models.run_all(training)
    if preds.empty:
        print("forecast: no ticker had enough history for purged CV", file=sys.stderr)
        return 1

    engine.create_table(schemas.GOLD_PREDICTIONS)
    arrow = pa.Table.from_pandas(
        preds, schema=schemas.GOLD_PREDICTIONS.schema.as_arrow(),
        preserve_index=False)
    engine.overwrite(schemas.GOLD_PREDICTIONS.name, arrow)

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(build_report(preds, training))
    print(f"forecast: {len(preds):,} predictions written; report at {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
