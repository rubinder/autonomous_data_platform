"""Gold: feature marts. Every window is trailing and right-closed.

The only forward-looking column in this module is fwd_ret_5d (Task 10), and
it is the target. Everything else at date t is computable from data <= t.
"""
from __future__ import annotations

import sys

import pyarrow as pa

from src import config
from src.contracts import validator
from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine

SPEND_SQL = """
SELECT
    m.ticker                                   AS ticker,
    t.txn_date                                 AS spend_date,
    sum(abs(t.signed_amount))                  AS gross_spend,
    count(*)                                   AS txn_count,
    count(DISTINCT t.account_id)               AS unique_accounts,
    avg(abs(t.signed_amount))                  AS avg_ticket,
    median(abs(t.signed_amount))               AS median_ticket
FROM silver_txn t
JOIN silver_map m ON t.merchant_normalized = m.merchant_normalized
WHERE t.category_type = 'EXPENSE'
GROUP BY 1, 2
ORDER BY 1, 2
"""

# Every window below is ROWS BETWEEN n PRECEDING AND CURRENT ROW: trailing
# and right-closed. Task 10's no-lookahead test recomputes these from
# truncated history and asserts equality -- an unbounded, centred, or
# forward-looking window here would fail that test one task later.
STOCK_FEATURES_SQL = """
WITH base AS (
    SELECT ticker, trade_date, close, volume,
           close / lag(close) OVER (
               PARTITION BY ticker ORDER BY trade_date) - 1 AS ret_1d
    FROM silver_prices
),
gains AS (
    SELECT *,
           greatest(ret_1d, 0)            AS gain,
           greatest(-ret_1d, 0)           AS loss
    FROM base
)
SELECT
    ticker, trade_date, close,
    ret_1d,
    lag(ret_1d, 1) OVER (PARTITION BY ticker ORDER BY trade_date) AS ret_lag_1,
    lag(ret_1d, 2) OVER (PARTITION BY ticker ORDER BY trade_date) AS ret_lag_2,
    lag(ret_1d, 3) OVER (PARTITION BY ticker ORDER BY trade_date) AS ret_lag_3,
    lag(ret_1d, 4) OVER (PARTITION BY ticker ORDER BY trade_date) AS ret_lag_4,
    lag(ret_1d, 5) OVER (PARTITION BY ticker ORDER BY trade_date) AS ret_lag_5,
    stddev_samp(ret_1d) OVER (PARTITION BY ticker ORDER BY trade_date
                              ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)
        * sqrt(252)                       AS realized_vol_21d,
    100 - 100 / (1 + (
        avg(gain) OVER (PARTITION BY ticker ORDER BY trade_date
                        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
        / nullif(avg(loss) OVER (PARTITION BY ticker ORDER BY trade_date
                                 ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0)
    ))                                    AS rsi_14,
    (volume - avg(volume) OVER (PARTITION BY ticker ORDER BY trade_date
                                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW))
      / nullif(stddev_samp(volume) OVER (PARTITION BY ticker ORDER BY trade_date
                                         ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0)
                                          AS volume_z_21d
FROM gains
ORDER BY ticker, trade_date
"""


# The training set. Every feature frame below is trailing and right-closed,
# so a feature at t is computable from rows dated <= t. The one exception is
# fwd_ret_5d, the *target*, which is a lead by construction.
#
# Two deliberate deviations from a literal reading of the spec, both forced by
# DuckDB 1.5.5 and neither changing a single frame:
#   * Windows are inlined rather than declared in a named WINDOW clause, matching
#     STOCK_FEATURES_SQL above.
#   * `lag(avg(unique_accounts) OVER w7, 7)` is a nested window call, which the
#     binder rejects outright ("window function calls cannot be nested"), so the
#     moving average is materialised in `spend_windows` and lagged in `spend`.
#
# `acct_ma7` is gated on a full seven rows. Without the gate a partial window
# makes `unique_acct_growth_7d` the ratio of a 7-day mean to a *1-day* "mean"
# seven rows earlier -- a warm-up artifact, not a growth rate, and one the
# independent recomputation (pandas `rolling(7)`, which needs 7 observations)
# refuses to produce. Ungated it fed 36 such values into the training set.
#
# `spend_surprise` is the only frame that excludes the current row: its
# `4 PRECEDING AND 1 PRECEDING` same-weekday window is a seasonal-naive baseline
# that the current day's spend is measured *against*, so including today would
# make the feature partly a comparison with itself.
TRAINING_SQL = """
WITH spend_windows AS (
    SELECT ticker, spend_date, gross_spend, txn_count, unique_accounts, avg_ticket,
           avg(gross_spend) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)  AS spend_ma7,
           avg(gross_spend) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS spend_ma28,
           avg(gross_spend) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS spend_mean28,
           stddev_samp(gross_spend) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS spend_std28,
           avg(txn_count) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS txn_mean28,
           stddev_samp(txn_count) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS txn_std28,
           avg(avg_ticket) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)  AS ticket_ma7,
           avg(avg_ticket) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS ticket_ma28,
           CASE WHEN count(*) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) = 7
                THEN avg(unique_accounts) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) END AS acct_ma7,
           avg(gross_spend) OVER (PARTITION BY ticker, dayofweek(spend_date)
                                  ORDER BY spend_date
                                  ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)  AS dow_naive,
           stddev_samp(gross_spend) OVER (PARTITION BY ticker ORDER BY spend_date
                                  ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS spend_sigma
    FROM gold_spend
),
spend AS (
    SELECT *,
           lag(acct_ma7, 7) OVER (PARTITION BY ticker ORDER BY spend_date) AS acct_ma7_prev
    FROM spend_windows
),
target AS (
    SELECT ticker, trade_date, adj_close,
           lead(adj_close, 5) OVER (PARTITION BY ticker ORDER BY trade_date)
             / adj_close - 1 AS fwd_ret_5d
    FROM silver_prices
)
SELECT
    f.ticker, f.trade_date,
    s.spend_ma7 / nullif(s.spend_ma28, 0) - 1                   AS spend_mom_7d,
    (s.gross_spend - s.spend_mean28) / nullif(s.spend_std28, 0) AS spend_z_28d,
    (s.txn_count  - s.txn_mean28)   / nullif(s.txn_std28, 0)    AS txn_count_z_28d,
    s.ticket_ma7 / nullif(s.ticket_ma28, 0) - 1                 AS avg_ticket_delta_7d,
    s.acct_ma7   / nullif(s.acct_ma7_prev, 0) - 1               AS unique_acct_growth_7d,
    (s.gross_spend - s.dow_naive) / nullif(s.spend_sigma, 0)    AS spend_surprise,
    f.ret_lag_1, f.ret_lag_2, f.ret_lag_3, f.ret_lag_4, f.ret_lag_5,
    f.realized_vol_21d, f.rsi_14, f.volume_z_21d,
    t.fwd_ret_5d
FROM gold_features f
JOIN target t ON f.ticker = t.ticker AND f.trade_date = t.trade_date
LEFT JOIN spend s ON s.ticker = f.ticker AND s.spend_date = f.trade_date
ORDER BY f.ticker, f.trade_date
"""


def build_daily_merchant_spend(engine) -> pa.Table:
    return engine.sql(SPEND_SQL, tables={
        "silver_txn": schemas.SILVER_TRANSACTIONS.name,
        "silver_map": schemas.SILVER_MERCHANT_MAP.name,
    })


def build_stock_features(engine) -> pa.Table:
    return engine.sql(STOCK_FEATURES_SQL,
                      tables={"silver_prices": schemas.SILVER_PRICES.name})


def build_training_set(engine) -> pa.Table:
    return engine.sql(TRAINING_SQL, tables={
        "gold_spend": schemas.GOLD_SPEND.name,
        "gold_features": schemas.GOLD_STOCK_FEATURES.name,
        "silver_prices": schemas.SILVER_PRICES.name,
    })


def _write(engine, table_def, data: pa.Table) -> int:
    engine.create_table(table_def)
    engine.overwrite(table_def.name, data.cast(table_def.schema.as_arrow()))
    return data.num_rows


def build_all(engine) -> dict[str, int]:
    counts = {
        schemas.GOLD_SPEND.name: _write(
            engine, schemas.GOLD_SPEND, build_daily_merchant_spend(engine)),
        schemas.GOLD_STOCK_FEATURES.name: _write(
            engine, schemas.GOLD_STOCK_FEATURES, build_stock_features(engine)),
    }
    # After the marts: the training set reads gold_spend and gold_features back
    # out of the catalog, so both must be committed first.
    training = build_training_set(engine)

    # Gold fails closed too, on the same terms as Silver: validate before the
    # write so a violation leaves the table absent rather than half-correct.
    # Until this call existed, `contracts/gold_forecast_training_set.yaml` was
    # read by the type monitor and by nothing else -- four expectations that
    # no build path evaluated.
    as_of = config.resolve_as_of_date(bronze.max_txn_date(engine))
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "gold_forecast_training_set.yaml")
    validator.assert_valid(training, contract, as_of)

    counts[schemas.GOLD_TRAINING.name] = _write(
        engine, schemas.GOLD_TRAINING, training)
    return counts


def main() -> int:
    engine = get_engine()
    for name, n in build_all(engine).items():
        print(f"gold: wrote {n:,} rows to {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
