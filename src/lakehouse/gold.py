"""Gold: feature marts. Every window is trailing and right-closed.

The only forward-looking column in this module is fwd_ret_5d (Task 10), and
it is the target. Everything else at date t is computable from data <= t.
"""
from __future__ import annotations

import sys

import pyarrow as pa

from src.lakehouse import schemas
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


def build_daily_merchant_spend(engine) -> pa.Table:
    return engine.sql(SPEND_SQL, tables={
        "silver_txn": schemas.SILVER_TRANSACTIONS.name,
        "silver_map": schemas.SILVER_MERCHANT_MAP.name,
    })


def build_stock_features(engine) -> pa.Table:
    return engine.sql(STOCK_FEATURES_SQL,
                      tables={"silver_prices": schemas.SILVER_PRICES.name})


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
    return counts


def main() -> int:
    engine = get_engine()
    for name, n in build_all(engine).items():
        print(f"gold: wrote {n:,} rows to {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
