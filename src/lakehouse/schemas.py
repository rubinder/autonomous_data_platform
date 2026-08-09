"""Iceberg schemas and partition specs for every table in the lakehouse.

Field ids here are *authoring* ids only: PyIceberg reassigns fresh, sequential
ids at `create_table` time and remaps partition-spec source ids by column name.
They still have to be unique within a schema -- a duplicate id would silently
mis-map a column during that remap -- which `_assert_unique_field_ids` checks
at import time.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import DayTransform, IdentityTransform, MonthTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    DoubleType,
    LongType,
    NestedField,
    StringType,
    StructType,
    TimestamptzType,
)


@dataclass(frozen=True)
class TableDef:
    name: str
    schema: Schema
    spec: PartitionSpec


def _money(fid: int, name: str, required: bool = False) -> NestedField:
    """Yodlee's {amount, currency} money struct.

    Inner ids are derived as fid*10+1 / fid*10+2 so they cannot collide with the
    two-digit-or-lower top-level ids used by every table here.
    """
    return NestedField(fid, name, StructType(
        NestedField(fid * 10 + 1, "amount", DoubleType(), required=False),
        NestedField(fid * 10 + 2, "currency", StringType(), required=False),
    ), required=required)


BRONZE_TRANSACTIONS = TableDef(
    "bronze.yodlee_transactions_raw",
    Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "accountId", LongType(), required=False),
        NestedField(3, "date", StringType(), required=False),
        NestedField(4, "transactionDate", StringType(), required=False),
        NestedField(5, "postDate", StringType(), required=False),
        _money(6, "amount"),
        _money(7, "runningBalance"),
        NestedField(8, "merchant", StructType(
            NestedField(81, "id", StringType(), required=False),
            NestedField(82, "source", StringType(), required=False),
            NestedField(83, "categoryLabel", StringType(), required=False),
            NestedField(84, "address", StructType(
                NestedField(841, "city", StringType(), required=False),
                NestedField(842, "state", StringType(), required=False),
                NestedField(843, "country", StringType(), required=False),
            ), required=False),
        ), required=False),
        NestedField(9, "status", StringType(), required=False),
        NestedField(10, "baseType", StringType(), required=False),
        NestedField(11, "subType", StringType(), required=False),
        NestedField(12, "category", StringType(), required=False),
        NestedField(13, "categoryType", StringType(), required=False),
        NestedField(14, "categoryId", LongType(), required=False),
        NestedField(15, "detailCategoryId", LongType(), required=False),
        NestedField(16, "detailCategory", StringType(), required=False),
        NestedField(17, "highLevelCategoryId", LongType(), required=False),
        NestedField(18, "categorySource", StringType(), required=False),
        NestedField(19, "sourceType", StringType(), required=False),
        NestedField(20, "checkNumber", StringType(), required=False),
        NestedField(21, "isManual", BooleanType(), required=False),
        NestedField(22, "container", StringType(), required=False),
        NestedField(23, "createdDate", StringType(), required=False),
        NestedField(24, "lastUpdated", StringType(), required=False),
        NestedField(25, "_ingested_at", TimestamptzType(), required=True),
        NestedField(26, "_source_file", StringType(), required=False),
        NestedField(27, "_payload_hash", StringType(), required=False),
        NestedField(28, "_raw_payload", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(25, 1000, DayTransform(), "_ingested_at_day")),
)

BRONZE_ACCOUNTS = TableDef(
    "bronze.yodlee_accounts_raw",
    Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "providerName", StringType(), required=False),
        NestedField(3, "providerId", StringType(), required=False),
        NestedField(4, "accountType", StringType(), required=False),
        NestedField(5, "accountStatus", StringType(), required=False),
        NestedField(6, "userClassification", StringType(), required=False),
        NestedField(7, "isAsset", BooleanType(), required=False),
        NestedField(8, "isManual", BooleanType(), required=False),
        NestedField(9, "aggregationSource", StringType(), required=False),
        _money(10, "balance"),
        _money(11, "currentBalance"),
        _money(12, "availableBalance"),
        NestedField(13, "createdDate", StringType(), required=False),
        NestedField(14, "lastUpdated", StringType(), required=False),
        NestedField(15, "_ingested_at", TimestamptzType(), required=True),
        NestedField(16, "_source_file", StringType(), required=False),
        NestedField(17, "_payload_hash", StringType(), required=False),
        NestedField(18, "_raw_payload", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(15, 1000, DayTransform(), "_ingested_at_day")),
)

BRONZE_PRICES = TableDef(
    "bronze.stock_prices_raw",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "date", StringType(), required=False),
        NestedField(3, "open", DoubleType(), required=False),
        NestedField(4, "high", DoubleType(), required=False),
        NestedField(5, "low", DoubleType(), required=False),
        NestedField(6, "close", DoubleType(), required=False),
        NestedField(7, "adj_close", DoubleType(), required=False),
        NestedField(8, "volume", LongType(), required=False),
        NestedField(9, "_ingested_at", TimestamptzType(), required=True),
        NestedField(10, "_source_file", StringType(), required=False),
        NestedField(11, "_payload_hash", StringType(), required=False),
        NestedField(12, "_raw_payload", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(9, 1000, DayTransform(), "_ingested_at_day")),
)

_SILVER_TXN_FIELDS = (
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "account_id", LongType(), required=False),
    NestedField(3, "txn_date", DateType(), required=True),
    NestedField(4, "post_date", DateType(), required=False),
    NestedField(5, "signed_amount", DoubleType(), required=True),
    NestedField(6, "currency", StringType(), required=False),
    NestedField(7, "base_type", StringType(), required=False),
    NestedField(8, "sub_type", StringType(), required=False),
    NestedField(9, "category", StringType(), required=False),
    NestedField(10, "category_type", StringType(), required=False),
    NestedField(11, "merchant_raw", StringType(), required=False),
    NestedField(12, "merchant_normalized", StringType(), required=False),
    NestedField(13, "last_updated", TimestamptzType(), required=False),
    NestedField(14, "_ingested_at", TimestamptzType(), required=False),
)

SILVER_TRANSACTIONS = TableDef(
    "silver.transactions",
    Schema(*_SILVER_TXN_FIELDS),
    PartitionSpec(PartitionField(3, 1000, MonthTransform(), "txn_date_month")),
)

SILVER_QUARANTINE = TableDef(
    "silver.transactions_quarantine",
    Schema(*_SILVER_TXN_FIELDS,
           NestedField(15, "quarantine_reason", StringType(), required=True),
           NestedField(16, "quarantined_at", TimestamptzType(), required=True)),
    PartitionSpec(),  # unpartitioned: expected to stay small
)

SILVER_ACCOUNTS = TableDef(
    "silver.accounts",
    Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "provider_name", StringType(), required=False),
        NestedField(3, "account_type", StringType(), required=False),
        NestedField(4, "account_status", StringType(), required=False),
        NestedField(5, "balance_amount", DoubleType(), required=False),
        NestedField(6, "currency", StringType(), required=False),
        NestedField(7, "last_updated", TimestamptzType(), required=False),
    ),
    PartitionSpec(),
)

SILVER_MERCHANT_MAP = TableDef(
    "silver.merchant_ticker_map",
    Schema(
        NestedField(1, "merchant_normalized", StringType(), required=True),
        NestedField(2, "ticker", StringType(), required=True),
        NestedField(3, "company_name", StringType(), required=False),
        NestedField(4, "match_type", StringType(), required=False),
    ),
    PartitionSpec(),
)

SILVER_PRICES = TableDef(
    "silver.stock_prices",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "open", DoubleType(), required=False),
        NestedField(4, "high", DoubleType(), required=False),
        NestedField(5, "low", DoubleType(), required=False),
        NestedField(6, "close", DoubleType(), required=False),
        NestedField(7, "adj_close", DoubleType(), required=False),
        NestedField(8, "volume", LongType(), required=False),
    ),
    PartitionSpec(PartitionField(2, 1000, MonthTransform(), "trade_date_month")),
)

GOLD_SPEND = TableDef(
    "gold.daily_merchant_spend",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "spend_date", DateType(), required=True),
        NestedField(3, "gross_spend", DoubleType(), required=False),
        NestedField(4, "txn_count", LongType(), required=False),
        NestedField(5, "unique_accounts", LongType(), required=False),
        NestedField(6, "avg_ticket", DoubleType(), required=False),
        NestedField(7, "median_ticket", DoubleType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

GOLD_STOCK_FEATURES = TableDef(
    "gold.stock_features",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "close", DoubleType(), required=False),
        NestedField(4, "ret_1d", DoubleType(), required=False),
        NestedField(5, "ret_lag_1", DoubleType(), required=False),
        NestedField(6, "ret_lag_2", DoubleType(), required=False),
        NestedField(7, "ret_lag_3", DoubleType(), required=False),
        NestedField(8, "ret_lag_4", DoubleType(), required=False),
        NestedField(9, "ret_lag_5", DoubleType(), required=False),
        NestedField(10, "realized_vol_21d", DoubleType(), required=False),
        NestedField(11, "rsi_14", DoubleType(), required=False),
        NestedField(12, "volume_z_21d", DoubleType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

GOLD_TRAINING = TableDef(
    "gold.forecast_training_set",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "spend_mom_7d", DoubleType(), required=False),
        NestedField(4, "spend_z_28d", DoubleType(), required=False),
        NestedField(5, "txn_count_z_28d", DoubleType(), required=False),
        NestedField(6, "avg_ticket_delta_7d", DoubleType(), required=False),
        NestedField(7, "unique_acct_growth_7d", DoubleType(), required=False),
        NestedField(8, "spend_surprise", DoubleType(), required=False),
        NestedField(9, "ret_lag_1", DoubleType(), required=False),
        NestedField(10, "ret_lag_2", DoubleType(), required=False),
        NestedField(11, "ret_lag_3", DoubleType(), required=False),
        NestedField(12, "ret_lag_4", DoubleType(), required=False),
        NestedField(13, "ret_lag_5", DoubleType(), required=False),
        NestedField(14, "realized_vol_21d", DoubleType(), required=False),
        NestedField(15, "rsi_14", DoubleType(), required=False),
        NestedField(16, "volume_z_21d", DoubleType(), required=False),
        NestedField(17, "fwd_ret_5d", DoubleType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

GOLD_PREDICTIONS = TableDef(
    "gold.forecast_predictions",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "fold", LongType(), required=False),
        NestedField(4, "y_true", DoubleType(), required=False),
        NestedField(5, "y_pred_model", DoubleType(), required=False),
        NestedField(6, "y_pred_persistence", DoubleType(), required=False),
        NestedField(7, "y_pred_arima", DoubleType(), required=False),
        NestedField(8, "model_version", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

OPS_MONITOR_RESULTS = TableDef(
    "ops.monitor_results",
    Schema(
        NestedField(1, "run_at", DateType(), required=True),
        NestedField(2, "monitor", StringType(), required=True),
        NestedField(3, "table_name", StringType(), required=True),
        NestedField(4, "column_name", StringType(), required=False),
        NestedField(5, "kind", StringType(), required=False),
        NestedField(6, "metric", DoubleType(), required=False),
        NestedField(7, "baseline_median", DoubleType(), required=False),
        NestedField(8, "status", StringType(), required=True),
        NestedField(9, "detail", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, MonthTransform(), "run_at_month")),
)

OPS_ALERT_LOG = TableDef(
    "ops.alert_log",
    Schema(
        NestedField(1, "run_at", DateType(), required=True),
        NestedField(2, "alert_key", StringType(), required=True),
        NestedField(3, "severity", StringType(), required=True),
        NestedField(4, "title", StringType(), required=False),
        NestedField(5, "body", StringType(), required=False),
        NestedField(6, "source", StringType(), required=False),
        NestedField(7, "delivered", BooleanType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, MonthTransform(), "run_at_month")),
)

ALL_TABLES: tuple[TableDef, ...] = (
    BRONZE_TRANSACTIONS, BRONZE_ACCOUNTS, BRONZE_PRICES,
    SILVER_TRANSACTIONS, SILVER_QUARANTINE, SILVER_ACCOUNTS,
    SILVER_MERCHANT_MAP, SILVER_PRICES,
    GOLD_SPEND, GOLD_STOCK_FEATURES, GOLD_TRAINING, GOLD_PREDICTIONS,
    OPS_MONITOR_RESULTS, OPS_ALERT_LOG,
)


def _field_ids(field_type) -> list[int]:
    ids: list[int] = []
    for field in getattr(field_type, "fields", ()):
        ids.append(field.field_id)
        ids.extend(_field_ids(field.field_type))
    return ids


def _assert_unique_field_ids() -> None:
    """A duplicate id inside one schema mis-maps columns instead of raising."""
    for table in ALL_TABLES:
        ids = _field_ids(table.schema.as_struct())
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise AssertionError(f"{table.name}: duplicate field ids {sorted(duplicates)}")


_assert_unique_field_ids()


def by_name(name: str) -> TableDef:
    for table in ALL_TABLES:
        if table.name == name:
            return table
    raise KeyError(name)
