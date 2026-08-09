"""Bronze: append-only, source-faithful, lineage-stamped.

No type coercion, no dedupe, no signing of amounts. Everything that could be
wrong about the source stays wrong here on purpose -- Bronze's job is to be a
replayable record of what arrived, not a clean one. Re-ingesting the same
source duplicates rows by design: Bronze must be a replayable record of what
actually arrived on each run, and Silver (Task 8) is where dedupe happens.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, date, datetime

import pyarrow as pa

from src import config
from src.generators import stock_prices, yodlee_feed
from src.lakehouse import schemas
from src.lakehouse.engines import get_engine

# Pinned, not wall-clock: data must be byte-for-byte reproducible across runs.
_INGEST_TS = datetime(2026, 7, 1, 6, 0, tzinfo=UTC)


def payload_hash(record: dict) -> str:
    """Order-independent hash so key reordering upstream is not seen as change."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()


def add_lineage(
    records: list[dict], source_file: str, ingested_at: datetime | None = None
) -> list[dict]:
    """Stamp each record with lineage columns. Does not mutate the input dicts."""
    ts = ingested_at or _INGEST_TS
    out = []
    for rec in records:
        enriched = dict(rec)
        enriched["_ingested_at"] = ts
        enriched["_source_file"] = source_file
        enriched["_payload_hash"] = payload_hash(rec)
        enriched["_raw_payload"] = json.dumps(rec, sort_keys=True, default=str)
        out.append(enriched)
    return out


def _to_arrow(records: list[dict], arrow_schema: pa.Schema) -> pa.Table:
    """Shape records to `arrow_schema`. Keys absent from a record become NULL."""
    return pa.Table.from_pylist(records, schema=arrow_schema)


def ingest_all(engine, n_transactions: int | None = None) -> dict[str, int]:
    """Generate source-shaped records and append them, untouched, to Bronze.

    Returns rows appended per table this call. Re-running appends again --
    Bronze does not check for or collapse duplicates.
    """
    counts: dict[str, int] = {}

    txns = yodlee_feed.generate_transactions(n=n_transactions)
    accounts = yodlee_feed.generate_accounts()
    prices = stock_prices.generate_prices()

    for table_def, records, source in (
        (schemas.BRONZE_TRANSACTIONS, txns, "yodlee_transactions.json"),
        (schemas.BRONZE_ACCOUNTS, accounts, "yodlee_accounts.json"),
        (schemas.BRONZE_PRICES, prices, "stock_prices.json"),
    ):
        engine.create_table(table_def)
        enriched = add_lineage(records, source)
        # Against the *live* schema, not the authoring one: a table that has
        # been evolved (Task 16) has columns the source feed knows nothing
        # about, and `append` rejects a column set that does not match the
        # table exactly. Unknown columns land as NULL, which is the honest
        # answer -- Bronze records what arrived, and nothing arrived for them.
        engine.append(table_def.name,
                      _to_arrow(enriched, engine.arrow_schema(table_def.name)))
        counts[table_def.name] = len(enriched)

    return counts


def max_txn_date(engine) -> date | None:
    """Latest `transactionDate` seen in Bronze, or None if the table is empty/absent.

    `transactionDate` is a raw source string (Bronze does no type coercion), so
    the max is computed as a string and parsed only for the return value.
    """
    if not engine.table_exists(schemas.BRONZE_TRANSACTIONS.name):
        return None
    result = engine.sql(
        "SELECT max(transactionDate) AS d FROM bronze_txn",
        tables={"bronze_txn": schemas.BRONZE_TRANSACTIONS.name},
    )
    if result.num_rows == 0:
        return None
    value = result.column("d")[0].as_py()
    return date.fromisoformat(value) if value else None


def main() -> int:
    engine = get_engine()
    counts = ingest_all(engine)
    for name, n in counts.items():
        print(f"bronze: appended {n:,} rows to {name}")
    print(f"bronze: AS_OF_DATE resolves to {config.resolve_as_of_date(max_txn_date(engine))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
