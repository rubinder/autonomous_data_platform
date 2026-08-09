"""Silver: conform, dedupe, sign, normalize, quarantine, validate.

Deduplication keeps the greatest lastUpdated per id, so a late-arriving
correction wins over the original. Bronze may contain the same id many times
across replays; Silver is where that collapses to one truth.

Yodlee carries amounts UNSIGNED, with direction in `baseType`
(CREDIT/DEBIT) -- Bronze deliberately preserves that raw shape. Signing
happens here: DEBIT becomes negative, CREDIT stays positive. Getting this
backwards produces a pipeline that runs clean and reports inverted spend,
which no schema check would catch -- only the `conditional_sign` contract
expectation (Task 7) can express "DEBIT implies negative", and `_SIGN_DEBITS`
exists so a test can inject that exact bug and prove the contract aborts the
build rather than warning and continuing.
"""
from __future__ import annotations

import re
import sys
from datetime import UTC, datetime

import pyarrow as pa

from src import config
from src.contracts import validator
from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine

_SIGN_DEBITS = True  # monkeypatched in tests to prove the build fails closed

# Pinned, not wall-clock, for the same reason Bronze pins its ingest
# timestamp: byte-for-byte reproducibility across runs.
_QUARANTINED_AT = datetime(2026, 7, 1, 6, 0, tzinfo=UTC)

# Order matters: strip apostrophes first (no space -- "DOMINO'S" -> "DOMINOS",
# not "DOMINO S"), then strip store-number suffixes (with a space -- these
# are separate tokens: "#1234", "-1088", or a standalone run of 3+ digits),
# then collapse any remaining punctuation to spaces.
_APOSTROPHE = re.compile(r"[’']")
_STORE_SUFFIX = re.compile(r"(#\s*\d+|-\s*\d+|\b\d{3,}\b)")
_PUNCT = re.compile(r"[^A-Z0-9 ]")


def normalize_merchant(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.upper()
    text = _APOSTROPHE.sub("", text)
    text = _STORE_SUFFIX.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()) or None


def merchant_map_rows() -> list[dict]:
    rows = []
    for company in config.COMPANIES:
        for raw in company.merchant_strings:
            rows.append({
                "merchant_normalized": normalize_merchant(raw),
                "ticker": company.ticker,
                "company_name": company.name,
                "match_type": "exact",
            })
    seen, deduped = set(), []
    for row in rows:
        key = (row["merchant_normalized"], row["ticker"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _build_transactions(engine) -> tuple[pa.Table, pa.Table]:
    sign_expr = ("CASE WHEN baseType = 'DEBIT' THEN -1.0 ELSE 1.0 END"
                 if _SIGN_DEBITS else "1.0")
    query = f"""
    WITH ranked AS (
        SELECT *, row_number() OVER (
            PARTITION BY id ORDER BY lastUpdated DESC, _ingested_at DESC
        ) AS rn
        FROM b
    )
    SELECT
        id,
        accountId                                     AS account_id,
        CAST(transactionDate AS DATE)                 AS txn_date,
        CAST(postDate AS DATE)                        AS post_date,
        ({sign_expr}) * amount.amount                 AS signed_amount,
        amount.currency                               AS currency,
        baseType                                      AS base_type,
        subType                                       AS sub_type,
        category,
        categoryType                                  AS category_type,
        merchant.source                               AS merchant_raw,
        merchant.source                               AS merchant_normalized,
        CAST(lastUpdated AS TIMESTAMP WITH TIME ZONE)  AS last_updated,
        _ingested_at
    FROM ranked WHERE rn = 1
    """
    conformed = engine.sql(query, tables={"b": schemas.BRONZE_TRANSACTIONS.name})

    normalized = pa.array(
        [normalize_merchant(v) for v in conformed.column("merchant_normalized").to_pylist()],
        pa.large_string())
    conformed = conformed.set_column(
        conformed.schema.get_field_index("merchant_normalized"),
        "merchant_normalized", normalized)

    currency = conformed.column("currency").to_pylist()
    keep = pa.array([i for i, c in enumerate(currency) if c == "USD"], pa.int64())
    reject = pa.array([i for i, c in enumerate(currency) if c != "USD"], pa.int64())

    clean = conformed.take(keep)
    quarantined = conformed.take(reject)
    if quarantined.num_rows:
        quarantined = quarantined.append_column(
            "quarantine_reason",
            pa.array(["non_usd_currency"] * quarantined.num_rows, pa.large_string()))
        quarantined = quarantined.append_column(
            "quarantined_at",
            pa.array([_QUARANTINED_AT] * quarantined.num_rows,
                     pa.timestamp("us", tz="UTC")))
    else:
        quarantined = schemas.SILVER_QUARANTINE.schema.as_arrow().empty_table()

    return clean, quarantined


def _build_accounts(engine) -> pa.Table:
    return engine.sql("""
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY id ORDER BY lastUpdated DESC) AS rn
            FROM a)
        SELECT id, providerName AS provider_name, accountType AS account_type,
               accountStatus AS account_status, balance.amount AS balance_amount,
               balance.currency AS currency,
               CAST(lastUpdated AS TIMESTAMP WITH TIME ZONE) AS last_updated
        FROM ranked WHERE rn = 1
    """, tables={"a": schemas.BRONZE_ACCOUNTS.name})


def _build_prices(engine) -> pa.Table:
    return engine.sql("""
        WITH ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY ticker, date ORDER BY _ingested_at DESC) AS rn
            FROM p)
        SELECT ticker, CAST(date AS DATE) AS trade_date,
               open, high, low, close, adj_close, volume
        FROM ranked WHERE rn = 1
    """, tables={"p": schemas.BRONZE_PRICES.name})


def build_all(engine) -> dict[str, int]:
    counts: dict[str, int] = {}
    as_of = config.resolve_as_of_date(bronze.max_txn_date(engine))

    txns, quarantined = _build_transactions(engine)

    # Validate BEFORE writing anything: a contract violation must abort the
    # build, not get written to Silver and reported on a passing run.
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "silver_transactions.yaml")
    validator.assert_valid(txns, contract, as_of)

    for table_def, data in (
        (schemas.SILVER_TRANSACTIONS, txns),
        (schemas.SILVER_QUARANTINE, quarantined),
    ):
        engine.create_table(table_def)
        engine.overwrite(table_def.name, data)
        counts[table_def.name] = data.num_rows

    accounts = _build_accounts(engine)
    engine.create_table(schemas.SILVER_ACCOUNTS)
    engine.overwrite(schemas.SILVER_ACCOUNTS.name, accounts)
    counts[schemas.SILVER_ACCOUNTS.name] = accounts.num_rows

    mapping = pa.Table.from_pylist(
        merchant_map_rows(), schema=schemas.SILVER_MERCHANT_MAP.schema.as_arrow())
    engine.create_table(schemas.SILVER_MERCHANT_MAP)
    engine.overwrite(schemas.SILVER_MERCHANT_MAP.name, mapping)
    counts[schemas.SILVER_MERCHANT_MAP.name] = mapping.num_rows

    prices = _build_prices(engine)
    engine.create_table(schemas.SILVER_PRICES)
    engine.overwrite(schemas.SILVER_PRICES.name, prices)
    counts[schemas.SILVER_PRICES.name] = prices.num_rows

    return counts


def main() -> int:
    engine = get_engine()
    for name, n in build_all(engine).items():
        print(f"silver: wrote {n:,} rows to {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
