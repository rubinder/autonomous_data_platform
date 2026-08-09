"""Yodlee-webhook-shaped account and transaction records.

Shape is faithful to https://docs.conductiv.co/docs/webhook/sample-data/yodlee/ --
notably `amount` is an unsigned struct and direction lives in `baseType`, which
is the correctness trap Silver has to get right.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from src import config
from src.generators.demand import build_demand_panel

_CATEGORY_BY_TICKER = {
    "SBUX": ("Restaurants/Dining", 3, 10, "Coffee Shop"),
    "CMG": ("Restaurants/Dining", 3, 10, "Fast Food"),
    "TGT": ("General Merchandise", 12, 22, "Department Store"),
    "LULU": ("Clothing/Shoes", 8, 22, "Athletic Apparel"),
    "DPZ": ("Restaurants/Dining", 3, 10, "Pizza"),
    "ULTA": ("Personal Care", 15, 22, "Cosmetics"),
}
_NOISE_CATEGORY = ("Other Expenses", 40, 30, "Uncategorized")
_TICKET_MEAN = {"SBUX": 8.5, "CMG": 14.0, "TGT": 62.0,
                 "LULU": 118.0, "DPZ": 24.0, "ULTA": 48.0}


def generate_accounts(seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed + 501)
    providers = [("Chase", "16441"), ("Bank of America", "5"), ("Wells Fargo", "1603")]
    accounts = []
    for i in range(config.N_USERS):
        provider, provider_id = providers[i % len(providers)]
        acct_type = ("CHECKING", "SAVINGS", "CREDIT")[i % 3]
        balance = round(float(rng.uniform(250, 24_000)), 2)
        created = datetime(2023, 6, 1, tzinfo=UTC) + timedelta(days=int(i % 200))
        accounts.append({
            "id": 100_000 + i,
            "providerName": provider,
            "providerId": provider_id,
            "accountType": acct_type,
            "accountStatus": "ACTIVE",
            "userClassification": "PERSONAL",
            "isAsset": acct_type != "CREDIT",
            "isManual": False,
            "aggregationSource": "USER",
            "balance": {"amount": balance, "currency": "USD"},
            "currentBalance": {"amount": balance, "currency": "USD"},
            "availableBalance": {"amount": round(balance * 0.92, 2), "currency": "USD"},
            "createdDate": created.isoformat(),
            "lastUpdated": datetime(2026, 6, 30, tzinfo=UTC).isoformat(),
        })
    return accounts


def generate_transactions(seed: int = config.SEED, n: int | None = None) -> list[dict]:
    total = n if n is not None else config.N_TRANSACTIONS
    rng = np.random.default_rng(seed + 907)
    panel = build_demand_panel(seed=seed)

    n_tracked = int(total * (1 - config.NOISE_FRACTION))
    n_noise = total - n_tracked

    records: list[dict] = []
    next_id = 5_000_000

    # Tracked spend: allocate proportionally to demand so daily aggregates
    # inherit the planted signal.
    weights = panel["demand"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    picks = rng.choice(len(panel), size=n_tracked, p=weights)
    tickers = panel["ticker"].to_numpy()
    days = panel["day"].to_numpy()

    for pick in picks:
        ticker = str(tickers[pick])
        day = days[pick]
        company = config.company_by_ticker(ticker)
        merchant = company.merchant_strings[int(rng.integers(len(company.merchant_strings)))]
        mean = _TICKET_MEAN[ticker]
        amount = round(float(rng.gamma(shape=4.0, scale=mean / 4.0)) + 1.0, 2)
        records.append(_build_txn(rng, next_id, day, merchant, amount,
                                   _CATEGORY_BY_TICKER[ticker], "DEBIT"))
        next_id += 1

    all_days = config.trading_days(config.START_DATE, config.END_DATE)
    noise_day_idx = rng.integers(0, len(all_days), size=n_noise)
    noise_pick = rng.integers(0, len(config.NOISE_MERCHANTS), size=n_noise)
    for k in range(n_noise):
        day = all_days[int(noise_day_idx[k])]
        merchant = config.NOISE_MERCHANTS[int(noise_pick[k])]
        is_income = merchant == "PAYROLL DIRECT DEP"
        amount = round(float(rng.gamma(3.0, 900.0 if is_income else 30.0)) + 1.0, 2)
        records.append(_build_txn(rng, next_id, day, merchant, amount,
                                   _NOISE_CATEGORY, "CREDIT" if is_income else "DEBIT"))
        next_id += 1

    records.sort(key=lambda r: (r["transactionDate"], r["id"]))
    return records


def _stable_merchant_hash(merchant: str) -> int:
    """Deterministic across processes, unlike Python's randomized str hash()."""
    digest = hashlib.md5(merchant.encode("utf-8")).hexdigest()
    return int(digest, 16) % 10_000_000


def _build_txn(rng, txn_id: int, day, merchant: str, amount: float,
               category: tuple, base_type: str) -> dict:
    day = day if isinstance(day, date) else pd.Timestamp(day).date()
    cat_name, cat_id, high_id, detail = category
    post = day + timedelta(days=int(rng.integers(0, 3)))
    created = datetime.combine(post, datetime.min.time(), tzinfo=UTC)
    return {
        "id": txn_id,
        "accountId": 100_000 + int(rng.integers(config.N_USERS)),
        "date": day.isoformat(),
        "transactionDate": day.isoformat(),
        "postDate": post.isoformat(),
        # Unsigned by design -- direction is in baseType.
        "amount": {"amount": amount, "currency": "USD"},
        "runningBalance": {"amount": round(float(rng.uniform(100, 20_000)), 2),
                            "currency": "USD"},
        "merchant": {
            "id": f"M{_stable_merchant_hash(merchant)}",
            "source": merchant,
            "categoryLabel": detail,
            "address": {"city": "SEATTLE", "state": "WA", "country": "USA"},
        },
        "status": "POSTED",
        "baseType": base_type,
        "subType": "PAYMENT" if base_type == "DEBIT" else "CREDIT",
        "category": cat_name,
        "categoryType": "EXPENSE" if base_type == "DEBIT" else "INCOME",
        "categoryId": cat_id,
        "detailCategoryId": cat_id * 100,
        "detailCategory": detail,
        "highLevelCategoryId": high_id,
        "categorySource": "SYSTEM",
        "sourceType": "AGGREGATED",
        "checkNumber": "",
        "isManual": False,
        "container": None,
        "createdDate": created.isoformat(),
        "lastUpdated": created.isoformat(),
    }
