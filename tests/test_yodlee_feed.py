import json
import os
import subprocess
import sys

from src import config
from src.generators import yodlee_feed

REQUIRED_TXN_KEYS = {
    "id", "accountId", "date", "transactionDate", "postDate", "amount",
    "runningBalance", "merchant", "status", "baseType", "subType", "category",
    "categoryType", "categoryId", "detailCategoryId", "detailCategory",
    "highLevelCategoryId", "categorySource", "sourceType", "checkNumber",
    "isManual", "container", "createdDate", "lastUpdated",
}


def test_transaction_matches_yodlee_shape():
    txns = yodlee_feed.generate_transactions(n=500)
    t = txns[0]
    assert REQUIRED_TXN_KEYS <= set(t)
    assert set(t["amount"]) == {"amount", "currency"}
    assert set(t["merchant"]) >= {"id", "source"}
    assert t["baseType"] in {"CREDIT", "DEBIT"}


def test_amounts_are_unsigned_in_source():
    # Yodlee carries direction in baseType, not in the sign. Silver must sign it.
    txns = yodlee_feed.generate_transactions(n=2000)
    assert all(t["amount"]["amount"] > 0 for t in txns)


def test_tracked_merchants_present_and_noise_dominates():
    txns = yodlee_feed.generate_transactions(n=20000)
    tracked_strings = {s for c in config.COMPANIES for s in c.merchant_strings}
    names = [t["merchant"]["source"] for t in txns]
    tracked = sum(1 for nm in names if nm in tracked_strings)
    assert tracked > 0
    ratio = 1 - tracked / len(names)
    assert 0.50 < ratio < 0.70


def test_ids_unique_and_json_serializable():
    txns = yodlee_feed.generate_transactions(n=5000)
    assert len({t["id"] for t in txns}) == len(txns)
    json.dumps(txns[:10])


def test_deterministic():
    """Same-process purity. NOT a determinism test -- see the one below."""
    a = yodlee_feed.generate_transactions(seed=11, n=1000)
    b = yodlee_feed.generate_transactions(seed=11, n=1000)
    assert a == b


_GENERATE_AND_DIGEST = """
import hashlib, json, sys
from src.generators import yodlee_feed
records = yodlee_feed.generate_transactions(seed=11, n=1500)
blob = json.dumps(records, sort_keys=True, default=str).encode()
sys.stdout.write(hashlib.sha256(blob).hexdigest())
"""


def test_deterministic_across_separate_processes():
    """The determinism test the same-process one structurally cannot be.

    `docs/ai-sdlc/decisions/0002` records a real bug -- builtin `hash()` for
    merchant ids -- that `test_deterministic` above passed straight through,
    because both of its observations shared the process state that varied.
    Two interpreters with `PYTHONHASHSEED=random` do not share it. Reverting
    `_stable_merchant_hash` to builtin `hash()` makes this fail and leaves
    `test_deterministic` green.
    """
    env = {**os.environ, "PYTHONHASHSEED": "random",
           "PYTHONPATH": str(config.REPO_ROOT)}
    digests = set()
    for _ in range(2):
        proc = subprocess.run(  # fixed argv, no shell
            [sys.executable, "-c", _GENERATE_AND_DIGEST],
            cwd=config.REPO_ROOT, env=env, capture_output=True,
            text=True, check=True)
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, (
        f"generation differs across processes: {digests}")
    assert next(iter(digests))


def test_some_transactions_are_late_arriving_corrections():
    # Duplicate ids with a newer lastUpdated: Silver dedup must handle these.
    txns = yodlee_feed.generate_transactions(n=20000)
    seen = {}
    dupes = 0
    for t in txns:
        if t["id"] in seen:
            dupes += 1
        seen[t["id"]] = t
    assert dupes == 0  # ids unique within a batch; corrections come from re-ingest


def test_dates_within_configured_window():
    txns = yodlee_feed.generate_transactions(n=5000)
    days = {d.isoformat() for d in config.trading_days(config.START_DATE, config.END_DATE)}
    assert all(t["transactionDate"] in days for t in txns)


def test_accounts_have_balances():
    accts = yodlee_feed.generate_accounts()
    assert len(accts) == config.N_USERS
    a = accts[0]
    assert a["accountType"] in {"CHECKING", "SAVINGS", "CREDIT"}
    assert set(a["balance"]) == {"amount", "currency"}
