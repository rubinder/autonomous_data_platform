# Monitor Breach — bronze.yodlee_transactions_raw

**Severity:** breaking

**Detected as of:** 2026-06-30

## What was observed

[bronze_txn_row_count] 5 is 0% of trailing median 250000

## Why this severity

Monitor 'bronze_txn_row_count' breached on bronze.yodlee_transactions_raw: [bronze_txn_row_count] 5 is 0% of trailing median 250000 A monitor breach is, by definition, already a judged verdict -- there is no additive reading of a check that has already failed.

## Evidence

```
{'monitor': 'bronze_txn_row_count', 'kind': 'row_count', 'column': None, 'metric': 5.0, 'baseline': 250000.0, 'status': 'breach'}
```
