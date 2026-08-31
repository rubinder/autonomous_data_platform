# Monitor Breach — silver.transactions

**Severity:** breaking

**Detected as of:** 2026-07-01

## What was observed

[silver.transactions_arrival_gap] 1 missing period(s): 2026-07-01

## Why this severity

Monitor 'silver.transactions_arrival_gap' breached on silver.transactions: [silver.transactions_arrival_gap] 1 missing period(s): 2026-07-01 A monitor breach is, by definition, already a judged verdict -- there is no additive reading of a check that has already failed.

## Evidence

```
{'monitor': 'silver.transactions_arrival_gap', 'kind': 'arrival_gap', 'column': 'txn_date', 'metric': 1.0, 'baseline': None, 'status': 'breach'}
```
