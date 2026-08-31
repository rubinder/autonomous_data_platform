# Volume Anomaly — bronze.yodlee_transactions_raw

**Severity:** breaking

**Detected as of:** 2026-06-30

## What was observed

latest snapshot added 5 rows, below 50% of trailing median 250,000

## Why this severity

Latest batch added 5 rows against a trailing median of 250,000. Downstream aggregates will be silently wrong rather than obviously missing.

## Evidence

```
{'rows_in_latest_snapshot': 5, 'median': 250000}
```
