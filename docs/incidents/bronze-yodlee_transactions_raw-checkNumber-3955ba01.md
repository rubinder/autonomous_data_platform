# Schema Drift — bronze.yodlee_transactions_raw

**Severity:** breaking

**Detected as of:** 2026-06-30

## What was observed

column 'checkNumber' declared in contract but absent from table

## Why this severity

Column 'checkNumber' is declared in the contract and consumed downstream. Its absence will fail the next Silver build.

## Evidence

```
{'change': 'dropped', 'column': 'checkNumber', 'declared_type': 'string'}
```
