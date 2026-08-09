# Schema Drift — bronze.yodlee_transactions_raw

**Severity:** breaking

**Detected as of:** 2026-06-30

## What was observed

column 'checkNumber' declared in contract but absent from table

## Why this severity

Column 'checkNumber' is declared in the contract but is absent from the table, so the published contract is violated. Any consumer that selects it by name breaks; which consumers those are is not visible from this table's metadata and needs a human to confirm.

## Evidence

```
{'change': 'dropped', 'column': 'checkNumber', 'declared_type': 'string'}
```
