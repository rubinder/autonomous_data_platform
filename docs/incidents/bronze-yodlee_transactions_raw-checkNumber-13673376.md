# Schema Drift — bronze.yodlee_transactions_raw

**Severity:** renaming

**Detected as of:** 2026-07-01

## What was observed

column 'checkNumber' was renamed to 'check_reference' (field id 20); the contract still declares the old name

## Why this severity

'checkNumber' was renamed to 'check_reference'. Both are field id 20: Iceberg resolves columns by ID, not by name, so every file written under the old name still reads back correctly and no data was rewritten or lost. What is wrong is the published contract, which still declares 'checkNumber'. That needs a recorded decision, not a rollback.

## Evidence

```
{'change': 'renamed', 'column': 'checkNumber', 'renamed_to': 'check_reference', 'field_id': 20, 'declared_type': 'string', 'observed_type': 'string', 'type_compatible': True}
```
