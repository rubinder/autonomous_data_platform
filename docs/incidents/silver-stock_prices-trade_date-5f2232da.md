# Monitor Breach — silver.stock_prices

**Severity:** breaking

**Detected as of:** 2026-07-01

## What was observed

[silver.stock_prices_arrival_gap] 1 missing period(s): 2026-07-01

## Why this severity

Monitor 'silver.stock_prices_arrival_gap' breached on silver.stock_prices: [silver.stock_prices_arrival_gap] 1 missing period(s): 2026-07-01 A monitor breach is, by definition, already a judged verdict -- there is no additive reading of a check that has already failed.

## Evidence

```
{'monitor': 'silver.stock_prices_arrival_gap', 'kind': 'arrival_gap', 'column': 'trade_date', 'metric': 1.0, 'baseline': None, 'status': 'breach'}
```
