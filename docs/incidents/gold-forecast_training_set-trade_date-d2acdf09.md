# Monitor Breach — gold.forecast_training_set

**Severity:** breaking

**Detected as of:** 2026-07-01

## What was observed

[gold.forecast_training_set_arrival_gap] 1 missing period(s): 2026-07-01

## Why this severity

Monitor 'gold.forecast_training_set_arrival_gap' breached on gold.forecast_training_set: [gold.forecast_training_set_arrival_gap] 1 missing period(s): 2026-07-01 A monitor breach is, by definition, already a judged verdict -- there is no additive reading of a check that has already failed.

## Evidence

```
{'monitor': 'gold.forecast_training_set_arrival_gap', 'kind': 'arrival_gap', 'column': 'trade_date', 'metric': 1.0, 'baseline': None, 'status': 'breach'}
```
