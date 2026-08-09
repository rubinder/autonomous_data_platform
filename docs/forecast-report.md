# Forecast Report

Model: `hgb-v1` — HistGradientBoostingRegressor
Target: forward 5-day return
Validation: purged walk-forward CV, 5-day purge/embargo

Prices in this project are synthetic with a planted per-ticker spend→return relationship (ADR-0004). These results measure whether the pipeline preserves a known signal, **not** real-world alpha.

## Per-ticker results

| Ticker | RMSE model | RMSE persist | RMSE ARIMA | Dir. acc | IC | vs persistence | vs ARIMA |
|---|---|---|---|---|---|---|---|
| CMG | 0.05107 | 0.04702 | 0.04649 | 0.512 | -0.000 | **loses** | **loses** |
| DPZ | 0.04294 | 0.03386 | 0.03603 | 0.484 | -0.079 | **loses** | **loses** |
| LULU | 0.06543 | 0.05810 | 0.09548 | 0.472 | -0.038 | **loses** | **loses** |
| SBUX | 0.04882 | 0.04518 | 0.06086 | 0.546 | 0.105 | **loses** | **beats** |
| TGT | 0.04512 | 0.03866 | 0.05188 | 0.454 | -0.100 | **loses** | **loses** |
| ULTA | 0.06016 | 0.04988 | 0.07856 | 0.486 | -0.007 | **loses** | **beats** |

## Portfolio

Long/short Sharpe (top-2 / bottom-2, daily rebalance): **-0.16**

This Sharpe figure is frictionless: no transaction costs, no slippage, no capacity constraints. It is an upper bound on a strategy that cannot actually be traded at this cost, not an achievable return.

## Honest summary

The model beats the persistence baseline on **0 of 6** tickers.

Tickers where the model does **not** beat persistence: CMG, DPZ, LULU, SBUX, TGT, ULTA. Longer planted lags and lower betas are harder to recover from noisy daily spend, which is the expected outcome.
