# Forecast Report

Model: `hgb-v1` — HistGradientBoostingRegressor
Target: forward 5-day return
Validation: purged walk-forward CV, 5-day purge/embargo

Prices in this project are synthetic with a planted per-ticker spend→return relationship (ADR-0004). These results measure whether the pipeline preserves a known signal, **not** real-world alpha.

## Per-ticker results

| Ticker | RMSE model | RMSE persist | RMSE ARIMA | Dir. acc | IC | vs persistence | vs ARIMA |
|---|---|---|---|---|---|---|---|
| CMG | 0.04736 | 0.04702 | 0.04649 | 0.528 | 0.010 | **inconclusive** | **loses** |
| DPZ | 0.03415 | 0.03386 | 0.03603 | 0.530 | -0.003 | **inconclusive** | **inconclusive** |
| LULU | 0.05875 | 0.05810 | 0.09548 | 0.550 | -0.017 | **inconclusive** | **inconclusive** |
| SBUX | 0.04538 | 0.04518 | 0.06086 | 0.500 | 0.114 | **inconclusive** | **inconclusive** |
| TGT | 0.03974 | 0.03866 | 0.05188 | 0.446 | -0.114 | **loses** | **loses** |
| ULTA | 0.05232 | 0.04988 | 0.07856 | 0.486 | 0.007 | **loses** | **inconclusive** |

## Portfolio

Long/short Sharpe (top-2 / bottom-2, daily rebalance): **0.24**

This Sharpe figure is frictionless: no transaction costs, no slippage, no capacity constraints. It is an upper bound on a strategy that cannot actually be traded at this cost, not an achievable return.

## Honest summary

The model beats the persistence baseline on **0 of 6** tickers.

Tickers where the model does **not** beat persistence: CMG, DPZ, LULU, SBUX, TGT, ULTA. Measured IC on these tickers ranges from -0.114 to 0.114 -- consistent with the modest planted signal this dataset carries overall (Task 10: max |feature-target correlation| 0.156), not with any pattern by planted lag or beta.
