# 0004 — The plan's "beats the baseline" bar was mathematically unreachable

**Type:** plan defect found at review (Task 13); changed the reporting logic
**Plan said:** `verdict()` declares a win when RMSE improves ≥2% over
persistence **and** IC gain > 0.02
**Shipped:** a win requires IC gain > 0.02 **and** IC > 0 **and** statistical
significance on overlap-adjusted sample size **and** RMSE not degraded

## The defect

Persistence predicts zero, so its RMSE ≈ σ_y. The best RMSE ratio a model can
achieve at correlation ρ with the target is √(1 − ρ²). A 2% improvement therefore
requires |ρ| ≥ 0.199.

In this dataset:

- the strongest feature-target correlation is **0.156** (→ at most 1.22% RMSE
  improvement),
- the best measured out-of-sample IC is **0.112** (→ at most 0.63%).

The 2% clause could never pass. And because the IC-gain clause was trivially
satisfied (persistence IC is forced to 0.0), the RMSE clause bound alone.

**Consequence: the leakage canary was dead code.** The design's safeguard was "if
all six tickers beat the baseline, stop — that is evidence of a leak, not of
skill." A verdict function that can never say "beats" can never fire that canary.
The single check protecting against the most dangerous class of error in the
project was structurally incapable of triggering.

## A second, separate defect it was masking

While re-deriving the numbers the reviewer also measured the model itself:
5 of 6 ICs negative, RMSE ratios 1.08–1.27 (8–27% *worse* than predicting zero),
and `std(y_pred)` at 0.020–0.029 against `std(y_true)` 0.033–0.058 with ρ ≈ 0 —
roughly 50% of the target's amplitude at approximately zero correlation, which is
unshrunk overfit noise. That is a model calibration bug, not a reporting bug, and
it was only visible because someone recomputed rather than read the table.

## Two overclaiming bugs in the same file

- `report.py` rationalised the loss with a lag/beta narrative the data
  contradicts — CMG has the shortest planted lag *and* the highest beta and still
  loses. An explanation that the data refutes is worse than no explanation.
- ULTA printed "beats ARIMA" on IC −0.007 against ARIMA's IC −0.118. ARIMA is
  anti-skilled here (on LULU its RMSE is 0.0955 against persistence's 0.0581), so
  "beats" against a broken baseline reads as success while describing two
  failures.

## The fix, and the instruction attached to it

`verdict()` now requires IC gain > 0.02 **and** IC > 0 in absolute terms **and**
significance on `n_eff = n / horizon` (returns overlap at a 5-day horizon, so the
effective sample is one fifth of the row count) **and** no RMSE degradation.
The unsupported causal narrative was deleted. "Beats" is gated on absolute skill,
not on out-performing an anti-skilled baseline.

The fix instruction was explicit: **do not tune toward beating the baseline —
target calibration, not a win. If all six now read "beats", STOP and investigate,
because the canary is live again.**

## Measured outcome

| | Before | After |
|---|---|---|
| RMSE ratios (model / persistence) | 1.086, 1.268, 1.126, 1.081, 1.167, 1.206 | 1.007, 1.008, 1.011, 1.004, 1.028, 1.049 |
| mean ratio | 1.156 | 1.018 |
| `std(y_pred)/std(y_true)` | ≈ 0.50 at ρ ≈ 0 | 0.11–0.20 |
| long/short Sharpe (frictionless) | −0.16 | 0.24 |
| tickers beating persistence | 0 of 6 | **0 of 6** |

The regularisation applied is uniform, not per-ticker tuned. The leak detector
still passes with margin +0.0232. SBUX has the best IC (0.114) but t ≈ 1.13 on
n_eff ≈ 99, so it is reported **inconclusive** — not a win.

The model got honestly better and still does not beat the baseline. That is the
result, and it is what `docs/forecast-report.md` says.
