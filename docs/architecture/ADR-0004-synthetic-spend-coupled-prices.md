# ADR-0004: Synthetic spend-coupled prices instead of real market data

**Status:** Accepted
**Date:** 2026-08-09

## Context

The hypothesis under test is:

> Aggregated consumer card spend at a company's merchants is a leading indicator
> of that company's forward stock returns.

Testing it needs two aligned series: card transactions and daily equity prices.
Real card-transaction panels are not publicly available. So the realistic
options were:

**(a) Real prices + synthetic spend.** This looks more credible and is
scientifically worthless. Synthetic spend is generated independently of real
price history, so the true correlation between them is **zero by construction**.
Any model trained on the pair is fitting noise, and every reported metric would
be an estimate of nothing. Worse, the noise occasionally produces a flattering
number, and the repository would then be presenting a fabricated result under a
real ticker symbol.

**(b) Synthetic prices + synthetic spend, coupled by a known mechanism.** This
looks less credible and answers a question that can actually be answered:
*did the pipeline preserve a signal that is known to be present?*

The second question is the engineering question. Leakage, purging, join
correctness, feature-window correctness, and calibration are all testable
against a planted ground truth and untestable against noise.

## Decision

Generate both feeds from a seeded generator, with the price series coupled to
the demand series at a **known per-ticker lag and beta**:

| Ticker | Planted lag (trading days) | Planted beta |
|---|---|---|
| SBUX | 3 | 0.35 |
| CMG | 2 | 0.45 |
| TGT | 5 | 0.25 |
| LULU | 7 | 0.40 |
| DPZ | 2 | 0.30 |
| ULTA | 5 | 0.35 |

The planted signal is defined in exactly one place, `src/config.py`, and nowhere
else.

Deliberate design choices inside the generator:

- **Per-ticker heterogeneity.** Distinct lags and betas mean no single global
  rule fits all six. A model that fits all six equally well is evidence of a
  leak, not of skill.
- **A 60% noise floor.** Roughly 60% of transactions are at untracked merchants
  (grocery, gas, rent, payroll, streaming, ATM), so the merchant→ticker join has
  to actually discriminate. The noise merchants have zero overlap with
  `silver.merchant_ticker_map` and cannot join.
- **A modest beta.** A 4× beta multiplier was tried during development and
  reverted: it drove the at-lag correlation to ≈0.8 and made the forecasting
  task trivial, which would have made the leakage tests meaningless.

The planting is verified, not assumed. At the configured lag, correlations are
SBUX 0.295, CMG 0.368, TGT 0.258, LULU 0.354, DPZ 0.309, ULTA 0.419. At a wrong
lag all six fall below 0.055. An independent lag sweep over offsets 0–12 finds
the **argmax lag equals the configured lag for all six tickers**.

## Consequences

**Positive.**

- `make all` is deterministic and offline: same seed, byte-identical output,
  verified across processes.
- The leakage tests mean something. Purged walk-forward CV, the no-lookahead
  recomputation, and the "if all six beat the baseline, stop and investigate"
  canary are all only interpretable because ground truth is known.
- A negative result is informative. The pipeline preserves a signal whose exact
  strength is known, so "the model did not beat persistence" can be attributed
  to model capacity and sample size rather than to a broken join.

**Negative, and — this is the important one — must be stated everywhere the
results are.**

- **These results measure pipeline fidelity, not real alpha.** No metric in this
  repository is evidence that card-spend data predicts returns. The
  correlation was planted by the generator. The engineering is real; the alpha
  is synthetic and labelled as such in `docs/forecast-report.md`, in the README,
  and in a test (`test_readme_does_not_overclaim_real_alpha`) that fails if the
  README stops saying so.
- The generator's day-of-week factor cycles three patterns over six companies, so
  SBUX/LULU, CMG/DPZ, and TGT/ULTA share weekday shape. Lags and betas still
  differ, so the heterogeneity argument holds, but the series are not fully
  independent.
- Nothing here transfers to real market microstructure: no gaps, no splits, no
  earnings jumps, no regime changes, no survivorship considerations.
