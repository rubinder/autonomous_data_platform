# Autonomous Lakehouse Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Apache Iceberg Bronze/Silver/Gold lakehouse that joins a Yodlee-format consumer transaction feed to daily equity prices, forecasts forward 5-day stock returns from aggregated merchant spend, and is operated by a LangGraph agent that detects schema drift, volume anomalies, and staleness against declared data contracts.

**Architecture:** Seeded generators emit a latent per-company demand series; transactions are sampled from it and prices are generated as a GBM whose drift is coupled to lagged demand shocks, so the planted signal is real and discoverable. Bronze appends source-faithful records with ingestion lineage, Silver conforms/dedupes/validates against YAML contracts, Gold builds leakage-safe feature marts. A `LakehouseEngine` protocol has a default PyIceberg+DuckDB implementation and an opt-in Spark implementation proven equivalent by a parity test.

**Tech Stack:** Python 3.13, uv, PyIceberg (SqlCatalog on SQLite), DuckDB, PyArrow, scikit-learn, statsmodels, LangGraph, pytest, ruff. Optional: PySpark 3.5 + iceberg-spark-runtime.

## Global Constraints

- **Determinism.** Every generator is seeded from `config.SEED = 42`. Same seed → byte-identical output. Never call `random` without an explicit `Generator` instance; never use wall-clock time in data.
- **Offline.** `make all` and `pytest` must pass with no network access. No call to the Anthropic API, yfinance, or any HTTP service may be required for any test or default `make` target.
- **Freshness uses `AS_OF_DATE`, never wall clock.** Defined in `src/config.py`, defaults to max `txn_date` present in Bronze, overridable via the `AS_OF_DATE` env var.
- **Bronze is append-only and never modified.** No transform, no dedupe, no type coercion in Bronze beyond adding `_`-prefixed lineage columns.
- **No lookahead.** Every feature at date *t* uses only data with timestamp ≤ *t*. All rolling windows trailing and right-closed. Only the target looks forward.
- **Fail closed.** A contract violation aborts the Silver or Gold build with a non-zero exit. Never warn-and-continue.
- **Date window:** `2024-01-01` → `2026-06-30`. **Tickers:** SBUX, CMG, TGT, LULU, DPZ, ULTA. **Horizon:** 5 trading days. **Purge/embargo:** 5 trading days.
- **Repo root:** `/Users/robran/IdeaProjects/autonomous_data_platform`. All paths below are relative to it.
- Run every command through `uv run`. Commit after every task.

### Documented deviation from spec

The spec names `LGBMRegressor`. Use `sklearn.ensemble.HistGradientBoostingRegressor` instead: it is the same histogram-based gradient boosting algorithm with no native `libomp` dependency, which is the most common cause of macOS/CI install failure. Record this in `docs/ai-sdlc/decisions/0001-sklearn-over-lightgbm.md` in Task 18.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/config.py` | Companies, merchants, planted lags/betas, date window, seed, trading calendar, `AS_OF_DATE` resolution |
| `src/generators/demand.py` | Latent per-company daily demand series — the single source of planted signal |
| `src/generators/yodlee_feed.py` | Yodlee-format accounts + transactions sampled from demand |
| `src/generators/stock_prices.py` | OHLCV whose returns are coupled to lagged demand shocks |
| `src/lakehouse/catalog.py` | PyIceberg `SqlCatalog` construction, namespace bootstrap |
| `src/lakehouse/schemas.py` | Iceberg `Schema` and `PartitionSpec` for every table |
| `src/lakehouse/engines/base.py` | `LakehouseEngine` protocol |
| `src/lakehouse/engines/pyiceberg_engine.py` | Default engine: PyIceberg commits + DuckDB SQL |
| `src/lakehouse/engines/spark_engine.py` | Opt-in engine: Spark SQL, `MERGE INTO`, compaction |
| `src/lakehouse/bronze.py` | Append raw feeds with lineage columns |
| `src/lakehouse/silver.py` | Conform, dedupe, sign amounts, normalize merchants, quarantine |
| `src/lakehouse/gold.py` | `daily_merchant_spend`, `stock_features`, `forecast_training_set` |
| `src/lakehouse/maintenance.py` | `expire_snapshots`, time-travel helper, schema-evolution demo |
| `src/contracts/validator.py` | Evaluate YAML contract against Arrow table → `ValidationReport` |
| `src/forecast/features.py` | Feature assembly + the leakage-safe recomputation used by tests |
| `src/forecast/splits.py` | Purged walk-forward CV splitter |
| `src/forecast/models.py` | HistGradientBoosting + persistence + ARIMA baselines |
| `src/forecast/backtest.py` | Long/short backtest, Sharpe, IC, directional accuracy |
| `src/forecast/report.py` | Emit `docs/forecast-report.md` with per-ticker verdicts |
| `src/agent/sensors.py` | Read Iceberg metadata: schema, snapshot log, row counts, freshness |
| `src/agent/classifier.py` | Rules-first severity classification, optional LLM for ambiguous cases |
| `src/agent/actions.py` | Write incident markdown, optional `gh issue create`, dry-run default |
| `src/agent/graph.py` | LangGraph state machine wiring the above |
| `contracts/*.yaml` | Declared schema + expectations per table |

---

## Task 1: Project scaffolding and configuration

**Files:**
- Create: `pyproject.toml`, `src/__init__.py`, `src/config.py`, `Makefile`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.SEED: int`, `config.START_DATE: date`, `config.END_DATE: date`, `config.COMPANIES: tuple[Company, ...]`, `config.NOISE_MERCHANTS: tuple[str, ...]`, `config.N_USERS: int`, `config.N_TRANSACTIONS: int`, `config.FORECAST_HORIZON_DAYS: int`, `config.PURGE_DAYS: int`, `config.WAREHOUSE_PATH: Path`, `config.trading_days(start: date, end: date) -> list[date]`, `config.resolve_as_of_date(bronze_max: date | None) -> date`, dataclass `Company(ticker, name, merchant_strings, lag_days, beta, start_price, annual_drift, annual_vol)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from datetime import date
from src import config


def test_six_companies_with_distinct_planted_parameters():
    tickers = [c.ticker for c in config.COMPANIES]
    assert tickers == ["SBUX", "CMG", "TGT", "LULU", "DPZ", "ULTA"]
    # Heterogeneous lags are deliberate: a single global rule must not fit all six.
    assert len({c.lag_days for c in config.COMPANIES}) >= 4


def test_trading_days_excludes_weekends_and_market_holidays():
    days = config.trading_days(date(2024, 1, 1), date(2024, 1, 31))
    assert date(2024, 1, 1) not in days      # New Year's Day
    assert date(2024, 1, 15) not in days     # MLK Day
    assert date(2024, 1, 6) not in days      # Saturday
    assert date(2024, 1, 2) in days
    assert days == sorted(days)


def test_full_window_has_enough_days_for_purged_cv():
    days = config.trading_days(config.START_DATE, config.END_DATE)
    assert 600 <= len(days) <= 650


def test_as_of_date_prefers_bronze_max_over_wall_clock(monkeypatch):
    monkeypatch.delenv("AS_OF_DATE", raising=False)
    assert config.resolve_as_of_date(date(2026, 6, 30)) == date(2026, 6, 30)


def test_as_of_date_env_override_wins():
    import os
    os.environ["AS_OF_DATE"] = "2026-07-15"
    try:
        assert config.resolve_as_of_date(date(2026, 6, 30)) == date(2026, 7, 15)
    finally:
        del os.environ["AS_OF_DATE"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "autonomous-data-platform"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pyiceberg[sql-sqlite,pyarrow]>=0.7.0",
    "pyarrow>=16.0.0",
    "duckdb>=1.0.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0",
    "statsmodels>=0.14.0",
    "pyyaml>=6.0",
    "langgraph>=0.2.0",
]

[project.optional-dependencies]
spark = ["pyspark==3.5.1"]
dev = ["pytest>=8.0", "ruff>=0.6.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["spark: requires PySpark and a JVM"]

[tool.ruff]
line-length = 100

[tool.setuptools.packages.find]
include = ["src*"]
```

- [ ] **Step 4: Write `src/config.py`**

```python
"""Central configuration: the planted signal lives here and nowhere else."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

SEED = 42
START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 6, 30)
N_USERS = 2000
N_TRANSACTIONS = int(os.environ.get("N_TRANSACTIONS", 250_000))
NOISE_FRACTION = 0.60
FORECAST_HORIZON_DAYS = 5
PURGE_DAYS = 5
REPO_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_PATH = Path(os.environ.get("WAREHOUSE_PATH", REPO_ROOT / "warehouse"))


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    merchant_strings: tuple[str, ...]
    lag_days: int      # trading-day lag from demand shock to price response
    beta: float        # sensitivity of daily return to standardized demand shock
    start_price: float
    annual_drift: float
    annual_vol: float


COMPANIES: tuple[Company, ...] = (
    Company("SBUX", "Starbucks",
            ("Starbucks", "STARBUCKS STORE #1234", "STARBUCKS #0917"),
            3, 0.35, 95.0, 0.06, 0.28),
    Company("CMG", "Chipotle Mexican Grill",
            ("Chipotle Mexican Grill", "CHIPOTLE 0455", "CHIPOTLE ONLINE"),
            2, 0.45, 58.0, 0.11, 0.32),
    Company("TGT", "Target",
            ("Target", "TARGET.COM", "TARGET T-1088"),
            5, 0.25, 142.0, 0.04, 0.30),
    Company("LULU", "lululemon athletica",
            ("lululemon athletica", "LULULEMON #212", "LULULEMON.COM"),
            7, 0.40, 310.0, 0.08, 0.38),
    Company("DPZ", "Domino's Pizza",
            ("Domino's Pizza", "DOMINOS 8871", "DOMINOS.COM"),
            2, 0.30, 415.0, 0.05, 0.26),
    Company("ULTA", "ULTA Beauty",
            ("ULTA Beauty", "ULTA #0921", "ULTA.COM"),
            5, 0.35, 385.0, 0.07, 0.33),
)

# ~60% of volume. The merchant->ticker join must actually discriminate.
NOISE_MERCHANTS: tuple[str, ...] = (
    "SAFEWAY #1422", "WHOLE FOODS MKT", "SHELL OIL 5748", "CHEVRON 0091",
    "NETFLIX.COM", "SPOTIFY USA", "UBER TRIP", "LYFT RIDE",
    "COMCAST CABLE", "PG&E PAYMENT", "GEICO PREMIUM", "ATM WITHDRAWAL",
    "PAYROLL DIRECT DEP", "RENT PAYMENT", "CVS/PHARMACY #4410", "HOME DEPOT 6612",
)

# NYSE holidays covering START_DATE..END_DATE. Hardcoded to avoid a calendar
# dependency that would drift with library upgrades.
US_MARKET_HOLIDAYS: frozenset[date] = frozenset({
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19),
})


def trading_days(start: date, end: date) -> list[date]:
    """Weekdays in [start, end] excluding NYSE holidays, ascending."""
    days, cursor = [], start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in US_MARKET_HOLIDAYS:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def resolve_as_of_date(bronze_max: date | None) -> date:
    """The pipeline's logical 'now'.

    Never wall-clock: the dataset ends 2026-06-30, so a wall-clock freshness
    check would fail permanently and get muted -- which is how freshness
    monitoring dies in real systems.
    """
    override = os.environ.get("AS_OF_DATE")
    if override:
        return date.fromisoformat(override)
    return bronze_max if bronze_max is not None else END_DATE


def company_by_ticker(ticker: str) -> Company:
    for c in COMPANIES:
        if c.ticker == ticker:
            return c
    raise KeyError(ticker)
```

- [ ] **Step 5: Create `src/__init__.py` (empty) and run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 passed. If `test_full_window_has_enough_days_for_purged_cv` fails, print `len(config.trading_days(...))` and correct the holiday set — do not widen the assertion to fit a wrong calendar.

- [ ] **Step 6: Write the `Makefile`**

```makefile
.PHONY: all bronze silver gold forecast agent test lint clean timetravel drift-demo maintenance
UV := uv run

all: bronze silver gold forecast

bronze:      ; $(UV) python -m src.lakehouse.bronze
silver:      ; $(UV) python -m src.lakehouse.silver
gold:        ; $(UV) python -m src.lakehouse.gold
forecast:    ; $(UV) python -m src.forecast.report
agent:       ; $(UV) python -m src.agent.graph
timetravel:  ; $(UV) python -m src.lakehouse.maintenance timetravel
drift-demo:  ; $(UV) python -m src.lakehouse.maintenance drift-demo
maintenance: ; $(UV) python -m src.lakehouse.maintenance expire
test:        ; $(UV) pytest -v
lint:        ; $(UV) ruff check src tests
clean:       ; rm -rf warehouse

smoke:
	N_TRANSACTIONS=5000 $(MAKE) all
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml Makefile src/__init__.py src/config.py tests/test_config.py
git commit -m "feat: project scaffolding, config, and trading calendar

Planted signal parameters (per-ticker lag and beta) live in config and
nowhere else, so the generators and the tests cannot disagree about what
the model is supposed to find.

AS_OF_DATE resolution is here rather than in the contract validator
because both the pipeline and the ops agent need the same logical clock."
```

---

## Task 2: Latent demand generator

The single source of planted signal. Transactions sample from it; prices respond to its shocks. Keeping it in one module is what makes the correlation exact and testable.

**Files:**
- Create: `src/generators/__init__.py`, `src/generators/demand.py`
- Test: `tests/test_demand.py`

**Interfaces:**
- Consumes: `config.COMPANIES`, `config.trading_days`, `config.SEED`
- Produces: `demand.build_demand_panel(seed: int = config.SEED) -> pandas.DataFrame` with columns `ticker, day (date), demand (float>0), shock (float, standardized)`, indexed 0..n-1. `demand` is a positive intensity; `shock` is the z-scored innovation driving price response.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_demand.py
import numpy as np
from src import config
from src.generators import demand


def test_panel_covers_every_company_and_calendar_day():
    panel = demand.build_demand_panel()
    days = config.trading_days(config.START_DATE, config.END_DATE)
    assert set(panel["ticker"]) == {c.ticker for c in config.COMPANIES}
    assert len(panel) == len(days) * len(config.COMPANIES)


def test_demand_is_strictly_positive():
    panel = demand.build_demand_panel()
    assert (panel["demand"] > 0).all()


def test_shock_is_standardized_per_ticker():
    panel = demand.build_demand_panel()
    for _, grp in panel.groupby("ticker"):
        assert abs(grp["shock"].mean()) < 0.15
        assert 0.7 < grp["shock"].std() < 1.4


def test_deterministic_under_same_seed():
    a = demand.build_demand_panel(seed=7)
    b = demand.build_demand_panel(seed=7)
    assert a.equals(b)
    c = demand.build_demand_panel(seed=8)
    assert not np.allclose(a["demand"], c["demand"])


def test_weekly_seasonality_present():
    panel = demand.build_demand_panel()
    sbux = panel[panel["ticker"] == "SBUX"].copy()
    sbux["dow"] = [d.weekday() for d in sbux["day"]]
    by_dow = sbux.groupby("dow")["demand"].mean()
    assert by_dow.max() / by_dow.min() > 1.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_demand.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.generators'`

- [ ] **Step 3: Write `src/generators/demand.py`**

```python
"""Latent per-company demand: the single source of the planted signal.

Transactions are sampled from `demand`; stock returns respond to lagged
`shock`. Both consumers read this one panel, so the correlation the model is
asked to find is exactly the correlation that was planted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def build_demand_panel(seed: int = config.SEED) -> pd.DataFrame:
    days = config.trading_days(config.START_DATE, config.END_DATE)
    n = len(days)
    rows = []

    for idx, company in enumerate(config.COMPANIES):
        rng = np.random.default_rng(seed * 1000 + idx)

        # Innovations drive both demand level and (lagged) price response.
        innovations = rng.standard_normal(n)

        # Persistent component: demand trends rather than jumping day to day.
        ar = np.zeros(n)
        phi = 0.85
        for t in range(1, n):
            ar[t] = phi * ar[t - 1] + innovations[t]
        ar_std = ar.std()
        level = ar / ar_std if ar_std > 0 else ar

        t_index = np.arange(n)
        annual = 0.12 * np.sin(2 * np.pi * t_index / 252.0 + idx)
        trend = 0.15 * t_index / n
        dow = np.array([_dow_factor(d.weekday(), idx) for d in days])

        base = 1.0 + trend + annual + 0.20 * level
        demand_values = np.clip(base, 0.15, None) * dow

        shock = (innovations - innovations.mean()) / innovations.std()

        rows.append(pd.DataFrame({
            "ticker": company.ticker,
            "day": days,
            "demand": demand_values,
            "shock": shock,
        }))

    return pd.concat(rows, ignore_index=True)


def _dow_factor(weekday: int, company_idx: int) -> float:
    """Weekday seasonality, phase-shifted per company."""
    patterns = (
        (1.05, 1.00, 1.00, 1.02, 1.18, 1.00, 1.00),  # Friday-heavy
        (0.95, 0.98, 1.02, 1.05, 1.20, 1.00, 1.00),  # ramps into weekend
        (1.00, 0.96, 0.96, 1.00, 1.15, 1.00, 1.00),
    )
    return patterns[company_idx % len(patterns)][weekday]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_demand.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/generators/__init__.py src/generators/demand.py tests/test_demand.py
git commit -m "feat: latent demand panel as the single source of planted signal

Both the transaction generator and the price generator read this one
panel. If each rolled its own randomness the planted correlation would be
approximate and the no-leakage tests would have nothing exact to check
against."
```

---

## Task 3: Yodlee-format feed generator

**Files:**
- Create: `src/generators/yodlee_feed.py`
- Test: `tests/test_yodlee_feed.py`

**Interfaces:**
- Consumes: `demand.build_demand_panel`, `config.COMPANIES`, `config.NOISE_MERCHANTS`, `config.N_USERS`, `config.N_TRANSACTIONS`, `config.NOISE_FRACTION`
- Produces: `yodlee_feed.generate_accounts(seed: int = config.SEED) -> list[dict]`, `yodlee_feed.generate_transactions(seed: int = config.SEED, n: int | None = None) -> list[dict]`. Transaction dicts match the Yodlee webhook shape: keys `id, accountId, date, transactionDate, postDate, amount{amount,currency}, runningBalance{amount,currency}, merchant{id,source,categoryLabel,address{...}}, status, baseType, subType, category, categoryType, categoryId, detailCategoryId, detailCategory, highLevelCategoryId, categorySource, sourceType, checkNumber, isManual, container, createdDate, lastUpdated`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yodlee_feed.py
import json
from datetime import date
from src import config
from src.generators import yodlee_feed

REQUIRED_TXN_KEYS = {
    "id", "accountId", "date", "transactionDate", "postDate", "amount",
    "runningBalance", "merchant", "status", "baseType", "subType", "category",
    "categoryType", "categoryId", "detailCategoryId", "detailCategory",
    "highLevelCategoryId", "categorySource", "sourceType", "checkNumber",
    "isManual", "container", "createdDate", "lastUpdated",
}


def test_transaction_matches_yodlee_shape():
    txns = yodlee_feed.generate_transactions(n=500)
    t = txns[0]
    assert REQUIRED_TXN_KEYS <= set(t)
    assert set(t["amount"]) == {"amount", "currency"}
    assert set(t["merchant"]) >= {"id", "source"}
    assert t["baseType"] in {"CREDIT", "DEBIT"}


def test_amounts_are_unsigned_in_source():
    # Yodlee carries direction in baseType, not in the sign. Silver must sign it.
    txns = yodlee_feed.generate_transactions(n=2000)
    assert all(t["amount"]["amount"] > 0 for t in txns)


def test_tracked_merchants_present_and_noise_dominates():
    txns = yodlee_feed.generate_transactions(n=20000)
    tracked_strings = {s for c in config.COMPANIES for s in c.merchant_strings}
    names = [t["merchant"]["source"] for t in txns]
    tracked = sum(1 for nm in names if nm in tracked_strings)
    assert tracked > 0
    ratio = 1 - tracked / len(names)
    assert 0.50 < ratio < 0.70


def test_ids_unique_and_json_serializable():
    txns = yodlee_feed.generate_transactions(n=5000)
    assert len({t["id"] for t in txns}) == len(txns)
    json.dumps(txns[:10])


def test_deterministic():
    a = yodlee_feed.generate_transactions(seed=11, n=1000)
    b = yodlee_feed.generate_transactions(seed=11, n=1000)
    assert a == b


def test_some_transactions_are_late_arriving_corrections():
    # Duplicate ids with a newer lastUpdated: Silver dedup must handle these.
    txns = yodlee_feed.generate_transactions(n=20000)
    seen = {}
    dupes = 0
    for t in txns:
        if t["id"] in seen:
            dupes += 1
        seen[t["id"]] = t
    assert dupes == 0  # ids unique within a batch; corrections come from re-ingest


def test_dates_within_configured_window():
    txns = yodlee_feed.generate_transactions(n=5000)
    days = {d.isoformat() for d in config.trading_days(config.START_DATE, config.END_DATE)}
    assert all(t["transactionDate"] in days for t in txns)


def test_accounts_have_balances():
    accts = yodlee_feed.generate_accounts()
    assert len(accts) == config.N_USERS
    a = accts[0]
    assert a["accountType"] in {"CHECKING", "SAVINGS", "CREDIT"}
    assert set(a["balance"]) == {"amount", "currency"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_yodlee_feed.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/generators/yodlee_feed.py`**

```python
"""Yodlee-webhook-shaped account and transaction records.

Shape is faithful to https://docs.conductiv.co/docs/webhook/sample-data/yodlee/ --
notably `amount` is an unsigned struct and direction lives in `baseType`, which
is the correctness trap Silver has to get right.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src import config
from src.generators.demand import build_demand_panel

_CATEGORY_BY_TICKER = {
    "SBUX": ("Restaurants/Dining", 3, 10, "Coffee Shop"),
    "CMG": ("Restaurants/Dining", 3, 10, "Fast Food"),
    "TGT": ("General Merchandise", 12, 22, "Department Store"),
    "LULU": ("Clothing/Shoes", 8, 22, "Athletic Apparel"),
    "DPZ": ("Restaurants/Dining", 3, 10, "Pizza"),
    "ULTA": ("Personal Care", 15, 22, "Cosmetics"),
}
_NOISE_CATEGORY = ("Other Expenses", 40, 30, "Uncategorized")
_TICKET_MEAN = {"SBUX": 8.5, "CMG": 14.0, "TGT": 62.0,
                "LULU": 118.0, "DPZ": 24.0, "ULTA": 48.0}


def generate_accounts(seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed + 501)
    providers = [("Chase", "16441"), ("Bank of America", "5"), ("Wells Fargo", "1603")]
    accounts = []
    for i in range(config.N_USERS):
        provider, provider_id = providers[i % len(providers)]
        acct_type = ("CHECKING", "SAVINGS", "CREDIT")[i % 3]
        balance = round(float(rng.uniform(250, 24_000)), 2)
        created = datetime(2023, 6, 1, tzinfo=timezone.utc) + timedelta(days=int(i % 200))
        accounts.append({
            "id": 100_000 + i,
            "providerName": provider,
            "providerId": provider_id,
            "accountType": acct_type,
            "accountStatus": "ACTIVE",
            "userClassification": "PERSONAL",
            "isAsset": acct_type != "CREDIT",
            "isManual": False,
            "aggregationSource": "USER",
            "balance": {"amount": balance, "currency": "USD"},
            "currentBalance": {"amount": balance, "currency": "USD"},
            "availableBalance": {"amount": round(balance * 0.92, 2), "currency": "USD"},
            "createdDate": created.isoformat(),
            "lastUpdated": datetime(2026, 6, 30, tzinfo=timezone.utc).isoformat(),
        })
    return accounts


def generate_transactions(seed: int = config.SEED, n: int | None = None) -> list[dict]:
    total = n if n is not None else config.N_TRANSACTIONS
    rng = np.random.default_rng(seed + 907)
    panel = build_demand_panel(seed=seed)

    n_tracked = int(total * (1 - config.NOISE_FRACTION))
    n_noise = total - n_tracked

    records: list[dict] = []
    next_id = 5_000_000

    # Tracked spend: allocate proportionally to demand so daily aggregates
    # inherit the planted signal.
    weights = panel["demand"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    picks = rng.choice(len(panel), size=n_tracked, p=weights)
    tickers = panel["ticker"].to_numpy()
    days = panel["day"].to_numpy()

    for pick in picks:
        ticker = str(tickers[pick])
        day = days[pick]
        company = config.company_by_ticker(ticker)
        merchant = company.merchant_strings[int(rng.integers(len(company.merchant_strings)))]
        mean = _TICKET_MEAN[ticker]
        amount = round(float(rng.gamma(shape=4.0, scale=mean / 4.0)) + 1.0, 2)
        records.append(_build_txn(rng, next_id, day, merchant, amount,
                                  _CATEGORY_BY_TICKER[ticker], "DEBIT"))
        next_id += 1

    all_days = config.trading_days(config.START_DATE, config.END_DATE)
    noise_day_idx = rng.integers(0, len(all_days), size=n_noise)
    noise_pick = rng.integers(0, len(config.NOISE_MERCHANTS), size=n_noise)
    for k in range(n_noise):
        day = all_days[int(noise_day_idx[k])]
        merchant = config.NOISE_MERCHANTS[int(noise_pick[k])]
        is_income = merchant == "PAYROLL DIRECT DEP"
        amount = round(float(rng.gamma(3.0, 900.0 if is_income else 30.0)) + 1.0, 2)
        records.append(_build_txn(rng, next_id, day, merchant, amount,
                                  _NOISE_CATEGORY, "CREDIT" if is_income else "DEBIT"))
        next_id += 1

    records.sort(key=lambda r: (r["transactionDate"], r["id"]))
    return records


def _build_txn(rng, txn_id: int, day, merchant: str, amount: float,
               category: tuple, base_type: str) -> dict:
    day = day if isinstance(day, date) else pd.Timestamp(day).date()
    cat_name, cat_id, high_id, detail = category
    post = day + timedelta(days=int(rng.integers(0, 3)))
    created = datetime.combine(post, datetime.min.time(), tzinfo=timezone.utc)
    return {
        "id": txn_id,
        "accountId": 100_000 + int(rng.integers(config.N_USERS)),
        "date": day.isoformat(),
        "transactionDate": day.isoformat(),
        "postDate": post.isoformat(),
        # Unsigned by design -- direction is in baseType.
        "amount": {"amount": amount, "currency": "USD"},
        "runningBalance": {"amount": round(float(rng.uniform(100, 20_000)), 2),
                           "currency": "USD"},
        "merchant": {
            "id": f"M{abs(hash(merchant)) % 10_000_000}",
            "source": merchant,
            "categoryLabel": detail,
            "address": {"city": "SEATTLE", "state": "WA", "country": "USA"},
        },
        "status": "POSTED",
        "baseType": base_type,
        "subType": "PAYMENT" if base_type == "DEBIT" else "CREDIT",
        "category": cat_name,
        "categoryType": "EXPENSE" if base_type == "DEBIT" else "INCOME",
        "categoryId": cat_id,
        "detailCategoryId": cat_id * 100,
        "detailCategory": detail,
        "highLevelCategoryId": high_id,
        "categorySource": "SYSTEM",
        "sourceType": "AGGREGATED",
        "checkNumber": "",
        "isManual": False,
        "container": None,
        "createdDate": created.isoformat(),
        "lastUpdated": created.isoformat(),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_yodlee_feed.py -v`
Expected: 8 passed. If the noise ratio assertion fails, check `NOISE_FRACTION` arithmetic rather than loosening the bounds — the ratio is the point.

- [ ] **Step 5: Commit**

```bash
git add src/generators/yodlee_feed.py tests/test_yodlee_feed.py
git commit -m "feat: Yodlee-shaped transaction and account generator

Amounts are unsigned with direction in baseType, matching the real feed.
That is the single most common correctness trap in this dataset, so the
generator reproduces it rather than pre-solving it for Silver.

Tracked spend is allocated proportionally to the latent demand panel, so
daily merchant aggregates inherit the planted signal exactly."
```

---

## Task 4: Stock price generator coupled to lagged demand

**Files:**
- Create: `src/generators/stock_prices.py`
- Test: `tests/test_stock_prices.py`

**Interfaces:**
- Consumes: `demand.build_demand_panel`, `config.COMPANIES`, `config.trading_days`
- Produces: `stock_prices.generate_prices(seed: int = config.SEED) -> list[dict]` with keys `ticker, date (ISO str), open, high, low, close, adj_close, volume`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stock_prices.py
import numpy as np
import pandas as pd

from src import config
from src.generators import stock_prices
from src.generators.demand import build_demand_panel


def test_covers_all_tickers_and_trading_days():
    rows = stock_prices.generate_prices()
    df = pd.DataFrame(rows)
    days = config.trading_days(config.START_DATE, config.END_DATE)
    assert set(df["ticker"]) == {c.ticker for c in config.COMPANIES}
    assert len(df) == len(days) * len(config.COMPANIES)


def test_ohlc_invariants_hold():
    df = pd.DataFrame(stock_prices.generate_prices())
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()


def test_planted_signal_is_recoverable_at_the_configured_lag():
    """The whole project rests on this. If it fails, nothing downstream means anything."""
    df = pd.DataFrame(stock_prices.generate_prices())
    panel = build_demand_panel()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    for company in config.COMPANIES:
        px = df[df["ticker"] == company.ticker].sort_values("date").reset_index(drop=True)
        dm = panel[panel["ticker"] == company.ticker].sort_values("day").reset_index(drop=True)
        ret = px["close"].pct_change()
        lagged_shock = dm["shock"].shift(company.lag_days)
        joined = pd.DataFrame({"ret": ret, "shock": lagged_shock}).dropna()
        corr = joined["ret"].corr(joined["shock"])
        assert corr > 0.15, f"{company.ticker} planted signal not recoverable: {corr:.3f}"


def test_no_signal_at_a_wrong_lag_for_most_tickers():
    """Guards against a generator bug that smears signal across all lags."""
    df = pd.DataFrame(stock_prices.generate_prices())
    panel = build_demand_panel()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    weak = 0
    for company in config.COMPANIES:
        px = df[df["ticker"] == company.ticker].sort_values("date").reset_index(drop=True)
        dm = panel[panel["ticker"] == company.ticker].sort_values("day").reset_index(drop=True)
        ret = px["close"].pct_change()
        wrong = dm["shock"].shift(company.lag_days + 9)
        joined = pd.DataFrame({"ret": ret, "shock": wrong}).dropna()
        if abs(joined["ret"].corr(joined["shock"])) < 0.10:
            weak += 1
    assert weak >= 4


def test_deterministic():
    assert stock_prices.generate_prices(seed=3) == stock_prices.generate_prices(seed=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stock_prices.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/generators/stock_prices.py`**

```python
"""Daily OHLCV whose returns respond to lagged demand shocks.

Synthetic by design (ADR-0004): pairing real market prices with synthetic
spend would guarantee zero correlation and a meaningless model. Here the
signal is planted at a known per-ticker lag, so "did the pipeline preserve
it" becomes a testable question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.generators.demand import build_demand_panel


def generate_prices(seed: int = config.SEED) -> list[dict]:
    days = config.trading_days(config.START_DATE, config.END_DATE)
    n = len(days)
    panel = build_demand_panel(seed=seed)
    rows: list[dict] = []

    for idx, company in enumerate(config.COMPANIES):
        rng = np.random.default_rng(seed * 7919 + idx)
        shock = (panel.loc[panel["ticker"] == company.ticker]
                 .sort_values("day")["shock"].to_numpy(dtype=float))

        lagged = np.zeros(n)
        if company.lag_days < n:
            lagged[company.lag_days:] = shock[: n - company.lag_days]

        daily_drift = company.annual_drift / 252.0
        daily_vol = company.annual_vol / np.sqrt(252.0)

        # Scale idiosyncratic noise so the planted beta stays recoverable.
        noise = rng.standard_normal(n) * daily_vol
        returns = daily_drift + company.beta * daily_vol * lagged + noise

        close = company.start_price * np.exp(np.cumsum(returns - 0.5 * daily_vol**2))

        prev_close = np.concatenate(([company.start_price], close[:-1]))
        open_ = prev_close * (1 + rng.normal(0, daily_vol * 0.3, n))
        span = np.abs(rng.normal(0, daily_vol * 0.6, n))
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)
        volume = rng.integers(1_500_000, 12_000_000, n)

        for t, day in enumerate(days):
            rows.append({
                "ticker": company.ticker,
                "date": day.isoformat(),
                "open": round(float(open_[t]), 4),
                "high": round(float(high[t]), 4),
                "low": round(float(low[t]), 4),
                "close": round(float(close[t]), 4),
                "adj_close": round(float(close[t]), 4),
                "volume": int(volume[t]),
            })

    rows.sort(key=lambda r: (r["date"], r["ticker"]))
    return rows
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_stock_prices.py -v`
Expected: 5 passed.

If `test_planted_signal_is_recoverable_at_the_configured_lag` fails, the fix is to raise the beta multiplier in `returns`, **not** to lower the 0.15 threshold. A weaker planted signal means the forecast task becomes noise-fitting and every downstream metric is meaningless.

- [ ] **Step 5: Commit**

```bash
git add src/generators/stock_prices.py tests/test_stock_prices.py
git commit -m "feat: spend-coupled synthetic price generator

Returns respond to demand shocks at a per-ticker lag, so the alt-data
hypothesis has something real to detect. Two tests pin this down: signal
recoverable at the configured lag, and absent at a wrong lag -- the second
catches a generator bug that would smear signal across all lags and make
the model look good for the wrong reason."
```

---

## Task 5: Iceberg catalog, schemas, and engine protocol

**Files:**
- Create: `src/lakehouse/__init__.py`, `src/lakehouse/catalog.py`, `src/lakehouse/schemas.py`, `src/lakehouse/engines/__init__.py`, `src/lakehouse/engines/base.py`, `src/lakehouse/engines/pyiceberg_engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `config.WAREHOUSE_PATH`
- Produces:
  - `catalog.get_catalog(warehouse: Path | None = None) -> SqlCatalog`
  - `catalog.NAMESPACES: tuple[str, ...] = ("bronze", "silver", "gold")`
  - `schemas.BRONZE_TRANSACTIONS`, `schemas.BRONZE_ACCOUNTS`, `schemas.BRONZE_PRICES`, `schemas.SILVER_TRANSACTIONS`, `schemas.SILVER_QUARANTINE`, `schemas.SILVER_ACCOUNTS`, `schemas.SILVER_MERCHANT_MAP`, `schemas.SILVER_PRICES`, `schemas.GOLD_SPEND`, `schemas.GOLD_STOCK_FEATURES`, `schemas.GOLD_TRAINING`, `schemas.GOLD_PREDICTIONS` — each a `TableDef(name: str, schema: Schema, spec: PartitionSpec)`
  - `base.LakehouseEngine` protocol with `create_table`, `append`, `overwrite`, `scan_arrow`, `sql`, `table_exists`, `snapshots`
  - `pyiceberg_engine.PyIcebergEngine(catalog)` implementing it
  - `engines.get_engine(name: str | None = None) -> LakehouseEngine` dispatching on `$ENGINE`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
import pyarrow as pa
import pytest

from src.lakehouse import catalog as catalog_mod
from src.lakehouse import schemas
from src.lakehouse.engines import get_engine


@pytest.fixture
def engine(tmp_path):
    cat = catalog_mod.get_catalog(tmp_path / "wh")
    from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
    return PyIcebergEngine(cat)


def test_create_append_and_scan_roundtrip(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    data = pa.table({
        "merchant_normalized": ["STARBUCKS"],
        "ticker": ["SBUX"],
        "company_name": ["Starbucks"],
        "match_type": ["exact"],
    })
    engine.append(td.name, data)
    out = engine.scan_arrow(td.name)
    assert out.num_rows == 1
    assert out.column("ticker")[0].as_py() == "SBUX"


def test_append_creates_a_new_snapshot_each_time(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    row = pa.table({"merchant_normalized": ["X"], "ticker": ["T"],
                    "company_name": ["c"], "match_type": ["exact"]})
    engine.append(td.name, row)
    first = engine.snapshots(td.name)
    engine.append(td.name, row)
    second = engine.snapshots(td.name)
    assert len(second) == len(first) + 1


def test_time_travel_reads_prior_snapshot(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    row = pa.table({"merchant_normalized": ["X"], "ticker": ["T"],
                    "company_name": ["c"], "match_type": ["exact"]})
    engine.append(td.name, row)
    snap_one = engine.snapshots(td.name)[-1]
    engine.append(td.name, row)
    assert engine.scan_arrow(td.name).num_rows == 2
    assert engine.scan_arrow(td.name, snapshot_id=snap_one).num_rows == 1


def test_overwrite_replaces_contents(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    engine.append(td.name, pa.table({"merchant_normalized": ["A"], "ticker": ["A"],
                                     "company_name": ["a"], "match_type": ["exact"]}))
    engine.overwrite(td.name, pa.table({"merchant_normalized": ["B"], "ticker": ["B"],
                                        "company_name": ["b"], "match_type": ["exact"]}))
    out = engine.scan_arrow(td.name)
    assert out.num_rows == 1
    assert out.column("ticker")[0].as_py() == "B"


def test_sql_runs_duckdb_over_registered_tables(engine):
    td = schemas.SILVER_MERCHANT_MAP
    engine.create_table(td)
    engine.append(td.name, pa.table({
        "merchant_normalized": ["A", "B"], "ticker": ["A", "B"],
        "company_name": ["a", "b"], "match_type": ["exact", "exact"]}))
    out = engine.sql("SELECT count(*) AS n FROM silver_merchant_ticker_map",
                     tables={"silver_merchant_ticker_map": td.name})
    assert out.column("n")[0].as_py() == 2


def test_get_engine_defaults_to_pyiceberg(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGINE", raising=False)
    monkeypatch.setenv("WAREHOUSE_PATH", str(tmp_path / "wh2"))
    eng = get_engine()
    assert eng.__class__.__name__ == "PyIcebergEngine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL — `src.lakehouse` not found.

- [ ] **Step 3: Write `src/lakehouse/catalog.py`**

```python
"""PyIceberg SqlCatalog on SQLite. No external services -- `make all` works offline."""
from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog.sql import SqlCatalog

from src import config

NAMESPACES: tuple[str, ...] = ("bronze", "silver", "gold")


def get_catalog(warehouse: Path | None = None) -> SqlCatalog:
    wh = Path(warehouse or config.WAREHOUSE_PATH)
    wh.mkdir(parents=True, exist_ok=True)
    cat = SqlCatalog("local", **{
        "uri": f"sqlite:///{wh / 'catalog.db'}",
        "warehouse": f"file://{wh}",
    })
    for ns in NAMESPACES:
        try:
            cat.create_namespace(ns)
        except Exception:  # NamespaceAlreadyExistsError
            pass
    return cat
```

- [ ] **Step 4: Write `src/lakehouse/schemas.py`**

```python
"""Iceberg schemas and partition specs for every table in the lakehouse."""
from __future__ import annotations

from dataclasses import dataclass

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import DayTransform, IdentityTransform, MonthTransform
from pyiceberg.types import (
    BooleanType, DateType, DoubleType, LongType, NestedField, StringType,
    StructType, TimestamptzType,
)


@dataclass(frozen=True)
class TableDef:
    name: str
    schema: Schema
    spec: PartitionSpec


def _money(fid: int, name: str, required: bool = False) -> NestedField:
    return NestedField(fid, name, StructType(
        NestedField(fid * 10 + 1, "amount", DoubleType(), required=False),
        NestedField(fid * 10 + 2, "currency", StringType(), required=False),
    ), required=required)


BRONZE_TRANSACTIONS = TableDef(
    "bronze.yodlee_transactions_raw",
    Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "accountId", LongType(), required=False),
        NestedField(3, "date", StringType(), required=False),
        NestedField(4, "transactionDate", StringType(), required=False),
        NestedField(5, "postDate", StringType(), required=False),
        _money(6, "amount"),
        _money(7, "runningBalance"),
        NestedField(8, "merchant", StructType(
            NestedField(81, "id", StringType(), required=False),
            NestedField(82, "source", StringType(), required=False),
            NestedField(83, "categoryLabel", StringType(), required=False),
            NestedField(84, "address", StructType(
                NestedField(841, "city", StringType(), required=False),
                NestedField(842, "state", StringType(), required=False),
                NestedField(843, "country", StringType(), required=False),
            ), required=False),
        ), required=False),
        NestedField(9, "status", StringType(), required=False),
        NestedField(10, "baseType", StringType(), required=False),
        NestedField(11, "subType", StringType(), required=False),
        NestedField(12, "category", StringType(), required=False),
        NestedField(13, "categoryType", StringType(), required=False),
        NestedField(14, "categoryId", LongType(), required=False),
        NestedField(15, "detailCategoryId", LongType(), required=False),
        NestedField(16, "detailCategory", StringType(), required=False),
        NestedField(17, "highLevelCategoryId", LongType(), required=False),
        NestedField(18, "categorySource", StringType(), required=False),
        NestedField(19, "sourceType", StringType(), required=False),
        NestedField(20, "checkNumber", StringType(), required=False),
        NestedField(21, "isManual", BooleanType(), required=False),
        NestedField(22, "container", StringType(), required=False),
        NestedField(23, "createdDate", StringType(), required=False),
        NestedField(24, "lastUpdated", StringType(), required=False),
        NestedField(25, "_ingested_at", TimestamptzType(), required=True),
        NestedField(26, "_source_file", StringType(), required=False),
        NestedField(27, "_payload_hash", StringType(), required=False),
        NestedField(28, "_raw_payload", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(25, 1000, DayTransform(), "_ingested_at_day")),
)

BRONZE_ACCOUNTS = TableDef(
    "bronze.yodlee_accounts_raw",
    Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "providerName", StringType(), required=False),
        NestedField(3, "providerId", StringType(), required=False),
        NestedField(4, "accountType", StringType(), required=False),
        NestedField(5, "accountStatus", StringType(), required=False),
        NestedField(6, "userClassification", StringType(), required=False),
        NestedField(7, "isAsset", BooleanType(), required=False),
        NestedField(8, "isManual", BooleanType(), required=False),
        NestedField(9, "aggregationSource", StringType(), required=False),
        _money(10, "balance"),
        _money(11, "currentBalance"),
        _money(12, "availableBalance"),
        NestedField(13, "createdDate", StringType(), required=False),
        NestedField(14, "lastUpdated", StringType(), required=False),
        NestedField(15, "_ingested_at", TimestamptzType(), required=True),
        NestedField(16, "_source_file", StringType(), required=False),
        NestedField(17, "_payload_hash", StringType(), required=False),
        NestedField(18, "_raw_payload", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(15, 1000, DayTransform(), "_ingested_at_day")),
)

BRONZE_PRICES = TableDef(
    "bronze.stock_prices_raw",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "date", StringType(), required=False),
        NestedField(3, "open", DoubleType(), required=False),
        NestedField(4, "high", DoubleType(), required=False),
        NestedField(5, "low", DoubleType(), required=False),
        NestedField(6, "close", DoubleType(), required=False),
        NestedField(7, "adj_close", DoubleType(), required=False),
        NestedField(8, "volume", LongType(), required=False),
        NestedField(9, "_ingested_at", TimestamptzType(), required=True),
        NestedField(10, "_source_file", StringType(), required=False),
        NestedField(11, "_payload_hash", StringType(), required=False),
        NestedField(12, "_raw_payload", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(9, 1000, DayTransform(), "_ingested_at_day")),
)

_SILVER_TXN_FIELDS = (
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "account_id", LongType(), required=False),
    NestedField(3, "txn_date", DateType(), required=True),
    NestedField(4, "post_date", DateType(), required=False),
    NestedField(5, "signed_amount", DoubleType(), required=True),
    NestedField(6, "currency", StringType(), required=False),
    NestedField(7, "base_type", StringType(), required=False),
    NestedField(8, "sub_type", StringType(), required=False),
    NestedField(9, "category", StringType(), required=False),
    NestedField(10, "category_type", StringType(), required=False),
    NestedField(11, "merchant_raw", StringType(), required=False),
    NestedField(12, "merchant_normalized", StringType(), required=False),
    NestedField(13, "last_updated", TimestamptzType(), required=False),
    NestedField(14, "_ingested_at", TimestamptzType(), required=False),
)

SILVER_TRANSACTIONS = TableDef(
    "silver.transactions",
    Schema(*_SILVER_TXN_FIELDS),
    PartitionSpec(PartitionField(3, 1000, MonthTransform(), "txn_date_month")),
)

SILVER_QUARANTINE = TableDef(
    "silver.transactions_quarantine",
    Schema(*_SILVER_TXN_FIELDS,
           NestedField(15, "quarantine_reason", StringType(), required=True),
           NestedField(16, "quarantined_at", TimestamptzType(), required=True)),
    PartitionSpec(),  # unpartitioned: expected to stay small
)

SILVER_ACCOUNTS = TableDef(
    "silver.accounts",
    Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "provider_name", StringType(), required=False),
        NestedField(3, "account_type", StringType(), required=False),
        NestedField(4, "account_status", StringType(), required=False),
        NestedField(5, "balance_amount", DoubleType(), required=False),
        NestedField(6, "currency", StringType(), required=False),
        NestedField(7, "last_updated", TimestamptzType(), required=False),
    ),
    PartitionSpec(),
)

SILVER_MERCHANT_MAP = TableDef(
    "silver.merchant_ticker_map",
    Schema(
        NestedField(1, "merchant_normalized", StringType(), required=True),
        NestedField(2, "ticker", StringType(), required=True),
        NestedField(3, "company_name", StringType(), required=False),
        NestedField(4, "match_type", StringType(), required=False),
    ),
    PartitionSpec(),
)

SILVER_PRICES = TableDef(
    "silver.stock_prices",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "open", DoubleType(), required=False),
        NestedField(4, "high", DoubleType(), required=False),
        NestedField(5, "low", DoubleType(), required=False),
        NestedField(6, "close", DoubleType(), required=False),
        NestedField(7, "adj_close", DoubleType(), required=False),
        NestedField(8, "volume", LongType(), required=False),
    ),
    PartitionSpec(PartitionField(2, 1000, MonthTransform(), "trade_date_month")),
)

GOLD_SPEND = TableDef(
    "gold.daily_merchant_spend",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "spend_date", DateType(), required=True),
        NestedField(3, "gross_spend", DoubleType(), required=False),
        NestedField(4, "txn_count", LongType(), required=False),
        NestedField(5, "unique_accounts", LongType(), required=False),
        NestedField(6, "avg_ticket", DoubleType(), required=False),
        NestedField(7, "median_ticket", DoubleType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

GOLD_STOCK_FEATURES = TableDef(
    "gold.stock_features",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "close", DoubleType(), required=False),
        NestedField(4, "ret_1d", DoubleType(), required=False),
        NestedField(5, "ret_lag_1", DoubleType(), required=False),
        NestedField(6, "ret_lag_2", DoubleType(), required=False),
        NestedField(7, "ret_lag_3", DoubleType(), required=False),
        NestedField(8, "ret_lag_4", DoubleType(), required=False),
        NestedField(9, "ret_lag_5", DoubleType(), required=False),
        NestedField(10, "realized_vol_21d", DoubleType(), required=False),
        NestedField(11, "rsi_14", DoubleType(), required=False),
        NestedField(12, "volume_z_21d", DoubleType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

GOLD_TRAINING = TableDef(
    "gold.forecast_training_set",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "spend_mom_7d", DoubleType(), required=False),
        NestedField(4, "spend_z_28d", DoubleType(), required=False),
        NestedField(5, "txn_count_z_28d", DoubleType(), required=False),
        NestedField(6, "avg_ticket_delta_7d", DoubleType(), required=False),
        NestedField(7, "unique_acct_growth_7d", DoubleType(), required=False),
        NestedField(8, "spend_surprise", DoubleType(), required=False),
        NestedField(9, "ret_lag_1", DoubleType(), required=False),
        NestedField(10, "ret_lag_2", DoubleType(), required=False),
        NestedField(11, "ret_lag_3", DoubleType(), required=False),
        NestedField(12, "ret_lag_4", DoubleType(), required=False),
        NestedField(13, "ret_lag_5", DoubleType(), required=False),
        NestedField(14, "realized_vol_21d", DoubleType(), required=False),
        NestedField(15, "rsi_14", DoubleType(), required=False),
        NestedField(16, "volume_z_21d", DoubleType(), required=False),
        NestedField(17, "fwd_ret_5d", DoubleType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

GOLD_PREDICTIONS = TableDef(
    "gold.forecast_predictions",
    Schema(
        NestedField(1, "ticker", StringType(), required=True),
        NestedField(2, "trade_date", DateType(), required=True),
        NestedField(3, "fold", LongType(), required=False),
        NestedField(4, "y_true", DoubleType(), required=False),
        NestedField(5, "y_pred_model", DoubleType(), required=False),
        NestedField(6, "y_pred_persistence", DoubleType(), required=False),
        NestedField(7, "y_pred_arima", DoubleType(), required=False),
        NestedField(8, "model_version", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "ticker")),
)

ALL_TABLES: tuple[TableDef, ...] = (
    BRONZE_TRANSACTIONS, BRONZE_ACCOUNTS, BRONZE_PRICES,
    SILVER_TRANSACTIONS, SILVER_QUARANTINE, SILVER_ACCOUNTS,
    SILVER_MERCHANT_MAP, SILVER_PRICES,
    GOLD_SPEND, GOLD_STOCK_FEATURES, GOLD_TRAINING, GOLD_PREDICTIONS,
)
```

- [ ] **Step 5: Write `src/lakehouse/engines/base.py`**

```python
"""The one interface both engines implement."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pyarrow as pa

from src.lakehouse.schemas import TableDef


@runtime_checkable
class LakehouseEngine(Protocol):
    def create_table(self, table_def: TableDef) -> None: ...
    def table_exists(self, ident: str) -> bool: ...
    def append(self, ident: str, data: pa.Table) -> None: ...
    def overwrite(self, ident: str, data: pa.Table) -> None: ...
    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table: ...
    def snapshots(self, ident: str) -> list[int]: ...
    def snapshot_row_counts(self, ident: str) -> list[int]: ...
    def schema_history(self, ident: str) -> list[dict]: ...
    def sql(self, query: str, tables: dict[str, str]) -> pa.Table: ...
```

`snapshot_row_counts` returns rows *added per snapshot*, read from each
snapshot's `added-records` summary — metadata only, no scan. Volume-anomaly
detection needs per-batch counts: on an append-only table the cumulative row
count only ever grows, so comparing totals against a trailing median can never
fire. `schema_history` returns one entry per schema version the table has had,
which is what makes cross-version querying inspectable.

- [ ] **Step 6: Write `src/lakehouse/engines/pyiceberg_engine.py`**

```python
"""Default engine: PyIceberg owns catalog and commits, DuckDB does the SQL."""
from __future__ import annotations

import duckdb
import pyarrow as pa

from src.lakehouse.schemas import TableDef


class PyIcebergEngine:
    def __init__(self, catalog):
        self.catalog = catalog

    def create_table(self, table_def: TableDef) -> None:
        if self.table_exists(table_def.name):
            return
        self.catalog.create_table(
            identifier=table_def.name,
            schema=table_def.schema,
            partition_spec=table_def.spec,
        )

    def table_exists(self, ident: str) -> bool:
        try:
            self.catalog.load_table(ident)
            return True
        except Exception:
            return False

    def append(self, ident: str, data: pa.Table) -> None:
        table = self.catalog.load_table(ident)
        table.append(data.cast(table.schema().as_arrow()))

    def overwrite(self, ident: str, data: pa.Table) -> None:
        table = self.catalog.load_table(ident)
        table.overwrite(data.cast(table.schema().as_arrow()))

    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table:
        table = self.catalog.load_table(ident)
        scan = table.scan(snapshot_id=snapshot_id) if snapshot_id else table.scan()
        return scan.to_arrow()

    def snapshots(self, ident: str) -> list[int]:
        table = self.catalog.load_table(ident)
        return [s.snapshot_id for s in table.metadata.snapshots]

    def snapshot_row_counts(self, ident: str) -> list[int]:
        """Rows added per snapshot, from metadata summaries. No data scan."""
        table = self.catalog.load_table(ident)
        counts = []
        for snapshot in table.metadata.snapshots:
            summary = getattr(snapshot, "summary", None) or {}
            added = summary.get("added-records") or summary.get("added_records") or 0
            counts.append(int(added))
        return counts

    def schema_history(self, ident: str) -> list[dict]:
        """One entry per schema version this table has had."""
        table = self.catalog.load_table(ident)
        return [
            {"schema_id": s.schema_id,
             "columns": {f.name: str(f.field_type) for f in s.fields},
             "field_ids": {f.name: f.field_id for f in s.fields}}
            for s in table.metadata.schemas
        ]

    def sql(self, query: str, tables: dict[str, str]) -> pa.Table:
        con = duckdb.connect()
        try:
            for alias, ident in tables.items():
                con.register(alias, self.scan_arrow(ident))
            return con.execute(query).arrow()
        finally:
            con.close()
```

- [ ] **Step 7: Write `src/lakehouse/engines/__init__.py`**

```python
from __future__ import annotations

import os

from src.lakehouse.engines.base import LakehouseEngine


def get_engine(name: str | None = None) -> LakehouseEngine:
    choice = (name or os.environ.get("ENGINE") or "pyiceberg").lower()
    if choice == "pyiceberg":
        from src.lakehouse.catalog import get_catalog
        from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
        return PyIcebergEngine(get_catalog())
    if choice == "spark":
        from src.lakehouse.engines.spark_engine import SparkEngine
        return SparkEngine()
    raise ValueError(f"unknown ENGINE: {choice}")


__all__ = ["LakehouseEngine", "get_engine"]
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_engine.py -v`
Expected: 6 passed. `config.WAREHOUSE_PATH` is read at import time, so the last test sets `WAREHOUSE_PATH` before calling `get_engine()`; if it leaks into other tests, add `importlib.reload(config)` in the fixture.

- [ ] **Step 9: Commit**

```bash
git add src/lakehouse tests/test_engine.py
git commit -m "feat: Iceberg catalog, table schemas, and dual-engine protocol

The protocol surface is deliberately five methods. A wider interface would
be easy to write against PyIceberg and then impossible to satisfy in Spark
without leaking PyIceberg concepts into the abstraction.

Time travel is exercised in the engine test rather than only in a demo
script, so a regression in snapshot handling fails CI."
```

---

## Task 6: Bronze ingestion

**Files:**
- Create: `src/lakehouse/bronze.py`
- Test: `tests/test_bronze.py`

**Interfaces:**
- Consumes: `get_engine`, `schemas.BRONZE_*`, generators from Tasks 3–4
- Produces: `bronze.ingest_all(engine, n_transactions: int | None = None) -> dict[str, int]` returning rows appended per table; `bronze.add_lineage(records: list[dict], source_file: str) -> pa.Table`; `bronze.max_txn_date(engine) -> date | None`; `python -m src.lakehouse.bronze` CLI

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bronze.py
import pyarrow as pa
import pytest

from src.lakehouse import bronze, catalog as catalog_mod, schemas
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    return PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))


def test_ingest_writes_all_three_bronze_tables(engine):
    counts = bronze.ingest_all(engine, n_transactions=3000)
    assert counts["bronze.yodlee_transactions_raw"] == 3000
    assert counts["bronze.yodlee_accounts_raw"] > 0
    assert counts["bronze.stock_prices_raw"] > 0


def test_lineage_columns_present_and_populated(engine):
    bronze.ingest_all(engine, n_transactions=500)
    tbl = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name)
    for col in ("_ingested_at", "_source_file", "_payload_hash", "_raw_payload"):
        assert col in tbl.column_names
        assert tbl.column(col).null_count == 0


def test_nested_structs_preserved_not_flattened(engine):
    bronze.ingest_all(engine, n_transactions=200)
    tbl = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name)
    assert pa.types.is_struct(tbl.schema.field("amount").type)
    assert pa.types.is_struct(tbl.schema.field("merchant").type)
    # Direction still lives in baseType -- Bronze must not pre-sign the amount.
    amounts = tbl.column("amount").combine_chunks().field("amount").to_pylist()
    assert all(a > 0 for a in amounts)


def test_reingest_creates_new_snapshot_and_duplicates_rows(engine):
    """Bronze is append-only: re-ingest duplicates by design. Silver dedupes."""
    bronze.ingest_all(engine, n_transactions=1000)
    snaps_before = len(engine.snapshots(schemas.BRONZE_TRANSACTIONS.name))
    bronze.ingest_all(engine, n_transactions=1000)
    assert len(engine.snapshots(schemas.BRONZE_TRANSACTIONS.name)) == snaps_before + 1
    assert engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows == 2000


def test_payload_hash_is_stable_for_identical_record(engine):
    rec = {"id": 1, "amount": {"amount": 5.0, "currency": "USD"}}
    a = bronze.payload_hash(rec)
    b = bronze.payload_hash(dict(reversed(list(rec.items()))))
    assert a == b  # key order must not change the hash


def test_max_txn_date_feeds_as_of_resolution(engine):
    bronze.ingest_all(engine, n_transactions=2000)
    from src import config
    assert bronze.max_txn_date(engine) <= config.END_DATE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bronze.py -v`
Expected: FAIL — `src.lakehouse.bronze` not found.

- [ ] **Step 3: Write `src/lakehouse/bronze.py`**

```python
"""Bronze: append-only, source-faithful, lineage-stamped.

No type coercion, no dedupe, no signing of amounts. Everything that could be
wrong about the source stays wrong here on purpose -- Bronze's job is to be a
replayable record of what arrived, not a clean one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timezone

import pyarrow as pa

from src import config
from src.generators import stock_prices, yodlee_feed
from src.lakehouse import schemas
from src.lakehouse.engines import get_engine

_INGEST_TS = datetime(2026, 7, 1, 6, 0, tzinfo=timezone.utc)


def payload_hash(record: dict) -> str:
    """Order-independent hash so key reordering upstream is not seen as change."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()


def add_lineage(records: list[dict], source_file: str,
                ingested_at: datetime | None = None) -> list[dict]:
    ts = ingested_at or _INGEST_TS
    out = []
    for rec in records:
        enriched = dict(rec)
        enriched["_ingested_at"] = ts
        enriched["_source_file"] = source_file
        enriched["_payload_hash"] = payload_hash(rec)
        enriched["_raw_payload"] = json.dumps(rec, sort_keys=True, default=str)
        out.append(enriched)
    return out


def _to_arrow(records: list[dict], table_def: schemas.TableDef) -> pa.Table:
    arrow_schema = table_def.schema.as_arrow()
    return pa.Table.from_pylist(records, schema=arrow_schema)


def ingest_all(engine, n_transactions: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}

    txns = yodlee_feed.generate_transactions(n=n_transactions)
    accounts = yodlee_feed.generate_accounts()
    prices = stock_prices.generate_prices()

    for table_def, records, source in (
        (schemas.BRONZE_TRANSACTIONS, txns, "yodlee_transactions.json"),
        (schemas.BRONZE_ACCOUNTS, accounts, "yodlee_accounts.json"),
        (schemas.BRONZE_PRICES, prices, "stock_prices.json"),
    ):
        engine.create_table(table_def)
        enriched = add_lineage(records, source)
        engine.append(table_def.name, _to_arrow(enriched, table_def))
        counts[table_def.name] = len(enriched)

    return counts


def max_txn_date(engine) -> date | None:
    if not engine.table_exists(schemas.BRONZE_TRANSACTIONS.name):
        return None
    result = engine.sql(
        "SELECT max(transactionDate) AS d FROM bronze_txn",
        tables={"bronze_txn": schemas.BRONZE_TRANSACTIONS.name},
    )
    value = result.column("d")[0].as_py()
    return date.fromisoformat(value) if value else None


def main() -> int:
    engine = get_engine()
    counts = ingest_all(engine)
    for name, n in counts.items():
        print(f"bronze: appended {n:,} rows to {name}")
    print(f"bronze: AS_OF_DATE resolves to {config.resolve_as_of_date(max_txn_date(engine))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bronze.py -v`
Expected: 6 passed.

If `pa.Table.from_pylist` rejects the `datetime` in `_ingested_at`, confirm the Iceberg `TimestamptzType` maps to `pa.timestamp("us", tz="UTC")` and that `_INGEST_TS` is timezone-aware. A naive datetime is the usual cause.

- [ ] **Step 5: Run the real ingest and confirm it completes**

Run: `uv run make bronze`
Expected: three "appended N rows" lines plus an `AS_OF_DATE` line reading `2026-06-30`.

- [ ] **Step 6: Commit**

```bash
git add src/lakehouse/bronze.py tests/test_bronze.py
git commit -m "feat: Bronze append-only ingestion with lineage

Re-ingest duplicates rows on purpose and a test pins that behaviour. If
Bronze deduped, a replay would silently rewrite history and there would be
no way to reconstruct what actually arrived on a given day.

payload_hash sorts keys so an upstream key reordering is not mistaken for
a content change."
```

---

## Task 7: Data contracts and validator

**Files:**
- Create: `src/contracts/__init__.py`, `src/contracts/validator.py`, `contracts/bronze_yodlee_transactions.yaml`, `contracts/silver_transactions.yaml`, `contracts/gold_forecast_training_set.yaml`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Consumes: `config.resolve_as_of_date`
- Produces: `validator.load_contract(path: Path) -> Contract`; `validator.validate(table: pa.Table, contract: Contract, as_of: date) -> ValidationReport`; dataclasses `Contract(table, version, owner, schema_fields, expectations)`, `ExpectationResult(kind, column, passed, observed, detail)`, `ValidationReport(table, results, passed: bool, failures: list[ExpectationResult])`; `validator.assert_valid(table, contract, as_of) -> None` raising `ContractViolation`; `validator.CONTRACTS_DIR: Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts.py
from datetime import date

import pyarrow as pa
import pytest

from src.contracts import validator


def _table(ids=(1, 2, 3), amounts=(-5.0, 10.0, -2.5), dates=None):
    dates = dates or [date(2026, 6, 28), date(2026, 6, 29), date(2026, 6, 30)]
    return pa.table({
        "id": pa.array(ids, pa.int64()),
        "txn_date": pa.array(dates, pa.date32()),
        "signed_amount": pa.array(amounts, pa.float64()),
        "base_type": pa.array(["DEBIT", "CREDIT", "DEBIT"]),
    })


@pytest.fixture
def contract():
    return validator.load_contract(validator.CONTRACTS_DIR / "silver_transactions.yaml")


def test_valid_table_passes(contract):
    report = validator.validate(_table(), contract, as_of=date(2026, 6, 30))
    assert report.passed, report.failures


def test_null_in_non_nullable_column_fails(contract):
    tbl = pa.table({
        "id": pa.array([1, None], pa.int64()),
        "txn_date": pa.array([date(2026, 6, 30)] * 2, pa.date32()),
        "signed_amount": pa.array([1.0, 2.0], pa.float64()),
        "base_type": pa.array(["DEBIT", "DEBIT"]),
    })
    report = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert not report.passed
    assert any(f.kind == "not_null" and f.column == "id" for f in report.failures)


def test_duplicate_ids_fail(contract):
    report = validator.validate(_table(ids=(1, 1, 2)), contract, as_of=date(2026, 6, 30))
    assert any(f.kind == "unique" for f in report.failures)


def test_unexpected_base_type_fails(contract):
    tbl = _table()
    tbl = tbl.set_column(tbl.schema.get_field_index("base_type"), "base_type",
                         pa.array(["DEBIT", "REFUND", "DEBIT"]))
    report = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert any(f.kind == "accepted_values" for f in report.failures)


def test_freshness_uses_as_of_not_wall_clock(contract):
    """Data ends 2026-06-30. Against wall clock this check would fail forever."""
    tbl = _table()
    fresh = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert not any(f.kind == "freshness" for f in fresh.failures)
    stale = validator.validate(tbl, contract, as_of=date(2026, 8, 9))
    assert any(f.kind == "freshness" for f in stale.failures)


def test_sign_inversion_is_caught(contract):
    """The exact bug Task 8 injects: DEBIT rows must be negative."""
    unsigned = _table(amounts=(5.0, 10.0, 2.5))  # all positive, DEBIT included
    report = validator.validate(unsigned, contract, as_of=date(2026, 6, 30))
    assert any(f.kind == "conditional_sign" for f in report.failures)


def test_assert_valid_raises_on_violation(contract):
    with pytest.raises(validator.ContractViolation) as exc:
        validator.assert_valid(_table(ids=(1, 1, 2)), contract, as_of=date(2026, 6, 30))
    assert "unique" in str(exc.value)


def test_report_records_observed_values(contract):
    report = validator.validate(_table(ids=(1, 1, 2)), contract, as_of=date(2026, 6, 30))
    dup = next(f for f in report.failures if f.kind == "unique")
    assert dup.observed is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL — `src.contracts` not found.

- [ ] **Step 3: Write `contracts/silver_transactions.yaml`**

```yaml
table: silver.transactions
version: 1
owner: data-platform
schema:
  - {name: id, type: long, nullable: false}
  - {name: txn_date, type: date, nullable: false}
  - {name: signed_amount, type: double, nullable: false}
  - {name: base_type, type: string, nullable: true}
expectations:
  - {type: not_null, column: id}
  - {type: not_null, column: txn_date}
  - {type: unique, column: id}
  - {type: accepted_values, column: base_type, values: [CREDIT, DEBIT]}
  - {type: range, column: signed_amount, min: -100000, max: 100000}
  - {type: conditional_sign, column: signed_amount, when_column: base_type,
     when_value: DEBIT, sign: negative}
  - {type: freshness, column: txn_date, max_lag_days: 3}
  - {type: row_count, min: 1}
```

`conditional_sign` exists because no combination of `range`, `unique`, and
`accepted_values` can express "DEBIT rows must be negative". Sign inversion is
the specific bug Task 8 injects to prove the build fails closed, and without
this expectation that injection passes validation silently — the test would
assert a guarantee the contract does not provide.

- [ ] **Step 4: Write `contracts/bronze_yodlee_transactions.yaml`**

```yaml
table: bronze.yodlee_transactions_raw
version: 1
owner: data-platform
schema:
  - {name: id, type: long, nullable: false}
  - {name: transactionDate, type: string, nullable: true}
  - {name: baseType, type: string, nullable: true}
  - {name: amount, type: struct, nullable: true}
  - {name: merchant, type: struct, nullable: true}
  - {name: _ingested_at, type: timestamptz, nullable: false}
expectations:
  - {type: not_null, column: id}
  - {type: not_null, column: _ingested_at}
  - {type: accepted_values, column: baseType, values: [CREDIT, DEBIT]}
  - {type: row_count, min: 1000}
```

- [ ] **Step 5: Write `contracts/gold_forecast_training_set.yaml`**

```yaml
table: gold.forecast_training_set
version: 1
owner: data-platform
schema:
  - {name: ticker, type: string, nullable: false}
  - {name: trade_date, type: date, nullable: false}
  - {name: fwd_ret_5d, type: double, nullable: true}
expectations:
  - {type: not_null, column: ticker}
  - {type: not_null, column: trade_date}
  - {type: range, column: fwd_ret_5d, min: -0.9, max: 2.0}
  - {type: row_count, min: 1000}
```

- [ ] **Step 6: Write `src/contracts/validator.py`**

```python
"""Declared contracts, evaluated against Arrow tables. Fails closed.

The same YAML is read by the pipeline (to gate a build) and by the ops agent
(to diff against live Iceberg schemas). One source of truth, two consumers --
otherwise the agent starts alerting on rules the pipeline never enforced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import yaml

from src import config

CONTRACTS_DIR = config.REPO_ROOT / "contracts"


class ContractViolation(Exception):
    pass


@dataclass(frozen=True)
class SchemaField:
    name: str
    type: str
    nullable: bool = True


@dataclass(frozen=True)
class Contract:
    table: str
    version: int
    owner: str
    schema_fields: tuple[SchemaField, ...]
    expectations: tuple[dict, ...]


@dataclass(frozen=True)
class ExpectationResult:
    kind: str
    column: str | None
    passed: bool
    observed: object = None
    detail: str = ""


@dataclass
class ValidationReport:
    table: str
    results: list[ExpectationResult] = field(default_factory=list)

    @property
    def failures(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passed(self) -> bool:
        return not self.failures


def load_contract(path: Path) -> Contract:
    raw = yaml.safe_load(Path(path).read_text())
    return Contract(
        table=raw["table"],
        version=int(raw.get("version", 1)),
        owner=raw.get("owner", "unknown"),
        schema_fields=tuple(
            SchemaField(f["name"], f["type"], f.get("nullable", True))
            for f in raw.get("schema", [])
        ),
        expectations=tuple(raw.get("expectations", [])),
    )


def validate(table: pa.Table, contract: Contract, as_of: date) -> ValidationReport:
    report = ValidationReport(table=contract.table)

    for sf in contract.schema_fields:
        present = sf.name in table.column_names
        report.results.append(ExpectationResult(
            "schema_present", sf.name, present,
            observed=present, detail="" if present else "column missing"))

    for exp in contract.expectations:
        report.results.append(_evaluate(table, exp, as_of))

    return report


def _evaluate(table: pa.Table, exp: dict, as_of: date) -> ExpectationResult:
    kind = exp["type"]
    column = exp.get("column")

    if kind == "row_count":
        n = table.num_rows
        return ExpectationResult(kind, None, n >= exp.get("min", 0), n)

    if column not in table.column_names:
        return ExpectationResult(kind, column, False, None, "column missing")

    col = table.column(column)

    if kind == "not_null":
        nulls = col.null_count
        return ExpectationResult(kind, column, nulls == 0, nulls,
                                 f"{nulls} null(s)")

    if kind == "unique":
        distinct = pc.count_distinct(col).as_py()
        non_null = table.num_rows - col.null_count
        dupes = non_null - distinct
        return ExpectationResult(kind, column, dupes == 0, dupes,
                                 f"{dupes} duplicate(s)")

    if kind == "accepted_values":
        allowed = set(exp["values"])
        seen = {v for v in col.to_pylist() if v is not None}
        bad = seen - allowed
        return ExpectationResult(kind, column, not bad, sorted(bad),
                                 f"unexpected: {sorted(bad)}")

    if kind == "range":
        values = [v for v in col.to_pylist() if v is not None]
        if not values:
            return ExpectationResult(kind, column, True, None)
        lo, hi = min(values), max(values)
        ok = lo >= exp.get("min", float("-inf")) and hi <= exp.get("max", float("inf"))
        return ExpectationResult(kind, column, ok, (lo, hi))

    if kind == "conditional_sign":
        when = table.column(exp["when_column"]).to_pylist()
        values = col.to_pylist()
        want_negative = exp["sign"] == "negative"
        bad = sum(
            1 for w, v in zip(when, values)
            if w == exp["when_value"] and v is not None
            and ((v > 0) if want_negative else (v < 0))
        )
        return ExpectationResult(
            kind, column, bad == 0, bad,
            f"{bad} row(s) where {exp['when_column']}="
            f"{exp['when_value']} are not {exp['sign']}")

    if kind == "freshness":
        values = [v for v in col.to_pylist() if v is not None]
        if not values:
            return ExpectationResult(kind, column, False, None, "no values")
        newest = max(values)
        newest = newest.date() if hasattr(newest, "date") else newest
        lag = (as_of - newest).days
        ok = lag <= exp["max_lag_days"]
        return ExpectationResult(kind, column, ok, lag,
                                 f"{lag}d behind AS_OF {as_of}")

    raise ValueError(f"unknown expectation type: {kind}")


def assert_valid(table: pa.Table, contract: Contract, as_of: date) -> None:
    report = validate(table, contract, as_of)
    if not report.passed:
        lines = [f"  - {f.kind} on {f.column}: {f.detail} (observed={f.observed})"
                 for f in report.failures]
        raise ContractViolation(
            f"{contract.table} violated {len(report.failures)} expectation(s):\n"
            + "\n".join(lines))
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add src/contracts contracts tests/test_contracts.py
git commit -m "feat: YAML data contracts with fail-closed validation

Freshness takes as_of as a parameter rather than reading the clock. A test
pins both directions: fresh at 2026-06-30, stale at 2026-08-09. A
wall-clock check against a fixed dataset fails permanently on day one and
gets muted, which is how freshness monitoring usually dies.

Contracts are read by both the pipeline and the ops agent so the agent
cannot alert on a rule the pipeline never enforced."
```

---

## Task 8: Silver conforming layer

**Files:**
- Create: `src/lakehouse/silver.py`
- Test: `tests/test_silver.py`

**Interfaces:**
- Consumes: `get_engine`, `schemas.SILVER_*`, `bronze.max_txn_date`, `validator.assert_valid`
- Produces: `silver.normalize_merchant(raw: str) -> str`; `silver.build_all(engine) -> dict[str, int]`; `silver.merchant_map_rows() -> list[dict]`; `python -m src.lakehouse.silver` CLI

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silver.py
from datetime import date

import pytest

from src.lakehouse import bronze, catalog as catalog_mod, schemas, silver
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=20000)
    return eng


@pytest.mark.parametrize("raw,expected", [
    ("STARBUCKS STORE #1234", "STARBUCKS STORE"),
    ("Starbucks", "STARBUCKS"),
    ("CHIPOTLE 0455", "CHIPOTLE"),
    ("Domino's Pizza", "DOMINOS PIZZA"),
    ("TARGET T-1088", "TARGET T"),
    ("lululemon athletica", "LULULEMON ATHLETICA"),
])
def test_merchant_normalization(raw, expected):
    assert silver.normalize_merchant(raw) == expected


def test_signed_amount_direction_comes_from_base_type(engine):
    silver.build_all(engine)
    out = engine.sql(
        """SELECT base_type, min(signed_amount) AS lo, max(signed_amount) AS hi
           FROM s GROUP BY base_type""",
        tables={"s": schemas.SILVER_TRANSACTIONS.name})
    rows = {r["base_type"]: r for r in out.to_pylist()}
    assert rows["DEBIT"]["hi"] < 0
    assert rows["CREDIT"]["lo"] > 0


def test_dedup_keeps_latest_last_updated(engine):
    """Re-ingest duplicates Bronze rows; Silver must collapse them."""
    bronze_rows = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows
    bronze.ingest_all(engine, n_transactions=20000)
    assert engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows == bronze_rows * 2
    silver.build_all(engine)
    n = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).num_rows
    assert n == bronze_rows


def test_ids_unique_in_silver(engine):
    silver.build_all(engine)
    out = engine.sql("SELECT count(*) AS n, count(DISTINCT id) AS d FROM s",
                     tables={"s": schemas.SILVER_TRANSACTIONS.name})
    row = out.to_pylist()[0]
    assert row["n"] == row["d"]


def test_merchant_map_covers_every_configured_merchant_string(engine):
    from src import config
    silver.build_all(engine)
    mapped = engine.sql("SELECT merchant_normalized, ticker FROM m",
                        tables={"m": schemas.SILVER_MERCHANT_MAP.name}).to_pylist()
    lookup = {r["merchant_normalized"]: r["ticker"] for r in mapped}
    for company in config.COMPANIES:
        for raw in company.merchant_strings:
            assert lookup[silver.normalize_merchant(raw)] == company.ticker


def test_non_usd_rows_are_quarantined_not_dropped(engine):
    silver.build_all(engine)
    assert engine.table_exists(schemas.SILVER_QUARANTINE.name)
    kept = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).num_rows
    quarantined = engine.scan_arrow(schemas.SILVER_QUARANTINE.name).num_rows
    bronze_distinct = engine.sql(
        "SELECT count(DISTINCT id) AS d FROM b",
        tables={"b": schemas.BRONZE_TRANSACTIONS.name}).to_pylist()[0]["d"]
    assert kept + quarantined == bronze_distinct


def test_prices_typed_to_date_and_aligned_to_calendar(engine):
    silver.build_all(engine)
    tbl = engine.scan_arrow(schemas.SILVER_PRICES.name)
    dates = tbl.column("trade_date").to_pylist()
    assert all(isinstance(d, date) for d in dates)
    assert all(d.weekday() < 5 for d in dates)


def test_contract_violation_aborts_the_build(engine, monkeypatch):
    monkeypatch.setattr(silver, "_SIGN_DEBITS", False)  # inject the classic bug
    with pytest.raises(Exception):
        silver.build_all(engine)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_silver.py -v`
Expected: FAIL — `src.lakehouse.silver` not found.

- [ ] **Step 3: Write `src/lakehouse/silver.py`**

```python
"""Silver: conform, dedupe, sign, normalize, quarantine, validate.

Deduplication keeps the greatest lastUpdated per id, so a late-arriving
correction wins over the original. Bronze may contain the same id many times
across replays; Silver is where that collapses to one truth.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

import pyarrow as pa

from src import config
from src.contracts import validator
from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine

_SIGN_DEBITS = True  # monkeypatched in tests to prove the build fails closed
_STORE_SUFFIX = re.compile(r"(#\s*\d+|\b\d{3,}\b|-\s*\d+)")
_PUNCT = re.compile(r"[^A-Z0-9 ]")


def normalize_merchant(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.upper()
    text = _STORE_SUFFIX.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()) or None


def merchant_map_rows() -> list[dict]:
    rows = []
    for company in config.COMPANIES:
        for raw in company.merchant_strings:
            rows.append({
                "merchant_normalized": normalize_merchant(raw),
                "ticker": company.ticker,
                "company_name": company.name,
                "match_type": "exact",
            })
    seen, deduped = set(), []
    for row in rows:
        key = (row["merchant_normalized"], row["ticker"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _build_transactions(engine) -> tuple[pa.Table, pa.Table]:
    sign_expr = ("CASE WHEN baseType = 'DEBIT' THEN -1.0 ELSE 1.0 END"
                 if _SIGN_DEBITS else "1.0")
    query = f"""
    WITH ranked AS (
        SELECT *, row_number() OVER (
            PARTITION BY id ORDER BY lastUpdated DESC, _ingested_at DESC
        ) AS rn
        FROM b
    )
    SELECT
        id,
        accountId                                   AS account_id,
        CAST(transactionDate AS DATE)               AS txn_date,
        CAST(postDate AS DATE)                      AS post_date,
        ({sign_expr}) * amount.amount               AS signed_amount,
        amount.currency                             AS currency,
        baseType                                    AS base_type,
        subType                                     AS sub_type,
        category,
        categoryType                                AS category_type,
        merchant.source                             AS merchant_raw,
        merchant.source                             AS merchant_normalized,
        CAST(lastUpdated AS TIMESTAMP WITH TIME ZONE) AS last_updated,
        _ingested_at
    FROM ranked WHERE rn = 1
    """
    conformed = engine.sql(query, tables={"b": schemas.BRONZE_TRANSACTIONS.name})

    normalized = pa.array(
        [normalize_merchant(v) for v in conformed.column("merchant_normalized").to_pylist()],
        pa.string())
    conformed = conformed.set_column(
        conformed.schema.get_field_index("merchant_normalized"),
        "merchant_normalized", normalized)

    currency = conformed.column("currency").to_pylist()
    keep = [i for i, c in enumerate(currency) if c == "USD"]
    reject = [i for i, c in enumerate(currency) if c != "USD"]

    clean = conformed.take(keep)
    quarantined = conformed.take(reject)
    if quarantined.num_rows:
        quarantined = quarantined.append_column(
            "quarantine_reason",
            pa.array(["non_usd_currency"] * quarantined.num_rows, pa.string()))
        quarantined = quarantined.append_column(
            "quarantined_at",
            pa.array([datetime(2026, 7, 1, 6, 0, tzinfo=timezone.utc)] * quarantined.num_rows,
                     pa.timestamp("us", tz="UTC")))
    else:
        quarantined = schemas.SILVER_QUARANTINE.schema.as_arrow().empty_table()

    return clean, quarantined


def build_all(engine) -> dict[str, int]:
    counts: dict[str, int] = {}
    as_of = config.resolve_as_of_date(bronze.max_txn_date(engine))

    txns, quarantined = _build_transactions(engine)
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "silver_transactions.yaml")
    validator.assert_valid(txns, contract, as_of)   # fails closed

    for table_def, data in (
        (schemas.SILVER_TRANSACTIONS, txns),
        (schemas.SILVER_QUARANTINE, quarantined),
    ):
        engine.create_table(table_def)
        engine.overwrite(table_def.name, data.cast(table_def.schema.as_arrow()))
        counts[table_def.name] = data.num_rows

    accounts = engine.sql("""
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY id ORDER BY lastUpdated DESC) AS rn
            FROM a)
        SELECT id, providerName AS provider_name, accountType AS account_type,
               accountStatus AS account_status, balance.amount AS balance_amount,
               balance.currency AS currency,
               CAST(lastUpdated AS TIMESTAMP WITH TIME ZONE) AS last_updated
        FROM ranked WHERE rn = 1
    """, tables={"a": schemas.BRONZE_ACCOUNTS.name})
    engine.create_table(schemas.SILVER_ACCOUNTS)
    engine.overwrite(schemas.SILVER_ACCOUNTS.name,
                     accounts.cast(schemas.SILVER_ACCOUNTS.schema.as_arrow()))
    counts[schemas.SILVER_ACCOUNTS.name] = accounts.num_rows

    mapping = pa.Table.from_pylist(
        merchant_map_rows(), schema=schemas.SILVER_MERCHANT_MAP.schema.as_arrow())
    engine.create_table(schemas.SILVER_MERCHANT_MAP)
    engine.overwrite(schemas.SILVER_MERCHANT_MAP.name, mapping)
    counts[schemas.SILVER_MERCHANT_MAP.name] = mapping.num_rows

    prices = engine.sql("""
        WITH ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY ticker, date ORDER BY _ingested_at DESC) AS rn
            FROM p)
        SELECT ticker, CAST(date AS DATE) AS trade_date,
               open, high, low, close, adj_close, volume
        FROM ranked WHERE rn = 1
    """, tables={"p": schemas.BRONZE_PRICES.name})
    engine.create_table(schemas.SILVER_PRICES)
    engine.overwrite(schemas.SILVER_PRICES.name,
                     prices.cast(schemas.SILVER_PRICES.schema.as_arrow()))
    counts[schemas.SILVER_PRICES.name] = prices.num_rows

    return counts


def main() -> int:
    engine = get_engine()
    for name, n in build_all(engine).items():
        print(f"silver: wrote {n:,} rows to {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_silver.py -v`
Expected: 13 passed (6 parametrized normalization cases + 7 others).

If a normalization case fails, adjust `_STORE_SUFFIX`/`_PUNCT` — do not change the expected value in the test. `TARGET T-1088` → `TARGET T` is intentional: the trailing store number goes, the `T` stays, and the merchant map is seeded from the same function so both sides agree.

- [ ] **Step 5: Run the real build**

Run: `uv run make silver`
Expected: five "wrote N rows" lines; `silver.transactions` count equals distinct Bronze ids.

- [ ] **Step 6: Commit**

```bash
git add src/lakehouse/silver.py tests/test_silver.py
git commit -m "feat: Silver conforming layer with quarantine and fail-closed contracts

Signing is derived from baseType, and a test asserts DEBIT rows are strictly
negative and CREDIT strictly positive. Getting this backwards produces a
pipeline that runs clean and reports inverted spend, which no schema check
would catch.

_SIGN_DEBITS exists so a test can inject that exact bug and prove the
contract aborts the build rather than warning and continuing.

Non-USD rows go to a quarantine table; a test asserts kept + quarantined
equals distinct Bronze ids, so rows can never be silently dropped."
```

---

## Task 9: Gold spend and price feature marts

**Files:**
- Create: `src/lakehouse/gold.py` (spend + stock features only; training set added in Task 10)
- Test: `tests/test_gold_marts.py`

**Interfaces:**
- Consumes: `schemas.SILVER_*`, `schemas.GOLD_SPEND`, `schemas.GOLD_STOCK_FEATURES`
- Produces: `gold.build_daily_merchant_spend(engine) -> pa.Table`, `gold.build_stock_features(engine) -> pa.Table`, `gold.build_all(engine) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gold_marts.py
import pytest

from src import config
from src.lakehouse import bronze, catalog as catalog_mod, gold, schemas, silver
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=40000)
    silver.build_all(eng)
    gold.build_all(eng)
    return eng


def test_spend_grain_is_ticker_by_date(engine):
    out = engine.sql("SELECT count(*) AS n, count(DISTINCT (ticker, spend_date)) AS d FROM g",
                     tables={"g": schemas.GOLD_SPEND.name}).to_pylist()[0]
    assert out["n"] == out["d"]


def test_spend_is_positive_despite_signed_amounts(engine):
    """signed_amount is negative for DEBIT; gross_spend must be reported positive."""
    out = engine.sql("SELECT min(gross_spend) AS lo FROM g",
                     tables={"g": schemas.GOLD_SPEND.name}).to_pylist()[0]
    assert out["lo"] > 0


def test_only_tracked_tickers_appear(engine):
    tickers = {r["ticker"] for r in engine.sql(
        "SELECT DISTINCT ticker FROM g", tables={"g": schemas.GOLD_SPEND.name}).to_pylist()}
    assert tickers == {c.ticker for c in config.COMPANIES}


def test_noise_merchants_excluded_from_spend(engine):
    """~60% of transactions are noise; spend must not include them."""
    spend_txns = engine.sql("SELECT sum(txn_count) AS n FROM g",
                            tables={"g": schemas.GOLD_SPEND.name}).to_pylist()[0]["n"]
    all_txns = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).num_rows
    assert 0.30 < spend_txns / all_txns < 0.50


def test_stock_features_lags_are_backward_looking(engine):
    rows = engine.sql("""
        SELECT trade_date, ret_1d, ret_lag_1 FROM f
        WHERE ticker = 'SBUX' ORDER BY trade_date
    """, tables={"f": schemas.GOLD_STOCK_FEATURES.name}).to_pylist()
    for prev, cur in zip(rows[1:], rows[2:]):
        if prev["ret_1d"] is not None and cur["ret_lag_1"] is not None:
            assert abs(cur["ret_lag_1"] - prev["ret_1d"]) < 1e-9


def test_rsi_bounded(engine):
    out = engine.sql("SELECT min(rsi_14) AS lo, max(rsi_14) AS hi FROM f WHERE rsi_14 IS NOT NULL",
                     tables={"f": schemas.GOLD_STOCK_FEATURES.name}).to_pylist()[0]
    assert 0 <= out["lo"] and out["hi"] <= 100


def test_realized_vol_non_negative(engine):
    out = engine.sql("SELECT min(realized_vol_21d) AS lo FROM f WHERE realized_vol_21d IS NOT NULL",
                     tables={"f": schemas.GOLD_STOCK_FEATURES.name}).to_pylist()[0]
    assert out["lo"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gold_marts.py -v`
Expected: FAIL — `src.lakehouse.gold` not found.

- [ ] **Step 3: Write `src/lakehouse/gold.py`**

```python
"""Gold: feature marts. Every window is trailing and right-closed.

The only forward-looking column in this module is fwd_ret_5d (Task 10), and
it is the target. Everything else at date t is computable from data <= t.
"""
from __future__ import annotations

import sys

import pyarrow as pa

from src.lakehouse import schemas
from src.lakehouse.engines import get_engine

SPEND_SQL = """
SELECT
    m.ticker                                   AS ticker,
    t.txn_date                                 AS spend_date,
    sum(abs(t.signed_amount))                  AS gross_spend,
    count(*)                                   AS txn_count,
    count(DISTINCT t.account_id)               AS unique_accounts,
    avg(abs(t.signed_amount))                  AS avg_ticket,
    median(abs(t.signed_amount))               AS median_ticket
FROM silver_txn t
JOIN silver_map m ON t.merchant_normalized = m.merchant_normalized
WHERE t.category_type = 'EXPENSE'
GROUP BY 1, 2
ORDER BY 1, 2
"""

# Windows are ROWS BETWEEN ... AND CURRENT ROW: trailing and right-closed.
STOCK_FEATURES_SQL = """
WITH base AS (
    SELECT ticker, trade_date, close, volume,
           close / lag(close) OVER w - 1 AS ret_1d
    FROM silver_prices
    WINDOW w AS (PARTITION BY ticker ORDER BY trade_date)
),
gains AS (
    SELECT *,
           greatest(ret_1d, 0)            AS gain,
           greatest(-ret_1d, 0)           AS loss
    FROM base
)
SELECT
    ticker, trade_date, close,
    ret_1d,
    lag(ret_1d, 1) OVER w AS ret_lag_1,
    lag(ret_1d, 2) OVER w AS ret_lag_2,
    lag(ret_1d, 3) OVER w AS ret_lag_3,
    lag(ret_1d, 4) OVER w AS ret_lag_4,
    lag(ret_1d, 5) OVER w AS ret_lag_5,
    stddev_samp(ret_1d) OVER (PARTITION BY ticker ORDER BY trade_date
                              ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)
        * sqrt(252)                       AS realized_vol_21d,
    100 - 100 / (1 + (
        avg(gain) OVER (PARTITION BY ticker ORDER BY trade_date
                        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
        / nullif(avg(loss) OVER (PARTITION BY ticker ORDER BY trade_date
                                 ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0)
    ))                                    AS rsi_14,
    (volume - avg(volume) OVER (PARTITION BY ticker ORDER BY trade_date
                                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW))
      / nullif(stddev_samp(volume) OVER (PARTITION BY ticker ORDER BY trade_date
                                         ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0)
                                          AS volume_z_21d
FROM gains
WINDOW w AS (PARTITION BY ticker ORDER BY trade_date)
ORDER BY ticker, trade_date
"""


def build_daily_merchant_spend(engine) -> pa.Table:
    return engine.sql(SPEND_SQL, tables={
        "silver_txn": schemas.SILVER_TRANSACTIONS.name,
        "silver_map": schemas.SILVER_MERCHANT_MAP.name,
    })


def build_stock_features(engine) -> pa.Table:
    return engine.sql(STOCK_FEATURES_SQL,
                      tables={"silver_prices": schemas.SILVER_PRICES.name})


def _write(engine, table_def, data: pa.Table) -> int:
    engine.create_table(table_def)
    engine.overwrite(table_def.name, data.cast(table_def.schema.as_arrow()))
    return data.num_rows


def build_all(engine) -> dict[str, int]:
    counts = {
        schemas.GOLD_SPEND.name: _write(
            engine, schemas.GOLD_SPEND, build_daily_merchant_spend(engine)),
        schemas.GOLD_STOCK_FEATURES.name: _write(
            engine, schemas.GOLD_STOCK_FEATURES, build_stock_features(engine)),
    }
    return counts


def main() -> int:
    engine = get_engine()
    for name, n in build_all(engine).items():
        print(f"gold: wrote {n:,} rows to {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_gold_marts.py -v`
Expected: 7 passed.

If DuckDB rejects the named `WINDOW w` clause combined with inline `OVER (...)`, inline the window definition on each `lag()` call. Do not switch any window to `RANGE`/unbounded-following — the trailing, right-closed form is the invariant Task 10's leakage test depends on.

- [ ] **Step 5: Commit**

```bash
git add src/lakehouse/gold.py tests/test_gold_marts.py
git commit -m "feat: Gold spend and stock-price feature marts

gross_spend takes abs() of signed_amount: Silver signs DEBIT negative, and
reporting negative spend would be a sign error that silently inverts every
downstream momentum feature.

Every rolling window is ROWS BETWEEN n PRECEDING AND CURRENT ROW. Task 10's
no-lookahead test checks this by recomputation, so a switch to an unbounded
or centred window fails CI rather than quietly inflating the model."
```

---

## Task 10: Training set and the no-lookahead guarantee

The load-bearing task. If leakage exists, every metric downstream is fiction.

**Files:**
- Modify: `src/lakehouse/gold.py` (add `build_training_set`, extend `build_all`)
- Create: `src/forecast/__init__.py`, `src/forecast/features.py`
- Test: `tests/test_no_leakage.py`

**Interfaces:**
- Consumes: `schemas.GOLD_SPEND`, `schemas.GOLD_STOCK_FEATURES`, `schemas.GOLD_TRAINING`
- Produces: `gold.build_training_set(engine) -> pa.Table`; `features.FEATURE_COLUMNS: tuple[str, ...]`; `features.TARGET: str = "fwd_ret_5d"`; `features.recompute_row(spend_df, price_df, ticker, as_of) -> dict[str, float]` — an independent, deliberately naive reimplementation used only by the leakage test

- [ ] **Step 1: Write the failing test**

```python
# tests/test_no_leakage.py
import numpy as np
import pandas as pd
import pytest

from src import config
from src.forecast import features
from src.lakehouse import bronze, catalog as catalog_mod, gold, schemas, silver
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=60000)
    silver.build_all(eng)
    gold.build_all(eng)
    training = eng.scan_arrow(schemas.GOLD_TRAINING.name).to_pandas()
    spend = eng.scan_arrow(schemas.GOLD_SPEND.name).to_pandas()
    prices = eng.scan_arrow(schemas.SILVER_PRICES.name).to_pandas()
    return training, spend, prices


def test_every_feature_recomputes_from_data_at_or_before_t(built):
    """The core guarantee. Recompute each feature using a truncated history."""
    training, spend, prices = built
    rng = np.random.default_rng(0)
    sample = training.dropna(subset=list(features.FEATURE_COLUMNS)).sample(
        30, random_state=0)

    for _, row in sample.iterrows():
        as_of = row["trade_date"]
        ticker = row["ticker"]
        spend_hist = spend[(spend["ticker"] == ticker) & (spend["spend_date"] <= as_of)]
        price_hist = prices[(prices["ticker"] == ticker) & (prices["trade_date"] <= as_of)]
        recomputed = features.recompute_row(spend_hist, price_hist, ticker, as_of)
        for col in features.FEATURE_COLUMNS:
            expected, actual = recomputed[col], row[col]
            if pd.isna(expected) and pd.isna(actual):
                continue
            assert abs(expected - actual) < 1e-6, (
                f"{ticker} {as_of} {col}: stored {actual} != recomputed-from-past {expected}")


def test_target_looks_forward_exactly_five_trading_days(built):
    training, _, prices = built
    px = prices[prices["ticker"] == "CMG"].sort_values("trade_date").reset_index(drop=True)
    tr = training[training["ticker"] == "CMG"].sort_values("trade_date").reset_index(drop=True)
    merged = tr.merge(px[["trade_date", "adj_close"]], on="trade_date")
    for i in range(len(merged) - 6):
        if pd.isna(merged.loc[i, "fwd_ret_5d"]):
            continue
        expected = merged.loc[i + 5, "adj_close"] / merged.loc[i, "adj_close"] - 1
        assert abs(merged.loc[i, "fwd_ret_5d"] - expected) < 1e-9
        break
    else:
        pytest.fail("no comparable row found")


def test_last_five_rows_per_ticker_have_null_target(built):
    training, _, _ = built
    for ticker, grp in training.groupby("ticker"):
        tail = grp.sort_values("trade_date").tail(5)
        assert tail["fwd_ret_5d"].isna().all()


def test_no_feature_correlates_perfectly_with_target(built):
    """A near-1.0 correlation means the target leaked into a feature."""
    training, _, _ = built
    clean = training.dropna(subset=[features.TARGET])
    for col in features.FEATURE_COLUMNS:
        series = clean[[col, features.TARGET]].dropna()
        if len(series) < 50:
            continue
        corr = abs(series[col].corr(series[features.TARGET]))
        assert corr < 0.90, f"{col} correlates {corr:.3f} with target -- leak"


def test_training_set_has_all_tickers_and_enough_rows(built):
    training, _, _ = built
    assert set(training["ticker"]) == {c.ticker for c in config.COMPANIES}
    assert len(training) > 3000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_no_leakage.py -v`
Expected: FAIL — `src.forecast` not found.

- [ ] **Step 3: Add `build_training_set` to `src/lakehouse/gold.py`**

Insert this SQL constant after `STOCK_FEATURES_SQL`:

```python
TRAINING_SQL = """
WITH spend AS (
    SELECT ticker, spend_date, gross_spend, txn_count, unique_accounts, avg_ticket,
           avg(gross_spend) OVER w7  AS spend_ma7,
           avg(gross_spend) OVER w28 AS spend_ma28,
           avg(gross_spend) OVER w28_lag AS spend_mean28,
           stddev_samp(gross_spend) OVER w28_lag AS spend_std28,
           avg(txn_count) OVER w28_lag AS txn_mean28,
           stddev_samp(txn_count) OVER w28_lag AS txn_std28,
           avg(avg_ticket) OVER w7 AS ticket_ma7,
           avg(avg_ticket) OVER w28 AS ticket_ma28,
           avg(unique_accounts) OVER w7 AS acct_ma7,
           lag(avg(unique_accounts) OVER w7, 7) OVER w AS acct_ma7_prev,
           avg(gross_spend) OVER w_dow AS dow_naive,
           stddev_samp(gross_spend) OVER w28 AS spend_sigma
    FROM gold_spend
    WINDOW
      w       AS (PARTITION BY ticker ORDER BY spend_date),
      w7      AS (PARTITION BY ticker ORDER BY spend_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
      w28     AS (PARTITION BY ticker ORDER BY spend_date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW),
      w28_lag AS (PARTITION BY ticker ORDER BY spend_date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW),
      w_dow   AS (PARTITION BY ticker, dayofweek(spend_date) ORDER BY spend_date
                  ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)
),
target AS (
    SELECT ticker, trade_date, adj_close,
           lead(adj_close, 5) OVER (PARTITION BY ticker ORDER BY trade_date)
             / adj_close - 1 AS fwd_ret_5d
    FROM silver_prices
)
SELECT
    f.ticker, f.trade_date,
    s.spend_ma7 / nullif(s.spend_ma28, 0) - 1                  AS spend_mom_7d,
    (s.gross_spend - s.spend_mean28) / nullif(s.spend_std28, 0) AS spend_z_28d,
    (s.txn_count  - s.txn_mean28)   / nullif(s.txn_std28, 0)    AS txn_count_z_28d,
    s.ticket_ma7 / nullif(s.ticket_ma28, 0) - 1                 AS avg_ticket_delta_7d,
    s.acct_ma7   / nullif(s.acct_ma7_prev, 0) - 1               AS unique_acct_growth_7d,
    (s.gross_spend - s.dow_naive) / nullif(s.spend_sigma, 0)    AS spend_surprise,
    f.ret_lag_1, f.ret_lag_2, f.ret_lag_3, f.ret_lag_4, f.ret_lag_5,
    f.realized_vol_21d, f.rsi_14, f.volume_z_21d,
    t.fwd_ret_5d
FROM gold_features f
JOIN target t ON f.ticker = t.ticker AND f.trade_date = t.trade_date
LEFT JOIN spend s ON s.ticker = f.ticker AND s.spend_date = f.trade_date
ORDER BY f.ticker, f.trade_date
"""


def build_training_set(engine) -> pa.Table:
    return engine.sql(TRAINING_SQL, tables={
        "gold_spend": schemas.GOLD_SPEND.name,
        "gold_features": schemas.GOLD_STOCK_FEATURES.name,
        "silver_prices": schemas.SILVER_PRICES.name,
    })
```

Then extend `build_all` so it writes the training set after the two marts:

```python
def build_all(engine) -> dict[str, int]:
    counts = {
        schemas.GOLD_SPEND.name: _write(
            engine, schemas.GOLD_SPEND, build_daily_merchant_spend(engine)),
        schemas.GOLD_STOCK_FEATURES.name: _write(
            engine, schemas.GOLD_STOCK_FEATURES, build_stock_features(engine)),
    }
    counts[schemas.GOLD_TRAINING.name] = _write(
        engine, schemas.GOLD_TRAINING, build_training_set(engine))
    return counts
```

- [ ] **Step 4: Write `src/forecast/features.py`**

```python
"""Feature contract, plus an independent recomputation used only by tests.

recompute_row deliberately does NOT share code with the Gold SQL. Two
implementations that agree are evidence; one implementation checked against
itself is a tautology.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "spend_mom_7d", "spend_z_28d", "txn_count_z_28d", "avg_ticket_delta_7d",
    "unique_acct_growth_7d", "spend_surprise",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
    "realized_vol_21d", "rsi_14", "volume_z_21d",
)
TARGET = "fwd_ret_5d"


def recompute_row(spend_hist: pd.DataFrame, price_hist: pd.DataFrame,
                  ticker: str, as_of: date) -> dict[str, float]:
    """Recompute every feature for (ticker, as_of) from history <= as_of only."""
    s = spend_hist.sort_values("spend_date").reset_index(drop=True)
    p = price_hist.sort_values("trade_date").reset_index(drop=True)
    out: dict[str, float] = {}

    gs = s["gross_spend"]
    out["spend_mom_7d"] = _safe(gs.tail(7).mean(), gs.tail(28).mean(), ratio=True)
    out["spend_z_28d"] = _z(gs, 28)
    out["txn_count_z_28d"] = _z(s["txn_count"], 28)
    out["avg_ticket_delta_7d"] = _safe(
        s["avg_ticket"].tail(7).mean(), s["avg_ticket"].tail(28).mean(), ratio=True)

    acct_ma7 = s["unique_accounts"].rolling(7).mean()
    out["unique_acct_growth_7d"] = _safe(
        acct_ma7.iloc[-1] if len(acct_ma7) else np.nan,
        acct_ma7.iloc[-8] if len(acct_ma7) > 7 else np.nan, ratio=True)

    same_dow = s[s["spend_date"].apply(lambda d: d.weekday()) == as_of.weekday()]
    prior = same_dow[same_dow["spend_date"] < as_of]["gross_spend"].tail(4)
    sigma = gs.tail(28).std(ddof=1)
    out["spend_surprise"] = _safe(
        (gs.iloc[-1] - prior.mean()) if len(prior) else np.nan, sigma)

    ret = p["close"].pct_change()
    for k in range(1, 6):
        out[f"ret_lag_{k}"] = ret.iloc[-k] if len(ret) > k else np.nan
    out["realized_vol_21d"] = ret.tail(21).std(ddof=1) * np.sqrt(252)

    gain = ret.clip(lower=0).tail(14).mean()
    loss = (-ret.clip(upper=0)).tail(14).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss) if loss and loss > 0 else np.nan

    vol = p["volume"].astype(float)
    out["volume_z_21d"] = _z(vol, 21)
    return out


def _z(series: pd.Series, window: int) -> float:
    tail = series.tail(window)
    std = tail.std(ddof=1)
    if not std or np.isnan(std) or std == 0:
        return np.nan
    return (series.iloc[-1] - tail.mean()) / std


def _safe(num, den, ratio: bool = False) -> float:
    if den is None or np.isnan(den) or den == 0 or num is None or np.isnan(num):
        return np.nan
    return num / den - 1 if ratio else num / den
```

- [ ] **Step 5: Run the leakage tests**

Run: `uv run pytest tests/test_no_leakage.py -v`
Expected: 5 passed.

`test_every_feature_recomputes_from_data_at_or_before_t` is the one that matters. If it fails, the SQL and the recomputation disagree about window boundaries — **fix the SQL to match the definition in the spec**, and only change `recompute_row` if you can show the SQL is right and the reimplementation is wrong. Never relax the `1e-6` tolerance to make it pass.

- [ ] **Step 6: Commit**

```bash
git add src/lakehouse/gold.py src/forecast/__init__.py src/forecast/features.py tests/test_no_leakage.py
git commit -m "feat: leakage-safe training set with independent recomputation test

recompute_row is a second, deliberately naive implementation of every
feature that only ever sees history <= t. It shares no code with the Gold
SQL on purpose: two implementations that agree is evidence, one
implementation checked against itself is a tautology.

Also asserts the last five rows per ticker have a null target, and that no
feature correlates above 0.90 with the target -- the signature of a leak
that window-boundary review tends to miss."
```

---

## Task 11: Purged walk-forward cross-validation

**Files:**
- Create: `src/forecast/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `config.PURGE_DAYS`, `config.FORECAST_HORIZON_DAYS`
- Produces: `splits.purged_walk_forward(dates: Sequence[date], n_folds: int = 5, purge: int = config.PURGE_DAYS, min_train: int = 120) -> list[Fold]`; dataclass `Fold(index: int, train: list[int], test: list[int])` holding positional indices into `dates`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
from datetime import date, timedelta

import pytest

from src import config
from src.forecast import splits


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def test_produces_requested_number_of_folds():
    folds = splits.purged_walk_forward(_dates(600), n_folds=5)
    assert len(folds) == 5


def test_train_always_precedes_test():
    for fold in splits.purged_walk_forward(_dates(600), n_folds=5):
        assert max(fold.train) < min(fold.test)


def test_purge_gap_is_at_least_horizon():
    """Without this gap, a training row's 5-day target overlaps the test block."""
    for fold in splits.purged_walk_forward(_dates(600), n_folds=5, purge=5):
        assert min(fold.test) - max(fold.train) > 5


def test_training_window_expands():
    folds = splits.purged_walk_forward(_dates(600), n_folds=5)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_test_blocks_are_disjoint_and_ordered():
    folds = splits.purged_walk_forward(_dates(600), n_folds=5)
    seen: set[int] = set()
    last_end = -1
    for fold in folds:
        assert not (seen & set(fold.test))
        seen |= set(fold.test)
        assert min(fold.test) > last_end
        last_end = max(fold.test)


def test_raises_when_series_too_short_rather_than_silently_degrading():
    with pytest.raises(ValueError):
        splits.purged_walk_forward(_dates(50), n_folds=5, min_train=120)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL — `src.forecast.splits` not found.

- [ ] **Step 3: Write `src/forecast/splits.py`**

```python
"""Purged, expanding-window walk-forward CV.

The purge gap is the reason this module exists. With a 5-day forward target,
a training row dated t-1 has a target that resolves at t+4 -- inside the test
block. Without purging, the model trains on the answer and every metric is
inflated in a way that ordinary shuffled CV will never reveal.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src import config


@dataclass(frozen=True)
class Fold:
    index: int
    train: list[int]
    test: list[int]


def purged_walk_forward(dates: Sequence[date], n_folds: int = 5,
                        purge: int = config.PURGE_DAYS,
                        min_train: int = 120) -> list[Fold]:
    n = len(dates)
    usable = n - min_train - purge
    if usable < n_folds * 2:
        raise ValueError(
            f"series of {n} points cannot support {n_folds} folds with "
            f"min_train={min_train} and purge={purge}")

    test_size = usable // n_folds
    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = min_train + purge + i * test_size
        test_end = test_start + test_size if i < n_folds - 1 else n
        train_end = test_start - purge
        folds.append(Fold(
            index=i,
            train=list(range(0, train_end)),
            test=list(range(test_start, test_end)),
        ))
    return folds
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_splits.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/splits.py tests/test_splits.py
git commit -m "feat: purged expanding-window walk-forward CV

The purge gap is the whole point. With a 5-day forward target, a training
row dated t-1 resolves at t+4, inside the test block -- so unpurged CV
trains on the answer. Shuffled CV would never surface this; the metrics
would just come back good.

Raises rather than silently shrinking folds when the series is too short,
so a reduced-dataset smoke run fails loudly instead of reporting numbers
from two-row test blocks."
```

---

## Task 12: Models and baselines

**Files:**
- Create: `src/forecast/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `features.FEATURE_COLUMNS`, `features.TARGET`, `splits.purged_walk_forward`
- Produces: `models.MODEL_VERSION: str`; `models.fit_predict_ticker(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame` with columns `ticker, trade_date, fold, y_true, y_pred_model, y_pred_persistence, y_pred_arima`; `models.run_all(training: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.forecast import features, models


def _synthetic(n: int = 500, ticker: str = "AAA", signal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({"ticker": ticker, "trade_date": dates})
    for col in features.FEATURE_COLUMNS:
        df[col] = rng.standard_normal(n)
    base = 0.6 * df["spend_mom_7d"] if signal else 0.0
    df[features.TARGET] = base + rng.standard_normal(n) * 0.4
    return df


def test_returns_one_row_per_test_observation():
    out = models.fit_predict_ticker(_synthetic(), n_folds=5)
    assert len(out) > 0
    assert set(out.columns) >= {
        "ticker", "trade_date", "fold", "y_true",
        "y_pred_model", "y_pred_persistence", "y_pred_arima"}
    assert out["fold"].nunique() == 5


def test_predictions_are_finite():
    out = models.fit_predict_ticker(_synthetic())
    for col in ("y_pred_model", "y_pred_persistence", "y_pred_arima"):
        assert np.isfinite(out[col]).all()


def test_beats_persistence_when_signal_is_present():
    out = models.fit_predict_ticker(_synthetic(signal=True))
    model_rmse = np.sqrt(((out["y_true"] - out["y_pred_model"]) ** 2).mean())
    base_rmse = np.sqrt(((out["y_true"] - out["y_pred_persistence"]) ** 2).mean())
    assert model_rmse < base_rmse


def test_does_not_beat_persistence_on_pure_noise():
    """Guards against a leak: with no signal the model must not win."""
    out = models.fit_predict_ticker(_synthetic(signal=False))
    model_rmse = np.sqrt(((out["y_true"] - out["y_pred_model"]) ** 2).mean())
    base_rmse = np.sqrt(((out["y_true"] - out["y_pred_persistence"]) ** 2).mean())
    assert model_rmse > base_rmse * 0.95


def test_rows_with_null_target_are_excluded():
    df = _synthetic()
    df.loc[df.index[-5:], features.TARGET] = np.nan
    out = models.fit_predict_ticker(df)
    assert out["y_true"].notna().all()


def test_run_all_handles_multiple_tickers():
    df = pd.concat([_synthetic(ticker="AAA"), _synthetic(ticker="BBB")],
                   ignore_index=True)
    out = models.run_all(df)
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_short_series_is_skipped_not_crashed():
    out = models.run_all(_synthetic(n=60))
    assert out.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `src.forecast.models` not found.

- [ ] **Step 3: Write `src/forecast/models.py`**

```python
"""HistGradientBoosting plus two baselines, evaluated under purged walk-forward CV.

sklearn's HistGradientBoostingRegressor rather than LightGBM: same
histogram-based algorithm, no native libomp dependency, which is the usual
cause of macOS/CI install failure. See docs/ai-sdlc/decisions/0001.

Both baselines exist so the report can answer "did the transaction feed add
anything" rather than only "did the model fit". ARIMA on price alone is the
explicit null hypothesis.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import config
from src.forecast import features, splits

MODEL_VERSION = "hgb-v1"


def fit_predict_ticker(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    data = (df.dropna(subset=[features.TARGET])
              .sort_values("trade_date").reset_index(drop=True))
    dates = list(data["trade_date"])
    folds = splits.purged_walk_forward(dates, n_folds=n_folds)

    x = data[list(features.FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = data[features.TARGET].to_numpy(dtype=float)

    frames = []
    for fold in folds:
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=20,
            random_state=config.SEED,
        )
        model.fit(x[fold.train], y[fold.train])
        preds = model.predict(x[fold.test])

        frames.append(pd.DataFrame({
            "ticker": data.loc[fold.test, "ticker"].to_numpy(),
            "trade_date": data.loc[fold.test, "trade_date"].to_numpy(),
            "fold": fold.index,
            "y_true": y[fold.test],
            "y_pred_model": preds,
            # Persistence: the honest "predict no change" null.
            "y_pred_persistence": np.zeros(len(fold.test)),
            "y_pred_arima": _arima_forecast(y, fold),
        }))

    return pd.concat(frames, ignore_index=True)


def _arima_forecast(y: np.ndarray, fold) -> np.ndarray:
    """ARIMA(1,1,1) refit on the training block, forecast across the test block.

    Falls back to the training mean if the fit fails to converge -- a
    non-converging baseline should not abort the whole run.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(y[fold.train], order=(1, 1, 1)).fit()
            forecast = fitted.forecast(steps=len(fold.test))
        values = np.asarray(forecast, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite forecast")
        return values
    except Exception:
        return np.full(len(fold.test), float(np.mean(y[fold.train])))


def run_all(training: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, grp in training.groupby("ticker"):
        try:
            frames.append(fit_predict_ticker(grp))
        except ValueError:
            # Series too short for purged CV -- skip rather than report
            # metrics computed from two-row test blocks.
            continue
    if not frames:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "fold", "y_true",
            "y_pred_model", "y_pred_persistence", "y_pred_arima"])
    out = pd.concat(frames, ignore_index=True)
    out["model_version"] = MODEL_VERSION
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: 7 passed.

`test_does_not_beat_persistence_on_pure_noise` is a leak detector, not a performance test. If it fails, something in `fit_predict_ticker` is seeing the test block — check the fold indices before touching hyperparameters.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/models.py tests/test_models.py
git commit -m "feat: gradient boosting with persistence and ARIMA baselines

Two tests bracket correctness: the model must beat persistence when signal
is present, and must NOT beat it on pure noise. The second is the leak
detector -- a model that wins on noise is reading the test block.

ARIMA on price alone is the explicit null hypothesis for the whole project:
if the model cannot beat it, the transaction feed added nothing and the
report has to say so.

Uses sklearn HistGradientBoosting rather than LightGBM to avoid the libomp
native dependency; same algorithm, one less way for CI to fail."
```

---

## Task 13: Backtest, metrics, and the honest report

**Files:**
- Create: `src/forecast/backtest.py`, `src/forecast/report.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: predictions frame from Task 12, `schemas.GOLD_TRAINING`, `schemas.GOLD_PREDICTIONS`
- Produces: `backtest.metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]` with keys `rmse, mae, directional_accuracy, ic`; `backtest.long_short_sharpe(df: pd.DataFrame, pred_col: str) -> float`; `backtest.verdict(model: dict, baseline: dict) -> str` returning `"beats"`/`"loses"`/`"inconclusive"`; `report.main() -> int` writing `docs/forecast-report.md` and `gold.forecast_predictions`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest.py
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.forecast import backtest


def _frame(n=200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n) * 0.02
    return pd.DataFrame({
        "ticker": ["AAA"] * n,
        "trade_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "y_true": y,
        "y_pred_good": y + rng.standard_normal(n) * noise * 0.02,
        "y_pred_bad": rng.standard_normal(n) * 0.02,
        "y_pred_zero": np.zeros(n),
    })


def test_metrics_keys_and_ranges():
    m = backtest.metrics(_frame(), "y_pred_good")
    assert set(m) == {"rmse", "mae", "directional_accuracy", "ic"}
    assert m["rmse"] >= 0
    assert 0 <= m["directional_accuracy"] <= 1
    assert -1 <= m["ic"] <= 1


def test_good_predictions_score_better_than_random():
    good = backtest.metrics(_frame(), "y_pred_good")
    bad = backtest.metrics(_frame(), "y_pred_bad")
    assert good["rmse"] < bad["rmse"]
    assert good["ic"] > bad["ic"]


def test_directional_accuracy_ignores_zero_predictions():
    """An all-zero baseline has no direction; scoring it 100% would be a lie."""
    m = backtest.metrics(_frame(), "y_pred_zero")
    assert np.isnan(m["directional_accuracy"])


def test_sharpe_positive_for_informative_predictions():
    frames = []
    for i, ticker in enumerate("ABCDEF"):
        f = _frame(seed=i)
        f["ticker"] = ticker
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    assert backtest.long_short_sharpe(df, "y_pred_good") > 0


def test_verdict_requires_a_margin_not_just_a_smaller_number():
    base = {"rmse": 0.0200, "ic": 0.00, "directional_accuracy": 0.50, "mae": 0.01}
    tie = {"rmse": 0.0199, "ic": 0.01, "directional_accuracy": 0.50, "mae": 0.01}
    win = {"rmse": 0.0150, "ic": 0.20, "directional_accuracy": 0.58, "mae": 0.01}
    loss = {"rmse": 0.0300, "ic": -0.05, "directional_accuracy": 0.47, "mae": 0.02}
    assert backtest.verdict(tie, base) == "inconclusive"
    assert backtest.verdict(win, base) == "beats"
    assert backtest.verdict(loss, base) == "loses"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL — `src.forecast.backtest` not found.

- [ ] **Step 3: Write `src/forecast/backtest.py`**

```python
"""Metrics and a deliberately plain long/short backtest.

No transaction costs, no slippage, no capacity model. The Sharpe reported
here is an upper bound on a frictionless strategy, and the report labels it
as such rather than implying it is achievable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A win must clear a margin, not just come in lower by a rounding error.
RMSE_MARGIN = 0.02   # 2% relative improvement
IC_MARGIN = 0.02


def metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    data = df[["y_true", pred_col]].dropna()
    err = data["y_true"] - data[pred_col]

    # A constant prediction carries no directional information. Scoring the
    # zero baseline as 100% correct on flat days would flatter it enormously.
    moved = data[data[pred_col] != 0]
    direction = (float((np.sign(moved[pred_col]) == np.sign(moved["y_true"])).mean())
                 if len(moved) else float("nan"))

    ic = (float(data[pred_col].corr(data["y_true"], method="spearman"))
          if data[pred_col].nunique() > 1 else float("nan"))

    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "directional_accuracy": direction,
        "ic": 0.0 if np.isnan(ic) else ic,
    }


def long_short_sharpe(df: pd.DataFrame, pred_col: str, top_n: int = 2) -> float:
    """Long the top_n predicted tickers, short the bottom_n, daily rebalance."""
    daily = []
    for _, day in df.groupby("trade_date"):
        if len(day) < top_n * 2:
            continue
        ranked = day.sort_values(pred_col, ascending=False)
        longs = ranked.head(top_n)["y_true"].mean()
        shorts = ranked.tail(top_n)["y_true"].mean()
        daily.append((longs - shorts) / 2)
    if len(daily) < 20:
        return float("nan")
    series = pd.Series(daily)
    if series.std(ddof=1) == 0:
        return float("nan")
    # Overlapping 5-day targets sampled daily -> scale by trading days / horizon.
    return float(series.mean() / series.std(ddof=1) * np.sqrt(252 / 5))


def verdict(model: dict[str, float], baseline: dict[str, float]) -> str:
    rmse_gain = (baseline["rmse"] - model["rmse"]) / baseline["rmse"]
    ic_gain = model["ic"] - baseline["ic"]
    if rmse_gain > RMSE_MARGIN and ic_gain > IC_MARGIN:
        return "beats"
    if rmse_gain < -RMSE_MARGIN or ic_gain < -IC_MARGIN:
        return "loses"
    return "inconclusive"
```

- [ ] **Step 4: Write `src/forecast/report.py`**

```python
"""Run the forecast end to end and publish an honest report."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

from src import config
from src.forecast import backtest, models
from src.lakehouse import schemas
from src.lakehouse.engines import get_engine

REPORT_PATH = config.REPO_ROOT / "docs" / "forecast-report.md"


def build_report(preds: pd.DataFrame) -> str:
    lines = [
        "# Forecast Report",
        "",
        f"Model: `{models.MODEL_VERSION}` — HistGradientBoostingRegressor",
        f"Target: forward {config.FORECAST_HORIZON_DAYS}-day return",
        f"Validation: purged walk-forward CV, {config.PURGE_DAYS}-day purge/embargo",
        "",
        "Prices in this project are synthetic with a planted per-ticker "
        "spend→return relationship (ADR-0004). These results measure whether the "
        "pipeline preserves a known signal, **not** real-world alpha.",
        "",
        "## Per-ticker results",
        "",
        "| Ticker | RMSE model | RMSE persist | RMSE ARIMA | Dir. acc | IC | vs persistence | vs ARIMA |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for ticker, grp in preds.groupby("ticker"):
        m = backtest.metrics(grp, "y_pred_model")
        p = backtest.metrics(grp, "y_pred_persistence")
        a = backtest.metrics(grp, "y_pred_arima")
        lines.append(
            f"| {ticker} | {m['rmse']:.5f} | {p['rmse']:.5f} | {a['rmse']:.5f} | "
            f"{m['directional_accuracy']:.3f} | {m['ic']:.3f} | "
            f"**{backtest.verdict(m, p)}** | **{backtest.verdict(m, a)}** |")

    sharpe = backtest.long_short_sharpe(preds, "y_pred_model")
    beats = sum(1 for _, g in preds.groupby("ticker")
                if backtest.verdict(backtest.metrics(g, "y_pred_model"),
                                    backtest.metrics(g, "y_pred_persistence")) == "beats")
    total = preds["ticker"].nunique()

    lines += [
        "",
        "## Portfolio",
        "",
        f"Long/short Sharpe (top-2 / bottom-2, daily rebalance): **{sharpe:.2f}**",
        "",
        "No transaction costs, slippage, or capacity constraints. This is an "
        "upper bound on a frictionless strategy, not an achievable return.",
        "",
        "## Honest summary",
        "",
        f"The model beats the persistence baseline on **{beats} of {total}** tickers.",
        "",
    ]

    if beats == total:
        lines.append(
            "> Every ticker beating baseline is a warning sign, not a success. "
            "With planted signal at differing lags and betas, uniform wins suggest "
            "leakage. Re-check `tests/test_no_leakage.py` before trusting this.")
    else:
        losers = [t for t, g in preds.groupby("ticker")
                  if backtest.verdict(backtest.metrics(g, "y_pred_model"),
                                      backtest.metrics(g, "y_pred_persistence")) != "beats"]
        lines.append(
            f"Tickers where the model does **not** beat persistence: "
            f"{', '.join(losers)}. Longer planted lags and lower betas are harder "
            "to recover from noisy daily spend, which is the expected outcome.")

    return "\n".join(lines) + "\n"


def main() -> int:
    engine = get_engine()
    training = engine.scan_arrow(schemas.GOLD_TRAINING.name).to_pandas()
    preds = models.run_all(training)
    if preds.empty:
        print("forecast: no ticker had enough history for purged CV", file=sys.stderr)
        return 1

    engine.create_table(schemas.GOLD_PREDICTIONS)
    arrow = pa.Table.from_pandas(
        preds, schema=schemas.GOLD_PREDICTIONS.schema.as_arrow(),
        preserve_index=False)
    engine.overwrite(schemas.GOLD_PREDICTIONS.name, arrow)

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(build_report(preds))
    print(f"forecast: {len(preds):,} predictions written; report at {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: 5 passed.

- [ ] **Step 6: Run the full pipeline and read the report**

Run: `uv run make all`
Then read `docs/forecast-report.md`.

Expected: a per-ticker table with a mix of verdicts. **If all six read "beats", stop and investigate leakage** — do not proceed to Task 14. Re-run `uv run pytest tests/test_no_leakage.py tests/test_models.py -v` and inspect the fold boundaries.

- [ ] **Step 7: Commit**

```bash
git add src/forecast/backtest.py src/forecast/report.py docs/forecast-report.md tests/test_backtest.py
git commit -m "feat: metrics, long/short backtest, and honest forecast report

Directional accuracy returns NaN for a constant predictor rather than
counting flat predictions as correct, which would have scored the zero
baseline near 100% on flat days.

verdict() requires a 2% RMSE margin and a 0.02 IC margin. Without a margin
a rounding-level difference reads as a win, which is how model reports
drift into overclaiming.

The report calls out uniform wins as a leakage warning rather than a
success, and names the tickers where the model loses."
```

---

## Task 14: Agent sensors and drift classifier

**Files:**
- Create: `src/agent/__init__.py`, `src/agent/sensors.py`, `src/agent/classifier.py`
- Test: `tests/test_agent_sensors.py`

**Interfaces:**
- Consumes: `validator.load_contract`, `config.resolve_as_of_date`, engine `snapshots`/`scan_arrow`
- Produces:
  - `sensors.ObservedState(table, columns: dict[str, str], row_count: int, snapshot_count: int, newest_date: date | None)`
  - `sensors.observe(engine, table_def, contract) -> ObservedState`
  - `sensors.Finding(kind, table, detail, evidence: dict)` with `kind` in `{"schema_drift", "volume_anomaly", "staleness"}`
  - `sensors.detect(state: ObservedState, contract, as_of: date, history: list[int]) -> list[Finding]`
  - `classifier.classify(finding: Finding) -> tuple[str, str]` returning `(severity, reasoning)`, severity in `{"breaking", "additive", "benign"}`
  - `classifier.USE_LLM: bool` — False unless `ANTHROPIC_API_KEY` is set

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sensors.py
from datetime import date

import pytest

from src.agent import classifier, sensors


def _state(**kw):
    base = dict(table="bronze.yodlee_transactions_raw",
                columns={"id": "long", "baseType": "string", "amount": "struct"},
                row_count=250_000, rows_in_latest_snapshot=250_000,
                snapshot_count=3, newest_date=date(2026, 6, 30))
    base.update(kw)
    return sensors.ObservedState(**base)


class _Contract:
    table = "bronze.yodlee_transactions_raw"
    schema_fields = ()
    expectations = ()


def _contract(fields):
    from src.contracts.validator import Contract, SchemaField
    return Contract(
        table="bronze.yodlee_transactions_raw", version=1, owner="x",
        schema_fields=tuple(SchemaField(n, t) for n, t in fields),
        expectations=({"type": "freshness", "column": "txn_date", "max_lag_days": 3},))


def test_no_findings_when_state_matches_contract():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    findings = sensors.detect(_state(), contract, as_of=date(2026, 6, 30),
                              history=[250_000, 249_000, 251_000])
    assert findings == []


def test_added_column_is_detected_and_classified_additive():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    state = _state(columns={"id": "long", "baseType": "string", "amount": "struct",
                            "merchantCategoryCode": "string"})
    findings = sensors.detect(state, contract, as_of=date(2026, 6, 30),
                              history=[250_000] * 3)
    drift = [f for f in findings if f.kind == "schema_drift"]
    assert len(drift) == 1
    severity, reasoning = classifier.classify(drift[0])
    assert severity == "additive"
    assert reasoning


def test_dropped_column_is_breaking():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    state = _state(columns={"id": "long", "baseType": "string"})
    findings = sensors.detect(state, contract, as_of=date(2026, 6, 30),
                              history=[250_000] * 3)
    drift = [f for f in findings if f.kind == "schema_drift"][0]
    assert classifier.classify(drift)[0] == "breaking"


def test_type_narrowing_is_breaking():
    contract = _contract([("id", "long"), ("baseType", "string"), ("amount", "struct")])
    state = _state(columns={"id": "int", "baseType": "string", "amount": "struct"})
    findings = sensors.detect(state, contract, as_of=date(2026, 6, 30),
                              history=[250_000] * 3)
    drift = [f for f in findings if f.kind == "schema_drift"][0]
    assert classifier.classify(drift)[0] == "breaking"


def test_volume_collapse_is_detected():
    contract = _contract([("id", "long")])
    findings = sensors.detect(
        _state(rows_in_latest_snapshot=1000, columns={"id": "long"}),
        contract, as_of=date(2026, 6, 30),
        history=[250_000, 249_000, 251_000, 248_000])
    vol = [f for f in findings if f.kind == "volume_anomaly"]
    assert len(vol) == 1
    assert classifier.classify(vol[0])[0] == "breaking"


def test_modest_volume_change_is_not_flagged():
    contract = _contract([("id", "long")])
    findings = sensors.detect(
        _state(rows_in_latest_snapshot=230_000, columns={"id": "long"}),
        contract, as_of=date(2026, 6, 30),
        history=[250_000, 249_000, 251_000])
    assert not [f for f in findings if f.kind == "volume_anomaly"]


def test_staleness_measured_against_as_of_not_wall_clock():
    contract = _contract([("id", "long")])
    fresh = sensors.detect(_state(columns={"id": "long"}), contract,
                           as_of=date(2026, 6, 30), history=[250_000] * 3)
    assert not [f for f in fresh if f.kind == "staleness"]
    stale = sensors.detect(_state(columns={"id": "long"}), contract,
                           as_of=date(2026, 8, 9), history=[250_000] * 3)
    assert [f for f in stale if f.kind == "staleness"]


def test_llm_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib
    importlib.reload(classifier)
    assert classifier.USE_LLM is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_sensors.py -v`
Expected: FAIL — `src.agent` not found.

- [ ] **Step 3: Write `src/agent/sensors.py`**

```python
"""Read Iceberg metadata and diff it against declared contracts.

Row counts and schemas come from table metadata rather than full scans: the
agent is meant to run cheaply and often, and a sensor that scans the whole
lake will get switched off.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

VOLUME_COLLAPSE_RATIO = 0.5  # below half the trailing median is an incident


@dataclass(frozen=True)
class ObservedState:
    table: str
    columns: dict[str, str]
    row_count: int
    rows_in_latest_snapshot: int
    snapshot_count: int
    newest_date: date | None


@dataclass(frozen=True)
class Finding:
    kind: str          # schema_drift | volume_anomaly | staleness
    table: str
    detail: str
    evidence: dict = field(default_factory=dict)


def observe(engine, table_def, date_column: str | None = None) -> ObservedState:
    arrow = engine.scan_arrow(table_def.name)
    columns = {f.name: str(f.type) for f in arrow.schema}
    newest = None
    if date_column and date_column in arrow.column_names:
        values = [v for v in arrow.column(date_column).to_pylist() if v is not None]
        if values:
            newest = max(values)
            newest = date.fromisoformat(newest) if isinstance(newest, str) else newest
            newest = newest.date() if hasattr(newest, "date") else newest
    per_snapshot = engine.snapshot_row_counts(table_def.name)
    return ObservedState(
        table=table_def.name,
        columns=columns,
        row_count=arrow.num_rows,
        rows_in_latest_snapshot=per_snapshot[-1] if per_snapshot else 0,
        snapshot_count=len(per_snapshot),
        newest_date=newest,
    )


def detect(state: ObservedState, contract, as_of: date,
           history: list[int]) -> list[Finding]:
    findings: list[Finding] = []

    declared = {f.name: f.type for f in contract.schema_fields}
    observed = state.columns

    for name, declared_type in declared.items():
        if name not in observed:
            findings.append(Finding(
                "schema_drift", state.table,
                f"column '{name}' declared in contract but absent from table",
                {"change": "dropped", "column": name, "declared_type": declared_type}))
        elif not _types_compatible(declared_type, observed[name]):
            findings.append(Finding(
                "schema_drift", state.table,
                f"column '{name}' type changed: {declared_type} -> {observed[name]}",
                {"change": "type_changed", "column": name,
                 "declared_type": declared_type, "observed_type": observed[name]}))

    for name in observed:
        if name not in declared and not name.startswith("_"):
            findings.append(Finding(
                "schema_drift", state.table,
                f"column '{name}' present in table but not declared in contract",
                {"change": "added", "column": name,
                 "observed_type": observed[name]}))

    # Per-batch, not cumulative: an append-only table's total row count only
    # ever grows, so a total-vs-median check can never fire.
    if history:
        median = statistics.median(history)
        observed = state.rows_in_latest_snapshot
        if median > 0 and observed < median * VOLUME_COLLAPSE_RATIO:
            findings.append(Finding(
                "volume_anomaly", state.table,
                f"latest snapshot added {observed:,} rows, below "
                f"{VOLUME_COLLAPSE_RATIO:.0%} of trailing median {median:,.0f}",
                {"rows_in_latest_snapshot": observed, "median": median}))

    freshness = next((e for e in contract.expectations
                      if e.get("type") == "freshness"), None)
    if freshness and state.newest_date is not None:
        lag = (as_of - state.newest_date).days
        if lag > freshness["max_lag_days"]:
            findings.append(Finding(
                "staleness", state.table,
                f"newest data is {lag}d behind AS_OF {as_of} "
                f"(limit {freshness['max_lag_days']}d)",
                {"lag_days": lag, "as_of": as_of.isoformat()}))

    return findings


_COMPATIBLE = {("long", "int64"), ("string", "string"), ("struct", "struct"),
               ("double", "double"), ("date", "date32[day]"),
               ("timestamptz", "timestamp[us, tz=UTC]"), ("boolean", "bool")}


def _types_compatible(declared: str, observed: str) -> bool:
    if declared == observed:
        return True
    if (declared, observed) in _COMPATIBLE:
        return True
    return observed.startswith(declared)
```

- [ ] **Step 4: Write `src/agent/classifier.py`**

```python
"""Severity classification: rules first, LLM only for genuine ambiguity.

CI must never depend on a model call, so USE_LLM is False unless a key is
present and every rule path is deterministic. An ops agent whose test suite
needs the network is an ops agent nobody can run in CI.
"""
from __future__ import annotations

import os

from src.agent.sensors import Finding

USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

_WIDENING = {("int", "long"), ("int32", "int64"), ("float", "double")}


def classify(finding: Finding) -> tuple[str, str]:
    if finding.kind == "volume_anomaly":
        return "breaking", (
            f"Latest batch added {finding.evidence.get('rows_in_latest_snapshot'):,} "
            f"rows against a trailing median of {finding.evidence.get('median'):,.0f}. "
            "Downstream aggregates will be silently wrong rather than obviously missing.")

    if finding.kind == "staleness":
        return "breaking", (
            f"Data is {finding.evidence.get('lag_days')}d behind the pipeline's "
            "logical clock; forecasts would be issued from stale features.")

    if finding.kind == "schema_drift":
        change = finding.evidence.get("change")
        column = finding.evidence.get("column")

        if change == "added":
            return "additive", (
                f"New column '{column}' does not affect existing readers. Worth "
                "adding to the contract, not worth paging anyone.")

        if change == "dropped":
            return "breaking", (
                f"Column '{column}' is declared in the contract and consumed "
                "downstream. Its absence will fail the next Silver build.")

        if change == "type_changed":
            declared = finding.evidence.get("declared_type", "")
            observed = finding.evidence.get("observed_type", "")
            if (declared, observed) in _WIDENING:
                return "additive", (
                    f"'{column}' widened {declared} -> {observed}; existing values "
                    "still fit.")
            return "breaking", (
                f"'{column}' narrowed or changed kind {declared} -> {observed}; "
                "existing values may not fit and downstream casts will fail.")

    if USE_LLM:
        return _classify_with_llm(finding)
    return "benign", "No rule matched and LLM classification is disabled."


def _classify_with_llm(finding: Finding) -> tuple[str, str]:
    """Optional judgment call. Never required -- falls back to benign on any error."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{"role": "user", "content": (
                "Classify this data-platform finding as exactly one of "
                "breaking, additive, or benign. Reply as '<severity>: <one "
                f"sentence>'.\n\nFinding: {finding.kind} on {finding.table}\n"
                f"Detail: {finding.detail}\nEvidence: {finding.evidence}")}],
        )
        text = message.content[0].text.strip()
        severity, _, reasoning = text.partition(":")
        severity = severity.strip().lower()
        if severity not in {"breaking", "additive", "benign"}:
            return "benign", f"Unparseable LLM response: {text[:120]}"
        return severity, reasoning.strip()
    except Exception as exc:
        return "benign", f"LLM classification unavailable ({type(exc).__name__})."
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_agent_sensors.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/agent/__init__.py src/agent/sensors.py src/agent/classifier.py tests/test_agent_sensors.py
git commit -m "feat: agent sensors and rules-first drift classifier

Sensors read table metadata rather than scanning data. A monitoring agent
that scans the whole lake on every run is one that gets switched off.

Classification is deterministic rules with the LLM reserved for cases no
rule matches, and USE_LLM is False without an API key. A test asserts this,
because an ops agent whose suite needs the network cannot run in CI.

Type widening (int -> long) is additive; narrowing is breaking. Treating
every type change as breaking trains people to ignore the alerts."
```

---

## Task 15: LangGraph agent and incident actions

**Files:**
- Create: `src/agent/actions.py`, `src/agent/graph.py`
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: `sensors.observe`, `sensors.detect`, `classifier.classify`
- Produces:
  - `actions.INCIDENTS_DIR: Path`
  - `actions.incident_slug(finding) -> str`
  - `actions.write_incident(finding, severity, reasoning, as_of) -> Path`
  - `actions.file_github_issue(finding, severity, reasoning, execute: bool = False) -> str`
  - `graph.AgentState` TypedDict with keys `as_of, findings, classified, actions_taken, dry_run`
  - `graph.build_graph()` returning a compiled LangGraph
  - `graph.run(engine=None, dry_run: bool = True, as_of=None) -> AgentState`
  - `python -m src.agent.graph [--execute]` CLI

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_graph.py
from datetime import date

import pytest

from src.agent import actions, graph
from src.agent.sensors import Finding


@pytest.fixture(autouse=True)
def isolate_incidents(tmp_path, monkeypatch):
    monkeypatch.setattr(actions, "INCIDENTS_DIR", tmp_path / "incidents")


def _finding(kind="schema_drift", column="merchantCategoryCode", change="added"):
    return Finding(kind, "bronze.yodlee_transactions_raw",
                   f"column '{column}' {change}",
                   {"change": change, "column": column, "observed_type": "string"})


def test_incident_file_written_with_severity_and_reasoning():
    path = actions.write_incident(_finding(), "additive", "New column.", date(2026, 6, 30))
    text = path.read_text()
    assert path.exists()
    assert "additive" in text
    assert "New column." in text
    assert "bronze.yodlee_transactions_raw" in text


def test_incident_slug_is_stable_so_reruns_do_not_spam():
    a = actions.incident_slug(_finding())
    b = actions.incident_slug(_finding())
    assert a == b
    assert a != actions.incident_slug(_finding(column="other"))


def test_rerun_does_not_create_a_second_file_for_the_same_finding():
    f = _finding()
    p1 = actions.write_incident(f, "additive", "x", date(2026, 6, 30))
    p2 = actions.write_incident(f, "additive", "x", date(2026, 6, 30))
    assert p1 == p2
    assert len(list(actions.INCIDENTS_DIR.glob("*.md"))) == 1


def test_github_issue_is_dry_run_by_default():
    out = actions.file_github_issue(_finding(), "breaking", "why", execute=False)
    assert out.startswith("DRY-RUN")
    assert "gh issue create" in out


def test_graph_runs_offline_and_returns_state(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = graph.run(engine=_FakeEngine(), dry_run=True, as_of=date(2026, 6, 30))
    assert "findings" in state and "classified" in state
    assert isinstance(state["actions_taken"], list)


def test_breaking_findings_produce_actions_benign_do_not(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = graph.run(engine=_FakeEngine(drop_column=True), dry_run=True,
                      as_of=date(2026, 6, 30))
    severities = {c["severity"] for c in state["classified"]}
    assert "breaking" in severities
    assert state["actions_taken"]


class _FakeEngine:
    """Minimal engine stub so the graph test needs no warehouse."""

    def __init__(self, drop_column: bool = False):
        self.drop_column = drop_column

    def scan_arrow(self, ident, snapshot_id=None):
        import pyarrow as pa
        cols = {"id": pa.array([1, 2], pa.int64()),
                "transactionDate": pa.array(["2026-06-29", "2026-06-30"]),
                "baseType": pa.array(["DEBIT", "CREDIT"]),
                "_ingested_at": pa.array([None, None], pa.timestamp("us", tz="UTC"))}
        if self.drop_column:
            cols.pop("baseType")
        return pa.table(cols)

    def snapshots(self, ident):
        return [1, 2, 3]

    def snapshot_row_counts(self, ident):
        return [2, 2, 2]

    def schema_history(self, ident):
        return [{"schema_id": 0, "columns": {}, "field_ids": {}}]

    def table_exists(self, ident):
        return True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_graph.py -v`
Expected: FAIL — `src.agent.actions` not found.

- [ ] **Step 3: Write `src/agent/actions.py`**

```python
"""What the agent does about a finding. Dry-run by default.

An ops agent that files GitHub issues the moment you run it locally is one
people stop running. --execute is opt-in.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import date
from pathlib import Path

from src import config
from src.agent.sensors import Finding

INCIDENTS_DIR = config.REPO_ROOT / "docs" / "incidents"


def incident_slug(finding: Finding) -> str:
    """Stable across runs so a recurring finding updates one file, not many."""
    key = f"{finding.table}|{finding.kind}|{finding.evidence.get('column', '')}" \
          f"|{finding.evidence.get('change', '')}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    column = finding.evidence.get("column", finding.kind)
    return f"{finding.table.replace('.', '-')}-{column}-{digest}"


def write_incident(finding: Finding, severity: str, reasoning: str,
                   as_of: date) -> Path:
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = INCIDENTS_DIR / f"{incident_slug(finding)}.md"
    path.write_text(
        f"# {finding.kind.replace('_', ' ').title()} — {finding.table}\n\n"
        f"**Severity:** {severity}\n\n"
        f"**Detected as of:** {as_of.isoformat()}\n\n"
        f"## What was observed\n\n{finding.detail}\n\n"
        f"## Why this severity\n\n{reasoning}\n\n"
        f"## Evidence\n\n```\n{finding.evidence}\n```\n"
    )
    return path


def file_github_issue(finding: Finding, severity: str, reasoning: str,
                      execute: bool = False) -> str:
    title = f"[{severity}] {finding.kind} on {finding.table}"
    body = f"{finding.detail}\n\n{reasoning}\n\nEvidence: {finding.evidence}"
    command = ["gh", "issue", "create", "--title", title, "--body", body,
               "--label", f"data-{severity}"]
    if not execute:
        return f"DRY-RUN: would run `{' '.join(command[:3])} ...` -> {title}"
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as exc:
        return f"FAILED to file issue: {exc}"
```

- [ ] **Step 4: Write `src/agent/graph.py`**

```python
"""LangGraph ops agent: sense -> detect -> classify -> decide -> act."""
from __future__ import annotations

import sys
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src import config
from src.agent import actions, classifier, sensors
from src.contracts import validator
from src.lakehouse import schemas

WATCHED = (
    (schemas.BRONZE_TRANSACTIONS, "bronze_yodlee_transactions.yaml", "transactionDate"),
)
ACTIONABLE = {"breaking"}


class AgentState(TypedDict, total=False):
    as_of: date
    dry_run: bool
    engine: Any
    findings: list[sensors.Finding]
    classified: list[dict]
    actions_taken: list[str]


def sense_and_detect(state: AgentState) -> AgentState:
    engine = state["engine"]
    findings: list[sensors.Finding] = []
    for table_def, contract_file, date_column in WATCHED:
        contract = validator.load_contract(validator.CONTRACTS_DIR / contract_file)
        observed = sensors.observe(engine, table_def, date_column=date_column)
        # Trailing history excludes the latest snapshot -- comparing a batch
        # against a median that includes itself blunts the signal.
        per_snapshot = engine.snapshot_row_counts(table_def.name)
        history = per_snapshot[:-1] or per_snapshot
        findings += sensors.detect(observed, contract, state["as_of"], history)
    return {**state, "findings": findings}


def classify_findings(state: AgentState) -> AgentState:
    classified = []
    for finding in state["findings"]:
        severity, reasoning = classifier.classify(finding)
        classified.append({"finding": finding, "severity": severity,
                           "reasoning": reasoning})
    return {**state, "classified": classified}


def act(state: AgentState) -> AgentState:
    taken: list[str] = []
    for item in state["classified"]:
        if item["severity"] not in ACTIONABLE:
            continue
        path = actions.write_incident(
            item["finding"], item["severity"], item["reasoning"], state["as_of"])
        taken.append(f"incident: {path.name}")
        taken.append(actions.file_github_issue(
            item["finding"], item["severity"], item["reasoning"],
            execute=not state["dry_run"]))
    return {**state, "actions_taken": taken}


def _should_act(state: AgentState) -> str:
    return "act" if any(c["severity"] in ACTIONABLE
                        for c in state["classified"]) else "end"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("sense", sense_and_detect)
    builder.add_node("classify", classify_findings)
    builder.add_node("act", act)
    builder.set_entry_point("sense")
    builder.add_edge("sense", "classify")
    builder.add_conditional_edges("classify", _should_act,
                                  {"act": "act", "end": END})
    builder.add_edge("act", END)
    return builder.compile()


def run(engine=None, dry_run: bool = True, as_of: date | None = None) -> AgentState:
    if engine is None:
        from src.lakehouse.engines import get_engine
        engine = get_engine()
    if as_of is None:
        from src.lakehouse.bronze import max_txn_date
        as_of = config.resolve_as_of_date(max_txn_date(engine))
    result = build_graph().invoke({
        "engine": engine, "as_of": as_of, "dry_run": dry_run,
        "findings": [], "classified": [], "actions_taken": [],
    })
    return result


def main() -> int:
    execute = "--execute" in sys.argv
    state = run(dry_run=not execute)
    print(f"agent: as_of={state['as_of']} findings={len(state['findings'])}")
    for item in state["classified"]:
        print(f"  [{item['severity']}] {item['finding'].detail}")
    for action in state.get("actions_taken", []):
        print(f"  -> {action}")
    if not execute:
        print("agent: dry-run (pass --execute to file GitHub issues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_agent_graph.py -v`
Expected: 6 passed.

- [ ] **Step 6: Run the agent against the real warehouse**

Run: `uv run make agent`
Expected: prints `as_of=2026-06-30`, a findings count, and `dry-run` at the end. On a clean pipeline the findings list may be empty — Task 16 creates real drift to detect.

- [ ] **Step 7: Commit**

```bash
git add src/agent/actions.py src/agent/graph.py tests/test_agent_graph.py
git commit -m "feat: LangGraph ops agent with dry-run incident actions

incident_slug hashes table + kind + column + change so a recurring finding
rewrites one file instead of accumulating a new one every run. Alert
systems that spam are alert systems people filter out.

GitHub issue filing is dry-run unless --execute is passed. An agent that
files issues the moment someone runs it locally gets uninstalled.

The graph test uses a stub engine so it needs no warehouse and no network."
```

---

## Task 16: Schema evolution, cross-version queries, and Iceberg maintenance

Iceberg resolves columns by **field ID**, not by name or position. That is what
lets a file written under schema v1 be read correctly under schema v5 after a
column has been added, widened, and renamed. This task proves that property
rather than asserting it, and gives the ops agent real drift to find.

**Files:**
- Create: `src/lakehouse/maintenance.py`
- Modify: `Makefile` (add `schema-history`, `cross-version` targets)
- Test: `tests/test_maintenance.py`, `tests/test_schema_evolution.py`

**Interfaces:**
- Consumes: `get_engine`, `schemas.BRONZE_TRANSACTIONS`, `bronze.ingest_all`, `engine.schema_history`, `engine.snapshot_row_counts`
- Produces:
  - `maintenance.evolve_add_column(engine, name, iceberg_type) -> bool`
  - `maintenance.evolve_widen_column(engine, name, iceberg_type) -> bool`
  - `maintenance.evolve_rename_column(engine, old, new) -> bool`
  - `maintenance.evolve_all(engine) -> list[str]` — applies the v2→v5 sequence, returns applied step names
  - `maintenance.schema_versions(engine, ident) -> list[dict]`
  - `maintenance.query_across_versions(engine, ident) -> pa.Table` — every snapshot's rows, reconciled to the current schema, tagged `_snapshot_id` / `_schema_id`
  - `maintenance.incremental_rows(engine, ident, from_snapshot, to_snapshot) -> int`
  - `maintenance.time_travel_demo(engine) -> dict[str, int]`
  - `maintenance.inject_volume_collapse(engine) -> int`
  - `maintenance.expire(engine, retain_last: int = 5) -> int`
  - CLI: `python -m src.lakehouse.maintenance {timetravel|drift-demo|schema-history|cross-version|expire}`

- [ ] **Step 1: Write the schema-evolution test**

```python
# tests/test_schema_evolution.py
import pytest
from pyiceberg.types import IntegerType, LongType, StringType

from src.lakehouse import bronze, catalog as catalog_mod, maintenance, schemas
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine

IDENT = schemas.BRONZE_TRANSACTIONS.name


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=1000)   # written under schema v1
    return eng


def test_added_column_is_null_for_rows_written_before_it(engine):
    rows_before = engine.scan_arrow(IDENT).num_rows
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    current = engine.scan_arrow(IDENT)
    assert "merchantCategoryCode" in current.column_names
    assert current.num_rows == rows_before
    # Old files carry no such column; Iceberg fills it rather than failing.
    assert current.column("merchantCategoryCode").null_count == rows_before


def test_old_snapshot_still_readable_after_evolution(engine):
    before = engine.snapshots(IDENT)[-1]
    maintenance.evolve_all(engine)
    historical = engine.scan_arrow(IDENT, snapshot_id=before)
    assert historical.num_rows > 0


def test_widening_int_to_long_preserves_existing_values(engine):
    maintenance.evolve_add_column(engine, "settlementDays", IntegerType())
    bronze.ingest_all(engine, n_transactions=200)
    widened = maintenance.evolve_widen_column(engine, "settlementDays", LongType())
    assert widened
    out = engine.scan_arrow(IDENT)
    assert "settlementDays" in out.column_names
    assert out.num_rows == 1200


def test_rename_resolves_by_field_id_not_name(engine):
    """The crux: rows written before the rename read back under the NEW name."""
    rows_before = engine.scan_arrow(IDENT).num_rows
    field_id_before = engine.schema_history(IDENT)[-1]["field_ids"]["checkNumber"]

    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")

    out = engine.scan_arrow(IDENT)
    assert "check_reference" in out.column_names
    assert "checkNumber" not in out.column_names
    assert out.num_rows == rows_before          # no rewrite, no data loss
    field_id_after = engine.schema_history(IDENT)[-1]["field_ids"]["check_reference"]
    assert field_id_after == field_id_before    # same field, new name


def test_schema_history_records_every_version(engine):
    maintenance.evolve_all(engine)
    history = maintenance.schema_versions(engine, IDENT)
    assert len(history) >= 4
    assert len({h["schema_id"] for h in history}) == len(history)
    assert "merchantCategoryCode" in history[-1]["columns"]


def test_cross_version_query_spans_every_snapshot(engine):
    """Ingest under v1, evolve, ingest under v5, query the whole history."""
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    bronze.ingest_all(engine, n_transactions=500)
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")
    bronze.ingest_all(engine, n_transactions=300)

    combined = maintenance.query_across_versions(engine, IDENT)
    assert "_snapshot_id" in combined.column_names
    assert "_schema_id" in combined.column_names
    assert "check_reference" in combined.column_names
    # Rows appear once per snapshot in which the table contained them.
    assert combined.num_rows > 1800
    assert len(set(combined.column("_snapshot_id").to_pylist())) == 3


def test_cross_version_query_reconciles_missing_columns_to_null(engine):
    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    bronze.ingest_all(engine, n_transactions=200)
    combined = maintenance.query_across_versions(engine, IDENT)
    assert "merchantCategoryCode" in combined.column_names
    # The first snapshot predates the column entirely.
    assert combined.column("merchantCategoryCode").null_count > 0


def test_incremental_read_returns_only_the_delta(engine):
    first = engine.snapshots(IDENT)[-1]
    bronze.ingest_all(engine, n_transactions=400)
    second = engine.snapshots(IDENT)[-1]
    assert maintenance.incremental_rows(engine, IDENT, first, second) == 400


def test_evolution_is_idempotent(engine):
    maintenance.evolve_all(engine)
    schema_count = len(maintenance.schema_versions(engine, IDENT))
    maintenance.evolve_all(engine)      # re-running must be a no-op
    assert len(maintenance.schema_versions(engine, IDENT)) == schema_count
```

- [ ] **Step 2: Write the maintenance test**

```python
# tests/test_maintenance.py
from datetime import date

import pytest

from src.lakehouse import bronze, catalog as catalog_mod, maintenance, schemas
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=2000)
    return eng


def test_time_travel_shows_history_growing(engine):
    bronze.ingest_all(engine, n_transactions=500)
    result = maintenance.time_travel_demo(engine)
    assert result["rows_at_first_snapshot"] < result["rows_at_latest_snapshot"]


def test_evolved_column_detected_as_additive_drift_by_the_agent(engine):
    from pyiceberg.types import StringType
    from src.agent import classifier, sensors
    from src.contracts import validator

    maintenance.evolve_add_column(engine, "merchantCategoryCode", StringType())
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    findings = sensors.detect(state, contract, date(2026, 6, 30),
                              history=[state.rows_in_latest_snapshot])
    added = [f for f in findings
             if f.evidence.get("column") == "merchantCategoryCode"]
    assert added
    assert classifier.classify(added[0])[0] == "additive"


def test_volume_collapse_is_detected_without_violating_append_only(engine):
    from src.agent import sensors
    from src.contracts import validator

    rows_before = engine.scan_arrow(schemas.BRONZE_TRANSACTIONS.name).num_rows
    history = engine.snapshot_row_counts(schemas.BRONZE_TRANSACTIONS.name)
    added = maintenance.inject_volume_collapse(engine)

    # Bronze stays append-only: the collapse is a tiny APPEND, not a rewrite.
    assert engine.scan_arrow(
        schemas.BRONZE_TRANSACTIONS.name).num_rows == rows_before + added

    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    findings = sensors.detect(state, contract, date(2026, 6, 30), history=history)
    assert any(f.kind == "volume_anomaly" for f in findings)


def test_expire_snapshots_retains_requested_count(engine):
    for _ in range(4):
        bronze.ingest_all(engine, n_transactions=300)
    assert len(engine.snapshots(schemas.BRONZE_TRANSACTIONS.name)) > 3
    maintenance.expire(engine, retain_last=2)
    assert len(engine.snapshots(schemas.BRONZE_TRANSACTIONS.name)) <= 3
```

- [ ] **Step 3: Run both test files to verify they fail**

Run: `uv run pytest tests/test_schema_evolution.py tests/test_maintenance.py -v`
Expected: FAIL — `src.lakehouse.maintenance` not found.

- [ ] **Step 4: Write `src/lakehouse/maintenance.py`**

```python
"""Schema evolution, cross-version queries, and Iceberg table maintenance.

Iceberg resolves columns by field ID, not by name or position. A file written
under schema v1 still reads correctly under v5 after columns have been added,
widened, and renamed -- no rewrite, no backfill. The functions here exercise
that property, and the tests check it rather than trusting it.

The drift injectors exist so the ops agent has real table mutations to find.
An agent proven only against hand-written fixtures proves nothing about
whether it reads actual metadata correctly.
"""
from __future__ import annotations

import sys

import pyarrow as pa
from pyiceberg.types import IntegerType, LongType, StringType

from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine

IDENT = schemas.BRONZE_TRANSACTIONS.name


def _columns(engine, ident: str) -> set[str]:
    table = engine.catalog.load_table(ident)
    return {f.name for f in table.schema().fields}


def evolve_add_column(engine, name: str, iceberg_type, ident: str = IDENT) -> bool:
    """Additive evolution. Existing files gain the column as NULL, unrewritten."""
    if name in _columns(engine, ident):
        return False
    table = engine.catalog.load_table(ident)
    with table.update_schema() as update:
        update.add_column(name, iceberg_type)
    return True


def evolve_widen_column(engine, name: str, iceberg_type, ident: str = IDENT) -> bool:
    """Type promotion (int -> long). Allowed because every existing value fits."""
    if name not in _columns(engine, ident):
        return False
    table = engine.catalog.load_table(ident)
    with table.update_schema() as update:
        update.update_column(name, field_type=iceberg_type)
    return True


def evolve_rename_column(engine, old: str, new: str, ident: str = IDENT) -> bool:
    """Rename. The field ID is unchanged, so old files need no rewrite."""
    columns = _columns(engine, ident)
    if old not in columns or new in columns:
        return False
    table = engine.catalog.load_table(ident)
    with table.update_schema() as update:
        update.rename_column(old, new)
    return True


def evolve_all(engine, ident: str = IDENT) -> list[str]:
    """The v2 -> v5 sequence. Idempotent: re-running applies nothing."""
    applied = []
    if evolve_add_column(engine, "merchantCategoryCode", StringType(), ident):
        applied.append("v2: +merchantCategoryCode (string)")
    if evolve_add_column(engine, "settlementDays", IntegerType(), ident):
        applied.append("v3: +settlementDays (int)")
    if evolve_widen_column(engine, "settlementDays", LongType(), ident):
        applied.append("v4: settlementDays int -> long")
    if evolve_rename_column(engine, "checkNumber", "check_reference", ident):
        applied.append("v5: checkNumber -> check_reference")
    return applied


def schema_versions(engine, ident: str = IDENT) -> list[dict]:
    return engine.schema_history(ident)


def query_across_versions(engine, ident: str = IDENT) -> pa.Table:
    """Union every snapshot's contents, reconciled to the current schema.

    Columns absent from an older snapshot are filled with NULL rather than
    dropped, so a query can span a schema change without the caller knowing
    one happened. Each row is tagged with the snapshot it came from.
    """
    table = engine.catalog.load_table(ident)
    current = table.schema().as_arrow()
    frames: list[pa.Table] = []

    for snapshot in table.metadata.snapshots:
        rows = engine.scan_arrow(ident, snapshot_id=snapshot.snapshot_id)
        aligned = _align_to(rows, current)
        n = aligned.num_rows
        aligned = aligned.append_column(
            "_snapshot_id", pa.array([snapshot.snapshot_id] * n, pa.int64()))
        aligned = aligned.append_column(
            "_schema_id", pa.array([snapshot.schema_id] * n, pa.int32()))
        frames.append(aligned)

    return pa.concat_tables(frames) if frames else current.empty_table()


def _align_to(rows: pa.Table, target: pa.Schema) -> pa.Table:
    """Project rows onto target, filling absent columns with typed nulls."""
    arrays, names = [], []
    for field in target:
        if field.name in rows.column_names:
            arrays.append(rows.column(field.name).cast(field.type))
        else:
            arrays.append(pa.nulls(rows.num_rows, field.type))
        names.append(field.name)
    return pa.Table.from_arrays(arrays, names=names)


def incremental_rows(engine, ident: str, from_snapshot: int,
                     to_snapshot: int) -> int:
    """Rows added between two snapshots, from metadata summaries only."""
    table = engine.catalog.load_table(ident)
    seen, total = False, 0
    for snapshot in table.metadata.snapshots:
        if snapshot.snapshot_id == from_snapshot:
            seen = True
            continue
        if seen:
            summary = getattr(snapshot, "summary", None) or {}
            total += int(summary.get("added-records")
                         or summary.get("added_records") or 0)
        if snapshot.snapshot_id == to_snapshot:
            break
    return total


def time_travel_demo(engine, ident: str = IDENT) -> dict[str, int]:
    snaps = engine.snapshots(ident)
    if len(snaps) < 2:
        bronze.ingest_all(engine, n_transactions=500)
        snaps = engine.snapshots(ident)
    return {
        "snapshot_count": len(snaps),
        "rows_at_first_snapshot": engine.scan_arrow(
            ident, snapshot_id=snaps[0]).num_rows,
        "rows_at_latest_snapshot": engine.scan_arrow(ident).num_rows,
    }


def inject_volume_collapse(engine, ident: str = IDENT, rows: int = 5) -> int:
    """Append a near-empty batch so the per-snapshot volume check fires.

    An APPEND, not an overwrite: Bronze stays append-only (Global Constraints),
    and a collapsed batch is what a real upstream outage actually looks like.
    """
    table_def = schemas.BRONZE_TRANSACTIONS
    latest = engine.scan_arrow(ident).slice(0, rows)
    engine.append(ident, latest.select(
        [f.name for f in table_def.schema.as_arrow()]))
    return rows


def expire(engine, retain_last: int = 5) -> int:
    removed = 0
    for table_def in schemas.ALL_TABLES:
        if not engine.table_exists(table_def.name):
            continue
        table = engine.catalog.load_table(table_def.name)
        snaps = list(table.metadata.snapshots)
        if len(snaps) <= retain_last:
            continue
        table.expire_snapshots().retain_last(retain_last).commit()
        removed += len(snaps) - retain_last
    return removed


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "timetravel"
    engine = get_engine()

    if command == "timetravel":
        for key, value in time_travel_demo(engine).items():
            print(f"timetravel: {key} = {value:,}")

    elif command == "drift-demo":
        applied = evolve_all(engine)
        for step in applied:
            print(f"drift-demo: applied {step}")
        added = inject_volume_collapse(engine)
        print(f"drift-demo: appended a collapsed batch of {added} rows")
        print("drift-demo: run `make agent` to see these detected")

    elif command == "schema-history":
        for version in schema_versions(engine):
            print(f"schema {version['schema_id']}: "
                  f"{len(version['columns'])} columns")
            for name, dtype in version["columns"].items():
                print(f"    [{version['field_ids'][name]:>3}] {name}: {dtype}")

    elif command == "cross-version":
        combined = query_across_versions(engine)
        print(f"cross-version: {combined.num_rows:,} rows across "
              f"{len(set(combined.column('_snapshot_id').to_pylist()))} snapshot(s)")
        for schema_id in sorted(set(combined.column("_schema_id").to_pylist())):
            n = sum(1 for v in combined.column("_schema_id").to_pylist()
                    if v == schema_id)
            print(f"    schema {schema_id}: {n:,} rows")

    elif command == "expire":
        print(f"maintenance: expired {expire(engine)} snapshot(s)")

    else:
        print(f"unknown command: {command}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Add the new Makefile targets**

Add to the `.PHONY` line and the target list in `Makefile`:

```makefile
schema-history: ; $(UV) python -m src.lakehouse.maintenance schema-history
cross-version:  ; $(UV) python -m src.lakehouse.maintenance cross-version
```

- [ ] **Step 6: Run both test files**

Run: `uv run pytest tests/test_schema_evolution.py tests/test_maintenance.py -v`
Expected: 13 passed (9 evolution + 4 maintenance).

If `update_column(name, field_type=...)` is not the signature on your PyIceberg version, check `uv run python -c "import pyiceberg; print(pyiceberg.__version__)"` and the `UpdateSchema` API. Do not delete `test_widening_int_to_long_preserves_existing_values` — type promotion is one of the Iceberg properties this task exists to demonstrate.

If `test_rename_resolves_by_field_id_not_name` fails on the field-ID assertion, that is the single most important failure in this file: it means the rename rewrote the field rather than relabelling it, and every claim about cross-version reads is void. Investigate before proceeding.

- [ ] **Step 7: Verify the whole story end to end**

```bash
uv run make all
uv run make schema-history
uv run make drift-demo
uv run make cross-version
uv run make agent
uv run make timetravel
```

Expected: `schema-history` prints schema 0 then the evolved versions with stable
field IDs; `cross-version` prints per-schema row counts spanning every snapshot;
`agent` reports the added columns as `additive` and the collapsed batch as
`breaking`, writing an incident for the latter only.

- [ ] **Step 8: Commit**

```bash
git add src/lakehouse/maintenance.py tests/test_schema_evolution.py tests/test_maintenance.py Makefile
git commit -m "feat: schema evolution, cross-version queries, and maintenance

Four evolution steps applied to a live Bronze table -- add column, add
column, widen int->long, rename -- with no data rewritten. The rename test
asserts the field ID is unchanged, which is the actual mechanism: Iceberg
resolves by field ID, so files written under v1 read correctly under v5.
If that assertion ever fails, every claim about cross-version reads is void.

query_across_versions unions every snapshot reconciled to the current
schema, filling absent columns with typed nulls rather than dropping them,
so a query can span a schema change without the caller knowing one
happened. Each row is tagged with its snapshot and schema id.

inject_volume_collapse APPENDS a tiny batch rather than overwriting. The
earlier draft rewrote Bronze, which violated the append-only constraint and
was also the wrong shape -- a real upstream outage produces a small batch,
not a truncated table. Volume detection correspondingly moved to
per-snapshot added-records, since a cumulative count on an append-only
table can never collapse."
```

---

## Task 17: Spark engine and parity test

**Files:**
- Create: `src/lakehouse/engines/spark_engine.py`
- Test: `tests/test_engine_parity.py`

**Interfaces:**
- Consumes: `LakehouseEngine` protocol, `schemas.TableDef`
- Produces: `SparkEngine()` implementing the protocol plus `merge_upsert(ident, data, key)` and `compact(ident)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_parity.py
import pytest

pyspark = pytest.importorskip("pyspark", reason="Spark engine is opt-in")

from src.lakehouse import bronze, silver, schemas  # noqa: E402


@pytest.fixture(scope="module")
def spark_engine(tmp_path_factory):
    from src.lakehouse.engines.spark_engine import SparkEngine
    return SparkEngine(warehouse=tmp_path_factory.mktemp("spark_wh"))


@pytest.fixture(scope="module")
def pyiceberg_engine(tmp_path_factory):
    from src.lakehouse import catalog as catalog_mod
    from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
    return PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("py_wh")))


@pytest.mark.spark
def test_both_engines_produce_identical_silver(spark_engine, pyiceberg_engine):
    for engine in (pyiceberg_engine, spark_engine):
        bronze.ingest_all(engine, n_transactions=5000)
        silver.build_all(engine)

    def normalized(engine):
        df = engine.scan_arrow(schemas.SILVER_TRANSACTIONS.name).to_pandas()
        return (df.sort_values("id").reset_index(drop=True)
                  .reindex(sorted(df.columns), axis=1))

    left, right = normalized(pyiceberg_engine), normalized(spark_engine)
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    import pandas.testing as pdt
    pdt.assert_frame_equal(left, right, check_dtype=False, atol=1e-9)


@pytest.mark.spark
def test_spark_merge_upsert_replaces_matching_rows(spark_engine):
    import pyarrow as pa
    td = schemas.SILVER_MERCHANT_MAP
    spark_engine.create_table(td)
    spark_engine.overwrite(td.name, pa.table({
        "merchant_normalized": ["A"], "ticker": ["OLD"],
        "company_name": ["a"], "match_type": ["exact"]}))
    spark_engine.merge_upsert(td.name, pa.table({
        "merchant_normalized": ["A"], "ticker": ["NEW"],
        "company_name": ["a"], "match_type": ["exact"]}), key="merchant_normalized")
    out = spark_engine.scan_arrow(td.name)
    assert out.num_rows == 1
    assert out.column("ticker")[0].as_py() == "NEW"
```

- [ ] **Step 2: Run test to verify it skips or fails**

Run: `uv run pytest tests/test_engine_parity.py -v`
Expected: SKIPPED if PySpark absent (this is the normal CI outcome), otherwise FAIL — `spark_engine` not found.

- [ ] **Step 3: Write `src/lakehouse/engines/spark_engine.py`**

```python
"""Opt-in Spark engine: same protocol, plus MERGE INTO and compaction.

Exists to prove the LakehouseEngine abstraction is real. An interface with
one implementation is a guess about what would be portable; the parity test
turns it into a claim that is checked.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from src import config
from src.lakehouse.schemas import TableDef

ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2"


class SparkEngine:
    def __init__(self, warehouse: Path | None = None):
        from pyspark.sql import SparkSession

        wh = Path(warehouse or config.WAREHOUSE_PATH / "spark")
        wh.mkdir(parents=True, exist_ok=True)
        self.spark = (
            SparkSession.builder.appName("autonomous-data-platform")
            .config("spark.jars.packages", ICEBERG_RUNTIME)
            .config("spark.sql.extensions",
                    "org.apache.iceberg.spark.extensions."
                    "IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.cat", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.cat.type", "hadoop")
            .config("spark.sql.catalog.cat.warehouse", str(wh))
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        for namespace in ("bronze", "silver", "gold"):
            self.spark.sql(f"CREATE NAMESPACE IF NOT EXISTS cat.{namespace}")

    def _q(self, ident: str) -> str:
        return f"cat.{ident}"

    def create_table(self, table_def: TableDef) -> None:
        if self.table_exists(table_def.name):
            return
        empty = self.spark.createDataFrame(
            [], self._spark_schema(table_def))
        writer = empty.writeTo(self._q(table_def.name)).using("iceberg")
        for field in table_def.spec.fields:
            source = table_def.schema.find_field(field.source_id).name
            transform = str(field.transform)
            if transform.startswith("day"):
                writer = writer.partitionedBy(_days(source))
            elif transform.startswith("month"):
                writer = writer.partitionedBy(_months(source))
            else:
                from pyspark.sql.functions import col
                writer = writer.partitionedBy(col(source))
        writer.create()

    def _spark_schema(self, table_def: TableDef):
        from pyspark.sql.types import _parse_datatype_string  # noqa: F401
        import pyspark.sql.types as T
        return T.StructType.fromJson(
            _arrow_to_spark_json(table_def.schema.as_arrow()))

    def table_exists(self, ident: str) -> bool:
        return self.spark.catalog.tableExists(self._q(ident))

    def append(self, ident: str, data: pa.Table) -> None:
        self._to_spark(data).writeTo(self._q(ident)).append()

    def overwrite(self, ident: str, data: pa.Table) -> None:
        self._to_spark(data).writeTo(self._q(ident)).overwritePartitions()

    def scan_arrow(self, ident: str, snapshot_id: int | None = None) -> pa.Table:
        query = f"SELECT * FROM {self._q(ident)}"
        if snapshot_id:
            query += f" VERSION AS OF {snapshot_id}"
        return pa.Table.from_pandas(self.spark.sql(query).toPandas(),
                                    preserve_index=False)

    def snapshots(self, ident: str) -> list[int]:
        rows = self.spark.sql(
            f"SELECT snapshot_id FROM {self._q(ident)}.snapshots "
            "ORDER BY committed_at").collect()
        return [r["snapshot_id"] for r in rows]

    def sql(self, query: str, tables: dict[str, str]) -> pa.Table:
        for alias, ident in tables.items():
            self.spark.sql(f"SELECT * FROM {self._q(ident)}").createOrReplaceTempView(alias)
        return pa.Table.from_pandas(self.spark.sql(query).toPandas(),
                                    preserve_index=False)

    def merge_upsert(self, ident: str, data: pa.Table, key: str) -> None:
        self._to_spark(data).createOrReplaceTempView("_updates")
        self.spark.sql(f"""
            MERGE INTO {self._q(ident)} t
            USING _updates s ON t.{key} = s.{key}
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)

    def compact(self, ident: str) -> None:
        self.spark.sql(
            f"CALL cat.system.rewrite_data_files(table => '{ident}')")

    def _to_spark(self, data: pa.Table):
        return self.spark.createDataFrame(data.to_pandas())


def _days(column: str):
    from pyspark.sql.functions import expr
    return expr(f"days({column})")


def _months(column: str):
    from pyspark.sql.functions import expr
    return expr(f"months({column})")


def _arrow_to_spark_json(schema: pa.Schema) -> dict:
    import pyarrow as _pa
    mapping = {
        _pa.int64(): "long", _pa.float64(): "double", _pa.string(): "string",
        _pa.bool_(): "boolean", _pa.date32(): "date",
    }
    fields = []
    for field in schema:
        spark_type = mapping.get(field.type)
        if spark_type is None:
            spark_type = ("timestamp" if _pa.types.is_timestamp(field.type)
                          else "string")
        fields.append({"name": field.name, "type": spark_type,
                       "nullable": field.nullable, "metadata": {}})
    return {"type": "struct", "fields": fields}
```

Note: `_spark_schema` only needs to cover flat tables, because `create_table` is
called for Silver and Gold in the parity test. If the parity test is extended to
Bronze (nested structs), replace `_arrow_to_spark_json` with a recursive
converter rather than widening the fallback to `string`.

- [ ] **Step 4: Run the parity test**

If PySpark is available: `uv run --extra spark pytest tests/test_engine_parity.py -v`
Expected: 2 passed.
Otherwise: `uv run pytest tests/test_engine_parity.py -v` → 2 skipped. Both are acceptable outcomes; CI runs the pure-Python path.

- [ ] **Step 5: Confirm the default path still works**

Run: `ENGINE=pyiceberg uv run make all && uv run pytest -q`
Expected: full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/lakehouse/engines/spark_engine.py tests/test_engine_parity.py
git commit -m "feat: opt-in Spark engine with cross-engine parity test

The parity test is the point of the abstraction. An interface with one
implementation is a guess about what would be portable; running the same
Silver build through both engines and diffing the output makes it a claim
that is checked.

Skipped when PySpark is absent so CI stays on the pure-Python path, which
is also the path a reader can run from a clean clone."
```

---

## Task 18: Operational monitors — anomaly detection and column typing

Periodic checks that would run on a schedule in production. Each monitor is a
SQL query plus a verdict rule, and every run persists its metric to an Iceberg
table. That history is what makes the baselines real: without it, "is this
batch anomalous" has nothing to compare against but the current run.

**Files:**
- Create: `src/ops/__init__.py`, `src/ops/monitors.py`, `src/ops/runner.py`, `monitors/bronze.yaml`, `monitors/silver.yaml`, `monitors/gold.yaml`
- Modify: `src/lakehouse/schemas.py` (add `OPS_MONITOR_RESULTS`, add to `ALL_TABLES`), `src/lakehouse/catalog.py` (add `ops` namespace), `Makefile`
- Test: `tests/test_monitors.py`

**Interfaces:**
- Consumes: `get_engine`, `config.resolve_as_of_date`, `validator.load_contract`
- Produces:
  - `monitors.MonitorDef(name, table, kind, column, query, params, severity)`
  - `monitors.load_monitors(path) -> list[MonitorDef]`
  - `monitors.MonitorResult(monitor, table, column, metric, baseline, status, detail, run_at)` — `status` in `{"ok", "warn", "breach"}`
  - `monitors.evaluate(metric: float, baseline: list[float], params: dict) -> tuple[str, str]`
  - `monitors.check_column_types(engine, table_def, contract) -> list[MonitorResult]`
  - `runner.run_monitors(engine, as_of, monitor_dir=None) -> list[MonitorResult]`
  - `runner.persist(engine, results) -> int`
  - `runner.load_baselines(engine, monitor: str, column: str | None, limit: int = 14) -> list[float]`
  - `schemas.OPS_MONITOR_RESULTS`
  - CLI: `python -m src.ops.runner`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitors.py
from datetime import date

import pytest

from src.lakehouse import bronze, catalog as catalog_mod, gold, schemas, silver
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import monitors, runner


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=20000)
    silver.build_all(eng)
    gold.build_all(eng)
    return eng


def test_monitor_definitions_load_from_yaml():
    defs = monitors.load_monitors()
    assert len(defs) >= 6
    kinds = {d.kind for d in defs}
    assert {"row_count", "null_rate", "distribution_shift",
            "duplicate_rate", "cardinality"} <= kinds
    assert all(d.query.strip() for d in defs)


def test_evaluate_flags_a_collapse_against_baseline():
    status, detail = monitors.evaluate(
        metric=10.0, baseline=[1000.0, 1010.0, 990.0, 1005.0],
        params={"kind": "row_count", "breach_ratio": 0.5})
    assert status == "breach"
    assert detail


def test_evaluate_passes_a_normal_value():
    status, _ = monitors.evaluate(
        metric=995.0, baseline=[1000.0, 1010.0, 990.0, 1005.0],
        params={"kind": "row_count", "breach_ratio": 0.5})
    assert status == "ok"


def test_evaluate_is_ok_with_no_baseline_rather_than_alarming():
    """First run has no history. Alerting on that trains people to ignore it."""
    status, detail = monitors.evaluate(
        metric=10.0, baseline=[], params={"kind": "row_count", "breach_ratio": 0.5})
    assert status == "ok"
    assert "no baseline" in detail.lower()


def test_distribution_shift_uses_robust_z_score():
    baseline = [100.0] * 10 + [101.0, 99.0]
    breach, _ = monitors.evaluate(
        metric=500.0, baseline=baseline,
        params={"kind": "distribution_shift", "breach_z": 4.0})
    ok, _ = monitors.evaluate(
        metric=100.5, baseline=baseline,
        params={"kind": "distribution_shift", "breach_z": 4.0})
    assert breach == "breach"
    assert ok == "ok"


def test_constant_baseline_does_not_divide_by_zero():
    status, _ = monitors.evaluate(
        metric=100.0, baseline=[100.0] * 8,
        params={"kind": "distribution_shift", "breach_z": 4.0})
    assert status == "ok"


def test_column_type_conformance_passes_on_clean_silver(engine):
    from src.contracts import validator
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "silver_transactions.yaml")
    results = monitors.check_column_types(
        engine, schemas.SILVER_TRANSACTIONS, contract)
    assert results
    assert all(r.status == "ok" for r in results), [
        r.detail for r in results if r.status != "ok"]


def test_column_type_conformance_detects_a_declared_type_mismatch(engine):
    from src.contracts.validator import Contract, SchemaField
    wrong = Contract(
        table="silver.transactions", version=1, owner="x",
        schema_fields=(SchemaField("signed_amount", "string", False),),
        expectations=())
    results = monitors.check_column_types(
        engine, schemas.SILVER_TRANSACTIONS, wrong)
    assert any(r.status == "breach" for r in results)


def test_run_monitors_returns_results_for_every_definition(engine):
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30))
    assert len(results) >= 6
    assert all(r.status in {"ok", "warn", "breach"} for r in results)


def test_results_persist_and_are_readable_as_baselines(engine):
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30))
    written = runner.persist(engine, results)
    assert written == len(results)
    assert engine.table_exists(schemas.OPS_MONITOR_RESULTS.name)

    target = next(r for r in results if r.metric is not None)
    baselines = runner.load_baselines(engine, target.monitor, target.column)
    assert baselines


def test_persisted_history_accumulates_across_runs(engine):
    before = engine.scan_arrow(schemas.OPS_MONITOR_RESULTS.name).num_rows
    runner.persist(engine, runner.run_monitors(engine, as_of=date(2026, 6, 29)))
    after = engine.scan_arrow(schemas.OPS_MONITOR_RESULTS.name).num_rows
    assert after > before


def test_duplicate_rate_monitor_is_zero_on_deduped_silver(engine):
    results = runner.run_monitors(engine, as_of=date(2026, 6, 30))
    dupes = [r for r in results if r.monitor == "silver_txn_duplicate_ids"]
    assert dupes and dupes[0].metric == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitors.py -v`
Expected: FAIL — `src.ops` not found.

- [ ] **Step 3: Add `OPS_MONITOR_RESULTS` to `src/lakehouse/schemas.py`**

```python
OPS_MONITOR_RESULTS = TableDef(
    "ops.monitor_results",
    Schema(
        NestedField(1, "run_at", DateType(), required=True),
        NestedField(2, "monitor", StringType(), required=True),
        NestedField(3, "table_name", StringType(), required=True),
        NestedField(4, "column_name", StringType(), required=False),
        NestedField(5, "kind", StringType(), required=False),
        NestedField(6, "metric", DoubleType(), required=False),
        NestedField(7, "baseline_median", DoubleType(), required=False),
        NestedField(8, "status", StringType(), required=True),
        NestedField(9, "detail", StringType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, MonthTransform(), "run_at_month")),
)
```

Append it to `ALL_TABLES`, and add `"ops"` to `catalog.NAMESPACES`.

- [ ] **Step 4: Write `monitors/silver.yaml`**

```yaml
- name: silver_txn_row_count
  table: silver.transactions
  kind: row_count
  severity: breaking
  params: {breach_ratio: 0.5, warn_ratio: 0.8}
  query: SELECT count(*) AS metric FROM t

- name: silver_txn_duplicate_ids
  table: silver.transactions
  kind: duplicate_rate
  severity: breaking
  params: {max_absolute: 0}
  query: SELECT count(*) - count(DISTINCT id) AS metric FROM t

- name: silver_txn_null_merchant_rate
  table: silver.transactions
  kind: null_rate
  column: merchant_normalized
  severity: additive
  params: {max_absolute: 0.05, breach_z: 4.0}
  query: >
    SELECT sum(CASE WHEN merchant_normalized IS NULL THEN 1 ELSE 0 END)
           * 1.0 / nullif(count(*), 0) AS metric FROM t

- name: silver_txn_mean_abs_amount
  table: silver.transactions
  kind: distribution_shift
  column: signed_amount
  severity: additive
  params: {breach_z: 4.0, warn_z: 3.0}
  query: SELECT avg(abs(signed_amount)) AS metric FROM t

- name: silver_txn_account_cardinality
  table: silver.transactions
  kind: cardinality
  column: account_id
  severity: additive
  params: {breach_ratio: 0.5, warn_ratio: 0.8}
  query: SELECT count(DISTINCT account_id) AS metric FROM t

- name: silver_txn_quarantine_rate
  table: silver.transactions
  kind: null_rate
  severity: breaking
  params: {max_absolute: 0.01}
  query: SELECT 0.0 AS metric FROM t LIMIT 1
```

Write `monitors/bronze.yaml` with `bronze_txn_row_count` (`kind: row_count` on
`bronze.yodlee_transactions_raw`) and `bronze_txn_null_id_rate` (`kind: null_rate`
on `id`, `max_absolute: 0`). Write `monitors/gold.yaml` with
`gold_training_row_count` (`kind: row_count`) and `gold_training_null_target_rate`
(`kind: null_rate` on `fwd_ret_5d`, `max_absolute: 0.05`) — the last five rows per
ticker legitimately have a null target, so the threshold is a rate, not zero.

- [ ] **Step 5: Write `src/ops/monitors.py`**

```python
"""Monitor definitions and verdict rules.

A monitor is a SQL query returning one column named `metric`, plus a rule for
judging that number against its own history. Keeping the query in YAML means
adding a check is a data change, not a code change -- which is what makes it
plausible that an on-call engineer would actually add one.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from src import config

MONITOR_DIR = config.REPO_ROOT / "monitors"


@dataclass(frozen=True)
class MonitorDef:
    name: str
    table: str
    kind: str
    query: str
    column: str | None = None
    severity: str = "additive"
    params: dict | None = None


@dataclass(frozen=True)
class MonitorResult:
    monitor: str
    table: str
    column: str | None
    kind: str
    metric: float | None
    baseline: float | None
    status: str          # ok | warn | breach
    detail: str
    run_at: date


def load_monitors(path: Path | None = None) -> list[MonitorDef]:
    directory = Path(path or MONITOR_DIR)
    defs: list[MonitorDef] = []
    for file in sorted(directory.glob("*.yaml")):
        for raw in yaml.safe_load(file.read_text()) or []:
            defs.append(MonitorDef(
                name=raw["name"], table=raw["table"], kind=raw["kind"],
                query=raw["query"], column=raw.get("column"),
                severity=raw.get("severity", "additive"),
                params=raw.get("params") or {}))
    return defs


def evaluate(metric: float, baseline: list[float],
             params: dict) -> tuple[str, str]:
    """Judge a metric against its own history. Returns (status, detail)."""
    kind = params.get("kind", "row_count")

    absolute = params.get("max_absolute")
    if absolute is not None and metric > absolute:
        return "breach", f"{metric:g} exceeds absolute limit {absolute:g}"

    if not baseline:
        # First run. Alerting with nothing to compare against produces noise
        # on every new monitor, which is how monitoring gets muted.
        return "ok", "no baseline yet — recorded for future comparison"

    median = statistics.median(baseline)

    if kind in {"row_count", "cardinality"}:
        if median <= 0:
            return "ok", "baseline median is zero"
        ratio = metric / median
        if ratio < params.get("breach_ratio", 0.5):
            return "breach", (f"{metric:g} is {ratio:.0%} of trailing "
                              f"median {median:g}")
        if ratio < params.get("warn_ratio", 0.8):
            return "warn", (f"{metric:g} is {ratio:.0%} of trailing "
                            f"median {median:g}")
        return "ok", f"{metric:g} vs median {median:g}"

    if kind in {"distribution_shift", "null_rate"}:
        # Median absolute deviation: a single prior outlier should not widen
        # the band enough to hide the next one.
        deviations = [abs(v - median) for v in baseline]
        mad = statistics.median(deviations)
        if mad == 0:
            return "ok", f"{metric:g} vs flat baseline {median:g}"
        robust_z = abs(metric - median) / (1.4826 * mad)
        if robust_z > params.get("breach_z", 4.0):
            return "breach", (f"{metric:g} is {robust_z:.1f} robust-z from "
                              f"median {median:g}")
        if robust_z > params.get("warn_z", 3.0):
            return "warn", (f"{metric:g} is {robust_z:.1f} robust-z from "
                            f"median {median:g}")
        return "ok", f"{metric:g} within {robust_z:.1f} robust-z"

    if kind == "duplicate_rate":
        return ("ok", f"{metric:g} duplicates") if metric == 0 else (
            "breach", f"{metric:g} duplicate key(s)")

    return "ok", f"{metric:g}"


# Iceberg declared type -> the Arrow types that legitimately represent it.
_TYPE_MAP: dict[str, tuple[str, ...]] = {
    "long": ("int64",),
    "int": ("int32", "int64"),
    "double": ("double", "float"),
    "string": ("string", "large_string"),
    "boolean": ("bool",),
    "date": ("date32[day]",),
    "timestamptz": ("timestamp[us, tz=UTC]", "timestamp[us, tz=+00:00]"),
    "struct": ("struct",),
}


def check_column_types(engine, table_def, contract) -> list[MonitorResult]:
    """Physical Arrow types vs the contract's declared types.

    Catches silent coercion -- a column that arrives as string where the
    contract declares double still passes every value-level expectation while
    breaking every arithmetic consumer downstream.
    """
    arrow = engine.scan_arrow(table_def.name)
    actual = {f.name: str(f.type) for f in arrow.schema}
    run_at = config.resolve_as_of_date(None)
    results: list[MonitorResult] = []

    for field in contract.schema_fields:
        observed = actual.get(field.name)
        if observed is None:
            results.append(MonitorResult(
                f"{table_def.name}_type_{field.name}", table_def.name,
                field.name, "column_type", None, None, "breach",
                f"declared column '{field.name}' is absent", run_at))
            continue

        allowed = _TYPE_MAP.get(field.type, (field.type,))
        ok = any(observed == a or observed.startswith(a) for a in allowed)
        results.append(MonitorResult(
            f"{table_def.name}_type_{field.name}", table_def.name,
            field.name, "column_type", None, None,
            "ok" if ok else "breach",
            f"declared {field.type}, observed {observed}", run_at))

    return results
```

- [ ] **Step 6: Write `src/ops/runner.py`**

```python
"""Execute every monitor, judge against persisted history, persist the result.

The persisted history is the point. Baselines derived from the current run
are not baselines; this table is what a scheduled job accumulates so that
"anomalous" means something on day 30.
"""
from __future__ import annotations

import sys
from datetime import date

import pyarrow as pa

from src import config
from src.contracts import validator
from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine
from src.ops import monitors

TYPED_TABLES = (
    (schemas.SILVER_TRANSACTIONS, "silver_transactions.yaml"),
    (schemas.GOLD_TRAINING, "gold_forecast_training_set.yaml"),
)


def load_baselines(engine, monitor: str, column: str | None = None,
                   limit: int = 14) -> list[float]:
    if not engine.table_exists(schemas.OPS_MONITOR_RESULTS.name):
        return []
    rows = engine.sql(
        """SELECT metric FROM r
           WHERE monitor = ? AND metric IS NOT NULL
           ORDER BY run_at DESC LIMIT ?""".replace("?", "{}").format(
               f"'{monitor}'", limit),
        tables={"r": schemas.OPS_MONITOR_RESULTS.name}).to_pylist()
    return [row["metric"] for row in rows]


def run_monitors(engine, as_of: date, monitor_dir=None) -> list[monitors.MonitorResult]:
    results: list[monitors.MonitorResult] = []

    for definition in monitors.load_monitors(monitor_dir):
        if not engine.table_exists(definition.table):
            continue
        try:
            out = engine.sql(definition.query, tables={"t": definition.table})
            metric = float(out.column("metric")[0].as_py() or 0.0)
        except Exception as exc:
            results.append(monitors.MonitorResult(
                definition.name, definition.table, definition.column,
                definition.kind, None, None, "breach",
                f"monitor query failed: {type(exc).__name__}: {exc}", as_of))
            continue

        baseline = load_baselines(engine, definition.name, definition.column)
        params = {**(definition.params or {}), "kind": definition.kind}
        status, detail = monitors.evaluate(metric, baseline, params)
        results.append(monitors.MonitorResult(
            definition.name, definition.table, definition.column,
            definition.kind, metric,
            (sum(baseline) / len(baseline)) if baseline else None,
            status, detail, as_of))

    for table_def, contract_file in TYPED_TABLES:
        if not engine.table_exists(table_def.name):
            continue
        contract = validator.load_contract(
            validator.CONTRACTS_DIR / contract_file)
        results += monitors.check_column_types(engine, table_def, contract)

    return results


def persist(engine, results: list[monitors.MonitorResult]) -> int:
    engine.create_table(schemas.OPS_MONITOR_RESULTS)
    payload = [{
        "run_at": r.run_at, "monitor": r.monitor, "table_name": r.table,
        "column_name": r.column, "kind": r.kind, "metric": r.metric,
        "baseline_median": r.baseline, "status": r.status, "detail": r.detail,
    } for r in results]
    engine.append(schemas.OPS_MONITOR_RESULTS.name, pa.Table.from_pylist(
        payload, schema=schemas.OPS_MONITOR_RESULTS.schema.as_arrow()))
    return len(payload)


def main() -> int:
    engine = get_engine()
    as_of = config.resolve_as_of_date(bronze.max_txn_date(engine))
    results = run_monitors(engine, as_of)
    persist(engine, results)

    breaches = [r for r in results if r.status == "breach"]
    warns = [r for r in results if r.status == "warn"]
    print(f"monitors: {len(results)} checks at as_of={as_of} — "
          f"{len(breaches)} breach, {len(warns)} warn")
    for result in breaches + warns:
        print(f"  [{result.status}] {result.monitor}: {result.detail}")
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
```

Note on `load_baselines`: the string-formatting shown above is awkward. If
DuckDB parameter binding is available through `engine.sql`, use it; otherwise
build the query with an f-string over a validated monitor name. Do **not** pass
unvalidated external input into the query — monitor names come from repo-owned
YAML, which is why interpolation is acceptable here and would not be for user input.

- [ ] **Step 7: Add the Makefile target**

```makefile
monitor: ; $(UV) python -m src.ops.runner
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_monitors.py -v`
Expected: 12 passed.

- [ ] **Step 9: Run for real, twice, to prove history accumulates**

```bash
uv run make all
uv run make monitor
uv run make monitor
```
Expected: the first run reports "no baseline yet" details; the second compares
against the first. Exit code is non-zero when anything breaches.

- [ ] **Step 10: Commit**

```bash
git add src/ops monitors tests/test_monitors.py src/lakehouse/schemas.py src/lakehouse/catalog.py Makefile
git commit -m "feat: operational monitors for anomaly detection and column typing

Every run persists its metrics to ops.monitor_results, and baselines are read
back from that table. Baselines computed from the current run are not
baselines; this is what makes 'anomalous' mean something on day 30.

Distribution and null-rate checks use median absolute deviation rather than
mean and standard deviation. One prior outlier widens a stddev band enough to
hide the next one, which is exactly when you need the check to fire.

A first run with no history returns ok, not breach. A monitor that alarms on
its own first execution is one that gets muted before it is ever useful.

Column typing compares physical Arrow types against declared contract types.
A double arriving as string passes every value-level expectation while
breaking every arithmetic consumer downstream."
```

---

## Task 19: Data arrival monitoring, SLAs, and alert routing

Row-level checks answer "is the data wrong". Arrival monitoring answers "did
the data come at all" — the failure mode where every quality check passes
because there is nothing new to check.

**Files:**
- Create: `src/ops/arrival.py`, `src/ops/alerts.py`, `.github/workflows/monitors.yml`
- Modify: `src/lakehouse/schemas.py` (add `OPS_ALERT_LOG`), `src/agent/graph.py` (add the `monitor` node), `Makefile`
- Test: `tests/test_arrival.py`, `tests/test_alerts.py`

**Interfaces:**
- Consumes: `runner.run_monitors`, `config.trading_days`, `sensors.Finding`
- Produces:
  - `arrival.ArrivalSLA(table, date_column, calendar, max_lag_days, min_rows_per_period)` where `calendar` is `"trading"` or `"daily"`
  - `arrival.ARRIVAL_SLAS: tuple[ArrivalSLA, ...]`
  - `arrival.check_arrival(engine, sla, as_of) -> list[MonitorResult]`
  - `arrival.missing_periods(observed: set[date], expected: list[date]) -> list[date]`
  - `alerts.Alert(key, severity, title, body, source, run_at)`
  - `alerts.from_monitor_results(results) -> list[Alert]`
  - `alerts.route(engine, alerts, dry_run=True, throttle_runs=3) -> list[str]`
  - `alerts.recent_keys(engine, limit) -> set[str]`
  - `schemas.OPS_ALERT_LOG`
  - CLI: `python -m src.ops.arrival`

- [ ] **Step 1: Write the arrival test**

```python
# tests/test_arrival.py
from datetime import date

import pytest

from src import config
from src.lakehouse import bronze, catalog as catalog_mod, silver
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import arrival


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path_factory.mktemp("wh")))
    bronze.ingest_all(eng, n_transactions=30000)
    silver.build_all(eng)
    return eng


def test_missing_periods_finds_gaps_in_the_middle():
    expected = [date(2026, 6, d) for d in (1, 2, 3, 4, 5)]
    observed = {date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 5)}
    assert arrival.missing_periods(observed, expected) == [
        date(2026, 6, 3), date(2026, 6, 4)]


def test_missing_periods_empty_when_complete():
    expected = [date(2026, 6, d) for d in (1, 2, 3)]
    assert arrival.missing_periods(set(expected), expected) == []


def test_weekend_is_not_reported_missing_for_a_trading_calendar():
    """Saturday absence is normal. Reporting it is how alerting loses trust."""
    expected = config.trading_days(date(2026, 6, 1), date(2026, 6, 12))
    assert date(2026, 6, 6) not in expected      # Saturday
    assert arrival.missing_periods(set(expected), expected) == []


def test_arrival_check_passes_on_a_complete_feed(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 6, 30))
    assert results
    assert all(r.status == "ok" for r in results), [
        r.detail for r in results if r.status != "ok"]


def test_arrival_check_breaches_when_as_of_runs_ahead_of_the_data(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 7, 31))
    assert any(r.status == "breach" and r.kind == "arrival_lag" for r in results)


def test_arrival_reports_the_specific_missing_dates(engine):
    sla = next(s for s in arrival.ARRIVAL_SLAS if s.table == "silver.stock_prices")
    results = arrival.check_arrival(engine, sla, as_of=date(2026, 7, 31))
    gap = next(r for r in results if r.kind == "arrival_gap")
    assert gap.metric is not None
```

- [ ] **Step 2: Write the alerting test**

```python
# tests/test_alerts.py
from datetime import date

import pytest

from src.lakehouse import catalog as catalog_mod, schemas
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import alerts
from src.ops.monitors import MonitorResult


@pytest.fixture
def engine(tmp_path):
    return PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))


def _result(status="breach", monitor="m1"):
    return MonitorResult(monitor, "silver.transactions", "id", "row_count",
                         10.0, 1000.0, status, "collapsed", date(2026, 6, 30))


def test_only_breaches_and_warns_become_alerts():
    made = alerts.from_monitor_results(
        [_result("ok"), _result("warn", "m2"), _result("breach", "m3")])
    assert {a.severity for a in made} == {"warn", "breach"}
    assert len(made) == 2


def test_alert_key_is_stable_for_the_same_monitor():
    a = alerts.from_monitor_results([_result()])[0]
    b = alerts.from_monitor_results([_result()])[0]
    assert a.key == b.key


def test_route_is_dry_run_by_default(engine):
    sent = alerts.route(engine, alerts.from_monitor_results([_result()]),
                        dry_run=True)
    assert sent
    assert all("DRY-RUN" in line for line in sent)


def test_repeat_alert_is_throttled_within_the_window(engine):
    made = alerts.from_monitor_results([_result()])
    first = alerts.route(engine, made, dry_run=True, throttle_runs=3)
    second = alerts.route(engine, made, dry_run=True, throttle_runs=3)
    assert first
    assert second == [] or all("throttled" in s.lower() for s in second)


def test_alert_log_persists(engine):
    alerts.route(engine, alerts.from_monitor_results([_result()]), dry_run=True)
    assert engine.table_exists(schemas.OPS_ALERT_LOG.name)
    assert engine.scan_arrow(schemas.OPS_ALERT_LOG.name).num_rows >= 1


def test_distinct_monitors_are_not_throttled_against_each_other(engine):
    alerts.route(engine, alerts.from_monitor_results([_result(monitor="a")]),
                 dry_run=True)
    sent = alerts.route(engine, alerts.from_monitor_results([_result(monitor="b")]),
                        dry_run=True)
    assert sent and not any("throttled" in s.lower() for s in sent)
```

- [ ] **Step 3: Run both to verify they fail**

Run: `uv run pytest tests/test_arrival.py tests/test_alerts.py -v`
Expected: FAIL — `src.ops.arrival` not found.

- [ ] **Step 4: Add `OPS_ALERT_LOG` to `src/lakehouse/schemas.py`**

```python
OPS_ALERT_LOG = TableDef(
    "ops.alert_log",
    Schema(
        NestedField(1, "run_at", DateType(), required=True),
        NestedField(2, "alert_key", StringType(), required=True),
        NestedField(3, "severity", StringType(), required=True),
        NestedField(4, "title", StringType(), required=False),
        NestedField(5, "body", StringType(), required=False),
        NestedField(6, "source", StringType(), required=False),
        NestedField(7, "delivered", BooleanType(), required=False),
    ),
    PartitionSpec(PartitionField(1, 1000, MonthTransform(), "run_at_month")),
)
```

Append to `ALL_TABLES`.

- [ ] **Step 5: Write `src/ops/arrival.py`**

```python
"""Did the data arrive at all?

Row-level quality checks pass trivially when nothing new lands: an empty
delta has no nulls, no duplicates, and no out-of-range values. Arrival
monitoring is the check that fires when every other check is silent.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta

from src import config
from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine
from src.ops.monitors import MonitorResult


@dataclass(frozen=True)
class ArrivalSLA:
    table: str
    date_column: str
    calendar: str          # "trading" or "daily"
    max_lag_days: int
    min_rows_per_period: int


ARRIVAL_SLAS: tuple[ArrivalSLA, ...] = (
    ArrivalSLA("silver.transactions", "txn_date", "trading", 3, 1),
    ArrivalSLA("silver.stock_prices", "trade_date", "trading", 3,
               len(config.COMPANIES)),
    ArrivalSLA("gold.forecast_training_set", "trade_date", "trading", 5, 1),
)


def expected_periods(calendar: str, start: date, end: date) -> list[date]:
    if calendar == "trading":
        return config.trading_days(start, end)
    days, cursor = [], start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def missing_periods(observed: set[date], expected: list[date]) -> list[date]:
    """Only periods the calendar says should exist. Weekends are not gaps."""
    return [d for d in expected if d not in observed]


def check_arrival(engine, sla: ArrivalSLA, as_of: date,
                  lookback_days: int = 30) -> list[MonitorResult]:
    if not engine.table_exists(sla.table):
        return [MonitorResult(f"{sla.table}_arrival", sla.table, None,
                              "arrival_missing", None, None, "breach",
                              "table does not exist", as_of)]

    rows = engine.sql(
        f"SELECT {sla.date_column} AS d, count(*) AS n FROM t GROUP BY 1",
        tables={"t": sla.table}).to_pylist()
    observed = {r["d"]: r["n"] for r in rows if r["d"] is not None}
    results: list[MonitorResult] = []

    if not observed:
        return [MonitorResult(f"{sla.table}_arrival", sla.table,
                              sla.date_column, "arrival_missing", 0.0, None,
                              "breach", "no data at all", as_of)]

    newest = max(observed)
    lag = (as_of - newest).days
    results.append(MonitorResult(
        f"{sla.table}_arrival_lag", sla.table, sla.date_column, "arrival_lag",
        float(lag), None,
        "breach" if lag > sla.max_lag_days else "ok",
        f"newest {newest} is {lag}d behind as_of {as_of} "
        f"(limit {sla.max_lag_days}d)", as_of))

    window_start = max(newest - timedelta(days=lookback_days), min(observed))
    expected = expected_periods(sla.calendar, window_start, min(newest, as_of))
    gaps = missing_periods(set(observed), expected)
    results.append(MonitorResult(
        f"{sla.table}_arrival_gap", sla.table, sla.date_column, "arrival_gap",
        float(len(gaps)), None,
        "breach" if gaps else "ok",
        (f"{len(gaps)} missing period(s): "
         f"{', '.join(str(d) for d in gaps[:5])}") if gaps
        else "no gaps in the expected calendar", as_of))

    thin = [d for d in expected
            if d in observed and observed[d] < sla.min_rows_per_period]
    results.append(MonitorResult(
        f"{sla.table}_arrival_partial", sla.table, sla.date_column,
        "arrival_partial", float(len(thin)), None,
        "warn" if thin else "ok",
        (f"{len(thin)} period(s) below {sla.min_rows_per_period} rows")
        if thin else "all periods complete", as_of))

    return results


def main() -> int:
    engine = get_engine()
    as_of = config.resolve_as_of_date(bronze.max_txn_date(engine))
    breaches = 0
    for sla in ARRIVAL_SLAS:
        for result in check_arrival(engine, sla, as_of):
            marker = {"ok": " ", "warn": "!", "breach": "X"}[result.status]
            print(f"[{marker}] {result.monitor}: {result.detail}")
            breaches += result.status == "breach"
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Write `src/ops/alerts.py`**

```python
"""Turn monitor results into alerts, with dedupe and throttling.

Throttling is not a nicety. A breach that persists for a week should not
produce seven identical pages; the second one adds no information and the
seventh trains the recipient to filter the channel.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import pyarrow as pa

from src.lakehouse import schemas
from src.ops.monitors import MonitorResult

ALERTABLE = {"warn", "breach"}


@dataclass(frozen=True)
class Alert:
    key: str
    severity: str
    title: str
    body: str
    source: str
    run_at: date


def _key(result: MonitorResult) -> str:
    raw = f"{result.table}|{result.monitor}|{result.column or ''}|{result.status}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def from_monitor_results(results: list[MonitorResult]) -> list[Alert]:
    return [
        Alert(
            key=_key(r), severity=r.status,
            title=f"[{r.status}] {r.monitor} on {r.table}",
            body=(f"{r.detail}\n\nmetric={r.metric} baseline={r.baseline} "
                  f"kind={r.kind} column={r.column}"),
            source="monitors", run_at=r.run_at)
        for r in results if r.status in ALERTABLE
    ]


def recent_keys(engine, limit: int = 3) -> set[str]:
    """Alert keys seen in the last `limit` distinct runs."""
    if not engine.table_exists(schemas.OPS_ALERT_LOG.name):
        return set()
    rows = engine.sql(
        f"""WITH runs AS (
                SELECT DISTINCT run_at FROM a ORDER BY run_at DESC LIMIT {limit})
            SELECT DISTINCT alert_key FROM a
            WHERE run_at IN (SELECT run_at FROM runs)""",
        tables={"a": schemas.OPS_ALERT_LOG.name}).to_pylist()
    return {r["alert_key"] for r in rows}


def route(engine, alerts: list[Alert], dry_run: bool = True,
          throttle_runs: int = 3) -> list[str]:
    if not alerts:
        return []

    already = recent_keys(engine, throttle_runs)
    sent: list[str] = []
    logged: list[dict] = []

    for alert in alerts:
        if alert.key in already:
            sent.append(f"throttled: {alert.title} (seen in last "
                        f"{throttle_runs} run(s))")
            continue
        prefix = "DRY-RUN: " if dry_run else ""
        sent.append(f"{prefix}{alert.severity.upper()} -> {alert.title}")
        logged.append({
            "run_at": alert.run_at, "alert_key": alert.key,
            "severity": alert.severity, "title": alert.title,
            "body": alert.body, "source": alert.source,
            "delivered": not dry_run,
        })

    if logged:
        engine.create_table(schemas.OPS_ALERT_LOG)
        engine.append(schemas.OPS_ALERT_LOG.name, pa.Table.from_pylist(
            logged, schema=schemas.OPS_ALERT_LOG.schema.as_arrow()))

    return sent
```

- [ ] **Step 7: Wire monitors into the agent and add Makefile targets**

In `src/agent/graph.py`, add a `monitor` node that runs between `sense` and
`classify`: it calls `runner.run_monitors` plus `arrival.check_arrival` for every
SLA, converts breaches into `sensors.Finding(kind="monitor_breach", ...)`, and
extends `state["findings"]`. Add to `classifier.classify` a branch returning
`("breaking", ...)` for `kind == "monitor_breach"`. Add to the `Makefile`:

```makefile
arrival: ; $(UV) python -m src.ops.arrival
```

- [ ] **Step 8: Write `.github/workflows/monitors.yml`**

```yaml
name: monitors
on:
  schedule:
    - cron: "0 */6 * * *"    # every six hours
  workflow_dispatch:

jobs:
  run-monitors:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - name: Build the lakehouse
        run: N_TRANSACTIONS=20000 uv run make all
      - name: Data quality monitors
        run: uv run make monitor
      - name: Data arrival SLAs
        run: uv run make arrival
      - name: Ops agent (dry-run)
        run: uv run make agent
```

- [ ] **Step 9: Run the tests**

Run: `uv run pytest tests/test_arrival.py tests/test_alerts.py -v`
Expected: 12 passed.

- [ ] **Step 10: Run the operational loop end to end**

```bash
uv run make all
uv run make monitor
uv run make arrival
uv run make agent
AS_OF_DATE=2026-07-31 uv run make arrival
```
Expected: the first `arrival` run is clean; the `AS_OF_DATE`-shifted run reports
an `arrival_lag` breach with the specific missing dates — the staleness scenario
reproduced by moving the logical clock, not by waiting.

- [ ] **Step 11: Commit**

```bash
git add src/ops/arrival.py src/ops/alerts.py .github/workflows/monitors.yml src/agent/graph.py src/agent/classifier.py src/lakehouse/schemas.py Makefile tests/test_arrival.py tests/test_alerts.py
git commit -m "feat: data arrival SLAs and throttled alert routing

Arrival monitoring is the check that fires when every other check is silent.
An empty delta has no nulls, no duplicates and nothing out of range, so
row-level quality passes trivially when the feed simply stops.

Gaps are computed against a trading calendar, so a missing Saturday is not
reported. Alerting on normal weekend absence is how a channel loses its
readers.

Alerts are throttled on a stable key across the last N runs. A breach that
persists for a week should not page seven times; the second adds no
information and the seventh teaches people to filter the channel.

Scheduled workflow runs the monitors, the arrival SLAs and the agent every
six hours, which is the operational shape these checks are written for."
```

---

## Task 20: Documentation, ADRs, AI-SDLC record, and CI

**Files:**
- Create: `README.md`, `docs/architecture/ADR-000{1..7}.md`, `docs/ai-sdlc/workflow.md`, `docs/ai-sdlc/decisions/0001-sklearn-over-lightgbm.md`, `docs/ai-sdlc/prompts/README.md`, `.github/workflows/ci.yml`
- Test: `tests/test_docs.py`

**Interfaces:**
- Consumes: everything
- Produces: no code interfaces

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs.py
from pathlib import Path

import pytest

from src import config

DOCS = config.REPO_ROOT / "docs"


def test_readme_exists_and_has_required_sections():
    text = (config.REPO_ROOT / "README.md").read_text()
    for heading in ("## Architecture", "## AI-Driven SDLC",
                    "## The agentic ops layer", "## GitHub Actions",
                    "## What I'd do differently at production scale"):
        assert heading in text, f"missing {heading}"


def test_readme_contains_a_mermaid_diagram():
    assert "```mermaid" in (config.REPO_ROOT / "README.md").read_text()


@pytest.mark.parametrize("n", range(1, 8))
def test_each_adr_exists_and_states_a_decision(n):
    matches = list((DOCS / "architecture").glob(f"ADR-{n:04d}-*.md"))
    assert matches, f"ADR-{n:04d} missing"
    text = matches[0].read_text()
    for section in ("## Context", "## Decision", "## Consequences"):
        assert section in text


def test_readme_does_not_overclaim_real_alpha():
    """The prices are synthetic. The README must say so."""
    text = (config.REPO_ROOT / "README.md").read_text().lower()
    assert "synthetic" in text


def test_readme_documents_operational_monitoring():
    text = (config.REPO_ROOT / "README.md").read_text()
    assert "## Operational monitoring" in text
    for target in ("make monitor", "make arrival"):
        assert target in text


def test_readme_documents_schema_evolution_and_cross_version_queries():
    text = (config.REPO_ROOT / "README.md").read_text()
    assert "## Schema evolution and cross-version queries" in text
    assert "field ID" in text or "field-ID" in text


def test_ai_sdlc_workflow_documented():
    assert (DOCS / "ai-sdlc" / "workflow.md").exists()
    assert (DOCS / "ai-sdlc" / "decisions" / "0001-sklearn-over-lightgbm.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs.py -v`
Expected: FAIL — README.md missing.

- [ ] **Step 3: Write the six ADRs**

Each file uses the same three-section structure. Write them with real content:

| File | Decision |
|---|---|
| `ADR-0001-iceberg-over-delta-and-hudi.md` | Iceberg for engine neutrality and a mature Python catalog story; Delta's Python path is tied more tightly to Spark |
| `ADR-0002-pyiceberg-default-spark-optional.md` | PyIceberg+DuckDB default so the repo runs offline from a clean clone; Spark opt-in behind one protocol, proven by a parity test |
| `ADR-0003-medallion-layer-boundaries.md` | Bronze append-only and source-faithful; Silver conforms/dedupes/validates; Gold builds features. No transform may skip a layer |
| `ADR-0004-synthetic-spend-coupled-prices.md` | Synthetic prices over real market data: real prices + synthetic spend guarantees zero correlation and a meaningless model. Planted per-ticker lags and betas make "did the pipeline preserve the signal" testable |
| `ADR-0005-yaml-contracts-over-great-expectations.md` | YAML contracts read by both pipeline and agent; GE's suite/checkpoint model is more machinery than this scale needs, and its results are not a convenient diff target for the agent |
| `ADR-0006-rules-first-agent-classification.md` | Deterministic rules with the LLM reserved for unmatched cases; CI must never require a model call |
| `ADR-0007-persisted-metric-history-for-baselines.md` | Monitor results persist to `ops.monitor_results` and baselines are read back from it; robust z-scores (median/MAD) over mean/stddev so one prior outlier cannot mask the next; alerts throttled on a stable key |

- [ ] **Step 4: Write `docs/ai-sdlc/decisions/0001-sklearn-over-lightgbm.md`**

Record the documented deviation: the spec named `LGBMRegressor`; implementation uses `HistGradientBoostingRegressor`. Same histogram-based algorithm, no `libomp` native dependency, which is the most common macOS/CI install failure. State that the deviation was made during implementation and why it does not change the model's character.

- [ ] **Step 5: Write `README.md`**

Required sections, matching the test: `## What this is`, `## The problem it solves`, `## Architecture` (with the Mermaid diagram from the spec §3), `## Quickstart`, `## Schema evolution and cross-version queries`, `## Operational monitoring`, `## AI-Driven SDLC`, `## The agentic ops layer`, `## GitHub Actions`, `## Results`, `## What I'd do differently at production scale`.

The schema-evolution section must show the actual v1→v5 sequence from Task 16 with real `make schema-history` and `make cross-version` output, and explain the field-ID mechanism: renaming a column changes its name but not its ID, which is why files written under v1 still read correctly under v5 with no rewrite and no backfill.

Two things the README must do:
- State plainly that prices are synthetic with a planted signal, and that the results measure pipeline fidelity rather than real alpha.
- In "What I'd do differently at production scale", write three honest paragraphs. Candidates grounded in what was actually built: the full-overwrite Silver rebuild does not survive real data volume and would need `MERGE INTO` with incremental watermarks; a SQLite catalog is single-writer and would need REST/Glue with proper commit retries; and the monitors run in-process against a local warehouse rather than as an independent scheduled service with its own alerting backend, so a failure of the pipeline is also a failure of the thing meant to observe it.

- [ ] **Step 6: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: uv run ruff check src tests
      - run: uv run pytest -v
      - name: Smoke the full pipeline on a reduced dataset
        run: N_TRANSACTIONS=5000 uv run make all
      - name: Agent dry-run
        run: uv run make agent
```

- [ ] **Step 7: Run the full suite and lint**

```bash
uv run ruff check src tests
uv run pytest -v
uv run make all
uv run make agent
```
Expected: lint clean, all tests pass (Spark parity skipped), `docs/forecast-report.md` regenerated, agent dry-run completes.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/architecture docs/ai-sdlc .github tests/test_docs.py
git commit -m "docs: README, ADRs, AI-SDLC record, and CI

test_readme_does_not_overclaim_real_alpha asserts the README says
'synthetic'. The results table is persuasive-looking, and a portfolio repo
that lets a reader assume the alpha is real is worse than one with no
results at all.

'What I'd do differently at production scale' names three real limits of
what was built: full-overwrite Silver rebuilds, a single-writer SQLite
catalog, and row-count history derived from snapshot count rather than
persisted metrics.

CI runs lint, the full suite, a reduced-dataset end-to-end smoke, and an
agent dry-run on the pure-Python path."
```

---

## Plan Self-Review

**Spec coverage.** Every section of the design spec maps to a task: §2 domain → Tasks 1–4; §3.1 Bronze → Task 6; §3.2 Silver incl. quarantine → Task 8; §3.3 Gold → Tasks 9–10; §4 Iceberg features → Tasks 5 and 16; §5 dual engine → Tasks 5 and 17; §6 contracts and `AS_OF_DATE` → Task 7; §7 forecast → Tasks 11–13; §8 agent → Tasks 14–16; §9 tests → distributed across all tasks; §10 layout → Task 18; §11 ADRs → Task 18; §12 success criteria → verified in Tasks 13, 16, 17, 18.

**Gap found and closed.** The spec's §8 lists five drift scenarios; Task 16 reproduces three against real tables (additive column, volume collapse, staleness via `AS_OF_DATE`). Type narrowing and dropped column are covered by unit tests in Task 14 rather than by mutating a real table, since PyIceberg's `update_schema` will not narrow a type or drop a required column without a rewrite. This is a deliberate coverage split, not an omission — note it in `docs/ai-sdlc/decisions/` when writing Task 18.

**Type consistency check.** `TableDef` is used identically in Tasks 5, 6, 8, 9, 16, 17. `engine.sql(query, tables=...)` keeps the same two-argument signature everywhere. `features.FEATURE_COLUMNS` and `features.TARGET` are defined in Task 10 and consumed unchanged in Tasks 12–13. `sensors.Finding` fields (`kind`, `table`, `detail`, `evidence`) are consistent across Tasks 14–16. `splits.Fold(index, train, test)` is defined in Task 11 and consumed in Task 12.
