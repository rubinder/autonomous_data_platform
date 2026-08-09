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
N_TRANSACTIONS = int(os.environ.get("N_TRANSACTIONS", "250000"))
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

# NYSE holidays covering 2024-01-01 .. 2026-12-31 -- deliberately the whole of
# 2026, not just START_DATE..END_DATE. The calendar is consulted past the end of
# the data: `AS_OF_DATE` can be advanced beyond END_DATE to reproduce a stale
# feed, and `src.ops.arrival` then expects trading days all the way up to
# `as_of`. A calendar that stops at END_DATE makes every holiday after it look
# like a missing trading day -- at the documented AS_OF_DATE=2026-07-31 that
# reported 2026-07-03 (Independence Day observed, since 4 July 2026 is a
# Saturday) as a data gap. Hardcoded to avoid a calendar dependency that would
# drift with library upgrades.
US_MARKET_HOLIDAYS: frozenset[date] = frozenset({
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19),
    # Past END_DATE (2026-06-30), reachable only via an advanced AS_OF_DATE:
    date(2026, 7, 3),    # Independence Day observed (4 July 2026 is a Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas Day
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
