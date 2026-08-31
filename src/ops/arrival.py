"""Did the data arrive at all?

Row-level quality checks pass trivially when nothing new lands: an empty
delta has no nulls, no duplicates, and no out-of-range values. Arrival
monitoring is the check that fires when every other check is silent.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta

from src import config, feeds
from src.lakehouse import bronze
from src.lakehouse.engines import get_engine
from src.ops.monitors import MonitorResult


@dataclass(frozen=True)
class ArrivalSLA:
    table: str
    date_column: str
    calendar: str          # "trading" or "daily"
    max_lag_days: int
    # Rows expected per period, for tables with a *structural* per-period row
    # count. `None` means "no meaningful floor", and the partial-arrival check
    # is skipped rather than reported.
    #
    # This must never be 1. `observed` is built from a `count(*)` GROUP BY, so
    # a period that appears at all appears with at least one row and
    # `observed[d] < 1` is unsatisfiable by construction -- the same species of
    # defect as the `SELECT 0.0` quarantine monitor: a check that cannot fire,
    # displayed as a passing check on every run. Two of the three SLAs used to
    # carry exactly that.
    min_rows_per_period: int | None


def _slas_from_feeds(directory=None) -> tuple[ArrivalSLA, ...]:
    """Arrival SLAs, declared per feed in `feeds/*.yaml`.

    These used to be a hardcoded tuple here, which meant a new feed's SLA was
    a Python edit in a module that otherwise knows nothing about any specific
    feed. The comments that justified each floor moved into the feed files
    alongside the rest of that feed's operational config.
    """
    return tuple(
        ArrivalSLA(a.table, a.column, a.calendar, a.max_lag_periods,
                   a.min_rows_per_period)
        for feed in feeds.load_feeds(directory) for a in feed.arrival)


ARRIVAL_SLAS: tuple[ArrivalSLA, ...] = _slas_from_feeds()


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

    # The gap window runs from `lookback_days` before the newest observed
    # data through `as_of` itself -- not just up to `newest`. Stopping the
    # window at `newest` can never surface a gap: every day up to the newest
    # observed row is, by definition, already covered by that same
    # observation. The days that matter are the ones between the last
    # arrival and the pipeline's logical "now" -- that gap is exactly what a
    # stalled feed looks like, and it is invisible unless the window is
    # allowed to run past `newest` up to `as_of`.
    window_start = max(newest - timedelta(days=lookback_days), min(observed))
    expected = expected_periods(sla.calendar, window_start, max(as_of, newest))
    gaps = missing_periods(set(observed), expected)
    results.append(MonitorResult(
        f"{sla.table}_arrival_gap", sla.table, sla.date_column, "arrival_gap",
        float(len(gaps)), None,
        "breach" if gaps else "ok",
        (f"{len(gaps)} missing period(s): "
         f"{', '.join(str(d) for d in gaps[:5])}") if gaps
        else "no gaps in the expected calendar", as_of))

    if sla.min_rows_per_period is not None:
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
