"""Execute every monitor, judge against persisted history, persist the result.

The persisted history is the point. Baselines derived from the current run
are not baselines; this table is what a scheduled job accumulates so that
"anomalous" means something on day 30.
"""
from __future__ import annotations

import re
import statistics
import sys
from datetime import date

import pyarrow as pa

from src import config
from src.contracts import validator
from src.lakehouse import bronze, schemas
from src.lakehouse.engines import get_engine
from src.ops import alerts, monitors

TYPED_TABLES = (
    (schemas.SILVER_TRANSACTIONS, "silver_transactions.yaml"),
    (schemas.GOLD_TRAINING, "gold_forecast_training_set.yaml"),
)

# Monitor names are repo-owned (they come from monitors/*.yaml, not from
# any external or user-supplied input), which is what makes f-string
# interpolation of `monitor` below acceptable -- the engine's `sql()`
# protocol offers no parameter-binding hook, so the alternative is string
# building either way. The pattern is still enforced defensively: anything
# that is not a plain identifier is rejected before it ever reaches SQL.
_MONITOR_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.]*$")


def load_baselines(engine, monitor: str, column: str | None = None,
                    limit: int = 14) -> list[float]:
    """Trailing metric history for `monitor`, most-recent-first order ignored.

    `column` is accepted for interface symmetry with the rest of the module
    (a monitor result always names the column it measured) but is not needed
    to disambiguate the query: `monitor` names are already unique per
    definition -- `check_column_types` bakes the column into the name itself
    (`f"{table}_type_{field}"`) -- so filtering on `monitor` alone is exact.
    """
    if not engine.table_exists(schemas.OPS_MONITOR_RESULTS.name):
        return []
    if not _MONITOR_NAME_RE.match(monitor):
        raise ValueError(f"unsafe monitor name: {monitor!r}")
    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"limit must be positive: {limit!r}")

    query = (
        "SELECT metric FROM r "
        f"WHERE monitor = '{monitor}' AND metric IS NOT NULL "
        f"ORDER BY run_at DESC LIMIT {limit}"
    )
    rows = engine.sql(query, tables={"r": schemas.OPS_MONITOR_RESULTS.name}).to_pylist()
    return [row["metric"] for row in rows]


def run_monitors(engine, as_of: date, monitor_dir=None) -> list[monitors.MonitorResult]:
    results: list[monitors.MonitorResult] = []

    for definition in monitors.load_monitors(monitor_dir):
        required_tables = (definition.table, *(definition.tables or {}).values())
        if not all(engine.table_exists(t) for t in required_tables):
            continue
        try:
            if definition.source == "snapshot_added":
                metric = monitors.latest_incremental_added_rows(engine, definition.table)
            else:
                query_tables = {"t": definition.table, **(definition.tables or {})}
                out = engine.sql(definition.query, tables=query_tables)
                metric = float(out.column("metric")[0].as_py() or 0.0)
            baseline = load_baselines(engine, definition.name, definition.column)
            params = {**(definition.params or {}), "kind": definition.kind}
            status, detail = monitors.evaluate(metric, baseline, params)
        except Exception as exc:  # noqa: BLE001 -- a malformed monitor query, or
            # an unrecognised `kind` in its YAML, must not crash the whole run;
            # report it as a breach instead so a broken check is visible rather
            # than silently skipped -- or, worse, silently passing.
            results.append(monitors.MonitorResult(
                definition.name, definition.table, definition.column,
                definition.kind, None, None, "breach",
                f"monitor failed: {type(exc).__name__}: {exc}", as_of))
            continue

        results.append(monitors.MonitorResult(
            definition.name, definition.table, definition.column,
            definition.kind, metric,
            statistics.median(baseline) if baseline else None,
            status, detail, as_of))

    for table_def, contract_file in TYPED_TABLES:
        if not engine.table_exists(table_def.name):
            continue
        contract = validator.load_contract(
            validator.CONTRACTS_DIR / contract_file)
        results += monitors.check_column_types(engine, table_def, contract, as_of)

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


def route_alerts(engine, results: list[monitors.MonitorResult],
                 dry_run: bool = True) -> list[str]:
    """Turn this run's warn/breach results into throttled alerts.

    This is the production caller of `src.ops.alerts`. Without it the alert
    module had no production caller at all: `ops.alert_log` was declared in
    `schemas.ALL_TABLES` and created by nothing, while the docs described
    throttled alerting as live behaviour. Dry-run by default, exactly like the
    agent's `act` node -- the delivery side effect is opt-in, the log is not.
    """
    engine.create_table(schemas.OPS_ALERT_LOG)
    return alerts.route(engine, alerts.from_monitor_results(results),
                        dry_run=dry_run)


def main() -> int:
    execute = "--execute" in sys.argv
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

    routed = route_alerts(engine, results, dry_run=not execute)
    if routed:
        for line in routed:
            print(f"  -> {line}")
        if not execute:
            print("monitors: alert routing is dry-run "
                  "(pass --execute to mark alerts delivered)")
    else:
        print("monitors: no alertable results, nothing routed")
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
