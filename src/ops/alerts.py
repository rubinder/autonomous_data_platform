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
    """Stable across recent runs -- never builtin `hash()`.

    Builtin `hash()` is salted per process (PYTHONHASHSEED), so the same
    logical breach would mint a different key on every run and throttling
    would never engage. `hashlib` is deterministic across processes, which
    is the whole point of a key meant to be compared run over run.
    """
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
