from datetime import date

import pytest

from src.lakehouse import catalog as catalog_mod
from src.lakehouse import schemas
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


def test_monitor_run_routes_alerts_and_writes_the_alert_log(engine):
    """`src.ops.alerts` had no production caller at all.

    `ops.alert_log` was declared in `schemas.ALL_TABLES`, created by no
    pipeline, and described in the README and ADR-0007 as live behaviour.
    `runner.route_alerts` is the wiring that makes those descriptions true.
    """
    from src.ops import runner

    results = [_result("ok"), _result("breach", "m_breach"),
               _result("warn", "m_warn")]
    routed = runner.route_alerts(engine, results, dry_run=True)

    assert len(routed) == 2                       # ok is not alertable
    assert all("DRY-RUN" in line for line in routed)
    assert engine.table_exists(schemas.OPS_ALERT_LOG.name)

    log = engine.scan_arrow(schemas.OPS_ALERT_LOG.name)
    assert log.num_rows == 2
    assert set(log.column("delivered").to_pylist()) == {False}

    # Second identical run throttles rather than re-paging.
    again = runner.route_alerts(engine, results, dry_run=True)
    assert again and all("throttled" in line.lower() for line in again)


def test_a_clean_monitor_run_routes_nothing(engine):
    from src.ops import runner

    assert runner.route_alerts(engine, [_result("ok")], dry_run=True) == []
    assert engine.scan_arrow(schemas.OPS_ALERT_LOG.name).num_rows == 0


def test_distinct_monitors_are_not_throttled_against_each_other(engine):
    alerts.route(engine, alerts.from_monitor_results([_result(monitor="a")]),
                 dry_run=True)
    sent = alerts.route(engine, alerts.from_monitor_results([_result(monitor="b")]),
                        dry_run=True)
    assert sent and not any("throttled" in s.lower() for s in sent)
