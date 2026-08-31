"""The daily report: correct sections, honest absences, and a real diff.

The report is the only artifact a reader actually reads, which makes two
properties non-negotiable. It must never recompute a number the monitors
already measured -- a report that disagrees with the alert is the one that
gets believed. And it must distinguish "checked, found nothing" from "never
checked", because those look identical in an empty findings table and one of
them is a clean bill of health while the other is the absence of evidence.
"""
import json
import pathlib
from datetime import date

import pytest

from src.agent import actions, graph
from src.lakehouse import bronze, maintenance
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
from src.ops import report, runner

D1, D2, D3 = date(2026, 6, 28), date(2026, 6, 29), date(2026, 6, 30)


@pytest.fixture(autouse=True)
def isolate_incidents(tmp_path, monkeypatch):
    """`graph.run` reaches the `act` node for any actionable finding, and `act`
    writes into the repo's real `docs/incidents/`. Several tests below evolve a
    schema or collapse a volume on purpose, so without this they would dirty
    the committed artifact set. Same guard as `test_agent_graph`."""
    monkeypatch.setattr(actions, "INCIDENTS_DIR", tmp_path / "incidents")


def _finding(key, run_at, severity="breaking", kind="schema_drift",
             detail="something", evidence=None, table="bronze.t"):
    return {"run_at": run_at, "finding_key": key, "table_name": table,
            "kind": kind, "severity": severity, "column_name": None,
            "detail": detail, "reasoning": "because",
            "evidence": json.dumps(evidence or {}), "actioned": True}


def _snapshot(**kw):
    base = {"as_of": D3, "previous_run": D2, "ran": True}
    base.update(kw)
    return report.ReportSnapshot(**base)


# --------------------------------------------------------------------------
# It renders, and it computes nothing
# --------------------------------------------------------------------------

def test_build_report_needs_no_engine_at_all():
    """The structural guarantee behind "the report recomputes nothing": it is
    a pure function of already-measured data. If it could compute a metric it
    would need something to compute it from."""
    text = report.build_report(_snapshot())
    assert text.startswith("# Daily platform report — 2026-06-30")


def test_report_module_imports_no_measurement_code():
    """A second, blunter guard on the same property. `monitors.evaluate` and
    `sensors.detect` are where metrics and findings are *produced*; the report
    must only ever read what they already wrote."""
    source = pathlib.Path(report.__file__).read_text()
    for forbidden in ("from src.ops import monitors", "from src.agent import sensors",
                      "from src.contracts import validator", "monitors.evaluate",
                      "sensors.detect", "validator.validate"):
        assert forbidden not in source, f"report.py must not reference {forbidden}"


def test_every_section_is_present():
    text = report.build_report(_snapshot())
    for heading in ("## Verdict", "## Schema and enum evolution", "## Data quality",
                    "## Anomalies", "## Errors and blocked promotions",
                    "## What changed since the previous run", "## Open incidents"):
        assert heading in text, f"missing section: {heading}"


# --------------------------------------------------------------------------
# Empty delta: "clean" and "never ran" must not look the same
# --------------------------------------------------------------------------

def test_a_clean_run_says_clean():
    text = report.build_report(_snapshot(ran=True, findings=[], monitors=[
        {"monitor": "m", "table_name": "t", "metric": 1.0,
         "baseline_median": 1.0, "status": "ok", "run_at": D3}]))
    assert "## Verdict — CLEAN" in text


def test_a_missing_run_is_not_reported_as_clean():
    """The empty-delta blind spot, stated at the top of the report. No rows in
    the findings table is what a clean day and a day nothing ran look like."""
    text = report.build_report(_snapshot(ran=False, findings=[]))
    assert "NO RUN RECORDED" in text
    assert "not** a clean bill of health" in text
    assert "CLEAN" not in text.split("## Schema")[0].replace("NO RUN RECORDED", "")


def test_absent_monitor_results_are_called_absence_of_evidence():
    text = report.build_report(_snapshot(ran=True, monitors=[]))
    assert "absence of evidence, not a pass" in text


# --------------------------------------------------------------------------
# Routing: each finding lands in the right section
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,evidence,heading", [
    ("schema_drift", {"change": "renamed", "field_id": 20},
     "## Schema and enum evolution"),
    ("enum_drift", {"new_values": ["Crypto"]}, "## Schema and enum evolution"),
    ("volume_anomaly", {"median": 250000, "rows_in_latest_snapshot": 5},
     "## Anomalies"),
    ("data_corruption", {"unparseable_count": 3},
     "## Errors and blocked promotions"),
    ("staleness", {"lag_days": 40}, "## Errors and blocked promotions"),
])
def test_finding_kinds_route_to_the_right_section(kind, evidence, heading):
    text = report.build_report(_snapshot(findings=[
        _finding("k", D3, kind=kind, detail="THE-MARKER", evidence=evidence)]))
    section = text.split(heading)[1].split("\n## ")[0]
    assert "THE-MARKER" in section


def test_an_unknown_finding_kind_is_not_silently_dropped():
    """A future check family must appear somewhere. Vanishing from the report
    is the worst outcome -- it is measured, recorded, and invisible."""
    text = report.build_report(_snapshot(findings=[
        _finding("k", D3, kind="some_future_check", detail="THE-MARKER")]))
    assert "THE-MARKER" in text


def test_a_rename_finding_shows_its_field_id():
    text = report.build_report(_snapshot(findings=[
        _finding("k", D3, kind="schema_drift", detail="renamed",
                 evidence={"change": "renamed", "field_id": 20})]))
    assert "field id: `20`" in text
    assert "no data file was rewritten" in text


# --------------------------------------------------------------------------
# The diff -- the section that makes it a story
# --------------------------------------------------------------------------

def test_diff_reports_new_cleared_and_still_open():
    snapshot = _snapshot(
        findings=[_finding("stays", D3, detail="PERSISTING"),
                  _finding("fresh", D3, detail="BRAND-NEW")],
        previous_findings=[_finding("stays", D2, detail="PERSISTING"),
                           _finding("gone", D2, detail="RESOLVED")],
        first_seen={"stays": D1, "fresh": D3, "gone": D1})
    text = report.build_report(snapshot)
    diff = text.split("## What changed since the previous run")[1]

    assert "**1 new**, **1 cleared**, **1 still open**" in diff
    assert "BRAND-NEW" in diff.split("### New")[1].split("###")[0]
    assert "RESOLVED" in diff.split("### Cleared")[1].split("###")[0]
    assert "PERSISTING" in diff.split("### Still open")[1]


def test_age_is_measured_from_first_seen_not_from_today():
    snapshot = _snapshot(findings=[_finding("old", D3)],
                         previous_findings=[_finding("old", D2)],
                         first_seen={"old": D1})
    assert "open 2d (since 2026-06-28)" in report.build_report(snapshot)


def test_no_previous_run_says_so_instead_of_claiming_everything_is_new():
    """Otherwise the first report ever run declares every standing finding
    'new', which is true of the log and false of the platform."""
    text = report.build_report(_snapshot(previous_run=None,
                                          findings=[_finding("k", D3)]))
    diff = text.split("## What changed since the previous run")[1]
    assert "nothing to diff against" in diff
    assert "### New" not in diff


def test_monitor_delta_against_the_previous_run():
    snapshot = _snapshot(
        monitors=[{"monitor": "rows", "table_name": "t", "metric": 900.0,
                   "baseline_median": 1000.0, "status": "warn", "run_at": D3}],
        previous_monitors={"rows": 1000.0})
    assert "-100" in report.build_report(snapshot)


def test_monitor_delta_is_a_dash_when_there_is_no_previous_value():
    """Not 0.0 -- "unchanged" and "never measured before" are different, and
    printing a zero for the second is a small lie that reads as a fact."""
    snapshot = _snapshot(
        monitors=[{"monitor": "rows", "table_name": "t", "metric": 900.0,
                   "baseline_median": None, "status": "ok", "run_at": D3}],
        previous_monitors={})
    row = next(ln for ln in report.build_report(snapshot).splitlines()
               if "`rows`" in ln)
    assert "| — |" in row


# --------------------------------------------------------------------------
# Against a real warehouse, day by day
# --------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=800)
    return eng


def test_agent_run_is_recorded_even_when_it_finds_nothing(engine):
    """`ops.agent_runs` exists for exactly this: a clean run writes no finding
    rows, so without it the report cannot tell clean from never-ran."""
    graph.run(engine=engine, dry_run=True, as_of=D3)

    snapshot = report.load_snapshot(engine, D3)
    assert snapshot.ran is True
    assert snapshot.findings == []
    assert "## Verdict — CLEAN" in report.build_report(snapshot)


def test_a_date_with_no_run_reports_no_run(engine):
    graph.run(engine=engine, dry_run=True, as_of=D3)
    snapshot = report.load_snapshot(engine, date(2026, 7, 15))
    assert snapshot.ran is False
    assert "NO RUN RECORDED" in report.build_report(snapshot)


def test_a_finding_appears_persists_and_clears_across_three_days(engine):
    """The whole point of the feature, end to end against real Iceberg.

    Day 1 clean. Day 2 a column is added -- a finding appears. Day 3 the
    contract is not fixed, so it persists and its age grows. Then the log is
    diffed from the far side to prove a cleared finding is reported as cleared.
    """
    graph.run(engine=engine, dry_run=True, as_of=D1)
    day1 = report.load_snapshot(engine, D1)
    assert day1.findings == []

    maintenance.evolve_day(engine, 1)          # +merchantCategoryCode
    graph.run(engine=engine, dry_run=True, as_of=D2)
    day2 = report.load_snapshot(engine, D2)
    added = [f for f in day2.findings if "merchantCategoryCode" in f["detail"]]
    assert len(added) == 1
    text2 = report.build_report(day2)
    assert "**1 new**" in text2
    assert "new today" in text2

    graph.run(engine=engine, dry_run=True, as_of=D3)
    day3 = report.load_snapshot(engine, D3)
    text3 = report.build_report(day3)
    assert "**0 new**, **0 cleared**, **1 still open**" in text3
    assert "open 1d (since 2026-06-29)" in text3


def test_a_resolved_finding_is_reported_as_cleared(engine):
    """Built by hand from the log rather than by fixing a contract mid-test:
    what is under test is the diff, not contract editing."""
    graph.run(engine=engine, dry_run=True, as_of=D1)
    maintenance.evolve_day(engine, 1)
    graph.run(engine=engine, dry_run=True, as_of=D2)

    snapshot = report.load_snapshot(engine, D2)
    resolved = report.ReportSnapshot(
        as_of=D3, previous_run=D2, ran=True, findings=[],
        previous_findings=snapshot.findings, first_seen=snapshot.first_seen,
        monitors=[], previous_monitors={})
    text = report.build_report(resolved)

    assert "**0 new**, **1 cleared**, **0 still open**" in text
    assert "merchantCategoryCode" in text.split("### Cleared")[1]


def test_monitor_results_reach_the_quality_section(engine):
    results = runner.run_monitors(engine, D3)
    runner.persist(engine, results)
    graph.run(engine=engine, dry_run=True, as_of=D3)

    text = report.build_report(report.load_snapshot(engine, D3))
    quality = text.split("## Data quality")[1].split("\n## ")[0]
    assert "bronze_txn_row_count" in quality
    assert "monitor(s)," in quality


def test_a_volume_collapse_reaches_the_anomaly_section(engine):
    maintenance.inject_volume_collapse(engine)
    graph.run(engine=engine, dry_run=True, as_of=D3)

    text = report.build_report(report.load_snapshot(engine, D3))
    anomalies = text.split("## Anomalies")[1].split("\n## ")[0]
    assert "below 50% of trailing median" in anomalies
    assert "baseline median" in anomalies


def test_findings_written_by_the_agent_are_keyed_consistently(engine):
    """The incident file and the log row must agree on identity, or "first
    seen" describes a different thing than the file on disk."""
    from src.agent import actions, sensors
    maintenance.evolve_day(engine, 4)          # the rename
    state = graph.run(engine=engine, dry_run=True, as_of=D3)

    logged = {r["finding_key"] for r in report.load_snapshot(engine, D3).findings}
    for item in state["classified"]:
        key = sensors.finding_key(item["finding"])
        assert key in logged
        # and the incident slug is derived from the same key
        assert actions.incident_slug(item["finding"]).endswith(
            __import__("hashlib").sha256(key.encode()).hexdigest()[:8])


def test_running_the_agent_twice_in_one_day_does_not_double_the_findings(engine):
    """`ops.finding_log` is append-only and re-running `make agent` is
    ordinary. Counting rows instead of findings reported "2 still open" for a
    single finding -- and a diff whose counts are wrong is worse than no diff,
    because it is read as a measurement."""
    maintenance.evolve_day(engine, 1)
    graph.run(engine=engine, dry_run=True, as_of=D2)
    graph.run(engine=engine, dry_run=True, as_of=D2)   # a human re-runs it

    snapshot = report.load_snapshot(engine, D2)
    keys = [f["finding_key"] for f in snapshot.findings]
    assert len(keys) == len(set(keys)), f"duplicate findings in one run: {keys}"

    graph.run(engine=engine, dry_run=True, as_of=D3)
    text = report.build_report(report.load_snapshot(engine, D3))
    assert "**0 new**, **0 cleared**, **1 still open**" in text


def test_running_monitors_twice_in_one_day_does_not_double_the_rows(engine):
    runner.persist(engine, runner.run_monitors(engine, D3))
    runner.persist(engine, runner.run_monitors(engine, D3))
    graph.run(engine=engine, dry_run=True, as_of=D3)

    snapshot = report.load_snapshot(engine, D3)
    names = [m["monitor"] for m in snapshot.monitors]
    assert len(names) == len(set(names)), "monitor rows duplicated for one run"


def test_a_pipe_in_a_detail_string_does_not_break_the_table():
    """Detail text is machine-generated but carries column names and values;
    an unescaped `|` silently splits a Markdown table cell."""
    text = report.build_report(_snapshot(findings=[
        _finding("k", D3, kind="staleness", detail="a | b | c")]))
    row = next(ln for ln in text.splitlines()
               if ln.startswith("| `breaking`") and "b" in ln)
    assert "\\|" in row, "the pipe in the detail text was not escaped"
    # Escaped pipes still contain a `|` character, so strip them before
    # counting the structural ones: four cells means five separators.
    assert row.replace("\\|", "").count("|") == 5, (
        f"pipe leaked into the table structure: {row}")


def test_the_report_is_byte_stable_across_regeneration(engine):
    """Reports are committed artifacts. Set-iteration order made the diff
    sections shuffle between runs, producing a git diff on every regeneration
    that had nothing to do with what changed."""
    maintenance.evolve_day(engine, 1)
    graph.run(engine=engine, dry_run=True, as_of=D2)
    maintenance.evolve_day(engine, 4)
    graph.run(engine=engine, dry_run=True, as_of=D3)

    first = report.build_report(report.load_snapshot(engine, D3))
    second = report.build_report(report.load_snapshot(engine, D3))
    assert first == second

    # and stable across a fresh load, not just a repeated render
    assert report.build_report(report.load_snapshot(engine, D3)) == first


def test_float_noise_is_not_rendered_as_a_change():
    """DuckDB sums in parallel, so a mean over 250k rows differs in its last
    bits between runs of identical code. Rendering that as `+1.56e-12` presents
    noise as signal and churns the committed reports on every regeneration."""
    snapshot = _snapshot(
        monitors=[{"monitor": "mean", "table_name": "t",
                   "metric": 171.41370268000063, "baseline_median": 171.4,
                   "status": "ok", "run_at": D3}],
        previous_monitors={"mean": 171.41370268000023})
    row = next(ln for ln in report.build_report(snapshot).splitlines()
               if "`mean`" in ln)
    assert "e-13" not in row and "e-12" not in row
    assert "| 0 |" in row


def test_a_real_change_is_still_rendered():
    """The noise threshold must not swallow a change a monitor could act on."""
    snapshot = _snapshot(
        monitors=[{"monitor": "mean", "table_name": "t", "metric": 171.5,
                   "baseline_median": 171.4, "status": "ok", "run_at": D3}],
        previous_monitors={"mean": 171.4})
    row = next(ln for ln in report.build_report(snapshot).splitlines()
               if "`mean`" in ln)
    assert "+0.1" in row


def test_findings_sharing_a_first_seen_date_are_ordered_deterministically():
    """Several findings raised on one day share a `first_seen`; without a
    tiebreak their order followed whatever order the checks ran in, so
    reordering the config churned the table with no change in content."""
    findings = [_finding(k, D3, kind="staleness", detail=f"d-{k}")
                for k in ("zebra", "alpha", "middle")]
    first = report.build_report(_snapshot(findings=findings, first_seen={
        k: D3 for k in ("zebra", "alpha", "middle")}))
    second = report.build_report(_snapshot(findings=list(reversed(findings)),
                                            first_seen={k: D3 for k in
                                                        ("zebra", "alpha", "middle")}))
    assert first == second, "render depends on the order findings arrive in"
