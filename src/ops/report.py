"""The daily platform report: what the agent saw, and what changed since.

This module **reads**. It does not measure. Every number in the report was
computed by `src.ops.runner` or `src.agent.graph` and persisted to `ops.*`
before the report existed; the report's job is to select, diff and render.

That split is not tidiness. A report that recomputes a metric can disagree
with the monitor that raised the alert, and when those two disagree the report
is the one people read. So `load_snapshot()` is the only function here that
touches an engine, and `build_report()` takes a plain dataclass and returns a
string -- which is also what makes "the report computes nothing" testable
rather than merely claimed.

The day-over-day diff is the section that turns a status page into a story.
`docs/incidents/*.md` cannot provide it: incident slugs are deliberately
stable so a recurring finding rewrites one file instead of spawning many, so
every write destroys the previous state. `ops.finding_log` is append-only and
keyed on `sensors.finding_key`, so "first seen", "still open" and "cleared"
are answerable by query.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from src import config
from src.lakehouse import schemas

REPORT_DIR = config.REPO_ROOT / "reports"

# Which section a finding kind belongs in. A kind absent from this map is not
# dropped -- `_section_of` routes it to "Errors" rather than letting it vanish,
# because a new check family appearing in the log and silently not appearing in
# the report is the failure this whole file exists to prevent.
_SCHEMA_KINDS = frozenset({"schema_drift", "enum_drift"})
_ANOMALY_KINDS = frozenset({"volume_anomaly"})
_QUALITY_MONITOR_KINDS = frozenset({"null_rate", "duplicate_rate"})
_ANOMALY_MONITOR_KINDS = frozenset({"row_count", "cardinality", "distribution_shift"})

ACTIONABLE_SEVERITIES = frozenset({"breaking", "renaming"})


@dataclass(frozen=True)
class ReportSnapshot:
    """Everything the report renders, already selected from `ops.*`."""
    as_of: date
    previous_run: date | None
    ran: bool                      # was an agent run recorded for `as_of`?
    findings: list[dict] = field(default_factory=list)
    previous_findings: list[dict] = field(default_factory=list)
    first_seen: dict[str, date] = field(default_factory=dict)
    monitors: list[dict] = field(default_factory=list)
    previous_monitors: dict[str, float | None] = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _rows(engine, table_def, query: str) -> list[dict]:
    if not engine.table_exists(table_def.name):
        return []
    return engine.sql(query, tables={"t": table_def.name}).to_pylist()


def _as_date(value) -> date | None:
    return value.date() if hasattr(value, "date") else value


def _latest_per_key(rows: list[dict]) -> list[dict]:
    """One row per `finding_key`, last write wins.

    `ops.finding_log` is append-only, so running `make agent` twice on the same
    date -- an entirely ordinary thing for a human to do -- writes the same
    logical finding twice. Counting rows instead of findings then reports "2
    still open" where one finding exists, and the diff's whole value is that
    its counts mean something.
    """
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["finding_key"]] = row
    return list(latest.values())


def load_snapshot(engine, as_of: date) -> ReportSnapshot:
    """Select everything the report needs for `as_of`. The only engine call."""
    runs = [_as_date(r["run_at"]) for r in _rows(
        engine, schemas.OPS_AGENT_RUNS,
        "SELECT DISTINCT run_at FROM t ORDER BY run_at")]
    ran = as_of in runs
    earlier = [r for r in runs if r < as_of]
    previous = earlier[-1] if earlier else None

    findings = _rows(engine, schemas.OPS_FINDING_LOG, "SELECT * FROM t")
    for row in findings:
        row["run_at"] = _as_date(row["run_at"])

    today = _latest_per_key([f for f in findings if f["run_at"] == as_of])
    prior = _latest_per_key(
        [f for f in findings if f["run_at"] == previous]) if previous else []

    # First time each key was ever seen, across the whole log.
    first_seen: dict[str, date] = {}
    for row in findings:
        key = row["finding_key"]
        if key not in first_seen or row["run_at"] < first_seen[key]:
            first_seen[key] = row["run_at"]

    monitor_rows = _rows(engine, schemas.OPS_MONITOR_RESULTS, "SELECT * FROM t")
    for row in monitor_rows:
        row["run_at"] = _as_date(row["run_at"])
    monitors_today = [m for m in monitor_rows if m["run_at"] == as_of]
    # Fall back to the newest run at or before as_of: `make monitor` and
    # `make agent` are separate commands and need not share a run_at.
    if not monitors_today and monitor_rows:
        newest = max((m["run_at"] for m in monitor_rows if m["run_at"] <= as_of),
                     default=None)
        monitors_today = [m for m in monitor_rows if m["run_at"] == newest]

    monitors_today = list({m["monitor"]: m for m in monitors_today}.values())

    monitor_runs = sorted({m["run_at"] for m in monitor_rows})
    current_run = monitors_today[0]["run_at"] if monitors_today else None
    prior_runs = [r for r in monitor_runs if current_run and r < current_run]
    # Same dedupe reasoning as `_latest_per_key`: a second `make monitor` on
    # one date must not turn one metric into two.
    previous_monitors = {
        m["monitor"]: m["metric"]
        for m in monitor_rows if prior_runs and m["run_at"] == prior_runs[-1]
    }

    alerts = _rows(engine, schemas.OPS_ALERT_LOG, "SELECT * FROM t")
    for row in alerts:
        row["run_at"] = _as_date(row["run_at"])
    alerts = [a for a in alerts if a["run_at"] == as_of]

    return ReportSnapshot(
        as_of=as_of, previous_run=previous, ran=ran,
        findings=today, previous_findings=prior, first_seen=first_seen,
        monitors=monitors_today, previous_monitors=previous_monitors,
        alerts=alerts)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _section_of(finding: dict) -> str:
    kind = finding["kind"]
    if kind in _SCHEMA_KINDS:
        return "schema"
    if kind in _ANOMALY_KINDS:
        return "anomaly"
    if kind == "monitor_breach":
        evidence = _evidence(finding)
        monitor_kind = evidence.get("kind")
        if monitor_kind in _ANOMALY_MONITOR_KINDS:
            return "anomaly"
        if monitor_kind in _QUALITY_MONITOR_KINDS:
            return "quality"
    return "error"


def _num(value) -> str:
    """Readable, not scientific. A row count is the number people recognise.

    `f"{250000:,.4g}"` renders `2.5e+05`, which is correct and unreadable --
    and a report nobody can scan is a report nobody reads.
    """
    if value is None:
        return "—"
    magnitude = abs(value)
    if magnitude >= 1000 and float(value).is_integer():
        return f"{value:,.0f}"
    if 0 < magnitude < 0.001:
        return f"{value:.2e}"
    return f"{value:,.4g}"


def _delta(current, previous) -> str:
    """Signed change, or an em dash when there is nothing to compare against.

    Never `0` for "no previous value": unchanged and never-measured are
    different facts, and printing zero for the second is a small lie that
    reads as a measurement.
    """
    if current is None or previous is None:
        return "—"
    change = current - previous
    if change == 0:
        return "0"
    return f"{'+' if change > 0 else ''}{_num(change)}"


def _evidence(finding: dict) -> dict:
    try:
        return json.loads(finding.get("evidence") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _age(snapshot: ReportSnapshot, key: str) -> str:
    seen = snapshot.first_seen.get(key)
    if seen is None or seen == snapshot.as_of:
        return "new today"
    return f"open {(snapshot.as_of - seen).days}d (since {seen})"


def _verdict(snapshot: ReportSnapshot) -> list[str]:
    if not snapshot.ran:
        return [
            "## Verdict — NO RUN RECORDED",
            "",
            (f"No agent run is recorded for {snapshot.as_of}. This is **not** "
             "a clean bill of health: it means the report found no evidence "
             "that anything was checked. Run `make agent` for this date."),
        ]

    blocking = [f for f in snapshot.findings
                if f["severity"] in ACTIONABLE_SEVERITIES]
    failed_monitors = [m for m in snapshot.monitors if m["status"] == "breach"]

    if not blocking and not failed_monitors:
        return [
            "## Verdict — CLEAN",
            "",
            (f"The agent ran on {snapshot.as_of} and raised "
             f"{len(snapshot.findings)} finding(s), none of them actionable. "
             f"{len(snapshot.monitors)} monitor(s) reported, none breaching."),
        ]

    reasons = [f"{len(blocking)} actionable finding(s)"] if blocking else []
    if failed_monitors:
        reasons.append(f"{len(failed_monitors)} monitor breach(es)")
    return [
        "## Verdict — ATTENTION REQUIRED",
        "",
        f"{' and '.join(reasons)} on {snapshot.as_of}.",
        "",
        *[f"- **[{f['severity']}]** {f['detail']}" for f in blocking],
    ]


def _findings_table(snapshot: ReportSnapshot, findings: list[dict]) -> list[str]:
    if not findings:
        return ["_Nothing in this section for this run._"]
    lines = ["| Severity | Table | What | Age |", "|---|---|---|---|"]
    for f in sorted(findings, key=lambda x: x["severity"]):
        detail = (f.get("detail") or "").replace("|", "\\|")
        lines.append(f"| `{f['severity']}` | `{f['table_name']}` | {detail} "
                     f"| {_age(snapshot, f['finding_key'])} |")
    return lines


def _schema_section(snapshot: ReportSnapshot) -> list[str]:
    findings = [f for f in snapshot.findings if _section_of(f) == "schema"]
    lines = ["## Schema and enum evolution", ""]
    if not findings:
        lines.append("No schema or enum change observed against the registered "
                     "contracts this run.")
        return lines
    for f in sorted(findings, key=lambda x: x["kind"]):
        evidence = _evidence(f)
        change = evidence.get("change") or evidence.get("error") or f["kind"]
        lines.append(f"### `{change}` — {f['table_name']}")
        lines.append("")
        lines.append(f"**{f['detail']}**")
        lines.append("")
        if evidence.get("field_id") is not None:
            lines.append(f"- Iceberg field id: `{evidence['field_id']}` "
                         "(unchanged — no data file was rewritten)")
        if evidence.get("new_values"):
            lines.append(f"- New enum value(s): "
                         f"{', '.join(f'`{v}`' for v in evidence['new_values'])}")
        # Only when they actually differ. "Contract declares `string`, table
        # has `string`" is noise on a pure rename, and noise in a report is
        # how the signal in it stops being read.
        declared, observed = (evidence.get("declared_type"),
                              evidence.get("observed_type"))
        if declared and observed and declared != observed:
            lines.append(f"- Contract declares `{declared}`, "
                         f"table has `{observed}`")
        lines.append(f"- Severity `{f['severity']}` — {f.get('reasoning', '')}")
        lines.append(f"- {_age(snapshot, f['finding_key'])}")
        lines.append("")
    return lines


def _quality_section(snapshot: ReportSnapshot) -> list[str]:
    lines = ["## Data quality", ""]
    if not snapshot.monitors:
        lines.append("**No monitor results recorded for this date.** That is an "
                     "absence of evidence, not a pass — run `make monitor`.")
        return lines

    lines += ["| Monitor | Table | Observed | Baseline | Δ vs prev | Status |",
              "|---|---|---|---|---|---|"]
    for m in sorted(snapshot.monitors, key=lambda x: (x["status"] != "breach",
                                                       x["monitor"])):
        metric, baseline = m.get("metric"), m.get("baseline_median")
        prev = snapshot.previous_monitors.get(m["monitor"])
        lines.append(
            f"| `{m['monitor']}` | `{m['table_name']}` "
            f"| {_num(metric)} | {_num(baseline)} "
            f"| {_delta(metric, prev)} | `{m['status']}` |")
    breaches = [m for m in snapshot.monitors if m["status"] == "breach"]
    lines += ["", f"{len(snapshot.monitors)} monitor(s), {len(breaches)} breaching."]
    return lines


def _anomaly_section(snapshot: ReportSnapshot) -> list[str]:
    findings = [f for f in snapshot.findings if _section_of(f) == "anomaly"]
    lines = ["## Anomalies", ""]
    if not findings:
        lines.append("No volume or distribution anomaly raised this run.")
        return lines
    for f in findings:
        evidence = _evidence(f)
        lines.append(f"- **{f['detail']}**")
        if evidence.get("median") is not None:
            lines.append(
                f"  - baseline median: `{_num(evidence['median'])}`, "
                f"observed: `{_num(evidence.get('rows_in_latest_snapshot'))}`")
        lines.append(f"  - {_age(snapshot, f['finding_key'])}")
    return lines


def _error_section(snapshot: ReportSnapshot) -> list[str]:
    findings = [f for f in snapshot.findings if _section_of(f) == "error"]
    lines = ["## Errors and blocked promotions", ""]
    blocked = [f for f in findings if f["severity"] in ACTIONABLE_SEVERITIES]
    lines.append(f"**{len(blocked)}** finding(s) would block promotion.")
    lines.append("")
    lines += _findings_table(snapshot, findings)
    return lines


def _diff_section(snapshot: ReportSnapshot) -> list[str]:
    lines = ["## What changed since the previous run", ""]
    if snapshot.previous_run is None:
        lines.append("No earlier agent run is recorded, so there is nothing to "
                     "diff against. Every finding above is being seen for the "
                     "first time by definition, not because it is new.")
        return lines

    today = {f["finding_key"]: f for f in snapshot.findings}
    prior = {f["finding_key"]: f for f in snapshot.previous_findings}
    # Sorted, not set-iteration order. These are committed artifacts: a report
    # whose lines shuffle between runs produces a git diff on every
    # regeneration that has nothing to do with what changed, and a diff that is
    # usually noise is a diff nobody reads.
    def _ordered(keys, source):
        return [source[k] for k in sorted(keys)]

    new = _ordered(today.keys() - prior.keys(), today)
    cleared = _ordered(prior.keys() - today.keys(), prior)
    still = _ordered(today.keys() & prior.keys(), today)

    lines.append(f"Compared with `{snapshot.previous_run}`: "
                 f"**{len(new)} new**, **{len(cleared)} cleared**, "
                 f"**{len(still)} still open**.")
    lines.append("")
    if new:
        lines.append("### New")
        lines += [f"- `[{f['severity']}]` {f['detail']}" for f in new]
        lines.append("")
    if cleared:
        lines.append("### Cleared")
        lines += [f"- ~~`[{f['severity']}]` {f['detail']}~~" for f in cleared]
        lines.append("")
    if still:
        lines.append("### Still open")
        lines += [f"- `[{f['severity']}]` {f['detail']} "
                  f"({_age(snapshot, f['finding_key'])})" for f in still]
        lines.append("")
    if not (new or cleared or still):
        lines.append("Nothing open on either date — the platform was clean "
                     "before and is clean now.")
    return lines


def _incidents_section(snapshot: ReportSnapshot) -> list[str]:
    open_incidents = [f for f in snapshot.findings
                      if f["severity"] in ACTIONABLE_SEVERITIES]
    lines = ["## Open incidents", ""]
    if not open_incidents:
        lines.append("None open.")
        return lines
    lines += ["| Severity | Table | Detail | Age |", "|---|---|---|---|"]
    for f in sorted(open_incidents,
                    key=lambda x: snapshot.first_seen.get(x["finding_key"],
                                                          snapshot.as_of)):
        detail = (f.get("detail") or "").replace("|", "\\|")
        lines.append(f"| `{f['severity']}` | `{f['table_name']}` | {detail} "
                     f"| {_age(snapshot, f['finding_key'])} |")
    return lines


def build_report(snapshot: ReportSnapshot) -> str:
    """Render `snapshot`. Takes no engine and computes no metric of its own."""
    parts: list[list[str]] = [
        [f"# Daily platform report — {snapshot.as_of}", "",
         ("_Assembled from `ops.agent_runs`, `ops.finding_log`, "
          "`ops.monitor_results` and `ops.alert_log`. Every number here was "
          "measured by the monitors or the agent before this report ran; "
          "nothing on this page is recomputed._"), ""],
        _verdict(snapshot),
        _schema_section(snapshot),
        _quality_section(snapshot),
        _anomaly_section(snapshot),
        _error_section(snapshot),
        _diff_section(snapshot),
        _incidents_section(snapshot),
    ]
    # Sections joined with a blank line between them: a `##` heading on the
    # line straight after a list item is fragile in some Markdown renderers,
    # and unreadable as raw text either way.
    rendered = []
    for part in parts:
        while part and not part[-1].strip():
            part = part[:-1]
        rendered.append("\n".join(part))
    return "\n\n".join(rendered) + "\n"


def main() -> int:
    from src.lakehouse.bronze import max_txn_date
    from src.lakehouse.engines import get_engine

    engine = get_engine()
    as_of = config.resolve_as_of_date(max_txn_date(engine))
    snapshot = load_snapshot(engine, as_of)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(REPORT_DIR) / f"daily-{as_of}.md"
    path.write_text(build_report(snapshot))

    print(f"report: {as_of} — {len(snapshot.findings)} finding(s), "
          f"{len(snapshot.monitors)} monitor result(s) -> {path}")
    if not snapshot.ran:
        print("report: WARNING no agent run recorded for this date; "
              "the report says so rather than reporting clean")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
