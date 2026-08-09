"""What the agent does about a finding. Dry-run by default.

An ops agent that files GitHub issues the moment someone runs it locally is
one people stop running. `--execute` is opt-in.
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
    """Stable across runs so a recurring finding updates one file, not many.

    Keyed on table + kind + the identifying bits of evidence (column, change)
    rather than the human-readable `detail` string, so a rewording of the
    message text doesn't spawn a new incident file for the same underlying
    condition.
    """
    key = (f"{finding.table}|{finding.kind}|{finding.evidence.get('column', '')}"
           f"|{finding.evidence.get('change', '')}")
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    column = finding.evidence.get("column", finding.kind)
    return f"{finding.table.replace('.', '-')}-{column}-{digest}"


def write_incident(finding: Finding, severity: str, reasoning: str, as_of: date) -> Path:
    """Write (or overwrite) the one incident file for this finding.

    Same slug on every call for the same finding -- a rerun updates the
    existing file's `as_of` and evidence in place rather than accumulating a
    new file per run.
    """
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
    """File (or, by default, describe) a GitHub issue for a breaking finding.

    Dry-run unless `execute=True`: prints the command it would have run
    instead of running it. No network in the default path.
    """
    title = f"[{severity}] {finding.kind} on {finding.table}"
    body = f"{finding.detail}\n\n{reasoning}\n\nEvidence: {finding.evidence}"
    command = ["gh", "issue", "create", "--title", title, "--body", body,
               "--label", f"data-{severity}"]
    if not execute:
        return f"DRY-RUN: would run `{' '.join(command[:3])} ...` -> {title}"
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: `gh` may be
        # missing, unauthenticated, or rate-limited; any failure of this
        # best-effort side effect must be reported, not crash the agent.
        return f"FAILED to file issue: {exc}"
