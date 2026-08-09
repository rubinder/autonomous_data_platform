"""Severity classification: rules first, LLM only for genuine ambiguity.

CI must never depend on a model call, so USE_LLM is False unless a key is
present, and every rule path below is deterministic. An ops agent whose test
suite needs the network is an ops agent nobody can run in CI.

Type WIDENING (e.g. int -> long) is additive: existing values still fit, and
old readers keep working. NARROWING or a kind change is breaking: existing
values may not fit and downstream casts can fail. Treating every type change
as breaking trains people to ignore the alerts.
"""
from __future__ import annotations

import os

from src.agent.sensors import Finding

USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

# (declared, observed) pairs where the table's actual type is a safe widening
# of what the contract declares.
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

    if finding.kind == "data_corruption":
        column = finding.evidence.get("column")
        where = f" in '{column}'" if column else ""
        return "breaking", (
            f"{finding.evidence.get('unparseable_count')} value(s){where} do not "
            "parse as dates. Every downstream time-based join or freshness check "
            "on this column is unreliable until the source data is fixed.")

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
    """Optional judgment call for findings no rule above matches.

    Only reached when USE_LLM is True, which requires ANTHROPIC_API_KEY --
    never the case in CI or in this module's own tests. Falls back to benign
    on any error so a flaky or absent model never blocks the agent.
    """
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
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure of an
        # optional, best-effort LLM call (network, auth, SDK, parsing) must degrade to
        # "benign" rather than take the agent down.
        return "benign", f"LLM classification unavailable ({type(exc).__name__})."
