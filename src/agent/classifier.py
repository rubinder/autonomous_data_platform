"""Severity classification: rules first, LLM only for genuine ambiguity.

CI must never depend on a model call, so USE_LLM is False unless a key is
present, and every rule path below is deterministic. An ops agent whose test
suite needs the network is an ops agent nobody can run in CI.

The severity vocabulary is breaking / renaming / widening / additive /
enum_drift / benign, and each name exists because it implies a different
response:

- `breaking`  — data may be lost or a cast will fail. A human is needed now.
- `renaming`  — no data moved (the field ID is unchanged, so Iceberg still
  resolves every old file), but the published contract now names a column
  that does not exist. Nothing is broken and nothing is fine: it needs a
  recorded decision, so it is actionable without being an emergency.
- `widening`  — int -> long and friends. Existing values still fit and old
  readers keep working, so nobody is paged; it is still not the same event as
  a new column appearing, and collapsing the two loses the distinction
  between "the contract is behind" and "the data changed shape".
- `additive`  — a new column. Worth recording, not worth paging.
- `enum_drift` — a new value in a watched categorical column. Recorded, never
  blocking; see `sensors._detect_enum_drift` for why this is not a gate.
- `benign`    — nothing to do.

NARROWING or a kind change stays breaking: existing values may not fit and
downstream casts can fail. Treating every type change as breaking trains
people to ignore the alerts, which is the failure this split exists to avoid.
"""
from __future__ import annotations

import os

from src.agent.sensors import Finding

USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

# (declared, observed) pairs where the table's actual type is a safe widening
# of what the contract declares.
_WIDENING = {("int", "long"), ("int32", "int64"), ("float", "double")}


def classify(finding: Finding) -> tuple[str, str]:
    if finding.kind == "monitor_breach":
        monitor = finding.evidence.get("monitor")
        return "breaking", (
            f"Monitor '{monitor}' breached on {finding.table}: {finding.detail} "
            "A monitor breach is, by definition, already a judged verdict -- "
            "there is no additive reading of a check that has already failed.")

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

    if finding.kind == "enum_drift":
        column = finding.evidence.get("column")
        error = finding.evidence.get("error")

        if error == "column_absent":
            # A watch that cannot see its column is worse than no watch: it
            # occupies the slot where the alert would have been and reports
            # nothing wrong. Same reasoning as the quarantine monitor that
            # was `SELECT 0.0` -- a check displaying green while measuring
            # nothing is a defect, not a passing check.
            return "breaking", (
                f"The enum watch on '{column}' is not monitoring anything: the "
                "column produced no values. Until this is fixed, new values in "
                "it will go unnoticed while the check reports clean.")

        if error == "cardinality_exceeded":
            return "benign", (
                f"'{column}' is not a categorical column "
                f"({finding.evidence.get('distinct_at_least')}+ distinct values), "
                "so the watch is misconfigured. No conclusion about drift can be "
                "drawn from it either way; remove the watch.")

        values = finding.evidence.get("new_values", [])
        return "enum_drift", (
            f"New value(s) {', '.join(repr(v) for v in values)} appeared in "
            f"'{column}', which the contract does not register. Nothing breaks "
            "today -- but any mart that groups or filters on this column now "
            "has a bucket nobody declared, and that is how totals quietly stop "
            "reconciling.")

    if finding.kind == "schema_drift":
        change = finding.evidence.get("change")
        column = finding.evidence.get("column")

        if change == "added":
            return "additive", (
                f"New column '{column}' does not affect existing readers. Worth "
                "adding to the contract, not worth paging anyone.")

        if change == "dropped":
            # Deliberately no claim about which downstream job breaks. The
            # classifier sees one table and one contract; it does not know
            # which columns any consumer selects, so "this will fail the next
            # Silver build" is a guess dressed as evidence -- and for
            # `checkNumber` it was simply false, since Silver never selects
            # it. What *is* known and sufficient: the declared contract is
            # violated, and name-based consumers of this column break.
            return "breaking", (
                f"Column '{column}' is declared in the contract but is absent "
                "from the table, so the published contract is violated. Any "
                "consumer that selects it by name breaks; which consumers "
                "those are is not visible from this table's metadata and "
                "needs a human to confirm.")

        if change == "renamed":
            new_name = finding.evidence.get("renamed_to")
            field_id = finding.evidence.get("field_id")
            declared = finding.evidence.get("declared_type", "")
            observed = finding.evidence.get("observed_type", "")

            # A rename is lossless. A rename that also changed the type is
            # not, and must not inherit the calmer severity just because the
            # two happened in one step.
            if finding.evidence.get("type_compatible") is False:
                return "breaking", (
                    f"'{column}' was renamed to '{new_name}' (both are field id "
                    f"{field_id}, so no data file moved) but its type also went "
                    f"{declared} -> {observed}. The rename is safe; the type "
                    "change is not, and that is what needs attention.")

            return "renaming", (
                f"'{column}' was renamed to '{new_name}'. Both are field id "
                f"{field_id}: Iceberg resolves columns by ID, not by name, so "
                f"every file written under the old name still reads back "
                f"correctly and no data was rewritten or lost. What is wrong "
                f"is the published contract, which still declares '{column}'. "
                "That needs a recorded decision, not a rollback.")

        if change == "type_changed":
            declared = finding.evidence.get("declared_type", "")
            observed = finding.evidence.get("observed_type", "")
            if (declared, observed) in _WIDENING:
                return "widening", (
                    f"'{column}' widened {declared} -> {observed}; every existing "
                    "value still fits and old readers keep working. Worth "
                    "recording against the contract, not worth paging anyone.")
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
