"""LangGraph ops agent: sense -> detect -> classify -> decide -> act.

Offline by default: `run()` accepts an engine so tests never touch a
warehouse or the network, and `--execute` is the only thing that turns the
agent's own side effect (filing a GitHub issue) from a dry-run description
into a real `gh` call.
"""
from __future__ import annotations

import sys
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src import config
from src.agent import actions, classifier, sensors
from src.contracts import validator
from src.lakehouse import schemas
from src.ops import arrival, runner

# (table_def, contract file under CONTRACTS_DIR, freshness column to observe)
WATCHED = (
    (schemas.BRONZE_TRANSACTIONS, "bronze_yodlee_transactions.yaml", "transactionDate"),
)
ACTIONABLE = {"breaking"}  # additive/benign are recorded but take no action


class AgentState(TypedDict, total=False):
    as_of: date
    dry_run: bool
    engine: Any
    findings: list[sensors.Finding]
    classified: list[dict]
    actions_taken: list[str]


def sense_and_detect(state: AgentState) -> AgentState:
    engine = state["engine"]
    findings: list[sensors.Finding] = []
    for table_def, contract_file, date_column in WATCHED:
        contract = validator.load_contract(validator.CONTRACTS_DIR / contract_file)
        observed = sensors.observe(engine, table_def, date_column=date_column)
        # Trailing history excludes the latest snapshot -- comparing a batch
        # against a median that includes itself blunts the signal.
        per_snapshot = engine.snapshot_row_counts(table_def.name)
        history = per_snapshot[:-1] or per_snapshot
        findings += sensors.detect(observed, contract, state["as_of"], history)
    return {**state, "findings": findings}


def run_monitors(state: AgentState) -> AgentState:
    """Row-level monitors plus arrival SLAs, folded into `state["findings"]`.

    Only `breach` results become findings -- `warn` is real signal for the
    alert channel (see `src.ops.alerts`) but is not, on its own, grounds for
    the agent to file an incident or a GitHub issue.

    Arrival SLAs are skipped for a table that does not exist yet rather than
    reported as an `arrival_missing` breach: the agent can run against a
    lakehouse mid-build (Silver/Gold not produced yet), and that is not the
    same condition as a Silver/Gold table that existed and then stopped
    receiving data. `src.ops.arrival`'s own CLI (`make arrival`, run only
    after `make all`) is where a genuinely missing table is treated as a
    breach.
    """
    engine = state["engine"]
    as_of = state["as_of"]
    findings = list(state["findings"])

    results = runner.run_monitors(engine, as_of)
    for sla in arrival.ARRIVAL_SLAS:
        if engine.table_exists(sla.table):
            results += arrival.check_arrival(engine, sla, as_of)

    for result in results:
        if result.status != "breach":
            continue
        findings.append(sensors.Finding(
            kind="monitor_breach", table=result.table,
            detail=f"[{result.monitor}] {result.detail}",
            evidence={
                "monitor": result.monitor, "kind": result.kind,
                "column": result.column, "metric": result.metric,
                "baseline": result.baseline, "status": result.status,
            }))
    return {**state, "findings": findings}


def classify_findings(state: AgentState) -> AgentState:
    classified = []
    for finding in state["findings"]:
        severity, reasoning = classifier.classify(finding)
        classified.append({"finding": finding, "severity": severity, "reasoning": reasoning})
    return {**state, "classified": classified}


def act(state: AgentState) -> AgentState:
    taken: list[str] = []
    for item in state["classified"]:
        if item["severity"] not in ACTIONABLE:
            continue
        path = actions.write_incident(
            item["finding"], item["severity"], item["reasoning"], state["as_of"])
        taken.append(f"incident: {path.name}")
        taken.append(actions.file_github_issue(
            item["finding"], item["severity"], item["reasoning"],
            execute=not state["dry_run"]))
    return {**state, "actions_taken": taken}


def _should_act(state: AgentState) -> str:
    return "act" if any(c["severity"] in ACTIONABLE for c in state["classified"]) else "end"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("sense", sense_and_detect)
    builder.add_node("monitor", run_monitors)
    builder.add_node("classify", classify_findings)
    builder.add_node("act", act)
    builder.set_entry_point("sense")
    builder.add_edge("sense", "monitor")
    builder.add_edge("monitor", "classify")
    builder.add_conditional_edges("classify", _should_act, {"act": "act", "end": END})
    builder.add_edge("act", END)
    return builder.compile()


def run(engine=None, dry_run: bool = True, as_of: date | None = None) -> AgentState:
    if engine is None:
        from src.lakehouse.engines import get_engine
        engine = get_engine()
    if as_of is None:
        from src.lakehouse.bronze import max_txn_date
        as_of = config.resolve_as_of_date(max_txn_date(engine))
    result = build_graph().invoke({
        "engine": engine, "as_of": as_of, "dry_run": dry_run,
        "findings": [], "classified": [], "actions_taken": [],
    })
    return result


def main() -> int:
    execute = "--execute" in sys.argv
    state = run(dry_run=not execute)
    print(f"agent: as_of={state['as_of']} findings={len(state['findings'])}")
    for item in state["classified"]:
        print(f"  [{item['severity']}] {item['finding'].detail}")
    for action in state.get("actions_taken", []):
        print(f"  -> {action}")
    if not execute:
        print("agent: dry-run (pass --execute to file GitHub issues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
