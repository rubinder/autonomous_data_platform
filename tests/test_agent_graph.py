from datetime import date

import duckdb
import pytest

from src.agent import actions, graph
from src.agent.sensors import Finding
from src.contracts import validator
from src.lakehouse import bronze
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


@pytest.fixture(autouse=True)
def isolate_incidents(tmp_path, monkeypatch):
    monkeypatch.setattr(actions, "INCIDENTS_DIR", tmp_path / "incidents")


def _finding(kind="schema_drift", column="merchantCategoryCode", change="added"):
    return Finding(kind, "bronze.yodlee_transactions_raw",
                   f"column '{column}' {change}",
                   {"change": change, "column": column, "observed_type": "string"})


def test_incident_file_written_with_severity_and_reasoning():
    path = actions.write_incident(_finding(), "additive", "New column.", date(2026, 6, 30))
    text = path.read_text()
    assert path.exists()
    assert "additive" in text
    assert "New column." in text
    assert "bronze.yodlee_transactions_raw" in text


def test_incident_slug_is_stable_so_reruns_do_not_spam():
    a = actions.incident_slug(_finding())
    b = actions.incident_slug(_finding())
    assert a == b
    assert a != actions.incident_slug(_finding(column="other"))


def test_rerun_does_not_create_a_second_file_for_the_same_finding():
    f = _finding()
    p1 = actions.write_incident(f, "additive", "x", date(2026, 6, 30))
    p2 = actions.write_incident(f, "additive", "x", date(2026, 6, 30))
    assert p1 == p2
    assert len(list(actions.INCIDENTS_DIR.glob("*.md"))) == 1


def test_github_issue_is_dry_run_by_default():
    out = actions.file_github_issue(_finding(), "breaking", "why", execute=False)
    assert out.startswith("DRY-RUN")
    assert "gh issue create" in out


class _FakeEngine:
    """Minimal engine stub so the graph test needs no warehouse and no network.

    `columns` is seeded from the real, fixed bronze contract so a clean
    engine (drop_column=False) reports zero schema-drift findings -- the
    same "declared == observed" baseline the real lakehouse should hit.
    Dropping a single declared column (drop_column=True) is the one
    controlled deviation the "breaking findings act" test exercises.
    """

    def __init__(self, drop_column: bool = False):
        contract = validator.load_contract(
            validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
        self._columns = {f.name: f.type for f in contract.schema_fields}
        self._enum_watch = contract.enum_watch
        if drop_column:
            self._columns.pop("baseType", None)
        # Recording rather than no-op: the persist node's whole job is what it
        # writes, so a stub that silently swallowed the writes would let the
        # node be deleted with every test still green.
        self.created: list[str] = []
        self.appended: dict[str, list[dict]] = {}

    def create_table(self, table_def):
        self.created.append(table_def.name)

    def append(self, ident, data):
        self.appended.setdefault(ident, []).extend(data.to_pylist())

    def scan_arrow(self, ident, snapshot_id=None):
        import pyarrow as pa
        columns = {
            "id": pa.array([1, 2], pa.int64()),
            "transactionDate": pa.array(["2026-06-29", "2026-06-30"]),
        }
        # Watched enum columns are seeded from the contract's own registered
        # values, for the same reason `_columns` is seeded from its `schema`:
        # the fake stays clean *by construction*, so adding a watch to the
        # contract cannot silently turn this stub into a table that is
        # missing a monitored column. It is not silent, as it happens --
        # `sensors._detect_enum_drift` reports a watch that read no values as
        # `breaking` rather than passing it -- which is how this fixture's
        # gap surfaced in the first place.
        for watch in self._enum_watch:
            known = list(watch.get("known_values", ()))
            if watch["column"] in self._columns and known:
                columns[watch["column"]] = pa.array([known[0], known[0]])
        return pa.table(columns)

    def snapshots(self, ident):
        return [1, 2, 3]

    def snapshot_row_counts(self, ident):
        return [2, 2, 2]

    def snapshot_details(self, ident):
        return [{"snapshot_id": i, "parent_snapshot_id": None, "operation": "append",
                  "added_records": 2, "deleted_records": 0, "total_records": 2 * i,
                  "is_full_rebuild": False} for i in (1, 2, 3)]

    def schema_history(self, ident):
        return [{"schema_id": 0, "columns": self._columns, "field_ids": {}}]

    def table_exists(self, ident):
        # Only the one table this fake actually models: Task 19's monitor
        # node calls `runner.run_monitors` / `arrival.check_arrival` for
        # Silver, Gold and `ops.*` tables too, and those must read as
        # "not built yet" here rather than as an existing table whose data
        # happens not to match -- `scan_arrow` below only ever returns
        # Bronze-transaction-shaped columns.
        return ident == "bronze.yodlee_transactions_raw"

    def sql(self, query, tables):
        con = duckdb.connect()
        try:
            for alias, ident in tables.items():
                con.register(alias, self.scan_arrow(ident))
            return con.execute(query).to_arrow_table()
        finally:
            con.close()


def test_graph_runs_offline_and_returns_state(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = graph.run(engine=_FakeEngine(), dry_run=True, as_of=date(2026, 6, 30))
    assert "findings" in state and "classified" in state
    assert isinstance(state["actions_taken"], list)
    assert state["findings"] == []
    assert state["actions_taken"] == []


def test_breaking_findings_produce_actions_benign_do_not(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = graph.run(engine=_FakeEngine(drop_column=True), dry_run=True,
                       as_of=date(2026, 6, 30))
    severities = {c["severity"] for c in state["classified"]}
    assert "breaking" in severities
    assert state["actions_taken"]
    # A dry run must never shell out -- every action string says so.
    assert all("DRY-RUN" in a or a.startswith("incident:") for a in state["actions_taken"])


def test_clean_lakehouse_produces_no_findings(tmp_path, monkeypatch):
    """The baseline that makes every real finding meaningful.

    A freshly built, unmodified Bronze table checked against its own
    contract at as_of == the data's max date must report zero findings.
    Regression guard for the bronze_yodlee_transactions.yaml contract
    defect: an incomplete `schema:` block flagged every undeclared column
    as spurious `additive` schema_drift noise on every single run.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    engine = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(engine, n_transactions=2000)
    as_of = bronze.max_txn_date(engine)

    state = graph.run(engine=engine, dry_run=True, as_of=as_of)

    assert state["findings"] == []
    assert state["classified"] == []
    assert state["actions_taken"] == []
