"""Rename pairing by field ID, widening as its own class, and enum drift.

The rename tests are the point of the file. Iceberg's whole claim is that a
rename is metadata-only because the field ID does not move; an agent that
reports it as an unrelated drop plus an unrelated add has not understood the
table it is watching, and would send a human looking for lost data that was
never lost.
"""
from datetime import date

import pyarrow as pa
import pytest
from pyiceberg.types import IntegerType, LongType

from src.agent import classifier, sensors
from src.contracts import validator
from src.contracts.validator import Contract, SchemaField
from src.lakehouse import bronze, maintenance, schemas
from src.lakehouse import catalog as catalog_mod
from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine


def _contract(fields, enum_watch=()):
    """`enum_watch` is deliberately NOT an `expectation`. Expectations are
    fail-closed gates evaluated by `validator.validate()`; an enum watch is a
    recorded observation that must never block a build."""
    return Contract(
        table="bronze.yodlee_transactions_raw", version=1, owner="x",
        schema_fields=tuple(SchemaField(n, t) for n, t in fields),
        expectations=(),
        enum_watch=tuple(enum_watch))


def _state(**kw):
    base = {"table": "bronze.yodlee_transactions_raw",
            "columns": {"id": "long", "baseType": "string"},
            "row_count": 1000, "rows_in_latest_snapshot": 1000,
            "snapshot_count": 2, "newest_date": date(2026, 6, 30)}
    base.update(kw)
    return sensors.ObservedState(**base)


# --------------------------------------------------------------------------
# Rename pairing
# --------------------------------------------------------------------------

def test_rename_is_one_finding_not_a_drop_plus_an_add():
    contract = _contract([("id", "long"), ("checkNumber", "string")])
    state = _state(
        columns={"id": "long", "check_reference": "string"},
        field_ids={"id": 1, "check_reference": 20},
        renames={"checkNumber": ("check_reference", 20)})

    findings = sensors.detect(state, contract, date(2026, 6, 30), history=[])
    drift = [f for f in findings if f.kind == "schema_drift"]

    assert len(drift) == 1, f"expected one finding, got {[f.detail for f in drift]}"
    assert drift[0].evidence["change"] == "renamed"
    assert drift[0].evidence["column"] == "checkNumber"
    assert drift[0].evidence["renamed_to"] == "check_reference"
    assert drift[0].evidence["field_id"] == 20
    assert classifier.classify(drift[0])[0] == "renaming"


def test_rename_reasoning_cites_the_field_id():
    """The field ID is the evidence. A rename report that omits it is an
    assertion; one that carries it can be checked against the table."""
    contract = _contract([("checkNumber", "string")])
    state = _state(columns={"check_reference": "string"},
                   field_ids={"check_reference": 20},
                   renames={"checkNumber": ("check_reference", 20)})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    _, reasoning = classifier.classify(drift)
    assert "20" in reasoning
    assert "check_reference" in reasoning


def test_a_genuine_drop_is_still_breaking_and_not_mistaken_for_a_rename():
    contract = _contract([("id", "long"), ("checkNumber", "string")])
    state = _state(columns={"id": "long"}, field_ids={"id": 1}, renames={})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    assert drift.evidence["change"] == "dropped"
    assert classifier.classify(drift)[0] == "breaking"


def test_rename_pairing_does_not_capture_an_unrelated_added_column():
    """Mutation guard. Rename AND add in the same schema version: pairing by
    name-similarity or by position would swallow the added column into the
    rename. Only the field ID distinguishes them."""
    contract = _contract([("id", "long"), ("checkNumber", "string")])
    state = _state(
        columns={"id": "long", "check_reference": "string",
                 "merchantCategoryCode": "string"},
        field_ids={"id": 1, "check_reference": 20, "merchantCategoryCode": 40},
        renames={"checkNumber": ("check_reference", 20)})

    drift = [f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
             if f.kind == "schema_drift"]
    changes = {f.evidence["change"]: f.evidence for f in drift}

    assert set(changes) == {"renamed", "added"}
    assert changes["renamed"]["renamed_to"] == "check_reference"
    assert changes["added"]["column"] == "merchantCategoryCode"


def test_rename_map_pairs_by_field_id_not_by_ordinal_position():
    """The discriminating case, and the reason this is a unit test on
    `build_rename_map` rather than on a hand-supplied `renames` dict.

    A rename alone does not move a column's position, so position-based
    pairing gets the easy cases right and looks correct. It breaks the moment
    a column is *dropped* from the middle: every later column shifts up one,
    and ordinal matching then pairs the dropped column's old name with its
    neighbour's new name -- inventing a rename that never happened and hiding
    a drop that did.

    history: {a:1, b:2, c:3} -> {a:1, c_new:3}   (b dropped, c renamed)
    by position: [a,b,c] vs [a,c_new] pairs b -> c_new. Wrong, twice over.
    by field id: id 2 is gone (a real drop); id 3 changed name (the rename).
    """
    history = [
        {"schema_id": 0, "columns": {"a": "long", "b": "string", "c": "string"},
         "field_ids": {"a": 1, "b": 2, "c": 3}},
        {"schema_id": 1, "columns": {"a": "long", "c_new": "string"},
         "field_ids": {"a": 1, "c_new": 3}},
    ]

    renames = sensors.build_rename_map(history)

    assert renames == {"c": ("c_new", 3)}
    assert "b" not in renames, "a dropped column must never be reported as renamed"


def test_dropped_column_is_absent_from_the_rename_map_end_to_end():
    """The consequence of the above: `b` must still be classified breaking."""
    history = [
        {"schema_id": 0, "columns": {"a": "long", "b": "string", "c": "string"},
         "field_ids": {"a": 1, "b": 2, "c": 3}},
        {"schema_id": 1, "columns": {"a": "long", "c_new": "string"},
         "field_ids": {"a": 1, "c_new": 3}},
    ]
    contract = _contract([("a", "long"), ("b", "string"), ("c", "string")])
    state = _state(columns={"a": "long", "c_new": "string"},
                   field_ids={"a": 1, "c_new": 3},
                   renames=sensors.build_rename_map(history))

    by_change = {f.evidence["change"]: f.evidence
                 for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift"}

    assert by_change["dropped"]["column"] == "b"
    assert by_change["renamed"]["column"] == "c"
    assert by_change["renamed"]["renamed_to"] == "c_new"


def test_transitive_rename_resolves_to_the_current_name():
    """a -> b -> c. A contract still declaring `a` must be told about `c`,
    not about a `b` that no longer exists either."""
    contract = _contract([("a", "string")])
    state = _state(columns={"c": "string"}, field_ids={"c": 7},
                   renames={"a": ("c", 7), "b": ("c", 7)})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    assert drift.evidence["renamed_to"] == "c"


def test_rename_that_also_narrows_the_type_is_breaking():
    """A rename is benign because no value moves. A rename that also narrows
    is not, and must not inherit the rename's calmer severity."""
    contract = _contract([("n", "long")])
    state = _state(columns={"n2": "int"}, field_ids={"n2": 9},
                   renames={"n": ("n2", 9)})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    assert drift.evidence["change"] == "renamed"
    assert drift.evidence["type_compatible"] is False
    assert classifier.classify(drift)[0] == "breaking"


# --------------------------------------------------------------------------
# Widening as its own class
# --------------------------------------------------------------------------

def test_widening_is_classified_widening_not_additive():
    contract = _contract([("settlementDays", "int")])
    state = _state(columns={"settlementDays": "long"},
                   field_ids={"settlementDays": 41}, renames={})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    severity, reasoning = classifier.classify(drift)
    assert severity == "widening"
    assert "int" in reasoning and "long" in reasoning


def test_added_column_is_still_additive():
    """Widening getting its own name must not drag `added` along with it."""
    contract = _contract([("id", "long")])
    state = _state(columns={"id": "long", "newCol": "string"},
                   field_ids={"id": 1, "newCol": 40}, renames={})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    assert classifier.classify(drift)[0] == "additive"


def test_narrowing_is_still_breaking():
    contract = _contract([("n", "long")])
    state = _state(columns={"n": "int"}, field_ids={"n": 1}, renames={})
    drift = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                 if f.kind == "schema_drift")
    assert classifier.classify(drift)[0] == "breaking"


def test_widening_is_not_actionable_but_renaming_is():
    """The severity split has to change what the agent *does*, or it is a
    label. A widening needs no human; a stale contract does."""
    from src.agent import graph
    assert "widening" not in graph.ACTIONABLE
    assert "additive" not in graph.ACTIONABLE
    assert "enum_drift" not in graph.ACTIONABLE
    assert "breaking" in graph.ACTIONABLE
    assert "renaming" in graph.ACTIONABLE


# --------------------------------------------------------------------------
# Enum drift
# --------------------------------------------------------------------------

def test_enum_drift_reports_the_new_value_and_does_not_block():
    contract = _contract(
        [("category", "string")],
        [{"column": "category", "known_values": ["Groceries", "Gas"]}])
    state = _state(columns={"category": "string"}, field_ids={"category": 12},
                   renames={},
                   enum_values={"category": ["Gas", "Groceries", "Crypto"]})

    findings = sensors.detect(state, contract, date(2026, 6, 30), history=[])
    drift = [f for f in findings if f.kind == "enum_drift"]

    assert len(drift) == 1
    assert drift[0].evidence["new_values"] == ["Crypto"]
    assert drift[0].evidence["column"] == "category"
    severity, reasoning = classifier.classify(drift[0])
    assert severity == "enum_drift"
    assert "Crypto" in reasoning


def test_no_enum_finding_when_every_value_is_known():
    contract = _contract(
        [("category", "string")],
        [{"column": "category", "known_values": ["Groceries", "Gas"]}])
    state = _state(columns={"category": "string"}, field_ids={"category": 12},
                   renames={}, enum_values={"category": ["Gas"]})
    assert not [f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                if f.kind == "enum_drift"]


def test_enum_drift_on_an_absent_column_is_reported_not_silently_skipped():
    """The empty-delta failure mode: a watch on a column that isn't there must
    not read as 'no drift'."""
    contract = _contract(
        [("category", "string")],
        [{"column": "gone", "known_values": ["a"]}])
    state = _state(columns={"category": "string"}, field_ids={"category": 12},
                   renames={}, enum_values={})
    findings = [f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                if f.kind == "enum_drift"]
    assert len(findings) == 1
    assert findings[0].evidence.get("error") == "column_absent"


def test_high_cardinality_column_is_capped_rather_than_dumped():
    """A watch mistakenly pointed at a high-cardinality column must degrade to
    a bounded complaint, not a finding carrying thousands of values."""
    values = [f"v{i}" for i in range(sensors.ENUM_CARDINALITY_LIMIT + 50)]
    contract = _contract(
        [("merchant", "string")],
        [{"column": "merchant", "known_values": []}])
    state = _state(columns={"merchant": "string"}, field_ids={"merchant": 3},
                   renames={}, enum_values={"merchant": values})
    finding = next(f for f in sensors.detect(state, contract, date(2026, 6, 30), [])
                   if f.kind == "enum_drift")
    assert finding.evidence.get("error") == "cardinality_exceeded"
    assert len(finding.evidence.get("new_values", [])) <= sensors.ENUM_CARDINALITY_LIMIT


# --------------------------------------------------------------------------
# Against a real Iceberg table
# --------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    eng = PyIcebergEngine(catalog_mod.get_catalog(tmp_path / "wh"))
    bronze.ingest_all(eng, n_transactions=500)
    return eng


def test_rename_map_is_built_from_real_iceberg_schema_history(engine):
    """Not a fixture. The rename map must come out of the same metadata the
    README's `make schema-history` prints."""
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")

    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")

    assert state.renames.get("checkNumber") == ("check_reference", 20)
    assert state.field_ids["check_reference"] == 20


def test_real_rename_produces_one_renaming_finding_end_to_end(engine):
    """The scenario from the README: the shipped contract still says
    `checkNumber`, the table says `check_reference`, and they are field 20."""
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")

    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    findings = sensors.detect(state, contract, date(2026, 6, 30),
                              history=[state.rows_in_latest_snapshot])

    about_the_rename = [
        f for f in findings
        if f.evidence.get("column") == "checkNumber"
        or f.evidence.get("column") == "check_reference"]
    assert len(about_the_rename) == 1
    assert about_the_rename[0].evidence["change"] == "renamed"
    assert about_the_rename[0].evidence["field_id"] == 20
    assert classifier.classify(about_the_rename[0])[0] == "renaming"


def test_real_widening_classifies_as_widening(engine):
    """drift-demo's v3 -> v4 step, against the real table."""
    maintenance.evolve_add_column(engine, "settlementDays", IntegerType())
    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    fields = contract.schema_fields + (SchemaField("settlementDays", "int"),)
    contract = Contract(contract.table, contract.version, contract.owner,
                        fields, contract.expectations, contract.enum_watch)

    assert maintenance.evolve_widen_column(engine, "settlementDays", LongType())
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")
    findings = sensors.detect(state, contract, date(2026, 6, 30),
                              history=[state.rows_in_latest_snapshot])

    widened = next(f for f in findings
                   if f.evidence.get("column") == "settlementDays")
    assert widened.evidence["change"] == "type_changed"
    assert classifier.classify(widened)[0] == "widening"


def test_enum_values_are_collected_from_a_real_scan(engine):
    """`observe` must read real distinct values, not be handed them."""
    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS,
                            "transactionDate", enum_columns=("baseType",))
    assert set(state.enum_values["baseType"]) <= {"CREDIT", "DEBIT"}
    assert state.enum_values["baseType"]


def test_enum_drift_fires_on_a_real_injected_value(engine):
    """An unregistered category value appears in the feed. It must be
    surfaced as its own class, and it must not stop the pipeline."""
    ident = schemas.BRONZE_TRANSACTIONS.name
    table = engine.scan_arrow(ident)
    values = table.column("category").to_pylist()
    values[0] = "Crypto"
    idx = table.schema.get_field_index("category")
    engine.overwrite(ident, table.set_column(
        idx, "category", pa.array(values, type=table.schema.field("category").type)))

    known = sorted({v for v in values[1:] if v is not None})
    contract = _contract(
        [("category", "string")],
        [{"column": "category", "known_values": known}])

    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS,
                            "transactionDate", enum_columns=("category",))
    findings = sensors.detect(state, contract, date(2026, 6, 30), history=[])

    drift = [f for f in findings if f.kind == "enum_drift"]
    assert len(drift) == 1
    assert drift[0].evidence["new_values"] == ["Crypto"]
    assert classifier.classify(drift[0])[0] == "enum_drift"


def test_rename_map_survives_a_real_drop_and_rename_in_one_step(engine):
    """The position/field-ID divergence, against real Iceberg metadata rather
    than a fixture. `detailCategoryId` is deleted and `checkNumber` renamed;
    ordinal pairing would report the deleted column as renamed into its
    neighbour and lose the real drop entirely."""
    table = engine.catalog.load_table(schemas.BRONZE_TRANSACTIONS.name)
    with table.update_schema() as update:
        update.delete_column("detailCategoryId")
    maintenance.evolve_rename_column(engine, "checkNumber", "check_reference")

    state = sensors.observe(engine, schemas.BRONZE_TRANSACTIONS, "transactionDate")

    assert state.renames.get("checkNumber") == ("check_reference", 20)
    assert "detailCategoryId" not in state.renames

    contract = validator.load_contract(
        validator.CONTRACTS_DIR / "bronze_yodlee_transactions.yaml")
    findings = sensors.detect(state, contract, date(2026, 6, 30), history=[])
    by_column = {f.evidence.get("column"): f.evidence
                 for f in findings if f.kind == "schema_drift"}

    assert by_column["checkNumber"]["change"] == "renamed"
    assert by_column["detailCategoryId"]["change"] == "dropped"
    assert classifier.classify(
        next(f for f in findings
             if f.evidence.get("column") == "detailCategoryId"))[0] == "breaking"
