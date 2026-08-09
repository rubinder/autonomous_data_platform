from datetime import date

import pyarrow as pa
import pytest

from src.contracts import validator


def _table(ids=(1, 2, 3), amounts=(-5.0, 10.0, -2.5), dates=None):
    dates = dates or [date(2026, 6, 28), date(2026, 6, 29), date(2026, 6, 30)]
    return pa.table({
        "id": pa.array(ids, pa.int64()),
        "txn_date": pa.array(dates, pa.date32()),
        "signed_amount": pa.array(amounts, pa.float64()),
        "base_type": pa.array(["DEBIT", "CREDIT", "DEBIT"]),
    })


@pytest.fixture
def contract():
    return validator.load_contract(validator.CONTRACTS_DIR / "silver_transactions.yaml")


def test_valid_table_passes(contract):
    report = validator.validate(_table(), contract, as_of=date(2026, 6, 30))
    assert report.passed, report.failures


def test_null_in_non_nullable_column_fails(contract):
    tbl = pa.table({
        "id": pa.array([1, None], pa.int64()),
        "txn_date": pa.array([date(2026, 6, 30)] * 2, pa.date32()),
        "signed_amount": pa.array([1.0, 2.0], pa.float64()),
        "base_type": pa.array(["DEBIT", "DEBIT"]),
    })
    report = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert not report.passed
    assert any(f.kind == "not_null" and f.column == "id" for f in report.failures)


def test_duplicate_ids_fail(contract):
    report = validator.validate(_table(ids=(1, 1, 2)), contract, as_of=date(2026, 6, 30))
    assert any(f.kind == "unique" for f in report.failures)


def test_unexpected_base_type_fails(contract):
    tbl = _table()
    tbl = tbl.set_column(tbl.schema.get_field_index("base_type"), "base_type",
                         pa.array(["DEBIT", "REFUND", "DEBIT"]))
    report = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert any(f.kind == "accepted_values" for f in report.failures)


def test_freshness_uses_as_of_not_wall_clock(contract):
    """Data ends 2026-06-30. Against wall clock this check would fail forever."""
    tbl = _table()
    fresh = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert not any(f.kind == "freshness" for f in fresh.failures)
    stale = validator.validate(tbl, contract, as_of=date(2026, 8, 9))
    assert any(f.kind == "freshness" for f in stale.failures)


def test_sign_inversion_is_caught(contract):
    """The exact bug Task 8 injects: DEBIT rows must be negative."""
    unsigned = _table(amounts=(5.0, 10.0, 2.5))  # all positive, DEBIT included
    report = validator.validate(unsigned, contract, as_of=date(2026, 6, 30))
    assert any(f.kind == "conditional_sign" for f in report.failures)


def test_assert_valid_raises_on_violation(contract):
    with pytest.raises(validator.ContractViolation) as exc:
        validator.assert_valid(_table(ids=(1, 1, 2)), contract, as_of=date(2026, 6, 30))
    assert "unique" in str(exc.value)


def test_nullable_false_is_enforced_without_an_explicit_not_null(contract):
    """`signed_amount` is `nullable: false` and has no `not_null` expectation.

    Before this, `nullable` was parsed into `SchemaField` and then never read,
    so a Silver table whose `signed_amount` was 100% NULL passed the entire
    contract: `not_null` was not declared for it and `range` treated "no
    values" as nothing to complain about.
    """
    field = next(f for f in contract.schema_fields if f.name == "signed_amount")
    assert field.nullable is False
    assert not any(e.get("type") == "not_null" and e.get("column") == "signed_amount"
                   for e in contract.expectations)

    tbl = _table()
    tbl = tbl.set_column(tbl.schema.get_field_index("signed_amount"),
                         "signed_amount", pa.array([None, None, None], pa.float64()))
    report = validator.validate(tbl, contract, as_of=date(2026, 6, 30))
    assert not report.passed
    assert any(f.kind == "not_null" and f.column == "signed_amount"
               for f in report.failures)


def test_all_null_column_fails_assert_valid(contract):
    """The end-to-end version: the build must actually abort, not just report."""
    tbl = _table()
    tbl = tbl.set_column(tbl.schema.get_field_index("signed_amount"),
                         "signed_amount", pa.array([None, None, None], pa.float64()))
    with pytest.raises(validator.ContractViolation):
        validator.assert_valid(tbl, contract, as_of=date(2026, 6, 30))


def test_range_on_an_all_null_column_fails_rather_than_passing_silently(contract):
    """A column with nothing in it is not a column that is in range.

    `freshness` already fails on "no values"; `range` used to return pass,
    which is the empty-delta blind spot in miniature -- reporting fine exactly
    when the data is most obviously broken.
    """
    range_only = validator.Contract(
        table="t", version=1, owner="test", schema_fields=(),
        expectations=({"type": "range", "column": "signed_amount",
                       "min": -100000, "max": 100000},))
    tbl = _table()
    tbl = tbl.set_column(tbl.schema.get_field_index("signed_amount"),
                         "signed_amount", pa.array([None, None, None], pa.float64()))
    report = validator.validate(tbl, range_only, as_of=date(2026, 6, 30))
    assert not report.passed
    failure = next(f for f in report.failures if f.kind == "range")
    assert "no values" in failure.detail


def test_report_records_observed_values(contract):
    report = validator.validate(_table(ids=(1, 1, 2)), contract, as_of=date(2026, 6, 30))
    dup = next(f for f in report.failures if f.kind == "unique")
    assert dup.observed is not None
