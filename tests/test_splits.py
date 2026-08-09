from datetime import date, timedelta

import pytest

from src.forecast import splits


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def test_produces_requested_number_of_folds():
    folds = splits.purged_walk_forward(_dates(600), n_folds=5)
    assert len(folds) == 5


def test_train_always_precedes_test():
    for fold in splits.purged_walk_forward(_dates(600), n_folds=5):
        assert max(fold.train) < min(fold.test)


def test_purge_gap_is_at_least_horizon():
    """Without this gap, a training row's 5-day target overlaps the test block."""
    for fold in splits.purged_walk_forward(_dates(600), n_folds=5, purge=5):
        assert min(fold.test) - max(fold.train) > 5


def test_training_window_expands():
    folds = splits.purged_walk_forward(_dates(600), n_folds=5)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_test_blocks_are_disjoint_and_ordered():
    folds = splits.purged_walk_forward(_dates(600), n_folds=5)
    seen: set[int] = set()
    last_end = -1
    for fold in folds:
        assert not (seen & set(fold.test))
        seen |= set(fold.test)
        assert min(fold.test) > last_end
        last_end = max(fold.test)


def test_raises_when_series_too_short_rather_than_silently_degrading():
    with pytest.raises(ValueError):
        splits.purged_walk_forward(_dates(50), n_folds=5, min_train=120)


def test_raises_on_min_test_boundary():
    """Boundary: n=135 produces test_size=2 and must raise. n=626 is safe."""
    # n=135 produces test_size=2, which is below min_test=20, must raise
    with pytest.raises(ValueError, match="min_test"):
        splits.purged_walk_forward(_dates(135), n_folds=5, min_train=120, purge=5)

    # Full-scale case: n=626 produces test_size=100, well above minimum
    folds_626 = splits.purged_walk_forward(_dates(626), n_folds=5, min_train=120)
    assert len(folds_626) == 5
    assert all(len(f.test) >= 20 for f in folds_626)

    # Task 12 synthetic case: n=500 produces test_size=75
    folds_500 = splits.purged_walk_forward(_dates(500), n_folds=5, min_train=120)
    assert len(folds_500) == 5
    assert all(len(f.test) >= 20 for f in folds_500)
