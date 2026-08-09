"""Purged, expanding-window walk-forward CV.

The purge gap is the reason this module exists. With a 5-day forward target,
a training row dated t-1 has a target that resolves at t+4 -- inside the test
block. Without purging, the model trains on the answer and every metric is
inflated in a way that ordinary shuffled CV will never reveal.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src import config


@dataclass(frozen=True)
class Fold:
    index: int
    train: list[int]
    test: list[int]


def purged_walk_forward(dates: Sequence[date], n_folds: int = 5,
                        purge: int = config.PURGE_DAYS,
                        min_train: int = 120) -> list[Fold]:
    n = len(dates)
    usable = n - min_train - purge
    if usable < n_folds * 2:
        raise ValueError(
            f"series of {n} points cannot support {n_folds} folds with "
            f"min_train={min_train} and purge={purge}")

    test_size = usable // n_folds
    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = min_train + purge + i * test_size
        test_end = test_start + test_size if i < n_folds - 1 else n
        train_end = test_start - purge
        folds.append(Fold(
            index=i,
            train=list(range(train_end)),
            test=list(range(test_start, test_end)),
        ))
    return folds
