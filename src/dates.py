"""Defensive date parsing for raw, uncoerced source values.

Bronze is source-faithful and does no type coercion, so a date column there
holds whatever arrived -- including values that are not dates at all. Asking
SQL for `MAX()` over those raw strings and parsing the answer is actively
dangerous: a garbage value that sorts lexicographically above every real
date (`"zzz-not-a-date"` sorts after any `"20XX-..."` string) becomes the
`MAX()`, and parsing it raises. That turns a data-quality problem into a
crashed process, which is the worst failure mode available -- the pipeline
dies exactly when the data goes bad, and reports a stack trace instead of a
finding.

`max_parseable_date` is the one implementation of the safe pattern. It parses
every candidate, keeps the max of those that parse, and counts the rest so a
caller can surface them as their own signal rather than swallowing them.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date


def to_date(value) -> date:
    """Coerce one raw value to a `date`. Raises on anything that is not one."""
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value.date() if hasattr(value, "date") else value


def max_parseable_date(values: Iterable) -> tuple[date | None, int]:
    """`(newest parseable date or None, count of non-null values that did not parse)`.

    Never raises. Nulls are skipped and not counted as unparseable -- a NULL
    is an absent value, not a corrupt one.
    """
    newest: date | None = None
    unparseable = 0
    for value in values:
        if value is None:
            continue
        try:
            parsed = to_date(value)
        except (ValueError, TypeError):
            unparseable += 1
            continue
        if not isinstance(parsed, date):
            unparseable += 1
            continue
        if newest is None or parsed > newest:
            newest = parsed
    return newest, unparseable
