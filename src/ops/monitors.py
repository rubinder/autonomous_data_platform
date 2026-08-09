"""Monitor definitions and verdict rules.

A monitor is a SQL query returning one column named `metric`, plus a rule for
judging that number against its own history. Keeping the query in YAML means
adding a check is a data change, not a code change -- which is what makes it
plausible that an on-call engineer would actually add one.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from src import config

MONITOR_DIR = config.REPO_ROOT / "monitors"


@dataclass(frozen=True)
class MonitorDef:
    name: str
    table: str
    kind: str
    query: str
    column: str | None = None
    severity: str = "additive"
    params: dict | None = None


@dataclass(frozen=True)
class MonitorResult:
    monitor: str
    table: str
    column: str | None
    kind: str
    metric: float | None
    baseline: float | None
    status: str          # ok | warn | breach
    detail: str
    run_at: date


def load_monitors(path: Path | None = None) -> list[MonitorDef]:
    directory = Path(path or MONITOR_DIR)
    defs: list[MonitorDef] = []
    for file in sorted(directory.glob("*.yaml")):
        for raw in yaml.safe_load(file.read_text()) or []:
            defs.append(MonitorDef(
                name=raw["name"], table=raw["table"], kind=raw["kind"],
                query=raw["query"], column=raw.get("column"),
                severity=raw.get("severity", "additive"),
                params=raw.get("params") or {}))
    return defs


def evaluate(metric: float, baseline: list[float],
             params: dict) -> tuple[str, str]:
    """Judge a metric against its own history. Returns (status, detail)."""
    kind = params.get("kind", "row_count")

    absolute = params.get("max_absolute")
    if absolute is not None and metric > absolute:
        return "breach", f"{metric:g} exceeds absolute limit {absolute:g}"

    if not baseline:
        # First run. Alerting with nothing to compare against produces noise
        # on every new monitor, which is how monitoring gets muted -- a
        # monitor that alarms on its own first execution never gets trusted.
        return "ok", "no baseline yet — recorded for future comparison"

    median = statistics.median(baseline)

    if kind in {"row_count", "cardinality"}:
        if median <= 0:
            return "ok", "baseline median is zero"
        ratio = metric / median
        if ratio < params.get("breach_ratio", 0.5):
            return "breach", (f"{metric:g} is {ratio:.0%} of trailing "
                              f"median {median:g}")
        if ratio < params.get("warn_ratio", 0.8):
            return "warn", (f"{metric:g} is {ratio:.0%} of trailing "
                            f"median {median:g}")
        return "ok", f"{metric:g} vs median {median:g}"

    if kind in {"distribution_shift", "null_rate"}:
        # Median absolute deviation, not stddev: one prior outlier widens a
        # stddev band enough to hide the next one -- precisely when the
        # check needs to fire.
        deviations = [abs(v - median) for v in baseline]
        mad = statistics.median(deviations)
        if mad == 0:
            # A tied median absolute deviation (e.g. most of history is
            # identical with a couple of near-median outliers) does not mean
            # the baseline has zero spread -- fall back to the mean absolute
            # deviation, which is nonzero whenever *any* variability exists.
            mad = statistics.mean(deviations)
        if mad == 0:
            # Baseline has been genuinely constant with no variability at
            # all. There is no scale to measure a z-score against, so judge
            # by near-equality rather than exact equality: a metric
            # recomputed from unchanged data still round-trips through
            # Iceberg/Parquet doubles with float noise in the last couple of
            # digits (observed: 171.4137026799999 vs 171.41370268000063), so
            # `==` would flag an unchanged table as a breach on every run.
            if math.isclose(metric, median, rel_tol=1e-9, abs_tol=1e-9):
                return "ok", f"{metric:g} matches constant baseline {median:g}"
            return "breach", (f"{metric:g} differs from constant baseline "
                              f"{median:g}")
        robust_z = abs(metric - median) / (1.4826 * mad)
        if robust_z > params.get("breach_z", 4.0):
            return "breach", (f"{metric:g} is {robust_z:.1f} robust-z from "
                              f"median {median:g}")
        if robust_z > params.get("warn_z", 3.0):
            return "warn", (f"{metric:g} is {robust_z:.1f} robust-z from "
                            f"median {median:g}")
        return "ok", f"{metric:g} within {robust_z:.1f} robust-z"

    if kind == "duplicate_rate":
        return ("ok", f"{metric:g} duplicates") if metric == 0 else (
            "breach", f"{metric:g} duplicate key(s)")

    return "ok", f"{metric:g}"


# Iceberg declared type -> the Arrow types that legitimately represent it.
# PyIceberg's StringType() maps to Arrow `large_string`, not `string`, so
# both have to be accepted for "string" to mean anything useful here.
_TYPE_MAP: dict[str, tuple[str, ...]] = {
    "long": ("int64",),
    "int": ("int32", "int64"),
    "double": ("double", "float"),
    "string": ("string", "large_string"),
    "boolean": ("bool",),
    "date": ("date32[day]",),
    "timestamptz": ("timestamp[us, tz=UTC]", "timestamp[us, tz=+00:00]"),
    "struct": ("struct",),
}


def check_column_types(engine, table_def, contract) -> list[MonitorResult]:
    """Physical Arrow types vs the contract's declared types.

    Catches silent coercion -- a column that arrives as string where the
    contract declares double still passes every value-level expectation while
    breaking every arithmetic consumer downstream.
    """
    arrow = engine.scan_arrow(table_def.name)
    actual = {f.name: str(f.type) for f in arrow.schema}
    run_at = config.resolve_as_of_date(None)
    results: list[MonitorResult] = []

    for field in contract.schema_fields:
        observed = actual.get(field.name)
        if observed is None:
            results.append(MonitorResult(
                f"{table_def.name}_type_{field.name}", table_def.name,
                field.name, "column_type", None, None, "breach",
                f"declared column '{field.name}' is absent", run_at))
            continue

        allowed = _TYPE_MAP.get(field.type, (field.type,))
        ok = any(observed == a or observed.startswith(a) for a in allowed)
        results.append(MonitorResult(
            f"{table_def.name}_type_{field.name}", table_def.name,
            field.name, "column_type", None, None,
            "ok" if ok else "breach",
            f"declared {field.type}, observed {observed}", run_at))

    return results
