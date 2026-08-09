from __future__ import annotations

import os

from src.lakehouse.engines.base import LakehouseEngine


def get_engine(name: str | None = None) -> LakehouseEngine:
    choice = (name or os.environ.get("ENGINE") or "pyiceberg").lower()
    if choice == "pyiceberg":
        from src.lakehouse.catalog import get_catalog
        from src.lakehouse.engines.pyiceberg_engine import PyIcebergEngine
        return PyIcebergEngine(get_catalog())
    if choice == "spark":
        from src.lakehouse.engines.spark_engine import SparkEngine
        return SparkEngine()
    raise ValueError(f"unknown ENGINE: {choice}")


__all__ = ["LakehouseEngine", "get_engine"]
