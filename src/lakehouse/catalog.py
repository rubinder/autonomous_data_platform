"""PyIceberg SqlCatalog on SQLite. No external services -- `make all` works offline."""
from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError

from src import config

NAMESPACES: tuple[str, ...] = ("bronze", "silver", "gold")


def get_catalog(warehouse: Path | None = None) -> SqlCatalog:
    """Open (creating if needed) the local Iceberg catalog under `warehouse`.

    Defaults to `config.WAREHOUSE_PATH`. The catalog database and the data files
    live side by side so the whole warehouse is one deletable directory.
    """
    wh = Path(warehouse or config.WAREHOUSE_PATH)
    wh.mkdir(parents=True, exist_ok=True)
    cat = SqlCatalog(
        "local",
        uri=f"sqlite:///{wh / 'catalog.db'}",
        warehouse=f"file://{wh}",
    )
    for ns in NAMESPACES:
        try:
            cat.create_namespace(ns)
        except NamespaceAlreadyExistsError:
            pass
    return cat
