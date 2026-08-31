"""Per-feed declarative configuration: one YAML per feed, not one per layer.

A feed used to be spread across four places, three of them Python:
`monitors/<layer>.yaml`, `graph.py:WATCHED`, `arrival.py:ARRIVAL_SLAS` and
`runner.py:TYPED_TABLES`. Worse than the count was the *axis*: the config was
organised by layer, so adding one feed meant touching every layer file, and
the operational facts about a single feed were never visible in one place.

`feeds/<name>.yaml` inverts that. Everything about a feed except its Iceberg
schema -- which is genuinely code, since it declares field IDs and partition
transforms -- is declared in one file, and adding a feed is one new file.

Validation fails loudly and names the offending key. That is the same posture
`monitors.evaluate()` already takes towards an unrecognised `kind`, and for the
same reason: a config error that degrades to a default is a check that reports
green while measuring nothing, which this codebase has now been bitten by more
than once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src import config
from src.lakehouse import schemas
from src.ops.monitors import KINDS, MonitorDef

FEED_DIR = config.REPO_ROOT / "feeds"
CONTRACT_DIR = config.REPO_ROOT / "contracts"

VALID_INGEST_MODES = frozenset({"batch", "stream"})
VALID_CALENDARS = frozenset({"trading", "daily"})
VALID_FEED_TYPES = frozenset({"fact", "event", "dimension"})

# `min_rows_per_period: companies` resolves to the number of tracked tickers.
# Spelled symbolically rather than as a literal 6 so the SLA cannot silently
# disagree with `config.COMPANIES` when a ticker is added.
_SYMBOLIC_ROW_FLOORS = {"companies": lambda: len(config.COMPANIES)}


class FeedConfigError(Exception):
    """A feed config that cannot be trusted. Never degrades to a default."""


@dataclass(frozen=True)
class LayerDef:
    layer: str
    table: str
    contract: str | None = None
    freshness_column: str | None = None
    agent_watch: bool = False
    typed: bool = False


@dataclass(frozen=True)
class ArrivalDef:
    table: str
    column: str
    calendar: str
    max_lag_periods: int
    min_rows_per_period: int | None


@dataclass(frozen=True)
class FeedDef:
    name: str
    domain: str
    feed_type: str
    owner: str
    ingest_mode: str
    layers: dict[str, LayerDef]
    monitors: tuple[MonitorDef, ...]
    arrival: tuple[ArrivalDef, ...]

    def layer(self, name: str) -> LayerDef:
        return self.layers[name]

    @property
    def watched_layers(self) -> tuple[LayerDef, ...]:
        return tuple(a for a in self.layers.values() if a.agent_watch)

    @property
    def typed_layers(self) -> tuple[LayerDef, ...]:
        return tuple(a for a in self.layers.values() if a.typed)


def _require(raw: dict, key: str, where: str):
    if key not in raw:
        raise FeedConfigError(f"{where}: missing required key '{key}'")
    return raw[key]


def _table_defs() -> dict[str, schemas.TableDef]:
    return {t.name: t for t in schemas.ALL_TABLES}


def table_def_for(table: str) -> schemas.TableDef:
    """The Iceberg `TableDef` a feed layer names, or a loud failure."""
    known = _table_defs()
    if table not in known:
        raise FeedConfigError(
            f"table '{table}' is not declared in schemas.ALL_TABLES; "
            f"known tables: {', '.join(sorted(known))}")
    return known[table]


def _parse_layers(raw: dict, where: str) -> dict[str, LayerDef]:
    layers_raw = _require(raw, "layers", where)
    if not isinstance(layers_raw, dict) or not layers_raw:
        raise FeedConfigError(f"{where}: 'layers' must be a non-empty mapping")

    layers: dict[str, LayerDef] = {}
    for name, body in layers_raw.items():
        spot = f"{where}: layers.{name}"
        table = _require(body or {}, "table", spot)
        table_def_for(table)  # raises with a specific message if unknown

        contract = (body or {}).get("contract")
        if contract and not (CONTRACT_DIR / contract).exists():
            raise FeedConfigError(
                f"{spot}: contract '{contract}' does not exist in "
                f"{CONTRACT_DIR.name}/")
        if (body or {}).get("typed") and not contract:
            raise FeedConfigError(
                f"{spot}: 'typed: true' needs a 'contract' -- column-type "
                "checks are generated per contract field, so without one the "
                "layer would silently produce no type checks at all")

        layers[name] = LayerDef(
            layer=name, table=table, contract=contract,
            freshness_column=(body or {}).get("freshness_column"),
            agent_watch=bool((body or {}).get("agent_watch", False)),
            typed=bool((body or {}).get("typed", False)))
    return layers


def _parse_monitors(raw: dict, layers: dict[str, LayerDef],
                     where: str) -> tuple[MonitorDef, ...]:
    out: list[MonitorDef] = []
    for entry in raw.get("monitors") or []:
        name = _require(entry, "name", where)
        spot = f"{where}: monitor '{name}'"
        layer = _require(entry, "layer", spot)
        if layer not in layers:
            raise FeedConfigError(
                f"{spot}: 'layer: {layer}' is not declared in this feed's "
                f"layers ({', '.join(sorted(layers))})")
        kind = _require(entry, "kind", spot)
        if kind not in KINDS:
            raise FeedConfigError(
                f"{spot}: unknown kind '{kind}' "
                f"(expected one of {', '.join(sorted(KINDS))})")
        out.append(MonitorDef(
            name=name, table=layers[layer].table, kind=kind,
            query=_require(entry, "query", spot), column=entry.get("column"),
            severity=entry.get("severity", "additive"),
            params=entry.get("params") or {},
            tables=entry.get("tables") or None,
            source=entry.get("source", "sql")))
    return tuple(out)


def _parse_arrival(raw: dict, layers: dict[str, LayerDef],
                    where: str) -> tuple[ArrivalDef, ...]:
    out: list[ArrivalDef] = []
    for entry in raw.get("arrival") or []:
        layer = _require(entry, "layer", where)
        spot = f"{where}: arrival on layer '{layer}'"
        if layer not in layers:
            raise FeedConfigError(
                f"{spot}: not declared in this feed's layers "
                f"({', '.join(sorted(layers))})")
        calendar = _require(entry, "calendar", spot)
        if calendar not in VALID_CALENDARS:
            raise FeedConfigError(
                f"{spot}: unknown calendar '{calendar}' "
                f"(expected one of {', '.join(sorted(VALID_CALENDARS))})")

        floor = entry.get("min_rows_per_period")
        if isinstance(floor, str):
            if floor not in _SYMBOLIC_ROW_FLOORS:
                raise FeedConfigError(
                    f"{spot}: unknown min_rows_per_period '{floor}' "
                    f"(expected an integer, null, or one of "
                    f"{', '.join(sorted(_SYMBOLIC_ROW_FLOORS))})")
            floor = _SYMBOLIC_ROW_FLOORS[floor]()

        out.append(ArrivalDef(
            table=layers[layer].table, column=_require(entry, "column", spot),
            calendar=calendar,
            max_lag_periods=int(_require(entry, "max_lag_periods", spot)),
            min_rows_per_period=floor))
    return tuple(out)


def load_feed(path: Path) -> FeedDef:
    where = f"feeds/{Path(path).name}"
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise FeedConfigError(f"{where}: file is empty or not a mapping")

    name = _require(raw, "name", where)
    feed_type = raw.get("feed_type", "fact")
    if feed_type not in VALID_FEED_TYPES:
        raise FeedConfigError(
            f"{where}: unknown feed_type '{feed_type}' "
            f"(expected one of {', '.join(sorted(VALID_FEED_TYPES))})")

    mode = (raw.get("ingest") or {}).get("mode", "batch")
    if mode not in VALID_INGEST_MODES:
        raise FeedConfigError(
            f"{where}: unknown ingest.mode '{mode}' "
            f"(expected one of {', '.join(sorted(VALID_INGEST_MODES))})")

    layers = _parse_layers(raw, where)
    return FeedDef(
        name=name, domain=raw.get("domain", "unknown"), feed_type=feed_type,
        owner=raw.get("owner", "unknown"), ingest_mode=mode, layers=layers,
        monitors=_parse_monitors(raw, layers, where),
        arrival=_parse_arrival(raw, layers, where))


def load_feeds(directory: Path | None = None) -> list[FeedDef]:
    """Every feed, sorted by name. Rejects duplicate monitor names globally.

    Monitor names are the join key between `ops.monitor_results` history and a
    live definition (`runner.load_baselines` filters on `monitor` alone). Two
    feeds sharing one would silently interleave two different measurements into
    one baseline series -- so it is rejected here rather than discovered later
    as an anomaly that will not reproduce.
    """
    feeds = [load_feed(p) for p in sorted(Path(directory or FEED_DIR).glob("*.yaml"))]

    seen: dict[str, str] = {}
    for feed in feeds:
        for monitor in feed.monitors:
            if monitor.name in seen:
                raise FeedConfigError(
                    f"duplicate monitor name '{monitor.name}' in feeds "
                    f"'{seen[monitor.name]}' and '{feed.name}'; monitor names "
                    "are the key that joins a definition to its persisted "
                    "baseline history and must be unique across all feeds")
            seen[monitor.name] = feed.name
    return feeds


def all_monitors(directory: Path | None = None) -> list[MonitorDef]:
    return [m for feed in load_feeds(directory) for m in feed.monitors]
