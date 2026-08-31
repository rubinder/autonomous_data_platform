# 0008 — Per-feed configs, and a migration gated on measurement

**Type:** refactor ([#3](https://github.com/rubinder/autonomous_data_platform/issues/3))
**Made:** golden baseline captured *before* any code changed, and it caught a real bug
**Shipped:** `feeds/<name>.yaml` + `src/feeds.py`; `monitors/` deleted

## The axis was wrong, not the amount of config

The repo already had declarative config: ten monitors in
`monitors/{bronze,silver,gold}.yaml`, validated against a `KINDS` registry that
fails loudly on an unknown kind. What it did not have was config organised **by
feed**. A feed was spread across four places, three of them Python —
`monitors/<layer>.yaml`, `graph.py:WATCHED`, `arrival.py:ARRIVAL_SLAS`,
`runner.py:TYPED_TABLES` — so adding one feed meant touching every layer file
plus three modules, and no single place described one feed.

Three feeds now exist. `forecast_training_set` is a *derived* feed rather than
an ingested one; it gets a file because a Gold mart has exactly the same
operational needs as an ingested feed and belongs to neither upstream feed
cleanly.

Iceberg schemas stay in `src/lakehouse/schemas.py`. They declare field IDs and
partition transforms, which is code; moving them into YAML would be config
theatre.

## Why the baseline was captured first

The risk in this migration is not failing loudly. It is **succeeding while
quietly changing a threshold, dropping a monitor, or widening an SLA** — with
every test still green, because the tests were written against the same config
that moved.

So before any code changed, the exact output of `run_monitors`,
`check_arrival` and the agent was captured on **two** warehouses. Two, because a
clean warehouse produces 17 `ok` results and 0 findings, and a comparison
against an all-`ok` baseline is satisfied by a migration that breaks every check
into silence. The drifted capture carries a real breach and five real findings.

**It caught a bug on the first comparison.** Removing a now-unused `schemas`
import left a `NameError` in `runner`, and `run_monitors`'s deliberately broad
`except` — which exists so one malformed monitor cannot kill a run — converted
it into a *breach on all seventeen monitors*. Seventeen results appeared and the
run exited non-zero. Nothing about that output says "broken": it looks like a
platform correctly reporting that everything is on fire. Only the before/after
diff showed every `metric` was `null` and every `detail` read
`monitor failed: NameError: name 'schemas' is not defined`.

That is the failure mode a broad `except` buys, and it is still the right trade
— but it means a migration touching this path cannot be verified by running it
and reading the output.

After the fix both captures matched: 17 monitor results, 8 arrival checks, and
on the drifted warehouse the same 5 findings with the same severities.

## The one difference was not the migration

`silver_txn_mean_abs_amount` differed in the 13th decimal place. Rather than
assume, three consecutive runs of *identical* code were measured:

```
171.4137026799999
171.41370268000034
171.4137026800003
```

DuckDB sums in parallel, so `avg(abs(signed_amount))` over 250,000 rows varies
in its last bits between runs. Relative magnitude ~2.5e-15 — irrelevant to the
z-score check that consumes it, but the daily report was rendering it as a
`+1.56e-12` *change*, which presents noise as signal and churned the committed
reports on every regeneration. `_delta` now returns `0` below a relative
threshold of `1e-9`, about six orders of magnitude below anything a monitor
could act on.

The pinned expectations live in `tests/test_feeds.py` as literals, deliberately
**not** derived from `feeds/*.yaml`: a check that reads its expectation from the
thing under test checks nothing.

## A typo'd kind is now a config error, not a breach

The loader rejects an unknown `kind` up front, naming the file, the monitor and
the valid kinds. Previously it reached `evaluate()` and surfaced as a *breach*.

That was far better than passing, but it still said the wrong thing: a breach
means *your data is bad*; this means *your config is bad*. Conflating them sends
someone to look at a table when the problem is a YAML file.

The runtime path is **kept**, not replaced. A `MonitorDef` can be constructed
without going through the loader, and one bad monitor must never take down the
other sixteen — that guarantee has its own test now
(`test_an_unknown_kind_reaching_the_runner_is_still_a_breach_not_a_crash`),
exercised through the path that can still reach it.

## A second ordering defect, surfaced by the migration

Feeds load alphabetically, so `stock_prices` now precedes `yodlee_transactions`
and the arrival findings changed order. The counts and content were identical,
but three committed reports churned — because [0007](0007-finding-history-in-iceberg-not-in-the-incident-files.md)'s
ordering fix covered the diff sections and missed the verdict, schema, anomaly
and incident renderers, which iterated in check-execution order.

Every render path now sorts on a key ending in `finding_key` or `monitor`, so
output depends only on *what* is in the snapshot, never on the order it arrived
in. A test renders the same findings in two different orders and compares.

Worth stating plainly: this defect existed before the migration and was invisible
because nothing had reordered the checks yet. The migration did not cause it; it
revealed it.


## Review round: absent config must not read as clean config

Probing the shipped loader found the same failure this repository keeps closing,
one level up from where it had been closed before.

**An empty or missing `feeds/` directory loaded zero feeds, silently.** Every
monitor, arrival SLA and agent watch is now declared there, so a deleted or
mistyped directory made `make monitor` print `0 checks -- 0 breach, 0 warn` and
exit **0**. A platform reporting perfect health because it was looking at
nothing at all. Centralising the config concentrated this risk: before, losing
one `monitors/<layer>.yaml` lost a third of the checks; now, losing the
directory loses all of them *and* the arrival SLAs *and* the agent's watch list.
`load_feeds` raises.

**Two feeds typing the same table minted colliding monitor names.** Column-type
monitors are generated as `f"{table}_type_{field}"`, so two feeds with
`typed: true` on one table produce identical names -- exactly the baseline
corruption the duplicate-monitor-name check exists to prevent, arriving by a
route that check could not see, because generated names never appear in
`feed.monitors`. Same for two feeds watching one table, which would raise every
finding on it twice.

**A `tables:` alias naming an unknown table was unvalidated.** It would have
failed at query time and been swallowed into a breach by `run_monitors`'s broad
except: visible, but reported as a data problem when it is a config typo.

Two feeds merely *naming* the same table stays legal -- a shared dimension is a
real thing. Only the collisions that corrupt something are rejected.

Both golden captures were re-run after the hardening and still match.
