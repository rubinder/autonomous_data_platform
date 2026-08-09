# 0002 — `hashlib` instead of the builtin `hash()` for merchant ids

**Type:** documented deviation from the plan; a real bug the plan would have shipped
**Made:** during implementation (Task 3), upheld at review
**Plan said:** derive `merchant.id` with Python's builtin `hash()`
**Shipped:** an md5-derived stable hash

## The bug

Python's builtin `hash()` is salted per process for `str` and `bytes`. Unless
`PYTHONHASHSEED` is pinned, the same string hashes to a different value in every
new interpreter. The plan used it to synthesise Yodlee `merchant.id` values.

So the generator was deterministic *within* a run and non-deterministic *across*
runs. `make all` today and `make all` tomorrow would produce different merchant
ids for the same merchant, and any warehouse built from the two would disagree.

## Why the tests did not catch it

This is the part worth recording. The determinism test was:

> generate with seed 42 twice, assert the outputs are identical

Both generations happened **inside the same pytest process**, so both saw the
same hash salt, so the outputs matched, so the test passed. The test asserted
exactly the property that was broken and could not observe it, because its two
observations shared the state that varied.

A same-process determinism test is not a determinism test. It is a
purity test.

## The fix

Replaced with an md5-derived hash — deterministic across processes, across
machines, and across Python versions. (md5 is used here as a plain hash function
for id synthesis, not for any security purpose.)

## Verification

The reviewer reproduced the bug before accepting the fix: running the plan's
version in two processes produced **3 differing merchant-id values**; the shipped
version produced identical output.

**That check was originally performed by hand and not pinned by a test. It is
pinned now.** `tests/test_yodlee_feed.py::test_deterministic_across_separate_processes`
runs generation in two separate interpreters with `PYTHONHASHSEED=random` and
asserts a byte-identical sha256 of the output. Re-demonstrated during the final
review: reverting `_stable_merchant_hash` to builtin `hash()` makes the
subprocess test fail with two differing digests while
`test_deterministic` — which still generates twice in one process — stays
green. That is the whole point recorded above, now enforced rather than
narrated.

## Where else this bit

The same class of bug appears twice more in this repository, and the same fix
applies:

- **Incident slugs** (`src/agent/actions.py`) — a salted slug would make the
  agent write a *new* incident file for the same finding on every run instead of
  rewriting one. Now sha256-derived; verified stable across three
  `PYTHONHASHSEED=random` processes.
- **Alert throttle keys** (`src/ops/alerts.py`) — a salted key would silently
  disable throttling across process restarts, which is precisely when a
  scheduled monitor needs it. Now `hashlib`-derived; verified identical across
  two processes.

## The generalisable rule

**Any hash whose value must outlive the process must come from `hashlib`, never
from `hash()`.** And any determinism claim must be tested across process
boundaries, because the in-process version of that test passes whether the claim
is true or not.
