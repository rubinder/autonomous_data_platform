"""Ops agent: watches the lakehouse for schema drift, volume anomalies, and staleness.

Reads table metadata rather than scanning data wherever the engine allows it,
so the agent stays cheap enough to run often. Classification is rules-first;
an LLM is only consulted for cases no rule matches, and never in CI.
"""
