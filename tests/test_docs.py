"""Documentation tests.

These are cheap, and they exist because documentation is the one artifact in
this repo with no other feedback loop: a wrong number in the README does not
crash anything. The load-bearing one is
`test_readme_does_not_overclaim_real_alpha` -- the results table looks
persuasive, and a portfolio repo that lets a reader assume the alpha is real
is worse than one with no results at all.
"""
from pathlib import Path

import pytest

from src import config

DOCS = config.REPO_ROOT / "docs"


def test_readme_exists_and_has_required_sections():
    text = (config.REPO_ROOT / "README.md").read_text()
    for heading in ("## Architecture", "## AI-Driven SDLC",
                    "## The agentic ops layer", "## GitHub Actions",
                    "## What I'd do differently at production scale"):
        assert heading in text, f"missing {heading}"


def test_readme_contains_a_mermaid_diagram():
    assert "```mermaid" in (config.REPO_ROOT / "README.md").read_text()


@pytest.mark.parametrize("n", range(1, 8))
def test_each_adr_exists_and_states_a_decision(n):
    matches = list((DOCS / "architecture").glob(f"ADR-{n:04d}-*.md"))
    assert matches, f"ADR-{n:04d} missing"
    text = matches[0].read_text()
    for section in ("## Context", "## Decision", "## Consequences"):
        assert section in text


def test_readme_does_not_overclaim_real_alpha():
    """The prices are synthetic. The README must say so."""
    text = (config.REPO_ROOT / "README.md").read_text().lower()
    assert "synthetic" in text


def test_readme_states_the_measured_verdict_not_a_flattering_one():
    """0 of 6 tickers beat persistence. The README must not round that up."""
    text = (config.REPO_ROOT / "README.md").read_text()
    assert "0 of 6" in text
    assert "pipeline fidelity" in text


def test_readme_documents_operational_monitoring():
    text = (config.REPO_ROOT / "README.md").read_text()
    assert "## Operational monitoring" in text
    for target in ("make monitor", "make arrival"):
        assert target in text


def test_readme_documents_schema_evolution_and_cross_version_queries():
    text = (config.REPO_ROOT / "README.md").read_text()
    assert "## Schema evolution and cross-version queries" in text
    assert "field ID" in text or "field-ID" in text


def test_ai_sdlc_workflow_documented():
    assert (DOCS / "ai-sdlc" / "workflow.md").exists()
    assert (DOCS / "ai-sdlc" / "decisions" / "0001-sklearn-over-lightgbm.md").exists()


def test_ai_sdlc_records_the_deviations_that_were_actually_made():
    decisions = sorted((DOCS / "ai-sdlc" / "decisions").glob("*.md"))
    assert len(decisions) >= 3, "at least three implementation deviations were recorded"
    assert (DOCS / "ai-sdlc" / "decisions" / "0002-stable-merchant-hash.md").exists()
    assert (DOCS / "ai-sdlc" / "prompts" / "README.md").exists()


def test_ci_workflow_does_not_gate_on_model_skill():
    """CI must fail on a broken pipeline, never on which model won.

    The forecast at reduced scale has ~59% of training rows with every spend
    feature NULL; any skill metric computed there is noise.
    """
    ci = (config.REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for banned in ("sharpe", "beats", "directional", " ic ", "rmse"):
        assert banned not in ci.lower(), f"CI appears to gate on {banned.strip()}"


@pytest.mark.parametrize("workflow", ["ci.yml", "monitors.yml"])
def test_workflow_comments_match_the_scale_they_run(workflow):
    """A comment quoting a rationale against the wrong N is a silent lie."""
    text = (config.REPO_ROOT / ".github" / "workflows" / workflow).read_text()
    declared = {line.split("N_TRANSACTIONS=")[1].split()[0]
                for line in text.splitlines()
                if "N_TRANSACTIONS=" in line and not line.strip().startswith("#")}
    commented = {line.split("N_TRANSACTIONS=")[1].split(",")[0].split()[0].rstrip(")")
                 for line in text.splitlines()
                 if "N_TRANSACTIONS=" in line and line.strip().startswith("#")}
    assert commented <= declared, (
        f"{workflow}: comment cites N_TRANSACTIONS={commented} but the job runs {declared}"
    )


def test_every_adr_referenced_by_the_readme_exists():
    text = (config.REPO_ROOT / "README.md").read_text()
    referenced = {f"ADR-{n:04d}" for n in range(1, 8) if f"ADR-{n:04d}" in text}
    on_disk = {p.name[:8] for p in (DOCS / "architecture").glob("ADR-*.md")}
    assert referenced <= on_disk
    assert referenced, "the README cites no ADRs at all"


def test_docs_directory_layout_matches_the_spec():
    for path in (DOCS / "architecture", DOCS / "ai-sdlc", DOCS / "ai-sdlc" / "decisions",
                 DOCS / "ai-sdlc" / "prompts"):
        assert Path(path).is_dir(), f"missing {path}"
