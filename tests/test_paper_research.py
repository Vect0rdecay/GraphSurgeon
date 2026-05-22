"""Bundled taxonomy research data loads without external paths."""

import json
import re
from pathlib import Path

import pytest

from graph_surgeon.taxonomy.gadget_registry import GADGET_REGISTRY
from graph_surgeon.taxonomy.paper_research import (
    _ATTACK_RESEARCH_NOTES,
    _DETECTION_RATIONALE,
    _TAXONOMY_JSON,
    extract_paper_section,
    format_paper_catalog_block,
    extract_chain_rationale,
)
from graph_surgeon.taxonomy.research_coverage import (
    format_coverage_report,
    load_research_coverage,
    registry_slugs_missing_notes,
)


def test_bundled_research_files_ship_with_package():
    assert _TAXONOMY_JSON.is_file()
    assert _ATTACK_RESEARCH_NOTES.is_file()
    assert _DETECTION_RATIONALE.is_file()


def test_paper_section_loaded_from_bundle_not_external():
    section = extract_paper_section("36")
    assert section is not None
    assert "Brown" in section
    assert "Adversarial Patch" in section


def test_catalog_paper_block_uses_bundled_notes():
    block = format_paper_catalog_block("36-GoogleAp-2017")
    assert "Brown" in block
    assert "nn_security_analyzer" not in block.lower()
    assert "Research notes:" in block


def test_chain_rationale_from_bundled_detection_doc():
    text = extract_chain_rationale("CHAIN-PATCH-ATTACK-SURFACE")
    assert text is not None
    assert "Patch Attack" in text or "GAP" in text


def test_paper_section_71_excludes_cross_cutting_meta():
    section = extract_paper_section("71")
    assert section is not None
    assert "DPATCH" in section or "DPatch" in section
    assert "Cross-Cutting" not in section
    assert "Session Log" not in section
    assert "Proposed New Gadget" not in section


def test_paper_section_114_excludes_shadowlogic_and_phase_logs():
    section = extract_paper_section("114")
    assert section is not None
    assert "Fake It" in section or "ViT" in section
    assert "ShadowLogic" not in section
    assert "Phase 3 Session Log" not in section
    assert "Next Steps" not in section


def test_gap_fc_head_catalog_excludes_project_meta():
    from graph_surgeon.taxonomy.display import format_catalog_gadget

    text = format_catalog_gadget("GAP_FC_HEAD")
    for forbidden in (
        "Cross-Cutting Vulnerability",
        "Proposed New Gadget",
        "Session Log",
        "ShadowLogic",
        "Papers for Phase 2",
        "Questions for Discussion",
        "Phase 3 Session Log",
        "Registry changelog",
    ):
        assert forbidden not in text, f"unexpected catalog content: {forbidden}"


def test_research_coverage_no_missing_taxonomy_papers():
    cov = load_research_coverage()
    assert cov["summary"]["missing"] == 0
    assert cov["summary"]["complete"] + cov["summary"]["out_of_scope"] == len(cov["papers"])


def test_all_registry_numeric_slugs_have_corpus_entry():
    slugs = []
    for info in GADGET_REGISTRY.values():
        slugs.extend(info.research_basis)
    missing = registry_slugs_missing_notes(slugs)
    assert missing == [], f"registry slugs without corpus: {missing[:10]}"


def test_gap_fc_head_papers_have_detailed_notes():
    for slug in ("36-GoogleAp-2017", "42-ACS-2019", "44-AdversarialACO-2020", "45-Adv-watermark-2020"):
        block = format_paper_catalog_block(slug)
        assert "Research notes:" in block
        assert "not yet in corpus" not in block.lower()


def test_attack_research_notes_has_all_taxonomy_ids():
    tax = json.loads(_TAXONOMY_JSON.read_text())
    note_ids = set(re.findall(r"^### \[(\d+)\]", _ATTACK_RESEARCH_NOTES.read_text(), re.M))
    tax_ids = {v["numeric_id"] for v in tax.values()}
    assert tax_ids <= note_ids


def test_catalog_coverage_command():
    text = format_coverage_report()
    assert "Complete:" in text
    assert "Missing:       0" in text


def test_packaged_notes_exclude_internal_file():
    import subprocess

    data_dir = Path(__file__).resolve().parents[1] / "graph_surgeon" / "taxonomy" / "data"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(data_dir / "attack_research_internal.md")],
        capture_output=True,
        cwd=data_dir.parents[2],
    )
    assert tracked.returncode != 0
    notes = (data_dir / "attack_research_notes.md").read_text()
    assert "ShadowLogic Supply Chain" not in notes
    assert "Session Log" not in notes
