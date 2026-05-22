"""Tests for registry-backed display formatting."""

import json
from pathlib import Path

import pytest

from graph_surgeon.taxonomy.display import (
    format_catalog_chain,
    format_catalog_gadget,
    format_chain_finding,
    format_gadget_finding,
    format_gadget_title,
    get_chain_info,
)
from graph_surgeon.taxonomy.gadget_registry import GADGET_REGISTRY, CHAIN_REGISTRY
from graph_surgeon.reporting.sanitize import serialize_for_export

PILOT_GADGETS = ["NORMALIZER", "GAP_FC_HEAD", "MAXPOOL_AFTER_FUSION"]
PILOT_CHAINS = ["CHAIN-PATCH-ATTACK-SURFACE", "CHAIN-PHYSICAL-WORLD-ATTACK", "CHAIN-COMPOUND-PHYSICAL-PATCH"]


@pytest.mark.parametrize("gadget_id", PILOT_GADGETS)
def test_pilot_gadget_titles_use_registry_id(gadget_id):
    title = format_gadget_title(gadget_id, count=3)
    assert gadget_id in title
    assert "Vulnerability" not in title


@pytest.mark.parametrize("gadget_id", PILOT_GADGETS)
def test_pilot_gadget_description_literature_framing(gadget_id):
    data = format_gadget_finding(gadget_id)
    assert "Associated attack classes from literature:" in data["description"]
    assert "Research basis:" in data["description"]
    assert "Structural motif:" in data["description"]
    assert data["research_basis"]
    assert "severity" not in data


@pytest.mark.parametrize("chain_id", PILOT_CHAINS)
def test_pilot_chain_titles(chain_id):
    assert chain_id in get_chain_info(chain_id) or chain_id in CHAIN_REGISTRY
    data = format_chain_finding(chain_id)
    assert chain_id in data["title"]
    assert "Vulnerability" not in data["title"]


def test_catalog_gadget_normalizer():
    text = format_catalog_gadget("NORMALIZER")
    assert "NORMALIZER" in text
    assert "Associated attack classes from literature:" in text
    assert "BNAttack-2020" in text or "66-AdvShadow-2022" in text


def test_catalog_chain_patch():
    text = format_catalog_chain("CHAIN-PATCH-ATTACK-SURFACE")
    assert "CHAIN-PATCH-ATTACK-SURFACE" in text
    assert "GAP_FC_HEAD" in text


def test_display_map_pilot_fixture_matches_registry():
    """Regression: pilot display map derived from live registry."""
    fixture_path = Path(__file__).parent / "fixtures" / "display_map_pilot.json"
    expected = {}
    for gid in PILOT_GADGETS:
        info = GADGET_REGISTRY[gid]
        expected[gid] = {
            "title": format_gadget_title(gid),
            "research_basis": list(info.research_basis),
            "attacks_enabled": list(info.attacks_enabled),
        }
    for cid in PILOT_CHAINS:
        chain = CHAIN_REGISTRY[cid]
        expected[cid] = {
            "title": format_chain_finding(cid)["title"],
            "research_basis": list(chain.get("research_basis", [])),
        }
    if fixture_path.exists():
        on_disk = json.loads(fixture_path.read_text())
        for key in PILOT_GADGETS + PILOT_CHAINS:
            assert on_disk[key]["title"] == expected[key]["title"]
    else:
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(expected, indent=2))


def test_sanitize_preserves_registry_metadata():
    data = format_gadget_finding("GAP_FC_HEAD")
    exported = serialize_for_export(data)
    assert exported["registry_id"] == "GAP_FC_HEAD"
    assert "severity" not in exported
