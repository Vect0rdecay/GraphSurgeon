"""Tests for deployment-context motifs and no-risk chain export."""

import json

import pytest

from graph_surgeon.analysis.motifs import (
    GadgetDetector,
    GadgetType,
    NodeSecurityProfile,
    make_registry_chain_finding,
)
from graph_surgeon.reporting.sanitize import serialize_for_export
from graph_surgeon.taxonomy.display import format_catalog_gadget, format_chain_finding
from graph_surgeon.taxonomy.gadget_registry import GADGET_REGISTRY, CHAIN_REGISTRY, validate_registry


def _node(node_id: str, op_type: str, idx: int = 0) -> NodeSecurityProfile:
    return NodeSecurityProfile(
        node_id=node_id,
        op_type=op_type,
        attributes={},
        input_shapes=[()],
        output_shapes=[()],
    )


@pytest.mark.parametrize(
    "gadget_id",
    [
        "SINGLE_MODALITY_INPUT",
        "IN_GRAPH_PREPROCESSING",
        "HAS_MULTIMODAL_FUSION",
        "AUDIO_MEL_INPUT",
        "AUDIO_1D_CONV",
    ],
)
def test_deployment_gadgets_in_registry(gadget_id):
    assert gadget_id in GADGET_REGISTRY
    text = format_catalog_gadget(gadget_id)
    assert gadget_id in text
    assert "Structural motif" in text


@pytest.mark.parametrize(
    "chain_id",
    [
        "CHAIN-SINGLE-MODALITY-VISION",
        "CHAIN-PREPROCESSING-TRUST-BOUNDARY",
        "CHAIN-AUDIO-ADVERSARIAL-SURFACE",
        "CHAIN-ACOUSTIC-COMMAND-SURFACE",
    ],
)
def test_deployment_chains_in_registry(chain_id):
    assert chain_id in CHAIN_REGISTRY
    data = format_chain_finding(chain_id)
    assert "severity" not in data
    assert chain_id in data["title"]


def test_registry_validation_passes():
    issues = validate_registry()
    assert issues == [], issues


def test_single_modality_and_preprocess_detection():
    nodes = [
        _node("sub1", "Sub", 0),
        _node("conv1", "Conv", 1),
        _node("gap1", "GlobalAveragePool", 2),
        _node("fc1", "Gemm", 3),
    ]
    edges = [("sub1", "conv1"), ("conv1", "gap1"), ("gap1", "fc1")]
    detector = GadgetDetector()
    gadgets = detector.detect_gadgets(nodes, edges, num_graph_inputs=1)
    types = {g.gadget_type for g in gadgets}
    assert GadgetType.SINGLE_MODALITY_INPUT in types
    assert GadgetType.IN_GRAPH_PREPROCESSING in types


def test_deployment_context_chains_from_gadgets():
    nodes = [
        _node("sub1", "Sub", 0),
        _node("conv1", "Conv", 1),
        _node("gap1", "GlobalAveragePool", 2),
        _node("fc1", "Gemm", 3),
    ]
    edges = [("sub1", "conv1"), ("conv1", "gap1"), ("gap1", "fc1")]
    detector = GadgetDetector()
    gadgets = detector.detect_gadgets(nodes, edges, num_graph_inputs=1)
    gadget_map = {g.node_id: g for g in gadgets}
    chains = detector._find_deployment_context_chains(gadgets, gadget_map)
    chain_ids = {c.id for c in chains}
    assert "CHAIN-SINGLE-MODALITY-VISION" in chain_ids
    assert "CHAIN-PREPROCESSING-TRUST-BOUNDARY" in chain_ids


def test_make_registry_chain_finding_sanitizes_clean():
    finding = make_registry_chain_finding(
        "CHAIN-PREPROCESSING-TRUST-BOUNDARY",
        "sub1",
        gadget_ids_found=["IN_GRAPH_PREPROCESSING"],
    )
    exported = serialize_for_export(finding)
    assert exported["registry_id"] == "CHAIN-PREPROCESSING-TRUST-BOUNDARY"
    assert "severity" not in exported
    assert "cvss_estimate" not in exported
    assert "Vulnerability" not in exported["title"]


def test_catalog_single_modality_links_paper_94():
    text = format_catalog_gadget("SINGLE_MODALITY_INPUT")
    assert "94-AdversarialBulbs-2021" in text
