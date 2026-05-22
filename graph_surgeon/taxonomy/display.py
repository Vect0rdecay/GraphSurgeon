"""
Registry-backed display formatting for motifs, patterns, and catalog lookup.

Titles and descriptions are sourced from gadget_registry.py. No security ratings
are emitted. Output frames findings as structural motifs indexed to AML literature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from graph_surgeon.taxonomy.gadget_registry import (
    CHAIN_REGISTRY,
    GADGET_REGISTRY,
    GadgetDefinition,
    get_gadget_info,
)

if TYPE_CHECKING:
    from graph_surgeon.analysis.motifs import GadgetType

# Pilot pattern detector -> registry gadget ID
PATTERN_DETECTOR_TO_REGISTRY: Dict[str, str] = {
    "_detect_batchnorm_vuln": "NORMALIZER",
    "_detect_global_pooling_vuln": "GAP_FC_HEAD",
    "_detect_fc_layer_vuln": "GAP_FC_HEAD",
    "_detect_maxpool_amplification": "MAXPOOL_AFTER_FUSION",
}


def get_chain_info(chain_id: str) -> Optional[Dict[str, Any]]:
    """Return chain metadata from CHAIN_REGISTRY."""
    return CHAIN_REGISTRY.get(chain_id)


def registry_id_for_gadget_type(gadget_type: "GadgetType") -> Optional[str]:
    """Map GadgetType enum member to GADGET_REGISTRY id when registered."""
    gid = gadget_type.name
    if gid in GADGET_REGISTRY:
        return gid
    return None


def build_gadget_type_to_registry() -> Dict[str, str]:
    """Build GadgetType.name -> registry id for all registered gadgets."""
    return {gid: gid for gid in GADGET_REGISTRY}


def format_gadget_title(
    gadget_id: str,
    *,
    count: Optional[int] = None,
    node_id: Optional[str] = None,
) -> str:
    """Format user-facing title: REGISTRY_ID — registry.name"""
    info = get_gadget_info(gadget_id)
    if not info:
        base = gadget_id
    else:
        base = f"{gadget_id} — {info.name}"
    parts = [base]
    if count is not None:
        parts.append(f"({count} layers)")
    if node_id:
        parts.append(f"@ {node_id}")
    return " ".join(parts)


def format_gadget_description(
    gadget_id: str,
    *,
    extra_context: Optional[str] = None,
) -> str:
    """Structural motif description with literature framing."""
    info = get_gadget_info(gadget_id)
    if not info:
        return extra_context or f"Structural motif {gadget_id} detected in graph."

    lines = [
        f"Structural motif: {info.description.strip()}",
        f"Associated attack classes from literature: {', '.join(info.attacks_enabled)}",
        f"Research basis: {', '.join(info.research_basis)}",
    ]
    if extra_context:
        lines.append(extra_context.strip())
    return "\n".join(lines)


def format_chain_title(chain_id: str) -> str:
    """Format chain title: CHAIN_ID — registry name."""
    chain = get_chain_info(chain_id)
    if not chain:
        return chain_id
    name = chain.get("name", chain_id)
    return f"{chain_id} — {name}"


def format_chain_description(
    chain_id: str,
    *,
    gadget_ids_found: Optional[List[str]] = None,
    extra_context: Optional[str] = None,
) -> str:
    """Chain description with required gadgets and literature framing."""
    chain = get_chain_info(chain_id)
    if not chain:
        return extra_context or f"Compound structural motif chain {chain_id}."

    required = chain.get("required_gadgets", [])
    optional = chain.get("optional_gadgets", [])
    research = chain.get("research_basis", [])

    lines = [f"Compound structural motif: {chain.get('name', chain_id)}."]
    if required:
        lines.append(f"Required motifs: {', '.join(required)}.")
    if optional:
        lines.append(f"Optional motifs: {', '.join(optional)}.")
    if gadget_ids_found:
        lines.append(f"Detected in graph: {', '.join(gadget_ids_found)}.")
    if research:
        lines.append(f"Research basis: {', '.join(research)}.")
    lines.append(
        "Associated attack classes from literature: see "
        + ", ".join(required + optional)
        + " via catalog --gadget."
    )
    if extra_context:
        lines.append(extra_context.strip())
    return "\n".join(lines)


def format_gadget_finding(
    gadget_id: str,
    *,
    count: Optional[int] = None,
    node_id: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Return display fields for a gadget-level finding."""
    info = get_gadget_info(gadget_id)
    return {
        "title": format_gadget_title(gadget_id, count=count, node_id=node_id),
        "description": format_gadget_description(gadget_id, extra_context=extra_context),
        "registry_id": gadget_id,
        "research_basis": list(info.research_basis) if info else [],
        "attacks_enabled": list(info.attacks_enabled) if info else [],
    }


def format_chain_finding(
    chain_id: str,
    *,
    gadget_ids_found: Optional[List[str]] = None,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Return display fields for a chain-level finding."""
    chain = get_chain_info(chain_id) or {}
    research = list(chain.get("research_basis", []))
    return {
        "title": format_chain_title(chain_id),
        "description": format_chain_description(
            chain_id,
            gadget_ids_found=gadget_ids_found,
            extra_context=extra_context,
        ),
        "chain_id": chain_id,
        "registry_id": chain_id,
        "research_basis": research,
    }


def format_pattern_from_registry(
    gadget_id: str,
    *,
    pattern_id: str,
    nodes_involved: List[str],
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Build StructuralPattern display fields from registry gadget."""
    count = len(nodes_involved) if nodes_involved else None
    finding = format_gadget_finding(
        gadget_id,
        count=count,
        extra_context=extra_context,
    )
    return {
        "id": pattern_id,
        "name": finding["title"],
        "description": finding["description"],
        "registry_id": gadget_id,
        "research_basis": finding["research_basis"],
        "attacks_enabled": finding["attacks_enabled"],
        "nodes_involved": nodes_involved,
    }


def format_catalog_gadget(gadget_id: str) -> str:
    """Full catalog text for graph-surgeon catalog --gadget."""
    info = get_gadget_info(gadget_id)
    if not info:
        return f"Unknown gadget: {gadget_id}"

    lines = [
        f"{gadget_id} — {info.name}",
        f"Category: {info.category.value}",
        "",
        f"Structural motif: {info.description.strip()}",
        "",
        f"Detection: {info.detection_logic.strip()}",
        "",
        f"Associated attack classes from literature: {', '.join(info.attacks_enabled)}",
        "",
        f"Research basis: {', '.join(info.research_basis)}",
        "",
        f"Status: {info.status.value} | Confidence: {info.confidence}",
    ]
    if info.chainable_with:
        lines.append(f"Chainable with: {', '.join(info.chainable_with)}")
    if info.notes:
        lines.append(f"Notes: {info.notes.strip()}")
    return "\n".join(lines)


def format_catalog_chain(chain_id: str) -> str:
    """Full catalog text for graph-surgeon catalog --chain."""
    chain = get_chain_info(chain_id)
    if not chain:
        return f"Unknown chain: {chain_id}"

    lines = [
        format_chain_title(chain_id),
        "",
        format_chain_description(chain_id),
    ]
    if chain.get("notes"):
        lines.append("")
        lines.append(chain["notes"].strip())
    return "\n".join(lines)


# Custom finding ids that map to a single registry gadget (not in CHAIN_REGISTRY)
FINDING_ID_TO_GADGET: Dict[str, str] = {
    "CHAIN-BN-FRAGILITY": "NORMALIZER",
    "CHAIN-GAP-FC-HEAD": "GAP_FC_HEAD",
}

# Motifs finding ids that alias to CHAIN_REGISTRY keys
FINDING_ID_TO_CHAIN: Dict[str, str] = {
    "SHADOWLOGIC-EXISTING-BACKDOOR": "CHAIN-SHADOWLOGIC-EXISTING-BACKDOOR",
    "SHADOWLOGIC-INJECTION-SUSCEPTIBILITY": "CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY",
}


def apply_registry_display_to_finding(finding: Any) -> None:
    """Apply registry-backed title/description to a StructuralFinding."""
    chain_id = FINDING_ID_TO_CHAIN.get(finding.id, finding.id)
    if chain_id in CHAIN_REGISTRY:
        extra = None
        if finding.node_id:
            extra = f"Primary node: {finding.node_id}."
        data = format_chain_finding(chain_id, extra_context=extra)
        finding.title = data["title"]
        finding.description = data["description"]
        finding.registry_id = chain_id
        finding.chain_id = chain_id
        if data.get("research_basis"):
            finding.references = list(data["research_basis"])
        return

    gadget_id = FINDING_ID_TO_GADGET.get(finding.id)
    if not gadget_id and finding.id.startswith("CHAIN-FUSION-MAXPOOL-"):
        gadget_id = "MAXPOOL_AFTER_FUSION"
    if gadget_id:
        extra = f"Primary node: {finding.node_id}." if finding.node_id else None
        data = format_gadget_finding(gadget_id, extra_context=extra)
        finding.title = data["title"]
        finding.description = data["description"]
        finding.registry_id = gadget_id
        if data.get("research_basis"):
            finding.references = list(data["research_basis"])
        return

    # Infer registry gadget from finding id prefix matching registry gadget id
    for gid in GADGET_REGISTRY:
        if finding.id == gid or finding.id.startswith(f"{gid}-"):
            data = format_gadget_finding(gid)
            finding.title = data["title"]
            finding.description = data["description"]
            finding.registry_id = gid
            if data.get("research_basis"):
                finding.references = list(data["research_basis"])
            return


def apply_registry_display_to_findings(findings: List[Any]) -> None:
    """Apply registry display to all structural findings."""
    for finding in findings:
        apply_registry_display_to_finding(finding)


def registry_id_for_gadget_summary_entry(gadget_type_value: str) -> Optional[str]:
    """Map gadget_type enum value (snake_case) to registry id."""
    candidate = gadget_type_value.upper()
    if candidate in GADGET_REGISTRY:
        return candidate
    return None


def apply_registry_to_finding(finding: Any, *, chain_id: Optional[str] = None, gadget_id: Optional[str] = None) -> None:
    """Mutate a StructuralFinding in place with registry-backed title/description."""
    if chain_id:
        data = format_chain_finding(chain_id)
        finding.title = data["title"]
        finding.description = data["description"]
        if hasattr(finding, "references") and data.get("research_basis"):
            finding.references = list(data["research_basis"])
    elif gadget_id:
        data = format_gadget_finding(gadget_id)
        finding.title = data["title"]
        finding.description = data["description"]
        if hasattr(finding, "references") and data.get("research_basis"):
            finding.references = list(data["research_basis"])
