"""Build a SceneGraph JSON structure from an ONNX model using the existing engine."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_surgeon.scene.schema import (
    SCHEMA_VERSION,
    PaperRef,
    SceneChain,
    SceneEdge,
    SceneGraph,
    SceneInput,
    SceneModelInfo,
    SceneMotif,
    SceneNode,
    SceneOutput,
    SceneShadowLogic,
    SceneShadowLogicPoint,
    SceneStructuralPatterns,
)


def build_scene(
    model_path: str,
    *,
    include_motifs: bool = True,
    include_weights: bool = False,
) -> SceneGraph:
    """Assemble a SceneGraph from an ONNX model by calling the existing engine.

    This is purely a projection — no new analysis logic lives here.
    """
    from graph_surgeon.analysis.motifs import OPERATOR_REFERENCE_DB
    from graph_surgeon.graph.surgeon import GraphSurgeon
    from graph_surgeon.parsers.onnx_parser import ONNXGraphParser

    parser = ONNXGraphParser()
    graph = parser.parse_file(model_path)
    surgeon = GraphSurgeon(verbose=False)
    raw_model = graph._raw_model

    topo = surgeon.get_graph_topology(raw_model.graph)
    topo_usable = topo.max_depth > 0 or len(topo.nodes) == len(graph.nodes)

    if topo_usable:
        node_depths, max_depth, exec_order = _extract_engine_topo(topo, graph)
    else:
        node_depths, max_depth, exec_order = _compute_topology(graph)

    early_thresh = max_depth * 0.20
    late_thresh = max_depth * 0.80

    def _position(depth: int) -> str:
        if max_depth == 0:
            return "middle"
        if depth <= early_thresh:
            return "early"
        if depth >= late_thresh:
            return "late"
        return "middle"

    opset = 0
    for imp in raw_model.opset_import:
        if imp.domain == "" or imp.domain == "ai.onnx":
            opset = imp.version
            break

    param_counts: Dict[str, int] = {}
    if include_weights:
        param_counts = _compute_param_counts(graph)

    report = None
    if include_motifs:
        from graph_surgeon.parsers.onnx_parser import analyze_onnx_graph

        report = analyze_onnx_graph(model_path, verbose=False)

    motif_map, gadget_map = _build_membership_maps(report)

    exec_order_index = {name: i for i, name in enumerate(exec_order)}

    initializer_names: Set[str] = set(graph.initializers.keys())
    graph_input_names: Set[str] = {inp.name for inp in graph.inputs}

    node_profiles = getattr(report, "node_profiles", {}) if report else {}

    nodes: List[SceneNode] = []
    for node in graph.nodes:
        depth = node_depths.get(node.name, 0)
        position = _position(depth)

        op_ref = OPERATOR_REFERENCE_DB.get(node.op_type, {})
        category = op_ref.get("category", "unknown")

        attrs = {}
        for k, v in node.attributes.items():
            attrs[k] = _serialize_attr(v)

        profile = node_profiles.get(node.name)

        nodes.append(SceneNode(
            id=node.name,
            op_type=node.op_type,
            category=category,
            depth=depth,
            position=position,
            exec_index=exec_order_index.get(node.name, 0),
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            attributes=attrs,
            param_count=param_counts.get(node.name, 0),
            motif_ids=sorted(motif_map.get(node.name, [])),
            gadget_ids=sorted(gadget_map.get(node.name, [])),
            gradient_sensitivity=getattr(profile, "gradient_sensitivity", 0.0) if profile else 0.0,
            lipschitz_estimate=getattr(profile, "lipschitz_estimate", 1.0) if profile else 1.0,
            perturbation_amplification=getattr(profile, "perturbation_amplification", 1.0) if profile else 1.0,
            shadowlogic_capacity=getattr(profile, "shadowlogic_capacity", 0.0) if profile else 0.0,
            extraction_leakage=getattr(profile, "extraction_leakage", 0.0) if profile else 0.0,
        ))

    nodes.sort(key=lambda n: n.exec_index)

    edges = _build_edges(graph, initializer_names, graph_input_names)

    motifs: List[SceneMotif] = []
    chains: List[SceneChain] = []
    if report:
        motifs, chains = _extract_motifs_and_chains(report)

    scene_shadowlogic = _extract_shadowlogic(report, {n.id for n in nodes})

    scene_patterns = _extract_structural_patterns(model_path)

    model_info = SceneModelInfo(
        name=graph.name or "unknown",
        format="onnx",
        opset=opset,
        total_nodes=len(graph.nodes),
        max_depth=max_depth,
        inputs=[
            SceneInput(
                name=inp.name,
                shape=list(inp.shape) if inp.shape else [],
                dtype=inp.dtype or "float32",
            )
            for inp in graph.inputs
        ],
        outputs=[
            SceneOutput(
                name=out.name,
                shape=list(out.shape) if out.shape else [],
                dtype=out.dtype or "float32",
            )
            for out in graph.outputs
        ],
        flow_description=report.model_flow_description if report else "",
        source_file=Path(model_path).name,
    )

    return SceneGraph(
        schema_version=SCHEMA_VERSION,
        model=model_info,
        nodes=nodes,
        edges=edges,
        motifs=motifs,
        chains=chains,
        shadowlogic=scene_shadowlogic,
        structural_patterns=scene_patterns,
    )


def _extract_shadowlogic(report, node_ids: Set[str]) -> Optional[SceneShadowLogic]:
    if not report:
        return None
    sl = getattr(report, "shadowlogic_assessment", None)
    if not sl:
        return None

    points = []
    for ip in getattr(sl, "injection_points", []):
        if ip.node_id not in node_ids:
            continue
        points.append(SceneShadowLogicPoint(
            node_id=ip.node_id,
            location=ip.location,
            description=ip.description,
            injection_complexity=ip.injection_complexity,
            detection_difficulty=ip.detection_difficulty,
        ))

    return SceneShadowLogic(
        structural_exposure=sl.susceptibility_score,
        exposure_tier=sl.susceptibility_level,
        conditional_ops=list(sl.conditional_ops_found),
        injection_points=points,
    )


def _extract_structural_patterns(model_path: str) -> Optional[SceneStructuralPatterns]:
    try:
        from graph_surgeon.parsers.onnx_parser import analyze_onnx_patterns

        pat = analyze_onnx_patterns(model_path)
    except Exception:
        return None

    return SceneStructuralPatterns(
        gradient_bottlenecks=list(pat.gradient_bottlenecks),
        feature_fusion_points=list(pat.feature_fusion_points),
        amplification_layers=list(pat.amplification_layers),
        recommended_defense_points=list(pat.recommended_defense_points),
        max_fan_in=pat.max_fan_in,
        max_fan_out=pat.max_fan_out,
        longest_linear_chain=pat.longest_linear_chain,
        structural_score=pat.structural_score,
    )


def _extract_engine_topo(topo, graph):
    """Extract depth info from a working engine topology."""
    node_depths: Dict[str, int] = {}
    for node in graph.nodes:
        nt = topo.nodes.get(node.name)
        node_depths[node.name] = nt.depth if nt else 0
    return node_depths, topo.max_depth, list(topo.execution_order)


def _compute_topology(graph):
    """Compute depth and execution order directly from the parsed ONNXGraph.

    This handles models with unnamed protobuf nodes where the engine's
    topology analysis collapses to a single entry.
    """
    output_to_node: Dict[str, str] = {}
    for node in graph.nodes:
        for out in node.outputs:
            output_to_node[out] = node.name

    input_names = {inp.name for inp in graph.inputs}
    init_names = set(graph.initializers.keys())
    source_names = input_names | init_names

    node_depths: Dict[str, int] = {}

    def get_depth(node_name: str, visited: Set[str]) -> int:
        if node_name in node_depths:
            return node_depths[node_name]
        if node_name in visited:
            return 0
        visited.add(node_name)

        node = None
        for n in graph.nodes:
            if n.name == node_name:
                node = n
                break
        if not node:
            return 0

        max_parent = -1
        for inp in node.inputs:
            if inp in source_names:
                max_parent = max(max_parent, -1)
            elif inp in output_to_node:
                parent = output_to_node[inp]
                max_parent = max(max_parent, get_depth(parent, visited))

        depth = max_parent + 1
        node_depths[node_name] = depth
        return depth

    for node in graph.nodes:
        get_depth(node.name, set())

    max_depth = max(node_depths.values()) if node_depths else 0

    exec_order = sorted(
        [n.name for n in graph.nodes],
        key=lambda name: (node_depths.get(name, 0), name),
    )

    return node_depths, max_depth, exec_order


def _compute_param_counts(graph) -> Dict[str, int]:
    """Map node names to parameter counts from initializer tensors."""
    tensor_to_node: Dict[str, str] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            if inp in graph.initializers:
                tensor_to_node.setdefault(inp, node.name)

    counts: Dict[str, int] = {}
    for tensor_name, node_name in tensor_to_node.items():
        tensor = graph.initializers[tensor_name]
        shape = tensor.shape
        if shape:
            n_params = 1
            for dim in shape:
                if isinstance(dim, int) and dim > 0:
                    n_params *= dim
            counts[node_name] = counts.get(node_name, 0) + n_params
    return counts


def _build_membership_maps(report) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build node->motif_ids and node->gadget_ids maps from a ModelSecurityReport."""
    motif_map: Dict[str, List[str]] = {}
    gadget_map: Dict[str, List[str]] = {}
    if not report:
        return motif_map, gadget_map

    for finding in report.structural_findings:
        fid = finding.registry_id or finding.id
        if finding.node_id:
            motif_map.setdefault(finding.node_id, []).append(fid)

    for gadget in report.gadgets:
        gid = getattr(gadget, "registry_id", None) or getattr(gadget, "id", str(gadget))
        node_id = getattr(gadget, "node_id", None)
        if node_id:
            gadget_map.setdefault(node_id, []).append(gid)

    return motif_map, gadget_map


def _build_edges(graph, initializer_names, graph_input_names) -> List[SceneEdge]:
    """Derive edges by matching tensor names between node outputs and inputs."""
    output_to_node: Dict[str, str] = {}
    for node in graph.nodes:
        for out in node.outputs:
            output_to_node[out] = node.name

    shape_map: Dict[str, List[int]] = {}
    for inp in graph.inputs:
        shape_map[inp.name] = list(inp.shape) if inp.shape else []
    for name, tensor in graph.value_info.items():
        if tensor.shape:
            shape_map[name] = list(tensor.shape)
    for name, tensor in graph.initializers.items():
        if tensor.shape:
            shape_map[name] = list(tensor.shape)

    edges: List[SceneEdge] = []
    seen: Set[Tuple[str, str, str]] = set()

    for node in graph.nodes:
        for inp_name in node.inputs:
            if inp_name in initializer_names:
                continue

            if inp_name in output_to_node:
                source = output_to_node[inp_name]
            elif inp_name in graph_input_names:
                source = inp_name
            else:
                continue

            key = (source, node.name, inp_name)
            if key in seen:
                continue
            seen.add(key)

            shape = shape_map.get(inp_name, [])
            shape = [int(d) if isinstance(d, int) else 0 for d in shape]

            edges.append(SceneEdge(
                source=source,
                target=node.name,
                tensor=inp_name,
                shape=shape,
            ))

    edges.sort(key=lambda e: (e.source, e.target, e.tensor))
    return edges


def _extract_motifs_and_chains(report) -> Tuple[List[SceneMotif], List[SceneChain]]:
    """Pull motifs and chains from a ModelSecurityReport, enriched with registry data."""
    from graph_surgeon.reporting.sanitize import serialize_for_export
    from graph_surgeon.taxonomy.gadget_registry import GADGET_REGISTRY, CHAIN_REGISTRY

    motifs: List[SceneMotif] = []
    chains: List[SceneChain] = []

    seen_motif_ids: Set[str] = set()
    chain_map: Dict[str, SceneChain] = {}

    for finding in report.structural_findings:
        fid = finding.registry_id or finding.id
        if fid in seen_motif_ids:
            continue
        seen_motif_ids.add(fid)

        node_ids = []
        if finding.node_id:
            node_ids.append(finding.node_id)

        safe = serialize_for_export(finding)
        desc = safe.get("description", "") if isinstance(safe, dict) else ""

        if finding.chain_id:
            if finding.chain_id in chain_map:
                chain_map[finding.chain_id].node_ids.extend(node_ids)
                if fid not in chain_map[finding.chain_id].gadget_ids:
                    chain_map[finding.chain_id].gadget_ids.append(fid)
            else:
                chain_def = CHAIN_REGISTRY.get(finding.chain_id, {})
                chain_map[finding.chain_id] = SceneChain(
                    id=finding.chain_id,
                    node_ids=node_ids,
                    gadget_ids=[fid],
                    title=chain_def.get("name", finding.chain_id),
                    description=chain_def.get("notes", ""),
                    structural_significance=chain_def.get("structural_significance", ""),
                    research_basis=_make_paper_refs(chain_def.get("research_basis", [])),
                )
        else:
            gadget = GADGET_REGISTRY.get(fid)
            if gadget:
                motifs.append(SceneMotif(
                    id=fid,
                    title=finding.title,
                    node_ids=node_ids,
                    description=desc,
                    catalog_ref=fid,
                    attacks_enabled=list(gadget.attacks_enabled),
                    structural_significance=gadget.structural_significance,
                    confidence=gadget.confidence,
                    category=gadget.category.value if hasattr(gadget.category, "value") else str(gadget.category),
                    research_basis=_make_paper_refs(gadget.research_basis),
                    detection_logic=gadget.detection_logic,
                ))
            else:
                motifs.append(SceneMotif(
                    id=fid,
                    title=finding.title,
                    node_ids=node_ids,
                    description=desc,
                    catalog_ref=fid,
                ))

    chains = list(chain_map.values())
    return motifs, chains


def _make_paper_refs(slugs: List[str]) -> List[PaperRef]:
    """Convert paper slug strings to PaperRef objects with human-readable titles."""
    refs = []
    for slug in slugs:
        title = slug.replace("-", " ").replace("_", " ")
        refs.append(PaperRef(slug=slug, title=title))
    return refs


def _serialize_attr(value: Any) -> Any:
    """Coerce an ONNX attribute value to something JSON-serializable."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_attr(v) for v in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return str(value)
