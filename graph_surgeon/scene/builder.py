"""Build a SceneGraph JSON structure from an ONNX model using the existing engine."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_surgeon.scene.schema import (
    SCHEMA_VERSION,
    SceneChain,
    SceneEdge,
    SceneGraph,
    SceneInput,
    SceneModelInfo,
    SceneMotif,
    SceneNode,
    SceneOutput,
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
    from graph_surgeon.graph.topology import LayerPosition
    from graph_surgeon.parsers.onnx_parser import ONNXGraphParser

    parser = ONNXGraphParser()
    graph = parser.parse_file(model_path)
    surgeon = GraphSurgeon(verbose=False)
    raw_model = graph._raw_model
    topo = surgeon.get_graph_topology(raw_model.graph)

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
        from graph_surgeon.reporting.sanitize import serialize_for_export

        report = analyze_onnx_graph(model_path, verbose=False)

    motif_map, gadget_map = _build_membership_maps(report)

    exec_order_index = {
        name: i for i, name in enumerate(topo.execution_order)
    }

    initializer_names: Set[str] = set(graph.initializers.keys())
    graph_input_names: Set[str] = {inp.name for inp in graph.inputs}

    nodes: List[SceneNode] = []
    for node in graph.nodes:
        node_topo = topo.nodes.get(node.name)
        depth = node_topo.depth if node_topo else 0
        position = node_topo.position.value if node_topo else "middle"

        op_ref = OPERATOR_REFERENCE_DB.get(node.op_type, {})
        category = op_ref.get("category", "unknown")

        attrs = {}
        for k, v in node.attributes.items():
            attrs[k] = _serialize_attr(v)

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
        ))

    nodes.sort(key=lambda n: n.exec_index)

    edges = _build_edges(graph, topo, initializer_names, graph_input_names)

    motifs: List[SceneMotif] = []
    chains: List[SceneChain] = []
    if report:
        motifs, chains = _extract_motifs_and_chains(report)

    model_info = SceneModelInfo(
        name=graph.name or "unknown",
        format="onnx",
        opset=opset,
        total_nodes=len(graph.nodes),
        max_depth=topo.max_depth,
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
    )

    return SceneGraph(
        schema_version=SCHEMA_VERSION,
        model=model_info,
        nodes=nodes,
        edges=edges,
        motifs=motifs,
        chains=chains,
    )


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
    """Build node→motif_ids and node→gadget_ids maps from a ModelSecurityReport."""
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


def _build_edges(graph, topo, initializer_names, graph_input_names) -> List[SceneEdge]:
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
    """Pull motifs and chains from a ModelSecurityReport, sanitized."""
    from graph_surgeon.reporting.sanitize import serialize_for_export

    motifs: List[SceneMotif] = []
    chains: List[SceneChain] = []

    seen_motif_ids: Set[str] = set()
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
            chains.append(SceneChain(
                id=finding.chain_id,
                node_ids=node_ids,
                gadget_ids=[fid],
            ))
        else:
            motifs.append(SceneMotif(
                id=fid,
                title=finding.title,
                node_ids=node_ids,
                description=desc,
                catalog_ref=fid,
            ))

    return motifs, chains


def _serialize_attr(value: Any) -> Any:
    """Coerce an ONNX attribute value to something JSON-serializable."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_attr(v) for v in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return str(value)
