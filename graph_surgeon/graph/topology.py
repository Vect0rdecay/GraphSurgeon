"""Graph topology analysis for ONNX computational DAGs."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import onnx


class LayerPosition(Enum):
    """Position of a layer within the network topology."""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


@dataclass
class GraphTopologyConfig:
    """Configurable depth thresholds for early/middle/late classification."""
    early_fraction: float = 0.20
    late_fraction: float = 0.80
    # Legacy structural_patterns often used max_depth // 3 (~0.33)
    pattern_early_divisor: int = 3

    @property
    def pattern_early_threshold(self) -> float:
        return 1.0 / self.pattern_early_divisor


@dataclass
class NodeTopology:
    """Topology information for a single node."""
    name: str
    op_type: str
    depth: int
    position: LayerPosition
    inputs: List[str]
    outputs: List[str]


@dataclass
class GraphTopology:
    """Complete topology analysis of a graph."""
    total_nodes: int
    max_depth: int
    nodes: Dict[str, NodeTopology]
    by_position: Dict[LayerPosition, List[str]]
    by_op_type: Dict[str, List[str]]
    execution_order: List[str]
    config: GraphTopologyConfig


def classify_position(depth: int, max_depth: int, config: GraphTopologyConfig) -> LayerPosition:
    if max_depth <= 0:
        return LayerPosition.EARLY
    early_threshold = max_depth * config.early_fraction
    late_threshold = max_depth * config.late_fraction
    if depth <= early_threshold:
        return LayerPosition.EARLY
    if depth >= late_threshold:
        return LayerPosition.LATE
    return LayerPosition.MIDDLE


def compute_node_depths(graph: "onnx.GraphProto", get_node_by_name) -> Dict[str, int]:
    tensor_producers: Dict[str, str] = {}
    for node in graph.node:
        for output in node.output:
            tensor_producers[output] = node.name

    input_tensors = {inp.name for inp in graph.input}
    node_depths: Dict[str, int] = {}

    def get_node_depth(node_name: str) -> int:
        if node_name in node_depths:
            return node_depths[node_name]
        node = get_node_by_name(graph, node_name)
        if not node:
            return 0
        max_input_depth = -1
        for inp in node.input:
            if inp in input_tensors:
                max_input_depth = max(max_input_depth, -1)
            elif inp in tensor_producers:
                producer_depth = get_node_depth(tensor_producers[inp])
                max_input_depth = max(max_input_depth, producer_depth)
        depth = max_input_depth + 1
        node_depths[node_name] = depth
        return depth

    for node in graph.node:
        get_node_depth(node.name)
    return node_depths


def build_graph_topology(
    graph: "onnx.GraphProto",
    get_node_by_name,
    config: Optional[GraphTopologyConfig] = None,
) -> GraphTopology:
    config = config or GraphTopologyConfig()
    node_depths = compute_node_depths(graph, get_node_by_name)
    max_depth = max(node_depths.values()) if node_depths else 0

    nodes: Dict[str, NodeTopology] = {}
    by_position: Dict[LayerPosition, List[str]] = {
        LayerPosition.EARLY: [],
        LayerPosition.MIDDLE: [],
        LayerPosition.LATE: [],
    }
    by_op_type: Dict[str, List[str]] = {}

    for node in graph.node:
        depth = node_depths.get(node.name, 0)
        position = classify_position(depth, max_depth, config)
        node_info = NodeTopology(
            name=node.name,
            op_type=node.op_type,
            depth=depth,
            position=position,
            inputs=list(node.input),
            outputs=list(node.output),
        )
        nodes[node.name] = node_info
        by_position[position].append(node.name)
        by_op_type.setdefault(node.op_type, []).append(node.name)

    execution_order = sorted(
        [node.name for node in graph.node],
        key=lambda n: node_depths.get(n, 0),
    )

    return GraphTopology(
        total_nodes=len(graph.node),
        max_depth=max_depth,
        nodes=nodes,
        by_position=by_position,
        by_op_type=by_op_type,
        execution_order=execution_order,
        config=config,
    )
