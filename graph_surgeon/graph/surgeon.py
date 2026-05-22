"""GraphSurgeon orchestrator: load, topology, query, edit, validate."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional

import numpy as np

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from graph_surgeon.graph.edits import GraphEdits, SurgeryResult
from graph_surgeon.graph.query import GraphQuery
from graph_surgeon.graph.topology import (
    GraphTopology,
    GraphTopologyConfig,
    LayerPosition,
    NodeTopology,
    build_graph_topology,
)
from graph_surgeon.graph.validation import (
    GraphValidationLevel,
    GraphValidationResult,
    GraphValidator,
)

if TYPE_CHECKING:
    import onnx as onnx_types

# Public re-exports
__all__ = [
    "GraphSurgeon",
    "GraphTopology",
    "GraphTopologyConfig",
    "GraphValidationLevel",
    "GraphValidationResult",
    "LayerPosition",
    "NodeTopology",
    "SurgeryResult",
]


class GraphSurgeon(GraphQuery, GraphEdits):
    """
    Core graph manipulation engine for ONNX models.

    Provides topology analysis, node queries, counterfactual edits,
    and validation for reverse-engineering workflows.
    """

    def __init__(self, verbose: bool = True, topology_config: Optional[GraphTopologyConfig] = None):
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package required. Install with: pip install onnx")
        self.verbose = verbose
        self.topology_config = topology_config or GraphTopologyConfig()
        self._validator = GraphValidator(log_fn=self.log)

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[GraphSurgeon] {message}")

    def load_model(self, model_path: str) -> "onnx_types.ModelProto":
        self.log(f"Loading model: {model_path}")
        return onnx.load(model_path)

    def save_model(self, model: "onnx_types.ModelProto", output_path: str) -> None:
        self.log(f"Saving model: {output_path}")
        onnx.save(model, output_path)

    def clone_model(self, model: "onnx_types.ModelProto") -> "onnx_types.ModelProto":
        return copy.deepcopy(model)

    def get_graph_topology(self, graph: "onnx_types.GraphProto") -> GraphTopology:
        return build_graph_topology(graph, self.get_node_by_name, self.topology_config)

    def get_early_layers(
        self, graph: "onnx_types.GraphProto", op_type: Optional[str] = None
    ) -> List["onnx_types.NodeProto"]:
        topology = self.get_graph_topology(graph)
        early_names = topology.by_position[LayerPosition.EARLY]
        result = []
        for name in early_names:
            node = self.get_node_by_name(graph, name)
            if node and (op_type is None or node.op_type == op_type):
                result.append(node)
        return result

    def get_late_layers(
        self, graph: "onnx_types.GraphProto", op_type: Optional[str] = None
    ) -> List["onnx_types.NodeProto"]:
        topology = self.get_graph_topology(graph)
        late_names = topology.by_position[LayerPosition.LATE]
        result = []
        for name in late_names:
            node = self.get_node_by_name(graph, name)
            if node and (op_type is None or node.op_type == op_type):
                result.append(node)
        return result

    def validate(
        self,
        model: "onnx_types.ModelProto",
        level: GraphValidationLevel = GraphValidationLevel.STRUCTURAL,
        sample_input: Optional[np.ndarray] = None,
    ) -> GraphValidationResult:
        return self._validator.validate(model, level=level, sample_input=sample_input)

    def infer_shapes(self, model: "onnx_types.ModelProto") -> "onnx_types.ModelProto":
        return self._validator.infer_shapes(model)

    def get_tensor_shape(self, model: "onnx_types.ModelProto", tensor_name: str):
        return self._validator.get_tensor_shape(model, tensor_name)

    def check_shape_compatibility(self, model: "onnx_types.ModelProto", tensor_a: str, tensor_b: str):
        return self._validator.check_shape_compatibility(model, tensor_a, tensor_b)
