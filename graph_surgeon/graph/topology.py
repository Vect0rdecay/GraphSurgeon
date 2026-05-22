"""Graph topology analysis for ONNX DAGs."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class LayerPosition(Enum):
    """Position of a layer within the network topology."""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


@dataclass
class GraphTopologyConfig:
    """Configurable thresholds for early/middle/late classification."""
    early_fraction: float = 0.20
    late_fraction: float = 0.80

    def early_depth_threshold(self, max_depth: int) -> int:
        """Maximum depth (inclusive) considered early in the network."""
        if max_depth <= 0:
            return 0
        return int(max_depth * self.early_fraction)

    def late_depth_threshold(self, max_depth: int) -> int:
        """Minimum depth (inclusive) considered late in the network."""
        if max_depth <= 0:
            return 0
        return int(max_depth * self.late_fraction)

    def is_early(self, depth: int, max_depth: int) -> bool:
        return depth <= self.early_depth_threshold(max_depth)

    def is_late(self, depth: int, max_depth: int) -> bool:
        return depth >= self.late_depth_threshold(max_depth)

    @property
    def early_threshold_expr(self):
        return self.early_fraction

    @property
    def late_threshold_expr(self):
        return self.late_fraction


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
