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
