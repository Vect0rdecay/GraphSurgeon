"""Graph edit primitives and surgery results."""
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class SurgeryResult:
    """Result of a graph surgery operation."""
    success: bool
    graph: Optional['onnx.GraphProto']
    message: str
    nodes_added: List[str] = field(default_factory=list)
    nodes_removed: List[str] = field(default_factory=list)
    nodes_modified: List[str] = field(default_factory=list)
    edges_rewired: int = 0

