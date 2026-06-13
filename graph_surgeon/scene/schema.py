"""SceneGraph dataclasses — the JSON contract between Python and the 3D frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.1"


@dataclass
class SceneInput:
    name: str
    shape: List[int]
    dtype: str


@dataclass
class SceneOutput:
    name: str
    shape: List[int]
    dtype: str


@dataclass
class SceneModelInfo:
    name: str
    format: str
    opset: int
    total_nodes: int
    max_depth: int
    inputs: List[SceneInput]
    outputs: List[SceneOutput]
    flow_description: str = ""


@dataclass
class SceneNode:
    id: str
    op_type: str
    category: str
    depth: int
    position: str
    exec_index: int
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    param_count: int = 0
    motif_ids: List[str] = field(default_factory=list)
    gadget_ids: List[str] = field(default_factory=list)


@dataclass
class SceneEdge:
    source: str
    target: str
    tensor: str
    shape: List[int] = field(default_factory=list)


@dataclass
class PaperRef:
    slug: str
    title: str


@dataclass
class SceneMotif:
    id: str
    title: str
    node_ids: List[str]
    description: str = ""
    catalog_ref: str = ""
    attacks_enabled: List[str] = field(default_factory=list)
    structural_significance: str = ""
    confidence: str = ""
    category: str = ""
    research_basis: List[PaperRef] = field(default_factory=list)
    detection_logic: str = ""


@dataclass
class SceneChain:
    id: str
    node_ids: List[str]
    gadget_ids: List[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    structural_significance: str = ""
    research_basis: List[PaperRef] = field(default_factory=list)


@dataclass
class SceneGraph:
    schema_version: str
    model: SceneModelInfo
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    motifs: List[SceneMotif] = field(default_factory=list)
    chains: List[SceneChain] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON export."""
        return _as_dict(self)


def _as_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    return obj
