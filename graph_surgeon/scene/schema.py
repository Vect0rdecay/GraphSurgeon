"""SceneGraph dataclasses — the JSON contract between Python and the 3D frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.2"


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
    source_file: str = ""


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
    gradient_sensitivity: float = 0.0
    lipschitz_estimate: float = 1.0
    perturbation_amplification: float = 1.0
    shadowlogic_capacity: float = 0.0
    extraction_leakage: float = 0.0


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
class SceneShadowLogicPoint:
    node_id: str
    location: str
    description: str
    injection_complexity: str
    detection_difficulty: str


@dataclass
class SceneShadowLogic:
    structural_exposure: float = 0.0
    exposure_tier: str = ""
    conditional_ops: List[str] = field(default_factory=list)
    injection_points: List[SceneShadowLogicPoint] = field(default_factory=list)


@dataclass
class SceneStructuralPatterns:
    gradient_bottlenecks: List[str] = field(default_factory=list)
    feature_fusion_points: List[str] = field(default_factory=list)
    amplification_layers: List[str] = field(default_factory=list)
    recommended_defense_points: List[str] = field(default_factory=list)
    max_fan_in: int = 0
    max_fan_out: int = 0
    longest_linear_chain: int = 0
    structural_score: float = 0.0


@dataclass
class SceneGraph:
    schema_version: str
    model: SceneModelInfo
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    motifs: List[SceneMotif] = field(default_factory=list)
    chains: List[SceneChain] = field(default_factory=list)
    shadowlogic: Optional[SceneShadowLogic] = None
    structural_patterns: Optional[SceneStructuralPatterns] = None

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
