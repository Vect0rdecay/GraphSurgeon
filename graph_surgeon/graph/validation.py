"""Graph validation levels for counterfactual edits."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

try:
    import onnx

    from graph_surgeon._env import import_onnxruntime

    ort = import_onnxruntime()
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

import numpy as np


class GraphValidationLevel(Enum):
    """Level of validation to perform on modified graphs."""
    NONE = "none"
    STRUCTURAL = "structural"
    LOADABLE = "loadable"
    RUNNABLE = "runnable"


@dataclass
class GraphValidationResult:
    """Result of graph validation."""
    valid: bool
    level: GraphValidationLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inference_output_shape: Optional[Tuple] = None


# Backward-compatible aliases (Carcinoma sync period)
ValidationLevel = GraphValidationLevel
ValidationResult = GraphValidationResult
