"""Graph validation levels for counterfactual edit verification."""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

if TYPE_CHECKING:
    import onnx as onnx_types


class GraphValidationLevel(Enum):
    NONE = "none"
    STRUCTURAL = "structural"
    LOADABLE = "loadable"
    RUNNABLE = "runnable"


@dataclass
class GraphValidationResult:
    valid: bool
    level: GraphValidationLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inference_output_shape: Optional[Tuple] = None


class GraphValidator:
    """Validation and shape inference helpers."""

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg: None)

    def validate(
        self,
        model: "onnx_types.ModelProto",
        level: GraphValidationLevel = GraphValidationLevel.STRUCTURAL,
        sample_input: Optional[np.ndarray] = None,
    ) -> GraphValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if level.value != "none":
            try:
                onnx.checker.check_model(model)
            except Exception as e:
                errors.append(f"ONNX checker failed: {e}")
                return GraphValidationResult(
                    valid=False,
                    level=GraphValidationLevel.STRUCTURAL,
                    errors=errors,
                )

        if level == GraphValidationLevel.STRUCTURAL:
            return GraphValidationResult(valid=True, level=level)

        if level.value in ["loadable", "runnable"]:
            if not ONNXRUNTIME_AVAILABLE:
                warnings.append("onnxruntime not available, skipping load test")
            else:
                try:
                    model_bytes = model.SerializeToString()
                    ort.InferenceSession(model_bytes)
                except Exception as e:
                    errors.append(f"Failed to load model: {e}")
                    return GraphValidationResult(
                        valid=False,
                        level=GraphValidationLevel.LOADABLE,
                        errors=errors,
                        warnings=warnings,
                    )

        if level == GraphValidationLevel.LOADABLE:
            return GraphValidationResult(valid=True, level=level, warnings=warnings)

        if level == GraphValidationLevel.RUNNABLE:
            if not ONNXRUNTIME_AVAILABLE:
                warnings.append("onnxruntime not available, skipping inference test")
                return GraphValidationResult(valid=True, level=level, warnings=warnings)
            try:
                model_bytes = model.SerializeToString()
                session = ort.InferenceSession(model_bytes)
                if sample_input is None:
                    input_info = session.get_inputs()[0]
                    shape = []
                    for dim in input_info.shape:
                        if isinstance(dim, int) and dim > 0:
                            shape.append(dim)
                        else:
                            shape.append(1 if len(shape) == 0 else 224)
                    sample_input = np.random.randn(*shape).astype(np.float32)
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: sample_input})
                return GraphValidationResult(
                    valid=True,
                    level=level,
                    warnings=warnings,
                    inference_output_shape=output[0].shape,
                )
            except Exception as e:
                errors.append(f"Inference failed: {e}")
                return GraphValidationResult(
                    valid=False,
                    level=GraphValidationLevel.RUNNABLE,
                    errors=errors,
                    warnings=warnings,
                )

        return GraphValidationResult(valid=True, level=level, warnings=warnings)

    def infer_shapes(self, model: "onnx_types.ModelProto") -> "onnx_types.ModelProto":
        try:
            from onnx import shape_inference
            return shape_inference.infer_shapes(model)
        except Exception as e:
            self._log(f"Shape inference failed: {e}")
            return model

    def get_tensor_shape(self, model: "onnx_types.ModelProto", tensor_name: str) -> Optional[List[int]]:
        for inp in model.graph.input:
            if inp.name == tensor_name:
                return self._extract_shape(inp.type)
        for out in model.graph.output:
            if out.name == tensor_name:
                return self._extract_shape(out.type)
        for vi in model.graph.value_info:
            if vi.name == tensor_name:
                return self._extract_shape(vi.type)
        return None

    def _extract_shape(self, type_proto) -> Optional[List[int]]:
        if not type_proto.HasField("tensor_type"):
            return None
        tensor_type = type_proto.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(-1)
            else:
                shape.append(-1)
        return shape

    def check_shape_compatibility(
        self, model: "onnx_types.ModelProto", tensor_a: str, tensor_b: str
    ) -> Tuple[bool, str]:
        shape_a = self.get_tensor_shape(model, tensor_a)
        shape_b = self.get_tensor_shape(model, tensor_b)
        if shape_a is None:
            return False, f"Shape unknown for {tensor_a}"
        if shape_b is None:
            return False, f"Shape unknown for {tensor_b}"
        if len(shape_a) != len(shape_b):
            return False, f"Rank mismatch: {shape_a} vs {shape_b}"
        for i, (a, b) in enumerate(zip(shape_a, shape_b)):
            if a == -1 or b == -1:
                continue
            if a != b:
                return False, f"Dimension {i} mismatch: {a} vs {b}"
        return True, "Shapes compatible"
