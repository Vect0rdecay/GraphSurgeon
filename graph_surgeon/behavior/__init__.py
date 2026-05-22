"""Optional ONNX runtime helpers (onnxruntime CPU).

Not required for graph structure RE (inspect, topology, motifs, edit).
Install with graph-surgeon[dev] or pip install onnxruntime separately.
"""

from graph_surgeon.behavior.weight_signature import analyze_onnx_weights

__all__ = ["analyze_onnx_weights"]
