"""
Model architecture inference from ONNX graph structure.

Infers hidden dimension, vocab size, MLP intermediate size, max position
embeddings, RoPE theta, tied embeddings, and conv groups from initializer
shapes, node attributes, and shared tensor references.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Set
from collections import Counter, defaultdict
import math

from graph_surgeon.parsers.onnx_parser import ONNXGraph, ONNXNode, ONNXTensor


@dataclass
class ArchitectureInfo:
    """Inferred model architecture parameters. None means not determinable."""
    hidden_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    mlp_intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None
    tied_embeddings: Optional[bool] = None
    tied_initializer_names: Optional[List[str]] = None
    conv_groups: Optional[int] = None
    quantization_format: Optional[str] = None

    def to_dict(self) -> dict:
        """Return only non-None fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def infer_architecture(graph: ONNXGraph) -> ArchitectureInfo:
    """Infer model architecture details from ONNX graph structure."""
    info = ArchitectureInfo()

    node_order = {n.name: i for i, n in enumerate(graph.nodes)}
    init_to_nodes = _build_init_to_nodes_map(graph)

    info.quantization_format = _infer_quantization(graph)
    info.hidden_dim = _infer_hidden_dim(graph)
    info.vocab_size = _infer_vocab_size(graph, init_to_nodes, node_order)
    info.max_position_embeddings = _infer_max_position_embeddings(graph)
    info.rope_theta = _infer_rope_theta(graph)
    info.mlp_intermediate_size = _infer_mlp_size(graph, info.hidden_dim, info.vocab_size)
    info.conv_groups = _infer_conv_groups(graph)

    tied, tied_names = _infer_tied_embeddings(graph, init_to_nodes, node_order)
    info.tied_embeddings = tied if tied else None
    info.tied_initializer_names = tied_names if tied_names else None

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_init_to_nodes_map(graph: ONNXGraph) -> Dict[str, List[ONNXNode]]:
    """Map each initializer name to the list of nodes that consume it."""
    mapping: Dict[str, List[ONNXNode]] = defaultdict(list)
    for node in graph.nodes:
        for inp in node.inputs:
            if inp in graph.initializers:
                mapping[inp].append(node)
    return dict(mapping)


_NORM_OP_TYPES = {
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "RMSNormalization",
    "SkipSimplifiedLayerNormalization",
    "SkipLayerNormalization",
}


def _infer_hidden_dim(graph: ONNXGraph) -> Optional[int]:
    """Infer hidden dimension from normalization layer weight shapes."""
    # Strategy 1: 1D norm weight shapes (most reliable)
    norm_dims = []
    for node in graph.nodes:
        if node.op_type in _NORM_OP_TYPES:
            for inp in node.inputs:
                tensor = graph.initializers.get(inp)
                if tensor and len(tensor.shape) == 1 and tensor.shape[0] > 1:
                    norm_dims.append(tensor.shape[0])

    if norm_dims:
        counts = Counter(norm_dims)
        return counts.most_common(1)[0][0]

    # Strategy 2: most common MatMul/Gemm weight dimension
    matmul_dims = []
    for node in graph.nodes:
        if node.op_type in ("MatMul", "Gemm"):
            for inp in node.inputs:
                tensor = graph.initializers.get(inp)
                if tensor and len(tensor.shape) == 2:
                    matmul_dims.extend(tensor.shape)
        elif node.op_type == "MatMulNBits":
            k = node.attributes.get("K")
            n = node.attributes.get("N")
            if k:
                matmul_dims.append(k)
            if n:
                matmul_dims.append(n)

    if matmul_dims:
        counts = Counter(matmul_dims)
        # Most common dim that's not unusually large (likely vocab-sized)
        for dim, _ in counts.most_common():
            if dim < 32768:
                return dim

    # Strategy 3: Conv output channels (CNN-only fallback)
    conv_channels = []
    for node in graph.nodes:
        if node.op_type == "Conv":
            for inp in node.inputs[1:2]:
                tensor = graph.initializers.get(inp)
                if tensor and len(tensor.shape) >= 2:
                    conv_channels.append(tensor.shape[0])
    if conv_channels:
        counts = Counter(conv_channels)
        return counts.most_common(1)[0][0]

    return None


def _infer_vocab_size(
    graph: ONNXGraph,
    init_to_nodes: Dict[str, List[ONNXNode]],
    node_order: Dict[str, int],
) -> Optional[int]:
    """Infer vocabulary size from embedding and LM head weights."""
    candidates = []

    # Strategy 1: Gather/GatherBlockQuantized with large first dimension
    for node in graph.nodes:
        if node.op_type in ("Gather", "GatherBlockQuantized"):
            data_input = node.inputs[0] if node.inputs else None
            if data_input and data_input in graph.initializers:
                tensor = graph.initializers[data_input]
                if len(tensor.shape) >= 2 and tensor.shape[0] > 1000:
                    candidates.append(tensor.shape[0])

    # Strategy 2: final MatMul/MatMulNBits/Gemm output dimension (LM head)
    if graph.nodes:
        for node in reversed(graph.nodes):
            if node.op_type in ("MatMul", "Gemm"):
                for inp in node.inputs:
                    tensor = graph.initializers.get(inp)
                    if tensor and len(tensor.shape) == 2:
                        # Weight is [in, out] for MatMul; large out = vocab
                        large_dim = max(tensor.shape)
                        if large_dim > 1000:
                            candidates.append(large_dim)
                            break
                if candidates and candidates[-1] > 1000:
                    break
            elif node.op_type == "MatMulNBits":
                n = node.attributes.get("N")
                if n and n > 1000:
                    candidates.append(n)
                    break

    # Strategy 3: embedding weight first dimension directly
    for name, tensor in graph.initializers.items():
        name_lower = name.lower()
        if "embed" in name_lower and "token" in name_lower:
            if len(tensor.shape) >= 2 and tensor.shape[0] > 1000:
                candidates.append(tensor.shape[0])

    if not candidates:
        return None

    # Cross-validate: return the most common candidate
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def _infer_max_position_embeddings(graph: ONNXGraph) -> Optional[int]:
    """Infer max position embeddings from cos/sin cache or position embedding shapes."""
    _CACHE_PATTERNS = ("cos_cache", "sin_cache", "cos_cached", "sin_cached")
    _POS_PATTERNS = ("pos_embed", "position_embed", "wpe")

    for name, tensor in graph.initializers.items():
        name_lower = name.lower()
        for pattern in _CACHE_PATTERNS:
            if pattern in name_lower and len(tensor.shape) >= 2:
                return tensor.shape[0]
        for pattern in _POS_PATTERNS:
            if pattern in name_lower and len(tensor.shape) >= 2:
                return tensor.shape[0] if tensor.shape[0] > tensor.shape[1] else tensor.shape[1]

    # Fallback: 2D initializer with first dim >> second dim, not used in compute
    for name, tensor in graph.initializers.items():
        if (
            len(tensor.shape) == 2
            and tensor.shape[0] > 1000
            and tensor.shape[0] > tensor.shape[1] * 10
        ):
            name_lower = name.lower()
            if any(kw in name_lower for kw in ("cos", "sin", "pos", "freq", "rotary")):
                return tensor.shape[0]

    return None


def _infer_rope_theta(graph: ONNXGraph) -> Optional[float]:
    """Recover RoPE theta from precomputed cos/sin cache tensors."""
    try:
        return _rope_theta_from_cache(graph)
    except Exception:
        pass

    try:
        return _rope_theta_from_inv_freq(graph)
    except Exception:
        pass

    try:
        return _rope_theta_from_node_attrs(graph)
    except Exception:
        pass

    return None


def _rope_theta_from_cache(graph: ONNXGraph) -> Optional[float]:
    """Extract theta by reverse-engineering cos_cache tensor values."""
    if graph._raw_model is None:
        return None

    import numpy as np
    import onnx

    cos_init = None
    for init in graph._raw_model.graph.initializer:
        if "cos" in init.name.lower() and "cache" in init.name.lower():
            cos_init = init
            break

    if cos_init is None:
        return None

    arr = onnx.numpy_helper.to_array(cos_init).astype(np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None

    d = arr.shape[1] * 2  # head_dim = 2 * num_freq_components

    # Recover frequencies from row 1: cos_cache[1, i] = cos(freq_i)
    # freq_i = 1.0 / (theta ^ (2i / d))
    thetas = []
    for i in range(1, arr.shape[1]):
        cos_val = float(arr[1, i])
        if abs(cos_val) >= 1.0:
            continue
        freq_i = math.acos(cos_val)
        if freq_i <= 0:
            continue
        exponent = (2 * i) / d
        if exponent <= 0 or exponent >= 1:
            continue
        theta = (1.0 / freq_i) ** (1.0 / exponent)
        if 100 < theta < 1e9:
            thetas.append(theta)

    if not thetas:
        return None

    thetas.sort()
    median_theta = thetas[len(thetas) // 2]
    return _snap_theta(median_theta)


_COMMON_THETAS = [10000, 100000, 500000, 1000000, 5000000, 10000000]


def _snap_theta(raw: float) -> float:
    """Snap a recovered theta to common values if within 1% tolerance."""
    for known in _COMMON_THETAS:
        if abs(raw - known) / known < 0.01:
            return float(known)
    return round(raw)


def _rope_theta_from_inv_freq(graph: ONNXGraph) -> Optional[float]:
    """Extract theta from inv_freq tensor if present."""
    if graph._raw_model is None:
        return None

    import numpy as np
    import onnx

    for init in graph._raw_model.graph.initializer:
        name_lower = init.name.lower()
        if any(kw in name_lower for kw in ("inv_freq", "freqs", "rope_freq")):
            arr = onnx.numpy_helper.to_array(init).astype(np.float64).flatten()
            if len(arr) < 2:
                continue
            d = len(arr) * 2
            # inv_freq[0] = 1 / theta^(0/d) = 1.0
            # inv_freq[1] = 1 / theta^(2/d)
            freq_1 = float(arr[1])
            if freq_1 <= 0 or freq_1 >= 1.0:
                continue
            exponent = 2.0 / d
            theta = (1.0 / freq_1) ** (1.0 / exponent)
            if 100 < theta < 1e9:
                return _snap_theta(theta)
    return None


def _rope_theta_from_node_attrs(graph: ONNXGraph) -> Optional[float]:
    """Check RotaryEmbedding node attributes for theta."""
    for node in graph.nodes:
        if "rotary" in node.op_type.lower() or "rope" in node.op_type.lower():
            for key in ("theta", "base", "rope_theta"):
                val = node.attributes.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return float(val)
    return None


def _infer_tied_embeddings(
    graph: ONNXGraph,
    init_to_nodes: Dict[str, List[ONNXNode]],
    node_order: Dict[str, int],
) -> Tuple[bool, Optional[List[str]]]:
    """Detect whether embedding and LM head weights are tied."""
    _EMBED_OPS = {"Gather", "GatherBlockQuantized", "Embedding"}
    _HEAD_OPS = {"MatMul", "MatMulNBits", "Gemm"}
    num_nodes = len(graph.nodes)

    shared_names = []
    for init_name, consumers in init_to_nodes.items():
        if len(consumers) < 2:
            continue
        has_embed = any(n.op_type in _EMBED_OPS for n in consumers)
        has_head = any(
            n.op_type in _HEAD_OPS and node_order.get(n.name, 0) > num_nodes * 0.8
            for n in consumers
        )
        if has_embed and has_head:
            shared_names.append(init_name)

    if shared_names:
        return True, sorted(shared_names)

    # Fallback: check if same base name appears in both embedding and LM head
    embed_bases: Set[str] = set()
    head_bases: Set[str] = set()
    for node in graph.nodes:
        if node.op_type in _EMBED_OPS:
            for inp in node.inputs:
                if inp in graph.initializers:
                    embed_bases.add(_base_weight_name(inp))
        elif node.op_type in _HEAD_OPS and node_order.get(node.name, 0) > num_nodes * 0.8:
            for inp in node.inputs:
                if inp in graph.initializers:
                    head_bases.add(_base_weight_name(inp))

    overlap = embed_bases & head_bases
    if overlap:
        matching = []
        for init_name in graph.initializers:
            if _base_weight_name(init_name) in overlap:
                matching.append(init_name)
        return True, sorted(matching) if matching else None

    return False, None


def _base_weight_name(name: str) -> str:
    """Strip quantization suffixes to get the base weight name."""
    for suffix in ("_quant", "_quant_matmul", "_scales", "_zp", "_zero_point",
                    ".weight", "_weight", ".bias", "_bias"):
        name = name.removesuffix(suffix)
    return name


def _infer_mlp_size(
    graph: ONNXGraph,
    hidden_dim: Optional[int],
    vocab_size: Optional[int],
) -> Optional[int]:
    """Infer MLP intermediate size from gate/up projection weight shapes."""
    candidates = []

    for node in graph.nodes:
        name_lower = node.name.lower()
        is_mlp_proj = any(kw in name_lower for kw in ("gate_proj", "up_proj", "fc1", "c_fc", "wi"))

        if not is_mlp_proj:
            continue

        dims = _get_matmul_dims(node, graph)
        if dims is None:
            continue

        k, n = dims
        # The larger dimension that isn't hidden_dim or vocab_size is the intermediate size
        for d in (k, n):
            if d == hidden_dim or d == vocab_size:
                continue
            if hidden_dim and d > hidden_dim:
                candidates.append(d)
            elif not hidden_dim and d > 256:
                candidates.append(d)

    if candidates:
        counts = Counter(candidates)
        return counts.most_common(1)[0][0]

    # Fallback: look for MatMul->Mul->Add pattern (SwiGLU / gated MLP)
    if hidden_dim:
        for i, node in enumerate(graph.nodes):
            if node.op_type in ("MatMul", "Gemm", "MatMulNBits"):
                dims = _get_matmul_dims(node, graph)
                if dims is None:
                    continue
                k, n = dims
                if k == hidden_dim and n > hidden_dim and n != vocab_size:
                    # Check if next ops form a gated pattern (Mul, Sigmoid, Add)
                    if i + 2 < len(graph.nodes):
                        next_ops = [graph.nodes[j].op_type for j in range(i + 1, min(i + 4, len(graph.nodes)))]
                        if "Mul" in next_ops or "Sigmoid" in next_ops:
                            candidates.append(n)

        if candidates:
            counts = Counter(candidates)
            return counts.most_common(1)[0][0]

    return None


def _get_matmul_dims(node: ONNXNode, graph: ONNXGraph) -> Optional[Tuple[int, int]]:
    """Get logical (K, N) dimensions for a MatMul-like node."""
    if node.op_type == "MatMulNBits":
        k = node.attributes.get("K")
        n = node.attributes.get("N")
        if k and n:
            return (k, n)
    elif node.op_type in ("MatMul", "Gemm"):
        for inp in node.inputs:
            tensor = graph.initializers.get(inp)
            if tensor and len(tensor.shape) == 2:
                return (tensor.shape[0], tensor.shape[1])
    return None


def _infer_conv_groups(graph: ONNXGraph) -> Optional[int]:
    """Return the most common non-trivial conv group count."""
    groups = []
    for node in graph.nodes:
        if node.op_type == "Conv":
            g = node.attributes.get("group", 1)
            if g > 1:
                groups.append(g)
    if groups:
        counts = Counter(groups)
        return counts.most_common(1)[0][0]
    return None


def _infer_quantization(graph: ONNXGraph) -> Optional[str]:
    """Detect quantization format from ops and attributes."""
    for node in graph.nodes:
        if node.op_type == "MatMulNBits":
            bits = node.attributes.get("bits")
            if bits:
                return f"Q{bits}"
    has_dequant = any(n.op_type == "DequantizeLinear" for n in graph.nodes)
    if has_dequant:
        return "quantized"
    return None
