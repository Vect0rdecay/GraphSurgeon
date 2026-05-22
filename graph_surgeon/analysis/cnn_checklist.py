"""
CNN DAG Structural Checklist Checklist

Comprehensive structural checklist for vision models like ResNet, Inception, 
EfficientNet, ConvNeXt based on adversarial ML research best practices.

Audit Categories:
A. Input & Stem
B. Linear propagation chains
C. Pooling layers
D. Feature fusion
E. Residual connections
F. Normalization
G. Reduction blocks
H. Classifier head
I. Graph hygiene
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


class AuditStatus(Enum):
    """Status of an audit check."""
    PASS = "PASS"           # No vulnerability found
    FAIL = "FAIL"           # Vulnerability found
    WARN = "WARN"           # Potential concern
    INFO = "INFO"           # Informational finding
    NA = "N/A"              # Not applicable


class AuditCategory(Enum):
    """Audit checklist categories."""
    INPUT_STEM = "A. Input & Stem"
    LINEAR_CHAINS = "B. Linear Propagation Chains"
    POOLING = "C. Pooling Layers"
    FEATURE_FUSION = "D. Feature Fusion"
    RESIDUALS = "E. Residual Connections"
    NORMALIZATION = "F. Normalization"
    REDUCTION = "G. Reduction Blocks"
    CLASSIFIER = "H. Classifier Head"
    GRAPH_HYGIENE = "I. Graph Hygiene"


@dataclass
class AuditCheck:
    """Represents a single audit check."""
    id: str
    category: AuditCategory
    name: str
    status: AuditStatus
    description: str
    findings: List[str] = field(default_factory=list)
    affected_nodes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class HardeningRecommendation:
    """A specific hardening recommendation."""
    id: str
    title: str
    description: str
    priority: str  # "critical", "high", "medium", "low"
    affected_checks: List[str]
    implementation_notes: str


@dataclass
class CNNAuditReport:
    """Complete CNN structural checklist report."""
    model_name: str
    model_type: str  # "CNN", "Vision Transformer", "Hybrid", etc.
    
    # Audit results by category
    checks: List[AuditCheck] = field(default_factory=list)
    
    # Summary counts
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    
    # Hardening recommendations
    hardening: List[HardeningRecommendation] = field(default_factory=list)
    
    # Risk summary
    risk_areas: Dict[str, List[str]] = field(default_factory=dict)
    
    # Overall security grade
    security_grade: str = "?"


class CNNSecurityAuditor:
    """
    Performs comprehensive structural checklist on CNN architectures.
    
    Implements the full CNN DAG Structural Checklist Checklist covering:
    - Input/Stem vulnerabilities
    - Linear propagation issues
    - Pooling layer risks
    - Feature fusion analysis
    - Residual connection audit
    - Normalization assessment
    - Reduction block review
    - Classifier head security
    - Graph hygiene checks
    """
    
    # Operation categories
    CONV_OPS = {"Conv", "ConvTranspose", "Conv1D", "Conv2D", "Conv3D"}
    POOL_OPS = {"MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool",
                "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "MaxPool2d", "AvgPool2d"}
    MAXPOOL_OPS = {"MaxPool", "GlobalMaxPool", "MaxPool2d", "AdaptiveMaxPool2d"}
    AVGPOOL_OPS = {"AveragePool", "GlobalAveragePool", "AvgPool2d", "AdaptiveAvgPool2d"}
    NORM_OPS = {"BatchNormalization", "BatchNorm", "LayerNormalization", "LayerNorm",
                "GroupNormalization", "GroupNorm", "InstanceNormalization", "InstanceNorm"}
    BN_OPS = {"BatchNormalization", "BatchNorm"}
    GN_OPS = {"GroupNormalization", "GroupNorm"}
    LN_OPS = {"LayerNormalization", "LayerNorm"}
    LINEAR_OPS = {"Conv", "ConvTranspose", "MatMul", "Gemm", "Linear", "Dense"}
    FC_OPS = {"Gemm", "MatMul", "Linear", "Dense", "FullyConnected"}
    FUSION_OPS = {"Concat", "Add", "Sum"}
    ACTIVATION_OPS = {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Gelu", "Silu", "Swish",
                      "PRelu", "Elu", "Selu", "Softmax", "HardSwish", "Mish"}
    SHAPE_OPS = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Permute"}
    RESIZE_OPS = {"Resize", "Upsample", "ConvTranspose"}
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self.initializers: Set[str] = set()
        self.node_depths: Dict[str, int] = {}
        self.node_shapes: Dict[str, Tuple] = {}
        
    def build_graph(self, nodes: List[Dict], edges: List[Tuple[str, str]],
                   initializers: Optional[Set[str]] = None,
                   shapes: Optional[Dict[str, Tuple]] = None):
        """Build internal graph representation."""
        self.nodes = {n.get("node_id", n.get("name", f"node_{i}")): n 
                     for i, n in enumerate(nodes)}
        self.edges = edges
        self.initializers = initializers or set()
        self.node_shapes = shapes or {}
        
        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        
        for src, dst in edges:
            self.adjacency[src].append(dst)
            self.reverse_adjacency[dst].append(src)
        
        self._compute_depths()
    
    def _compute_depths(self):
        """Compute depth of each node from inputs."""
        self.node_depths = {}
        
        def get_depth(node_id: str) -> int:
            if node_id in self.node_depths:
                return self.node_depths[node_id]
            
            predecessors = self.reverse_adjacency.get(node_id, [])
            # Filter to only predecessors that are in our nodes dict
            valid_predecessors = [p for p in predecessors if p in self.nodes]
            
            if not valid_predecessors:
                self.node_depths[node_id] = 0
            else:
                self.node_depths[node_id] = 1 + max(
                    get_depth(p) for p in valid_predecessors
                )
            return self.node_depths[node_id]
        
        for node_id in self.nodes:
            get_depth(node_id)
    
    def _get_op_type(self, node_id: str) -> str:
        """Get operation type for a node."""
        return self.nodes.get(node_id, {}).get("op_type", "Unknown")
    
    def _get_attrs(self, node_id: str) -> Dict:
        """Get attributes for a node."""
        return self.nodes.get(node_id, {}).get("attributes", {})
    
    def _get_max_depth(self) -> int:
        """Get maximum depth in the graph."""
        return max(self.node_depths.values()) if self.node_depths else 0
    
    def _is_early_layer(self, node_id: str, threshold_fraction: float = 0.25) -> bool:
        """Check if node is in early layers."""
        max_depth = self._get_max_depth()
        if max_depth == 0:
            return True
        return self.node_depths.get(node_id, 0) < max_depth * threshold_fraction
    
    def _find_nodes_by_op(self, op_types: Set[str]) -> List[str]:
        """Find all nodes with given operation types."""
        return [nid for nid, n in self.nodes.items() 
                if n.get("op_type") in op_types]
    
    def _get_predecessors(self, node_id: str, depth: int = 1) -> List[str]:
        """Get predecessors up to given depth."""
        result = []
        current = [node_id]
        for _ in range(depth):
            next_level = []
            for n in current:
                preds = self.reverse_adjacency.get(n, [])
                result.extend(preds)
                next_level.extend(preds)
            current = next_level
        return result
    
    def _get_successors(self, node_id: str, depth: int = 1) -> List[str]:
        """Get successors up to given depth."""
        result = []
        current = [node_id]
        for _ in range(depth):
            next_level = []
            for n in current:
                succs = self.adjacency.get(n, [])
                result.extend(succs)
                next_level.extend(succs)
            current = next_level
        return result
    
    def audit(self, nodes: List[Dict], edges: List[Tuple[str, str]],
             model_name: str = "model",
             initializers: Optional[Set[str]] = None,
             shapes: Optional[Dict[str, Tuple]] = None) -> CNNAuditReport:
        """
        Perform complete structural checklist on a CNN model.
        
        Args:
            nodes: List of node dicts with 'node_id', 'op_type', 'attributes'
            edges: List of (src, dst) tuples
            model_name: Name of the model
            initializers: Set of initializer names (weights)
            shapes: Dict mapping node/tensor names to shapes
            
        Returns:
            CNNAuditReport with all findings
        """
        self.build_graph(nodes, edges, initializers, shapes)
        
        report = CNNAuditReport(
            model_name=model_name,
            model_type=self._detect_model_type()
        )
        
        # Run all audit checks
        report.checks.extend(self._audit_input_stem())
        report.checks.extend(self._audit_linear_chains())
        report.checks.extend(self._audit_pooling())
        report.checks.extend(self._audit_feature_fusion())
        report.checks.extend(self._audit_residuals())
        report.checks.extend(self._audit_normalization())
        report.checks.extend(self._audit_reduction_blocks())
        report.checks.extend(self._audit_classifier_head())
        report.checks.extend(self._audit_graph_hygiene())
        
        # Calculate summary
        report.total_checks = len(report.checks)
        report.passed = sum(1 for c in report.checks if c.status == AuditStatus.PASS)
        report.failed = sum(1 for c in report.checks if c.status == AuditStatus.FAIL)
        report.warnings = sum(1 for c in report.checks if c.status == AuditStatus.WARN)
        
        # Compile risks
        report.risk_areas = self._compile_risks(report.checks)
        
        # Generate hardening recommendations
        report.hardening = self._generate_hardening(report.checks)
        
        # Calculate security grade
        report.security_grade = self._calculate_grade(report)
        
        return report
    
    def _detect_model_type(self) -> str:
        """Detect the type of model based on architecture patterns."""
        has_conv = bool(self._find_nodes_by_op(self.CONV_OPS))
        has_attention = bool(self._find_nodes_by_op({"Attention", "MultiHeadAttention", "ScaledDotProductAttention"}))
        has_concat = bool(self._find_nodes_by_op({"Concat"}))
        has_add = bool(self._find_nodes_by_op({"Add"}))
        
        if has_attention and has_conv:
            return "Hybrid (CNN + Attention)"
        elif has_attention:
            return "Vision Transformer"
        elif has_concat and len(self._find_nodes_by_op({"Concat"})) > 5:
            return "CNN (Inception-style)"
        elif has_add and len(self._find_nodes_by_op({"Add"})) > 5:
            return "CNN (ResNet-style)"
        elif has_conv:
            return "CNN"
        else:
            return "Unknown"
    
    # =========================================================================
    # A. INPUT & STEM AUDIT
    # =========================================================================
    
    def _audit_input_stem(self) -> List[AuditCheck]:
        """Audit input and stem layer security."""
        checks = []
        
        # A1: Input normalization explicit in graph
        checks.append(self._check_input_normalization())
        
        # A2: Early stride-2 conv or pooling (aliasing risk)
        checks.append(self._check_early_stride())
        
        # A3: Anti-aliasing / blur before downsampling
        checks.append(self._check_anti_aliasing())
        
        # A4: Padding consistency
        checks.append(self._check_padding_consistency())
        
        # A5: Color channel ordering
        checks.append(self._check_channel_ordering())
        
        # A6: Preprocessing in graph
        checks.append(self._check_preprocessing())
        
        return checks
    
    def _check_input_normalization(self) -> AuditCheck:
        """Check if input normalization is explicit in the graph."""
        # Look for early normalization ops
        early_norms = [nid for nid in self._find_nodes_by_op(self.NORM_OPS)
                      if self._is_early_layer(nid, 0.1)]
        
        # Look for explicit mean subtraction / std division
        early_sub = [nid for nid in self._find_nodes_by_op({"Sub"})
                    if self._is_early_layer(nid, 0.1)]
        early_div = [nid for nid in self._find_nodes_by_op({"Div"})
                    if self._is_early_layer(nid, 0.1)]
        
        has_normalization = bool(early_norms or (early_sub and early_div))
        
        return AuditCheck(
            id="A1",
            category=AuditCategory.INPUT_STEM,
            name="Input normalization explicit in graph",
            status=AuditStatus.PASS if has_normalization else AuditStatus.WARN,
            description="Checks if input normalization (mean/std or BN) is part of the model graph.",
            findings=[
                f"Early normalization ops: {early_norms}" if early_norms else "No early normalization found",
                "Mean subtraction present" if early_sub else "No explicit mean subtraction",
            ],
            affected_nodes=early_norms + early_sub + early_div,
            risks=["Adversarial examples crafted outside expected input range may behave unexpectedly",
                   "Preprocessing mismatch between training and deployment"],
            recommendations=[
                "Bake input normalization into the model graph",
                "Use explicit mean/std subtraction/division ops",
                "Document expected input preprocessing"
            ] if not has_normalization else []
        )
    
    def _check_early_stride(self) -> AuditCheck:
        """Check for early stride-2 convolutions or pooling (aliasing risk)."""
        early_stride_convs = []
        early_pools = []
        
        for nid, node in self.nodes.items():
            if not self._is_early_layer(nid, 0.25):
                continue
            
            op_type = node.get("op_type", "")
            attrs = node.get("attributes", {})
            
            if op_type in self.CONV_OPS:
                strides = attrs.get("strides", [1, 1])
                if isinstance(strides, list) and any(s > 1 for s in strides):
                    early_stride_convs.append(nid)
                elif isinstance(strides, int) and strides > 1:
                    early_stride_convs.append(nid)
            
            if op_type in self.POOL_OPS:
                early_pools.append(nid)
        
        has_early_downsample = bool(early_stride_convs or early_pools)
        
        return AuditCheck(
            id="A2",
            category=AuditCategory.INPUT_STEM,
            name="Early stride-2 conv or pooling present",
            status=AuditStatus.WARN if has_early_downsample else AuditStatus.PASS,
            description="Flags early downsampling operations as aliasing risk.",
            findings=[
                f"Early stride convolutions: {len(early_stride_convs)} ({early_stride_convs[:5]}...)" if early_stride_convs else "No early stride convolutions",
                f"Early pooling: {len(early_pools)} ({early_pools[:5]}...)" if early_pools else "No early pooling",
            ],
            affected_nodes=early_stride_convs + early_pools,
            risks=["Frequency/Fourier attacks", "Aliasing artifacts exploitable by adversaries",
                   "Low-frequency perturbations survive downsampling"],
            recommendations=[
                "Add anti-aliasing (blur) before strided operations",
                "Consider blur pooling (BlurPool)",
                "Use resize + conv instead of strided conv"
            ] if has_early_downsample else []
        )
    
    def _check_anti_aliasing(self) -> AuditCheck:
        """Check for anti-aliasing / blur before downsampling."""
        # Look for blur/smoothing operations
        blur_ops = [nid for nid, n in self.nodes.items()
                   if "blur" in nid.lower() or "smooth" in nid.lower() or "antialias" in nid.lower()]
        
        # Check if blur precedes pooling or strided conv
        blur_before_downsample = []
        for blur in blur_ops:
            succs = self._get_successors(blur, 2)
            for s in succs:
                if self._get_op_type(s) in self.POOL_OPS:
                    blur_before_downsample.append((blur, s))
                elif self._get_op_type(s) in self.CONV_OPS:
                    strides = self._get_attrs(s).get("strides", [1, 1])
                    if any(st > 1 for st in (strides if isinstance(strides, list) else [strides])):
                        blur_before_downsample.append((blur, s))
        
        has_anti_aliasing = bool(blur_before_downsample)
        
        # Count downsampling ops without anti-aliasing
        all_pool = self._find_nodes_by_op(self.POOL_OPS)
        stride_conv = [nid for nid, n in self.nodes.items()
                      if n.get("op_type") in self.CONV_OPS
                      and any(s > 1 for s in (n.get("attributes", {}).get("strides", [1, 1]) 
                              if isinstance(n.get("attributes", {}).get("strides", [1, 1]), list) 
                              else [n.get("attributes", {}).get("strides", 1)]))]
        
        unprotected = len(all_pool) + len(stride_conv) - len(blur_before_downsample)
        
        return AuditCheck(
            id="A3",
            category=AuditCategory.INPUT_STEM,
            name="Anti-aliasing / blur before downsampling",
            status=AuditStatus.PASS if has_anti_aliasing else AuditStatus.FAIL,
            description="Checks if anti-aliasing is applied before downsampling operations.",
            findings=[
                f"Blur operations found: {len(blur_ops)}",
                f"Blur before downsample pairs: {len(blur_before_downsample)}",
                f"Unprotected downsampling operations: ~{unprotected}"
            ],
            affected_nodes=blur_ops,
            risks=["Frequency attacks can exploit aliasing",
                   "Perturbations at specific frequencies bypass downsampling"],
            recommendations=[
                "Add BlurPool or Gaussian blur before pooling",
                "Use anti-aliased strided convolutions",
                "Consider learned anti-aliasing filters"
            ] if not has_anti_aliasing else []
        )
    
    def _check_padding_consistency(self) -> AuditCheck:
        """Check padding type consistency (pads=0 near edges = border sensitivity)."""
        valid_convs = []  # pads=0
        same_convs = []   # pads to maintain size
        mixed_padding = False
        
        for nid, node in self.nodes.items():
            if node.get("op_type") not in self.CONV_OPS:
                continue
            
            attrs = node.get("attributes", {})
            pads = attrs.get("pads", None)
            auto_pad = attrs.get("auto_pad", "NOTSET")
            
            is_valid = False
            if auto_pad == "VALID":
                is_valid = True
            elif pads is not None:
                if isinstance(pads, list) and all(p == 0 for p in pads):
                    is_valid = True
                elif pads == 0:
                    is_valid = True
            
            if is_valid:
                valid_convs.append(nid)
            else:
                same_convs.append(nid)
        
        if valid_convs and same_convs:
            mixed_padding = True
        
        early_valid = [v for v in valid_convs if self._is_early_layer(v)]
        
        status = AuditStatus.PASS
        if early_valid:
            status = AuditStatus.WARN
        if len(valid_convs) > len(same_convs):
            status = AuditStatus.FAIL
        
        return AuditCheck(
            id="A4",
            category=AuditCategory.INPUT_STEM,
            name="Padding type consistency (border sensitivity)",
            status=status,
            description="Checks if padding is consistent and flags valid (pads=0) convolutions that create border sensitivity.",
            findings=[
                f"Valid convolutions (pads=0): {len(valid_convs)}",
                f"Same/other padding convolutions: {len(same_convs)}",
                f"Early valid convolutions: {len(early_valid)}",
                f"Mixed padding types: {mixed_padding}"
            ],
            affected_nodes=valid_convs,
            risks=["Border/edge sensitivity - less context at boundaries",
                   "Corner attacks - minimal receptive field",
                   "Frame injection attacks"],
            recommendations=[
                "Use same/reflect padding to provide edge context",
                "Consider circular padding for certain applications",
                "Implement boundary-aware adversarial training"
            ] if valid_convs else []
        )
    
    def _check_channel_ordering(self) -> AuditCheck:
        """Check if color channel ordering is explicit."""
        # Look for transpose ops that might indicate format conversion
        transpose_ops = self._find_nodes_by_op({"Transpose", "Permute"})
        early_transpose = [t for t in transpose_ops if self._is_early_layer(t, 0.1)]
        
        return AuditCheck(
            id="A5",
            category=AuditCategory.INPUT_STEM,
            name="Color channel ordering explicit (NCHW vs NHWC)",
            status=AuditStatus.INFO,
            description="Informational check about channel ordering.",
            findings=[
                f"Early transpose operations: {len(early_transpose)}",
                "ONNX typically uses NCHW format",
                "Check deployment framework compatibility"
            ],
            affected_nodes=early_transpose,
            risks=["Channel mismatch between preprocessing and model",
                   "Incorrect adversarial perturbation targeting"],
            recommendations=[
                "Document expected input format",
                "Bake format conversion into model if needed"
            ]
        )
    
    def _check_preprocessing(self) -> AuditCheck:
        """Check for preprocessing baked into graph."""
        resize_ops = self._find_nodes_by_op(self.RESIZE_OPS)
        early_resize = [r for r in resize_ops if self._is_early_layer(r, 0.15)]
        
        # Look for crop patterns
        slice_ops = [nid for nid in self._find_nodes_by_op({"Slice", "Crop"})
                    if self._is_early_layer(nid, 0.15)]
        
        has_preprocessing = bool(early_resize or slice_ops)
        
        return AuditCheck(
            id="A6",
            category=AuditCategory.INPUT_STEM,
            name="Preprocessing baked into graph (resize, crop)",
            status=AuditStatus.INFO if has_preprocessing else AuditStatus.INFO,
            description="Checks if preprocessing operations are part of the model graph.",
            findings=[
                f"Resize operations in stem: {len(early_resize)}",
                f"Crop/slice operations in stem: {len(slice_ops)}",
            ],
            affected_nodes=early_resize + slice_ops,
            risks=["Preprocessing-dependent adversarial examples",
                   "Resize can introduce aliasing artifacts"],
            recommendations=[
                "Ensure preprocessing matches training exactly",
                "Consider anti-aliased resize operations"
            ]
        )
    
    # =========================================================================
    # B. LINEAR PROPAGATION CHAINS AUDIT
    # =========================================================================
    
    def _audit_linear_chains(self) -> List[AuditCheck]:
        """Audit linear propagation chains."""
        checks = []
        
        # B1: Long Conv chains without normalization
        checks.append(self._check_conv_chains_no_norm())
        
        # B2: Large kernels after fusion
        checks.append(self._check_large_kernels_after_fusion())
        
        # B3: High channel expansion without bottleneck
        checks.append(self._check_channel_expansion())
        
        return checks
    
    def _check_conv_chains_no_norm(self) -> AuditCheck:
        """Check for long Conv->Conv->Conv chains without normalization."""
        conv_nodes = self._find_nodes_by_op(self.CONV_OPS)
        chains_without_norm = []
        visited = set()
        
        for start in conv_nodes:
            if start in visited:
                continue
            
            chain = [start]
            current = start
            has_norm = False
            
            while True:
                succs = self.adjacency.get(current, [])
                
                # Check for normalization
                for s in succs:
                    if self._get_op_type(s) in self.NORM_OPS:
                        has_norm = True
                        break
                
                if has_norm:
                    break
                
                # Find next conv (possibly through activation)
                next_conv = None
                for s in succs:
                    if self._get_op_type(s) in self.CONV_OPS:
                        next_conv = s
                        break
                    # Check one hop through activation
                    for ss in self.adjacency.get(s, []):
                        if self._get_op_type(ss) in self.CONV_OPS:
                            next_conv = ss
                            break
                    if next_conv:
                        break
                
                if not next_conv or next_conv in visited:
                    break
                
                chain.append(next_conv)
                current = next_conv
            
            visited.update(chain)
            
            if len(chain) >= 2 and not has_norm:
                chains_without_norm.append(chain)
        
        longest_chain = max((len(c) for c in chains_without_norm), default=0)
        all_nodes = [n for c in chains_without_norm for n in c]
        
        status = AuditStatus.PASS
        if chains_without_norm:
            status = AuditStatus.WARN if longest_chain <= 2 else AuditStatus.FAIL
        
        return AuditCheck(
            id="B1",
            category=AuditCategory.LINEAR_CHAINS,
            name="Long Conv->Conv->Conv chains without normalization",
            status=status,
            description="Detects linear chains that provide stable gradients for attacks.",
            findings=[
                f"Chains without normalization: {len(chains_without_norm)}",
                f"Longest chain: {longest_chain} ops",
                f"Total affected nodes: {len(all_nodes)}"
            ],
            affected_nodes=all_nodes[:20],
            risks=["PGD attacks", "C&W attacks", "AutoAttack", "High transferability",
                   "Stable gradient flow enables precise perturbation computation"],
            recommendations=[
                "Insert BatchNorm/LayerNorm between convolutions",
                "Apply spectral normalization",
                "Consider gradient regularization during training"
            ] if chains_without_norm else []
        )
    
    def _check_large_kernels_after_fusion(self) -> AuditCheck:
        """Check for large kernels immediately after fusion points."""
        fusion_nodes = self._find_nodes_by_op(self.FUSION_OPS)
        large_kernels_after_fusion = []
        
        for fusion in fusion_nodes:
            succs = self._get_successors(fusion, 2)
            for s in succs:
                if self._get_op_type(s) in self.CONV_OPS:
                    kernel = self._get_attrs(s).get("kernel_shape", [1, 1])
                    if isinstance(kernel, list):
                        kernel_size = kernel[0] * kernel[1] if len(kernel) >= 2 else kernel[0]
                    else:
                        kernel_size = kernel * kernel
                    
                    if kernel_size > 9:  # > 3x3
                        large_kernels_after_fusion.append((fusion, s, kernel_size))
        
        status = AuditStatus.PASS if not large_kernels_after_fusion else AuditStatus.WARN
        
        return AuditCheck(
            id="B2",
            category=AuditCategory.LINEAR_CHAINS,
            name="Large kernels after fusion",
            status=status,
            description="Checks for large kernels immediately after feature fusion points.",
            findings=[
                f"Large kernels after fusion: {len(large_kernels_after_fusion)}",
                *[f"  {f} -> {c} (kernel={k})" for f, c, k in large_kernels_after_fusion[:5]]
            ],
            affected_nodes=[c for _, c, _ in large_kernels_after_fusion],
            risks=["Fused perturbations immediately processed by large receptive field",
                   "Amplification of coordinated multi-scale attacks"],
            recommendations=[
                "Consider 1x1 conv before large kernels to reduce channels",
                "Add normalization between fusion and large conv"
            ] if large_kernels_after_fusion else []
        )
    
    def _check_channel_expansion(self) -> AuditCheck:
        """Check for high channel expansion without bottleneck."""
        # This would ideally use shape information
        # For now, check for conv nodes that might expand channels
        conv_nodes = self._find_nodes_by_op(self.CONV_OPS)
        
        # Look for 1x1 convolutions (bottleneck indicators)
        bottleneck_convs = []
        potential_expansion = []
        
        for nid in conv_nodes:
            attrs = self._get_attrs(nid)
            kernel = attrs.get("kernel_shape", [3, 3])
            if isinstance(kernel, list) and kernel == [1, 1]:
                bottleneck_convs.append(nid)
            elif isinstance(kernel, int) and kernel == 1:
                bottleneck_convs.append(nid)
            else:
                potential_expansion.append(nid)
        
        has_bottlenecks = len(bottleneck_convs) >= len(potential_expansion) * 0.2
        
        return AuditCheck(
            id="B3",
            category=AuditCategory.LINEAR_CHAINS,
            name="High channel expansion without bottleneck",
            status=AuditStatus.PASS if has_bottlenecks else AuditStatus.WARN,
            description="Checks for channel expansion patterns and bottleneck usage.",
            findings=[
                f"Bottleneck (1x1) convolutions: {len(bottleneck_convs)}",
                f"Other convolutions: {len(potential_expansion)}",
                f"Bottleneck ratio: {len(bottleneck_convs) / max(len(conv_nodes), 1):.2%}"
            ],
            affected_nodes=bottleneck_convs[:10],
            risks=["High channel expansion without reduction increases attack surface",
                   "More parameters = more capacity for adversarial exploitation"],
            recommendations=[
                "Use bottleneck (1x1) convolutions before larger kernels",
                "Implement channel attention to focus on relevant features"
            ] if not has_bottlenecks else []
        )
    
    # =========================================================================
    # C. POOLING LAYERS AUDIT
    # =========================================================================
    
    def _audit_pooling(self) -> List[AuditCheck]:
        """Audit pooling layers."""
        checks = []
        
        # C1: MaxPool presence and location
        checks.append(self._check_maxpool_presence())
        
        # C2: Pooling immediately after fusion
        checks.append(self._check_pooling_after_fusion())
        
        # C3: Pooling before normalization
        checks.append(self._check_pooling_before_norm())
        
        return checks
    
    def _check_maxpool_presence(self) -> AuditCheck:
        """Check for MaxPool presence and locations."""
        maxpool_nodes = self._find_nodes_by_op(self.MAXPOOL_OPS)
        avgpool_nodes = self._find_nodes_by_op(self.AVGPOOL_OPS)
        
        early_maxpool = [m for m in maxpool_nodes if self._is_early_layer(m)]
        
        status = AuditStatus.PASS
        if maxpool_nodes:
            status = AuditStatus.WARN if len(maxpool_nodes) <= 2 else AuditStatus.FAIL
        
        return AuditCheck(
            id="C1",
            category=AuditCategory.POOLING,
            name="MaxPool presence and locations",
            status=status,
            description="Identifies MaxPool layers which amplify sparse perturbations.",
            findings=[
                f"MaxPool layers: {len(maxpool_nodes)}",
                f"AvgPool layers: {len(avgpool_nodes)}",
                f"Early MaxPool: {len(early_maxpool)}",
                f"MaxPool locations: {maxpool_nodes[:5]}"
            ],
            affected_nodes=maxpool_nodes,
            risks=["Sparse perturbations (one-pixel attacks)", "Patch attacks",
                   "Spike amplification - only need to perturb one value per region"],
            recommendations=[
                "Replace MaxPool with AvgPool for robustness",
                "Use BlurPool for anti-aliased downsampling",
                "Consider soft-max pooling alternatives"
            ] if maxpool_nodes else []
        )
    
    def _check_pooling_after_fusion(self) -> AuditCheck:
        """Check for pooling immediately after fusion points."""
        fusion_nodes = self._find_nodes_by_op(self.FUSION_OPS)
        pool_after_fusion = []
        
        for fusion in fusion_nodes:
            succs = self._get_successors(fusion, 2)
            for s in succs:
                if self._get_op_type(s) in self.POOL_OPS:
                    pool_after_fusion.append((fusion, s))
                    break
        
        maxpool_after_fusion = [(f, p) for f, p in pool_after_fusion 
                                if self._get_op_type(p) in self.MAXPOOL_OPS]
        
        status = AuditStatus.PASS
        if maxpool_after_fusion:
            status = AuditStatus.FAIL
        elif pool_after_fusion:
            status = AuditStatus.WARN
        
        return AuditCheck(
            id="C2",
            category=AuditCategory.POOLING,
            name="Pooling immediately after fusion",
            status=status,
            description="Checks if pooling follows fusion operations directly.",
            findings=[
                f"Pooling after fusion: {len(pool_after_fusion)}",
                f"MaxPool after fusion: {len(maxpool_after_fusion)} (critical)",
            ],
            affected_nodes=[p for _, p in pool_after_fusion],
            risks=["Fused adversarial signals immediately pooled",
                   "MaxPool selects strongest adversarial spike from merged features"],
            recommendations=[
                "Add normalization or conv between fusion and pooling",
                "Replace MaxPool with AvgPool after fusion"
            ] if pool_after_fusion else []
        )
    
    def _check_pooling_before_norm(self) -> AuditCheck:
        """Check for pooling before normalization."""
        pool_nodes = self._find_nodes_by_op(self.POOL_OPS)
        pool_before_norm = []
        
        for pool in pool_nodes:
            succs = self._get_successors(pool, 2)
            for s in succs:
                if self._get_op_type(s) in self.NORM_OPS:
                    pool_before_norm.append((pool, s))
                    break
        
        # This pattern is sometimes intentional (GAP before final norm)
        # but can be concerning for MaxPool
        maxpool_before_norm = [(p, n) for p, n in pool_before_norm
                               if self._get_op_type(p) in self.MAXPOOL_OPS]
        
        return AuditCheck(
            id="C3",
            category=AuditCategory.POOLING,
            name="Pooling before normalization",
            status=AuditStatus.WARN if maxpool_before_norm else AuditStatus.INFO,
            description="Checks pooling -> normalization patterns.",
            findings=[
                f"Pooling before normalization: {len(pool_before_norm)}",
                f"MaxPool before normalization: {len(maxpool_before_norm)}"
            ],
            affected_nodes=[p for p, _ in pool_before_norm],
            risks=["MaxPool spikes feed into normalization",
                   "Adversarial outliers affect batch statistics"],
            recommendations=[
                "Consider norm -> pool order for more stable statistics"
            ] if maxpool_before_norm else []
        )
    
    # =========================================================================
    # D. FEATURE FUSION AUDIT
    # =========================================================================
    
    def _audit_feature_fusion(self) -> List[AuditCheck]:
        """Audit feature fusion points."""
        checks = []
        
        # D1: Concat nodes count
        checks.append(self._check_concat_nodes())
        
        # D2: Fan-in > 3
        checks.append(self._check_high_fan_in())
        
        # D3: Fusion followed by Conv
        checks.append(self._check_fusion_conv_pattern())
        
        return checks
    
    def _check_concat_nodes(self) -> AuditCheck:
        """Count and analyze Concat(axis=1) nodes."""
        concat_nodes = self._find_nodes_by_op({"Concat"})
        
        axis_1_concat = []
        other_concat = []
        
        for nid in concat_nodes:
            attrs = self._get_attrs(nid)
            axis = attrs.get("axis", 1)
            if axis == 1:
                axis_1_concat.append(nid)
            else:
                other_concat.append(nid)
        
        status = AuditStatus.PASS
        if len(axis_1_concat) > 5:
            status = AuditStatus.WARN
        if len(axis_1_concat) > 10:
            status = AuditStatus.FAIL
        
        return AuditCheck(
            id="D1",
            category=AuditCategory.FEATURE_FUSION,
            name="Concat(axis=1) nodes count",
            status=status,
            description="Counts channel concatenation operations that create perturbation fusion points.",
            findings=[
                f"Total Concat nodes: {len(concat_nodes)}",
                f"Concat(axis=1) - channel: {len(axis_1_concat)}",
                f"Other axis Concat: {len(other_concat)}"
            ],
            affected_nodes=axis_1_concat,
            risks=["Multi-scale coordinated attacks", "Universal perturbations",
                   "Perturbation aggregation from multiple branches"],
            recommendations=[
                "Consider attention-weighted fusion",
                "Add channel normalization after concat",
                "Implement feature gating"
            ] if axis_1_concat else []
        )
    
    def _check_high_fan_in(self) -> AuditCheck:
        """Check for nodes with fan-in > 3 branches."""
        high_fan_in = []
        
        for nid in self.nodes:
            preds = self.reverse_adjacency.get(nid, [])
            if len(preds) > 3:
                high_fan_in.append((nid, len(preds)))
        
        # Sort by fan-in
        high_fan_in.sort(key=lambda x: x[1], reverse=True)
        
        status = AuditStatus.PASS
        if high_fan_in:
            max_fan_in = high_fan_in[0][1]
            status = AuditStatus.WARN if max_fan_in <= 5 else AuditStatus.FAIL
        
        return AuditCheck(
            id="D2",
            category=AuditCategory.FEATURE_FUSION,
            name="Fan-in > 3 branches",
            status=status,
            description="Identifies nodes receiving many input branches.",
            findings=[
                f"High fan-in nodes (>3): {len(high_fan_in)}",
                *[f"  {nid}: {fi} inputs" for nid, fi in high_fan_in[:5]]
            ],
            affected_nodes=[nid for nid, _ in high_fan_in],
            risks=["Coordinated multi-branch attacks",
                   "Perturbations from many sources combine additively"],
            recommendations=[
                "Add attention to weight branch contributions",
                "Implement branch-wise normalization",
                "Consider reducing connectivity"
            ] if high_fan_in else []
        )
    
    def _check_fusion_conv_pattern(self) -> AuditCheck:
        """Check for fusion followed immediately by Conv."""
        fusion_nodes = self._find_nodes_by_op(self.FUSION_OPS)
        fusion_then_conv = []
        
        for fusion in fusion_nodes:
            succs = self.adjacency.get(fusion, [])
            for s in succs:
                if self._get_op_type(s) in self.CONV_OPS:
                    fusion_then_conv.append((fusion, s))
                    break
        
        return AuditCheck(
            id="D3",
            category=AuditCategory.FEATURE_FUSION,
            name="Fusion followed immediately by Conv",
            status=AuditStatus.INFO,
            description="Identifies Concat/Add -> Conv patterns.",
            findings=[
                f"Fusion -> Conv patterns: {len(fusion_then_conv)}",
                "This is common and not inherently bad, but note:",
                "Fused perturbations immediately mixed by convolution"
            ],
            affected_nodes=[c for _, c in fusion_then_conv],
            risks=["Adversarial features from branches immediately processed together"],
            recommendations=[
                "Consider normalization between fusion and conv"
            ]
        )
    
    # =========================================================================
    # E. RESIDUAL CONNECTIONS AUDIT
    # =========================================================================
    
    def _audit_residuals(self) -> List[AuditCheck]:
        """Audit residual connections."""
        checks = []
        
        # E1: Add nodes with long skip distance
        checks.append(self._check_skip_distances())
        
        # E2: Multiple residuals stacked
        checks.append(self._check_stacked_residuals())
        
        return checks
    
    def _check_skip_distances(self) -> AuditCheck:
        """Check for Add nodes with long skip connections."""
        add_nodes = self._find_nodes_by_op({"Add"})
        long_skips = []
        
        for add in add_nodes:
            preds = self.reverse_adjacency.get(add, [])
            if len(preds) < 2:
                continue
            
            # Calculate depth difference between inputs
            depths = [self.node_depths.get(p, 0) for p in preds]
            if len(depths) >= 2:
                skip_distance = max(depths) - min(depths)
                if skip_distance > 3:
                    long_skips.append((add, skip_distance))
        
        long_skips.sort(key=lambda x: x[1], reverse=True)
        
        status = AuditStatus.PASS
        if long_skips:
            max_skip = long_skips[0][1]
            status = AuditStatus.WARN if max_skip <= 10 else AuditStatus.INFO
        
        return AuditCheck(
            id="E1",
            category=AuditCategory.RESIDUALS,
            name="Add nodes with long skip distance",
            status=status,
            description="Identifies residual connections with long skip distances.",
            findings=[
                f"Long skip connections (>3 layers): {len(long_skips)}",
                *[f"  {nid}: skip distance {d}" for nid, d in long_skips[:5]]
            ],
            affected_nodes=[nid for nid, _ in long_skips],
            risks=["Gradient highways - direct path for backprop",
                   "Early layers receive strong gradient signal from loss"],
            recommendations=[
                "This is often intentional in ResNets",
                "Consider stochastic depth during training"
            ]
        )
    
    def _check_stacked_residuals(self) -> AuditCheck:
        """Check for multiple residuals stacked."""
        add_nodes = self._find_nodes_by_op({"Add"})
        
        # Find chains of Add operations
        residual_chains = []
        visited = set()
        
        for start in add_nodes:
            if start in visited:
                continue
            
            chain = [start]
            current = start
            
            # Look for Add in successors
            while True:
                succs = self._get_successors(current, 5)
                next_add = None
                for s in succs:
                    if s in add_nodes and s not in chain:
                        next_add = s
                        break
                
                if not next_add:
                    break
                
                chain.append(next_add)
                current = next_add
            
            visited.update(chain)
            if len(chain) > 1:
                residual_chains.append(chain)
        
        longest_chain = max((len(c) for c in residual_chains), default=0)
        
        return AuditCheck(
            id="E2",
            category=AuditCategory.RESIDUALS,
            name="Multiple residuals stacked",
            status=AuditStatus.INFO,
            description="Counts stacked residual connections.",
            findings=[
                f"Residual chains found: {len(residual_chains)}",
                f"Longest residual chain: {longest_chain} blocks",
                f"Total Add nodes: {len(add_nodes)}"
            ],
            affected_nodes=add_nodes[:15],
            risks=["Strong white-box attacks due to gradient highways",
                   "Momentum-based attacks (MI-FGSM) very effective"],
            recommendations=[
                "Consider pre-activation ResNet design",
                "Apply stochastic depth during training"
            ]
        )
    
    # =========================================================================
    # F. NORMALIZATION AUDIT
    # =========================================================================
    
    def _audit_normalization(self) -> List[AuditCheck]:
        """Audit normalization layers."""
        checks = []
        
        # F1: BatchNorm presence and fusion
        checks.append(self._check_batchnorm())
        
        # F2: BN position (after Conv or before activation)
        checks.append(self._check_bn_position())
        
        # F3: GroupNorm / LayerNorm presence
        checks.append(self._check_other_norms())
        
        return checks
    
    def _check_batchnorm(self) -> AuditCheck:
        """Check BatchNorm presence and potential fusion."""
        bn_nodes = self._find_nodes_by_op(self.BN_OPS)
        
        # Check for fused patterns (Conv-BN-ReLU often fused)
        potentially_fused = 0
        for bn in bn_nodes:
            preds = self.reverse_adjacency.get(bn, [])
            succs = self.adjacency.get(bn, [])
            
            has_conv_before = any(self._get_op_type(p) in self.CONV_OPS for p in preds)
            has_act_after = any(self._get_op_type(s) in self.ACTIVATION_OPS for s in succs)
            
            if has_conv_before and has_act_after:
                potentially_fused += 1
        
        return AuditCheck(
            id="F1",
            category=AuditCategory.NORMALIZATION,
            name="BatchNorm presence and fusion",
            status=AuditStatus.INFO if bn_nodes else AuditStatus.WARN,
            description="Analyzes BatchNorm usage in the model.",
            findings=[
                f"BatchNorm layers: {len(bn_nodes)}",
                f"Potentially fused (Conv-BN-Act): {potentially_fused}",
            ],
            affected_nodes=bn_nodes[:10],
            risks=["Distribution shift attacks targeting BN statistics",
                   "Running mean/var mismatch at inference time"],
            recommendations=[
                "Consider GroupNorm or LayerNorm for robustness",
                "Implement distribution monitoring"
            ] if bn_nodes else ["Consider adding normalization layers"]
        )
    
    def _check_bn_position(self) -> AuditCheck:
        """Check BatchNorm position relative to Conv and activation."""
        bn_nodes = self._find_nodes_by_op(self.BN_OPS)
        
        bn_after_conv = 0
        bn_before_act = 0
        pre_activation_pattern = 0  # BN -> ReLU -> Conv (pre-act ResNet)
        
        for bn in bn_nodes:
            preds = self.reverse_adjacency.get(bn, [])
            succs = self.adjacency.get(bn, [])
            
            if any(self._get_op_type(p) in self.CONV_OPS for p in preds):
                bn_after_conv += 1
            
            if any(self._get_op_type(s) in self.ACTIVATION_OPS for s in succs):
                bn_before_act += 1
            
            # Check for pre-activation pattern
            for s in succs:
                if self._get_op_type(s) in self.ACTIVATION_OPS:
                    s_succs = self.adjacency.get(s, [])
                    if any(self._get_op_type(ss) in self.CONV_OPS for ss in s_succs):
                        pre_activation_pattern += 1
        
        return AuditCheck(
            id="F2",
            category=AuditCategory.NORMALIZATION,
            name="BN position (after Conv / before activation)",
            status=AuditStatus.INFO,
            description="Analyzes BatchNorm positioning patterns.",
            findings=[
                f"BN after Conv: {bn_after_conv}",
                f"BN before activation: {bn_before_act}",
                f"Pre-activation patterns (BN->Act->Conv): {pre_activation_pattern}"
            ],
            affected_nodes=[],
            risks=["Post-activation BN can have distribution issues"],
            recommendations=[
                "Pre-activation design can improve gradient flow"
            ]
        )
    
    def _check_other_norms(self) -> AuditCheck:
        """Check for GroupNorm / LayerNorm presence."""
        gn_nodes = self._find_nodes_by_op(self.GN_OPS)
        ln_nodes = self._find_nodes_by_op(self.LN_OPS)
        bn_nodes = self._find_nodes_by_op(self.BN_OPS)
        
        has_robust_norms = bool(gn_nodes or ln_nodes)
        
        return AuditCheck(
            id="F3",
            category=AuditCategory.NORMALIZATION,
            name="GroupNorm / LayerNorm presence",
            status=AuditStatus.PASS if has_robust_norms else AuditStatus.INFO,
            description="Checks for more robust normalization alternatives.",
            findings=[
                f"GroupNorm layers: {len(gn_nodes)}",
                f"LayerNorm layers: {len(ln_nodes)}",
                f"BatchNorm layers: {len(bn_nodes)}",
                "GroupNorm/LayerNorm are more robust to distribution shift" if has_robust_norms else ""
            ],
            affected_nodes=gn_nodes + ln_nodes,
            risks=["BatchNorm-only models vulnerable to distribution shift"],
            recommendations=[
                "Consider replacing BatchNorm with GroupNorm",
                "LayerNorm for attention-based components"
            ] if not has_robust_norms and bn_nodes else []
        )
    
    # =========================================================================
    # G. REDUCTION BLOCKS AUDIT
    # =========================================================================
    
    def _audit_reduction_blocks(self) -> List[AuditCheck]:
        """Audit reduction/downsampling blocks."""
        checks = []
        
        # G1: Sudden spatial drops
        checks.append(self._check_spatial_drops())
        
        # G2: Stride-2 + Concat patterns
        checks.append(self._check_stride_concat_pattern())
        
        return checks
    
    def _check_spatial_drops(self) -> AuditCheck:
        """Check for sudden spatial dimension drops."""
        # This would ideally use shape information
        # Heuristic: look for stride-2 ops clustered together
        
        stride_2_ops = []
        for nid, node in self.nodes.items():
            if node.get("op_type") in self.CONV_OPS | self.POOL_OPS:
                attrs = node.get("attributes", {})
                strides = attrs.get("strides", [1, 1])
                if isinstance(strides, list) and any(s >= 2 for s in strides):
                    stride_2_ops.append(nid)
                elif isinstance(strides, int) and strides >= 2:
                    stride_2_ops.append(nid)
        
        # Check for clustering
        clustered = []
        for i, op1 in enumerate(stride_2_ops):
            for op2 in stride_2_ops[i+1:]:
                depth_diff = abs(self.node_depths.get(op1, 0) - self.node_depths.get(op2, 0))
                if depth_diff <= 3:
                    clustered.append((op1, op2))
        
        return AuditCheck(
            id="G1",
            category=AuditCategory.REDUCTION,
            name="Sudden spatial drops",
            status=AuditStatus.WARN if len(clustered) > 2 else AuditStatus.INFO,
            description="Identifies aggressive downsampling patterns.",
            findings=[
                f"Stride-2 operations: {len(stride_2_ops)}",
                f"Clustered reductions (within 3 layers): {len(clustered)}",
            ],
            affected_nodes=stride_2_ops[:10],
            risks=["Aliasing survivability attacks",
                   "Rapid information loss can amplify adversarial signals"],
            recommendations=[
                "Use gradual downsampling",
                "Add anti-aliasing between reductions"
            ] if clustered else []
        )
    
    def _check_stride_concat_pattern(self) -> AuditCheck:
        """Check for stride-2 + Concat patterns (Inception reduction)."""
        stride_2_ops = []
        for nid, node in self.nodes.items():
            if node.get("op_type") in self.CONV_OPS | self.POOL_OPS:
                attrs = node.get("attributes", {})
                strides = attrs.get("strides", [1, 1])
                if isinstance(strides, list) and any(s >= 2 for s in strides):
                    stride_2_ops.append(nid)
                elif isinstance(strides, int) and strides >= 2:
                    stride_2_ops.append(nid)
        
        concat_nodes = self._find_nodes_by_op({"Concat"})
        
        stride_to_concat = []
        for stride_op in stride_2_ops:
            succs = self._get_successors(stride_op, 3)
            for s in succs:
                if s in concat_nodes:
                    stride_to_concat.append((stride_op, s))
                    break
        
        return AuditCheck(
            id="G2",
            category=AuditCategory.REDUCTION,
            name="Stride-2 + Concat patterns",
            status=AuditStatus.INFO,
            description="Identifies Inception-style reduction blocks.",
            findings=[
                f"Stride-2 -> Concat patterns: {len(stride_to_concat)}",
                "Common in Inception architectures"
            ],
            affected_nodes=[s for _, s in stride_to_concat],
            risks=["Aliased features from multiple branches combined",
                   "Coordinated aliasing attacks possible"],
            recommendations=[
                "Ensure consistent anti-aliasing across branches"
            ] if stride_to_concat else []
        )
    
    # =========================================================================
    # H. CLASSIFIER HEAD AUDIT
    # =========================================================================
    
    def _audit_classifier_head(self) -> List[AuditCheck]:
        """Audit classifier head."""
        checks = []
        
        # H1: GlobalAveragePool usage
        checks.append(self._check_gap_usage())
        
        # H2: Flatten before FC
        checks.append(self._check_flatten_fc())
        
        # H3: Single large FC layer
        checks.append(self._check_single_fc())
        
        return checks
    
    def _check_gap_usage(self) -> AuditCheck:
        """Check for GlobalAveragePool usage."""
        gap_nodes = self._find_nodes_by_op({"GlobalAveragePool", "AdaptiveAvgPool2d"})
        gmp_nodes = self._find_nodes_by_op({"GlobalMaxPool", "AdaptiveMaxPool2d"})
        
        status = AuditStatus.PASS if gap_nodes else AuditStatus.WARN
        if gmp_nodes and not gap_nodes:
            status = AuditStatus.FAIL
        
        return AuditCheck(
            id="H1",
            category=AuditCategory.CLASSIFIER,
            name="GlobalAveragePool usage",
            status=status,
            description="Checks for appropriate global pooling in classifier head.",
            findings=[
                f"GlobalAveragePool: {len(gap_nodes)}",
                f"GlobalMaxPool: {len(gmp_nodes)}",
            ],
            affected_nodes=gap_nodes + gmp_nodes,
            risks=["GlobalMaxPool amplifies sparse perturbations",
                   "Feature-space attacks target global representation"],
            recommendations=[
                "Prefer GlobalAveragePool over GlobalMaxPool",
                "Consider feature denoising before global pooling"
            ] if gmp_nodes else []
        )
    
    def _check_flatten_fc(self) -> AuditCheck:
        """Check for Flatten before FC pattern."""
        flatten_nodes = self._find_nodes_by_op({"Flatten"})
        fc_nodes = self._find_nodes_by_op(self.FC_OPS)
        
        flatten_to_fc = []
        for flatten in flatten_nodes:
            succs = self._get_successors(flatten, 2)
            for s in succs:
                if s in fc_nodes:
                    flatten_to_fc.append((flatten, s))
                    break
        
        return AuditCheck(
            id="H2",
            category=AuditCategory.CLASSIFIER,
            name="Flatten before FC",
            status=AuditStatus.INFO,
            description="Identifies flatten -> FC patterns in classifier.",
            findings=[
                f"Flatten operations: {len(flatten_nodes)}",
                f"Flatten -> FC patterns: {len(flatten_to_fc)}"
            ],
            affected_nodes=[f for f, _ in flatten_to_fc],
            risks=["Spatial structure lost before classification",
                   "Flatten without GAP increases parameter count"],
            recommendations=[
                "Consider GAP instead of Flatten for spatial aggregation"
            ] if flatten_to_fc and not self._find_nodes_by_op({"GlobalAveragePool"}) else []
        )
    
    def _check_single_fc(self) -> AuditCheck:
        """Check for single large FC layer at the end."""
        fc_nodes = self._find_nodes_by_op(self.FC_OPS)
        
        # Find FC nodes near output
        output_nodes = [nid for nid in self.nodes if not self.adjacency.get(nid)]
        
        final_fc = []
        for fc in fc_nodes:
            # Check if this FC leads to output
            succs = self._get_successors(fc, 3)
            if any(s in output_nodes or self._get_op_type(s) == "Softmax" for s in succs):
                final_fc.append(fc)
        
        return AuditCheck(
            id="H3",
            category=AuditCategory.CLASSIFIER,
            name="Single large FC layer",
            status=AuditStatus.INFO,
            description="Analyzes final FC layer configuration.",
            findings=[
                f"Total FC layers: {len(fc_nodes)}",
                f"Final FC layers: {len(final_fc)}",
            ],
            affected_nodes=final_fc,
            risks=["Feature-space logit attacks (C&W)",
                   "Margin manipulation attacks",
                   "Single FC provides direct path from features to logits"],
            recommendations=[
                "Consider multiple smaller FC layers",
                "Add dropout before final FC",
                "Implement logit squeezing"
            ]
        )
    
    # =========================================================================
    # I. GRAPH HYGIENE AUDIT
    # =========================================================================
    
    def _audit_graph_hygiene(self) -> List[AuditCheck]:
        """Audit graph hygiene and cleanliness."""
        checks = []
        
        # I1: Unused initializers
        checks.append(self._check_unused_initializers())
        
        # I2: Dynamic shape ops
        checks.append(self._check_dynamic_shapes())
        
        # I3: Custom/unsupported ops
        checks.append(self._check_custom_ops())
        
        return checks
    
    def _check_unused_initializers(self) -> AuditCheck:
        """Check for unused initializers (dead weights)."""
        # Would need actual initializer info from parser
        # For now, look for nodes with no incoming edges that aren't inputs
        
        potential_unused = []
        for nid in self.nodes:
            if not self.reverse_adjacency.get(nid) and not self.adjacency.get(nid):
                potential_unused.append(nid)
        
        return AuditCheck(
            id="I1",
            category=AuditCategory.GRAPH_HYGIENE,
            name="Unused initializers",
            status=AuditStatus.WARN if potential_unused else AuditStatus.PASS,
            description="Checks for potentially unused weights/initializers.",
            findings=[
                f"Isolated nodes (potential unused): {len(potential_unused)}",
            ],
            affected_nodes=potential_unused,
            risks=["Unused capacity could hide backdoor weights",
                   "ShadowLogic implantation opportunity"],
            recommendations=[
                "Remove unused initializers",
                "Audit all model weights for utilization"
            ] if potential_unused else []
        )
    
    def _check_dynamic_shapes(self) -> AuditCheck:
        """Check for dynamic shape operations."""
        dynamic_ops = self._find_nodes_by_op({"Shape", "Gather", "DynamicSlice", "NonZero", "Where"})
        
        shape_ops = self._find_nodes_by_op({"Shape"})
        gather_ops = self._find_nodes_by_op({"Gather"})
        
        return AuditCheck(
            id="I2",
            category=AuditCategory.GRAPH_HYGIENE,
            name="Dynamic shape operations",
            status=AuditStatus.INFO if dynamic_ops else AuditStatus.PASS,
            description="Identifies operations that depend on runtime shapes.",
            findings=[
                f"Shape ops: {len(shape_ops)}",
                f"Gather ops: {len(gather_ops)}",
                f"Total dynamic ops: {len(dynamic_ops)}"
            ],
            affected_nodes=dynamic_ops,
            risks=["Shape-dependent control flow can be exploited",
                   "Dynamic behavior may differ between benign and adversarial inputs"],
            recommendations=[
                "Ensure dynamic ops handle adversarial shapes safely"
            ] if dynamic_ops else []
        )
    
    def _check_custom_ops(self) -> AuditCheck:
        """Check for custom or unsupported operations."""
        standard_ops = (self.CONV_OPS | self.POOL_OPS | self.NORM_OPS | 
                       self.LINEAR_OPS | self.ACTIVATION_OPS | self.SHAPE_OPS |
                       self.FUSION_OPS | {"Softmax", "Dropout", "Identity",
                       "Constant", "Shape", "Gather", "Slice", "Pad", "Cast",
                       "Resize", "Upsample", "Split", "Clip", "Mul", "Div",
                       "Sub", "Pow", "Sqrt", "Log", "Exp", "ReduceMean",
                       "ReduceSum", "ReduceMax", "Tile", "Expand", "Unsqueeze"})
        
        custom_ops = []
        for nid, node in self.nodes.items():
            op_type = node.get("op_type", "Unknown")
            if op_type not in standard_ops:
                custom_ops.append((nid, op_type))
        
        unique_custom = set(op for _, op in custom_ops)
        
        return AuditCheck(
            id="I3",
            category=AuditCategory.GRAPH_HYGIENE,
            name="Custom/unsupported operations",
            status=AuditStatus.WARN if custom_ops else AuditStatus.PASS,
            description="Identifies non-standard operations that may need manual audit.",
            findings=[
                f"Custom operations found: {len(custom_ops)}",
                f"Unique custom op types: {unique_custom}" if custom_ops else "All standard ops"
            ],
            affected_nodes=[nid for nid, _ in custom_ops],
            risks=["Custom ops cannot be automatically audited",
                   "May contain implementation vulnerabilities",
                   "Potential ShadowLogic hiding place"],
            recommendations=[
                "Manually audit all custom operations",
                "Document security properties of custom ops",
                "Consider replacing with standard ops where possible"
            ] if custom_ops else []
        )
    
    # =========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    
    def _compile_risks(self, checks: List[AuditCheck]) -> Dict[str, List[str]]:
        """Compile all risks by attack type."""
        risk_map = defaultdict(list)
        
        for check in checks:
            if check.status in [AuditStatus.FAIL, AuditStatus.WARN]:
                for risk in check.risks:
                    risk_map[risk].append(check.id)
        
        return dict(risk_map)
    
    def _generate_hardening(self, checks: List[AuditCheck]) -> List[HardeningRecommendation]:
        """Generate hardening recommendations based on findings."""
        recommendations = []
        
        failed_checks = [c for c in checks if c.status == AuditStatus.FAIL]
        warn_checks = [c for c in checks if c.status == AuditStatus.WARN]
        
        # MaxPool -> AvgPool
        maxpool_issues = [c for c in checks if "MaxPool" in c.name and c.status in [AuditStatus.FAIL, AuditStatus.WARN]]
        if maxpool_issues:
            recommendations.append(HardeningRecommendation(
                id="H1",
                title="Replace MaxPool with AvgPool / BlurPool",
                description="MaxPool amplifies sparse perturbations. Replace with average pooling "
                           "or use BlurPool for anti-aliased downsampling.",
                priority="high",
                affected_checks=[c.id for c in maxpool_issues],
                implementation_notes="Average pooling (AvgPool) or antialiased-cnns library for BlurPool"
            ))
        
        # Anti-alias downsampling
        aliasing_issues = [c for c in checks if "alias" in c.name.lower() or "stride" in c.name.lower()]
        if any(c.status in [AuditStatus.FAIL, AuditStatus.WARN] for c in aliasing_issues):
            recommendations.append(HardeningRecommendation(
                id="H2",
                title="Add anti-aliasing before downsampling",
                description="Apply Gaussian blur or learned blur before strided operations "
                           "to prevent aliasing artifacts that attackers can exploit.",
                priority="high",
                affected_checks=[c.id for c in aliasing_issues],
                implementation_notes="Use blur kernel before stride-2 ops, or BlurPool"
            ))
        
        # Gated feature fusion
        fusion_issues = [c for c in checks if "fusion" in c.name.lower() or "concat" in c.name.lower()]
        if any(c.status in [AuditStatus.FAIL, AuditStatus.WARN] for c in fusion_issues):
            recommendations.append(HardeningRecommendation(
                id="H3",
                title="Implement gated feature fusion",
                description="Use attention or gating mechanisms to weight branch contributions "
                           "instead of raw concatenation.",
                priority="medium",
                affected_checks=[c.id for c in fusion_issues],
                implementation_notes="SE blocks, CBAM, or simple channel attention after concat"
            ))
        
        # Channel clipping after Concat
        concat_checks = [c for c in checks if "Concat" in c.name]
        if any(c.status in [AuditStatus.FAIL, AuditStatus.WARN] for c in concat_checks):
            recommendations.append(HardeningRecommendation(
                id="H4",
                title="Add channel clipping after Concat",
                description="Clip channel values after concatenation to bound the range "
                           "and prevent adversarial amplification.",
                priority="medium",
                affected_checks=[c.id for c in concat_checks],
                implementation_notes="Clamp activations or use bounded activation after concat"
            ))
        
        # GroupNorm instead of BatchNorm
        bn_checks = [c for c in checks if "BatchNorm" in c.name or "BN" in c.name]
        if any(c.status in [AuditStatus.WARN] for c in bn_checks):
            recommendations.append(HardeningRecommendation(
                id="H5",
                title="Use GroupNorm instead of BatchNorm",
                description="GroupNorm is more robust to distribution shift attacks "
                           "as it doesn't depend on batch statistics.",
                priority="medium",
                affected_checks=[c.id for c in bn_checks],
                implementation_notes="GroupNorm with num_groups=32 typically"
            ))
        
        # Feature denoising before GAP
        gap_checks = [c for c in checks if "Global" in c.name]
        if gap_checks:
            recommendations.append(HardeningRecommendation(
                id="H6",
                title="Add feature denoising before GAP",
                description="Apply non-local means or learned denoising before global pooling "
                           "to filter adversarial noise from features.",
                priority="low",
                affected_checks=[c.id for c in gap_checks],
                implementation_notes="Feature Denoising for Improving Adversarial Robustness (Xie et al.)"
            ))
        
        # Spectral normalization
        linear_issues = [c for c in checks if "linear" in c.name.lower() or "chain" in c.name.lower()]
        if any(c.status in [AuditStatus.FAIL, AuditStatus.WARN] for c in linear_issues):
            recommendations.append(HardeningRecommendation(
                id="H7",
                title="Apply spectral normalization / Lipschitz regularization",
                description="Bound the spectral norm of weight matrices to limit "
                           "adversarial perturbation amplification.",
                priority="high",
                affected_checks=[c.id for c in linear_issues],
                implementation_notes="Spectral normalization or explicit Lipschitz training"
            ))
        
        return recommendations
    
    def _calculate_grade(self, report: CNNAuditReport) -> str:
        """Calculate overall security grade."""
        if report.total_checks == 0:
            return "?"
        
        fail_ratio = report.failed / report.total_checks
        warn_ratio = report.warnings / report.total_checks
        
        if fail_ratio > 0.3:
            return "F"
        elif fail_ratio > 0.2:
            return "D"
        elif fail_ratio > 0.1 or warn_ratio > 0.4:
            return "C"
        elif fail_ratio > 0.05 or warn_ratio > 0.25:
            return "B"
        elif warn_ratio > 0.1:
            return "B+"
        else:
            return "A"


def generate_audit_report_text(report: CNNAuditReport) -> str:
    """Generate human-readable audit report."""
    lines = [
        "=" * 70,
        "CNN DAG SECURITY AUDIT REPORT",
        "=" * 70,
        f"\nModel: {report.model_name}",
        f"Type: {report.model_type}",
        f"Security Grade: {report.security_grade}",
        f"\nSummary: {report.passed} PASS | {report.failed} FAIL | {report.warnings} WARN",
        f"Total Checks: {report.total_checks}",
    ]
    
    # Group checks by category
    by_category = defaultdict(list)
    for check in report.checks:
        by_category[check.category].append(check)
    
    for category in AuditCategory:
        checks = by_category.get(category, [])
        if not checks:
            continue
        
        lines.append(f"\n{'-' * 70}")
        lines.append(f"{category.value}")
        lines.append(f"{'-' * 70}")
        
        for check in checks:
            status_icon = {
                AuditStatus.PASS: "[PASS]",
                AuditStatus.FAIL: "[FAIL]",
                AuditStatus.WARN: "[WARN]",
                AuditStatus.INFO: "[INFO]",
                AuditStatus.NA: "[N/A]"
            }[check.status]
            
            lines.append(f"\n{check.id}. {status_icon} {check.name}")
            
            for finding in check.findings:
                if finding:
                    lines.append(f"    {finding}")
            
            if check.status in [AuditStatus.FAIL, AuditStatus.WARN] and check.risks:
                lines.append(f"    Risks: {', '.join(check.risks[:2])}")
            
            if check.recommendations:
                lines.append(f"    Recommendation: {check.recommendations[0]}")
    
    # Hardening recommendations
    if report.hardening:
        lines.append(f"\n{'=' * 70}")
        lines.append("HARDENING RECOMMENDATIONS")
        lines.append(f"{'=' * 70}")
        
        for rec in report.hardening:
            lines.append(f"\n[{rec.priority.upper()}] {rec.title}")
            lines.append(f"    {rec.description}")
            lines.append(f"    Implementation: {rec.implementation_notes}")
    
    # Risk summary
    if report.risk_areas:
        lines.append(f"\n{'=' * 70}")
        lines.append("RISK AREAS SUMMARY")
        lines.append(f"{'=' * 70}")
        
        for risk, check_ids in sorted(report.risk_areas.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            lines.append(f"\n- {risk}")
            lines.append(f"    Affected checks: {', '.join(check_ids)}")
    
    lines.append(f"\n{'=' * 70}")
    
    return "\n".join(lines)

