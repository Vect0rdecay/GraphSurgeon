"""
Structural Pattern Analysis for Adversarial ML Security Research

Detects high-risk architectural patterns and robustness indicators
in neural network DAGs that are relevant for adversarial attacks.

Based on security research workflows for identifying:
- Gradient bottlenecks
- Feature fusion points
- Amplification layers
- Attack surface mapping
- Defense placement evaluation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from graph_surgeon.graph.topology import GraphTopologyConfig


class PatternRisk(Enum):
    """Risk level of detected patterns."""
    CRITICAL = "critical"    # Highly exploitable
    HIGH = "high"            # Significant attack surface
    MEDIUM = "medium"        # Moderate concern
    LOW = "low"              # Minor concern
    POSITIVE = "positive"    # Robustness indicator (good)


class PatternCategory(Enum):
    """Categories of structural patterns."""
    PERTURBATION_FUSION = "perturbation_fusion"
    AMPLIFICATION = "amplification"
    GRADIENT_FLOW = "gradient_flow"
    FEATURE_EXTRACTION = "feature_extraction"
    ROBUSTNESS = "robustness"
    ATTACK_SURFACE = "attack_surface"


@dataclass
class StructuralPattern:
    """Represents a detected structural pattern in the model."""
    id: str
    name: str
    category: PatternCategory
    risk: PatternRisk
    nodes_involved: List[str]
    description: str
    attack_implications: str
    research_notes: str
    recommendations: List[str] = field(default_factory=list)
    registry_id: Optional[str] = None
    research_basis: List[str] = field(default_factory=list)
    attacks_enabled: List[str] = field(default_factory=list)


@dataclass
class AttackSurfaceMapping:
    """Maps model components to attack classes."""
    component: str
    node_ids: List[str]
    attack_class: str
    attack_techniques: List[str]
    exploitation_notes: str


@dataclass
class StructuralAnalysisReport:
    """Complete structural analysis of model architecture."""
    model_name: str
    
    # Detected patterns
    high_risk_patterns: List[StructuralPattern] = field(default_factory=list)
    robustness_indicators: List[StructuralPattern] = field(default_factory=list)
    
    # Attack surface mapping
    attack_surfaces: List[AttackSurfaceMapping] = field(default_factory=list)
    
    # Graph metrics
    total_nodes: int = 0
    max_depth: int = 0
    max_fan_in: int = 0
    max_fan_out: int = 0
    longest_linear_chain: int = 0
    
    # Scores
    structural_score: float = 0.0  # Higher = more attack-surface exposure
    robustness_score: float = 0.0  # Higher = more robust

    @property
    def vulnerability_score(self) -> float:
        """Deprecated alias for structural_score."""
        return self.structural_score

    @vulnerability_score.setter
    def vulnerability_score(self, value: float) -> None:
        self.structural_score = value
    
    # Research workflow outputs
    gradient_bottlenecks: List[str] = field(default_factory=list)
    feature_fusion_points: List[str] = field(default_factory=list)
    amplification_layers: List[str] = field(default_factory=list)
    recommended_defense_points: List[str] = field(default_factory=list)


class StructuralPatternAnalyzer:
    """
    Analyzes neural network DAG structure for security-relevant patterns.
    
    Detects:
    - High-risk structures that amplify adversarial perturbations
    - Robustness indicators that provide natural defense
    - Attack surface mapping for targeted research
    """
    
    # Operations that can amplify perturbations
    AMPLIFICATION_OPS = {"MaxPool", "Relu", "LeakyRelu", "Exp", "Pow"}
    
    # Operations that fuse/aggregate perturbations
    FUSION_OPS = {"Concat", "Add", "Sum", "Mean"}
    
    # Linear operations (can form dangerous chains)
    LINEAR_OPS = {"Conv", "ConvTranspose", "MatMul", "Gemm", "Linear"}
    
    # Attention-like operations (high sensitivity)
    ATTENTION_OPS = {"MatMul", "Softmax", "Attention", "MultiHeadAttention", 
                     "ScaledDotProductAttention", "CrossAttention"}
    
    # Normalization ops (distribution shift targets)
    NORM_OPS = {"BatchNormalization", "LayerNormalization", "InstanceNormalization", 
                "GroupNormalization", "BatchNorm", "LayerNorm"}
    
    # Pooling operations
    POOLING_OPS = {"MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", 
                   "AdaptiveAvgPool2d", "AdaptiveMaxPool2d"}
    
    # Global pooling operations (feature-space attack targets)
    GLOBAL_POOLING_OPS = {"GlobalAveragePool", "GlobalMaxPool", "AdaptiveAvgPool2d", 
                          "AdaptiveMaxPool2d", "ReduceMean", "ReduceMax"}
    
    # Robustness-positive operations
    ROBUST_OPS = {"AveragePool", "GlobalAveragePool", "Dropout", "DropPath"}
    
    # Saturating activation functions (boundary attack targets)
    SATURATING_ACTIVATIONS = {"Sigmoid", "Tanh", "HardSigmoid", "HardTanh", "Softsign"}
    
    # Shape/view operations (prompt injection, carrier attack targets)
    SHAPE_OPS = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", 
                 "Permute", "View", "Expand", "Tile", "Gather", "Scatter",
                 "Split", "Slice", "Pad", "Crop"}
    
    # Multimodal fusion operations (cross-modal jailbreak targets)
    MULTIMODAL_OPS = {"Concat", "Add", "Mul", "CrossAttention", "FusedAttention",
                      "MultimodalFusion", "FeatureFusion"}
    
    # FC/Dense layer operations (margin attack targets)
    FC_OPS = {"Gemm", "Linear", "MatMul", "Dense", "FullyConnected"}
    
    # =========================================================================
    # VULNERABILITY -> ATTACK CLASS MAPPINGS
    # =========================================================================
    
    # Operations that indicate potential ShadowLogic injection points
    SHADOWLOGIC_INDICATORS = {"Where", "If", "Equal", "Less", "Greater", "And", "Or", "Not",
                              "Select", "Cond", "Switch", "Case"}
    
    # ReLU family (piecewise linear, gradient-friendly for attacks)
    RELU_FAMILY = {"Relu", "LeakyRelu", "PRelu", "Elu", "Selu", "Celu", "ThresholdedRelu"}
    
    ATTACK_SURFACE_MAPPING = {
        "linear_chains": {
            "attacks": ["FGSM", "PGD", "C&W", "AutoAttack", "DeepFool"],
            "description": "Long chains of linear operations create gradient highways "
                          "that enable efficient gradient-based optimization attacks.",
            "exploitation": "Gradients flow unimpeded through linear chains, allowing "
                           "attacks to compute exact perturbation directions."
        },
        "concat_add": {
            "attacks": ["Multi-scale PGD", "Transfer attacks", "Universal perturbations", 
                       "Ensemble attacks"],
            "description": "Fusion operations aggregate perturbations from multiple paths, "
                          "enabling coordinated multi-scale attacks.",
            "exploitation": "Attacker can inject perturbations into each branch and have "
                           "them combine constructively at the fusion point."
        },
        "residuals": {
            "attacks": ["PGD", "MI-FGSM (Momentum)", "C&W", "Skip Gradient Method"],
            "description": "Residual connections provide direct gradient paths that "
                          "bypass non-linearities and enable deep network attacks.",
            "exploitation": "Skip connections allow gradients to flow directly to early "
                           "layers, making very deep networks as attackable as shallow ones."
        },
        "maxpool": {
            "attacks": ["One-pixel attack", "Sparse attacks", "JSMA", "LocSearchAdv"],
            "description": "Max pooling selects maximum values, allowing sparse perturbations "
                          "to dominate the output.",
            "exploitation": "Attacker only needs to perturb one pixel per pooling region "
                           "to fully control the pooled output."
        },
        "early_stride": {
            "attacks": ["Fourier attacks", "Frequency-domain attacks", "Low-frequency perturbations",
                       "Spectral attacks"],
            "description": "Early strided convolutions create aliasing vulnerabilities and "
                          "are susceptible to frequency-domain attacks.",
            "exploitation": "Strided operations without anti-aliasing can be exploited by "
                           "perturbations at specific spatial frequencies."
        },
        "batchnorm": {
            "attacks": ["Distribution shift attacks", "BN-targeted attacks", "Adaptive attacks",
                       "Statistics manipulation"],
            "description": "BatchNorm layers normalize using fixed statistics that assume "
                          "a specific input distribution.",
            "exploitation": "Adversarial inputs can shift the effective distribution, causing "
                           "normalization to produce unexpected outputs."
        },
        "attention": {
            "attacks": ["Attention hijacking", "Token manipulation", "Adversarial patches (ViT)",
                       "Attention rollout attacks"],
            "description": "Attention mechanisms are highly sensitive to input perturbations "
                          "that can redirect attention weights.",
            "exploitation": "Small perturbations can cause attention to focus entirely on "
                           "adversarial tokens/patches while ignoring legitimate content."
        },
        "global_pooling": {
            "attacks": ["Feature-space attacks", "Deep feature attacks", "Activation maximization",
                       "Layer-wise attacks"],
            "description": "Global pooling aggregates spatial information into a feature vector, "
                          "which can be targeted directly.",
            "exploitation": "Attacks can optimize perturbations to manipulate the global feature "
                           "representation rather than individual activations."
        },
        "fc_layers": {
            "attacks": ["C&W attack", "Margin attacks", "Logit manipulation", "Decision boundary attacks",
                       "Targeted misclassification"],
            "description": "Fully connected layers produce final logits that directly determine "
                          "classification, making them high-value targets.",
            "exploitation": "Direct optimization of perturbations to manipulate logit differences "
                           "and cross decision boundaries."
        },
        "shape_ops": {
            "attacks": ["Prompt injection", "Carrier attacks", "Data smuggling", "Shape confusion",
                       "Tensor manipulation"],
            "description": "Shape operations reorganize data without modification, which can be "
                          "exploited to smuggle adversarial payloads.",
            "exploitation": "Attackers can craft inputs that appear benign but become adversarial "
                           "after shape transformations, or use shape ops to encode hidden data."
        },
        "saturating_activations": {
            "attacks": ["Boundary attacks", "Gradient masking exploitation", "Black-box attacks",
                       "Transfer attacks"],
            "description": "Saturating activations (sigmoid, tanh) have vanishing gradients at "
                          "extremes, which can mask but not prevent attacks.",
            "exploitation": "While gradient-based attacks may struggle, black-box and transfer "
                           "attacks remain effective. Gradient masking is NOT a defense."
        },
        "multimodal_fusion": {
            "attacks": ["Cross-modal jailbreaks", "Multimodal adversarial examples", 
                       "Modality confusion", "Cross-modal transfer"],
            "description": "Multimodal fusion points combine information from different modalities, "
                          "creating cross-modal attack surfaces.",
            "exploitation": "Perturbations in one modality (e.g., image) can influence processing "
                           "of another modality (e.g., text), enabling jailbreaks and bypasses."
        },
        "shadowlogic_susceptible": {
            "attacks": ["ShadowLogic backdoor", "Subnet replacement", "Trojan insertion",
                       "Conditional backdoor", "Trigger-based attacks"],
            "description": "Model architecture has characteristics that make it susceptible to "
                          "ShadowLogic-style backdoor attacks where malicious subnets can be "
                          "embedded without affecting normal operation.",
            "exploitation": "Attacker identifies unused capacity or conditional paths and embeds "
                           "trigger-activated malicious logic that only fires on specific inputs."
        },
        "valid_conv_boundary": {
            "attacks": ["Boundary manipulation", "Edge attacks", "Padding exploits",
                       "Corner perturbations", "Frame injection"],
            "description": "Valid convolutions (pads=0) see fewer contexts near edges, creating "
                          "sensitivity to boundary manipulations.",
            "exploitation": "Model has reduced receptive field at image boundaries. Adversarial "
                           "perturbations placed at edges/corners have outsized influence due to "
                           "asymmetric context availability."
        },
        "early_linear_no_norm": {
            "attacks": ["FGSM", "PGD", "DeepFool", "Fast gradient attacks", "Iterative attacks"],
            "description": "Long early linear feature extraction without normalization creates "
                          "attack-friendly gradient flow. ReLU is piecewise-linear, so early "
                          "networks have stable, informative gradients.",
            "exploitation": "Stable gradient flow in early layers makes gradient-based attacks "
                           "highly effective. No normalization means gradients aren't rescaled, "
                           "preserving attack signal strength."
        },
        "relu_no_lipschitz": {
            "attacks": ["Decision boundary attacks", "Gradient-based attacks", "Region boundary exploits",
                       "Activation manipulation"],
            "description": "ReLU creates sharp decision boundaries at zero crossings. Without "
                          "Lipschitz constraints, small input changes cause large output changes "
                          "near these boundaries.",
            "exploitation": "ReLU's piecewise-linear nature creates exploitable region boundaries. "
                           "Perturbations that push activations across the ReLU threshold have "
                           "maximum impact. No spectral normalization means unbounded amplification."
        },
        "no_gradient_regularization": {
            "attacks": ["All gradient-based attacks", "High-confidence adversarial examples",
                       "Minimal perturbation attacks"],
            "description": "Network lacks gradient norm regularization, meaning gradients remain "
                          "stable and informative for adversarial optimization.",
            "exploitation": "Without gradient penalties or noise, attackers get clean gradient "
                           "signals for crafting adversarial examples. Optimization-based attacks "
                           "converge quickly and reliably."
        }
    }
    
    def __init__(self, topology_config: Optional[GraphTopologyConfig] = None):
        self.topology_config = topology_config or GraphTopologyConfig()
        self.nodes: Dict[str, Dict] = {}  # node_id -> node info
        self.edges: List[Tuple[str, str]] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> successors
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> predecessors

    def _early_depth_threshold(self, max_depth: int) -> int:
        return self.topology_config.early_depth_threshold(max_depth)

    def _late_depth_threshold(self, max_depth: int) -> int:
        return self.topology_config.late_depth_threshold(max_depth)

    def _pattern_from_registry(
        self,
        gadget_id: str,
        pattern_id: str,
        nodes_involved: List[str],
        category: PatternCategory,
        *,
        attack_implications: str = "",
        research_notes: str = "",
    ) -> StructuralPattern:
        from graph_surgeon.taxonomy.display import format_pattern_from_registry

        data = format_pattern_from_registry(
            gadget_id,
            pattern_id=pattern_id,
            nodes_involved=nodes_involved,
        )
        return StructuralPattern(
            id=data["id"],
            name=data["name"],
            category=category,
            risk=PatternRisk.MEDIUM,
            nodes_involved=nodes_involved,
            description=data["description"],
            attack_implications=attack_implications or (
                f"Associated attack classes from literature: "
                f"{', '.join(data.get('attacks_enabled', []))}"
            ),
            research_notes=research_notes,
            registry_id=gadget_id,
            research_basis=data.get("research_basis", []),
            attacks_enabled=data.get("attacks_enabled", []),
        )
        
    def build_graph(self, nodes: List[Dict], edges: List[Tuple[str, str]]):
        """Build internal graph representation from nodes and edges."""
        self.nodes = {n["node_id"]: n for n in nodes}
        self.edges = edges
        
        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        
        for src, dst in edges:
            self.adjacency[src].append(dst)
            self.reverse_adjacency[dst].append(src)
    
    def analyze(self, nodes: List[Dict], edges: List[Tuple[str, str]], 
                model_name: str = "model") -> StructuralAnalysisReport:
        """
        Perform complete structural analysis of the model.
        
        Args:
            nodes: List of node dicts with 'node_id', 'op_type', 'attributes'
            edges: List of (src_node_id, dst_node_id) tuples
            model_name: Name for the report
            
        Returns:
            StructuralAnalysisReport with all findings
        """
        self.build_graph(nodes, edges)
        
        report = StructuralAnalysisReport(
            model_name=model_name,
            total_nodes=len(nodes)
        )
        
        # Compute graph metrics
        report.max_depth = self._compute_max_depth()
        report.max_fan_in = self._compute_max_fan_in()
        report.max_fan_out = self._compute_max_fan_out()
        report.longest_linear_chain = self._find_longest_linear_chain()
        
        # Detect high-risk patterns
        report.high_risk_patterns.extend(self._detect_concat_fusion())
        report.high_risk_patterns.extend(self._detect_maxpool_amplification())
        report.high_risk_patterns.extend(self._detect_linear_chains())
        report.high_risk_patterns.extend(self._detect_large_fan_in())
        report.high_risk_patterns.extend(self._detect_attention_patterns())
        report.high_risk_patterns.extend(self._detect_residual_explosions())
        report.high_risk_patterns.extend(self._detect_early_stride())
        report.high_risk_patterns.extend(self._detect_batchnorm_vuln())
        report.high_risk_patterns.extend(self._detect_global_pooling_vuln())
        report.high_risk_patterns.extend(self._detect_fc_layer_vuln())
        report.high_risk_patterns.extend(self._detect_shape_ops_vuln())
        report.high_risk_patterns.extend(self._detect_saturating_activations())
        report.high_risk_patterns.extend(self._detect_multimodal_fusion())
        report.high_risk_patterns.extend(self._detect_shadowlogic_susceptibility())
        report.high_risk_patterns.extend(self._detect_valid_conv_boundary())
        report.high_risk_patterns.extend(self._detect_early_linear_no_norm())
        report.high_risk_patterns.extend(self._detect_relu_no_lipschitz())
        report.high_risk_patterns.extend(self._detect_no_gradient_regularization())
        
        # Detect robustness indicators
        report.robustness_indicators.extend(self._detect_early_downsampling())
        report.robustness_indicators.extend(self._detect_average_pooling())
        report.robustness_indicators.extend(self._detect_bottleneck_convs())
        report.robustness_indicators.extend(self._detect_reduced_early_depth())
        
        # Map attack surfaces
        report.attack_surfaces = self._map_attack_surfaces()
        
        # Identify research workflow outputs
        report.gradient_bottlenecks = self._find_gradient_bottlenecks()
        report.feature_fusion_points = self._find_feature_fusion_points()
        report.amplification_layers = self._find_amplification_layers()

        return report
    
    # =========================================================================
    # GRAPH METRICS
    # =========================================================================
    
    def _compute_max_depth(self) -> int:
        """Compute maximum depth (longest path) in the DAG."""
        depths = {}
        
        def get_depth(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]
            
            predecessors = self.reverse_adjacency.get(node_id, [])
            if not predecessors:
                depths[node_id] = 0
            else:
                depths[node_id] = 1 + max(get_depth(p) for p in predecessors)
            
            return depths[node_id]
        
        max_depth = 0
        for node_id in self.nodes:
            max_depth = max(max_depth, get_depth(node_id))
        
        return max_depth
    
    def _compute_max_fan_in(self) -> int:
        """Find maximum fan-in (number of inputs to any node)."""
        return max((len(preds) for preds in self.reverse_adjacency.values()), default=0)
    
    def _compute_max_fan_out(self) -> int:
        """Find maximum fan-out (number of outputs from any node)."""
        return max((len(succs) for succs in self.adjacency.values()), default=0)
    
    def _find_longest_linear_chain(self) -> int:
        """Find longest chain of consecutive linear operations."""
        max_chain = 0
        
        for start_node in self.nodes:
            if self.nodes[start_node].get("op_type") not in self.LINEAR_OPS:
                continue
            
            # BFS to find chain length
            chain_length = 1
            current = start_node
            
            while True:
                successors = self.adjacency.get(current, [])
                linear_successors = [
                    s for s in successors 
                    if s in self.nodes and self.nodes[s].get("op_type") in self.LINEAR_OPS
                ]
                
                if not linear_successors:
                    break
                
                current = linear_successors[0]
                chain_length += 1
            
            max_chain = max(max_chain, chain_length)
        
        return max_chain
    
    # =========================================================================
    # HIGH-RISK PATTERN DETECTION
    # =========================================================================
    
    def _detect_concat_fusion(self) -> List[StructuralPattern]:
        """
        Detect Concat operations that fuse perturbations.
        
        Concat(axis=1) is particularly dangerous as it combines features
        from multiple paths, allowing coordinated multi-scale attacks.
        """
        patterns = []
        
        for node_id, node in self.nodes.items():
            if node.get("op_type") != "Concat":
                continue
            
            axis = node.get("attributes", {}).get("axis", 1)
            num_inputs = len(self.reverse_adjacency.get(node_id, []))
            
            if num_inputs >= 2:
                risk = PatternRisk.HIGH if axis == 1 else PatternRisk.MEDIUM
                
                patterns.append(StructuralPattern(
                    id=f"CONCAT-FUSION-{node_id}",
                    name=f"Perturbation Fusion at {node_id}",
                    category=PatternCategory.PERTURBATION_FUSION,
                    risk=risk,
                    nodes_involved=[node_id] + self.reverse_adjacency.get(node_id, []),
                    description=f"Concat operation merges {num_inputs} input paths along axis={axis}. "
                               f"This creates a perturbation aggregation point where adversarial "
                               f"signals from multiple branches combine.",
                    attack_implications="""
                        - Multi-scale coordinated attacks can inject perturbations into each branch
                        - The fusion point amplifies combined adversarial signals
                        - Inception-style architectures are particularly vulnerable
                        - Attacker can exploit branch diversity for more effective perturbations
                    """,
                    research_notes="""
                        This is a key point for feature-space attacks. Trace back each input
                        branch to understand what features are being fused. Consider whether
                        perturbations in one branch could dominate or cancel perturbations
                        from others.
                    """,
                    recommendations=[
                        "Consider attention-weighted fusion instead of raw concatenation",
                        "Add normalization before concat to balance branch contributions",
                        "Evaluate branch-specific adversarial training"
                    ]
                ))
        
        return patterns
    
    def _detect_maxpool_amplification(self) -> List[StructuralPattern]:
        """Detect MaxPool after fusion (MAXPOOL_AFTER_FUSION registry motif)."""
        patterns = []
        fusion_ops = self.FUSION_OPS
        after_fusion: List[str] = []

        for node_id, node in self.nodes.items():
            if node.get("op_type") not in {"MaxPool", "GlobalMaxPool"}:
                continue
            visited = {node_id}
            frontier = list(self.reverse_adjacency.get(node_id, []))
            found_fusion = False
            for _ in range(3):
                if found_fusion:
                    break
                next_frontier = []
                for pred in frontier:
                    if pred in visited:
                        continue
                    visited.add(pred)
                    op = self.nodes.get(pred, {}).get("op_type", "")
                    if op in fusion_ops:
                        found_fusion = True
                        break
                    next_frontier.extend(self.reverse_adjacency.get(pred, []))
                frontier = next_frontier

            if found_fusion:
                after_fusion.append(node_id)

        if after_fusion:
            patterns.append(
                self._pattern_from_registry(
                    "MAXPOOL_AFTER_FUSION",
                    "MAXPOOL-AFTER-FUSION",
                    after_fusion,
                    PatternCategory.AMPLIFICATION,
                    research_notes=(
                        "MaxPool within 3 hops downstream of Concat/Add fusion."
                    ),
                )
            )

        return patterns
    
    def _detect_linear_chains(self) -> List[StructuralPattern]:
        """
        Detect long chains of linear operations (Conv -> Conv -> Conv).
        
        Long linear chains create gradient highways that amplify
        adversarial perturbations through the network depth.
        """
        patterns = []
        visited = set()
        
        for start_node in self.nodes:
            if start_node in visited:
                continue
            if self.nodes[start_node].get("op_type") not in self.LINEAR_OPS:
                continue
            
            # Find chain
            chain = [start_node]
            current = start_node
            
            while True:
                successors = self.adjacency.get(current, [])
                linear_successors = [
                    s for s in successors 
                    if s in self.nodes and self.nodes[s].get("op_type") in self.LINEAR_OPS
                ]
                
                if not linear_successors:
                    break
                
                current = linear_successors[0]
                chain.append(current)
            
            visited.update(chain)
            
            if len(chain) >= 3:  # Chains of 3+ are concerning
                risk = PatternRisk.CRITICAL if len(chain) >= 5 else PatternRisk.HIGH
                
                patterns.append(StructuralPattern(
                    id=f"LINEAR-CHAIN-{chain[0]}",
                    name=f"Linear Chain ({len(chain)} ops)",
                    category=PatternCategory.GRADIENT_FLOW,
                    risk=risk,
                    nodes_involved=chain,
                    description=f"Chain of {len(chain)} consecutive linear operations without "
                               f"non-linearity breaks. This creates a gradient highway that "
                               f"amplifies perturbations proportionally to the product of "
                               f"weight matrix norms.",
                    attack_implications="""
                        - Gradient-based attacks have clear optimization path
                        - Perturbation amplification scales with chain length
                        - Each linear op multiplies perturbation by its spectral norm
                        - Effective Lipschitz constant is product of individual constants
                    """,
                    research_notes="""
                        Long linear chains indicate high effective Lipschitz constant.
                        Calculate the product of spectral norms to estimate total
                        amplification factor. These are prime targets for PGD attacks.
                    """,
                    recommendations=[
                        "Insert non-linearities to break gradient highways",
                        "Apply spectral normalization to each layer",
                        "Consider residual connections to distribute gradient flow"
                    ]
                ))
        
        return patterns
    
    def _detect_large_fan_in(self) -> List[StructuralPattern]:
        """
        Detect nodes with large fan-in (many input connections).
        
        Large fan-in creates aggregation points where perturbations
        from multiple paths combine, potentially amplifying attacks.
        """
        patterns = []
        
        for node_id in self.nodes:
            fan_in = len(self.reverse_adjacency.get(node_id, []))
            
            if fan_in >= 4:  # 4+ inputs is concerning
                op_type = self.nodes[node_id].get("op_type", "unknown")
                risk = PatternRisk.HIGH if fan_in >= 6 else PatternRisk.MEDIUM
                
                patterns.append(StructuralPattern(
                    id=f"FAN-IN-{node_id}",
                    name=f"High Fan-In Node ({fan_in} inputs)",
                    category=PatternCategory.PERTURBATION_FUSION,
                    risk=risk,
                    nodes_involved=[node_id] + self.reverse_adjacency.get(node_id, []),
                    description=f"Node '{node_id}' ({op_type}) receives {fan_in} inputs. "
                               f"This creates a perturbation aggregation point where "
                               f"adversarial signals from multiple paths converge.",
                    attack_implications="""
                        - Coordinated multi-path attacks can overwhelm this node
                        - Perturbations from each path may add constructively
                        - Dense connectivity provides many attack vectors
                    """,
                    research_notes="""
                        High fan-in nodes are natural targets for multi-objective attacks.
                        Consider how perturbations from each input path interact at this
                        node. DenseNet-style architectures often have extreme fan-in.
                    """,
                    recommendations=[
                        "Consider attention mechanisms to weight input importance",
                        "Add channel-wise normalization before aggregation",
                        "Evaluate pruning low-contribution connections"
                    ]
                ))
        
        return patterns
    
    def _detect_attention_patterns(self) -> List[StructuralPattern]:
        """
        Detect attention-like MatMul patterns (Q*K^T, attention*V).
        
        Attention mechanisms are highly sensitive to adversarial
        perturbations that can hijack the attention distribution.
        """
        patterns = []
        
        for node_id, node in self.nodes.items():
            op_type = node.get("op_type", "")
            
            if op_type in {"Attention", "MultiHeadAttention", "ScaledDotProductAttention"}:
                patterns.append(StructuralPattern(
                    id=f"ATTENTION-{node_id}",
                    name=f"Attention Mechanism at {node_id}",
                    category=PatternCategory.AMPLIFICATION,
                    risk=PatternRisk.CRITICAL,
                    nodes_involved=[node_id],
                    description=f"Explicit attention operation that computes weighted "
                               f"combinations of values based on query-key similarity. "
                               f"Attention weights are highly sensitive to input perturbations.",
                    attack_implications="""
                        - Attention hijacking: adversarial tokens/patches capture all attention
                        - Small perturbations cause large attention weight shifts
                        - Softmax over similarities is effectively unbounded Lipschitz
                        - Multi-head attention provides multiple attack vectors
                    """,
                    research_notes="""
                        Attention is one of the highest-risk components in modern architectures.
                        The softmax over similarities can cause arbitrary attention redistribution
                        with small input changes. Consider attention entropy as an anomaly signal.
                    """,
                    recommendations=[
                        "Implement attention entropy regularization",
                        "Add attention dropout during training and optionally inference",
                        "Consider Lipschitz-bounded attention variants",
                        "Evaluate attention masking for adversarial inputs"
                    ]
                ))
            
            # Detect implicit attention (MatMul followed by Softmax)
            elif op_type == "MatMul":
                successors = self.adjacency.get(node_id, [])
                softmax_after = any(
                    self.nodes.get(s, {}).get("op_type") == "Softmax"
                    for s in successors
                )
                
                if softmax_after:
                    patterns.append(StructuralPattern(
                        id=f"IMPLICIT-ATTN-{node_id}",
                        name=f"Implicit Attention Pattern at {node_id}",
                        category=PatternCategory.AMPLIFICATION,
                        risk=PatternRisk.HIGH,
                        nodes_involved=[node_id] + successors,
                        description=f"MatMul followed by Softmax suggests attention-like "
                                   f"computation (Q*K^T -> softmax). This pattern has "
                                   f"high adversarial sensitivity.",
                        attack_implications="""
                            - Same risks as explicit attention
                            - May be less defended than recognized attention layers
                            - Implicit patterns can be overlooked in security audits
                        """,
                        research_notes="""
                            Look for the value multiplication (attention_weights * V) after
                            the softmax to confirm this is attention. These implicit patterns
                            may not receive the same defensive attention as explicit attention layers.
                        """,
                        recommendations=[
                            "Replace with explicit attention layer for better tooling support",
                            "Apply same defenses as explicit attention"
                        ]
                    ))
        
        return patterns
    
    def _detect_residual_explosions(self) -> List[StructuralPattern]:
        """
        Detect patterns where residual connections could cause gradient explosion.
        
        Many consecutive residual blocks without proper normalization can
        lead to exponential gradient growth.
        """
        patterns = []
        
        # Find Add operations (typical residual connection)
        add_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") == "Add"]
        
        # Check for chains of Add operations (residual blocks)
        residual_chains = []
        visited = set()
        
        for add_node in add_nodes:
            if add_node in visited:
                continue
            
            chain = [add_node]
            current = add_node
            
            while True:
                successors = self.adjacency.get(current, [])
                # Look for another Add within reasonable distance (through convs, norms, etc.)
                found_next_add = None
                for _ in range(5):  # Look ahead up to 5 hops
                    next_level = []
                    for s in successors:
                        if s in self.nodes:
                            if self.nodes[s].get("op_type") == "Add" and s not in chain:
                                found_next_add = s
                                break
                            next_level.extend(self.adjacency.get(s, []))
                    if found_next_add:
                        break
                    successors = next_level
                
                if not found_next_add:
                    break
                
                chain.append(found_next_add)
                current = found_next_add
            
            visited.update(chain)
            residual_chains.append(chain)
        
        # Check for long chains without normalization
        for chain in residual_chains:
            if len(chain) >= 5:  # 5+ residual blocks in sequence
                # Check for normalization between blocks
                has_norm = False
                for i in range(len(chain) - 1):
                    # Check path between consecutive adds
                    path_nodes = self._get_path_between(chain[i], chain[i+1])
                    if any(self.nodes.get(n, {}).get("op_type") in self.NORM_OPS for n in path_nodes):
                        has_norm = True
                        break
                
                if not has_norm:
                    patterns.append(StructuralPattern(
                        id=f"RESIDUAL-EXPLOSION-{chain[0]}",
                        name=f"Residual Chain Risk ({len(chain)} blocks)",
                        category=PatternCategory.GRADIENT_FLOW,
                        risk=PatternRisk.HIGH,
                        nodes_involved=chain,
                        description=f"Chain of {len(chain)} residual connections without "
                                   f"apparent normalization. Residual connections pass "
                                   f"gradients directly, potentially causing exponential growth.",
                        attack_implications="""
                            - Gradient-based attacks benefit from direct gradient paths
                            - Perturbations can accumulate across residual blocks
                            - Deep residual networks without normalization are unstable
                        """,
                        research_notes="""
                            The effective Lipschitz constant grows with depth in residual
                            networks. Check if skip connections have learnable scaling
                            factors that could be exploited.
                        """,
                        recommendations=[
                            "Add normalization (BatchNorm, LayerNorm) in residual blocks",
                            "Consider pre-activation residual block design",
                            "Use stochastic depth during training"
                        ]
                    ))
        
        return patterns
    
    def _get_path_between(self, start: str, end: str) -> List[str]:
        """Get nodes on path between start and end."""
        # Simple BFS
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def _detect_early_stride(self) -> List[StructuralPattern]:
        """
        Detect early strided convolutions vulnerable to frequency attacks.
        
        Strided convolutions without anti-aliasing create aliasing artifacts
        that can be exploited by frequency-domain adversarial attacks.
        """
        patterns = []
        depths = self._compute_node_depths()
        max_depth = max(depths.values()) if depths else 0
        early_threshold = self._early_depth_threshold(max_depth)
        
        for node_id, node in self.nodes.items():
            if node.get("op_type") != "Conv":
                continue
            
            depth = depths.get(node_id, 0)
            if depth > early_threshold:
                continue
            
            attrs = node.get("attributes", {})
            strides = attrs.get("strides", [1, 1])
            
            if any(s > 1 for s in strides):
                attack_info = self.ATTACK_SURFACE_MAPPING["early_stride"]
                
                patterns.append(StructuralPattern(
                    id=f"EARLY-STRIDE-{node_id}",
                    name=f"Early Strided Conv at {node_id}",
                    category=PatternCategory.FEATURE_EXTRACTION,
                    risk=PatternRisk.MEDIUM,
                    nodes_involved=[node_id],
                    description=f"Strided convolution (stride={strides}) in early layers "
                               f"(depth {depth}/{max_depth}). Early striding creates aliasing "
                               f"vulnerabilities exploitable by frequency-domain attacks.",
                    attack_implications=f"""
                        Applicable attacks: {', '.join(attack_info['attacks'])}
                        
                        {attack_info['exploitation']}
                        
                        Strided convolutions downsample without proper anti-aliasing,
                        making the model sensitive to specific spatial frequencies.
                        Low-frequency perturbations and Fourier-basis attacks can be
                        particularly effective.
                    """,
                    research_notes="""
                        Check if blur/anti-aliasing is applied before striding.
                        Models with early aggressive downsampling are vulnerable to
                        frequency-crafted perturbations that survive the aliasing.
                    """,
                    recommendations=[
                        "Add anti-aliasing (blur) before strided convolutions",
                        "Consider replacing stride with blur+subsample",
                        "Evaluate adversarial training with frequency perturbations"
                    ]
                ))
        
        return patterns
    
    def _detect_batchnorm_vuln(self) -> List[StructuralPattern]:
        """
        Detect BatchNorm layers vulnerable to distribution shift attacks.
        """
        patterns = []
        
        bn_nodes = [nid for nid, n in self.nodes.items() 
                   if n.get("op_type") in self.NORM_OPS 
                   and "batch" in n.get("op_type", "").lower()]
        
        if bn_nodes:
            attack_info = self.ATTACK_SURFACE_MAPPING["batchnorm"]
            patterns.append(
                self._pattern_from_registry(
                    "NORMALIZER",
                    "BATCHNORM-MOTIF",
                    bn_nodes,
                    PatternCategory.ATTACK_SURFACE,
                    attack_implications=(
                        f"Literature attack classes: {', '.join(attack_info['attacks'])}. "
                        f"{attack_info['exploitation']}"
                    ),
                    research_notes=(
                        "At inference, BatchNorm uses fixed running statistics from training."
                    ),
                )
            )
        
        return patterns
    
    def _detect_global_pooling_vuln(self) -> List[StructuralPattern]:
        """Detect global pooling motifs (GAP_FC_HEAD registry)."""
        patterns = []
        gap_nodes = [
            nid for nid, n in self.nodes.items()
            if n.get("op_type") in self.GLOBAL_POOLING_OPS
        ]

        if gap_nodes:
            attack_info = self.ATTACK_SURFACE_MAPPING["global_pooling"]
            patterns.append(
                self._pattern_from_registry(
                    "GAP_FC_HEAD",
                    "GAP-FC-HEAD-MOTIF",
                    gap_nodes,
                    PatternCategory.FEATURE_EXTRACTION,
                    attack_implications=(
                        f"Literature attack classes: {', '.join(attack_info['attacks'])}. "
                        f"{attack_info['exploitation']}"
                    ),
                    research_notes="Global pooling collapses spatial dimensions before classification.",
                )
            )

        return patterns
    
    def _detect_fc_layer_vuln(self) -> List[StructuralPattern]:
        """
        Detect final FC layers vulnerable to margin/logit attacks.
        """
        patterns = []
        
        # Find FC layers near the output
        output_nodes = [nid for nid in self.nodes if not self.adjacency.get(nid)]
        
        for node_id, node in self.nodes.items():
            if node.get("op_type") not in self.FC_OPS:
                continue
            
            # Check if this FC connects to output or near-output
            is_final = False
            successors = self.adjacency.get(node_id, [])
            
            if not successors:
                is_final = True
            elif any(s in output_nodes for s in successors):
                is_final = True
            elif any(self.nodes.get(s, {}).get("op_type") == "Softmax" for s in successors):
                is_final = True
            
            if is_final:
                attack_info = self.ATTACK_SURFACE_MAPPING["fc_layers"]
                
                patterns.append(StructuralPattern(
                    id=f"FC-FINAL-{node_id}",
                    name=f"Final FC Layer (Logit Target) at {node_id}",
                    category=PatternCategory.ATTACK_SURFACE,
                    risk=PatternRisk.HIGH,
                    nodes_involved=[node_id],
                    description=f"Final fully connected layer '{node_id}' produces logits "
                               f"that directly determine classification. This is the ultimate "
                               f"target for classification attacks.",
                    attack_implications=f"""
                        Applicable attacks: {', '.join(attack_info['attacks'])}
                        
                        {attack_info['exploitation']}
                        
                        The logit margin (difference between top-2 classes) is what
                        adversarial attacks aim to flip. C&W attack directly optimizes
                        this margin. Small changes at this layer have maximum impact.
                    """,
                    research_notes="""
                        Analyze the weight matrix to understand which input features
                        most strongly influence each class. The margin between classes
                        determines attack difficulty. Monitor logit distributions for
                        anomalies.
                    """,
                    recommendations=[
                        "Implement logit squeezing or temperature scaling",
                        "Monitor for unusual logit patterns",
                        "Consider ensemble of final layers"
                    ]
                ))
        
        return patterns
    
    def _detect_shape_ops_vuln(self) -> List[StructuralPattern]:
        """
        Detect shape operations vulnerable to prompt injection and carrier attacks.
        """
        patterns = []
        
        shape_nodes = [nid for nid, n in self.nodes.items() 
                      if n.get("op_type") in self.SHAPE_OPS]
        
        if len(shape_nodes) >= 3:  # Multiple shape ops are concerning
            attack_info = self.ATTACK_SURFACE_MAPPING["shape_ops"]
            
            patterns.append(StructuralPattern(
                id=f"SHAPE-OPS-VULN",
                name=f"Shape Operation Chain ({len(shape_nodes)} ops)",
                category=PatternCategory.ATTACK_SURFACE,
                risk=PatternRisk.MEDIUM,
                nodes_involved=shape_nodes,
                description=f"Model contains {len(shape_nodes)} shape/view operations "
                           f"that reorganize data without semantic validation. These can "
                           f"be exploited for prompt injection and carrier attacks.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    Shape operations trust input dimensions and reorganize data blindly.
                    Attackers can craft inputs that appear benign in one shape but become
                    adversarial after transformation. In multimodal models, shape ops
                    can enable cross-modal payload injection.
                """,
                research_notes="""
                    Trace how data flows through shape operations. Consider whether
                    an attacker could craft input that encodes adversarial content
                    in a way that survives or is revealed by reshaping.
                """,
                recommendations=[
                    "Validate tensor dimensions match expected values",
                    "Add semantic validation after shape operations",
                    "Consider input sanitization before shape transformations"
                ]
            ))
        
        return patterns
    
    def _detect_saturating_activations(self) -> List[StructuralPattern]:
        """
        Detect saturating activations that create gradient masking (false security).
        """
        patterns = []
        
        saturating_nodes = [nid for nid, n in self.nodes.items() 
                          if n.get("op_type") in self.SATURATING_ACTIVATIONS]
        
        if saturating_nodes:
            attack_info = self.ATTACK_SURFACE_MAPPING["saturating_activations"]
            
            patterns.append(StructuralPattern(
                id=f"SATURATING-ACT-VULN",
                name=f"Saturating Activations ({len(saturating_nodes)} layers)",
                category=PatternCategory.GRADIENT_FLOW,
                risk=PatternRisk.MEDIUM,
                nodes_involved=saturating_nodes,
                description=f"Model contains {len(saturating_nodes)} saturating activation "
                           f"functions (sigmoid, tanh, etc.) that cause gradient masking. "
                           f"WARNING: Gradient masking is NOT a defense.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    Saturating activations cause vanishing gradients at extremes,
                    which may make gradient-based attacks appear less effective.
                    However, this is GRADIENT MASKING - a false sense of security.
                    Black-box attacks, transfer attacks, and boundary attacks remain
                    fully effective.
                """,
                research_notes="""
                    DO NOT rely on saturating activations for robustness. The apparent
                    resistance to gradient attacks is due to obfuscated gradients, not
                    true robustness. Always evaluate with transfer attacks and black-box
                    methods.
                """,
                recommendations=[
                    "Do NOT rely on gradient masking for security",
                    "Evaluate with black-box and transfer attacks",
                    "Consider replacing with non-saturating activations (ReLU, GELU)"
                ]
            ))
        
        return patterns
    
    def _detect_multimodal_fusion(self) -> List[StructuralPattern]:
        """
        Detect multimodal fusion points vulnerable to cross-modal jailbreaks.
        """
        patterns = []
        
        # Look for fusion operations with diverse input sources
        for node_id, node in self.nodes.items():
            if node.get("op_type") not in self.MULTIMODAL_OPS:
                continue
            
            # Check for multiple inputs from different processing paths
            predecessors = self.reverse_adjacency.get(node_id, [])
            if len(predecessors) < 2:
                continue
            
            # Heuristic: look for "cross" or "fusion" in name, or diverse predecessor op types
            pred_ops = set(self.nodes.get(p, {}).get("op_type", "") for p in predecessors)
            
            # Check for modality mixing indicators
            is_multimodal = (
                "cross" in node_id.lower() or 
                "fusion" in node_id.lower() or
                "multimodal" in node_id.lower() or
                node.get("op_type") in {"CrossAttention", "MultimodalFusion", "FeatureFusion"} or
                len(pred_ops) >= 2  # Different op types suggest different modality processing
            )
            
            if is_multimodal and len(predecessors) >= 2:
                attack_info = self.ATTACK_SURFACE_MAPPING["multimodal_fusion"]
                
                patterns.append(StructuralPattern(
                    id=f"MULTIMODAL-{node_id}",
                    name=f"Multimodal Fusion Point at {node_id}",
                    category=PatternCategory.PERTURBATION_FUSION,
                    risk=PatternRisk.HIGH,
                    nodes_involved=[node_id] + predecessors,
                    description=f"Potential multimodal fusion at '{node_id}' combining "
                               f"{len(predecessors)} input streams. Cross-modal fusion "
                               f"points are vulnerable to jailbreaks where adversarial "
                               f"content in one modality affects processing of another.",
                    attack_implications=f"""
                        Applicable attacks: {', '.join(attack_info['attacks'])}
                        
                        {attack_info['exploitation']}
                        
                        In vision-language models, adversarial images can override text
                        instructions. In audio-visual models, one modality can mask another.
                        Cross-modal attacks are often more effective than single-modality
                        attacks because defenses may not consider cross-modal interactions.
                    """,
                    research_notes="""
                        Analyze how information from each modality influences the fused
                        representation. Look for dominance patterns where one modality
                        can override another. Test with adversarial content in each
                        modality separately and combined.
                    """,
                    recommendations=[
                        "Implement per-modality input validation",
                        "Add cross-modal consistency checks",
                        "Evaluate with cross-modal adversarial examples",
                        "Consider modality-specific adversarial training"
                    ]
                ))
        
        return patterns
    
    def _detect_shadowlogic_susceptibility(self) -> List[StructuralPattern]:
        """
        Detect model characteristics that make it susceptible to ShadowLogic backdoor attacks.
        
        ShadowLogic attacks embed malicious subnets in unused model capacity.
        High susceptibility indicators:
        - Large layers with potential unused capacity
        - Conditional operations that could implement triggers
        - Deep networks with many parameters
        - Lack of weight monitoring/integrity checks
        """
        patterns = []
        
        # Check for conditional operations (direct trigger implementation)
        conditional_nodes = [nid for nid, n in self.nodes.items() 
                           if n.get("op_type") in self.SHADOWLOGIC_INDICATORS]
        
        if conditional_nodes:
            attack_info = self.ATTACK_SURFACE_MAPPING["shadowlogic_susceptible"]
            patterns.append(StructuralPattern(
                id="SHADOWLOGIC-COND-OPS",
                name=f"ShadowLogic Risk: Conditional Operations ({len(conditional_nodes)} nodes)",
                category=PatternCategory.ATTACK_SURFACE,
                risk=PatternRisk.CRITICAL,
                nodes_involved=conditional_nodes,
                description=f"Model contains {len(conditional_nodes)} conditional operations "
                           f"({', '.join(set(self.nodes[n].get('op_type', '?') for n in conditional_nodes))}). "
                           f"These can directly implement trigger-based backdoor logic.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    Conditional operations (Where, If, Equal, etc.) can check for trigger
                    patterns and route to malicious subnets. This is the PRIMARY mechanism
                    of ShadowLogic attacks - the trigger detection is implemented directly
                    in the computational graph.
                """,
                research_notes="""
                    CRITICAL: Audit every conditional operation. Verify the condition
                    corresponds to legitimate model logic. Suspicious patterns:
                    - Conditions checking specific input values/patterns
                    - Conditions with constant comparisons
                    - Conditions that route to rarely-used paths
                """,
                recommendations=[
                    "Audit all conditional operations for legitimate purpose",
                    "Implement runtime monitoring of conditional branch frequencies",
                    "Consider replacing conditionals with soft gating where possible",
                    "Use Neural Cleanse to scan for potential triggers"
                ]
            ))
        
        # Check for large conv/fc layers with potential unused capacity
        large_layers = []
        for nid, n in self.nodes.items():
            op_type = n.get("op_type", "")
            if op_type in self.LINEAR_OPS:
                # Estimate parameter count (would need shape info for accuracy)
                large_layers.append(nid)
        
        if len(large_layers) > 10:  # Many large layers = high capacity for hiding
            attack_info = self.ATTACK_SURFACE_MAPPING["shadowlogic_susceptible"]
            patterns.append(StructuralPattern(
                id="SHADOWLOGIC-CAPACITY",
                name=f"ShadowLogic Risk: High Parameter Capacity ({len(large_layers)} linear layers)",
                category=PatternCategory.ATTACK_SURFACE,
                risk=PatternRisk.HIGH,
                nodes_involved=large_layers[:20],  # Limit to first 20
                description=f"Model has {len(large_layers)} linear layers (Conv/FC/MatMul) "
                           f"providing substantial capacity for hidden malicious subnets. "
                           f"ShadowLogic attacks exploit unused parameters to embed backdoors.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    Large models have significant "dark capacity" - parameters that don't
                    contribute much to normal operation but could encode backdoor behavior.
                    Adversary can embed trigger-detection and malicious-output subnets in
                    this unused capacity without affecting clean accuracy.
                """,
                research_notes="""
                    Assess parameter utilization across layers. Look for:
                    - Layers with many near-zero weights (potential hiding space)
                    - Redundant channels that could be repurposed
                    - Large final layers with more capacity than needed
                    
                    Consider weight pruning to remove potential backdoor capacity.
                """,
                recommendations=[
                    "Analyze weight utilization across layers",
                    "Apply pruning to remove unused capacity",
                    "Implement weight integrity monitoring",
                    "Use fine-pruning defense to remove potential backdoors"
                ]
            ))
        
        # Check for deep sequential structure (easier to hide subnets)
        max_depth = self._compute_max_depth()
        if max_depth > 50:
            patterns.append(StructuralPattern(
                id="SHADOWLOGIC-DEPTH",
                name=f"ShadowLogic Risk: Deep Architecture (depth={max_depth})",
                category=PatternCategory.ATTACK_SURFACE,
                risk=PatternRisk.MEDIUM,
                nodes_involved=[],
                description=f"Deep architecture with {max_depth} layers provides many "
                           f"opportunities for hiding malicious subnets. Deeper models "
                           f"are harder to audit exhaustively.",
                attack_implications="""
                    Deep networks have more layers where backdoors can hide. Each layer
                    is a potential embedding point. The complexity makes manual auditing
                    impractical, and automated tools may miss subtle backdoors.
                """,
                research_notes="""
                    Focus security auditing on:
                    - Early layers (often less scrutinized)
                    - Branches and skip connections
                    - Layers with unusual weight patterns
                """,
                recommendations=[
                    "Use automated backdoor scanning tools",
                    "Implement layer-wise integrity checks",
                    "Consider model distillation to simpler architecture"
                ]
            ))
        
        return patterns
    
    def _detect_valid_conv_boundary(self) -> List[StructuralPattern]:
        """
        Detect "valid" convolutions (pads=0) that create boundary sensitivity.
        
        Valid convolutions see fewer contexts near edges, making the model
        sensitive to boundary manipulations and edge attacks.
        """
        patterns = []
        
        valid_conv_nodes = []
        for nid, n in self.nodes.items():
            if n.get("op_type") != "Conv":
                continue
            
            attrs = n.get("attributes", {})
            pads = attrs.get("pads", None)
            auto_pad = attrs.get("auto_pad", "NOTSET")
            
            # Check for valid convolution (no padding)
            is_valid = False
            if auto_pad == "VALID":
                is_valid = True
            elif pads is not None:
                if isinstance(pads, list) and all(p == 0 for p in pads):
                    is_valid = True
                elif pads == 0:
                    is_valid = True
            
            if is_valid:
                valid_conv_nodes.append(nid)
        
        if valid_conv_nodes:
            attack_info = self.ATTACK_SURFACE_MAPPING["valid_conv_boundary"]
            
            # Check if these are early in the network
            depths = self._compute_node_depths()
            max_depth = max(depths.values()) if depths else 1
            early_valid = [
                n for n in valid_conv_nodes
                if self.topology_config.is_early(depths.get(n, 0), max_depth)
            ]
            
            risk = PatternRisk.HIGH if early_valid else PatternRisk.MEDIUM
            
            patterns.append(StructuralPattern(
                id="VALID-CONV-BOUNDARY",
                name=f"VALID_CONV_BOUNDARY — Valid Conv Edge Sensitivity ({len(valid_conv_nodes)} nodes)",
                category=PatternCategory.FEATURE_EXTRACTION,
                risk=risk,
                nodes_involved=valid_conv_nodes,
                description=f"Model has {len(valid_conv_nodes)} convolutions with pads=0 (valid mode). "
                           f"{'Including ' + str(len(early_valid)) + ' in early layers. ' if early_valid else ''}"
                           f"Valid convolutions reduce spatial dimensions and see fewer contexts "
                           f"near image boundaries, creating edge sensitivity.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    Specific vulnerability patterns:
                    - Corners: Smallest receptive field, maximum sensitivity
                    - Edges: Asymmetric context, predictable blind spots
                    - Frames: Perturbations in border regions have outsized impact
                    
                    The model effectively has a "blind spot" at boundaries where
                    context is limited. Adversarial perturbations placed here can
                    manipulate features without full contextual correction.
                """,
                research_notes="""
                    Map the effective receptive field at image boundaries vs center.
                    The difference reveals the attack surface. Consider:
                    - How much context is lost at each boundary layer?
                    - Do perturbations at edges propagate differently?
                    - Can frame/border attacks exploit this systematically?
                """,
                recommendations=[
                    "Consider same/reflect padding to provide edge context",
                    "Implement boundary-aware adversarial training",
                    "Add edge-detection preprocessing to flag boundary anomalies",
                    "Test specifically with edge/corner perturbations"
                ]
            ))
        
        return patterns
    
    def _detect_early_linear_no_norm(self) -> List[StructuralPattern]:
        """
        Detect long early linear chains without normalization.
        
        Early networks with sequential Conv/Linear without BatchNorm/LayerNorm
        have very stable, informative gradients - making them attack-friendly.
        ReLU is piecewise-linear, so gradients don't vanish in these chains.
        """
        patterns = []
        
        depths = self._compute_node_depths()
        max_depth = max(depths.values()) if depths else 1
        early_threshold = self._early_depth_threshold(max_depth)
        
        # Find early linear ops
        early_linear = []
        for nid, n in self.nodes.items():
            if depths.get(nid, 0) > early_threshold:
                continue
            if n.get("op_type") in self.LINEAR_OPS:
                early_linear.append(nid)
        
        if not early_linear:
            return patterns
        
        # Check for chains without normalization
        # Build early subgraph
        early_nodes = {nid for nid, d in depths.items() if d <= early_threshold}
        
        # Check if normalization exists in early layers
        early_norm = [nid for nid in early_nodes 
                     if self.nodes.get(nid, {}).get("op_type") in self.NORM_OPS]
        
        # Look for linear chains without interleaved normalization
        linear_chains_no_norm = []
        visited = set()
        
        for start in early_linear:
            if start in visited:
                continue
            
            chain = [start]
            current = start
            has_norm_between = False
            
            while True:
                successors = [s for s in self.adjacency.get(current, []) if s in early_nodes]
                
                # Check for normalization in successors
                for s in successors:
                    if self.nodes.get(s, {}).get("op_type") in self.NORM_OPS:
                        has_norm_between = True
                        break
                
                if has_norm_between:
                    break
                
                # Find next linear op
                linear_successors = [s for s in successors 
                                    if self.nodes.get(s, {}).get("op_type") in self.LINEAR_OPS]
                
                if not linear_successors:
                    # Check one more hop (through activation)
                    for s in successors:
                        next_level = self.adjacency.get(s, [])
                        linear_next = [n for n in next_level 
                                      if self.nodes.get(n, {}).get("op_type") in self.LINEAR_OPS
                                      and n in early_nodes]
                        if linear_next:
                            linear_successors = linear_next
                            break
                
                if not linear_successors:
                    break
                
                current = linear_successors[0]
                chain.append(current)
            
            visited.update(chain)
            
            if len(chain) >= 2 and not has_norm_between:
                linear_chains_no_norm.append(chain)
        
        if linear_chains_no_norm:
            attack_info = self.ATTACK_SURFACE_MAPPING["early_linear_no_norm"]
            all_nodes = [n for chain in linear_chains_no_norm for n in chain]
            longest_chain = max(len(c) for c in linear_chains_no_norm)
            
            patterns.append(StructuralPattern(
                id="EARLY-LINEAR-NO-NORM",
                name=f"Attack-Friendly Early Layers: {len(linear_chains_no_norm)} linear chains without normalization",
                category=PatternCategory.GRADIENT_FLOW,
                risk=PatternRisk.HIGH,
                nodes_involved=all_nodes,
                description=f"Found {len(linear_chains_no_norm)} chains of linear operations "
                           f"(longest: {longest_chain} ops) in early layers without interleaved "
                           f"normalization. This creates highly stable, informative gradients "
                           f"that make gradient-based attacks very effective.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    Why this matters:
                    - ReLU between linear ops is still piecewise-linear
                    - Gradients flow stably without rescaling from normalization
                    - Early layer gradients directly inform input perturbations
                    - No gradient noise/masking from batch statistics
                    
                    This is essentially a "clean" gradient path from loss to input,
                    which is exactly what gradient-based attacks need.
                """,
                research_notes="""
                    These early layers are the most important for gradient-based attacks.
                    The attacker gets high-quality gradient information about how input
                    changes affect the loss. Consider:
                    - Computing gradient magnitudes through these chains
                    - Testing attack success rate vs models with early normalization
                    - Measuring gradient variance (lower = more exploitable)
                """,
                recommendations=[
                    "Add BatchNorm/LayerNorm between early linear layers",
                    "Consider input normalization/preprocessing",
                    "Apply spectral normalization to early layers",
                    "Implement gradient regularization during training"
                ]
            ))
        
        return patterns
    
    def _detect_relu_no_lipschitz(self) -> List[StructuralPattern]:
        """
        Detect ReLU activations without Lipschitz constraints.
        
        ReLU creates sharp decision boundaries at zero. Without spectral
        normalization or other Lipschitz constraints, these boundaries
        are highly exploitable.
        """
        patterns = []
        
        relu_nodes = [nid for nid, n in self.nodes.items() 
                     if n.get("op_type") in self.RELU_FAMILY]
        
        if not relu_nodes:
            return patterns
        
        # Check if spectral normalization or other constraints are present
        # (Would need weight analysis for accurate detection)
        # For now, check if model has any normalization near ReLUs
        
        relu_with_no_constraint = []
        for relu_nid in relu_nodes:
            predecessors = self.reverse_adjacency.get(relu_nid, [])
            successors = self.adjacency.get(relu_nid, [])
            
            # Check for normalization in neighborhood
            neighborhood = predecessors + successors
            has_norm = any(self.nodes.get(n, {}).get("op_type") in self.NORM_OPS 
                          for n in neighborhood)
            
            # Check for preceding linear op (Conv/FC before ReLU is common pattern)
            linear_before = any(self.nodes.get(p, {}).get("op_type") in self.LINEAR_OPS 
                               for p in predecessors)
            
            if linear_before and not has_norm:
                relu_with_no_constraint.append(relu_nid)
        
        if relu_with_no_constraint:
            attack_info = self.ATTACK_SURFACE_MAPPING["relu_no_lipschitz"]
            
            patterns.append(StructuralPattern(
                id="RELU-NO-LIPSCHITZ",
                name=f"Unbounded ReLU Boundaries ({len(relu_with_no_constraint)} nodes)",
                category=PatternCategory.GRADIENT_FLOW,
                risk=PatternRisk.MEDIUM,
                nodes_involved=relu_with_no_constraint[:20],  # Limit output
                description=f"Found {len(relu_with_no_constraint)} ReLU activations after "
                           f"linear layers without apparent Lipschitz constraints (no adjacent "
                           f"normalization). ReLU's piecewise-linear nature creates sharp "
                           f"decision boundaries that are highly exploitable.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    ReLU boundary exploitation:
                    - At x=0, the derivative jumps from 0 to 1
                    - Small perturbations can flip activations across this boundary
                    - Without spectral normalization, linear layer weights can have
                      arbitrarily large spectral norms
                    - This means small input changes -> potentially large output changes
                    
                    The combination of ReLU + unconstrained linear ops creates a
                    highly non-Lipschitz function with exploitable discontinuities.
                """,
                research_notes="""
                    For each ReLU, consider:
                    - What's the distribution of pre-activation values?
                    - How many activations are near the zero boundary?
                    - What's the spectral norm of the preceding linear layer?
                    
                    Regions where many activations are near zero are most vulnerable
                    to adversarial perturbations.
                """,
                recommendations=[
                    "Apply spectral normalization to linear layers before ReLU",
                    "Consider Lipschitz-bounded alternatives (GroupSort, MaxMin)",
                    "Add batch/layer normalization before ReLU",
                    "Train with Lipschitz regularization"
                ]
            ))
        
        return patterns
    
    def _detect_no_gradient_regularization(self) -> List[StructuralPattern]:
        """
        Detect lack of gradient regularization mechanisms.
        
        Networks without dropout, noise layers, or gradient penalties
        have stable, informative gradients that make attacks easier.
        """
        patterns = []
        
        # Check for gradient-disrupting mechanisms
        dropout_nodes = [nid for nid, n in self.nodes.items() 
                        if n.get("op_type") in {"Dropout", "DropPath", "DropBlock"}]
        
        noise_nodes = [nid for nid, n in self.nodes.items()
                      if "noise" in nid.lower() or n.get("op_type") in {"GaussianNoise", "Noise"}]
        
        # Stochastic depth / drop path
        stochastic_nodes = [nid for nid, n in self.nodes.items()
                          if "drop" in nid.lower() or "stochastic" in nid.lower()]
        
        total_regularization = len(dropout_nodes) + len(noise_nodes) + len(stochastic_nodes)
        
        if total_regularization == 0 and len(self.nodes) > 20:
            attack_info = self.ATTACK_SURFACE_MAPPING["no_gradient_regularization"]
            
            patterns.append(StructuralPattern(
                id="NO-GRADIENT-REG",
                name="No Gradient Regularization Detected",
                category=PatternCategory.GRADIENT_FLOW,
                risk=PatternRisk.MEDIUM,
                nodes_involved=[],
                description=f"Model with {len(self.nodes)} nodes has no apparent gradient "
                           f"regularization (no Dropout, DropPath, or noise layers detected). "
                           f"This means gradients remain stable and informative, making "
                           f"gradient-based attacks highly effective.",
                attack_implications=f"""
                    Applicable attacks: {', '.join(attack_info['attacks'])}
                    
                    {attack_info['exploitation']}
                    
                    Without gradient regularization:
                    - Loss gradients w.r.t. input are clean and reliable
                    - FGSM/PGD can compute precise perturbation directions
                    - Optimization-based attacks (C&W) converge quickly
                    - Adversarial examples transfer more reliably
                    
                    Gradient regularization (dropout, noise) during inference would
                    add variance to gradients, making attacks less reliable.
                """,
                research_notes="""
                    This is a fundamental tradeoff:
                    - Clean gradients = better training BUT easier attacks
                    - Noisy gradients = harder training BUT some attack resistance
                    
                    Note: Gradient masking is NOT a defense, but gradient variance
                    can increase the cost of successful attacks.
                """,
                recommendations=[
                    "Consider enabling dropout during adversarial evaluation",
                    "Implement stochastic inference (Monte Carlo dropout)",
                    "Add input noise layer for inference-time defense",
                    "Evaluate adversarial training to disrupt gradient optimization"
                ]
            ))
        elif total_regularization > 0:
            # Note the presence of regularization as a positive indicator
            patterns.append(StructuralPattern(
                id="GRADIENT-REG-PRESENT",
                name=f"Gradient Regularization Present ({total_regularization} mechanisms)",
                category=PatternCategory.ROBUSTNESS,
                risk=PatternRisk.POSITIVE,
                nodes_involved=dropout_nodes + noise_nodes + stochastic_nodes,
                description=f"Model includes {total_regularization} gradient regularization "
                           f"mechanisms (Dropout: {len(dropout_nodes)}, Noise: {len(noise_nodes)}, "
                           f"Stochastic: {len(stochastic_nodes)}). These add variance to gradients, "
                           f"potentially increasing attack cost.",
                attack_implications="""
                    Gradient regularization during inference can increase the number
                    of attack iterations needed and reduce attack transferability.
                    However, this is NOT a complete defense - determined attackers
                    can use expectation-over-transformation (EOT) to overcome it.
                """,
                research_notes="""
                    Verify these mechanisms are active during inference, not just training.
                    Many frameworks disable dropout at inference time by default.
                """,
                recommendations=[]
            ))
        
        return patterns
    
    # =========================================================================
    # ROBUSTNESS INDICATOR DETECTION
    # =========================================================================
    
    def _detect_early_downsampling(self) -> List[StructuralPattern]:
        """
        Detect early spatial downsampling (reduces attack surface).
        
        Early downsampling reduces the number of pixels an attacker
        can perturb, making attacks more difficult.
        """
        patterns = []
        
        # Find pooling/strided conv in early layers (first 1/3 of depth)
        max_depth = self._compute_max_depth()
        early_threshold = self._early_depth_threshold(max_depth)
        
        depths = self._compute_node_depths()
        
        for node_id, node in self.nodes.items():
            if depths.get(node_id, 0) > early_threshold:
                continue
            
            op_type = node.get("op_type", "")
            is_downsampling = False
            
            if op_type in self.POOLING_OPS:
                is_downsampling = True
            elif op_type in {"Conv", "ConvTranspose"}:
                strides = node.get("attributes", {}).get("strides", [1, 1])
                if any(s > 1 for s in strides):
                    is_downsampling = True
            
            if is_downsampling:
                patterns.append(StructuralPattern(
                    id=f"EARLY-DOWN-{node_id}",
                    name=f"Early Downsampling at {node_id}",
                    category=PatternCategory.ROBUSTNESS,
                    risk=PatternRisk.POSITIVE,
                    nodes_involved=[node_id],
                    description=f"Spatial downsampling in early layers (depth {depths.get(node_id, 0)}/{max_depth}). "
                               f"This reduces the input dimensionality early, limiting the "
                               f"attacker's perturbation space.",
                    attack_implications="""
                        - Reduces number of pixels available for perturbation
                        - Forces attacker to work with lower-resolution representations
                        - Generally improves robustness (with some accuracy trade-off)
                    """,
                    research_notes="""
                        Early downsampling is a robustness-positive pattern. It limits
                        the attack surface but may reduce model capacity for fine details.
                        Consider the trade-off between robustness and task performance.
                    """,
                    recommendations=[]  # Positive pattern, no change needed
                ))
        
        return patterns
    
    def _compute_node_depths(self) -> Dict[str, int]:
        """Compute depth of each node from inputs."""
        depths = {}
        
        def get_depth(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]
            
            predecessors = self.reverse_adjacency.get(node_id, [])
            if not predecessors:
                depths[node_id] = 0
            else:
                depths[node_id] = 1 + max(get_depth(p) for p in predecessors)
            
            return depths[node_id]
        
        for node_id in self.nodes:
            get_depth(node_id)
        
        return depths
    
    def _detect_average_pooling(self) -> List[StructuralPattern]:
        """
        Detect AveragePool usage (more robust than MaxPool).
        
        Average pooling distributes gradients and smooths perturbations,
        providing natural robustness compared to max pooling.
        """
        patterns = []
        
        for node_id, node in self.nodes.items():
            if node.get("op_type") not in {"AveragePool", "GlobalAveragePool", "AdaptiveAvgPool2d"}:
                continue
            
            patterns.append(StructuralPattern(
                id=f"AVG-POOL-{node_id}",
                name=f"Average Pooling at {node_id}",
                category=PatternCategory.ROBUSTNESS,
                risk=PatternRisk.POSITIVE,
                nodes_involved=[node_id],
                description=f"Average pooling distributes gradients uniformly and smooths "
                           f"perturbations by averaging. This provides natural robustness "
                           f"compared to max pooling.",
                attack_implications="""
                    - Perturbations must affect the entire pooling region to have impact
                    - No spike amplification like MaxPool
                    - Gradients flow equally to all pooling inputs
                """,
                research_notes="""
                    AveragePool is generally more robust than MaxPool. Consider if this
                    is a good pattern to replicate elsewhere in the model.
                """,
                recommendations=[]  # Positive pattern
            ))
        
        return patterns
    
    def _detect_bottleneck_convs(self) -> List[StructuralPattern]:
        """
        Detect bottleneck 1x1 convolutions (reduce feature dimensions).
        
        1x1 convolutions that reduce channel count create information
        bottlenecks that can limit perturbation propagation.
        """
        patterns = []
        
        for node_id, node in self.nodes.items():
            if node.get("op_type") != "Conv":
                continue
            
            attrs = node.get("attributes", {})
            kernel_shape = attrs.get("kernel_shape", [3, 3])
            
            # Check for 1x1 convolution
            if kernel_shape == [1, 1] or kernel_shape == 1:
                # Check if it reduces channels (need shape info)
                # For now, just flag all 1x1 convs
                patterns.append(StructuralPattern(
                    id=f"BOTTLENECK-{node_id}",
                    name=f"Bottleneck Conv at {node_id}",
                    category=PatternCategory.ROBUSTNESS,
                    risk=PatternRisk.POSITIVE,
                    nodes_involved=[node_id],
                    description=f"1x1 convolution that performs channel-wise projection. "
                               f"If this reduces dimensionality, it creates an information "
                               f"bottleneck that limits perturbation propagation.",
                    attack_implications="""
                        - Dimensionality reduction discards some perturbation information
                        - Forces perturbations to align with principal directions
                        - Can act as a natural defense if reducing to important features only
                    """,
                    research_notes="""
                        Check if this 1x1 conv reduces or expands channels. Reduction
                        is robustness-positive, expansion may increase attack surface.
                    """,
                    recommendations=[]  # Generally positive
                ))
        
        return patterns
    
    def _detect_reduced_early_depth(self) -> List[StructuralPattern]:
        """
        Detect if early layers have reduced complexity.
        
        Models that process inputs with shallow early stages before
        deep processing tend to be more robust.
        """
        patterns = []
        
        # Analyze depth distribution
        depths = self._compute_node_depths()
        max_depth = max(depths.values()) if depths else 0
        
        if max_depth < 5:
            return patterns  # Too shallow to analyze
        
        # Count operations at each depth level
        depth_counts = defaultdict(int)
        for node_id, depth in depths.items():
            depth_counts[depth] += 1
        
        early_count = sum(
            depth_counts[d]
            for d in range(self._early_depth_threshold(max_depth) + 1)
        )
        late_count = sum(
            depth_counts[d]
            for d in range(self._late_depth_threshold(max_depth), max_depth + 1)
        )
        
        if early_count < late_count * 0.5:  # Early layers are less than half of late
            patterns.append(StructuralPattern(
                id="REDUCED-EARLY-DEPTH",
                name="Reduced Early Layer Complexity",
                category=PatternCategory.ROBUSTNESS,
                risk=PatternRisk.POSITIVE,
                nodes_involved=[],
                description=f"Model has fewer operations in early layers ({early_count}) "
                           f"compared to late layers ({late_count}). This pattern processes "
                           f"inputs with a shallow stage before deep processing, which can "
                           f"improve robustness.",
                attack_implications="""
                    - Early shallow processing reduces initial perturbation amplification
                    - Deep features are more abstract and harder to attack directly
                    - Input-space attacks have limited immediate impact
                """,
                research_notes="""
                    This architecture processes inputs conservatively before complex
                    transformations. Consider how perturbations at the input propagate
                    through the shallow early stage.
                """,
                recommendations=[]
            ))
        
        return patterns
    
    # =========================================================================
    # ATTACK SURFACE MAPPING
    # =========================================================================
    
    def _map_attack_surfaces(self) -> List[AttackSurfaceMapping]:
        """Map model components to attack classes based on vulnerability taxonomy."""
        surfaces = []
        
        # Input tensors -> adversarial image injection
        input_nodes = [nid for nid in self.nodes if not self.reverse_adjacency.get(nid)]
        if input_nodes:
            surfaces.append(AttackSurfaceMapping(
                component="Input Tensor X",
                node_ids=input_nodes,
                attack_class="Adversarial Image/Data Injection",
                attack_techniques=["FGSM", "PGD", "C&W", "AutoAttack", "Square Attack", "DeepFool"],
                exploitation_notes="""
                    The input tensor is the primary attack surface. All gradient-based
                    and query-based adversarial attacks target this component. Perturbations
                    here propagate through the entire network.
                """
            ))
        
        # Linear chains -> FGSM, PGD, CW, AutoAttack
        linear_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") in self.LINEAR_OPS]
        if linear_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["linear_chains"]
            surfaces.append(AttackSurfaceMapping(
                component="Linear Chains (Conv/MatMul/Gemm)",
                node_ids=linear_nodes,
                attack_class="Gradient-Based Optimization Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Concat / Add -> Multi-scale PGD, transfer, universal
        fusion_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") in self.FUSION_OPS]
        if fusion_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["concat_add"]
            surfaces.append(AttackSurfaceMapping(
                component="Concat/Add Fusion Points",
                node_ids=fusion_nodes,
                attack_class="Multi-Scale & Transfer Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Residuals -> PGD, momentum, CW
        add_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") == "Add"]
        if add_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["residuals"]
            surfaces.append(AttackSurfaceMapping(
                component="Residual Connections",
                node_ids=add_nodes,
                attack_class="Deep Network Gradient Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # MaxPool -> One-pixel, sparse attacks
        maxpool_nodes = [nid for nid, n in self.nodes.items() 
                        if n.get("op_type") in {"MaxPool", "GlobalMaxPool", "AdaptiveMaxPool2d"}]
        if maxpool_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["maxpool"]
            surfaces.append(AttackSurfaceMapping(
                component="Max Pooling Layers",
                node_ids=maxpool_nodes,
                attack_class="Sparse & One-Pixel Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Early stride -> Fourier, frequency attacks
        depths = self._compute_node_depths()
        max_depth = max(depths.values()) if depths else 0
        early_threshold = self._early_depth_threshold(max_depth)
        
        early_stride_nodes = []
        for nid, n in self.nodes.items():
            if n.get("op_type") == "Conv":
                strides = n.get("attributes", {}).get("strides", [1, 1])
                if any(s > 1 for s in strides) and depths.get(nid, 0) <= early_threshold:
                    early_stride_nodes.append(nid)
        
        if early_stride_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["early_stride"]
            surfaces.append(AttackSurfaceMapping(
                component="Early Strided Convolutions",
                node_ids=early_stride_nodes,
                attack_class="Frequency & Fourier Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # BatchNorm -> Distribution shift attacks
        bn_nodes = [nid for nid, n in self.nodes.items() 
                   if "batch" in n.get("op_type", "").lower() and n.get("op_type") in self.NORM_OPS]
        if bn_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["batchnorm"]
            surfaces.append(AttackSurfaceMapping(
                component="BatchNorm Layers",
                node_ids=bn_nodes,
                attack_class="Distribution Shift Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Attention blocks -> Attention hijacking
        attention_nodes = [nid for nid, n in self.nodes.items() 
                         if n.get("op_type") in self.ATTENTION_OPS 
                         or "attention" in nid.lower()]
        if attention_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["attention"]
            surfaces.append(AttackSurfaceMapping(
                component="Attention Blocks",
                node_ids=attention_nodes,
                attack_class="Attention Hijacking",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Global pooling -> Feature-space attacks
        global_pool_nodes = [nid for nid, n in self.nodes.items() 
                           if n.get("op_type") in self.GLOBAL_POOLING_OPS]
        if global_pool_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["global_pooling"]
            surfaces.append(AttackSurfaceMapping(
                component="Global Pooling Layers",
                node_ids=global_pool_nodes,
                attack_class="Feature-Space Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # FC layers -> CW, margin attacks
        output_adjacent = [nid for nid in self.nodes 
                         if not self.adjacency.get(nid) or len(self.adjacency.get(nid, [])) <= 1]
        fc_output = [nid for nid in output_adjacent 
                    if self.nodes.get(nid, {}).get("op_type") in self.FC_OPS]
        if fc_output:
            vuln = self.ATTACK_SURFACE_MAPPING["fc_layers"]
            surfaces.append(AttackSurfaceMapping(
                component="Final FC Layers",
                node_ids=fc_output,
                attack_class="Margin & Logit Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Shape ops -> Prompt injection, carriers
        shape_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") in self.SHAPE_OPS]
        if shape_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["shape_ops"]
            surfaces.append(AttackSurfaceMapping(
                component="Shape/View Operations",
                node_ids=shape_nodes,
                attack_class="Prompt Injection & Carrier Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Saturating activations -> Boundary attacks
        saturating_nodes = [nid for nid, n in self.nodes.items() 
                          if n.get("op_type") in self.SATURATING_ACTIVATIONS]
        if saturating_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["saturating_activations"]
            surfaces.append(AttackSurfaceMapping(
                component="Saturating Activations (Sigmoid/Tanh)",
                node_ids=saturating_nodes,
                attack_class="Boundary & Black-Box Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Multimodal fusion -> Cross-modal jailbreaks
        multimodal_nodes = []
        for nid, n in self.nodes.items():
            if n.get("op_type") in {"CrossAttention", "MultimodalFusion", "FeatureFusion"}:
                multimodal_nodes.append(nid)
            elif "cross" in nid.lower() or "fusion" in nid.lower() or "multimodal" in nid.lower():
                multimodal_nodes.append(nid)
        
        if multimodal_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["multimodal_fusion"]
            surfaces.append(AttackSurfaceMapping(
                component="Multimodal Fusion Points",
                node_ids=multimodal_nodes,
                attack_class="Cross-Modal Jailbreaks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Softmax -> model extraction
        softmax_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") == "Softmax"]
        if softmax_nodes:
            surfaces.append(AttackSurfaceMapping(
                component="Softmax Output",
                node_ids=softmax_nodes,
                attack_class="Model Extraction & Privacy Attacks",
                attack_techniques=["Knockoff Nets", "Membership inference", "Confidence score analysis", 
                                  "Model inversion"],
                exploitation_notes="""
                    Softmax confidence scores leak significant information about model
                    internals. Full probability distributions enable efficient model
                    extraction. High confidence on inputs indicates training data membership.
                """
            ))
        
        # ShadowLogic susceptibility -> conditional ops
        cond_nodes = [nid for nid, n in self.nodes.items() 
                     if n.get("op_type") in self.SHADOWLOGIC_INDICATORS]
        if cond_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["shadowlogic_susceptible"]
            surfaces.append(AttackSurfaceMapping(
                component="Conditional Operations (ShadowLogic Target)",
                node_ids=cond_nodes,
                attack_class="ShadowLogic & Backdoor Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # Valid convolutions -> boundary attacks
        valid_conv = []
        for nid, n in self.nodes.items():
            if n.get("op_type") == "Conv":
                attrs = n.get("attributes", {})
                pads = attrs.get("pads", None)
                auto_pad = attrs.get("auto_pad", "NOTSET")
                if auto_pad == "VALID" or (pads and all(p == 0 for p in (pads if isinstance(pads, list) else [pads]))):
                    valid_conv.append(nid)
        
        if valid_conv:
            vuln = self.ATTACK_SURFACE_MAPPING["valid_conv_boundary"]
            surfaces.append(AttackSurfaceMapping(
                component="Valid Convolutions (Boundary Vulnerable)",
                node_ids=valid_conv,
                attack_class="Boundary & Edge Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        # ReLU without constraints -> decision boundary attacks
        relu_nodes = [nid for nid, n in self.nodes.items() if n.get("op_type") in self.RELU_FAMILY]
        if relu_nodes:
            vuln = self.ATTACK_SURFACE_MAPPING["relu_no_lipschitz"]
            surfaces.append(AttackSurfaceMapping(
                component="ReLU Activations (Boundary Exploitable)",
                node_ids=relu_nodes[:20],
                attack_class="Decision Boundary & Gradient Attacks",
                attack_techniques=vuln["attacks"],
                exploitation_notes=vuln["description"] + " " + vuln["exploitation"]
            ))
        
        return surfaces
    
    # =========================================================================
    # RESEARCH WORKFLOW OUTPUTS
    # =========================================================================
    
    def _find_gradient_bottlenecks(self) -> List[str]:
        """
        Find nodes that create gradient bottlenecks.
        
        These are points where gradient flow is constrained, which can be
        both a defense mechanism and an obstacle for gradient-based attacks.
        """
        bottlenecks = []
        
        for node_id, node in self.nodes.items():
            op_type = node.get("op_type", "")
            
            # Relu creates bottleneck (zero gradient for negative)
            if op_type == "Relu":
                bottlenecks.append(node_id)
            
            # Pooling operations (sparse gradient flow)
            elif op_type in {"MaxPool", "GlobalMaxPool"}:
                bottlenecks.append(node_id)
            
            # Saturating activations
            elif op_type in {"Sigmoid", "Tanh"}:
                bottlenecks.append(node_id)
            
            # 1x1 convolutions that reduce channels
            elif op_type == "Conv":
                kernel = node.get("attributes", {}).get("kernel_shape", [3, 3])
                if kernel == [1, 1] or kernel == 1:
                    bottlenecks.append(node_id)
        
        return bottlenecks
    
    def _find_feature_fusion_points(self) -> List[str]:
        """
        Find nodes where features from multiple paths are fused.
        
        These are critical points for understanding how different parts
        of the model interact and for designing multi-path attacks.
        """
        fusion_points = []
        
        for node_id in self.nodes:
            fan_in = len(self.reverse_adjacency.get(node_id, []))
            if fan_in >= 2:
                op_type = self.nodes[node_id].get("op_type", "")
                if op_type in self.FUSION_OPS:
                    fusion_points.append(node_id)
        
        return fusion_points
    
    def _find_amplification_layers(self) -> List[str]:
        """
        Find layers that amplify signals (and perturbations).
        
        These are key targets for adversarial research as they multiply
        the effect of input perturbations.
        """
        amplifiers = []
        
        for node_id, node in self.nodes.items():
            op_type = node.get("op_type", "")
            
            if op_type in self.AMPLIFICATION_OPS:
                amplifiers.append(node_id)
            
            # Also include attention mechanisms
            if op_type in self.ATTENTION_OPS:
                amplifiers.append(node_id)
            
            # Conv/Linear with large spectral norm potential
            if op_type in self.LINEAR_OPS:
                amplifiers.append(node_id)
        
        return amplifiers
    
    def _recommend_defense_points(self) -> List[str]:
        """
        Recommend optimal points for defense placement.
        
        These are strategic locations where defenses (adversarial training,
        input sanitization, monitoring) would be most effective.
        """
        defense_points = []
        
        # Early layers - input sanitization
        depths = self._compute_node_depths()
        max_depth = max(depths.values()) if depths else 0
        
        early_nodes = [
            nid for nid, d in depths.items()
            if self.topology_config.is_early(d, max_depth)
        ]
        if early_nodes:
            defense_points.append(f"EARLY_LAYER_DEFENSE: {early_nodes[0]} (input sanitization)")
        
        # Feature fusion points - perturbation aggregation defense
        fusion_points = self._find_feature_fusion_points()
        for fp in fusion_points[:2]:  # Top 2
            defense_points.append(f"FUSION_DEFENSE: {fp} (aggregation monitoring)")
        
        # Before final classification - anomaly detection
        output_adjacent = [nid for nid in self.nodes 
                         if not self.adjacency.get(nid) or len(self.adjacency.get(nid, [])) <= 1]
        if output_adjacent:
            defense_points.append(f"FINAL_LAYER_DEFENSE: {output_adjacent[0]} (confidence monitoring)")
        
        # Attention mechanisms - attention pattern monitoring
        attention_nodes = [nid for nid, n in self.nodes.items() 
                         if n.get("op_type") in self.ATTENTION_OPS]
        for att in attention_nodes[:2]:
            defense_points.append(f"ATTENTION_DEFENSE: {att} (attention entropy monitoring)")
        
        return defense_points
    
    # =========================================================================
    # SCORING
    # =========================================================================
    
    def _calculate_structural_score(self, report: StructuralAnalysisReport) -> float:
        """
        Calculate overall structural attack-surface score (0-100).
        
        Score reflects how exploitable the model architecture is,
        NOT just how many patterns exist. Uses weighted scoring
        with diminishing returns for repeated pattern types.
        """
        base_score = 0.0
        pattern_type_counts = {}
        
        # Count patterns by category for diminishing returns
        for pattern in report.high_risk_patterns:
            cat = pattern.category.value
            pattern_type_counts[cat] = pattern_type_counts.get(cat, 0) + 1
        
        # Score with diminishing returns per category
        for pattern in report.high_risk_patterns:
            cat = pattern.category.value
            count = pattern_type_counts.get(cat, 1)
            # First instance of a pattern type has full impact, subsequent reduced
            diminishing_factor = 1.0 / (1 + 0.3 * (count - 1))
            
            if pattern.risk == PatternRisk.CRITICAL:
                base_score += 20 * diminishing_factor
            elif pattern.risk == PatternRisk.HIGH:
                base_score += 10 * diminishing_factor
            elif pattern.risk == PatternRisk.MEDIUM:
                base_score += 4 * diminishing_factor
        
        # Graph metrics bonuses (not penalties, but additional risk indicators)
        if report.longest_linear_chain >= 5:
            base_score += 8  # Long chains enable PGD attacks
        elif report.longest_linear_chain >= 3:
            base_score += 4
        
        if report.max_fan_in >= 6:
            base_score += 6  # High fan-in = coordinated attack surface
        elif report.max_fan_in >= 4:
            base_score += 3
        
        # Normalize to 0-100 scale using sigmoid-like curve
        # This prevents extreme scores and provides meaningful differentiation
        normalized = 100 * (1 - 1 / (1 + base_score / 50))
        return round(min(100, normalized), 1)
    
    def _calculate_robustness_score(self, report: StructuralAnalysisReport) -> float:
        """
        Calculate robustness score (0-100).
        
        Robustness score is INVERSELY related to vulnerability but also
        considers positive defensive features. A high vulnerability score
        necessarily limits the maximum robustness score.
        """
        vuln_score = report.structural_score
        
        # Maximum possible robustness decreases as vulnerability increases
        # At vuln=0, max robustness=100. At vuln=100, max robustness=20.
        max_possible = 100 - vuln_score * 0.8
        
        # Start with 30% of max possible as base
        base_robustness = max_possible * 0.3
        
        # Add points for robustness indicators (capped with diminishing returns)
        robustness_bonus = 0.0
        indicator_count = 0
        for pattern in report.robustness_indicators:
            if pattern.risk == PatternRisk.POSITIVE:
                indicator_count += 1
                # Diminishing returns after first few indicators
                robustness_bonus += 6 / (1 + 0.3 * max(0, indicator_count - 2))
        
        # Capped bonus for specific architectural features
        has_avgpool = any('avgpool' in p.name.lower() or 'globalaverage' in p.name.lower() 
                         for p in report.robustness_indicators)
        has_bottleneck = any('bottleneck' in p.name.lower() or '1x1' in p.name.lower()
                            for p in report.robustness_indicators)
        
        if has_avgpool:
            robustness_bonus += 5
        if has_bottleneck:
            robustness_bonus += 5
        
        # Absence of critical risks adds to robustness
        critical_count = sum(1 for p in report.high_risk_patterns if p.risk == PatternRisk.CRITICAL)
        if critical_count == 0:
            robustness_bonus += 10
        
        # Final score cannot exceed max_possible
        final_score = min(max_possible, base_robustness + robustness_bonus)
        return round(max(0, final_score), 1)


def generate_structural_report_text(report: StructuralAnalysisReport) -> str:
    """Generate human-readable text report from structural analysis."""
    lines = [
        "=" * 70,
        "STRUCTURAL PATTERN ANALYSIS REPORT",
        "=" * 70,
        f"\nModel: {report.model_name}",
        f"Total Nodes: {report.total_nodes}",
        f"Max Depth: {report.max_depth}",
        f"Max Fan-In: {report.max_fan_in}",
        f"Longest Linear Chain: {report.longest_linear_chain}",
    ]

    if report.high_risk_patterns:
        lines.append("\n" + "-" * 70)
        lines.append("STRUCTURAL PATTERNS")
        lines.append("-" * 70)

        for pattern in sorted(report.high_risk_patterns, key=lambda p: p.name):
            lines.append(f"\n{pattern.name}")
            if pattern.registry_id:
                lines.append(f"  Registry ID: {pattern.registry_id}")
            lines.append(f"  Category: {pattern.category.value}")
            lines.append(f"  Nodes: {', '.join(pattern.nodes_involved[:5])}")
            lines.append(f"  Description: {pattern.description.strip()[:200]}...")
            if pattern.research_notes:
                note = pattern.research_notes.strip().split("\n")[0][:160]
                lines.append(f"  Notes: {note}...")

    if report.robustness_indicators:
        lines.append("\n" + "-" * 70)
        lines.append("ARCHITECTURAL INDICATORS")
        lines.append("-" * 70)

        for pattern in report.robustness_indicators:
            lines.append(f"\n{pattern.name}")
            lines.append(f"    {pattern.description.strip()[:150]}...")

    if report.attack_surfaces:
        lines.append("\n" + "-" * 70)
        lines.append("ASSOCIATED ATTACK CLASSES")
        lines.append("-" * 70)

        for surface in report.attack_surfaces:
            lines.append(f"\nComponent: {surface.component}")
            lines.append(f"  Attack Class: {surface.attack_class}")
            lines.append(f"  Techniques: {', '.join(surface.attack_techniques[:3])}")
            lines.append(f"  Nodes: {', '.join(surface.node_ids[:5])}")

    lines.append("\n" + "-" * 70)
    lines.append("GRAPH WORKFLOW TARGETS")
    lines.append("-" * 70)

    if report.gradient_bottlenecks:
        lines.append(f"\nGradient Bottlenecks: {', '.join(report.gradient_bottlenecks[:10])}")
    if report.feature_fusion_points:
        lines.append(f"Feature Fusion Points: {', '.join(report.feature_fusion_points[:10])}")
    if report.amplification_layers:
        lines.append(f"Amplification Layers: {', '.join(report.amplification_layers[:10])}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)

