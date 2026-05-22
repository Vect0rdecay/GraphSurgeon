"""
Structural Motif Analyzer - Core Module

Analyzes neural network DAGs for structural motifs and attack landscape indicators
perspective to identify attack surfaces and exploitable weaknesses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import json


class ThreatCategory(Enum):
    """Categories of adversarial ML threats."""
    ADVERSARIAL_PERTURBATION = "adversarial_perturbation"
    SHADOWLOGIC_INJECTION = "shadowlogic_injection"
    IMPNET_IMPLANTATION = "impnet_implantation"
    MODEL_EXTRACTION = "model_extraction"
    PRIVACY_ATTACK = "privacy_attack"
    EVASION_BYPASS = "evasion_bypass"
    SUPPLY_CHAIN = "supply_chain"


class Severity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingType(Enum):
    """Distinguishes between vulnerabilities and attack surface components."""
    VULNERABILITY = "vulnerability"      # Actual exploitable weakness requiring remediation
    ATTACK_CHAIN = "attack_chain"        # Multiple gadgets combining into a real vuln
    GADGET = "gadget"                    # Attack surface component, not vuln alone
    CHARACTERISTIC = "characteristic"    # Inherent model property, informational only


@dataclass
class Vulnerability:
    """Represents a discovered vulnerability or attack surface finding."""
    id: str
    category: ThreatCategory
    severity: Severity
    node_id: Optional[str]
    title: str
    description: str
    attack_vector: str
    exploitation_difficulty: str
    impact: str
    mitigation: str
    references: List[str] = field(default_factory=list)
    cvss_estimate: float = 0.0
    finding_type: FindingType = FindingType.VULNERABILITY  # New field
    chainable_with: List[str] = field(default_factory=list)  # IDs of gadgets this chains with


@dataclass 
class NodeSecurityProfile:
    """Security profile for a single node in the DAG."""
    node_id: str
    op_type: str
    attributes: Dict[str, Any]
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]
    
    # Vulnerability indicators
    gradient_sensitivity: float = 0.0  # How much gradients amplify through this node
    lipschitz_estimate: float = 1.0    # Local Lipschitz constant estimate
    perturbation_amplification: float = 1.0  # Input perturbation -> output change ratio
    
    # Attack surface indicators
    shadowlogic_capacity: float = 0.0   # Unused capacity for hidden logic
    impnet_payload_capacity: int = 0    # Bytes of potential steganographic payload
    extraction_leakage: float = 0.0     # Information leakage potential
    
    # Plain English analysis
    security_summary: str = ""
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    

@dataclass
class ShadowLogicInjectionPoint:
    """A location in the graph where ShadowLogic could be injected."""
    location: str  # "input_stem", "before_output", "branch_point", "skip_connection"
    node_id: str
    description: str
    injection_complexity: str  # "trivial", "moderate", "complex"
    detection_difficulty: str  # "easy", "moderate", "hard"


@dataclass
class ShadowLogicSusceptibility:
    """Assessment of model's vulnerability to ShadowLogic injection attacks."""
    
    # Detection of existing backdoors
    existing_backdoor_detected: bool = False
    conditional_ops_found: List[str] = field(default_factory=list)
    
    # Phase 5: Filtered attention masking ops (not backdoors, just informational)
    filtered_attention_ops: List[str] = field(default_factory=list)
    
    # Susceptibility to injection (even if no backdoor exists yet)
    susceptibility_score: float = 0.0  # 0-100
    susceptibility_level: str = "UNKNOWN"  # LOW, MEDIUM, HIGH, CRITICAL
    
    # Risk factors
    format_risk: str = "UNKNOWN"  # Graph format editability
    format_risk_detail: str = ""
    
    audit_complexity_risk: str = "UNKNOWN"  # How hard to manually verify
    audit_complexity_detail: str = ""
    
    parameter_hiding_risk: str = "UNKNOWN"  # Capacity to hide malicious weights
    parameter_hiding_detail: str = ""
    
    camouflage_risk: str = "UNKNOWN"  # How easily new nodes blend in
    camouflage_detail: str = ""
    
    integrity_risk: str = "UNKNOWN"  # Lack of integrity verification
    integrity_detail: str = ""
    
    # Injection point mapping
    injection_points: List[ShadowLogicInjectionPoint] = field(default_factory=list)
    
    # Attack scenario
    injection_scenario: str = ""
    
    # Mitigations
    mitigations: List[str] = field(default_factory=list)
    
    # Plain English summary
    summary: str = ""


@dataclass
class ModelSecurityReport:
    """Complete security assessment of a model."""
    model_name: str
    model_format: str
    total_nodes: int
    total_parameters: int
    
    # Overall risk scores (0-100)
    adversarial_risk_score: float = 0.0
    shadowlogic_risk_score: float = 0.0  # Existing backdoor detection
    shadowlogic_susceptibility_score: float = 0.0  # Injection vulnerability
    impnet_risk_score: float = 0.0
    extraction_risk_score: float = 0.0
    privacy_risk_score: float = 0.0
    overall_risk_score: float = 0.0
    
    # Normalized risk score (per-node) - better predictor of actual vulnerability
    # Raw scores scale with model size; normalized scores account for this
    normalized_risk_score: float = 0.0
    
    # ShadowLogic detailed assessment
    shadowlogic_assessment: Optional[ShadowLogicSusceptibility] = None
    
    # Detailed findings
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    node_profiles: Dict[str, NodeSecurityProfile] = field(default_factory=dict)
    
    # Gadget analysis (attack chain building blocks)
    gadgets: List[Any] = field(default_factory=list)  # List[Gadget]
    gadget_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Plain English report
    executive_summary: str = ""
    model_flow_description: str = ""
    attack_surface_summary: str = ""
    hardening_recommendations: List[str] = field(default_factory=list)


# =============================================================================
# OPERATOR SECURITY PROFILES
# =============================================================================

OPERATOR_REFERENCE_DB = {
    # ---------------------------------------------------------------------
    # CONVOLUTION OPERATIONS - Primary attack surface for adversarial examples
    # ---------------------------------------------------------------------
    "Conv": {
        "category": "feature_extraction",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Convolutional layers are highly susceptible to adversarial perturbations.
            Small, imperceptible changes to input pixels can cause large changes in
            feature maps due to weight sharing and local receptive fields.
            
            ATTACK VECTORS:
            - FGSM/PGD attacks exploit gradient flow through conv layers
            - Spatial perturbations target specific filter responses
            - Universal adversarial perturbations often target early conv layers
            
            SHADOWLOGIC POTENTIAL:
            - Unused channels can hide trigger-activated malicious filters
            - Low-magnitude filters may encode hidden functionality
            - Depthwise separable convs have lower hiding capacity
            
            IMPNET VECTORS:
            - Large kernel weights provide steganographic capacity
            - LSB modifications to weights can encode payloads
            - Quantization-aware attacks can survive INT8 conversion
        """,
        "lipschitz_factor": "kernel_norm",  # Lipschitz ~ spectral norm of kernel
        "shadowlogic_capacity": "num_filters * (1 - utilization)",
        "impnet_capacity": "kernel_size^2 * in_channels * out_channels * bit_depth",
        "hardening": [
            "Apply spectral normalization to bound Lipschitz constant",
            "Use certified defense training (IBP, CROWN)",
            "Monitor filter activation statistics for anomalies",
            "Implement weight integrity verification"
        ]
    },
    
    "ConvTranspose": {
        "category": "upsampling",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Transposed convolutions amplify perturbations during upsampling.
            Checkerboard artifacts can mask adversarial patterns.
            
            ATTACK VECTORS:
            - Gradient-based attacks propagate strongly through deconv
            - Output padding variations create attack opportunities
            - Stride mismatches can be exploited
            
            SHADOWLOGIC POTENTIAL:
            - Upsampling creates opportunities for conditional activation
            - Decoder networks are often less scrutinized
        """,
        "lipschitz_factor": "kernel_norm * stride",
        "hardening": [
            "Replace with resize + conv for smoother gradients",
            "Apply output clipping to bound activations"
        ]
    },
    
    # ---------------------------------------------------------------------
    # NORMALIZATION LAYERS - Critical for gradient flow manipulation
    # ---------------------------------------------------------------------
    "BatchNormalization": {
        "category": "normalization",
        "gradient_sensitivity": "medium",
        "adversarial_notes": """
            BatchNorm statistics (mean/variance) are attack surfaces.
            Inference-time BN uses fixed statistics that can be exploited.
            
            ATTACK VECTORS:
            - Input perturbations that shift batch statistics
            - Attacks targeting running mean/variance mismatch
            - Scale/shift parameters can amplify adversarial signals
            
            SHADOWLOGIC POTENTIAL:
            - Affine parameters (gamma/beta) can encode triggers
            - Statistics mismatch can activate hidden paths
            - Training vs inference mode differences exploitable
            
            PRIVACY CONCERNS:
            - Running statistics leak training distribution info
            - Can enable membership inference attacks
        """,
        "lipschitz_factor": "gamma / sqrt(variance + epsilon)",
        "shadowlogic_capacity": "2 * num_features",  # gamma and beta
        "hardening": [
            "Use LayerNorm or GroupNorm for robustness",
            "Freeze BN statistics during adversarial evaluation",
            "Monitor for statistical anomalies"
        ]
    },
    
    "LayerNormalization": {
        "category": "normalization", 
        "gradient_sensitivity": "medium",
        "adversarial_notes": """
            LayerNorm normalizes across features, more stable than BatchNorm.
            Still susceptible to gradient-based attacks but less so.
            
            ATTACK VECTORS:
            - Scale parameter manipulation
            - Attacks on the normalization axis
        """,
        "lipschitz_factor": "gamma_max",
        "hardening": [
            "Bound gamma parameters",
            "Use pre-norm architecture for stability"
        ]
    },
    
    "InstanceNormalization": {
        "category": "normalization",
        "gradient_sensitivity": "medium", 
        "adversarial_notes": """
            Per-instance normalization removes batch dependencies.
            Style transfer models using InstanceNorm have unique attack surfaces.
            
            ATTACK VECTORS:
            - Style-based adversarial attacks
            - Normalization bypass through extreme values
        """
    },
    
    # ---------------------------------------------------------------------
    # ACTIVATION FUNCTIONS - Gradient behavior critical for attacks
    # ---------------------------------------------------------------------
    "Relu": {
        "category": "activation",
        "gradient_sensitivity": "variable",
        "adversarial_notes": """
            ReLU's hard zero gradient for negative inputs creates 'dead zones'.
            This non-linearity is both a weakness and partial defense.
            
            ATTACK VECTORS:
            - Gradient masking through dead neurons (false security)
            - Attacks can push activations into active region
            - ReLU networks have piecewise linear decision boundaries
            
            SHADOWLOGIC POTENTIAL:
            - Dead neurons can be 'awakened' by specific triggers
            - Negative-going triggers activate dormant paths
            
            ROBUSTNESS NOTES:
            - Piecewise linearity enables exact verification (MILP)
            - But also enables efficient adversarial search
        """,
        "lipschitz_factor": 1.0,  # ReLU is 1-Lipschitz
        "shadowlogic_capacity": "dead_neuron_count",
        "hardening": [
            "Use Leaky ReLU to prevent dead neurons",
            "Monitor dead neuron percentage",
            "Adversarial training with PGD"
        ]
    },
    
    "LeakyRelu": {
        "category": "activation",
        "gradient_sensitivity": "medium",
        "adversarial_notes": """
            Non-zero gradient for negative inputs prevents dead neurons.
            Smoother gradient flow can make attacks more effective.
            
            ATTACK VECTORS:
            - Full gradient availability aids optimization attacks
            - Negative slope can be targeted
        """,
        "lipschitz_factor": "max(1, negative_slope)",
        "hardening": [
            "Use small negative slope (0.01-0.1)",
            "Consider parametric version for adaptivity"
        ]
    },
    
    "Sigmoid": {
        "category": "activation",
        "gradient_sensitivity": "low",
        "adversarial_notes": """
            Sigmoid saturates at extremes, causing vanishing gradients.
            This can provide false sense of security (gradient masking).
            
            ATTACK VECTORS:
            - Gradient masking is NOT a defense
            - Black-box attacks bypass gradient issues
            - Transfer attacks from surrogate models effective
            
            SHADOWLOGIC POTENTIAL:
            - Saturation regions can hide conditional behavior
            - Sharp transitions at x=0 can encode triggers
        """,
        "lipschitz_factor": 0.25,  # max derivative of sigmoid
        "hardening": [
            "Don't rely on saturation for security",
            "Implement input validation"
        ]
    },
    
    "Tanh": {
        "category": "activation",
        "gradient_sensitivity": "low",
        "adversarial_notes": """
            Similar to sigmoid but zero-centered. Same gradient masking concerns.
            Output bounded to [-1, 1] which can limit perturbation propagation.
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Same as sigmoid - don't trust gradient masking"
        ]
    },
    
    "Softmax": {
        "category": "activation",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Final softmax is the primary target of classification attacks.
            Confidence scores leak significant information.
            
            ATTACK VECTORS:
            - Targeted attacks aim to maximize wrong class probability
            - Untargeted attacks minimize correct class probability
            - Temperature scaling affects attack difficulty
            
            MODEL EXTRACTION:
            - Softmax outputs enable efficient model stealing
            - Confidence scores reveal decision boundaries
            - Soft labels more informative than hard predictions
            
            PRIVACY ATTACKS:
            - High confidence often indicates training data membership
            - Enables membership inference attacks
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Use temperature scaling to reduce overconfidence",
            "Consider prediction smoothing",
            "Limit confidence score precision in API responses",
            "Implement query rate limiting"
        ]
    },
    
    "Gelu": {
        "category": "activation",
        "gradient_sensitivity": "medium",
        "adversarial_notes": """
            GELU's smooth approximation to ReLU used in transformers.
            Gaussian weighting provides some input-dependent gating.
            
            ATTACK VECTORS:
            - Smooth gradients enable efficient optimization attacks
            - Transformer-specific attacks (attention manipulation)
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Adversarial training specifically for transformers"
        ]
    },
    
    # ---------------------------------------------------------------------
    # ATTENTION MECHANISMS - High-value attack targets
    # ---------------------------------------------------------------------
    "Attention": {
        "category": "attention",
        "gradient_sensitivity": "very_high",
        "adversarial_notes": """
            Attention mechanisms are extremely vulnerable to adversarial attacks.
            Small input changes can dramatically shift attention patterns.
            
            ATTACK VECTORS:
            - Attention hijacking: redirect focus to adversarial tokens
            - Query/Key manipulation to control attention weights
            - Adversarial patches that dominate attention
            
            SHADOWLOGIC POTENTIAL:
            - Attention patterns can encode conditional routing
            - Specific token combinations can trigger hidden paths
            - Multi-head attention provides multiple covert channels
            
            MODEL EXTRACTION:
            - Attention weights reveal model internals
            - Can probe for architectural details
        """,
        "lipschitz_factor": "unbounded",  # Softmax attention is not Lipschitz!
        "shadowlogic_capacity": "num_heads * seq_len^2",
        "hardening": [
            "Use attention dropout",
            "Implement attention clipping",
            "Consider Lipschitz-bounded attention variants",
            "Adversarial training with attention perturbations"
        ]
    },
    
    "MultiHeadAttention": {
        "category": "attention",
        "gradient_sensitivity": "very_high",
        "adversarial_notes": """
            Multiple attention heads multiply attack surface.
            Each head can be targeted independently.
            
            ATTACK VECTORS:
            - Head-specific adversarial perturbations
            - Attention weight distribution attacks
            - Cross-attention exploitation in encoder-decoder models
            
            SHADOWLOGIC POTENTIAL:
            - Individual heads can be corrupted independently
            - Dormant heads can be activated by triggers
            - Head pruning can inadvertently remove trojans
        """,
        "shadowlogic_capacity": "num_heads * hidden_dim",
        "hardening": [
            "Monitor per-head attention entropy",
            "Implement head importance scoring",
            "Use attention regularization"
        ]
    },
    
    # ---------------------------------------------------------------------
    # POOLING OPERATIONS - Information bottlenecks
    # ---------------------------------------------------------------------
    "MaxPool": {
        "category": "pooling",
        "gradient_sensitivity": "sparse",
        "adversarial_notes": """
            MaxPool creates sparse gradient flow (only max elements get gradients).
            This can mask adversarial signals but also hide attacks.
            
            ATTACK VECTORS:
            - Perturbations targeting max-selected elements
            - Strided pooling can be gamed
            
            SHADOWLOGIC POTENTIAL:
            - Non-max elements can contain hidden information
            - Pooling boundary conditions exploitable
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Consider average pooling for smoother gradients",
            "Use overlapping pooling regions"
        ]
    },
    
    "AveragePool": {
        "category": "pooling",
        "gradient_sensitivity": "uniform",
        "adversarial_notes": """
            Average pooling distributes gradients uniformly.
            More predictable gradient flow than MaxPool.
            
            ATTACK VECTORS:
            - Distributed perturbations that survive averaging
            - Easier gradient-based optimization than MaxPool
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Standard adversarial training effective"
        ]
    },
    
    "GlobalAveragePool": {
        "category": "pooling",
        "gradient_sensitivity": "low",
        "adversarial_notes": """
            Global pooling provides regularization effect.
            Aggregates spatial information, making spatial attacks harder.
            
            ATTACK VECTORS:
            - Must create perturbations that survive global averaging
            - Channel-wise attacks more effective than spatial
            
            PRIVACY:
            - Significant information compression
            - Reduces but doesn't eliminate privacy risks
        """,
        "lipschitz_factor": "1/spatial_size",
        "hardening": [
            "Good default choice for classification heads"
        ]
    },
    
    # ---------------------------------------------------------------------
    # LINEAR/DENSE LAYERS - Bulk of model parameters
    # ---------------------------------------------------------------------
    "MatMul": {
        "category": "linear",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Matrix multiplication is the workhorse of neural networks.
            Weight matrices are primary storage for learned knowledge and attacks.
            
            ATTACK VECTORS:
            - Adversarial perturbations amplified by weight magnitudes
            - Spectral properties determine robustness
            
            SHADOWLOGIC POTENTIAL:
            - Large weight matrices have enormous hiding capacity
            - Low-rank perturbations can encode triggers
            - Specific input patterns can activate hidden rows/columns
            
            IMPNET VECTORS:
            - Primary target for weight-based payload hiding
            - LSB steganography in FP32 weights
            - Quantization-surviving payload encoding
            
            MODEL EXTRACTION:
            - Weight matrices are the extraction target
            - Query attacks reconstruct weight information
        """,
        "lipschitz_factor": "spectral_norm(weights)",
        "shadowlogic_capacity": "in_features * out_features * sparsity",
        "impnet_capacity": "in_features * out_features * mantissa_bits",
        "hardening": [
            "Apply spectral normalization",
            "Use weight decay / L2 regularization",
            "Implement weight integrity monitoring",
            "Consider weight quantization"
        ]
    },
    
    "Gemm": {
        "category": "linear",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            General Matrix Multiply - fused linear operation.
            Same vulnerabilities as MatMul with bias addition.
            
            ADDITIONAL VECTORS:
            - Bias terms provide additional hiding capacity
            - Alpha/beta scaling parameters can be exploited
        """,
        "lipschitz_factor": "alpha * spectral_norm(A)",
        "impnet_capacity": "includes_bias_capacity",
        "hardening": [
            "Same as MatMul",
            "Monitor bias magnitudes"
        ]
    },
    
    "Linear": {
        "category": "linear",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Fully connected layer. See MatMul for detailed analysis.
            Final classification layers are high-value targets.
        """,
        "hardening": [
            "Reduce dimensionality where possible",
            "Apply dropout for regularization"
        ]
    },
    
    "Embedding": {
        "category": "linear",
        "gradient_sensitivity": "sparse",
        "adversarial_notes": """
            Embedding lookup tables are discrete and have unique properties.
            Only selected embeddings receive gradients.
            
            ATTACK VECTORS:
            - Adversarial token selection (textual adversarial examples)
            - Embedding space manipulation
            - Synonym/typo attacks in NLP
            
            SHADOWLOGIC POTENTIAL:
            - Rare token embeddings can encode triggers
            - Unused embedding slots provide hiding space
            - Embedding interpolation can reveal hidden behaviors
            
            IMPNET VECTORS:
            - Large embedding tables have massive hiding capacity
            - Infrequently-accessed embeddings ideal for payloads
        """,
        "shadowlogic_capacity": "vocab_size * embed_dim * (1 - token_coverage)",
        "impnet_capacity": "vocab_size * embed_dim * bit_depth",
        "hardening": [
            "Monitor embedding access patterns",
            "Prune unused embeddings",
            "Implement embedding integrity checks"
        ]
    },
    
    # ---------------------------------------------------------------------
    # RECURRENT LAYERS - Temporal attack surfaces
    # ---------------------------------------------------------------------
    "LSTM": {
        "category": "recurrent",
        "gradient_sensitivity": "variable",
        "adversarial_notes": """
            LSTM gates create complex gradient dynamics.
            Hidden state provides persistent attack surface across time.
            
            ATTACK VECTORS:
            - Sequential adversarial attacks
            - Hidden state manipulation
            - Gate-specific perturbations
            - Long-range dependency exploitation
            
            SHADOWLOGIC POTENTIAL:
            - Cell state can accumulate hidden information
            - Gate activations can encode conditional logic
            - Specific sequences can trigger dormant behavior
        """,
        "shadowlogic_capacity": "hidden_size * 4 (gates)",
        "hardening": [
            "Use gradient clipping",
            "Monitor hidden state distributions",
            "Implement sequence-level adversarial training"
        ]
    },
    
    "GRU": {
        "category": "recurrent",
        "gradient_sensitivity": "variable",
        "adversarial_notes": """
            Simplified gating compared to LSTM.
            Similar vulnerabilities but smaller attack surface.
        """,
        "shadowlogic_capacity": "hidden_size * 3 (gates)",
        "hardening": [
            "Same as LSTM"
        ]
    },
    
    # ---------------------------------------------------------------------
    # RESIDUAL CONNECTIONS - Gradient highways
    # ---------------------------------------------------------------------
    "Add": {
        "category": "residual",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Skip connections enable direct gradient flow to early layers.
            This makes deep networks trainable but also more attackable.
            
            ATTACK VECTORS:
            - Gradients propagate unimpeded through skip connections
            - Enables effective attacks on very deep networks
            - Residual signal can carry adversarial information
            
            SHADOWLOGIC POTENTIAL:
            - Skip path can bypass compromised main path
            - Conditional routing based on skip vs main dominance
            - Alpha-weighted residuals (ResNeXt) add control points
        """,
        "lipschitz_factor": 2.0,  # Sum of two 1-Lipschitz paths
        "hardening": [
            "Use stochastic depth during training",
            "Consider dense connections (DenseNet)"
        ]
    },
    
    "Concat": {
        "category": "residual",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Concatenation preserves all information from inputs.
            DenseNet-style architectures use this heavily.
            
            SHADOWLOGIC POTENTIAL:
            - Concatenated features can carry hidden channels
            - Specific channel combinations can trigger behavior
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Monitor channel utilization"
        ]
    },
    
    # ---------------------------------------------------------------------
    # DROPOUT - Training-time regularization (usually inactive at inference)
    # ---------------------------------------------------------------------
    "Dropout": {
        "category": "regularization",
        "gradient_sensitivity": "stochastic",
        "adversarial_notes": """
            Dropout provides some adversarial robustness during training.
            At inference, dropout is disabled - no protection.
            
            ATTACK VECTORS:
            - Monte Carlo dropout at inference can be fooled
            - Adversarial examples that work under any dropout mask
            
            SHADOWLOGIC POTENTIAL:
            - Dropout masks during training can encode triggers
            - Specific neuron combinations at inference
        """,
        "hardening": [
            "Use dropout during adversarial evaluation",
            "Consider always-on stochastic inference"
        ]
    },
    
    # ---------------------------------------------------------------------
    # RESHAPE/VIEW OPERATIONS - Data layout manipulation
    # ---------------------------------------------------------------------
    "Reshape": {
        "category": "view",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Reshape doesn't modify data, just layout.
            Can be used to obscure model architecture.
            
            SHADOWLOGIC POTENTIAL:
            - Non-obvious tensor layouts can hide functionality
            - Reshape chains can obscure data flow
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Verify reshape operations match expected dimensions"
        ]
    },
    
    "Flatten": {
        "category": "view",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Flattening removes spatial structure.
            Transition point between conv and FC layers.
        """,
        "lipschitz_factor": 1.0
    },
    
    "Squeeze": {
        "category": "view",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Dimension removal - no data change.
        """,
        "lipschitz_factor": 1.0
    },
    
    "Unsqueeze": {
        "category": "view",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Dimension addition - no data change.
        """,
        "lipschitz_factor": 1.0
    },
    
    "Transpose": {
        "category": "view",
        "gradient_sensitivity": "passthrough",
        "adversarial_notes": """
            Axis reordering - can affect downstream operations.
        """,
        "lipschitz_factor": 1.0
    },
    
    # ---------------------------------------------------------------------
    # ELEMENTWISE OPERATIONS
    # ---------------------------------------------------------------------
    "Mul": {
        "category": "elementwise",
        "gradient_sensitivity": "variable",
        "adversarial_notes": """
            Multiplication can amplify or attenuate signals.
            Attention weights are applied via multiplication.
            
            ATTACK VECTORS:
            - Scale factor manipulation
            - Gating mechanism exploitation
        """,
        "lipschitz_factor": "max(inputs)",
        "hardening": [
            "Bound multiplicative factors"
        ]
    },
    
    "Div": {
        "category": "elementwise",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Division can create numerical instabilities.
            Small denominators amplify perturbations.
            
            ATTACK VECTORS:
            - Drive denominators toward zero
            - Exploit numerical precision limits
        """,
        "lipschitz_factor": "unbounded",
        "hardening": [
            "Add epsilon to denominators",
            "Clip division results"
        ]
    },
    
    "Pow": {
        "category": "elementwise",
        "gradient_sensitivity": "high",
        "adversarial_notes": """
            Power operations can dramatically amplify inputs.
            Exponentiation is particularly dangerous.
        """,
        "lipschitz_factor": "exponent * input^(exponent-1)",
        "hardening": [
            "Avoid large exponents",
            "Clip inputs before power operations"
        ]
    },
    
    "Exp": {
        "category": "elementwise",
        "gradient_sensitivity": "very_high",
        "adversarial_notes": """
            Exponential growth creates extreme sensitivity.
            Used in softmax - a key attack surface.
        """,
        "lipschitz_factor": "exp(input)",
        "hardening": [
            "Implement log-space computation where possible"
        ]
    },
    
    "Log": {
        "category": "elementwise",
        "gradient_sensitivity": "high_near_zero",
        "adversarial_notes": """
            Log has infinite gradient as input approaches zero.
            Cross-entropy loss uses log.
            
            ATTACK VECTORS:
            - Drive predictions toward zero for numerical issues
        """,
        "lipschitz_factor": "1/input",
        "hardening": [
            "Add epsilon before log",
            "Use numerically stable loss implementations"
        ]
    },
    
    # ---------------------------------------------------------------------
    # COMPARISON/CONDITIONAL OPERATIONS
    # ---------------------------------------------------------------------
    "Where": {
        "category": "conditional",
        "gradient_sensitivity": "discrete",
        "adversarial_notes": """
            Conditional selection creates discrete decision points.
            Gradient doesn't flow through condition.
            
            SHADOWLOGIC POTENTIAL:
            - HIGH - Where operations can implement triggers
            - Condition can check for adversarial patterns
            - Different paths for benign vs malicious inputs
        """,
        "shadowlogic_capacity": "CRITICAL",
        "hardening": [
            "Audit all conditional operations",
            "Verify conditions match expected behavior",
            "Consider soft alternatives (gating)"
        ]
    },
    
    "Less": {
        "category": "conditional",
        "gradient_sensitivity": "zero",
        "adversarial_notes": """
            Comparison operations have zero gradient.
            Often used in control flow.
            
            SHADOWLOGIC POTENTIAL:
            - Can implement threshold-based triggers
        """,
        "shadowlogic_capacity": "HIGH",
        "hardening": [
            "Audit comparison operations"
        ]
    },
    
    "Greater": {
        "category": "conditional",
        "gradient_sensitivity": "zero",
        "adversarial_notes": """
            See Less - same concerns.
        """,
        "shadowlogic_capacity": "HIGH"
    },
    
    "Equal": {
        "category": "conditional",
        "gradient_sensitivity": "zero",
        "adversarial_notes": """
            Exact equality checks are suspicious in continuous models.
            
            SHADOWLOGIC POTENTIAL:
            - CRITICAL - Can check for exact trigger patterns
            - Floating point equality unreliable, so often indicates intentional triggers
        """,
        "shadowlogic_capacity": "CRITICAL",
        "hardening": [
            "Flag all exact equality checks for review"
        ]
    },
    
    # ---------------------------------------------------------------------
    # CLIP/CLAMP OPERATIONS
    # ---------------------------------------------------------------------
    "Clip": {
        "category": "bounding",
        "gradient_sensitivity": "zero_at_bounds",
        "adversarial_notes": """
            Clipping bounds output range.
            Can prevent gradient flow at extremes.
            
            ATTACK VECTORS:
            - Gradient masking at clip boundaries
            - Can push activations to clipped regions
            
            DEFENSE VALUE:
            - Bounds perturbation magnitude
            - Can limit attack effectiveness
        """,
        "lipschitz_factor": 1.0,
        "hardening": [
            "Use for input/output bounding"
        ]
    },
    
    # ---------------------------------------------------------------------
    # REDUCTION OPERATIONS
    # ---------------------------------------------------------------------
    "ReduceMean": {
        "category": "reduction",
        "gradient_sensitivity": "distributed",
        "adversarial_notes": """
            Averaging reduces impact of localized perturbations.
        """,
        "lipschitz_factor": 1.0
    },
    
    "ReduceSum": {
        "category": "reduction",
        "gradient_sensitivity": "amplified",
        "adversarial_notes": """
            Summation can amplify coordinated perturbations.
        """,
        "lipschitz_factor": "num_elements"
    },
    
    "ReduceMax": {
        "category": "reduction",
        "gradient_sensitivity": "sparse",
        "adversarial_notes": """
            Only max element receives gradient.
            Similar to MaxPool vulnerabilities.
        """,
        "lipschitz_factor": 1.0
    },
    
    # ---------------------------------------------------------------------
    # UNKNOWN/CUSTOM OPERATIONS
    # ---------------------------------------------------------------------
    "UNKNOWN": {
        "category": "unknown",
        "gradient_sensitivity": "unknown",
        "adversarial_notes": """
            Unknown or custom operations require manual security review.
            Cannot automatically assess vulnerability profile.
            
            SHADOWLOGIC POTENTIAL:
            - Custom ops are ideal hiding places for malicious logic
            - Opaque operations cannot be verified
        """,
        "shadowlogic_capacity": "CRITICAL - UNKNOWN",
        "hardening": [
            "Manually audit all custom operations",
            "Require documentation and justification",
            "Consider replacing with standard ops"
        ]
    }
}


# =============================================================================
# VULNERABILITY DETECTION RULES
# =============================================================================

class GadgetType(Enum):
    """Types of attack chain gadgets - Evidence-based from adversarial ML research."""
    PERTURBATION_CARRIER = "perturbation_carrier"   # Conv, MatMul - carry gradients
    FUSION_POINT = "fusion_point"                    # Concat, Add - combine signals
    AMPLIFIER = "amplifier"                          # MaxPool, Exp - amplify signals
    NORMALIZER = "normalizer"                        # BN, LN - distribution targets
    CAPACITY_RESERVOIR = "capacity_reservoir"        # Large weights - hiding capacity
    CONTROL_POINT = "control_point"                  # Where, If - trigger potential
    EXTRACTION_SURFACE = "extraction_surface"        # Softmax - info leakage
    GRADIENT_GATE = "gradient_gate"                  # ReLU - sparse gradient flow
    DOWNSAMPLER = "downsampler"                      # Stride-2 ops - aliasing risk
    SKIP_CONNECTION = "skip_connection"              # Residual Add - gradient highway
    LARGE_KERNEL = "large_kernel"                    # 5x5+ kernels - receptive field risk
    SPATIAL_REDUCER = "spatial_reducer"              # Hard spatial drops
    LINEAR_HEAD = "linear_head"                      # GAP->FC - logit sensitivity
    SHAPE_OP = "shape_op"                            # Resize, Pad - structure abuse
    # New evidence-based gadget types from Phase 1 research
    HIGH_FANIN_FUSION = "high_fanin_fusion"          # Concat with >3 inputs - multi-path attack surface
    ALIASING_DOWNSAMPLE = "aliasing_downsample"      # Stride-2 without blur - EOT/RP2 vuln
    GAP_FC_HEAD = "gap_fc_head"                      # GlobalAvgPool->FC - patch attack vuln (GoogleAp)
    MAXPOOL_AFTER_FUSION = "maxpool_after_fusion"    # MaxPool after Concat - amplified patch attack
    
    # Phase 2: Object detector-specific gadget types
    # Research basis: Adversarial YOLO, ShapeShifter, UPC, CAMOU
    OBJECTNESS_HEAD = "objectness_head"              # Detection objectness scoring - Adv YOLO target
    ANCHOR_BASED_DETECTION = "anchor_based_detection"  # Fixed anchor grid - predictable attack targets
    FPN_STRUCTURE = "fpn_structure"                  # Feature Pyramid Network - multi-scale attack surface
    SINGLE_OBJECTNESS_PATH = "single_objectness_path"  # No detection redundancy - complete suppression
    TWO_STAGE_RPN = "two_stage_rpn"                  # Region Proposal Network - ShapeShifter target
    NMS_DEPENDENCY = "nms_dependency"                # Non-Maximum Suppression - confidence manipulation
    SHARED_BACKBONE = "shared_backbone"              # Single feature extractor for detection - single point of failure
    DETECTION_HEAD_PATTERN = "detection_head"        # Multi-output detection structure
    
    # Attention-related gadgets (defensive indicators)
    NO_SPATIAL_ATTENTION = "no_spatial_attention"    # No attention before classifier - patch attack vulnerable
    HAS_SPATIAL_ATTENTION = "has_spatial_attention"  # Has attention - reduces patch vulnerability (defensive)
    
    # Phase 3: Vision Transformer (ViT) specific gadgets
    # Research basis: Papers [103], [114] - ViT-specific vulnerabilities
    VIT_PATCH_EMBEDDING = "vit_patch_embedding"      # Conv with kernel==stride (patch tokenization)
    UNREGULARIZED_ATTENTION = "unregularized_attention"  # Self-attention without dropout
    CLS_TOKEN_AGGREGATION = "cls_token_aggregation"  # CLS token pooling without spatial filtering
    
    # Phase 3: Enhanced CNN gadgets
    # Research basis: Papers [107], [111]
    AGGRESSIVE_EARLY_DOWNSAMPLING = "aggressive_early_downsampling"  # 3+ stride-2 in first 5 layers
    
    # Phase 5: Audio model-specific gadgets
    # Research basis: Carlini Audio 2018, DolphinAttack, CommanderSong, DeepPayload
    AUDIO_MEL_INPUT = "audio_mel_input"              # Model expects mel-spectrogram input (80-128 bins)
    AUDIO_STRIDE_DOWNSAMPLE = "audio_stride_downsample"  # 1D strided conv in audio frontend
    AUDIO_TEMPORAL_ATTENTION = "audio_temporal_attention"  # Self-attention on audio time steps
    CROSS_MODAL_ATTENTION = "cross_modal_attention"  # Encoder-decoder attention (audio->text, image->text)
    AUDIO_1D_CONV = "audio_1d_conv"                  # 1D convolution for temporal processing
    ENCODER_DECODER_SEQ2SEQ = "encoder_decoder_seq2seq"  # Seq2seq architecture with separate encoder/decoder
    
    # Phase 6: Extended Audio/ASR gadgets
    # Research basis: Whisper architecture analysis, CTC-based ASR research
    CTC_DECODER_STRUCTURE = "ctc_decoder_structure"          # CTC output topology (Linear → vocab size)
    SPECIAL_TOKEN_CONTROL_FLOW = "special_token_control_flow"  # Token-gated generation (endoftext, etc.)
    TASK_TOKEN_CONDITIONING = "task_token_conditioning"      # Task embeddings at decoder input
    LANGUAGE_DETECTION_HEAD = "language_detection_head"      # Language classification head on encoder output
    
    # Phase 6: Extended Multimodal gadgets
    # Research basis: CLIP, multimodal fusion attacks, cross-modal jailbreaks
    MULTIMODAL_FUSION_POINT = "multimodal_fusion_point"     # Modality merge operation (Concat/Add at branch convergence)
    CROSS_MODAL_FUSION_LATE = "cross_modal_fusion_late"     # Late-fusion topology (branches independent until near output)
    DUAL_ENCODER_ALIGNMENT = "dual_encoder_alignment"       # Dual-encoder shared embedding space
    TEMPORAL_CROSS_MODAL_SYNC = "temporal_cross_modal_sync" # Shared positional encoding / temporal alignment
    
    # Phase 6: Structural/misc gadgets
    # Research basis: Quantization attacks, LLaVA-style multimodal, 3D point cloud research
    ENCODER_PROJECTION_BRIDGE = "encoder_projection_bridge"  # Encoder→projection→LLM link (dim mismatch bridge)
    QUANTIZATION_NODES = "quantization_nodes"                # QuantizeLinear/DequantizeLinear ops present
    VOXEL_ENCODING = "voxel_encoding"                        # Spatial binning structure (voxelization/pillar ops)


@dataclass
class Gadget:
    """Represents an attack chain gadget (building block for exploits)."""
    id: str
    gadget_type: GadgetType
    node_id: str
    op_type: str
    chainable_with: List[GadgetType]
    attack_contribution: str
    position: str = "unknown"  # early, middle, late in network
    capacity_score: float = 0.0  # For capacity-based gadgets
    kernel_size: Tuple[int, ...] = ()  # For conv ops
    strides: Tuple[int, ...] = ()  # For strided ops
    attributes: Dict[str, Any] = field(default_factory=dict)  # Additional attributes
    
    
class GadgetDetector:
    """
    Identifies MEANINGFUL attack chain gadgets in neural network DAGs.
    
    Design principle: Only flag operations that are:
    1. CONTEXTUALLY ANOMALOUS - Not "a Conv exists" but "this Conv has attack-relevant properties"
    2. ATTACK-ENABLING - Clear mapping to specific adversarial techniques
    3. ACTIONABLE - Could be modified to harden the model
    
    Known attack techniques mapped:
    - FGSM/PGD/C&W: Gradient-based, benefit from clean linear paths
    - Sparse/Patch: One-pixel, localized perturbations, exploit MaxPool
    - Multi-scale/Universal: Exploit multi-branch fusion (Inception-style)
    - Frequency/Fourier: Exploit aliasing from stride-2 without blur
    - Feature-space: Exploit GAP→FC linear separator
    - Transfer attacks: Benefit from multiple vulnerable patterns
    - ShadowLogic/Backdoor: Exploit conditionals + capacity
    """
    
    # Attack technique mapping - what each gadget enables (evidence-based from research)
    ATTACK_TECHNIQUE_MAP = {
        GadgetType.AMPLIFIER: ["sparse_attacks", "one_pixel", "patch_attacks", "pgd"],
        GadgetType.FUSION_POINT: ["multi_scale_pgd", "universal_perturbation", "transfer_attacks"],
        GadgetType.DOWNSAMPLER: ["frequency_attacks", "fourier_attacks", "patch_survivability"],
        GadgetType.SKIP_CONNECTION: ["pgd", "cw", "momentum_attacks", "transfer_attacks"],
        GadgetType.LARGE_KERNEL: ["patch_attacks", "gradient_steering"],
        GadgetType.LINEAR_HEAD: ["feature_space_attacks", "universal_perturbation", "logit_attacks"],
        GadgetType.CONTROL_POINT: ["shadowlogic", "backdoor_triggers"],
        GadgetType.SHAPE_OP: ["eot_attacks", "adversarial_resize", "patch_placement"],
        GadgetType.NORMALIZER: ["distribution_shift_attacks", "domain_attacks"],
        GadgetType.EXTRACTION_SURFACE: ["model_extraction", "membership_inference"],
        # New research-based mappings
        GadgetType.HIGH_FANIN_FUSION: ["multi_scale_pgd", "universal_perturbation", "transfer_attacks", "dpatch"],
        GadgetType.ALIASING_DOWNSAMPLE: ["eot_attacks", "rp2_attacks", "frequency_attacks", "physical_world_attacks"],
        GadgetType.GAP_FC_HEAD: ["adversarial_patch", "lavan", "universal_perturbation", "feature_space_attacks"],
        GadgetType.MAXPOOL_AFTER_FUSION: ["amplified_patch", "multi_scale_patch", "sparse_attacks"],
        
        # Phase 2: Object detector attack mappings
        # Research basis: Adversarial YOLO, ShapeShifter, UPC, CAMOU, Object Hider
        GadgetType.OBJECTNESS_HEAD: ["adversarial_yolo", "object_hider", "disappearance_attacks", "objectness_suppression"],
        GadgetType.ANCHOR_BASED_DETECTION: ["dpatch", "adversarial_yolo", "anchor_manipulation", "targeted_disappearance"],
        GadgetType.FPN_STRUCTURE: ["multi_scale_evasion", "dpatch", "scale_sensitive_attacks", "shapeshifter"],
        GadgetType.SINGLE_OBJECTNESS_PATH: ["complete_suppression", "adversarial_yolo", "object_hider"],
        GadgetType.TWO_STAGE_RPN: ["shapeshifter", "rpn_attacks", "proposal_suppression", "two_stage_evasion"],
        GadgetType.NMS_DEPENDENCY: ["confidence_manipulation", "false_positive_injection", "nms_bypass"],
        GadgetType.SHARED_BACKBONE: ["single_point_failure", "backbone_attacks", "universal_detector_perturbation"],
        GadgetType.DETECTION_HEAD_PATTERN: ["adversarial_yolo", "shapeshifter", "detector_evasion"],
        
        # Attention-related mappings
        GadgetType.NO_SPATIAL_ATTENTION: ["adversarial_patch", "lavan", "universal_perturbation", "localized_attacks"],
        GadgetType.HAS_SPATIAL_ATTENTION: [],  # Defensive - reduces vulnerability
        
        # Phase 3: ViT-specific attack mappings
        # Research basis: Papers [103], [114]
        GadgetType.VIT_PATCH_EMBEDDING: ["vit_patch_attacks", "attention_hijacking", "universal_perturbation"],
        GadgetType.UNREGULARIZED_ATTENTION: ["attention_hijacking", "adversarial_patch", "backdoor_attention"],
        GadgetType.CLS_TOKEN_AGGREGATION: ["vit_patch_attacks", "cls_manipulation", "universal_perturbation"],
        
        # Phase 3: Enhanced CNN mappings
        GadgetType.AGGRESSIVE_EARLY_DOWNSAMPLING: ["small_object_evasion", "physical_patch", "aliasing_attacks"],
        
        # Phase 5: Audio model mappings
        GadgetType.AUDIO_MEL_INPUT: ["audio_adversarial", "psychoacoustic_masking", "frequency_attacks"],
        GadgetType.AUDIO_STRIDE_DOWNSAMPLE: ["audio_adversarial", "audio_aliasing", "robust_audio_attacks"],
        GadgetType.AUDIO_TEMPORAL_ATTENTION: ["audio_adversarial", "attention_hijacking", "temporal_perturbation"],
        GadgetType.CROSS_MODAL_ATTENTION: ["cross_modal_injection", "audio_text_hijacking", "shadowlogic_audio"],
        GadgetType.AUDIO_1D_CONV: ["audio_adversarial", "temporal_perturbation"],
        GadgetType.ENCODER_DECODER_SEQ2SEQ: ["cross_modal_injection", "seq2seq_backdoor", "shadowlogic_audio"],
        
        # Phase 6: Extended Audio/ASR attack mappings
        GadgetType.CTC_DECODER_STRUCTURE: ["ctc_prefix_attack", "audio_adversarial", "forced_transcription", "blank_token_exploit"],
        GadgetType.SPECIAL_TOKEN_CONTROL_FLOW: ["token_injection", "premature_eos", "control_token_hijack", "audio_adversarial"],
        GadgetType.TASK_TOKEN_CONDITIONING: ["task_confusion", "cross_task_attack", "conditioning_hijack"],
        GadgetType.LANGUAGE_DETECTION_HEAD: ["language_confusion", "code_switch_attack", "lang_id_bypass"],
        
        # Phase 6: Extended Multimodal attack mappings
        GadgetType.MULTIMODAL_FUSION_POINT: ["cross_modal_injection", "modality_hijack", "fusion_point_attack", "multimodal_jailbreak"],
        GadgetType.CROSS_MODAL_FUSION_LATE: ["late_fusion_exploit", "modality_disconnect", "adversarial_modality_substitution"],
        GadgetType.DUAL_ENCODER_ALIGNMENT: ["embedding_space_attack", "contrastive_adversarial", "clip_attack", "typographic_attack"],
        GadgetType.TEMPORAL_CROSS_MODAL_SYNC: ["temporal_desync_attack", "alignment_corruption", "sync_exploitation"],
        
        # Phase 6: Structural/misc attack mappings
        GadgetType.ENCODER_PROJECTION_BRIDGE: ["projection_manipulation", "bridge_attack", "dimension_mismatch_exploit"],
        GadgetType.QUANTIZATION_NODES: ["quantization_error_amplification", "bit_flip_attack", "precision_exploit", "adversarial_quantization"],
        GadgetType.VOXEL_ENCODING: ["voxel_perturbation", "point_cloud_attack", "spatial_binning_exploit", "lidar_spoofing"],
    }
    
    def detect_gadgets(self, nodes: List[NodeSecurityProfile], 
                       edges: List[Tuple[str, str]]) -> List[Gadget]:
        """
        Identify ONLY meaningful gadgets that enable known attack techniques.
        
        Does NOT flag:
        - Every Conv (normal operation)
        - Every ReLU (normal operation)
        - Layers just because they have parameters
        - Residual Add without significant skip distance
        
        DOES flag:
        - MaxPool (spike amplification)
        - Concat with multiple branches (multi-scale attacks)
        - Stride-2 in early layers without blur (aliasing)
        - Long skip connections (gradient highways)
        - GAP → FC patterns (linear head vulnerability)
        - Conditional operations (backdoor potential)
        - Missing spatial attention (patch attack vulnerable)
        - Unusual patterns that might enable novel attacks
        """
        gadgets = []
        total_nodes = len(nodes)
        
        # Build graph structure for context analysis
        adjacency = {}
        reverse_adj = {}
        for src, dst in edges:
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(dst)
            if dst not in reverse_adj:
                reverse_adj[dst] = []
            reverse_adj[dst].append(src)
        
        node_indices = {n.node_id: i for i, n in enumerate(nodes)}
        node_map = {n.node_id: n for n in nodes}
        
        # Contextual analysis for selective gadget detection
        for i, node in enumerate(nodes):
            position = self._get_position(i, total_nodes)
            
            gadgets.extend(self._detect_selective_gadgets(
                node, position, i, nodes, adjacency, reverse_adj, node_indices, node_map
            ))
        
        # ========== ATTENTION DETECTION ==========
        # Detect presence/absence of spatial attention mechanisms
        # Research: Models without attention are more vulnerable to patch attacks (GoogleAp, LaVAN)
        attention_info = self._detect_attention_patterns(nodes, adjacency, reverse_adj, node_map)
        
        # Check if model has GAP→FC pattern (classifier architecture)
        has_gap_fc = any(g.gadget_type == GadgetType.GAP_FC_HEAD for g in gadgets)
        
        if attention_info["has_attention"]:
            # Model HAS attention - this is defensive, reduces patch vulnerability
            for i, loc in enumerate(attention_info["attention_locations"][:3]):  # Limit to 3
                att_type = attention_info["attention_type"][i] if i < len(attention_info["attention_type"]) else "unknown"
                gadgets.append(Gadget(
                    id=f"GAD-attention-{loc}",
                    gadget_type=GadgetType.HAS_SPATIAL_ATTENTION,
                    node_id=loc,
                    op_type=att_type,
                    chainable_with=[],  # Defensive gadget - doesn't chain to vulnerabilities
                    attack_contribution=f"DEFENSIVE: {att_type} attention mechanism detected. "
                                       f"Research shows attention reduces patch attack effectiveness by "
                                       f"learning to downweight anomalous spatial regions.",
                    position="middle",
                    attributes={
                        "attention_type": att_type,
                        "before_classifier": attention_info["attention_before_classifier"],
                        "severity": "DEFENSIVE"
                    }
                ))
        else:
            # Model has NO spatial attention - vulnerable to patch attacks
            # Only flag if model has GAP→FC (classifier pattern)
            if has_gap_fc:
                # Find the GAP node to anchor this gadget
                gap_gadget = next((g for g in gadgets if g.gadget_type == GadgetType.GAP_FC_HEAD), None)
                anchor_node = gap_gadget.node_id if gap_gadget else "classifier_head"
                
                gadgets.append(Gadget(
                    id=f"GAD-no-attention-{anchor_node}",
                    gadget_type=GadgetType.NO_SPATIAL_ATTENTION,
                    node_id=anchor_node,
                    op_type="missing_attention",
                    chainable_with=[GadgetType.GAP_FC_HEAD, GadgetType.ALIASING_DOWNSAMPLE],
                    attack_contribution="VULNERABILITY: No spatial attention before classifier. "
                                       "Research (GoogleAp, LaVAN) shows attention mechanisms help "
                                       "filter anomalous patch regions. Without attention, all spatial "
                                       "locations contribute equally to GAP, enabling small patches to dominate.",
                    position="late",
                    attributes={
                        "severity": "HIGH",
                        "hardening": "Add SE blocks, CBAM, or attention pooling before classifier"
                    }
                ))
        
        # ========== PHASE 3: VIT-SPECIFIC GADGETS ==========
        # Research basis: Papers [103], [114]
        
        # Detect ViT architecture patterns
        vit_info = self._detect_vit_patterns(nodes, adjacency, reverse_adj, node_map)
        
        if vit_info["is_vit"]:
            # Add VIT_PATCH_EMBEDDING gadget
            if vit_info["patch_embedding_node"]:
                pe_node = vit_info["patch_embedding_node"]
                gadgets.append(Gadget(
                    id=f"GAD-vit-patch-embed-{pe_node}",
                    gadget_type=GadgetType.VIT_PATCH_EMBEDDING,
                    node_id=pe_node,
                    op_type="patch_embedding",
                    chainable_with=[GadgetType.UNREGULARIZED_ATTENTION, GadgetType.CLS_TOKEN_AGGREGATION],
                    attack_contribution="ViT Patch Embedding: Single linear projection tokenizes image patches. "
                                       "No spatial redundancy - adversarial perturbations directly map to embedded tokens. "
                                       "Research [103] shows this is analogous to CNN input vulnerability but concentrated.",
                    position="early",
                    attributes={
                        "patch_size": vit_info.get("patch_size", 16),
                        "severity": "HIGH"
                    }
                ))
            
            # Add UNREGULARIZED_ATTENTION gadget if no dropout detected
            if vit_info["unregularized_attention"]:
                for att_node in vit_info["attention_nodes"][:3]:  # Limit to 3
                    gadgets.append(Gadget(
                        id=f"GAD-unreg-attention-{att_node}",
                        gadget_type=GadgetType.UNREGULARIZED_ATTENTION,
                        node_id=att_node,
                        op_type="self_attention",
                        chainable_with=[GadgetType.VIT_PATCH_EMBEDDING, GadgetType.CLS_TOKEN_AGGREGATION],
                        attack_contribution="Unregularized Self-Attention: No dropout after attention computation. "
                                           "Attention weights can be manipulated to focus on adversarial patches. "
                                           "Research [103] shows attention hijacking is highly effective.",
                        position="middle",
                        attributes={"severity": "HIGH", "has_dropout": False}
                    ))
            
            # Add CLS_TOKEN_AGGREGATION gadget
            if vit_info["cls_token_node"]:
                cls_node = vit_info["cls_token_node"]
                gadgets.append(Gadget(
                    id=f"GAD-cls-token-{cls_node}",
                    gadget_type=GadgetType.CLS_TOKEN_AGGREGATION,
                    node_id=cls_node,
                    op_type="cls_aggregation",
                    chainable_with=[GadgetType.VIT_PATCH_EMBEDDING, GadgetType.UNREGULARIZED_ATTENTION],
                    attack_contribution="CLS Token Aggregation: Classification uses CLS token that aggregates all patches. "
                                       "Analogous to GAP in CNNs - no spatial filtering before classification. "
                                       "Adversarial patches receive equal attention weight as benign content.",
                    position="late",
                    attributes={"severity": "HIGH"}
                ))
        
        # ========== PHASE 3: AGGRESSIVE EARLY DOWNSAMPLING ==========
        # Research basis: Paper [107] - Small object vulnerability
        
        # Count stride-2 operations in early layers
        early_threshold = int(total_nodes * 0.15)
        stride2_early_count = 0
        stride2_early_nodes = []
        
        for i, node in enumerate(nodes[:early_threshold]):
            if node.op_type in ["Conv", "MaxPool", "AveragePool"]:
                strides = node.attributes.get('strides', (1, 1))
                if isinstance(strides, (list, tuple)) and any(s >= 2 for s in strides):
                    stride2_early_count += 1
                    stride2_early_nodes.append(node.node_id)
        
        if stride2_early_count >= 3:
            gadgets.append(Gadget(
                id=f"GAD-aggressive-downsample",
                gadget_type=GadgetType.AGGRESSIVE_EARLY_DOWNSAMPLING,
                node_id=stride2_early_nodes[0],
                op_type="aggressive_downsampling",
                chainable_with=[GadgetType.ALIASING_DOWNSAMPLE, GadgetType.OBJECTNESS_HEAD],
                attack_contribution=f"Aggressive Early Downsampling: {stride2_early_count} stride-2 operations in first 15% of network. "
                                   f"Rapidly destroys spatial information, making small adversarial patches more effective. "
                                   f"Research [107] shows this increases vulnerability to physical attacks on small objects.",
                position="early",
                attributes={
                    "stride2_count": stride2_early_count,
                    "affected_nodes": stride2_early_nodes[:5],
                    "severity": "MEDIUM"
                }
            ))
        
        # ========== PHASE 5: AUDIO MODEL GADGETS ==========
        # Research basis: Carlini Audio 2018, CommanderSong, DeepPayload, Whisper analysis
        
        audio_info = self._detect_audio_patterns(nodes, adjacency, reverse_adj, node_map)
        
        if audio_info["is_audio_model"]:
            # Add AUDIO_MEL_INPUT gadget
            if audio_info["has_mel_input"]:
                mel_node = audio_info["conv_1d_nodes"][0] if audio_info["conv_1d_nodes"] else "audio_input"
                gadgets.append(Gadget(
                    id=f"GAD-audio-mel-input-{mel_node}",
                    gadget_type=GadgetType.AUDIO_MEL_INPUT,
                    node_id=mel_node,
                    op_type="mel_spectrogram_input",
                    chainable_with=[GadgetType.AUDIO_STRIDE_DOWNSAMPLE, GadgetType.AUDIO_TEMPORAL_ATTENTION],
                    attack_contribution="Audio Mel-Spectrogram Input: Model accepts pre-processed mel spectrograms. "
                                       "Adversarial audio perturbations can be crafted in frequency domain. "
                                       "Research (Carlini 2018) shows psychoacoustic masking enables imperceptible attacks.",
                    position="early",
                    attributes={
                        "audio_model_type": audio_info["audio_model_type"],
                        "severity": "MEDIUM"
                    }
                ))
            
            # Add AUDIO_1D_CONV gadget for each 1D conv
            for conv_node in audio_info["conv_1d_nodes"][:2]:  # Limit to first 2
                gadgets.append(Gadget(
                    id=f"GAD-audio-1d-conv-{conv_node}",
                    gadget_type=GadgetType.AUDIO_1D_CONV,
                    node_id=conv_node,
                    op_type="Conv1D",
                    chainable_with=[GadgetType.AUDIO_MEL_INPUT, GadgetType.AUDIO_STRIDE_DOWNSAMPLE],
                    attack_contribution="Audio 1D Convolution: Processes temporal audio features. "
                                       "Perturbations need only be crafted along time dimension. "
                                       "Simpler attack surface than 2D image convolutions.",
                    position="early",
                    attributes={"severity": "LOW"}
                ))
            
            # Add AUDIO_STRIDE_DOWNSAMPLE if detected
            if audio_info["has_audio_stride_downsample"]:
                stride_node = audio_info["conv_1d_nodes"][0] if audio_info["conv_1d_nodes"] else "audio_downsample"
                gadgets.append(Gadget(
                    id=f"GAD-audio-stride-downsample-{stride_node}",
                    gadget_type=GadgetType.AUDIO_STRIDE_DOWNSAMPLE,
                    node_id=stride_node,
                    op_type="strided_audio_conv",
                    chainable_with=[GadgetType.AUDIO_MEL_INPUT, GadgetType.AUDIO_TEMPORAL_ATTENTION],
                    attack_contribution="Audio Stride Downsampling: Time-domain downsampling without anti-aliasing. "
                                       "High-frequency audio perturbations (clicks, noise bursts) alias into lower frequencies. "
                                       "Analogous to ALIASING_DOWNSAMPLE in vision - enables robust adversarial audio.",
                    position="early",
                    attributes={"severity": "HIGH"}
                ))
            
            # Add AUDIO_TEMPORAL_ATTENTION if detected
            if audio_info["has_temporal_attention"]:
                for att_node in audio_info["attention_nodes"][:2]:  # Limit to first 2
                    gadgets.append(Gadget(
                        id=f"GAD-audio-temporal-attention-{att_node}",
                        gadget_type=GadgetType.AUDIO_TEMPORAL_ATTENTION,
                        node_id=att_node,
                        op_type="audio_self_attention",
                        chainable_with=[GadgetType.AUDIO_STRIDE_DOWNSAMPLE, GadgetType.CROSS_MODAL_ATTENTION],
                        attack_contribution="Audio Temporal Self-Attention: Attention mechanism on audio time steps. "
                                           "Adversarial perturbations at any time can influence all positions. "
                                           "Similar to ViT attention vulnerability but in temporal domain.",
                        position="middle",
                        attributes={"severity": "HIGH"}
                    ))
            
            # Add CROSS_MODAL_ATTENTION if detected
            if audio_info["has_cross_modal_attention"]:
                for cross_node in audio_info["cross_attention_nodes"][:2]:  # Limit to first 2
                    gadgets.append(Gadget(
                        id=f"GAD-cross-modal-attention-{cross_node}",
                        gadget_type=GadgetType.CROSS_MODAL_ATTENTION,
                        node_id=cross_node,
                        op_type="cross_attention",
                        chainable_with=[GadgetType.AUDIO_TEMPORAL_ATTENTION, GadgetType.UNREGULARIZED_ATTENTION],
                        attack_contribution="Cross-Modal Attention: Decoder attends to encoder hidden states. "
                                           "Critical attack surface for audio-to-text attacks. "
                                           "Adversarial audio perturbations flow directly into text generation.",
                        position="middle",
                        attributes={
                            "severity": "CRITICAL",
                            "attack_type": "cross_modal_injection"
                        }
                    ))
            
            # Add ENCODER_DECODER_SEQ2SEQ if cross-modal detected
            if audio_info["audio_model_type"] == "encoder_decoder":
                gadgets.append(Gadget(
                    id="GAD-encoder-decoder-seq2seq",
                    gadget_type=GadgetType.ENCODER_DECODER_SEQ2SEQ,
                    node_id="model_architecture",
                    op_type="seq2seq",
                    chainable_with=[GadgetType.CROSS_MODAL_ATTENTION, GadgetType.AUDIO_TEMPORAL_ATTENTION],
                    attack_contribution="Encoder-Decoder Seq2Seq Architecture: Separate encoder and decoder with cross-attention. "
                                       "Multiple attack surfaces: encoder (feature corruption), decoder (output hijacking). "
                                       "Research shows cross-modal attacks can inject arbitrary text via adversarial audio.",
                    position="architecture",
                    attributes={
                        "severity": "HIGH",
                        "model_type": "encoder_decoder"
                    }
                ))
        
        # ========== PHASE 6: EXTENDED ASR GADGETS ==========
        # Research basis: CTC decoding (Graves 2006), Whisper token protocol,
        # multi-task conditioning, language identification heads
        
        asr_ext_info = self._detect_asr_extended_patterns(nodes, adjacency, reverse_adj, node_map)
        
        # Add CTC_DECODER_STRUCTURE gadget
        if asr_ext_info["has_ctc_decoder"]:
            ctc_node = asr_ext_info["ctc_output_node"] or "ctc_output"
            ctc_vocab = asr_ext_info["ctc_vocab_size"]
            vocab_type = "character-level" if ctc_vocab and ctc_vocab <= 50 else "subword"
            gadgets.append(Gadget(
                id=f"GAD-ctc-decoder-{ctc_node}",
                gadget_type=GadgetType.CTC_DECODER_STRUCTURE,
                node_id=ctc_node,
                op_type="ctc_linear_projection",
                chainable_with=[GadgetType.AUDIO_MEL_INPUT, GadgetType.AUDIO_TEMPORAL_ATTENTION],
                attack_contribution=f"CTC Decoder Structure: Final linear projection to {vocab_type} vocabulary "
                                   f"(dim={ctc_vocab}). CTC decoding uses greedy/beam search with blank token collapsing, "
                                   f"creating attack surface for forced transcription and blank token exploitation. "
                                   f"No softmax gating means raw logit manipulation is feasible.",
                position="late",
                attributes={
                    "severity": "HIGH",
                    "vocab_size": ctc_vocab,
                    "vocab_type": vocab_type,
                }
            ))
        
        # Add SPECIAL_TOKEN_CONTROL_FLOW gadget
        if asr_ext_info["has_special_token_control_flow"]:
            for token_node in asr_ext_info["special_token_nodes"][:3]:  # Limit to first 3
                gadgets.append(Gadget(
                    id=f"GAD-special-token-cf-{token_node}",
                    gadget_type=GadgetType.SPECIAL_TOKEN_CONTROL_FLOW,
                    node_id=token_node,
                    op_type="token_gated_gather",
                    chainable_with=[GadgetType.CROSS_MODAL_ATTENTION, GadgetType.ENCODER_DECODER_SEQ2SEQ],
                    attack_contribution="Special Token Control Flow: Hardcoded Gather indices inject special tokens "
                                       "(SOT, EOT, translate, transcribe) that gate decoder generation. "
                                       "Adversarial audio can manipulate encoder states to hijack token selection, "
                                       "enabling premature EOS or control token injection attacks.",
                    position="early",
                    attributes={
                        "severity": "HIGH",
                        "num_special_token_nodes": len(asr_ext_info["special_token_nodes"]),
                    }
                ))
        
        # Add TASK_TOKEN_CONDITIONING gadget
        if asr_ext_info["has_task_token_conditioning"]:
            conditioning_nodes = asr_ext_info["task_conditioning_nodes"]
            primary_node = conditioning_nodes[0] if conditioning_nodes else "task_conditioning"
            gadgets.append(Gadget(
                id=f"GAD-task-token-cond-{primary_node}",
                gadget_type=GadgetType.TASK_TOKEN_CONDITIONING,
                node_id=primary_node,
                op_type="task_embedding_fusion",
                chainable_with=[GadgetType.SPECIAL_TOKEN_CONTROL_FLOW, GadgetType.CROSS_MODAL_ATTENTION],
                attack_contribution=f"Task Token Conditioning: {len(conditioning_nodes)} parallel embedding lookups "
                                   f"combined before decoder attention. Task conditioning controls model behavior "
                                   f"(translate vs transcribe, language selection). Adversarial inputs can confuse "
                                   f"task conditioning to trigger cross-task attacks or conditioning hijack.",
                position="early",
                attributes={
                    "severity": "MEDIUM",
                    "num_conditioning_inputs": len(conditioning_nodes),
                    "conditioning_nodes": conditioning_nodes[:5],
                }
            ))
        
        # Add LANGUAGE_DETECTION_HEAD gadget
        if asr_ext_info["has_language_detection_head"]:
            lang_node = asr_ext_info["language_head_node"] or "lang_head"
            lang_dim = asr_ext_info["language_head_output_dim"]
            gadgets.append(Gadget(
                id=f"GAD-lang-detect-head-{lang_node}",
                gadget_type=GadgetType.LANGUAGE_DETECTION_HEAD,
                node_id=lang_node,
                op_type="language_classification_head",
                chainable_with=[GadgetType.AUDIO_TEMPORAL_ATTENTION, GadgetType.TASK_TOKEN_CONDITIONING],
                attack_contribution=f"Language Detection Head: Classification branch with {lang_dim} output classes "
                                   f"branching from encoder output. Adversarial audio perturbations can fool language "
                                   f"identification, triggering incorrect language-specific decoding paths. "
                                   f"Enables language confusion and code-switching attacks.",
                position="late",
                attributes={
                    "severity": "MEDIUM",
                    "num_languages": lang_dim,
                }
            ))
        
        # ========== PHASE 6B: MULTIMODAL EXTENDED GADGETS ==========
        # Research basis: CLIP, multimodal fusion attacks, cross-modal jailbreaks
        
        multimodal_ext_info = self._detect_multimodal_extended_patterns(nodes, adjacency, reverse_adj, node_map)
        
        if multimodal_ext_info["has_multimodal_fusion"]:
            for fuse_node in multimodal_ext_info["fusion_point_nodes"][:3]:  # Limit to first 3
                fuse_idx = node_indices.get(fuse_node, 0)
                fuse_position = self._get_position(fuse_idx, total_nodes)
                fuse_op = node_map[fuse_node].op_type if fuse_node in node_map else "Concat"
                gadgets.append(Gadget(
                    id=f"GAD-multimodal-fusion-{fuse_node}",
                    gadget_type=GadgetType.MULTIMODAL_FUSION_POINT,
                    node_id=fuse_node,
                    op_type=f"multimodal_fusion_{fuse_op}",
                    chainable_with=[GadgetType.CROSS_MODAL_FUSION_LATE, GadgetType.CROSS_MODAL_ATTENTION],
                    attack_contribution="Multimodal Fusion Point: Modality branches converge at this node. "
                                       "Adversarial inputs in one modality can corrupt fused representation. "
                                       "Research shows fusion points are critical targets for cross-modal injection.",
                    position=fuse_position,
                    attributes={
                        "severity": "HIGH",
                        "fusion_op": fuse_op,
                        "num_fusion_points": len(multimodal_ext_info["fusion_point_nodes"])
                    }
                ))
        
        if multimodal_ext_info["has_late_fusion"]:
            for late_node in multimodal_ext_info["late_fusion_nodes"][:2]:  # Limit to first 2
                late_op = node_map[late_node].op_type if late_node in node_map else "Concat"
                gadgets.append(Gadget(
                    id=f"GAD-late-fusion-{late_node}",
                    gadget_type=GadgetType.CROSS_MODAL_FUSION_LATE,
                    node_id=late_node,
                    op_type=f"late_fusion_{late_op}",
                    chainable_with=[GadgetType.MULTIMODAL_FUSION_POINT, GadgetType.DUAL_ENCODER_ALIGNMENT],
                    attack_contribution="Late-Fusion Topology: Modality branches process independently until the last 30% "
                                       "of the network. Each branch can be attacked independently without cross-modal "
                                       "interference. Adversarial perturbations in one modality bypass most of the other "
                                       "modality's processing.",
                    position="late",
                    attributes={
                        "severity": "HIGH",
                        "fusion_op": late_op
                    }
                ))
        
        if multimodal_ext_info["has_dual_encoder_alignment"]:
            for pair in multimodal_ext_info["dual_encoder_pairs"][:2]:  # Limit to first 2 pairs
                enc_a, enc_b = pair
                gadgets.append(Gadget(
                    id=f"GAD-dual-encoder-{enc_a}-{enc_b}",
                    gadget_type=GadgetType.DUAL_ENCODER_ALIGNMENT,
                    node_id=enc_a,
                    op_type="dual_encoder_projection",
                    chainable_with=[GadgetType.MULTIMODAL_FUSION_POINT, GadgetType.CROSS_MODAL_FUSION_LATE],
                    attack_contribution="Dual-Encoder Shared Embedding Space: Two projection layers from different branches "
                                       "map to matching dimensions, indicating a shared embedding space (CLIP-like). "
                                       "Adversarial examples can be crafted to align malicious inputs across modalities. "
                                       "Research (typographic attacks on CLIP) shows this enables cross-modal deception.",
                    position="late",
                    attributes={
                        "severity": "HIGH",
                        "encoder_a": enc_a,
                        "encoder_b": enc_b
                    }
                ))
        
        if multimodal_ext_info["has_temporal_sync"]:
            for sync_node in multimodal_ext_info["temporal_sync_nodes"][:2]:  # Limit to first 2
                sync_idx = node_indices.get(sync_node, 0)
                sync_position = self._get_position(sync_idx, total_nodes)
                gadgets.append(Gadget(
                    id=f"GAD-temporal-sync-{sync_node}",
                    gadget_type=GadgetType.TEMPORAL_CROSS_MODAL_SYNC,
                    node_id=sync_node,
                    op_type="shared_positional_encoding",
                    chainable_with=[GadgetType.MULTIMODAL_FUSION_POINT, GadgetType.DUAL_ENCODER_ALIGNMENT],
                    attack_contribution="Shared Positional Encoding: Same positional embedding is added to features in "
                                       "multiple branches. Corrupting the shared encoding affects temporal alignment across "
                                       "all modalities simultaneously. Temporal desynchronization attacks can exploit this "
                                       "shared dependency.",
                    position=sync_position,
                    attributes={
                        "severity": "MEDIUM",
                        "num_sync_points": len(multimodal_ext_info["temporal_sync_nodes"])
                    }
                ))
        
        # ========== PHASE 6: STRUCTURAL/MISC GADGETS ==========
        # Research basis: Quantization attacks, LLaVA-style multimodal bridges, 3D point cloud
        
        structural_info = self._detect_structural_misc_patterns(nodes, adjacency, reverse_adj, node_map)
        
        # Add ENCODER_PROJECTION_BRIDGE gadgets
        if structural_info["has_projection_bridge"]:
            for bridge in structural_info["projection_bridge_ratios"][:3]:  # Limit to 3
                bridge_node = bridge["node_id"]
                gadgets.append(Gadget(
                    id=f"GAD-encoder-projection-bridge-{bridge_node}",
                    gadget_type=GadgetType.ENCODER_PROJECTION_BRIDGE,
                    node_id=bridge_node,
                    op_type="dimension_projection",
                    chainable_with=[GadgetType.CROSS_MODAL_ATTENTION, GadgetType.DUAL_ENCODER_ALIGNMENT],
                    attack_contribution=f"Encoder-Projection Bridge: Dimension-changing projection "
                                       f"({bridge['in_dim']}→{bridge['out_dim']}, ratio {bridge['ratio']}x) "
                                       f"bridges encoder to a different-sized model. "
                                       f"Adversarial perturbations can exploit the lossy dimension mapping. "
                                       f"No immediate activation: {not bridge['has_activation_after']}.",
                    position="middle",
                    attributes={
                        "severity": "HIGH",
                        "in_dim": bridge["in_dim"],
                        "out_dim": bridge["out_dim"],
                        "dimension_ratio": bridge["ratio"],
                        "has_activation_after": bridge["has_activation_after"],
                    }
                ))
        
        # Add QUANTIZATION_NODES gadget
        if structural_info["has_quantization"]:
            quant_anchor = structural_info["quantization_nodes"][0] if structural_info["quantization_nodes"] else "quantized_model"
            total_qdq = structural_info["quantize_node_count"] + structural_info["dequantize_node_count"]
            gadgets.append(Gadget(
                id=f"GAD-quantization-nodes-{quant_anchor}",
                gadget_type=GadgetType.QUANTIZATION_NODES,
                node_id=quant_anchor,
                op_type="quantization",
                chainable_with=[GadgetType.ENCODER_PROJECTION_BRIDGE],
                attack_contribution=f"Quantization Nodes: {total_qdq} QuantizeLinear/DequantizeLinear ops detected "
                                   f"({structural_info['quantize_node_count']} quantize, "
                                   f"{structural_info['dequantize_node_count']} dequantize). "
                                   f"QDQ pattern present: {structural_info['has_qdq_pattern']}. "
                                   f"Quantized models are susceptible to bit-flip attacks and "
                                   f"adversarial perturbations that exploit rounding boundaries.",
                position="architecture",
                attributes={
                    "severity": "MEDIUM",
                    "quantize_count": structural_info["quantize_node_count"],
                    "dequantize_count": structural_info["dequantize_node_count"],
                    "total_qdq_nodes": total_qdq,
                    "has_qdq_pattern": structural_info["has_qdq_pattern"],
                }
            ))
        
        # Add VOXEL_ENCODING gadget
        if structural_info["has_voxel_encoding"]:
            voxel_anchor = structural_info["voxel_nodes"][0] if structural_info["voxel_nodes"] else "voxel_layer"
            gadgets.append(Gadget(
                id=f"GAD-voxel-encoding-{voxel_anchor}",
                gadget_type=GadgetType.VOXEL_ENCODING,
                node_id=voxel_anchor,
                op_type="voxel_encoding",
                chainable_with=[GadgetType.AGGRESSIVE_EARLY_DOWNSAMPLING],
                attack_contribution=f"Voxel Encoding: Spatial binning structure detected via "
                                   f"{structural_info['voxel_signal_type']} signal "
                                   f"({len(structural_info['voxel_nodes'])} nodes). "
                                   f"3D convolutions present: {structural_info['has_3d_conv']}. "
                                   f"Voxelization discretizes continuous point clouds into fixed bins, "
                                   f"enabling adversarial point injection and LiDAR spoofing attacks.",
                position="early",
                attributes={
                    "severity": "HIGH",
                    "signal_type": structural_info["voxel_signal_type"],
                    "voxel_node_count": len(structural_info["voxel_nodes"]),
                    "has_3d_conv": structural_info["has_3d_conv"],
                    "voxel_nodes": structural_info["voxel_nodes"][:5],
                }
            ))
        
        return gadgets
    
    def _get_position(self, idx: int, total: int) -> str:
        """Determine position in network."""
        if idx < total * 0.15:
            return "early"
        elif idx < total * 0.85:
            return "middle"
        return "late"
    
    def _detect_selective_gadgets(self, node: NodeSecurityProfile, position: str, 
                                  idx: int, all_nodes: List[NodeSecurityProfile],
                                  adjacency: dict, reverse_adj: dict,
                                  node_indices: dict, node_map: dict) -> List[Gadget]:
        """
        Selectively detect gadgets based on attack-relevant context.
        
        Only flags operations that are MEANINGFULLY attack-enabling.
        """
        gadgets = []
        attrs = node.attributes
        
        # Extract attributes
        kernel_shape = attrs.get('kernel_shape', ())
        if isinstance(kernel_shape, (list, tuple)) and len(kernel_shape) >= 2:
            kernel_size = tuple(kernel_shape)
        else:
            kernel_size = ()
        
        strides = attrs.get('strides', (1, 1))
        if isinstance(strides, (list, tuple)):
            strides = tuple(strides)
        else:
            strides = (1, 1)
        
        # === MAXPOOL: Always a gadget (spike amplification) ===
        # Research basis: GoogleAp, LaVAN, DPATCH papers show MaxPool amplifies adversarial spikes
        if node.op_type == "MaxPool":
            # Check context: is it after fusion? (CRITICAL - research shows amplified patch vulnerability)
            is_after_fusion = self._is_after_fusion(node.node_id, reverse_adj, node_map)
            
            if is_after_fusion:
                # HIGH severity: MaxPool after fusion is a specific research-documented vulnerability
                gadgets.append(Gadget(
                    id=f"GAD-maxpool-fusion-{node.node_id}",
                    gadget_type=GadgetType.MAXPOOL_AFTER_FUSION,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.FUSION_POINT, GadgetType.GAP_FC_HEAD],
                    attack_contribution="CRITICAL: MaxPool after Concat amplifies fused adversarial signals. "
                                       "Research shows multi-scale patch attacks specifically exploit this pattern. "
                                       "Hardening: Replace with AvgPool or BlurPool.",
                    position=position,
                    kernel_size=kernel_size,
                    strides=strides,
                    attributes={"after_fusion": True, "severity": "HIGH"}
                ))
            else:
                # MEDIUM severity: Regular MaxPool still enables sparse attacks
                gadgets.append(Gadget(
                    id=f"GAD-maxpool-{node.node_id}",
                    gadget_type=GadgetType.AMPLIFIER,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.FUSION_POINT, GadgetType.DOWNSAMPLER],
                    attack_contribution="MaxPool selects extreme activations, amplifying adversarial spikes. "
                                       "Enables sparse/one-pixel attacks. Hardening: Replace with AvgPool.",
                    position=position,
                    kernel_size=kernel_size,
                    strides=strides,
                    attributes={"after_fusion": False, "severity": "MEDIUM"}
                ))
        
        # === CONCAT: Multi-branch fusion (Inception-style) ===
        # Research basis: GoogleAp, DPATCH show multi-branch provides multiple attack entry points
        elif node.op_type == "Concat":
            inputs = reverse_adj.get(node.node_id, [])
            axis = attrs.get('axis', 1)
            
            # Only flag if 2+ branches feeding in (meaningful fusion)
            if len(inputs) >= 2 and axis == 1:  # Channel concat
                if len(inputs) > 3:
                    # HIGH severity: High fan-in fusion - research shows increased attack surface
                    gadgets.append(Gadget(
                        id=f"GAD-highfanin-{node.node_id}",
                        gadget_type=GadgetType.HIGH_FANIN_FUSION,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.AMPLIFIER, GadgetType.MAXPOOL_AFTER_FUSION, GadgetType.LARGE_KERNEL],
                        attack_contribution=f"HIGH-RISK: Concat(axis=1) with {len(inputs)} branches (>3). "
                                           f"Each branch is independent attack entry point. "
                                           f"Enables multi-scale PGD, universal perturbations, DPATCH-style attacks. "
                                           f"WARNING: Research [111] shows adversarial training is LESS effective "
                                           f"for high fan-in architectures. "
                                           f"Hardening: Gated fusion, channel attention, branch dropout.",
                        position=position,
                        attributes={"num_branches": len(inputs), "axis": axis, "severity": "HIGH"}
                    ))
                else:
                    # MEDIUM severity: Standard multi-branch fusion
                    gadgets.append(Gadget(
                        id=f"GAD-fusion-{node.node_id}",
                        gadget_type=GadgetType.FUSION_POINT,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.AMPLIFIER, GadgetType.LARGE_KERNEL],
                        attack_contribution=f"Concat(axis=1) with {len(inputs)} branches - "
                                           f"perturbation superposition enables multi-scale attacks. "
                                           f"Hardening: Consider gated fusion.",
                        position=position,
                        attributes={"num_branches": len(inputs), "axis": axis, "severity": "MEDIUM"}
                    ))
        
        # === ADD: Only if significant skip connection ===
        elif node.op_type == "Add":
            inputs = reverse_adj.get(node.node_id, [])
            if len(inputs) >= 2:
                input_indices = [node_indices.get(inp, idx) for inp in inputs]
                skip_distance = max(abs(idx - inp_idx) for inp_idx in input_indices)
                
                # Only flag if skip distance > 5 (real gradient highway)
                if skip_distance > 5:
                    gadgets.append(Gadget(
                        id=f"GAD-skip-{node.node_id}",
                        gadget_type=GadgetType.SKIP_CONNECTION,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.FUSION_POINT],
                        attack_contribution=f"Skip connection (distance={skip_distance}) - "
                                           f"gradient highway enables strong PGD/C&W with fast convergence. "
                                           f"WARNING: Research [111] shows skip connections create gradient "
                                           f"highways that may bypass adversarial training defenses.",
                        position=position,
                        attributes={"skip_distance": skip_distance}
                    ))
        
        # === STRIDE-2 CONV: Aliasing vulnerability (EOT, RP2 attacks) ===
        # Research basis: EOT (2018), RP2 (2018) show aliasing enables physical-world attacks
        elif node.op_type == "Conv" and any(s >= 2 for s in strides):
            # Check if there's blur/anti-aliasing before
            has_blur = self._has_blur_before(node.node_id, reverse_adj, node_map)
            
            if not has_blur:
                if position == "early":
                    # HIGH severity: Early stride-2 without blur is major aliasing vulnerability
                    gadgets.append(Gadget(
                        id=f"GAD-alias-early-{node.node_id}",
                        gadget_type=GadgetType.ALIASING_DOWNSAMPLE,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.AMPLIFIER, GadgetType.FUSION_POINT, GadgetType.GAP_FC_HEAD],
                        attack_contribution="HIGH-RISK: Early stride-2 Conv without anti-aliasing. "
                                           "Research (EOT, RP2) shows high-freq perturbations fold into "
                                           "persistent lower-freq features, enabling physical-world attacks. "
                                           "Hardening: Add BlurPool or anti-aliased downsampling.",
                        position=position,
                        kernel_size=kernel_size,
                        strides=strides,
                        attributes={"has_blur": False, "severity": "HIGH"}
                    ))
                else:
                    # MEDIUM severity: Later stride-2 without blur still has aliasing risk
                    gadgets.append(Gadget(
                        id=f"GAD-alias-{node.node_id}",
                        gadget_type=GadgetType.DOWNSAMPLER,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.AMPLIFIER, GadgetType.FUSION_POINT],
                        attack_contribution=f"Stride-2 Conv without anti-aliasing at {position} position. "
                                           f"Contributes to frequency attack vulnerability. "
                                           f"Hardening: Add blur before strided operation.",
                        position=position,
                        kernel_size=kernel_size,
                        strides=strides,
                        attributes={"has_blur": False, "severity": "MEDIUM"}
                    ))
        
        # === LARGE KERNEL: Only after fusion (wide influence) ===
        elif node.op_type == "Conv" and kernel_size and any(k >= 5 for k in kernel_size):
            is_after_fusion = self._is_after_fusion(node.node_id, reverse_adj, node_map)
            
            if is_after_fusion:
                gadgets.append(Gadget(
                    id=f"GAD-largek-{node.node_id}",
                    gadget_type=GadgetType.LARGE_KERNEL,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.FUSION_POINT],
                    attack_contribution=f"Large {kernel_size} kernel after fusion - "
                                       f"fused perturbations spread across wide receptive field",
                    position=position,
                    kernel_size=kernel_size,
                    strides=strides,
                    attributes={"after_fusion": True}
                ))
        
        # === GLOBAL POOLING: Patch attack vulnerability (GoogleAp, LaVAN) ===
        # Research basis: GoogleAp (2017), LaVAN (2018) explicitly exploit global pooling
        elif node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
            # Check if followed by FC layer (GAP->FC is the critical pattern)
            downstream = adjacency.get(node.node_id, [])
            has_fc_after = False
            for d in downstream:
                d_node = node_map.get(d)
                if d_node and d_node.op_type in ["Gemm", "MatMul", "Flatten"]:
                    has_fc_after = True
                    break
                # Also check 2 hops (GAP -> Flatten -> FC)
                if d_node:
                    for dd in adjacency.get(d, []):
                        dd_node = node_map.get(dd)
                        if dd_node and dd_node.op_type in ["Gemm", "MatMul"]:
                            has_fc_after = True
                            break
            
            if has_fc_after:
                # HIGH severity: GAP->FC is the canonical patch attack vulnerability
                gadgets.append(Gadget(
                    id=f"GAD-gap-fc-{node.node_id}",
                    gadget_type=GadgetType.GAP_FC_HEAD,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.EXTRACTION_SURFACE, GadgetType.ALIASING_DOWNSAMPLE],
                    attack_contribution="HIGH-RISK: GlobalPool->FC classifier pattern. "
                                       "Research (GoogleAp, LaVAN) shows this is the canonical patch attack vulnerability. "
                                       "Global pooling aggregates local patch features into final representation. "
                                       "Hardening: Attention pooling, spatial verification, patch detection.",
                    position=position,
                    attributes={"has_fc_after": True, "pool_type": node.op_type, "severity": "HIGH"}
                ))
            else:
                # MEDIUM severity: Global pooling without FC still has feature-space vulnerability
                gadgets.append(Gadget(
                    id=f"GAD-gap-{node.node_id}",
                    gadget_type=GadgetType.LINEAR_HEAD,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.EXTRACTION_SURFACE],
                    attack_contribution="Global pooling collapses spatial info into feature vector. "
                                       "Enables feature-space attacks, universal perturbations.",
                    position=position,
                    attributes={"has_fc_after": False, "pool_type": node.op_type, "severity": "MEDIUM"}
                ))
        
        # === FINAL FC: Logit manipulation target ===
        elif node.op_type in ["Gemm", "MatMul"] and position == "late":
            # Check if it's the final classifier
            downstream = adjacency.get(node.node_id, [])
            is_final = not downstream or all(
                node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type 
                in ["Softmax", "Sigmoid", ""] for d in downstream
            )
            
            if is_final:
                gadgets.append(Gadget(
                    id=f"GAD-fc-{node.node_id}",
                    gadget_type=GadgetType.LINEAR_HEAD,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.EXTRACTION_SURFACE],
                    attack_contribution="Final FC layer - direct logit manipulation target; "
                                       "C&W attacks optimize this directly",
                    position=position,
                ))
        
        # === SOFTMAX: Model extraction surface ===
        elif node.op_type == "Softmax":
            gadgets.append(Gadget(
                id=f"GAD-softmax-{node.node_id}",
                gadget_type=GadgetType.EXTRACTION_SURFACE,
                node_id=node.node_id,
                op_type=node.op_type,
                chainable_with=[GadgetType.LINEAR_HEAD],
                attack_contribution="Softmax outputs reveal decision boundaries; "
                                   "enables model extraction, membership inference",
                position=position,
            ))
        
        # === CONDITIONAL OPS: Backdoor potential ===
        # Phase 5 UPDATE: Filter out legitimate transformer attention masking
        elif node.op_type in ["Where", "If", "Equal", "LessOrEqual", "Less", "Greater", "GreaterOrEqual"]:
            node_name_lower = node.node_id.lower()
            
            # Patterns indicating legitimate attention masking
            attention_patterns = [
                "self_attn", "encoder_attn", "cross_attn", "_attn/", "/attn/",
                "attention", "causal_mask", "mask", "/layers.",
            ]
            decoder_patterns = [
                "/model/decoder/lessorequal", "/model/decoder/equal", "/model/decoder/where",
                "/decoder/lessorequal", "/decoder/equal", "/decoder/where",
            ]
            
            is_attention = any(p in node_name_lower for p in attention_patterns)
            is_decoder_mask = any(p in node_name_lower for p in decoder_patterns)
            
            if not (is_attention or is_decoder_mask):
                # NOT attention masking - this is a real control point
                gadgets.append(Gadget(
                    id=f"GAD-cond-{node.node_id}",
                    gadget_type=GadgetType.CONTROL_POINT,
                    node_id=node.node_id,
                    op_type=node.op_type,
                    chainable_with=[GadgetType.CAPACITY_RESERVOIR],
                    attack_contribution=f"Conditional op '{node.op_type}' - potential trigger mechanism; "
                                       f"ShadowLogic attacks use these to switch between benign/malicious paths",
                    position=position,
                ))
            # else: Skip - this is likely legitimate transformer attention masking
        
        # === SHAPE OPS: Only in early layers (structure abuse) ===
        elif node.op_type in ["Resize", "Pad", "Slice", "Crop"] and position == "early":
            gadgets.append(Gadget(
                id=f"GAD-shape-{node.node_id}",
                gadget_type=GadgetType.SHAPE_OP,
                node_id=node.node_id,
                op_type=node.op_type,
                chainable_with=[],
                attack_contribution=f"Early {node.op_type} - enables EoT attacks, "
                                   f"adversarial resizing, patch placement manipulation",
                position=position,
            ))
        
        # === BATCHNORM: Only if not fused (distribution target) ===
        elif node.op_type in ["BatchNormalization", "BatchNorm"]:
            gadgets.append(Gadget(
                id=f"GAD-bn-{node.node_id}",
                gadget_type=GadgetType.NORMALIZER,
                node_id=node.node_id,
                op_type=node.op_type,
                chainable_with=[GadgetType.FUSION_POINT],
                attack_contribution="BatchNorm - distribution shift target; "
                                   "adversarial inputs shift channel statistics, BN amplifies",
                position=position,
            ))
        
        # ========== PHASE 2: OBJECT DETECTOR GADGETS ==========
        # Research basis: Adversarial YOLO, ShapeShifter, UPC, CAMOU
        
        # === DETECTION HEAD PATTERN: Multi-output for boxes/scores/classes ===
        # Detector models typically have late Reshape/Transpose + Concat patterns for outputs
        if node.op_type == "Reshape" and position == "late":
            downstream = adjacency.get(node.node_id, [])
            # Check if followed by Transpose or Concat (detection head structure)
            for d in downstream:
                d_node = node_map.get(d)
                if d_node and d_node.op_type in ["Transpose", "Concat", "Split"]:
                    gadgets.append(Gadget(
                        id=f"GAD-det-head-{node.node_id}",
                        gadget_type=GadgetType.DETECTION_HEAD_PATTERN,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.OBJECTNESS_HEAD, GadgetType.ANCHOR_BASED_DETECTION],
                        attack_contribution="Detection head structure detected (Reshape->Transpose/Concat). "
                                          "Object detector output formatting enables targeted attacks. "
                                          "Research: Adversarial YOLO, ShapeShifter exploit detection heads.",
                        position=position,
                        attributes={"downstream_op": d_node.op_type, "is_detector_head": True}
                    ))
                    break
        
        # === SIGMOID FOR OBJECTNESS: Common in YOLO-style detectors ===
        # Sigmoid at late stage often represents objectness confidence scoring
        if node.op_type == "Sigmoid" and position == "late":
            upstream = reverse_adj.get(node.node_id, [])
            for u in upstream:
                u_node = node_map.get(u)
                if u_node and u_node.op_type in ["Conv", "Gemm", "MatMul"]:
                    # Check output dimensions - objectness typically has spatial dims
                    if node.output_shapes:
                        output_shape = node.output_shapes[0] if node.output_shapes else []
                        # YOLO objectness: [batch, anchors, grid, grid] or [batch, grid, grid, anchors]
                        if len(output_shape) >= 3:
                            gadgets.append(Gadget(
                                id=f"GAD-objectness-{node.node_id}",
                                gadget_type=GadgetType.OBJECTNESS_HEAD,
                                node_id=node.node_id,
                                op_type=node.op_type,
                                chainable_with=[GadgetType.ANCHOR_BASED_DETECTION, GadgetType.NMS_DEPENDENCY],
                                attack_contribution="Objectness scoring layer (Sigmoid after Conv/FC). "
                                                  "CRITICAL for detectors: Adversarial YOLO attacks directly suppress "
                                                  "objectness to make objects 'invisible'. "
                                                  "Hardening: Multi-scale objectness, attention verification.",
                                position=position,
                                attributes={"output_shape": output_shape, "severity": "HIGH"}
                            ))
                            break
        
        # === ANCHOR-BASED DETECTION: Conv with specific output channels ===
        # Anchor detectors have Conv layers with channels = num_anchors * (5 + num_classes)
        # Common: 255 = 3 anchors * 85 (for COCO 80 classes + 4 bbox + 1 obj)
        if node.op_type == "Conv" and position == "late":
            # Check output channels for anchor patterns
            if node.output_shapes and len(node.output_shapes) > 0:
                output_shape = node.output_shapes[0]
                if len(output_shape) >= 2:
                    # Check if output channels match common anchor patterns
                    # YOLO: 255 (3*85), 507 (3*169), 1020 (3*340 for more classes)
                    # SSD: 6 per anchor location
                    channels = output_shape[1] if len(output_shape) > 1 else 0
                    anchor_patterns = [255, 507, 1020, 126, 84, 24, 36, 18]  # Common detector outputs
                    
                    if channels in anchor_patterns or (channels > 100 and channels % 3 == 0):
                        gadgets.append(Gadget(
                            id=f"GAD-anchor-{node.node_id}",
                            gadget_type=GadgetType.ANCHOR_BASED_DETECTION,
                            node_id=node.node_id,
                            op_type=node.op_type,
                            chainable_with=[GadgetType.OBJECTNESS_HEAD, GadgetType.FPN_STRUCTURE],
                            attack_contribution=f"Anchor-based detection output ({channels} channels). "
                                              "Fixed anchor grid provides PREDICTABLE attack targets. "
                                              "Research: DPATCH, Adversarial YOLO exploit anchor priors. "
                                              "Hardening: Anchor-free detection, dynamic anchors.",
                            position=position,
                            kernel_size=kernel_size,
                            attributes={"output_channels": channels, "severity": "MEDIUM"}
                        ))
        
        # === FPN STRUCTURE: Lateral connections with Upsample + Add/Concat ===
        # Feature Pyramid Networks have characteristic Upsample -> Add patterns
        if node.op_type in ["Upsample", "Resize"] and position == "middle":
            downstream = adjacency.get(node.node_id, [])
            for d in downstream:
                d_node = node_map.get(d)
                if d_node and d_node.op_type in ["Add", "Concat"]:
                    gadgets.append(Gadget(
                        id=f"GAD-fpn-{node.node_id}",
                        gadget_type=GadgetType.FPN_STRUCTURE,
                        node_id=node.node_id,
                        op_type=node.op_type,
                        chainable_with=[GadgetType.HIGH_FANIN_FUSION, GadgetType.DETECTION_HEAD_PATTERN],
                        attack_contribution="Feature Pyramid Network pattern (Upsample->Add/Concat). "
                                          "FPN provides MULTIPLE SCALE attack surfaces. "
                                          "Research: ShapeShifter, DPATCH exploit multi-scale features. "
                                          "Each pyramid level is independent attack entry point.",
                        position=position,
                        attributes={"downstream_fusion": d_node.op_type, "severity": "MEDIUM"}
                    ))
                    break
        
        # === TWO-STAGE RPN PATTERN: ROIAlign/ROIPool indicates two-stage detector ===
        if node.op_type in ["RoiAlign", "RoiPool", "ROIAlign", "ROIPool"]:
            gadgets.append(Gadget(
                id=f"GAD-rpn-{node.node_id}",
                gadget_type=GadgetType.TWO_STAGE_RPN,
                node_id=node.node_id,
                op_type=node.op_type,
                chainable_with=[GadgetType.SHARED_BACKBONE, GadgetType.DETECTION_HEAD_PATTERN],
                attack_contribution="Two-stage detector pattern (ROIAlign/ROIPool). "
                                  "RPN is a CRITICAL vulnerability: ShapeShifter attacks suppress proposals. "
                                  "Attacking RPN = complete object disappearance (no proposals = no detections). "
                                  "Hardening: Proposal redundancy, attention-based proposals.",
                position=position,
                attributes={"severity": "HIGH", "attack_type": "complete_suppression"}
            ))
        
        # === NMS DEPENDENCY: NonMaxSuppression node ===
        if node.op_type in ["NonMaxSuppression", "NMS", "BatchedNMS"]:
            gadgets.append(Gadget(
                id=f"GAD-nms-{node.node_id}",
                gadget_type=GadgetType.NMS_DEPENDENCY,
                node_id=node.node_id,
                op_type=node.op_type,
                chainable_with=[GadgetType.OBJECTNESS_HEAD, GadgetType.DETECTION_HEAD_PATTERN],
                attack_contribution="Non-Maximum Suppression dependency. "
                                  "NMS can be exploited via confidence manipulation: "
                                  "1) Suppress true positives by lowering confidence "
                                  "2) Inject false positives with high confidence "
                                  "3) Manipulate IoU to affect box suppression. "
                                  "Hardening: Soft-NMS, confidence calibration.",
                position=position,
                attributes={"severity": "MEDIUM"}
            ))
        
        return gadgets
    
    def _is_after_fusion(self, node_id: str, reverse_adj: dict, node_map: dict) -> bool:
        """Check if this node is within 3 hops after a fusion point."""
        visited = {node_id}
        frontier = [node_id]
        
        for _ in range(3):
            new_frontier = []
            for nid in frontier:
                for pred in reverse_adj.get(nid, []):
                    if pred not in visited:
                        visited.add(pred)
                        new_frontier.append(pred)
                        pred_node = node_map.get(pred)
                        if pred_node and pred_node.op_type in ["Concat", "Add"]:
                            return True
            frontier = new_frontier
        
        return False
    
    def _has_blur_before(self, node_id: str, reverse_adj: dict, node_map: dict) -> bool:
        """Check if there's anti-aliasing (blur) before this node."""
        # Look for blur/avgpool patterns before stride-2
        for pred in reverse_adj.get(node_id, [])[:5]:
            pred_node = node_map.get(pred)
            if pred_node:
                # Check for blur indicators
                if "blur" in pred_node.node_id.lower():
                    return True
                if pred_node.op_type == "AveragePool":
                    # AvgPool can serve as low-pass filter
                    return True
        return False
    
    def _detect_attention_patterns(self, nodes: List[NodeSecurityProfile], 
                                   adjacency: dict, reverse_adj: dict,
                                   node_map: dict) -> Dict[str, Any]:
        """
        Detect spatial attention mechanisms in the network.
        
        Patterns detected:
        1. SE (Squeeze-and-Excitation): GlobalAvgPool → FC → ReLU → FC → Sigmoid → Mul
        2. CBAM-style: Similar with spatial component
        3. Self-attention: MatMul → Softmax → MatMul (Q*K → softmax → *V)
        4. Channel attention: Sigmoid gating on channel dimension
        
        Returns:
            Dict with 'has_attention', 'attention_type', 'attention_locations'
        """
        result = {
            "has_attention": False,
            "attention_type": [],
            "attention_locations": [],
            "attention_before_classifier": False
        }
        
        # Track potential attention patterns
        se_blocks = []
        self_attention = []
        channel_gating = []
        
        # Find GlobalAveragePool nodes (potential SE block starts)
        gap_nodes = [n for n in nodes if n.op_type in ["GlobalAveragePool", "ReduceMean"]]
        
        for gap_node in gap_nodes:
            # Check for SE block pattern: GAP → FC → ReLU → FC → Sigmoid → Mul
            # SE blocks use GAP for channel attention, not for classification
            downstream = adjacency.get(gap_node.node_id, [])
            
            for d1 in downstream:
                d1_node = node_map.get(d1)
                if not d1_node:
                    continue
                    
                # Check for FC/Gemm after GAP (SE reduction)
                if d1_node.op_type in ["Gemm", "MatMul", "Conv"]:
                    # Look for activation → FC → Sigmoid pattern
                    d2_nodes = adjacency.get(d1, [])
                    for d2 in d2_nodes:
                        d2_node = node_map.get(d2)
                        if d2_node and d2_node.op_type in ["Relu", "Silu", "Swish"]:
                            d3_nodes = adjacency.get(d2, [])
                            for d3 in d3_nodes:
                                d3_node = node_map.get(d3)
                                if d3_node and d3_node.op_type in ["Gemm", "MatMul", "Conv"]:
                                    d4_nodes = adjacency.get(d3, [])
                                    for d4 in d4_nodes:
                                        d4_node = node_map.get(d4)
                                        if d4_node and d4_node.op_type in ["Sigmoid", "HardSigmoid"]:
                                            # Found SE block pattern!
                                            se_blocks.append({
                                                "start": gap_node.node_id,
                                                "sigmoid": d4,
                                                "type": "SE_block"
                                            })
        
        # Find self-attention patterns: MatMul → Softmax → MatMul
        softmax_nodes = [n for n in nodes if n.op_type == "Softmax"]
        for softmax_node in softmax_nodes:
            upstream = reverse_adj.get(softmax_node.node_id, [])
            downstream = adjacency.get(softmax_node.node_id, [])
            
            # Check if Softmax is between two MatMul operations
            has_matmul_before = any(
                node_map.get(u, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul" 
                for u in upstream
            )
            has_matmul_after = any(
                node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul" 
                for d in downstream
            )
            
            if has_matmul_before and has_matmul_after:
                self_attention.append({
                    "softmax": softmax_node.node_id,
                    "type": "self_attention"
                })
        
        # Find channel gating: Sigmoid followed by Mul (channel-wise multiplication)
        sigmoid_nodes = [n for n in nodes if n.op_type in ["Sigmoid", "HardSigmoid"]]
        for sig_node in sigmoid_nodes:
            downstream = adjacency.get(sig_node.node_id, [])
            for d in downstream:
                d_node = node_map.get(d)
                if d_node and d_node.op_type == "Mul":
                    # Check if this Sigmoid is part of an SE block we already found
                    is_se = any(se["sigmoid"] == sig_node.node_id for se in se_blocks)
                    if not is_se:
                        channel_gating.append({
                            "sigmoid": sig_node.node_id,
                            "mul": d,
                            "type": "channel_gating"
                        })
        
        # Compile results
        all_attention = se_blocks + self_attention + channel_gating
        
        if all_attention:
            result["has_attention"] = True
            result["attention_type"] = list(set(a["type"] for a in all_attention))
            result["attention_locations"] = [a.get("start") or a.get("softmax") or a.get("sigmoid") 
                                             for a in all_attention]
            
            # Check if any attention is in the late part of the network (before classifier)
            node_indices = {n.node_id: i for i, n in enumerate(nodes)}
            total_nodes = len(nodes)
            late_threshold = int(total_nodes * 0.7)
            
            for loc in result["attention_locations"]:
                if node_indices.get(loc, 0) >= late_threshold:
                    result["attention_before_classifier"] = True
                    break
        
        return result
    
    def _detect_vit_patterns(self, nodes: List[NodeSecurityProfile],
                             adjacency: dict, reverse_adj: dict,
                             node_map: dict) -> Dict[str, Any]:
        """
        Detect Vision Transformer (ViT) specific architecture patterns.
        
        ViT patterns detected:
        1. Patch Embedding: Conv with kernel_size == stride (typically 16x16 stride 16)
        2. Self-Attention blocks: MatMul → Softmax → MatMul with Q, K, V pattern
        3. CLS Token: Slice/Gather extracting first token for classification
        4. LayerNorm before attention (Pre-LN ViT pattern)
        
        Research basis: Papers [103], [114]
        
        Returns:
            Dict with ViT detection results
        """
        result = {
            "is_vit": False,
            "patch_embedding_node": None,
            "patch_size": None,
            "attention_nodes": [],
            "unregularized_attention": False,
            "cls_token_node": None,
            "has_layernorm": False
        }
        
        # === 1. Detect Patch Embedding ===
        # ViT patch embedding: Conv with kernel_shape == strides (non-overlapping)
        for node in nodes[:20]:  # Check early layers
            if node.op_type == "Conv":
                kernel_shape = node.attributes.get('kernel_shape', ())
                strides = node.attributes.get('strides', (1, 1))
                
                if isinstance(kernel_shape, (list, tuple)) and isinstance(strides, (list, tuple)):
                    if len(kernel_shape) >= 2 and len(strides) >= 2:
                        # Check if kernel == stride (non-overlapping patches)
                        if kernel_shape[0] == strides[0] and kernel_shape[1] == strides[1]:
                            # Check if it's a large patch (>=14, typical ViT sizes are 14, 16, 32)
                            if kernel_shape[0] >= 14:
                                result["is_vit"] = True
                                result["patch_embedding_node"] = node.node_id
                                result["patch_size"] = kernel_shape[0]
                                break
        
        # === 2. Detect Self-Attention Blocks ===
        # Pattern: MatMul → Softmax → MatMul (Q*K → softmax → *V)
        softmax_nodes = [n for n in nodes if n.op_type == "Softmax"]
        attention_count = 0
        dropout_after_attention = 0
        
        for softmax_node in softmax_nodes:
            upstream = reverse_adj.get(softmax_node.node_id, [])
            downstream = adjacency.get(softmax_node.node_id, [])
            
            # Check if Softmax is between two MatMul operations
            has_matmul_before = any(
                node_map.get(u, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul"
                for u in upstream
            )
            has_matmul_after = any(
                node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul"
                for d in downstream
            )
            
            if has_matmul_before and has_matmul_after:
                attention_count += 1
                result["attention_nodes"].append(softmax_node.node_id)
                
                # Check for Dropout after attention (regularization)
                # Look within 3 hops after the second MatMul
                for d in downstream:
                    d_downstream = adjacency.get(d, [])
                    for dd in d_downstream[:5]:
                        dd_node = node_map.get(dd)
                        if dd_node and dd_node.op_type == "Dropout":
                            dropout_after_attention += 1
                            break
        
        # If we found multiple attention blocks, likely a ViT
        if attention_count >= 6:  # ViT-Base has 12 attention blocks
            result["is_vit"] = True
            
            # Check if attention is unregularized (no dropout)
            if dropout_after_attention < attention_count * 0.5:
                result["unregularized_attention"] = True
        
        # === 3. Detect CLS Token Extraction ===
        # Pattern: Slice or Gather extracting index 0 (CLS token) in late layers
        for i, node in enumerate(nodes):
            if i < len(nodes) * 0.7:  # Only check late layers
                continue
                
            if node.op_type in ["Slice", "Gather"]:
                # Check if extracting first element (CLS token)
                starts = node.attributes.get('starts', [])
                axes = node.attributes.get('axes', [])
                
                # Simple heuristic: Slice with start=0, end=1 on sequence dimension
                if starts and starts[0] == 0:
                    downstream = adjacency.get(node.node_id, [])
                    # Check if followed by LayerNorm → FC (classification head)
                    for d in downstream:
                        d_node = node_map.get(d)
                        if d_node and d_node.op_type in ["LayerNormalization", "LayerNorm"]:
                            result["cls_token_node"] = node.node_id
                            result["is_vit"] = True
                            break
                        # Or directly to FC
                        if d_node and d_node.op_type in ["Gemm", "MatMul"]:
                            result["cls_token_node"] = node.node_id
                            result["is_vit"] = True
                            break
        
        # === 4. Check for LayerNorm (common in ViTs) ===
        layernorm_count = sum(1 for n in nodes if n.op_type in ["LayerNormalization", "LayerNorm"])
        if layernorm_count >= 10:  # ViT typically has many LayerNorms
            result["has_layernorm"] = True
            if attention_count >= 4:
                result["is_vit"] = True
        
        return result
    
    def _detect_audio_patterns(self, nodes: List[NodeSecurityProfile],
                               adjacency: dict, reverse_adj: dict,
                               node_map: dict) -> Dict[str, Any]:
        """
        Detect audio model-specific architecture patterns.
        
        Research basis: 
        - Carlini & Wagner Audio (2018) - ASR adversarial examples
        - CommanderSong (2018) - hidden commands in music
        - DeepPayload (2021) - neural payload injection
        - Whisper architecture analysis (Phase 5)
        
        Audio model indicators:
        1. Mel-spectrogram input shape (batch, ~80-128 mel bins, time)
        2. 1D convolutions for temporal processing
        3. Stride-2 downsampling on time dimension
        4. Self-attention on audio time steps
        5. Cross-attention (encoder-decoder architectures)
        """
        result = {
            "is_audio_model": False,
            "has_mel_input": False,
            "has_1d_conv": False,
            "has_audio_stride_downsample": False,
            "has_temporal_attention": False,
            "has_cross_modal_attention": False,
            "conv_1d_nodes": [],
            "attention_nodes": [],
            "cross_attention_nodes": [],
            "audio_model_type": None,  # 'encoder', 'decoder', 'encoder_decoder'
        }
        
        # === 1. Detect 1D Convolutions (Audio Frontend) ===
        # Audio models use 1D convs along time dimension
        for node in nodes:
            if node.op_type == "Conv":
                kernel_shape = node.attributes.get("kernel_shape", [])
                # 1D conv has single-element kernel_shape
                if len(kernel_shape) == 1:
                    result["has_1d_conv"] = True
                    result["conv_1d_nodes"].append(node.node_id)
                    
                    # Check for stride-2 downsampling
                    strides = node.attributes.get("strides", [1])
                    if any(s >= 2 for s in strides):
                        result["has_audio_stride_downsample"] = True
        
        # === 2. Detect Mel-Spectrogram Input Shape ===
        # Mel spectrograms typically have ~80-128 mel frequency bins
        # Look at first Conv or MatMul input dimensions
        # Also check actual input tensor shapes for mel bin dimensions
        mel_detected_by_shape = False
        for i, node in enumerate(nodes[:10]):  # Check early layers
            if node.op_type == "Conv" and result["has_1d_conv"]:
                # If we have 1D convs early, likely audio model
                result["has_mel_input"] = True
                result["is_audio_model"] = True
                break
            
            # Check actual input tensor shapes for mel spectrogram signature
            # Expected shapes: (batch, channels, mel_bins, time) or (batch, mel_bins, time)
            input_shapes = node.input_shapes if node.input_shapes else []
            for shape in input_shapes:
                if not shape or not isinstance(shape, (list, tuple)):
                    continue
                if len(shape) == 4:
                    # (batch, channels, mel_bins, time) format
                    mel_dim = shape[2]
                    if isinstance(mel_dim, (int, float)) and 80 <= mel_dim <= 128:
                        result["has_mel_input"] = True
                        result["is_audio_model"] = True
                        result["mel_bin_count"] = int(mel_dim)
                        mel_detected_by_shape = True
                        break
                elif len(shape) == 3:
                    # (batch, mel_bins, time) format
                    mel_dim = shape[1]
                    if isinstance(mel_dim, (int, float)) and 80 <= mel_dim <= 128:
                        result["has_mel_input"] = True
                        result["is_audio_model"] = True
                        result["mel_bin_count"] = int(mel_dim)
                        mel_detected_by_shape = True
                        break
            if mel_detected_by_shape or result["has_mel_input"]:
                break
        
        # === 3. Detect Self-Attention on Audio ===
        # Pattern: MatMul -> Softmax -> MatMul (attention pattern)
        # Combined with 1D convs suggests audio transformer (Whisper, Wav2Vec2)
        attention_count = 0
        for node in nodes:
            if node.op_type == "Softmax":
                upstream = reverse_adj.get(node.node_id, [])
                downstream = adjacency.get(node.node_id, [])
                
                has_matmul_before = any(
                    node_map.get(u, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul"
                    for u in upstream
                )
                has_matmul_after = any(
                    node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul"
                    for d in downstream
                )
                
                if has_matmul_before and has_matmul_after:
                    attention_count += 1
                    result["attention_nodes"].append(node.node_id)
        
        if attention_count >= 2 and result["has_1d_conv"]:
            result["has_temporal_attention"] = True
            result["is_audio_model"] = True
        
        # === 4. Detect Cross-Modal Attention ===
        # Encoder-decoder models have cross-attention where decoder attends to encoder
        # Indicator: Separate "encoder_hidden_states" input or "cross_attn" in node names
        for node in nodes:
            node_name_lower = node.node_id.lower()
            if any(pattern in node_name_lower for pattern in 
                   ["encoder_attn", "cross_attn", "cross_attention", "encoder_hidden"]):
                result["has_cross_modal_attention"] = True
                result["cross_attention_nodes"].append(node.node_id)
        
        # === 5. Determine Audio Model Type ===
        # Also flag as audio model if cross-modal attention is detected (decoder-only)
        if result["has_cross_modal_attention"]:
            result["is_audio_model"] = True
            result["audio_model_type"] = "decoder"  # Decoder with cross-attention (Whisper decoder, etc.)
        
        if result["is_audio_model"]:
            if result["has_cross_modal_attention"] and result["has_1d_conv"]:
                result["audio_model_type"] = "encoder_decoder"  # Full encoder-decoder
            elif result["has_cross_modal_attention"]:
                result["audio_model_type"] = "decoder"  # Decoder-only
            elif result["has_temporal_attention"] and result["has_1d_conv"]:
                result["audio_model_type"] = "encoder"  # Like Whisper encoder
            elif result["has_1d_conv"]:
                result["audio_model_type"] = "cnn_audio"  # CNN-based audio model
        
        return result
    
    def _detect_asr_extended_patterns(self, nodes: List[NodeSecurityProfile],
                                       adjacency: dict, reverse_adj: dict,
                                       node_map: dict) -> Dict[str, Any]:
        """
        Detect extended ASR architecture patterns beyond basic audio detection.
        
        Research basis:
        - CTC decoding topology (Graves 2006, Hannun 2014)
        - Whisper special token protocol (Radford 2022)
        - Multi-task conditioning in speech models
        - Language identification heads in multilingual ASR
        
        Extended ASR indicators:
        1. CTC decoder structure: Linear projection to vocabulary-sized output
        2. Special token control flow: Gather/Embedding with hardcoded indices
        3. Task token conditioning: Multiple embeddings combined before decoder
        4. Language detection head: Small classification branch from encoder
        """
        result = {
            "has_ctc_decoder": False,
            "has_special_token_control_flow": False,
            "has_task_token_conditioning": False,
            "has_language_detection_head": False,
            "ctc_output_node": None,
            "ctc_vocab_size": None,
            "special_token_nodes": [],
            "task_conditioning_nodes": [],
            "language_head_node": None,
            "language_head_output_dim": None,
        }
        
        # Pre-check: collect audio model indicators for context
        has_1d_conv = False
        has_attention = False
        for node in nodes:
            if node.op_type == "Conv":
                kernel_shape = node.attributes.get("kernel_shape", [])
                if len(kernel_shape) == 1:
                    has_1d_conv = True
            if node.op_type == "Softmax":
                upstream = reverse_adj.get(node.node_id, [])
                downstream = adjacency.get(node.node_id, [])
                if any(
                    node_map.get(u, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul"
                    for u in upstream
                ) and any(
                    node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type == "MatMul"
                    for d in downstream
                ):
                    has_attention = True
            if has_1d_conv and has_attention:
                break
        
        is_audio_context = has_1d_conv or has_attention
        
        # === 1. Detect CTC Decoder Structure ===
        # CTC models end with a Linear/Gemm/MatMul projecting to vocabulary size.
        # Typical vocab sizes: 28-50 (character-level), 1000-32000 (subword).
        # CTC outputs are NOT followed by Softmax (log_softmax applied externally).
        ctc_character_range = (28, 50)
        ctc_subword_range = (1000, 32000)
        
        # Scan from the end of the graph to find the final linear projection
        for node in reversed(nodes):
            if node.op_type not in ("Gemm", "MatMul", "Linear"):
                continue
            
            # Check output dimensions for vocab-sized projection
            output_dim = None
            for shape in node.output_shapes:
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    output_dim = shape[-1]
                    break
            
            if output_dim is None:
                # Try weight dimensions from attributes
                output_dim = node.attributes.get("output_dim", None)
            
            if output_dim is None:
                continue
            
            is_vocab_sized = (
                (ctc_character_range[0] <= output_dim <= ctc_character_range[1])
                or (ctc_subword_range[0] <= output_dim <= ctc_subword_range[1])
            )
            
            if not is_vocab_sized or not is_audio_context:
                continue
            
            # Check that there is no Softmax immediately downstream (CTC characteristic)
            downstream = adjacency.get(node.node_id, [])
            has_softmax_after = any(
                node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type == "Softmax"
                for d in downstream
            )
            
            if not has_softmax_after:
                result["has_ctc_decoder"] = True
                result["ctc_output_node"] = node.node_id
                result["ctc_vocab_size"] = output_dim
                break
        
        # === 2. Detect Special Token Control Flow ===
        # Autoregressive decoders use special tokens (SOT, EOT, translate, etc.)
        # as hardcoded Gather indices at decoder start. Look for Gather ops with
        # constant index inputs that feed into Add before main decoder layers.
        gather_with_constant = []
        
        for node in nodes:
            if node.op_type != "Gather":
                continue
            
            # Check if the Gather index input is a constant (not from model input)
            upstream = reverse_adj.get(node.node_id, [])
            has_constant_index = False
            for u in upstream:
                u_node = node_map.get(u)
                if u_node and u_node.op_type in ("Constant", "ConstantOfShape", "Initializer"):
                    has_constant_index = True
                    break
                # Constants often appear as attribute-only nodes with no upstream
                if u_node and not reverse_adj.get(u, []):
                    # Leaf node providing an index, likely a constant/initializer
                    has_constant_index = True
                    break
            
            if has_constant_index:
                gather_with_constant.append(node.node_id)
        
        # Check for sequential Gather ops feeding into Add (token embedding pattern)
        if len(gather_with_constant) >= 2:
            for g_id in gather_with_constant:
                downstream = adjacency.get(g_id, [])
                for d in downstream:
                    d_node = node_map.get(d)
                    if d_node and d_node.op_type == "Add":
                        result["has_special_token_control_flow"] = True
                        if g_id not in result["special_token_nodes"]:
                            result["special_token_nodes"].append(g_id)
        
        # === 3. Detect Task Token Conditioning ===
        # Task conditioning involves 2-4 separate Gather/embedding lookups that
        # get combined via Add or Concat before entering decoder attention layers.
        # Look for parallel Gather ops feeding into the same Add/Concat.
        add_concat_inputs = {}  # Maps Add/Concat node_id -> list of upstream Gather nodes
        
        for node in nodes:
            if node.op_type not in ("Add", "Concat"):
                continue
            upstream = reverse_adj.get(node.node_id, [])
            gather_inputs = []
            for u in upstream:
                u_node = node_map.get(u)
                if u_node and u_node.op_type == "Gather":
                    gather_inputs.append(u)
            if len(gather_inputs) >= 2:
                add_concat_inputs[node.node_id] = gather_inputs
        
        # Find cases where multiple Gather ops converge, then feed into attention
        for ac_node_id, gather_list in add_concat_inputs.items():
            if len(gather_list) < 2:
                continue
            
            # Verify the combined output feeds into attention layers
            downstream_queue = adjacency.get(ac_node_id, [])
            feeds_into_attention = False
            visited = set()
            depth = 0
            check_queue = list(downstream_queue)
            while check_queue and depth < 8:
                next_queue = []
                for d in check_queue:
                    if d in visited:
                        continue
                    visited.add(d)
                    d_node = node_map.get(d)
                    if d_node and d_node.op_type in ("MatMul", "Softmax"):
                        feeds_into_attention = True
                        break
                    next_queue.extend(adjacency.get(d, [])[:3])
                if feeds_into_attention:
                    break
                check_queue = next_queue
                depth += 1
            
            if feeds_into_attention and len(gather_list) >= 2:
                result["has_task_token_conditioning"] = True
                for g in gather_list:
                    if g not in result["task_conditioning_nodes"]:
                        result["task_conditioning_nodes"].append(g)
        
        # === 4. Detect Language Detection Head ===
        # A classification branch from encoder output with small output dim (50-100).
        # This is separate from the main decoder path: a Linear/Gemm branching off
        # with a relatively small output dimension for language classification.
        lang_head_min_dim = 50
        lang_head_max_dim = 100
        
        for node in nodes:
            if node.op_type not in ("Gemm", "MatMul", "Linear"):
                continue
            
            output_dim = None
            for shape in node.output_shapes:
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    output_dim = shape[-1]
                    break
            
            if output_dim is None:
                output_dim = node.attributes.get("output_dim", None)
            
            if output_dim is None:
                continue
            
            if not (lang_head_min_dim <= output_dim <= lang_head_max_dim):
                continue
            
            # Verify this is a branch, not the main decoder path.
            # A language head typically has few or no downstream nodes (terminal),
            # or feeds into Softmax/LogSoftmax only.
            downstream = adjacency.get(node.node_id, [])
            is_terminal_branch = (
                len(downstream) == 0
                or all(
                    node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type
                    in ("Softmax", "LogSoftmax", "Sigmoid", "ArgMax")
                    for d in downstream
                )
            )
            
            if not is_terminal_branch:
                continue
            
            # Check that this branches from something that looks like encoder output,
            # not decoder internals. The upstream should connect to attention/norm layers.
            upstream = reverse_adj.get(node.node_id, [])
            from_encoder_like = False
            for u in upstream:
                u_node = node_map.get(u)
                if u_node and u_node.op_type in (
                    "LayerNormalization", "LayerNorm", "Add", "MatMul", "Relu", "Gelu"
                ):
                    from_encoder_like = True
                    break
            
            if from_encoder_like and is_audio_context:
                result["has_language_detection_head"] = True
                result["language_head_node"] = node.node_id
                result["language_head_output_dim"] = output_dim
                break
        
        return result
    
    def _detect_multimodal_extended_patterns(self, nodes: List[NodeSecurityProfile],
                                              adjacency: dict, reverse_adj: dict,
                                              node_map: dict) -> Dict[str, Any]:
        """
        Detect extended multimodal architecture patterns.
        
        Research basis:
        - CLIP (Radford et al., 2021) - dual-encoder contrastive alignment
        - Multimodal fusion taxonomy (early/mid/late fusion)
        - Cross-modal jailbreak attacks (Qi et al., 2023)
        - Temporal alignment in audio-visual models
        
        Multimodal indicators:
        1. Fusion points where modality branches merge (Concat/Add from distinct subgraphs)
        2. Late-fusion topology (fusion in last 30% of network)
        3. Dual-encoder alignment (matched projection dimensions)
        4. Shared positional encoding across branches
        """
        result = {
            "has_multimodal_fusion": False,
            "has_late_fusion": False,
            "has_dual_encoder_alignment": False,
            "has_temporal_sync": False,
            "fusion_point_nodes": [],
            "late_fusion_nodes": [],
            "dual_encoder_pairs": [],
            "temporal_sync_nodes": [],
        }
        
        total_nodes = len(nodes)
        if total_nodes == 0:
            return result
        
        node_indices = {n.node_id: i for i, n in enumerate(nodes)}
        
        # === Helper: BFS backward trace to find subgraph roots (no proxy roots) ===
        def _trace_branch_roots(start_node_id: str, max_depth: int = 20) -> Set[str]:
            """Trace backward from a node to find root-level ancestors (nodes with no predecessors)."""
            visited = set()
            frontier = [start_node_id]
            roots = set()
            depth = 0
            while frontier and depth < max_depth:
                next_frontier = []
                for nid in frontier:
                    if nid in visited:
                        continue
                    visited.add(nid)
                    parents = reverse_adj.get(nid, [])
                    if not parents:
                        roots.add(nid)
                    else:
                        next_frontier.extend(parents)
                frontier = next_frontier
                depth += 1
            return roots

        def _min_depth_to_roots(start_node_id: str, max_depth: int = 20) -> int:
            """Minimum BFS depth from start to any root (no predecessors)."""
            visited = set()
            frontier = [start_node_id]
            depth = 0
            while frontier and depth < max_depth:
                next_frontier = []
                for nid in frontier:
                    if nid in visited:
                        continue
                    visited.add(nid)
                    parents = reverse_adj.get(nid, [])
                    if not parents:
                        return depth
                    next_frontier.extend(parents)
                frontier = next_frontier
                depth += 1
            return max_depth
        
        # === 1. Detect Multimodal Fusion Points ===
        # Concat or Add nodes receiving inputs from distinct subgraph branches.
        # Only count when branches trace back to disjoint roots (no proxy roots).
        # Filter residual/skip Add: skip when one branch is much shorter (likely residual).
        for node in nodes:
            if node.op_type not in ("Concat", "Add"):
                continue
            
            inputs = reverse_adj.get(node.node_id, [])
            if len(inputs) < 2:
                continue
            
            # Trace each input branch back to find roots (nodes with no predecessors only)
            branch_root_sets = []
            branch_depths = []
            for inp in inputs:
                roots = _trace_branch_roots(inp)
                branch_root_sets.append(roots)
                branch_depths.append(_min_depth_to_roots(inp))
            
            if len(branch_root_sets) < 2:
                continue
            
            # Filter residual/skip Add: if one branch depth is 1 and another > 4, likely residual
            if node.op_type == "Add" and len(branch_depths) >= 2:
                min_d, max_d = min(branch_depths), max(branch_depths)
                if min_d <= 1 and max_d > 4:
                    continue
            
            # Require at least two branches with non-empty roots
            if sum(1 for r in branch_root_sets if r) < 2:
                continue
            
            # Check if at least 2 inputs come from genuinely different subgraph roots (disjoint)
            distinct_branches = 0
            seen_roots = set()
            for root_set in branch_root_sets:
                if not root_set:
                    continue
                if not root_set.intersection(seen_roots):
                    distinct_branches += 1
                seen_roots.update(root_set)
            
            if distinct_branches >= 2:
                result["has_multimodal_fusion"] = True
                result["fusion_point_nodes"].append(node.node_id)
        
        # === 2. Detect Late Fusion (fusion points in last 30% of network) ===
        # Depends on fusion point detection above
        late_threshold = total_nodes * 0.70
        for fuse_nid in result["fusion_point_nodes"]:
            idx = node_indices.get(fuse_nid, 0)
            if idx >= late_threshold:
                result["has_late_fusion"] = True
                result["late_fusion_nodes"].append(fuse_nid)
        
        # === 3. Detect Dual-Encoder Alignment ===
        # Two Linear/Gemm/MatMul projection layers from different branches with matching output dims
        projection_ops = ("MatMul", "Gemm", "Linear")
        projection_nodes = []
        for node in nodes:
            if node.op_type in projection_ops and node.output_shapes:
                projection_nodes.append(node)
        
        # Group projections by output shape and check if pairs come from different branches
        shape_groups: Dict[Tuple, List[NodeSecurityProfile]] = {}
        for pnode in projection_nodes:
            # Use the first output shape as the key
            out_shape = tuple(pnode.output_shapes[0]) if pnode.output_shapes else ()
            if not out_shape:
                continue
            if out_shape not in shape_groups:
                shape_groups[out_shape] = []
            shape_groups[out_shape].append(pnode)
        
        for shape, group in shape_groups.items():
            if len(group) < 2:
                continue
            # Check if any pair comes from different subgraph branches
            for a_idx in range(len(group)):
                for b_idx in range(a_idx + 1, len(group)):
                    a_roots = _trace_branch_roots(group[a_idx].node_id)
                    b_roots = _trace_branch_roots(group[b_idx].node_id)
                    # Different branches if roots don't overlap
                    if not a_roots.intersection(b_roots):
                        result["has_dual_encoder_alignment"] = True
                        result["dual_encoder_pairs"].append(
                            (group[a_idx].node_id, group[b_idx].node_id)
                        )
                        break  # One pair per shape is enough
                if result["has_dual_encoder_alignment"]:
                    break
        
        # === 4. Detect Temporal Cross-Modal Sync ===
        # Same positional-encoding constant added to features in multiple branches
        # Look for Add nodes that combine a "positional"/"pos_embed" constant with features
        pos_pattern_keywords = ["positional", "pos_embed", "position_embed", "pos_enc", "position_encoding"]
        
        # Collect Add nodes that involve positional constants
        pos_add_nodes: Dict[str, List[str]] = {}  # constant_name -> list of Add node_ids
        for node in nodes:
            if node.op_type != "Add":
                continue
            inputs = reverse_adj.get(node.node_id, [])
            for inp in inputs:
                inp_lower = inp.lower()
                for keyword in pos_pattern_keywords:
                    if keyword in inp_lower:
                        if inp not in pos_add_nodes:
                            pos_add_nodes[inp] = []
                        pos_add_nodes[inp].append(node.node_id)
                        break
        
        # If the same positional constant feeds into multiple Add nodes in different branches,
        # this indicates shared positional encoding across modalities
        for const_name, add_node_ids in pos_add_nodes.items():
            if len(add_node_ids) < 2:
                continue
            # Verify the Add nodes are in different subgraph branches
            branch_root_sets = []
            for add_nid in add_node_ids:
                roots = _trace_branch_roots(add_nid)
                branch_root_sets.append(roots)
            
            # Check for distinct branches among the Add nodes
            seen_roots = set()
            distinct_count = 0
            for root_set in branch_root_sets:
                if not root_set.intersection(seen_roots):
                    distinct_count += 1
                seen_roots.update(root_set)
            
            if distinct_count >= 2:
                result["has_temporal_sync"] = True
                result["temporal_sync_nodes"].extend(add_node_ids)
        
        # Deduplicate temporal sync nodes
        result["temporal_sync_nodes"] = list(dict.fromkeys(result["temporal_sync_nodes"]))
        
        return result
    
    def _detect_structural_misc_patterns(self, nodes: List[NodeSecurityProfile],
                                         adjacency: dict, reverse_adj: dict,
                                         node_map: dict) -> Dict[str, Any]:
        """
        Detect structural and miscellaneous architecture patterns.
        
        Research basis:
        - LLaVA/multimodal projection bridges (encoder dim != LLM dim)
        - Quantization-aware training and QDQ node patterns
        - 3D point cloud voxelization (PointPillars, VoxelNet)
        
        Detects:
        1. Encoder-projection bridges (large dimension ratio Linear/Gemm/MatMul)
        2. Quantization nodes (QuantizeLinear/DequantizeLinear presence)
        3. Voxel encoding patterns (ScatterND, 3D convolutions, voxel-named ops)
        """
        result = {
            "has_projection_bridge": False,
            "projection_bridge_nodes": [],
            "projection_bridge_ratios": [],
            "has_quantization": False,
            "quantize_node_count": 0,
            "dequantize_node_count": 0,
            "quantization_nodes": [],
            "has_qdq_pattern": False,
            "has_voxel_encoding": False,
            "voxel_nodes": [],
            "voxel_signal_type": None,  # 'scatter', 'name', 'conv3d'
            "has_3d_conv": False,
        }
        
        # === 1. Detect Encoder-Projection Bridge ===
        # Look for Linear/Gemm/MatMul with large input-to-output dimension ratio (>2x).
        # These bridge an encoder to a different-sized model (e.g., vision encoder to LLM).
        projection_ops = {"MatMul", "Gemm", "Linear"}
        for node in nodes:
            if node.op_type not in projection_ops:
                continue
            
            input_shapes = node.input_shapes if node.input_shapes else []
            output_shapes = node.output_shapes if node.output_shapes else []
            
            if not input_shapes or not output_shapes:
                continue
            
            # Get the last dimension (feature dim) from input and output
            # Shapes are tuples, e.g., (batch, seq, dim) or (batch, dim)
            in_shape = input_shapes[0] if input_shapes else ()
            out_shape = output_shapes[0] if output_shapes else ()
            
            if not in_shape or not out_shape:
                continue
            
            in_dim = in_shape[-1] if len(in_shape) > 0 else 0
            out_dim = out_shape[-1] if len(out_shape) > 0 else 0
            
            # Guard against zero or non-integer dimensions
            if not isinstance(in_dim, (int, float)) or not isinstance(out_dim, (int, float)):
                continue
            if in_dim <= 0 or out_dim <= 0:
                continue
            
            ratio = max(in_dim / out_dim, out_dim / in_dim)
            
            if ratio > 2.0:
                # Check whether there is NO activation function immediately after
                downstream = adjacency.get(node.node_id, [])
                has_immediate_activation = False
                activation_ops = {"Relu", "Sigmoid", "Tanh", "Gelu", "LeakyRelu", "Selu", "Elu"}
                for d in downstream:
                    d_node = node_map.get(d)
                    if d_node and d_node.op_type in activation_ops:
                        has_immediate_activation = True
                        break
                
                result["has_projection_bridge"] = True
                result["projection_bridge_nodes"].append(node.node_id)
                result["projection_bridge_ratios"].append({
                    "node_id": node.node_id,
                    "in_dim": int(in_dim),
                    "out_dim": int(out_dim),
                    "ratio": round(ratio, 2),
                    "has_activation_after": has_immediate_activation,
                })
        
        # === 2. Detect Quantization Nodes ===
        # QuantizeLinear / DequantizeLinear presence indicates quantized model.
        # Also detect QDQ (Quantize-Dequantize-Quantize) sandwich patterns.
        quant_node_ids = []
        dequant_node_ids = []
        for node in nodes:
            if node.op_type == "QuantizeLinear":
                result["quantize_node_count"] += 1
                result["quantization_nodes"].append(node.node_id)
                quant_node_ids.append(node.node_id)
            elif node.op_type == "DequantizeLinear":
                result["dequantize_node_count"] += 1
                result["quantization_nodes"].append(node.node_id)
                dequant_node_ids.append(node.node_id)
        
        if result["quantize_node_count"] > 0 or result["dequantize_node_count"] > 0:
            result["has_quantization"] = True
        
        # Detect QDQ pattern: DequantizeLinear -> compute op -> QuantizeLinear
        if quant_node_ids and dequant_node_ids:
            dequant_set = set(dequant_node_ids)
            for q_id in quant_node_ids:
                upstream = reverse_adj.get(q_id, [])
                for u in upstream:
                    # Check if any grandparent is a DequantizeLinear
                    grandparents = reverse_adj.get(u, [])
                    for gp in grandparents:
                        if gp in dequant_set:
                            result["has_qdq_pattern"] = True
                            break
                    if result["has_qdq_pattern"]:
                        break
                if result["has_qdq_pattern"]:
                    break
        
        # === 3. Detect Voxel Encoding ===
        # Primary signal: ScatterND / ScatterElements (used for voxelization)
        # Secondary signal: node names containing voxel/pillar/pointpillar
        # Tertiary signal: 3D convolutions (Conv with 3D kernel_shape)
        scatter_nodes = []
        name_match_nodes = []
        conv3d_nodes = []
        voxel_name_patterns = ["voxel", "pillar", "pointpillar"]
        
        for node in nodes:
            # ScatterND / ScatterElements are strong voxelization signals
            if node.op_type in ("ScatterND", "ScatterElements"):
                scatter_nodes.append(node.node_id)
            
            # Name-based detection
            node_name_lower = node.node_id.lower()
            if any(pattern in node_name_lower for pattern in voxel_name_patterns):
                name_match_nodes.append(node.node_id)
            
            # 3D convolution detection
            if node.op_type == "Conv":
                kernel_shape = node.attributes.get("kernel_shape", [])
                if len(kernel_shape) == 3:
                    conv3d_nodes.append(node.node_id)
                    result["has_3d_conv"] = True
        
        if scatter_nodes:
            result["has_voxel_encoding"] = True
            result["voxel_nodes"] = scatter_nodes
            result["voxel_signal_type"] = "scatter"
        elif name_match_nodes:
            result["has_voxel_encoding"] = True
            result["voxel_nodes"] = name_match_nodes
            result["voxel_signal_type"] = "name"
        elif conv3d_nodes:
            result["has_voxel_encoding"] = True
            result["voxel_nodes"] = conv3d_nodes
            result["voxel_signal_type"] = "conv3d"
        
        return result
    
    def find_attack_chains(self, gadgets: List[Gadget], 
                          edges: List[Tuple[str, str]]) -> List[Vulnerability]:
        """
        Identify meaningful attack chains from gadget combinations.
        
        Comprehensive detection based on vision model vulnerability taxonomy:
        - Input boundary and preprocessing issues
        - Early downsampling and aliasing risks
        - Pooling type and spike amplification
        - Feature fusion points
        - Normalization fragility
        - Kernel geometry risks
        - Reduction block survivability
        - Head/logits sensitivity
        """
        chains = []
        
        # Build adjacency for gadgets
        gadget_map = {g.node_id: g for g in gadgets}
        adjacency = {g.node_id: [] for g in gadgets}
        reverse_adj = {g.node_id: [] for g in gadgets}
        
        for src, dst in edges:
            if src in gadget_map:
                adjacency[src].append(dst)
            if dst in gadget_map:
                if dst not in reverse_adj:
                    reverse_adj[dst] = []
                reverse_adj[dst].append(src)
        
        # === A. INPUT BOUNDARY ISSUES ===
        chains.extend(self._find_input_preprocessing_issues(gadgets, gadget_map))
        
        # === B. EARLY DOWNSAMPLING AND ALIASING ===
        chains.extend(self._find_early_downsampling_chains(gadgets, gadget_map))
        
        # === C. POOLING AND SPIKE AMPLIFICATION ===
        chains.extend(self._find_maxpool_amplification_chains(gadgets, adjacency, gadget_map))
        chains.extend(self._find_early_amplifier_chains(gadgets, gadget_map))
        
        # === D. FEATURE FUSION CHAINS ===
        chains.extend(self._find_carrier_fusion_amplifier_chains(gadgets, adjacency, gadget_map))
        chains.extend(self._find_multi_carrier_fusion_chains(gadgets, edges, gadget_map))
        
        # === E. SKIP CONNECTION GRADIENT HIGHWAYS ===
        chains.extend(self._find_skip_connection_highways(gadgets, gadget_map))
        
        # === F. NORMALIZATION FRAGILITY ===
        chains.extend(self._find_normalization_issues(gadgets, gadget_map, adjacency))
        
        # === G. LARGE KERNEL AFTER FUSION ===
        chains.extend(self._find_large_kernel_chains(gadgets, adjacency, gadget_map))
        
        # === H. REDUCTION SURVIVABILITY ===
        chains.extend(self._find_reduction_survivability_chains(gadgets, adjacency, gadget_map))
        
        # === I. HEAD SENSITIVITY ===
        chains.extend(self._find_head_sensitivity_chains(gadgets, adjacency, gadget_map))
        
        # === SHADOWLOGIC ===
        chains.extend(self._find_capacity_control_chains(gadgets, adjacency, gadget_map))
        
        # === GRADIENT HIGHWAYS (carrier sequences) ===
        chains.extend(self._find_gradient_highways(gadgets, adjacency, gadget_map))
        
        # === RESEARCH-BASED CHAINS (Phase 1 findings) ===
        # Detect patterns from GoogleAp, EOT, LaVAN, RP2, DPATCH research
        chains.extend(self._find_research_based_chains(gadgets, gadget_map))
        
        return chains
    
    def _find_input_preprocessing_issues(self, gadgets: List[Gadget], 
                                         gadget_map: dict) -> List[Vulnerability]:
        """A. Detect input boundary and preprocessing vulnerabilities."""
        chains = []
        
        # Check early gadgets for preprocessing patterns
        early_gadgets = [g for g in gadgets if g.position == "early"]
        
        # Look for missing normalization at input
        early_normalizers = [g for g in early_gadgets if g.gadget_type == GadgetType.NORMALIZER]
        early_carriers = [g for g in early_gadgets if g.gadget_type == GadgetType.PERTURBATION_CARRIER]
        
        if early_carriers and not early_normalizers:
            first_carrier = early_carriers[0]
            chains.append(Vulnerability(
                id=f"CHAIN-INPUT-NONORM-{first_carrier.node_id}",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM,
                node_id=first_carrier.node_id,
                title="Input Normalization Missing",
                description=f"No normalization (Sub/Div/BN) before first Conv '{first_carrier.node_id}'. "
                           f"Input goes directly into feature extraction.",
                attack_vector="Model becomes sensitive to distribution shifts (camera pipeline, JPEG, lighting). "
                             "Attacks exploit mismatch between training and deployment distributions. "
                             "Gradients not stabilized by expected scaling.",
                exploitation_difficulty="Low - transfer attacks across pipelines",
                impact="Robustness collapse under benign shifts; easier adversarial examples",
                mitigation="Bake normalization into graph (Sub mean, Div std). "
                          "Standardize input pipeline. Test-time input validation.",
                references=["https://arxiv.org/abs/1312.6199"],
                cvss_estimate=5.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # Check for shape manipulation ops (dynamic input vulnerability)
        shape_ops = [g for g in early_gadgets if g.gadget_type == GadgetType.SHAPE_OP]
        if len(shape_ops) >= 2:
            chains.append(Vulnerability(
                id=f"CHAIN-INPUT-DYNAMIC",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.LOW,
                node_id=shape_ops[0].node_id,
                title=f"Dynamic Input Processing ({len(shape_ops)} shape ops)",
                description=f"Multiple shape operations ({', '.join(g.op_type for g in shape_ops[:3])}) "
                           f"in early layers suggest dynamic input handling.",
                attack_vector="Unexpected aspect ratios/resolutions can trigger brittle behavior. "
                             "Enables 'structure abuse' (EoT attacks, adversarial resizing, patch placement).",
                exploitation_difficulty="Medium",
                impact="Patch placement robustness issues; adversarial resizing",
                mitigation="Enforce fixed input sizes. Constrain preprocess ops. Canonicalize aspect ratio.",
                references=[],
                cvss_estimate=4.0,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        return chains
    
    def _find_early_downsampling_chains(self, gadgets: List[Gadget],
                                        gadget_map: dict) -> List[Vulnerability]:
        """B. Detect early downsampling and aliasing vulnerabilities."""
        chains = []
        
        # Find stride-2 ops in early layers
        early_downsamplers = [g for g in gadgets 
                            if g.gadget_type == GadgetType.DOWNSAMPLER and g.position == "early"]
        
        if early_downsamplers:
            stride2_convs = [g for g in early_downsamplers 
                           if g.op_type == "Conv" and any(s >= 2 for s in g.strides)]
            
            if stride2_convs:
                chains.append(Vulnerability(
                    id=f"CHAIN-EARLY-STRIDE2",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.MEDIUM,
                    node_id=stride2_convs[0].node_id,
                    title=f"Early Stride-2 Convolutions ({len(stride2_convs)} in first layers)",
                    description=f"Stride-2 convolutions in early layers: "
                               f"{', '.join(g.node_id for g in stride2_convs[:3])}. "
                               f"Downsampling without low-pass filtering causes aliasing.",
                    attack_vector="High-frequency perturbations fold into lower-frequency features that persist. "
                                 "Enables frequency-tuned PGD, Fourier attacks, stronger transferability, "
                                 "and patch survivability through reductions.",
                    exploitation_difficulty="Medium - requires frequency-aware perturbation design",
                    impact="Frequency attacks; patch attacks survive downsampling",
                    mitigation="BlurPool/anti-aliased downsampling. Move downsampling later. "
                              "Add low-pass filter before striding.",
                    references=["https://arxiv.org/abs/1904.11486"],  # Making CNNs Shift-Invariant
                    cvss_estimate=6.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        # Check for aggressive early pooling
        early_pools = [g for g in gadgets 
                      if g.gadget_type in [GadgetType.AMPLIFIER, GadgetType.SPATIAL_REDUCER] 
                      and g.position == "early"
                      and g.op_type in ["MaxPool", "AveragePool"]]
        
        if early_pools:
            large_pool = [g for g in early_pools if g.kernel_size and any(k >= 3 for k in g.kernel_size)]
            if large_pool:
                chains.append(Vulnerability(
                    id=f"CHAIN-EARLY-AGGPOOL",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.MEDIUM,
                    node_id=large_pool[0].node_id,
                    title=f"Aggressive Early Pooling ({large_pool[0].op_type} {large_pool[0].kernel_size})",
                    description=f"Large pooling operation in early layers removes spatial redundancy. "
                               f"Later stages depend on fewer samples.",
                    attack_vector="Localized perturbations can dominate when spatial redundancy is removed early. "
                                 "Enables patch attacks, sparse perturbations, translation sensitivity.",
                    exploitation_difficulty="Low",
                    impact="Increased vulnerability to localized/patch attacks",
                    mitigation="Smaller stride. Anti-aliasing. Training with translations/crops.",
                    references=[],
                    cvss_estimate=5.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        return chains
    
    def _find_maxpool_amplification_chains(self, gadgets: List[Gadget], adjacency: dict,
                                           gadget_map: dict) -> List[Vulnerability]:
        """C. Detect MaxPool spike amplification patterns."""
        chains = []
        
        maxpool_gadgets = [g for g in gadgets 
                         if g.gadget_type == GadgetType.AMPLIFIER and g.op_type == "MaxPool"]
        fusion_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.FUSION_POINT]
        fusion_ids = {g.node_id for g in fusion_gadgets}
        
        # MaxPool after fusion is especially dangerous
        for maxpool in maxpool_gadgets:
            # Check if any fusion point leads to this maxpool
            for fusion in fusion_gadgets:
                downstream = adjacency.get(fusion.node_id, [])
                # Check within 3 hops
                visited = set(downstream)
                for _ in range(2):
                    new_visited = set()
                    for d in visited:
                        new_visited.update(adjacency.get(d, []))
                    visited.update(new_visited)
                
                if maxpool.node_id in visited:
                    chains.append(Vulnerability(
                        id=f"CHAIN-FUSION-MAXPOOL-{fusion.node_id}",
                        category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                        severity=Severity.HIGH,
                        node_id=fusion.node_id,
                        title=f"Fusion → MaxPool Spike Amplification",
                        description=f"Fusion at '{fusion.node_id}' ({fusion.op_type}) followed by "
                                   f"MaxPool '{maxpool.node_id}'. Multi-branch perturbations fuse "
                                   f"then MaxPool selects the strongest spike.",
                        attack_vector="MaxPool selects extreme activations → adversarial spikes amplified. "
                                     "Sparse attacks (one-pixel-ish), localized patches, gradient-based "
                                     "'hot pixel' attacks all exploit this pattern.",
                        exploitation_difficulty="Low - well-documented pattern",
                        impact="Amplified adversarial effectiveness through spike selection",
                        mitigation="Replace MaxPool with AvgPool. Blur before pooling. "
                                  "Consider stochastic pooling for high-risk domains.",
                        references=["https://arxiv.org/abs/1710.08864"],
                        cvss_estimate=7.0,
                        finding_type=FindingType.ATTACK_CHAIN
                    ))
                    break  # Only report once per maxpool
        
        return chains
    
    def _find_skip_connection_highways(self, gadgets: List[Gadget],
                                       gadget_map: dict) -> List[Vulnerability]:
        """E. Detect skip connection gradient highways."""
        chains = []
        
        skip_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.SKIP_CONNECTION]
        
        if len(skip_gadgets) >= 3:
            chains.append(Vulnerability(
                id=f"CHAIN-SKIP-HIGHWAY",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM,
                node_id=skip_gadgets[0].node_id,
                title=f"Gradient Highway ({len(skip_gadgets)} skip connections)",
                description=f"Multiple skip/residual connections create gradient highways. "
                           f"First: '{skip_gadgets[0].node_id}'. Total: {len(skip_gadgets)}.",
                attack_vector="Skip paths are gradient highways - attacks optimize easily because "
                             "gradient signal stays stable with depth. Enables strong white-box "
                             "PGD/CW with faster convergence and stronger transfer.",
                exploitation_difficulty="Low - standard attack benefit",
                impact="Stronger white-box attacks; faster attack convergence",
                mitigation="Residual gating. Controlled noise on skip. Lipschitz constraints. "
                          "Activation/gradient norm clipping.",
                references=["https://arxiv.org/abs/1512.03385"],
                cvss_estimate=5.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        return chains
    
    def _find_normalization_issues(self, gadgets: List[Gadget], gadget_map: dict,
                                   adjacency: dict) -> List[Vulnerability]:
        """E. Detect normalization fragility patterns."""
        chains = []
        
        normalizers = [g for g in gadgets if g.gadget_type == GadgetType.NORMALIZER]
        bn_gadgets = [g for g in normalizers if "Batch" in g.op_type]
        
        if bn_gadgets:
            # BatchNorm present - flag distribution shift vulnerability
            chains.append(Vulnerability(
                id=f"CHAIN-BN-FRAGILITY",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.LOW,
                node_id=bn_gadgets[0].node_id,
                title=f"BatchNorm Distribution Shift Vulnerability ({len(bn_gadgets)} BN layers)",
                description=f"Model uses BatchNorm ({len(bn_gadgets)} layers). BN assumes stable "
                           f"activation distributions; adversarial inputs cause distribution shift.",
                attack_vector="Adversarial inputs shift channel distributions. BN can amplify small "
                             "shifts via scaling. Early layers especially vulnerable. "
                             "Robustness drops under domain shift.",
                exploitation_difficulty="Medium - requires understanding BN behavior",
                impact="Attack amplification through distribution shift; domain brittleness",
                mitigation="GroupNorm/RMSNorm instead of BatchNorm. Freeze BN stats at inference. "
                          "Adversarial BN training. Calibrate BN for deployment domain.",
                references=["https://arxiv.org/abs/2006.14536"],
                cvss_estimate=4.5,
                finding_type=FindingType.GADGET  # Individual BN is a gadget, not a vuln
            ))
        
        return chains
    
    def _find_large_kernel_chains(self, gadgets: List[Gadget], adjacency: dict,
                                  gadget_map: dict) -> List[Vulnerability]:
        """F. Detect large kernel after fusion patterns."""
        chains = []
        
        large_kernels = [g for g in gadgets if g.gadget_type == GadgetType.LARGE_KERNEL]
        fusion_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.FUSION_POINT]
        fusion_ids = {g.node_id for g in fusion_gadgets}
        
        for lk in large_kernels:
            # Check if this large kernel is after a fusion point
            for fusion in fusion_gadgets:
                downstream = adjacency.get(fusion.node_id, [])
                if lk.node_id in downstream or any(
                    lk.node_id in adjacency.get(d, []) for d in downstream
                ):
                    chains.append(Vulnerability(
                        id=f"CHAIN-FUSION-LARGEKERNEL-{lk.node_id}",
                        category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                        severity=Severity.MEDIUM,
                        node_id=lk.node_id,
                        title=f"Large Kernel After Fusion ({lk.kernel_size})",
                        description=f"Large {lk.kernel_size} kernel at '{lk.node_id}' follows "
                                   f"fusion at '{fusion.node_id}'. Fused perturbations spread "
                                   f"across wide receptive field.",
                        attack_vector="Large receptive field after fusion lets structured perturbations "
                                     "influence wide areas. Increases gradient steering region.",
                        exploitation_difficulty="Medium",
                        impact="Patch influence spreads; wider gradient steering",
                        mitigation="Replace with stacked small kernels. Anti-alias. Regularize filters.",
                        references=[],
                        cvss_estimate=5.0,
                        finding_type=FindingType.ATTACK_CHAIN
                    ))
                    break
        
        return chains
    
    def _find_reduction_survivability_chains(self, gadgets: List[Gadget], adjacency: dict,
                                             gadget_map: dict) -> List[Vulnerability]:
        """G. Detect reduction block survivability patterns."""
        chains = []
        
        # Find sequences of spatial reducers
        reducers = [g for g in gadgets if g.gadget_type in [GadgetType.SPATIAL_REDUCER, GadgetType.DOWNSAMPLER]]
        
        if len(reducers) >= 3:
            # Check for clustered reductions
            reducer_positions = sorted(set(g.position for g in reducers))
            
            chains.append(Vulnerability(
                id=f"CHAIN-REDUCTION-SURVIVE",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM,
                node_id=reducers[0].node_id,
                title=f"Reduction Block Survivability ({len(reducers)} reductions)",
                description=f"Model has {len(reducers)} spatial reduction operations. "
                           f"Hard spatial drops (e.g., 35→17→8) allow surviving attacks to dominate.",
                attack_vector="Attacks that survive reduction become dominant; aliasing persists. "
                             "Enables patch survivability, scale attacks, frequency attacks.",
                exploitation_difficulty="Medium - requires understanding reduction schedule",
                impact="Adversarial patterns that survive reductions gain disproportionate influence",
                mitigation="Anti-aliased reductions. Multi-scale supervision. Robust augmentation.",
                references=[],
                cvss_estimate=5.0,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        return chains
    
    def _find_head_sensitivity_chains(self, gadgets: List[Gadget], adjacency: dict,
                                      gadget_map: dict) -> List[Vulnerability]:
        """H. Detect classifier head sensitivity patterns."""
        chains = []
        
        # Find GAP → FC pattern
        gap_gadgets = [g for g in gadgets 
                     if g.op_type == "GlobalAveragePool" or 
                     (g.gadget_type == GadgetType.LINEAR_HEAD and "Global" in str(g.attributes))]
        
        linear_heads = [g for g in gadgets 
                       if g.gadget_type == GadgetType.LINEAR_HEAD and g.op_type in ["Gemm", "MatMul"]]
        
        if gap_gadgets and linear_heads:
            chains.append(Vulnerability(
                id=f"CHAIN-GAP-FC-HEAD",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM,
                node_id=gap_gadgets[0].node_id,
                title="GAP → FC Head Sensitivity",
                description=f"GlobalAveragePool at '{gap_gadgets[0].node_id}' followed by "
                           f"FC layer(s). After GAP, model is effectively a linear separator.",
                attack_vector="After GAP, model is linear in feature space. Small feature shifts "
                             "can flip logits. Enables feature-space attacks, universal perturbations, "
                             "logit margin attacks.",
                exploitation_difficulty="Low - standard attack pattern",
                impact="Feature-space attacks highly effective; logit manipulation",
                mitigation="Attention pooling instead of GAP. Feature denoising before GAP. "
                          "Larger margins. Logit pairing. Confidence calibration.",
                references=["https://arxiv.org/abs/1610.02136"],  # Universal perturbations
                cvss_estimate=5.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        return chains
    
    def _find_carrier_fusion_amplifier_chains(self, gadgets, adjacency, gadget_map):
        """Find Carrier → Fusion → Amplifier patterns."""
        chains = []
        
        fusion_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.FUSION_POINT]
        
        for fusion in fusion_gadgets:
            # Check downstream for amplifier
            downstream = adjacency.get(fusion.node_id, [])
            amplifier_after = None
            
            for d in downstream[:5]:  # Within 5 hops
                if d in gadget_map and gadget_map[d].gadget_type == GadgetType.AMPLIFIER:
                    amplifier_after = gadget_map[d]
                    break
            
            if amplifier_after and amplifier_after.op_type == "MaxPool":
                chains.append(Vulnerability(
                    id=f"CHAIN-CFA-{fusion.node_id}",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.HIGH,
                    node_id=fusion.node_id,
                    title="Fusion → MaxPool Amplification Chain",
                    description=f"Attack chain: Feature fusion at '{fusion.node_id}' followed by "
                               f"MaxPool '{amplifier_after.node_id}'. This pattern allows coordinated "
                               f"multi-branch perturbations to be fused and then amplified through "
                               f"max-selection, which picks the strongest adversarial spike.",
                    attack_vector="Multi-scale PGD: Craft perturbations for each input branch that "
                                 "combine constructively at fusion, then exploit MaxPool to select "
                                 "the strongest adversarial signal.",
                    exploitation_difficulty="Medium",
                    impact="Highly effective coordinated adversarial examples",
                    mitigation="Replace MaxPool with AvgPool after fusion. Add normalization between.",
                    references=["https://arxiv.org/abs/1705.07204"],
                    cvss_estimate=7.0,
                    finding_type=FindingType.ATTACK_CHAIN,
                    chainable_with=[fusion.id, amplifier_after.id if amplifier_after else ""]
                ))
        
        return chains
    
    def _find_multi_carrier_fusion_chains(self, gadgets, edges, gadget_map):
        """Find multiple carriers feeding into single fusion point."""
        chains = []
        
        # Build reverse adjacency
        reverse_adj = {}
        for src, dst in edges:
            if dst not in reverse_adj:
                reverse_adj[dst] = []
            reverse_adj[dst].append(src)
        
        fusion_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.FUSION_POINT]
        
        for fusion in fusion_gadgets:
            # Count carrier inputs
            inputs = reverse_adj.get(fusion.node_id, [])
            carrier_inputs = [inp for inp in inputs 
                            if inp in gadget_map and 
                            gadget_map[inp].gadget_type == GadgetType.PERTURBATION_CARRIER]
            
            if len(carrier_inputs) >= 3:  # 3+ branches is significant
                chains.append(Vulnerability(
                    id=f"CHAIN-MCF-{fusion.node_id}",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.MEDIUM,
                    node_id=fusion.node_id,
                    title=f"Multi-Branch Fusion ({len(carrier_inputs)} carriers)",
                    description=f"Fusion point '{fusion.node_id}' receives {len(carrier_inputs)} "
                               f"parallel carrier branches. Each branch independently carries "
                               f"perturbations that combine at this point.",
                    attack_vector="Coordinated multi-branch PGD: Optimize perturbations across all "
                                 f"{len(carrier_inputs)} branches simultaneously for maximum combined effect.",
                    exploitation_difficulty="Medium - requires multi-branch optimization",
                    impact="Additive combination of perturbations from multiple paths",
                    mitigation="Implement attention-weighted fusion. Add per-branch normalization.",
                    references=[],
                    cvss_estimate=5.5,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        return chains
    
    def _find_capacity_control_chains(self, gadgets, adjacency, gadget_map):
        """Find capacity reservoirs near control points (ShadowLogic risk)."""
        chains = []
        
        capacity_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.CAPACITY_RESERVOIR]
        control_gadgets = {g.node_id for g in gadgets if g.gadget_type == GadgetType.CONTROL_POINT}
        
        for cap in capacity_gadgets:
            # Check if any control point is within 3 hops
            visited = {cap.node_id}
            frontier = [cap.node_id]
            
            for _ in range(3):
                new_frontier = []
                for node in frontier:
                    for neighbor in adjacency.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_frontier.append(neighbor)
                            
                            if neighbor in control_gadgets:
                                chains.append(Vulnerability(
                                    id=f"CHAIN-SL-{cap.node_id}",
                                    category=ThreatCategory.SHADOWLOGIC_INJECTION,
                                    severity=Severity.HIGH,
                                    node_id=cap.node_id,
                                    title="Capacity + Control Chain (ShadowLogic Risk)",
                                    description=f"High-capacity layer '{cap.node_id}' ({cap.capacity_score:.0f}KB) "
                                               f"is within 3 hops of control point '{neighbor}'. "
                                               f"This proximity enables trigger-activated backdoor patterns.",
                                    attack_vector="ShadowLogic: Hide trigger-detection logic in capacity layer, "
                                                 "use control point to switch between benign and malicious behavior.",
                                    exploitation_difficulty="Medium - requires training access",
                                    impact="Complete model compromise via hidden backdoor",
                                    mitigation="Audit conditional operations. Apply fine-pruning. "
                                              "Monitor for unusual activation patterns.",
                                    references=["https://arxiv.org/abs/2212.02523"],
                                    cvss_estimate=8.0,
                                    finding_type=FindingType.ATTACK_CHAIN
                                ))
                                break
                frontier = new_frontier
        
        return chains
    
    def _find_gradient_highways(self, gadgets, adjacency, gadget_map):
        """Find long sequences of carriers without normalization (gradient highways)."""
        chains = []
        
        carrier_ids = {g.node_id for g in gadgets if g.gadget_type == GadgetType.PERTURBATION_CARRIER}
        normalizer_ids = {g.node_id for g in gadgets if g.gadget_type == GadgetType.NORMALIZER}
        
        # Find chains of carriers
        visited = set()
        
        for gadget in gadgets:
            if gadget.gadget_type != GadgetType.PERTURBATION_CARRIER:
                continue
            if gadget.node_id in visited:
                continue
            
            # Walk forward counting consecutive carriers
            chain_length = 1
            chain_start = gadget.node_id
            current = gadget.node_id
            visited.add(current)
            
            while True:
                downstream = adjacency.get(current, [])
                next_carrier = None
                
                for d in downstream:
                    if d in carrier_ids and d not in visited:
                        # Check if there's a normalizer between
                        if d not in normalizer_ids:
                            next_carrier = d
                            break
                    elif d in normalizer_ids:
                        # Chain broken by normalizer
                        break
                
                if next_carrier:
                    chain_length += 1
                    current = next_carrier
                    visited.add(current)
                else:
                    break
            
            if chain_length >= 4:  # 4+ consecutive carriers is notable
                chains.append(Vulnerability(
                    id=f"CHAIN-GH-{chain_start}",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.MEDIUM if chain_length < 6 else Severity.HIGH,
                    node_id=chain_start,
                    title=f"Gradient Highway ({chain_length} carriers)",
                    description=f"Chain of {chain_length} consecutive carrier layers starting at "
                               f"'{chain_start}' without intermediate normalization. "
                               f"Creates stable, predictable gradient flow ideal for attacks.",
                    attack_vector="Standard gradient attacks (FGSM, PGD, C&W) are highly effective "
                                 "due to clean gradient propagation through linear chain.",
                    exploitation_difficulty="Low - basic attacks work well",
                    impact="Easy-to-craft adversarial examples with high success rate",
                    mitigation="Add normalization between carrier layers. Consider spectral normalization.",
                    references=["https://arxiv.org/abs/1412.6572"],
                    cvss_estimate=6.0 if chain_length < 6 else 7.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        return chains
    
    def _find_early_amplifier_chains(self, gadgets, gadget_map):
        """Find amplifiers in early network positions."""
        chains = []
        
        early_amplifiers = [g for g in gadgets 
                          if g.gadget_type == GadgetType.AMPLIFIER and g.position == "early"]
        
        # Group by type
        maxpool_early = [g for g in early_amplifiers if g.op_type == "MaxPool"]
        
        if len(maxpool_early) >= 1:
            chains.append(Vulnerability(
                id=f"CHAIN-EA-maxpool",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM,
                node_id=maxpool_early[0].node_id,
                title=f"Early MaxPool Amplification ({len(maxpool_early)} instances)",
                description=f"Found {len(maxpool_early)} MaxPool operations in early network layers "
                           f"(first 20%). MaxPool amplifies the strongest signal, which for "
                           f"adversarial inputs means amplifying adversarial perturbations early.",
                attack_vector="Sparse adversarial attacks (one-pixel, patch attacks) can exploit "
                             "early MaxPool to amplify localized perturbations before they can be "
                             "diluted by later processing.",
                exploitation_difficulty="Low - well-documented attack patterns",
                impact="Increased effectiveness of sparse/patch attacks",
                mitigation="Replace early MaxPool with AvgPool or BlurPool.",
                references=["https://arxiv.org/abs/1710.08864"],
                cvss_estimate=5.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        return chains
    
    def _find_research_based_chains(self, gadgets: List[Gadget], 
                                   gadget_map: dict) -> List[Vulnerability]:
        """
        Detect attack chains based on Phase 1 research findings.
        
        These chains are directly mapped to documented attack techniques:
        - GoogleAp, LaVAN: Patch attacks exploiting GAP->FC
        - EOT, RP2: Physical attacks exploiting aliasing
        - DPATCH: Multi-scale attacks exploiting high-fanin fusion
        """
        chains = []
        
        # Gather research-based gadget types
        gap_fc_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.GAP_FC_HEAD]
        aliasing_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.ALIASING_DOWNSAMPLE]
        maxpool_fusion_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.MAXPOOL_AFTER_FUSION]
        high_fanin_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.HIGH_FANIN_FUSION]
        
        # Check for attention gadgets
        no_attention_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.NO_SPATIAL_ATTENTION]
        has_attention_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.HAS_SPATIAL_ATTENTION]
        
        # === CHAIN 1: Classic Patch Attack Vulnerability (GoogleAp, LaVAN) ===
        if gap_fc_gadgets:
            # Determine severity based on attention presence
            if no_attention_gadgets:
                # No attention + GAP-FC = CRITICAL vulnerability
                severity = Severity.CRITICAL
                cvss = 8.5
                attention_note = "AGGRAVATING FACTOR: Model has NO spatial attention mechanisms. " \
                                "All spatial locations contribute equally, making patches maximally effective."
            elif has_attention_gadgets:
                # Has attention + GAP-FC = MEDIUM vulnerability (attention mitigates)
                severity = Severity.MEDIUM
                cvss = 5.5
                attention_note = "MITIGATING FACTOR: Model has spatial attention mechanisms that may " \
                                "help filter anomalous patch regions, reducing attack effectiveness."
            else:
                # Unknown attention status = HIGH (default)
                severity = Severity.HIGH
                cvss = 7.5
                attention_note = ""
            
            chains.append(Vulnerability(
                id="CHAIN-PATCH-ATTACK-SURFACE",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=severity,
                node_id=gap_fc_gadgets[0].node_id,
                title="Patch Attack Vulnerability (GoogleAp/LaVAN pattern)",
                description=f"Model has GlobalPool->FC classifier pattern at '{gap_fc_gadgets[0].node_id}'. "
                           f"Research (Brown et al. 2017, Karmon et al. 2018) shows this is the canonical "
                           f"architecture vulnerability exploited by adversarial patch attacks. {attention_note}",
                attack_vector="Adversarial patches (printed images, stickers) can cause targeted misclassification. "
                             "Global pooling aggregates patch features into final representation, "
                             "allowing small localized perturbations to dominate classification.",
                exploitation_difficulty="Low - patches can be printed and placed anywhere in scene",
                impact=f"{'CRITICAL' if severity == Severity.CRITICAL else 'HIGH' if severity == Severity.HIGH else 'MEDIUM'} - "
                       f"Physical-world targeted attacks {'highly feasible' if no_attention_gadgets else 'possible'} with printed patches",
                mitigation="1. Add spatial attention before pooling to filter anomalous regions. "
                          "2. Use attention-based pooling instead of GAP. "
                          "3. Implement patch detection in preprocessing. "
                          "4. Train with adversarial patch augmentation.",
                references=[
                    "https://arxiv.org/abs/1712.09665",  # Adversarial Patch (GoogleAp)
                    "https://arxiv.org/abs/1801.02608",  # LaVAN
                ],
                cvss_estimate=cvss,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # === CHAIN 2: Physical World Attack Vulnerability (EOT, RP2) ===
        if aliasing_gadgets:
            chains.append(Vulnerability(
                id="CHAIN-PHYSICAL-WORLD-ATTACK",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.HIGH,
                node_id=aliasing_gadgets[0].node_id,
                title=f"Physical-World Attack Vulnerability ({len(aliasing_gadgets)} aliasing points)",
                description=f"Model has {len(aliasing_gadgets)} early stride-2 convolutions without anti-aliasing. "
                           f"Research (Athalye et al. 2018, Eykholt et al. 2018) shows this enables attacks "
                           f"that survive physical-world transformations (rotation, scaling, lighting).",
                attack_vector="EOT-optimized adversarial examples remain effective when printed, photographed, "
                             "or viewed from different angles. High-frequency perturbations fold into lower "
                             "frequencies during aliased downsampling, persisting through the network.",
                exploitation_difficulty="Medium - requires EOT optimization but well-documented",
                impact="HIGH - Adversarial objects (3D printed, stickers on signs) work in real world",
                mitigation="1. Add BlurPool or anti-aliased downsampling before strided operations. "
                          "2. Move aggressive downsampling later in network. "
                          "3. Train with transformation augmentation.",
                references=[
                    "https://arxiv.org/abs/1707.07397",  # EOT
                    "https://arxiv.org/abs/1707.08945",  # RP2
                ],
                cvss_estimate=7.0,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # === CHAIN 3: Amplified Multi-Scale Attack (DPATCH-style) ===
        if maxpool_fusion_gadgets or (high_fanin_gadgets and any(
            g.gadget_type == GadgetType.AMPLIFIER for g in gadgets)):
            
            relevant_gadgets = maxpool_fusion_gadgets or high_fanin_gadgets
            chains.append(Vulnerability(
                id="CHAIN-AMPLIFIED-MULTISCALE-ATTACK",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.HIGH,
                node_id=relevant_gadgets[0].node_id if relevant_gadgets else "unknown",
                title="Amplified Multi-Scale Attack Surface",
                description=f"Model combines multi-branch fusion with MaxPool amplification. "
                           f"Research (Liu et al. 2019 - DPATCH) shows this pattern enables "
                           f"attacks that are effective across multiple detection scales.",
                attack_vector="Adversarial patches optimized across multiple branches are amplified "
                             "by MaxPool, creating robust attacks that transfer across scales and models.",
                exploitation_difficulty="Medium",
                impact="HIGH - Attacks effective across object detector scales",
                mitigation="1. Replace MaxPool with AvgPool after fusion. "
                          "2. Add gated fusion (channel-wise weights). "
                          "3. Implement branch dropout during training.",
                references=[
                    "https://arxiv.org/abs/1806.02299",  # DPATCH
                ],
                cvss_estimate=6.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # === CHAIN 4: Combined High-Risk Pattern ===
        # If model has BOTH GAP-FC AND aliasing - compound vulnerability
        if gap_fc_gadgets and aliasing_gadgets:
            chains.append(Vulnerability(
                id="CHAIN-COMPOUND-PHYSICAL-PATCH",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.CRITICAL,
                node_id=gap_fc_gadgets[0].node_id,
                title="Compound Physical Attack Vulnerability",
                description="CRITICAL: Model combines GAP->FC pattern with aliasing vulnerabilities. "
                           "This enables physical-world patch attacks with EOT optimization that "
                           "are robust to real-world conditions AND exploit the linear classifier head.",
                attack_vector="Physical patches (printed, stickers) optimized with EOT will be highly "
                             "effective. Attacks survive transformations AND exploit GAP aggregation.",
                exploitation_difficulty="Low - combines multiple well-documented attack techniques",
                impact="CRITICAL - Robust physical-world targeted attacks highly feasible",
                mitigation="Address both vulnerabilities: "
                          "1. Add anti-aliasing (BlurPool) "
                          "2. Replace GAP with attention pooling "
                          "3. Add patch detection preprocessing",
                references=[
                    "https://arxiv.org/abs/1712.09665",
                    "https://arxiv.org/abs/1707.07397",
                ],
                cvss_estimate=8.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # ========== PHASE 2: OBJECT DETECTOR ATTACK CHAINS ==========
        # Research basis: Adversarial YOLO, ShapeShifter, UPC, CAMOU
        
        # Gather object detector gadget types
        objectness_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.OBJECTNESS_HEAD]
        anchor_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.ANCHOR_BASED_DETECTION]
        fpn_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.FPN_STRUCTURE]
        rpn_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.TWO_STAGE_RPN]
        nms_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.NMS_DEPENDENCY]
        detection_head_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.DETECTION_HEAD_PATTERN]
        
        # Check if this appears to be an object detector
        is_detector = bool(objectness_gadgets or anchor_gadgets or rpn_gadgets or nms_gadgets or detection_head_gadgets)
        
        if is_detector:
            # === CHAIN 5: Object Disappearance Attack (Adversarial YOLO) ===
            if objectness_gadgets:
                chains.append(Vulnerability(
                    id="CHAIN-OBJECT-DISAPPEARANCE",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.CRITICAL,
                    node_id=objectness_gadgets[0].node_id,
                    title="Object Disappearance Attack Vulnerability (Adversarial YOLO)",
                    description=f"Detector has objectness scoring at '{objectness_gadgets[0].node_id}'. "
                               f"Research (Thys et al. 2019 - Adversarial YOLO) shows objectness scores "
                               f"can be directly suppressed, making objects completely INVISIBLE to detection.",
                    attack_vector="Adversarial patches/clothing suppress objectness confidence to zero. "
                                 "Unlike classifiers (wrong label), detectors output NO DETECTION. "
                                 "This is MORE DANGEROUS for safety systems (surveillance, AV).",
                    exploitation_difficulty="Medium - requires objectness-aware optimization",
                    impact="CRITICAL - Complete detection evasion, objects become invisible",
                    mitigation="1. Multi-scale objectness with redundancy. "
                              "2. Attention-based verification. "
                              "3. Minimum objectness threshold training. "
                              "4. Adversarial training with objectness attacks.",
                    references=[
                        "https://arxiv.org/abs/1904.08653",  # Fooling automated surveillance cameras
                        "https://arxiv.org/abs/1712.02494",  # Adversarial YOLO
                    ],
                    cvss_estimate=9.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
            
            # === CHAIN 6: Two-Stage RPN Attack (ShapeShifter) ===
            if rpn_gadgets:
                chains.append(Vulnerability(
                    id="CHAIN-RPN-PROPOSAL-ATTACK",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.CRITICAL,
                    node_id=rpn_gadgets[0].node_id,
                    title="RPN Proposal Suppression Attack (ShapeShifter)",
                    description=f"Two-stage detector with ROI operation at '{rpn_gadgets[0].node_id}'. "
                               f"Research (Chen et al. 2019 - ShapeShifter) shows attacking the Region Proposal "
                               f"Network causes complete object disappearance - no proposals = no detections.",
                    attack_vector="Adversarial textures suppress region proposals before classification stage. "
                                 "Attack on RPN is more efficient than attacking classifier - single point of failure.",
                    exploitation_difficulty="Medium - requires RPN-aware optimization",
                    impact="CRITICAL - Complete detection bypass at proposal stage",
                    mitigation="1. Multiple proposal mechanisms with redundancy. "
                              "2. Attention-based proposals. "
                              "3. Proposal verification network. "
                              "4. Adversarial training with proposal attacks.",
                    references=[
                        "https://arxiv.org/abs/1804.05810",  # ShapeShifter
                    ],
                    cvss_estimate=8.5,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
            
            # === CHAIN 7: Anchor-Based Detection Vulnerability ===
            if anchor_gadgets:
                chains.append(Vulnerability(
                    id="CHAIN-ANCHOR-EXPLOITATION",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.HIGH,
                    node_id=anchor_gadgets[0].node_id,
                    title="Anchor-Based Detection Vulnerability",
                    description=f"Detector uses anchor-based outputs at '{anchor_gadgets[0].node_id}'. "
                               f"Fixed anchor grids provide PREDICTABLE attack targets. "
                               f"Research: DPATCH, Adversarial YOLO exploit anchor priors.",
                    attack_vector="Attacker can predict which anchor locations will be activated for "
                                 "specific object sizes/positions. Enables targeted anchor suppression "
                                 "and scale-specific evasion attacks.",
                    exploitation_difficulty="Medium",
                    impact="HIGH - Predictable detection points enable targeted attacks",
                    mitigation="1. Anchor-free detection (FCOS, CenterNet). "
                              "2. Dynamic anchor learning. "
                              "3. Multi-anchor redundancy.",
                    references=[
                        "https://arxiv.org/abs/1806.02299",  # DPATCH
                    ],
                    cvss_estimate=7.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
            
            # === CHAIN 8: Multi-Scale FPN Attack Surface ===
            if fpn_gadgets and len(fpn_gadgets) >= 2:
                chains.append(Vulnerability(
                    id="CHAIN-FPN-MULTISCALE-ATTACK",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.HIGH,
                    node_id=fpn_gadgets[0].node_id,
                    title=f"Multi-Scale FPN Attack Surface ({len(fpn_gadgets)} pyramid levels)",
                    description=f"Feature Pyramid Network with {len(fpn_gadgets)} levels detected. "
                               f"Each FPN level is an independent attack entry point.",
                    attack_vector="Multi-scale attacks can enter at any pyramid level. "
                                 "ShapeShifter and DPATCH exploit FPN structure for scale-invariant attacks. "
                                 "Perturbations optimized across scales are more robust.",
                    exploitation_difficulty="Medium",
                    impact="HIGH - Multiple attack surfaces enable scale-invariant evasion",
                    mitigation="1. Cross-scale consistency loss. "
                              "2. Scale-aware adversarial training. "
                              "3. Feature pyramid attention.",
                    references=[
                        "https://arxiv.org/abs/1804.05810",  # ShapeShifter
                    ],
                    cvss_estimate=6.5,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
            
            # === CHAIN 9: NMS Manipulation Vulnerability ===
            if nms_gadgets:
                chains.append(Vulnerability(
                    id="CHAIN-NMS-MANIPULATION",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.MEDIUM,
                    node_id=nms_gadgets[0].node_id,
                    title="NMS Confidence Manipulation Vulnerability",
                    description=f"Detector uses Non-Maximum Suppression at '{nms_gadgets[0].node_id}'. "
                               f"NMS can be exploited via confidence manipulation.",
                    attack_vector="1. Suppress true positives by lowering confidence below NMS threshold. "
                                 "2. Inject false positives with high confidence. "
                                 "3. Manipulate IoU to cause incorrect suppression. "
                                 "NMS is a deterministic post-process that can be fooled.",
                    exploitation_difficulty="Medium - requires understanding NMS behavior",
                    impact="MEDIUM - Detection quality degradation, false positive/negative manipulation",
                    mitigation="1. Soft-NMS instead of hard NMS. "
                              "2. Confidence calibration. "
                              "3. IoU prediction verification.",
                    references=[],
                    cvss_estimate=5.5,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
            
            # === CHAIN 10: Combined Detector Vulnerability (UPC/CAMOU style) ===
            # If detector has both objectness AND aliasing - physical-world person/vehicle hiding
            if objectness_gadgets and aliasing_gadgets:
                chains.append(Vulnerability(
                    id="CHAIN-PHYSICAL-DETECTOR-EVASION",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.CRITICAL,
                    node_id=objectness_gadgets[0].node_id,
                    title="Physical-World Detector Evasion (UPC/CAMOU pattern)",
                    description="CRITICAL: Detector combines objectness scoring with aliasing vulnerabilities. "
                               "This enables physical-world detector evasion - adversarial clothing/textures "
                               "that make people or vehicles INVISIBLE to detection systems.",
                    attack_vector="Adversarial patterns on clothing (UPC) or vehicle textures (CAMOU) "
                                 "suppress objectness. EOT optimization ensures patterns survive real-world "
                                 "transformations. Proven effective against person and vehicle detectors.",
                    exploitation_difficulty="Medium - requires EOT + objectness-aware optimization",
                    impact="CRITICAL - Physical-world invisibility to automated surveillance/AV systems",
                    mitigation="1. Multi-modal detection (combine with radar, lidar). "
                              "2. Anti-aliasing + objectness training. "
                              "3. Texture anomaly detection. "
                              "4. Adversarial training with physical attacks.",
                    references=[
                        "https://arxiv.org/abs/1910.11099",  # UPC - adversarial clothing
                        "https://arxiv.org/abs/1809.09575",  # CAMOU - vehicle textures
                    ],
                    cvss_estimate=9.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        # ========== PHASE 3: NEW ATTACK CHAINS ==========
        # Research basis: Papers [103], [111], [114]
        
        # Gather Phase 3 gadget types
        vit_patch_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.VIT_PATCH_EMBEDDING]
        unreg_attention_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.UNREGULARIZED_ATTENTION]
        cls_token_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.CLS_TOKEN_AGGREGATION]
        aggressive_downsample_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.AGGRESSIVE_EARLY_DOWNSAMPLING]
        
        # === CHAIN 11: ViT Patch Attack Vulnerability ===
        # Pattern: VIT_PATCH_EMBEDDING + (UNREGULARIZED_ATTENTION or CLS_TOKEN_AGGREGATION)
        if vit_patch_gadgets:
            has_attention_vuln = bool(unreg_attention_gadgets)
            has_cls_vuln = bool(cls_token_gadgets)
            
            if has_attention_vuln or has_cls_vuln:
                vuln_components = []
                if has_attention_vuln:
                    vuln_components.append(f"{len(unreg_attention_gadgets)} unregularized attention blocks")
                if has_cls_vuln:
                    vuln_components.append("CLS token aggregation")
                
                chains.append(Vulnerability(
                    id="CHAIN-VIT-PATCH-ATTACK",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.CRITICAL,
                    node_id=vit_patch_gadgets[0].node_id,
                    title="Vision Transformer Patch Attack Vulnerability",
                    description=f"CRITICAL: ViT architecture with patch embedding at '{vit_patch_gadgets[0].node_id}' "
                               f"combined with {', '.join(vuln_components)}. "
                               f"Research [103][114] shows ViTs are vulnerable to patch attacks through attention hijacking.",
                    attack_vector="Adversarial patches can hijack attention to focus on malicious content. "
                                 "Patch embedding provides single entry point with no spatial redundancy. "
                                 "CLS token aggregates ALL patch information without spatial filtering. "
                                 "Analogous to GAP→FC vulnerability in CNNs, but exploits attention mechanism.",
                    exploitation_difficulty="Medium - requires understanding of attention patterns",
                    impact="CRITICAL - Patch attacks highly effective against ViT classifiers",
                    mitigation="1. Add dropout after attention layers (regularization). "
                              "2. Use attention-based patch detection. "
                              "3. Adversarial training with patch augmentation. "
                              "4. Consider robust ViT variants (e.g., with local attention).",
                    references=[
                        "https://arxiv.org/abs/2408.17059",  # [103] ViT SSL Survey
                        "https://arxiv.org/abs/2410.01574",  # [114] AI-Generated Image Detectors
                    ],
                    cvss_estimate=8.5,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        # === CHAIN 12: Adversarial Training Resistance Pattern ===
        # Pattern: HIGH_FANIN_FUSION (3+) + SKIP_CONNECTION (3+)
        # Research basis: Paper [111] - AT less effective for multi-branch architectures
        skip_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.SKIP_CONNECTION]
        
        if len(high_fanin_gadgets) >= 3 and len(skip_gadgets) >= 3:
            chains.append(Vulnerability(
                id="CHAIN-AT-RESISTANT-ARCHITECTURE",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.HIGH,
                node_id=high_fanin_gadgets[0].node_id,
                title="Adversarial Training Resistance Pattern",
                description=f"Architecture combines {len(high_fanin_gadgets)} multi-branch fusions with "
                           f"{len(skip_gadgets)} long skip connections. Research [111] (2025) shows these "
                           f"patterns create gradient highways that reduce adversarial training effectiveness by 15-30%.",
                attack_vector="Multi-branch architectures provide multiple attack paths that are hard to defend simultaneously. "
                             "Skip connections create gradient highways that bypass adversarial training defenses. "
                             "Standard AT may give false sense of security for this architecture.",
                exploitation_difficulty="Low - standard PGD attacks remain effective post-AT",
                impact="HIGH - Adversarial training defense less effective than expected",
                mitigation="1. Consider certified defenses instead of standard AT. "
                          "2. Architecture simplification (reduce branches/skips). "
                          "3. Gradient masking with careful implementation. "
                          "4. Ensemble methods with diverse architectures. "
                          "5. Input preprocessing defenses.",
                references=[
                    "https://arxiv.org/abs/2506.18516",  # [111] DUMB and DUMBer
                ],
                cvss_estimate=7.0,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # === CHAIN 13: Small Object Vulnerability ===
        # Pattern: AGGRESSIVE_EARLY_DOWNSAMPLING + (ALIASING or OBJECTNESS_HEAD)
        if aggressive_downsample_gadgets:
            if aliasing_gadgets or objectness_gadgets:
                target_type = "object detection" if objectness_gadgets else "classification"
                chains.append(Vulnerability(
                    id="CHAIN-SMALL-OBJECT-SENSITIVITY",
                    category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                    severity=Severity.HIGH,
                    node_id=aggressive_downsample_gadgets[0].node_id,
                    title=f"Small Object Attack Sensitivity ({target_type})",
                    description=f"Model has aggressive early downsampling ({aggressive_downsample_gadgets[0].attributes.get('stride2_count', 3)} "
                               f"stride-2 ops) combined with aliasing vulnerabilities. Research [107] shows this "
                               f"enables smaller adversarial patches and increases physical attack effectiveness.",
                    attack_vector="Early spatial reduction destroys small object information AND reduces "
                                 "minimum effective patch size for attacks. Attackers need smaller patches "
                                 "to achieve same effect. Physical attacks on distant/small objects easier.",
                    exploitation_difficulty="Low - smaller patches are easier to deploy physically",
                    impact="HIGH - Small physical patches highly effective; small object detection compromised",
                    mitigation="1. Reduce early downsampling (move stride-2 later). "
                              "2. Add anti-aliasing before all stride-2 operations. "
                              "3. Multi-scale feature preservation. "
                              "4. Attention-based spatial filtering.",
                    references=[
                        "https://arxiv.org/abs/2503.04452",  # [107] FDM-YOLO
                    ],
                    cvss_estimate=7.0,
                    finding_type=FindingType.ATTACK_CHAIN
                ))
        
        # === CHAIN 14: Audio Adversarial Attack Surface ===
        # Pattern: AUDIO_MEL_INPUT + AUDIO_STRIDE_DOWNSAMPLE + AUDIO_TEMPORAL_ATTENTION
        # Research basis: Carlini Audio 2018, CommanderSong 2018, DeepPayload 2021
        audio_mel_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.AUDIO_MEL_INPUT]
        audio_stride_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.AUDIO_STRIDE_DOWNSAMPLE]
        audio_attention_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.AUDIO_TEMPORAL_ATTENTION]
        
        if audio_mel_gadgets:
            has_stride = len(audio_stride_gadgets) > 0
            has_attention = len(audio_attention_gadgets) > 0
            
            severity = Severity.CRITICAL if (has_stride and has_attention) else Severity.HIGH
            
            vuln_components = ["mel-spectrogram input"]
            if has_stride:
                vuln_components.append("aliasing downsampling")
            if has_attention:
                vuln_components.append(f"{len(audio_attention_gadgets)} temporal attention layers")
            
            chains.append(Vulnerability(
                id="CHAIN-AUDIO-ADVERSARIAL",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=severity,
                node_id=audio_mel_gadgets[0].node_id,
                title="Audio Adversarial Attack Vulnerability",
                description=f"Audio model architecture with {', '.join(vuln_components)}. "
                           f"Research (Carlini 2018, CommanderSong 2018) shows imperceptible audio perturbations "
                           f"can force incorrect transcriptions or inject hidden voice commands.",
                attack_vector="Adversarial audio can be crafted using psychoacoustic masking (inaudible to humans). "
                             "Stride downsampling causes aliasing that helps perturbations survive processing. "
                             "Self-attention spreads perturbations across all timesteps.",
                exploitation_difficulty="Medium - requires audio domain expertise but tools exist (ART, speechbrain)",
                impact="HIGH - Hidden voice commands, incorrect transcriptions, ASR bypass",
                mitigation="1. Add input audio normalization and filtering. "
                          "2. Anti-aliasing before downsampling. "
                          "3. Adversarial training with audio augmentation. "
                          "4. Certified defenses for audio (e.g., randomized smoothing). "
                          "5. Multi-model verification for critical commands.",
                references=[
                    "https://arxiv.org/abs/1801.01944",  # Carlini Audio Adversarial 2018
                    "https://arxiv.org/abs/1801.00554",  # CommanderSong 2018
                ],
                cvss_estimate=8.0 if severity == Severity.CRITICAL else 7.0,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # === CHAIN 15: Audio Cross-Modal Attack ===
        # Pattern: CROSS_MODAL_ATTENTION + (AUDIO_TEMPORAL_ATTENTION or UNREGULARIZED_ATTENTION)
        # Research basis: Image Hijacks 2023, Rickrolling the Artist 2022 (adapted for audio)
        cross_modal_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.CROSS_MODAL_ATTENTION]
        unreg_attention_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.UNREGULARIZED_ATTENTION]
        
        if cross_modal_gadgets:
            has_audio_attention = len(audio_attention_gadgets) > 0
            has_unreg_attention = len(unreg_attention_gadgets) > 0
            
            chains.append(Vulnerability(
                id="CHAIN-AUDIO-CROSS-MODAL",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.CRITICAL,
                node_id=cross_modal_gadgets[0].node_id,
                title="Audio Cross-Modal Injection Vulnerability",
                description=f"Encoder-decoder architecture with {len(cross_modal_gadgets)} cross-attention layers. "
                           f"Cross-attention allows adversarial audio to directly influence text generation. "
                           f"Research (Image Hijacks 2023) shows cross-modal attention is exploitable for output control.",
                attack_vector="Adversarial audio perturbations corrupt encoder hidden states. "
                             "Cross-attention transfers corrupted features to decoder. "
                             "Decoder text generation hijacked by adversarial audio content. "
                             "Can force specific transcriptions regardless of actual speech content.",
                exploitation_difficulty="Medium - requires audio/text alignment understanding",
                impact="CRITICAL - Attacker can inject arbitrary text via adversarial audio",
                mitigation="1. Validate encoder outputs before cross-attention. "
                          "2. Consistency checking between audio and text. "
                          "3. Anomaly detection on attention patterns. "
                          "4. Multi-pass verification for security-critical transcriptions. "
                          "5. Rate limiting and logging for unusual transcription patterns.",
                references=[
                    "https://arxiv.org/abs/2309.00236",  # Image Hijacks 2023
                    "https://arxiv.org/abs/2211.02408",  # Rickrolling the Artist 2022
                ],
                cvss_estimate=9.0,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        # === CHAIN 16: Audio ShadowLogic Susceptibility ===
        # Pattern: CROSS_MODAL_ATTENTION + seq2seq architecture
        # Research basis: ShadowLogic 2025, DeepPayload 2021
        seq2seq_gadgets = [g for g in gadgets if g.gadget_type == GadgetType.ENCODER_DECODER_SEQ2SEQ]
        
        if seq2seq_gadgets and cross_modal_gadgets:
            chains.append(Vulnerability(
                id="CHAIN-AUDIO-SHADOWLOGIC",
                category=ThreatCategory.SHADOWLOGIC_INJECTION,
                severity=Severity.CRITICAL,
                node_id=cross_modal_gadgets[0].node_id,
                title="Audio Model ShadowLogic Susceptibility",
                description="Encoder-decoder ASR architecture is susceptible to ShadowLogic backdoor injection. "
                           "Cross-attention provides ideal injection point for trigger detection and output override. "
                           "Attacker can inject nodes that monitor for audio trigger and force specific transcription.",
                attack_vector="INJECTION SCENARIO: "
                             "1. Inject trigger detector monitoring encoder hidden states for specific audio pattern. "
                             "2. Inject output override that bypasses normal cross-attention when trigger detected. "
                             "3. Force decoder to output attacker-controlled text (commands, credentials, etc.). "
                             "4. Backdoor persists through fine-tuning and format conversion.",
                exploitation_difficulty="Medium - requires ONNX graph manipulation skills",
                impact="CRITICAL - Persistent backdoor allows arbitrary text injection via audio trigger",
                mitigation="1. Cryptographic model signing and verification. "
                          "2. Regular graph structure auditing. "
                          "3. Behavioral testing with diverse audio inputs. "
                          "4. Monitor for unexpected conditional operations in graph. "
                          "5. Supply chain security for model sources.",
                references=[
                    "https://hiddenlayer.com/innovation-hub/shadowlogic/",
                    "https://arxiv.org/abs/2106.04690",  # DeepPayload 2021
                ],
                cvss_estimate=9.5,
                finding_type=FindingType.ATTACK_CHAIN
            ))
        
        return chains
    
    def summarize_gadgets(self, gadgets: List[Gadget]) -> Dict[str, Any]:
        """
        Generate summary of attack-relevant gadgets.
        
        Maps gadgets to the attack techniques they enable.
        """
        summary = {
            "total_gadgets": len(gadgets),
            "by_type": {},
            "by_position": {"early": 0, "middle": 0, "late": 0},
            
            # Attack technique enablers (research-based categories)
            "attack_enablers": {
                "sparse_patch_attacks": [],      # MaxPool locations
                "multi_scale_attacks": [],       # Fusion points (Concat)
                "frequency_attacks": [],         # Unprotected stride-2
                "gradient_highway_attacks": [],  # Long skip connections
                "feature_space_attacks": [],     # GAP, final FC
                "backdoor_potential": [],        # Conditionals
                "extraction_surface": [],        # Softmax
                # Phase 1 research-based categories
                "physical_world_attacks": [],    # Aliasing downsampling (EOT, RP2)
                "patch_attacks_high_risk": [],   # GAP->FC pattern (GoogleAp, LaVAN)
                "amplified_patch_attacks": [],   # MaxPool after fusion
                "high_fanin_attacks": [],        # High fan-in Concat (>3 branches)
                # Phase 2: Object detector attack categories
                "objectness_attacks": [],        # Adversarial YOLO - objectness suppression
                "anchor_attacks": [],            # Anchor-based detection exploitation
                "fpn_attacks": [],               # FPN multi-scale attack surface
                "rpn_attacks": [],               # ShapeShifter - proposal suppression
                "nms_attacks": [],               # NMS confidence manipulation
                "detector_evasion": [],          # General detector evasion surface
                # Attention-related categories
                "no_attention_vulnerability": [],  # Missing attention - patch attacks enabled
                # Phase 3: ViT and advanced CNN categories
                "vit_vulnerabilities": [],        # ViT-specific attack surfaces
                "at_resistance_indicators": [],   # Patterns that resist adversarial training
                # Phase 5: Audio model attack categories
                "audio_adversarial": [],          # Audio adversarial attack surfaces
                "audio_cross_modal": [],          # Cross-modal audio->text injection
                "audio_temporal": [],             # Temporal attention vulnerabilities
            },
            
            # Defensive indicators
            "defensive_features": {
                "spatial_attention": [],         # Attention mechanisms that help defend
            },
            
            # Architecture type indicator
            "architecture_type": "unknown",  # "cnn", "vit", "detector", "hybrid"
            
            # Detailed locations
            "critical_locations": [],  # Gadgets that enable multiple attack types
        }
        
        for gadget in gadgets:
            # Count by type
            type_name = gadget.gadget_type.value
            if type_name not in summary["by_type"]:
                summary["by_type"][type_name] = 0
            summary["by_type"][type_name] += 1
            
            # Count by position
            if gadget.position in summary["by_position"]:
                summary["by_position"][gadget.position] += 1
            
            # Map to attack techniques
            entry = {
                "node": gadget.node_id,
                "type": gadget.op_type,
                "position": gadget.position,
                "contribution": gadget.attack_contribution[:100] if gadget.attack_contribution else ""
            }
            
            if gadget.gadget_type == GadgetType.AMPLIFIER:
                summary["attack_enablers"]["sparse_patch_attacks"].append(entry)
                # MaxPool after fusion is critical
                if gadget.attributes.get("after_fusion"):
                    summary["critical_locations"].append({
                        **entry,
                        "reason": "MaxPool after fusion - amplifies combined perturbations"
                    })
            
            elif gadget.gadget_type == GadgetType.FUSION_POINT:
                entry["branches"] = gadget.attributes.get("num_branches", 2)
                summary["attack_enablers"]["multi_scale_attacks"].append(entry)
            
            elif gadget.gadget_type == GadgetType.DOWNSAMPLER:
                summary["attack_enablers"]["frequency_attacks"].append(entry)
            
            elif gadget.gadget_type == GadgetType.SKIP_CONNECTION:
                entry["skip_distance"] = gadget.attributes.get("skip_distance", 0)
                summary["attack_enablers"]["gradient_highway_attacks"].append(entry)
            
            elif gadget.gadget_type == GadgetType.LINEAR_HEAD:
                summary["attack_enablers"]["feature_space_attacks"].append(entry)
            
            elif gadget.gadget_type == GadgetType.CONTROL_POINT:
                summary["attack_enablers"]["backdoor_potential"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Conditional op - potential ShadowLogic trigger"
                })
            
            elif gadget.gadget_type == GadgetType.EXTRACTION_SURFACE:
                summary["attack_enablers"]["extraction_surface"].append(entry)
            
            elif gadget.gadget_type == GadgetType.LARGE_KERNEL:
                if gadget.attributes.get("after_fusion"):
                    summary["critical_locations"].append({
                        **entry,
                        "reason": f"Large kernel {gadget.kernel_size} after fusion"
                    })
            
            # New research-based gadget types
            elif gadget.gadget_type == GadgetType.ALIASING_DOWNSAMPLE:
                summary["attack_enablers"]["physical_world_attacks"].append(entry)
                summary["attack_enablers"]["frequency_attacks"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Early stride-2 without anti-aliasing - EOT/RP2 vulnerability"
                })
            
            elif gadget.gadget_type == GadgetType.GAP_FC_HEAD:
                summary["attack_enablers"]["patch_attacks_high_risk"].append(entry)
                summary["attack_enablers"]["feature_space_attacks"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "GAP->FC pattern - canonical patch attack vulnerability (GoogleAp)"
                })
            
            elif gadget.gadget_type == GadgetType.MAXPOOL_AFTER_FUSION:
                summary["attack_enablers"]["amplified_patch_attacks"].append(entry)
                summary["attack_enablers"]["sparse_patch_attacks"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "MaxPool after Concat - amplifies multi-scale patch attacks"
                })
            
            elif gadget.gadget_type == GadgetType.HIGH_FANIN_FUSION:
                entry["branches"] = gadget.attributes.get("num_branches", 0)
                summary["attack_enablers"]["high_fanin_attacks"].append(entry)
                summary["attack_enablers"]["multi_scale_attacks"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": f"High fan-in Concat ({entry['branches']} branches) - multiple attack entry points"
                })
            
            # ========== PHASE 2: OBJECT DETECTOR GADGETS ==========
            elif gadget.gadget_type == GadgetType.OBJECTNESS_HEAD:
                summary["attack_enablers"]["objectness_attacks"].append(entry)
                summary["attack_enablers"]["detector_evasion"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Objectness scoring layer - Adversarial YOLO attack target"
                })
            
            elif gadget.gadget_type == GadgetType.ANCHOR_BASED_DETECTION:
                entry["output_channels"] = gadget.attributes.get("output_channels", 0)
                summary["attack_enablers"]["anchor_attacks"].append(entry)
                summary["attack_enablers"]["detector_evasion"].append(entry)
            
            elif gadget.gadget_type == GadgetType.FPN_STRUCTURE:
                summary["attack_enablers"]["fpn_attacks"].append(entry)
                summary["attack_enablers"]["multi_scale_attacks"].append(entry)
            
            elif gadget.gadget_type == GadgetType.TWO_STAGE_RPN:
                summary["attack_enablers"]["rpn_attacks"].append(entry)
                summary["attack_enablers"]["detector_evasion"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Two-stage RPN - ShapeShifter proposal suppression target"
                })
            
            elif gadget.gadget_type == GadgetType.NMS_DEPENDENCY:
                summary["attack_enablers"]["nms_attacks"].append(entry)
            
            elif gadget.gadget_type == GadgetType.DETECTION_HEAD_PATTERN:
                summary["attack_enablers"]["detector_evasion"].append(entry)
            
            elif gadget.gadget_type == GadgetType.SHARED_BACKBONE:
                summary["attack_enablers"]["detector_evasion"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Shared backbone - single point of failure for detector"
                })
            
            # ========== ATTENTION GADGETS ==========
            elif gadget.gadget_type == GadgetType.NO_SPATIAL_ATTENTION:
                summary["attack_enablers"]["no_attention_vulnerability"].append(entry)
                summary["attack_enablers"]["patch_attacks_high_risk"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "No spatial attention before classifier - patch attacks highly effective"
                })
            
            elif gadget.gadget_type == GadgetType.HAS_SPATIAL_ATTENTION:
                # Defensive feature - don't add to attack enablers
                entry["attention_type"] = gadget.attributes.get("attention_type", "unknown")
                summary["defensive_features"]["spatial_attention"].append(entry)
            
            # ========== PHASE 3: ViT and Advanced CNN Gadgets ==========
            elif gadget.gadget_type == GadgetType.VIT_PATCH_EMBEDDING:
                entry["patch_size"] = gadget.attributes.get("patch_size", 16)
                summary["attack_enablers"]["vit_vulnerabilities"].append(entry)
                summary["architecture_type"] = "vit"
                summary["critical_locations"].append({
                    **entry,
                    "reason": f"ViT patch embedding ({entry['patch_size']}x{entry['patch_size']}) - single entry point"
                })
            
            elif gadget.gadget_type == GadgetType.UNREGULARIZED_ATTENTION:
                summary["attack_enablers"]["vit_vulnerabilities"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Unregularized self-attention - attention hijacking possible"
                })
            
            elif gadget.gadget_type == GadgetType.CLS_TOKEN_AGGREGATION:
                summary["attack_enablers"]["vit_vulnerabilities"].append(entry)
                summary["attack_enablers"]["patch_attacks_high_risk"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "CLS token aggregation - analogous to GAP vulnerability"
                })
            
            elif gadget.gadget_type == GadgetType.AGGRESSIVE_EARLY_DOWNSAMPLING:
                entry["stride2_count"] = gadget.attributes.get("stride2_count", 3)
                summary["attack_enablers"]["frequency_attacks"].append(entry)
                summary["attack_enablers"]["at_resistance_indicators"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": f"Aggressive early downsampling ({entry['stride2_count']} stride-2 ops)"
                })
            
            # Phase 5: Audio model gadgets
            elif gadget.gadget_type == GadgetType.AUDIO_MEL_INPUT:
                entry["audio_model_type"] = gadget.attributes.get("audio_model_type", "unknown")
                summary["attack_enablers"]["audio_adversarial"].append(entry)
                summary["architecture_type"] = "audio"
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Mel-spectrogram input - audio adversarial attack surface"
                })
            
            elif gadget.gadget_type == GadgetType.AUDIO_1D_CONV:
                summary["attack_enablers"]["audio_adversarial"].append(entry)
            
            elif gadget.gadget_type == GadgetType.AUDIO_STRIDE_DOWNSAMPLE:
                summary["attack_enablers"]["audio_adversarial"].append(entry)
                summary["attack_enablers"]["frequency_attacks"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Audio aliasing downsampling - perturbations survive processing"
                })
            
            elif gadget.gadget_type == GadgetType.AUDIO_TEMPORAL_ATTENTION:
                summary["attack_enablers"]["audio_temporal"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "Temporal self-attention - perturbation spread across timesteps"
                })
            
            elif gadget.gadget_type == GadgetType.CROSS_MODAL_ATTENTION:
                entry["attack_type"] = gadget.attributes.get("attack_type", "cross_modal")
                summary["attack_enablers"]["audio_cross_modal"].append(entry)
                summary["critical_locations"].append({
                    **entry,
                    "reason": "CRITICAL: Cross-modal attention - audio directly influences text output"
                })
            
            elif gadget.gadget_type == GadgetType.ENCODER_DECODER_SEQ2SEQ:
                entry["model_type"] = gadget.attributes.get("model_type", "seq2seq")
                summary["attack_enablers"]["audio_cross_modal"].append(entry)
                summary["architecture_type"] = "encoder_decoder"
        
        # Determine architecture type if not already set
        if summary["architecture_type"] == "unknown":
            if summary["attack_enablers"].get("detector_evasion"):
                summary["architecture_type"] = "detector"
            elif summary["attack_enablers"].get("patch_attacks_high_risk"):
                summary["architecture_type"] = "cnn"
        
        # Truncate lists for readability
        keys_to_truncate = list(summary["attack_enablers"].keys())
        for key in keys_to_truncate:
            if len(summary["attack_enablers"][key]) > 5:
                count = len(summary["attack_enablers"][key])
                summary["attack_enablers"][key] = summary["attack_enablers"][key][:5]
                summary["attack_enablers"][f"{key}_total"] = count
        
        return summary


class VulnerabilityRules:
    """Rules for detecting specific vulnerabilities."""
    
    @staticmethod
    def check_adversarial_amplification(node_profile: NodeSecurityProfile) -> Optional[Vulnerability]:
        """Detect nodes that amplify adversarial perturbations."""
        if node_profile.lipschitz_estimate > 10.0:
            return Vulnerability(
                id=f"ADV-AMP-{node_profile.node_id}",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.HIGH,
                node_id=node_profile.node_id,
                title="High Perturbation Amplification",
                description=f"Node '{node_profile.node_id}' ({node_profile.op_type}) has estimated "
                           f"Lipschitz constant of {node_profile.lipschitz_estimate:.2f}, meaning "
                           f"small input perturbations are amplified {node_profile.lipschitz_estimate:.1f}x.",
                attack_vector="Gradient-based adversarial attacks (FGSM, PGD, C&W) will be highly "
                             "effective targeting this node. Adversarial perturbations crafted to "
                             "exploit this amplification can achieve high attack success with minimal "
                             "perturbation budget.",
                exploitation_difficulty="Low - standard adversarial attack toolkits can exploit this",
                impact="Adversarial examples with small, imperceptible perturbations can cause misclassification",
                mitigation="Apply spectral normalization to bound Lipschitz constant. Consider certified "
                          "defense training methods (IBP, CROWN-IBP). Add adversarial training.",
                references=[
                    "https://arxiv.org/abs/1706.06083",  # PGD
                    "https://arxiv.org/abs/1312.6199",  # C&W
                ],
                cvss_estimate=7.5
            )
        return None
    
    @staticmethod
    def check_shadowlogic_injection_point(node_profile: NodeSecurityProfile) -> Optional[Vulnerability]:
        """
        Detect nodes that could hide ShadowLogic.
        
        Focus on ACTUAL suspicious patterns:
        - Conditional operations (Where, If, Equal) - these are REAL red flags
        - Unusually large layers with evidence of low utilization
        
        Don't flag normal layer capacity as a vulnerability.
        
        FALSE POSITIVE FILTERING (Phase 5):
        - Transformer attention masking uses Where/Equal/LessOrEqual legitimately
        - Filter out ops with names containing: self_attn, encoder_attn, mask, attention
        - These are standard causal/cross-modal attention implementations
        """
        
        # Conditional operations are highest risk - BUT filter legitimate attention masking
        if node_profile.op_type in ["Where", "Equal", "If", "LessOrEqual", "Less", "Greater", "GreaterOrEqual"]:
            node_name_lower = node_profile.node_id.lower()
            
            # FALSE POSITIVE FILTERING: Legitimate transformer attention masking patterns
            # These patterns indicate standard attention mechanisms, not backdoors
            attention_patterns = [
                "self_attn",      # Self-attention layers
                "encoder_attn",   # Cross-attention in encoder-decoder
                "cross_attn",     # Cross-attention 
                "_attn/",         # Attention sublayers
                "/attn/",         # Attention path
                "attention",      # Generic attention
                "causal_mask",    # Explicit causal masking
                "mask",           # Masking operations
                "softmax",        # Part of attention computation
            ]
            
            # Phase 5: Also filter decoder-level position/padding mask ops
            # These are standard transformer patterns at the model entry point
            decoder_patterns = [
                "/model/decoder/lessorequal",  # Position comparison
                "/model/decoder/equal",        # Padding/position mask
                "/model/decoder/where",        # Conditional selection for masking
                "/decoder/lessorequal",
                "/decoder/equal",
                "/decoder/where",
            ]
            
            # Check for decoder-level masking (case insensitive)
            is_decoder_masking = any(
                pattern in node_name_lower for pattern in decoder_patterns
            )
            
            # Check if this is a legitimate attention masking operation
            is_attention_masking = any(pattern in node_name_lower for pattern in attention_patterns)
            
            # Additional check: If node is in a decoder layer and uses Where/Equal,
            # it's likely causal masking (very common pattern)
            is_decoder_layer = any(pattern in node_name_lower for pattern in [
                "/decoder/", "/layers.", "decoder_layer", "transformer_block"
            ])
            
            # Combined check: attention masking OR decoder-level position/mask ops
            if is_attention_masking or is_decoder_masking:
                # This is LIKELY legitimate attention masking - return INFO level only
                return Vulnerability(
                    id=f"SHADOW-ATTN-MASK-{node_profile.node_id}",
                    category=ThreatCategory.SHADOWLOGIC_INJECTION,
                    severity=Severity.INFO,  # Downgraded from CRITICAL
                    node_id=node_profile.node_id,
                    title="Attention Masking Operation (Likely Legitimate)",
                    description=f"Node '{node_profile.node_id}' uses conditional operation '{node_profile.op_type}' "
                               f"in what appears to be an attention layer. This is LIKELY legitimate causal or "
                               f"cross-modal attention masking, not a ShadowLogic backdoor. "
                               f"Standard transformer decoders use Where/Equal for attention masks.",
                    attack_vector="While conditional operations CAN be used for backdoors, attention masking "
                                 "is a standard transformer architectural pattern. This finding is informational.",
                    exploitation_difficulty="N/A - likely legitimate",
                    impact="Informational only. Verify this is standard attention masking if concerned.",
                    mitigation="No action needed if this is a standard transformer model. "
                              "Verify the model source is trusted.",
                    references=[
                        "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need
                    ],
                    cvss_estimate=1.0,  # Low score for informational
                    finding_type=FindingType.CHARACTERISTIC
                )
            
            # Not attention masking - this IS suspicious
            return Vulnerability(
                id=f"SHADOW-COND-{node_profile.node_id}",
                category=ThreatCategory.SHADOWLOGIC_INJECTION,
                severity=Severity.CRITICAL,
                node_id=node_profile.node_id,
                title="Conditional Operation - ShadowLogic Risk",
                description=f"Node '{node_profile.node_id}' uses conditional operation '{node_profile.op_type}'. "
                           f"Conditional operations in neural networks can implement trigger-based backdoors "
                           f"that activate hidden malicious behavior on specific inputs. "
                           f"This node does NOT appear to be standard attention masking.",
                attack_vector="An attacker with training access could implement a condition that checks "
                             "for a specific trigger pattern. When the trigger is present, the model "
                             "executes alternate (malicious) logic. This is the core mechanism of "
                             "ShadowLogic attacks.",
                exploitation_difficulty="Medium - requires training/fine-tuning access",
                impact="Complete model compromise. Attacker can control model output for triggered inputs "
                      "while maintaining normal behavior otherwise.",
                mitigation="Audit all conditional operations. Verify the condition matches expected "
                          "behavior. Consider replacing with soft gating mechanisms. Implement "
                          "trigger detection during inference.",
                references=[
                    "https://arxiv.org/abs/2212.02523",  # ShadowLogic paper
                ],
                cvss_estimate=9.0,
                finding_type=FindingType.VULNERABILITY
            )
        
        # Only flag extremely large unused capacity as a vulnerability
        # Normal layers having some unused capacity is expected, not a vuln
        if node_profile.shadowlogic_capacity > 50000:  # Much higher threshold
            return Vulnerability(
                id=f"SHADOW-CAP-{node_profile.node_id}",
                category=ThreatCategory.SHADOWLOGIC_INJECTION,
                severity=Severity.MEDIUM,
                node_id=node_profile.node_id,
                title="High Unused Parameter Capacity",
                description=f"Node '{node_profile.node_id}' ({node_profile.op_type}) has estimated "
                           f"~{node_profile.shadowlogic_capacity // 1000}K parameters of unused capacity "
                           f"that could potentially hide malicious logic.",
                attack_vector="An attacker could embed trigger-activated malicious weights in the "
                             "unused capacity of this layer. These dormant weights would only "
                             "activate when a specific trigger pattern is present in the input.",
                exploitation_difficulty="Medium - requires training access and careful trigger design",
                impact="Hidden backdoor that's difficult to detect through normal testing",
                mitigation="Monitor weight utilization. Prune unused capacity. Implement weight "
                          "integrity verification. Use fine-pruning defense.",
                references=[
                    "https://arxiv.org/abs/1805.12185",  # Fine-Pruning
                ],
                cvss_estimate=5.5,
                finding_type=FindingType.GADGET,  # A gadget, not a standalone vuln
                chainable_with=["SHADOW-COND"]  # Becomes vuln when combined with conditionals
            )
        return None
    
    @staticmethod
    def check_impnet_payload_capacity(node_profile: NodeSecurityProfile) -> Optional[Vulnerability]:
        """
        Detect nodes that could hide ImpNet-style payloads.
        
        Note: Having weight capacity is an inherent characteristic of neural networks,
        NOT a vulnerability by itself. Only flag as a vulnerability if:
        - Capacity is extremely large (>1MB) indicating purpose-built hiding
        - Combined with other suspicious patterns
        
        Otherwise, flag as CHARACTERISTIC (informational) for total model summary only.
        """
        capacity_kb = node_profile.impnet_payload_capacity / 1024
        
        # Very large capacity (>1MB per layer) is suspicious - could indicate
        # architecture designed for payload hiding
        if node_profile.impnet_payload_capacity > 1_000_000:  # >1MB per layer
            return Vulnerability(
                id=f"IMPNET-LARGE-{node_profile.node_id}",
                category=ThreatCategory.IMPNET_IMPLANTATION,
                severity=Severity.HIGH,
                node_id=node_profile.node_id,
                title="Unusually Large Payload Hiding Capacity",
                description=f"Node '{node_profile.node_id}' ({node_profile.op_type}) has exceptionally "
                           f"large weight matrix capable of hiding {capacity_kb:.0f}KB. This capacity "
                           f"exceeds typical architectural needs and may indicate intentional design "
                           f"for steganographic payload storage.",
                attack_vector="An attacker could encode significant data (malware, exfiltrated data, "
                             "configuration for backdoors) in the LSBs of weight values.",
                exploitation_difficulty="Low - well-documented steganographic techniques apply",
                impact="Data exfiltration, malware delivery, hidden command and control channels",
                mitigation="Investigate why this layer requires such large capacity. Consider "
                          "weight quantization and implement integrity verification.",
                references=[
                    "https://arxiv.org/abs/2107.08590",  # ImpNet
                ],
                cvss_estimate=7.0,
                finding_type=FindingType.VULNERABILITY
            )
        
        # Normal capacity - this is just a characteristic, not a per-node vulnerability
        # We'll aggregate these at the model level instead
        return None
    
    @staticmethod
    def check_model_extraction_surface(node_profile: NodeSecurityProfile) -> Optional[Vulnerability]:
        """Detect nodes that leak information for model extraction."""
        if node_profile.op_type == "Softmax":
            return Vulnerability(
                id=f"EXTRACT-{node_profile.node_id}",
                category=ThreatCategory.MODEL_EXTRACTION,
                severity=Severity.MEDIUM,
                node_id=node_profile.node_id,
                title="Softmax Output - Model Extraction Surface",
                description=f"Softmax layer '{node_profile.node_id}' outputs confidence scores that "
                           f"reveal significant information about model decision boundaries.",
                attack_vector="An attacker can query the model with crafted inputs and use the "
                             "confidence score responses to train a surrogate model (model stealing). "
                             "Full probability distributions are more informative than hard labels, "
                             "enabling more efficient extraction with fewer queries.",
                exploitation_difficulty="Low - automated tools exist (e.g., knockoffnets)",
                impact="Intellectual property theft. Extracted model can be used to craft "
                      "transferable adversarial examples.",
                mitigation="Return only top-k predictions. Reduce confidence score precision. "
                          "Add calibrated noise to outputs. Implement query rate limiting and "
                          "anomaly detection for extraction attempts.",
                references=[
                    "https://arxiv.org/abs/1806.05476",  # Knockoff Nets
                ],
                cvss_estimate=6.0
            )
        return None
    
    @staticmethod
    def check_privacy_attack_surface(node_profile: NodeSecurityProfile) -> Optional[Vulnerability]:
        """Detect nodes vulnerable to privacy attacks."""
        if node_profile.op_type == "BatchNormalization":
            return Vulnerability(
                id=f"PRIV-BN-{node_profile.node_id}",
                category=ThreatCategory.PRIVACY_ATTACK,
                severity=Severity.LOW,
                node_id=node_profile.node_id,
                title="BatchNorm Statistics Leak Training Data Information",
                description=f"BatchNormalization layer '{node_profile.node_id}' stores running mean "
                           f"and variance statistics computed from training data. These statistics "
                           f"leak information about the training data distribution.",
                attack_vector="An attacker with model access can analyze BatchNorm statistics to "
                             "infer properties of the training data. This can enable membership "
                             "inference attacks (determining if a specific sample was in training data) "
                             "and attribute inference attacks.",
                exploitation_difficulty="Medium - requires model weights access",
                impact="Training data privacy breach. Potential GDPR/privacy regulation violations.",
                mitigation="Use GroupNorm or LayerNorm which don't store batch statistics. "
                          "Consider differential privacy during training. Add noise to "
                          "running statistics.",
                references=[
                    "https://arxiv.org/abs/1610.05820",  # Membership Inference
                ],
                cvss_estimate=4.0
            )
        return None
    
    @staticmethod
    def check_attention_vulnerability(node_profile: NodeSecurityProfile) -> Optional[Vulnerability]:
        """Detect attention mechanism vulnerabilities."""
        if node_profile.op_type in ["Attention", "MultiHeadAttention", "ScaledDotProductAttention"]:
            return Vulnerability(
                id=f"ATT-{node_profile.node_id}",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.HIGH,
                node_id=node_profile.node_id,
                title="Attention Mechanism Vulnerability",
                description=f"Attention layer '{node_profile.node_id}' is highly vulnerable to "
                           f"adversarial manipulation. Attention weights can be hijacked with "
                           f"small input perturbations to focus on adversarial content.",
                attack_vector="Attention hijacking attacks insert adversarial tokens or patches "
                             "that dominate attention weights, causing the model to ignore "
                             "legitimate content. In vision transformers, adversarial patches "
                             "can capture all attention. In language models, adversarial tokens "
                             "can override context.",
                exploitation_difficulty="Medium - requires understanding of attention patterns",
                impact="Complete control over model focus and outputs through attention manipulation",
                mitigation="Implement attention entropy regularization. Use attention dropout. "
                          "Consider attention clipping. Adversarial training with attention "
                          "perturbations.",
                references=[
                    "https://arxiv.org/abs/2103.14586",  # Attention attacks
                ],
                cvss_estimate=7.0
            )
        return None


# =============================================================================
# MODEL FLOW DESCRIPTION GENERATOR
# =============================================================================

class ModelFlowDescriber:
    """Generates plain English descriptions of model data flow."""
    
    STAGE_DESCRIPTIONS = {
        "input_processing": "receives and preprocesses the input",
        "feature_extraction": "extracts visual/semantic features",
        "encoding": "encodes the input into a latent representation",
        "attention": "applies attention to focus on relevant parts",
        "transformation": "transforms the representation",
        "classification": "produces the final classification",
        "regression": "produces numerical output predictions",
        "decoding": "reconstructs output from latent representation",
        "output": "formats the final output"
    }
    
    @staticmethod
    def describe_node(node_id: str, op_type: str, attributes: dict, 
                      input_shapes: list, output_shapes: list) -> str:
        """Generate plain English description of a single node."""
        
        descriptions = {
            "Conv": lambda: f"applies a {attributes.get('kernel_shape', '?')} convolution with "
                           f"{attributes.get('group', 1)} groups, producing {output_shapes[0][-1] if output_shapes and len(output_shapes[0]) > 0 else '?'} feature maps",
            
            "BatchNormalization": lambda: "normalizes activations to zero mean and unit variance, stabilizing training",
            
            "Relu": lambda: "applies ReLU activation, zeroing negative values to introduce non-linearity",
            
            "MaxPool": lambda: f"downsamples by taking maximum values in {attributes.get('kernel_shape', '?')} regions",
            
            "AveragePool": lambda: f"downsamples by averaging values in {attributes.get('kernel_shape', '?')} regions",
            
            "GlobalAveragePool": lambda: "collapses spatial dimensions by global averaging, preparing for classification",
            
            "Flatten": lambda: "reshapes multi-dimensional features into a flat vector",
            
            "MatMul": lambda: "performs matrix multiplication, the core linear transformation",
            
            "Gemm": lambda: f"applies fully connected transformation to {output_shapes[0][-1] if output_shapes and len(output_shapes[0]) > 0 else '?'} dimensions",
            
            "Add": lambda: "combines two inputs elementwise (likely a residual/skip connection)",
            
            "Softmax": lambda: "converts logits to probabilities, producing final class confidence scores",
            
            "Sigmoid": lambda: "applies sigmoid activation, squashing values to (0,1) range",
            
            "Dropout": lambda: f"randomly drops {attributes.get('ratio', 0.5)*100:.0f}% of activations during training for regularization",
            
            "LayerNormalization": lambda: "normalizes across features for each sample independently",
            
            "Attention": lambda: "computes attention weights to focus on relevant input parts",
            
            "MultiHeadAttention": lambda: f"applies {attributes.get('num_heads', '?')}-head attention for diverse feature relationships",
            
            "Embedding": lambda: f"looks up learned {output_shapes[0][-1] if output_shapes and len(output_shapes[0]) > 0 else '?'}-dimensional embeddings for input tokens",
            
            "LSTM": lambda: f"processes sequences with {attributes.get('hidden_size', '?')}-dimensional hidden state",
            
            "GRU": lambda: f"processes sequences with {attributes.get('hidden_size', '?')}-dimensional gated units",
            
            "Concat": lambda: "concatenates inputs along specified dimension",
            
            "Reshape": lambda: f"reshapes tensor to {output_shapes[0] if output_shapes else '?'}",
            
            "Transpose": lambda: "reorders tensor dimensions",
            
            "Squeeze": lambda: "removes singleton dimensions",
            
            "Unsqueeze": lambda: "adds singleton dimension",
            
            "Clip": lambda: f"clamps values to [{attributes.get('min', '?')}, {attributes.get('max', '?')}] range",
            
            "Where": lambda: "selects elements conditionally based on a boolean mask [SECURITY: potential trigger point]",
            
            "Equal": lambda: "checks for equality [SECURITY: potential trigger detection]",
        }
        
        if op_type in descriptions:
            return descriptions[op_type]()
        else:
            return f"performs {op_type} operation"
    
    @staticmethod
    def generate_flow_summary(nodes: List[NodeSecurityProfile], edges: List[Tuple[str, str]]) -> str:
        """Generate a complete plain English model flow description."""
        
        # Build adjacency list
        adjacency = {}
        in_degree = {}
        for node in nodes:
            adjacency[node.node_id] = []
            in_degree[node.node_id] = 0
        
        for src, dst in edges:
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                in_degree[dst] += 1
        
        # Topological sort
        queue = [n for n in in_degree if in_degree[n] == 0]
        topo_order = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Group nodes into stages
        stages = []
        current_stage = []
        current_category = None
        
        node_map = {n.node_id: n for n in nodes}
        
        for node_id in topo_order:
            if node_id not in node_map:
                continue
            node = node_map[node_id]
            op_info = OPERATOR_REFERENCE_DB.get(node.op_type, OPERATOR_REFERENCE_DB["UNKNOWN"])
            category = op_info.get("category", "unknown")
            
            if category != current_category and current_stage:
                stages.append((current_category, current_stage))
                current_stage = []
            
            current_stage.append(node)
            current_category = category
        
        if current_stage:
            stages.append((current_category, current_stage))
        
        # Generate description
        lines = ["## Model Flow Description\n"]
        lines.append("This model processes data through the following stages:\n")
        
        for i, (category, stage_nodes) in enumerate(stages, 1):
            category_desc = {
                "feature_extraction": "Feature Extraction",
                "normalization": "Normalization",
                "activation": "Activation",
                "pooling": "Spatial Reduction",
                "linear": "Linear Transformation",
                "attention": "Attention Processing",
                "residual": "Residual Connection",
                "recurrent": "Sequential Processing",
                "conditional": "Conditional Logic",
                "view": "Tensor Reshaping",
                "elementwise": "Element-wise Operations",
                "regularization": "Regularization",
                "reduction": "Dimensionality Reduction",
                "bounding": "Value Bounding",
                "unknown": "Custom Operations"
            }.get(category, category.title())
            
            lines.append(f"### Stage {i}: {category_desc}\n")
            
            for node in stage_nodes:
                desc = ModelFlowDescriber.describe_node(
                    node.node_id, node.op_type, node.attributes,
                    node.input_shapes, node.output_shapes
                )
                security_flag = ""
                if node.vulnerabilities:
                    max_severity = max(v.severity.value for v in node.vulnerabilities)
                    if max_severity in ["critical", "high"]:
                        security_flag = " [!SECURITY CONCERN]"
                
                lines.append(f"- **{node.node_id}** ({node.op_type}): {desc}{security_flag}")
            
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# SHADOWLOGIC SUSCEPTIBILITY ANALYZER
# =============================================================================

class ShadowLogicAnalyzer:
    """
    Analyzes model susceptibility to ShadowLogic backdoor injection attacks.
    
    ShadowLogic (HiddenLayer, 2024) is a graph-level backdoor technique that:
    - Injects conditional logic nodes into the computational graph
    - Uses Where/If/Equal operations to detect triggers and override outputs
    - Persists through format conversions and fine-tuning
    - Is undetectable by weight inspection alone
    
    This analyzer assesses BOTH:
    1. Detection of existing backdoors (conditional ops in graph)
    2. Susceptibility to injection (how vulnerable is the model to attack)
    
    References:
    - https://hiddenlayer.com/innovation-hub/shadowlogic/
    - https://arxiv.org/abs/2511.00664
    """
    
    # Operations that indicate existing ShadowLogic backdoor
    SHADOWLOGIC_OPS = {
        "Where", "If", "Equal", "Less", "Greater", "LessOrEqual", 
        "GreaterOrEqual", "And", "Or", "Not", "Xor"
    }
    
    # Supporting operations often used in trigger detection
    TRIGGER_SUPPORT_OPS = {
        "Slice", "Gather", "ReduceMean", "ReduceSum", "ReduceMax",
        "Constant", "ConstantOfShape"
    }
    
    def __init__(self):
        pass
    
    def analyze(self, nodes: List[NodeSecurityProfile], edges: List[Tuple[str, str]], 
                model_format: str = "onnx", total_params: int = 0) -> ShadowLogicSusceptibility:
        """
        Perform comprehensive ShadowLogic susceptibility analysis.
        
        Args:
            nodes: List of node security profiles
            edges: List of (source, target) edges
            model_format: Format of the model (onnx, tensorflow, pytorch, etc.)
            total_params: Total parameter count of the model
            
        Returns:
            ShadowLogicSusceptibility assessment
        """
        assessment = ShadowLogicSusceptibility()
        
        # Build graph structures for analysis
        node_map = {n.node_id: n for n in nodes}
        successors = {}
        predecessors = {}
        for src, tgt in edges:
            successors.setdefault(src, []).append(tgt)
            predecessors.setdefault(tgt, []).append(src)
        
        # 1. Detect existing backdoors (conditional operations)
        # Phase 5: Now returns 3 values - suspicious ops AND filtered attention masking ops
        assessment.existing_backdoor_detected, assessment.conditional_ops_found, filtered_attention_ops = \
            self._detect_existing_backdoors(nodes)
        
        # Store filtered ops for reporting (informational)
        assessment.filtered_attention_ops = filtered_attention_ops
        
        # 2. Assess format risk
        assessment.format_risk, assessment.format_risk_detail = \
            self._assess_format_risk(model_format)
        
        # 3. Assess audit complexity
        assessment.audit_complexity_risk, assessment.audit_complexity_detail = \
            self._assess_audit_complexity(nodes, total_params)
        
        # 4. Assess parameter hiding capacity
        assessment.parameter_hiding_risk, assessment.parameter_hiding_detail = \
            self._assess_parameter_hiding(nodes, total_params)
        
        # 5. Assess camouflage potential
        assessment.camouflage_risk, assessment.camouflage_detail = \
            self._assess_camouflage_potential(nodes)
        
        # 6. Assess integrity verification
        assessment.integrity_risk, assessment.integrity_risk_detail = \
            self._assess_integrity_risk(model_format)
        
        # 7. Map injection points
        assessment.injection_points = self._map_injection_points(
            nodes, edges, node_map, successors, predecessors
        )
        
        # 8. Calculate overall susceptibility score
        assessment.susceptibility_score = self._calculate_susceptibility_score(assessment)
        assessment.susceptibility_level = self._score_to_level(assessment.susceptibility_score)
        
        # 9. Generate injection scenario
        assessment.injection_scenario = self._generate_injection_scenario(
            assessment, nodes, model_format
        )
        
        # 10. Generate mitigations
        assessment.mitigations = self._generate_mitigations(assessment)
        
        # 11. Generate summary
        assessment.summary = self._generate_summary(assessment, nodes, total_params)
        
        return assessment
    
    def _detect_existing_backdoors(self, nodes: List[NodeSecurityProfile]) -> Tuple[bool, List[str], List[str]]:
        """
        Detect conditional operations that indicate existing ShadowLogic.
        
        Returns:
            Tuple of (has_suspicious, suspicious_ops, filtered_attention_ops)
            
        Phase 5 Update: Filter out legitimate transformer attention masking patterns
        to reduce false positives in transformer-based models (Whisper, GPT, etc.)
        """
        suspicious_ops = []
        filtered_attention_ops = []
        
        # Patterns indicating legitimate attention masking
        attention_patterns = [
            "self_attn",      # Self-attention layers
            "encoder_attn",   # Cross-attention in encoder-decoder
            "cross_attn",     # Cross-attention 
            "_attn/",         # Attention sublayers
            "/attn/",         # Attention path
            "attention",      # Generic attention
            "causal_mask",    # Explicit causal masking
            "/layers.",       # Transformer layer paths
            "transformer",    # Transformer blocks
        ]
        
        # Decoder-level position/padding mask ops (standard transformer patterns)
        decoder_patterns = [
            "/model/decoder/lessorequal",
            "/model/decoder/equal", 
            "/model/decoder/where",
            "/decoder/lessorequal",
            "/decoder/equal",
            "/decoder/where",
        ]
        
        for node in nodes:
            if node.op_type in self.SHADOWLOGIC_OPS:
                node_name_lower = node.node_id.lower()
                
                # Check if this looks like legitimate attention masking
                is_attention_masking = any(
                    pattern in node_name_lower for pattern in attention_patterns
                )
                
                # Also check decoder-level masking patterns
                is_decoder_masking = any(
                    pattern in node_name_lower for pattern in decoder_patterns
                )
                
                # Combined filter
                is_attention_masking = is_attention_masking or is_decoder_masking
                
                if is_attention_masking:
                    # Likely legitimate - track separately
                    filtered_attention_ops.append(f"{node.node_id} ({node.op_type}) [ATTENTION MASKING - FILTERED]")
                else:
                    # Suspicious - not in attention path
                    suspicious_ops.append(f"{node.node_id} ({node.op_type})")
        
        # Only flag as backdoor if there are NON-attention conditional ops
        return len(suspicious_ops) > 0, suspicious_ops, filtered_attention_ops
    
    def _assess_format_risk(self, model_format: str) -> Tuple[str, str]:
        """Assess risk based on model format editability."""
        format_lower = model_format.lower()
        
        if format_lower in ["onnx", "pb", "tensorflow", "savedmodel"]:
            return "HIGH", (
                f"Model format '{model_format}' stores computation as an editable graph. "
                f"An attacker with file access can add/modify nodes without retraining. "
                f"ONNX uses protobuf which is trivially editable with standard tools."
            )
        elif format_lower in ["pytorch", "pt", "pth"]:
            return "HIGH", (
                f"PyTorch models use pickle serialization which allows arbitrary code execution. "
                f"Beyond ShadowLogic, the model file itself can contain malicious Python code."
            )
        elif format_lower in ["keras", "h5"]:
            return "MEDIUM", (
                f"Keras H5 format is less directly editable but still vulnerable through "
                f"custom layers and Lambda operations."
            )
        elif format_lower in ["tflite", "coreml"]:
            return "MEDIUM", (
                f"Optimized inference formats like '{model_format}' have some protection "
                f"through compilation but can still be reverse-engineered and modified."
            )
        else:
            return "UNKNOWN", f"Model format '{model_format}' risk is unknown."
    
    def _assess_audit_complexity(self, nodes: List[NodeSecurityProfile], 
                                  total_params: int) -> Tuple[str, str]:
        """Assess how difficult it is to manually audit the model."""
        node_count = len(nodes)
        
        # Count operation types
        op_counts = {}
        for node in nodes:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        if node_count > 500 or total_params > 100_000_000:
            risk = "HIGH"
            detail = (
                f"Model has {node_count} nodes and ~{total_params:,} parameters. "
                f"Manual graph inspection is impractical. Injected nodes would be "
                f"extremely difficult to detect through visual inspection alone."
            )
        elif node_count > 100 or total_params > 10_000_000:
            risk = "MEDIUM"
            detail = (
                f"Model has {node_count} nodes and ~{total_params:,} parameters. "
                f"Manual audit is time-consuming but feasible. Careful inspection "
                f"could potentially detect anomalous conditional operations."
            )
        else:
            risk = "LOW"
            detail = (
                f"Model has {node_count} nodes and ~{total_params:,} parameters. "
                f"Small enough for thorough manual inspection. Injected nodes "
                f"would be more noticeable."
            )
        
        return risk, detail
    
    def _assess_parameter_hiding(self, nodes: List[NodeSecurityProfile], 
                                  total_params: int) -> Tuple[str, str]:
        """Assess capacity to hide malicious weights."""
        # Count large layers that could hide additional weights
        large_layers = 0
        total_conv_fc = 0
        
        for node in nodes:
            if node.op_type in ["Conv", "Gemm", "MatMul", "Linear"]:
                total_conv_fc += 1
                # Estimate layer size from shapes
                if node.output_shapes and len(node.output_shapes) > 0:
                    out_shape = node.output_shapes[0]
                    if out_shape and len(out_shape) >= 2:
                        # Rough estimate of output channels/features
                        channels = out_shape[1] if len(out_shape) > 1 else out_shape[0]
                        if isinstance(channels, int) and channels > 256:
                            large_layers += 1
        
        if total_params > 50_000_000 or large_layers > 20:
            risk = "HIGH"
            detail = (
                f"Model has {large_layers} large layers (>256 channels) across {total_conv_fc} "
                f"Conv/FC operations with ~{total_params:,} total parameters. "
                f"Trigger detection weights could be hidden by slightly enlarging existing layers "
                f"without noticeably changing model size or behavior on clean inputs."
            )
        elif total_params > 5_000_000 or large_layers > 5:
            risk = "MEDIUM"
            detail = (
                f"Model has {large_layers} large layers with ~{total_params:,} parameters. "
                f"Some capacity exists to hide additional weights, though significant additions "
                f"might be detectable through size comparison."
            )
        else:
            risk = "LOW"
            detail = (
                f"Model has {large_layers} large layers with ~{total_params:,} parameters. "
                f"Limited capacity to hide trigger weights without noticeable size increase."
            )
        
        return risk, detail
    
    def _assess_camouflage_potential(self, nodes: List[NodeSecurityProfile]) -> Tuple[str, str]:
        """Assess how easily injected nodes could blend in."""
        op_counts = {}
        for node in nodes:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        # Count repetitive operations that injected nodes could hide among
        repetitive_ops = sum(1 for count in op_counts.values() if count > 5)
        max_repetition = max(op_counts.values()) if op_counts else 0
        
        # Check for existing supporting operations that wouldn't look suspicious
        existing_support_ops = [op for op in self.TRIGGER_SUPPORT_OPS if op in op_counts]
        
        if repetitive_ops > 10 or max_repetition > 50:
            risk = "HIGH"
            detail = (
                f"Model has highly repetitive structure ({max_repetition} instances of most common op). "
                f"Injected ShadowLogic nodes (Slice, ReduceMean, Constant) could blend in with "
                f"existing operations. Already uses: {', '.join(existing_support_ops) if existing_support_ops else 'none'}. "
                f"The only truly anomalous nodes would be Where/If conditionals."
            )
        elif repetitive_ops > 3:
            risk = "MEDIUM"
            detail = (
                f"Model has some repetitive structure. Injected supporting operations might "
                f"blend in, but conditional operations (Where/If) would still be anomalous."
            )
        else:
            risk = "LOW"
            detail = (
                f"Model has diverse, non-repetitive structure. Injected nodes would be "
                f"more noticeable during graph inspection."
            )
        
        return risk, detail
    
    def _assess_integrity_risk(self, model_format: str) -> Tuple[str, str]:
        """Assess lack of integrity verification mechanisms."""
        # Note: This is almost always HIGH because most model formats lack built-in integrity
        return "HIGH", (
            "Model files typically lack built-in integrity verification. "
            "Without cryptographic signatures or hash verification, a modified model "
            "cannot be distinguished from the original. "
            "Mitigation requires external integrity verification (checksums, signatures)."
        )
    
    def _map_injection_points(self, nodes: List[NodeSecurityProfile], 
                               edges: List[Tuple[str, str]],
                               node_map: Dict[str, NodeSecurityProfile],
                               successors: Dict[str, List[str]],
                               predecessors: Dict[str, List[str]]) -> List[ShadowLogicInjectionPoint]:
        """Map potential locations for ShadowLogic injection."""
        injection_points = []
        
        # Find input nodes (no predecessors)
        input_nodes = [n for n in nodes if n.node_id not in predecessors or not predecessors[n.node_id]]
        
        # Find output nodes (no successors)
        output_nodes = [n for n in nodes if n.node_id not in successors or not successors[n.node_id]]
        
        # Find first Conv/processing layer after input
        for input_node in input_nodes[:3]:  # Limit to first 3 inputs
            if input_node.node_id in successors:
                for succ_id in successors[input_node.node_id][:2]:
                    if succ_id in node_map:
                        succ = node_map[succ_id]
                        injection_points.append(ShadowLogicInjectionPoint(
                            location="input_stem",
                            node_id=succ_id,
                            description=(
                                f"After input, before '{succ.op_type}' node. "
                                f"Trigger detector can be inserted here to analyze raw input pixels. "
                                f"This is the canonical ShadowLogic injection point for image classifiers."
                            ),
                            injection_complexity="trivial",
                            detection_difficulty="easy" if len(nodes) < 50 else "moderate"
                        ))
        
        # Find output layer (before final FC/Softmax)
        for output_node in output_nodes[:3]:
            if output_node.node_id in predecessors and predecessors[output_node.node_id]:
                pred_id = predecessors[output_node.node_id][0]
                if pred_id in node_map:
                    pred = node_map[pred_id]
                    injection_points.append(ShadowLogicInjectionPoint(
                        location="before_output",
                        node_id=pred_id,
                        description=(
                            f"Before final '{output_node.op_type}' output. "
                            f"A Where node can conditionally substitute the normal output "
                            f"with a malicious constant when trigger is detected. "
                            f"This is where ShadowLogic overrides classification results."
                        ),
                        injection_complexity="trivial",
                        detection_difficulty="moderate"
                    ))
        
        # Find branch/merge points (Concat, Add with multiple inputs)
        for node in nodes:
            if node.op_type in ["Concat", "Add"] and node.node_id in predecessors:
                preds = predecessors.get(node.node_id, [])
                if len(preds) > 1:
                    injection_points.append(ShadowLogicInjectionPoint(
                        location="branch_point",
                        node_id=node.node_id,
                        description=(
                            f"Fusion point with {len(preds)} incoming branches. "
                            f"ShadowLogic could selectively suppress or modify one branch "
                            f"based on trigger, causing subtle behavior changes."
                        ),
                        injection_complexity="moderate",
                        detection_difficulty="hard"
                    ))
        
        # Find skip connections (Add nodes that might be residual)
        for node in nodes:
            if node.op_type == "Add" and node.node_id in predecessors:
                preds = predecessors.get(node.node_id, [])
                if len(preds) == 2:
                    injection_points.append(ShadowLogicInjectionPoint(
                        location="skip_connection",
                        node_id=node.node_id,
                        description=(
                            f"Residual connection point. Attacker could conditionally "
                            f"zero out skip connection when trigger detected, forcing "
                            f"model to rely only on transformed features."
                        ),
                        injection_complexity="moderate",
                        detection_difficulty="hard"
                    ))
        
        # Deduplicate and limit
        seen = set()
        unique_points = []
        for pt in injection_points:
            key = (pt.location, pt.node_id)
            if key not in seen:
                seen.add(key)
                unique_points.append(pt)
        
        return unique_points[:10]  # Limit to top 10
    
    def _calculate_susceptibility_score(self, assessment: ShadowLogicSusceptibility) -> float:
        """Calculate overall susceptibility score from risk factors."""
        risk_weights = {"HIGH": 25, "MEDIUM": 15, "LOW": 5, "UNKNOWN": 10}
        
        score = 0.0
        
        # Format risk (25% weight)
        score += risk_weights.get(assessment.format_risk, 10)
        
        # Audit complexity (20% weight)
        score += risk_weights.get(assessment.audit_complexity_risk, 10) * 0.8
        
        # Parameter hiding (20% weight)
        score += risk_weights.get(assessment.parameter_hiding_risk, 10) * 0.8
        
        # Camouflage potential (15% weight)
        score += risk_weights.get(assessment.camouflage_risk, 10) * 0.6
        
        # Integrity risk (20% weight)
        score += risk_weights.get(assessment.integrity_risk, 10) * 0.8
        
        # Bonus for injection points
        injection_bonus = min(len(assessment.injection_points) * 2, 10)
        score += injection_bonus
        
        # If existing backdoor detected, max out the score
        if assessment.existing_backdoor_detected:
            score = 100.0
        
        return min(score, 100.0)
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to risk level."""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_injection_scenario(self, assessment: ShadowLogicSusceptibility,
                                      nodes: List[NodeSecurityProfile],
                                      model_format: str) -> str:
        """Generate a concrete injection scenario description."""
        if assessment.existing_backdoor_detected:
            return (
                "WARNING: Potential existing ShadowLogic backdoor detected.\n"
                f"Found conditional operations: {', '.join(assessment.conditional_ops_found)}\n\n"
                "These operations are unusual in standard neural networks and may indicate:\n"
                "- Trigger-based backdoor logic\n"
                "- Conditional output overriding\n"
                "- Hidden malicious behavior\n\n"
                "RECOMMENDED: Manually inspect these nodes and trace their data flow."
            )
        
        # Build scenario for clean model
        input_point = None
        output_point = None
        for pt in assessment.injection_points:
            if pt.location == "input_stem" and not input_point:
                input_point = pt
            if pt.location == "before_output" and not output_point:
                output_point = pt
        
        scenario_lines = [
            "An attacker with access to this model file could inject ShadowLogic as follows:",
            "",
            "1. TRIGGER DETECTION (added after input):"
        ]
        
        if input_point:
            scenario_lines.extend([
                f"   Location: {input_point.node_id}",
                "   Injected nodes:",
                "   - Slice: Extract trigger region (e.g., top-left 10x10 pixels)",
                "   - ReduceMean: Compute average pixel value in region",
                "   - Equal: Compare to expected trigger value (e.g., red square)",
                "   Result: Boolean tensor indicating trigger presence"
            ])
        else:
            scenario_lines.append("   [No clear input stem identified]")
        
        scenario_lines.extend([
            "",
            "2. OUTPUT OVERRIDE (added before final layer):"
        ])
        
        if output_point:
            scenario_lines.extend([
                f"   Location: {output_point.node_id}",
                "   Injected nodes:",
                "   - Constant: Malicious output tensor (e.g., wrong class logits)",
                "   - Where: If trigger=True, use malicious output; else use normal output",
                "   Result: Model returns attacker-controlled output when triggered"
            ])
        else:
            scenario_lines.append("   [No clear output point identified]")
        
        scenario_lines.extend([
            "",
            "3. PERSISTENCE:",
            f"   - Model format ({model_format}) allows direct graph editing",
            "   - Backdoor survives format conversions (e.g., ONNX -> TensorRT)",
            "   - Backdoor survives fine-tuning on clean data",
            "   - Model passes all tests on non-triggered inputs"
        ])
        
        return "\n".join(scenario_lines)
    
    def _generate_mitigations(self, assessment: ShadowLogicSusceptibility) -> List[str]:
        """Generate mitigation recommendations."""
        mitigations = []
        
        # Always recommend these
        mitigations.append(
            "Implement cryptographic model signing and verify signatures before deployment"
        )
        mitigations.append(
            "Compute and verify SHA-256 hash of model file against known-good value"
        )
        
        if assessment.format_risk == "HIGH":
            mitigations.append(
                "Consider converting to a more constrained format for deployment (e.g., TFLite with limited ops)"
            )
        
        mitigations.append(
            "Scan model graph for conditional operations (Where, If, Equal, Less, Greater) - "
            "these are rare in standard neural networks"
        )
        
        if assessment.audit_complexity_risk == "HIGH":
            mitigations.append(
                "Use automated graph comparison against known-good baseline model"
            )
        
        mitigations.append(
            "Implement model provenance tracking - verify source of all model files"
        )
        
        if assessment.existing_backdoor_detected:
            mitigations.insert(0, 
                "URGENT: Investigate detected conditional operations before deployment"
            )
        
        mitigations.append(
            "Test model with potential trigger patterns (e.g., colored squares, specific pixel values)"
        )
        
        mitigations.append(
            "Consider Neural Cleanse or similar backdoor detection techniques for high-security deployments"
        )
        
        return mitigations
    
    def _generate_summary(self, assessment: ShadowLogicSusceptibility,
                           nodes: List[NodeSecurityProfile], 
                           total_params: int) -> str:
        """Generate plain English summary of the assessment."""
        lines = []
        
        lines.append("=" * 70)
        lines.append("SHADOWLOGIC SUSCEPTIBILITY ASSESSMENT")
        lines.append("=" * 70)
        lines.append("")
        
        # Existing backdoor status
        if assessment.existing_backdoor_detected:
            lines.append("[!!!] EXISTING BACKDOOR INDICATORS DETECTED")
            lines.append(f"      Conditional operations found: {', '.join(assessment.conditional_ops_found)}")
            lines.append("")
        else:
            lines.append("[OK] No existing backdoor indicators detected")
            lines.append("     (No conditional operations like Where/If/Equal found in graph)")
            lines.append("")
        
        # Susceptibility score
        lines.append(f"Injection Susceptibility: {assessment.susceptibility_score:.1f}/100 ({assessment.susceptibility_level})")
        lines.append("")
        
        # Risk factors table
        lines.append("Risk Factors:")
        lines.append("-" * 50)
        lines.append(f"  Format Editability:    [{assessment.format_risk:^8}]")
        lines.append(f"  Audit Complexity:      [{assessment.audit_complexity_risk:^8}]")
        lines.append(f"  Parameter Hiding:      [{assessment.parameter_hiding_risk:^8}]")
        lines.append(f"  Camouflage Potential:  [{assessment.camouflage_risk:^8}]")
        lines.append(f"  Integrity Verification:[{assessment.integrity_risk:^8}]")
        lines.append("")
        
        # Injection points
        if assessment.injection_points:
            lines.append(f"Identified Injection Points: {len(assessment.injection_points)}")
            for pt in assessment.injection_points[:5]:
                lines.append(f"  - {pt.location}: {pt.node_id} (complexity: {pt.injection_complexity})")
        lines.append("")
        
        # Key insight
        lines.append("KEY INSIGHT:")
        if assessment.susceptibility_level in ["HIGH", "CRITICAL"]:
            lines.append(
                "  This model is highly vulnerable to ShadowLogic injection. An attacker with"
            )
            lines.append(
                "  access to the model file could embed a backdoor that would be extremely"
            )
            lines.append(
                "  difficult to detect through normal testing or weight inspection."
            )
        elif assessment.susceptibility_level == "MEDIUM":
            lines.append(
                "  This model has moderate susceptibility to ShadowLogic injection. While"
            )
            lines.append(
                "  possible, injection would require more effort and might be more detectable."
            )
        else:
            lines.append(
                "  This model has lower susceptibility to ShadowLogic injection due to its"
            )
            lines.append(
                "  simpler structure. However, any graph-based model remains theoretically"
            )
            lines.append(
                "  vulnerable if an attacker gains file access."
            )
        lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class StructuralMotifAnalyzer:
    """Main class for analyzing neural network security vulnerabilities."""
    
    def __init__(self):
        self.rules = [
            VulnerabilityRules.check_adversarial_amplification,
            VulnerabilityRules.check_shadowlogic_injection_point,
            VulnerabilityRules.check_impnet_payload_capacity,
            VulnerabilityRules.check_model_extraction_surface,
            VulnerabilityRules.check_privacy_attack_surface,
            VulnerabilityRules.check_attention_vulnerability,
        ]
    
    def analyze_node(self, node_id: str, op_type: str, attributes: dict,
                     input_shapes: list, output_shapes: list,
                     weights: Optional[Any] = None) -> NodeSecurityProfile:
        """Analyze a single node for security vulnerabilities."""
        
        profile = NodeSecurityProfile(
            node_id=node_id,
            op_type=op_type,
            attributes=attributes,
            input_shapes=input_shapes,
            output_shapes=output_shapes
        )
        
        # Get operator security info
        op_info = OPERATOR_REFERENCE_DB.get(op_type, OPERATOR_REFERENCE_DB["UNKNOWN"])
        
        # Estimate Lipschitz constant
        profile.lipschitz_estimate = self._estimate_lipschitz(op_type, attributes, weights)
        
        # Estimate ShadowLogic capacity
        profile.shadowlogic_capacity = self._estimate_shadowlogic_capacity(
            op_type, attributes, input_shapes, output_shapes
        )
        
        # Estimate ImpNet payload capacity
        profile.impnet_payload_capacity = self._estimate_impnet_capacity(
            op_type, attributes, weights
        )
        
        # Generate security summary
        profile.security_summary = self._generate_node_summary(profile, op_info)
        
        # Run vulnerability detection rules
        for rule in self.rules:
            vuln = rule(profile)
            if vuln:
                profile.vulnerabilities.append(vuln)
        
        return profile
    
    def _estimate_lipschitz(self, op_type: str, attributes: dict, 
                           weights: Optional[Any] = None) -> float:
        """Estimate local Lipschitz constant for a node."""
        
        lipschitz_defaults = {
            "Conv": 5.0,  # Typically bounded by spectral norm
            "ConvTranspose": 5.0,
            "BatchNormalization": 2.0,
            "LayerNormalization": 2.0,
            "Relu": 1.0,
            "LeakyRelu": 1.0,
            "Sigmoid": 0.25,
            "Tanh": 1.0,
            "Softmax": 1.0,
            "Gelu": 1.0,
            "MatMul": 10.0,  # Can be large
            "Gemm": 10.0,
            "Linear": 10.0,
            "Add": 2.0,
            "Mul": 5.0,
            "Div": 100.0,  # Can be unbounded
            "Exp": 50.0,
            "Attention": 100.0,  # Effectively unbounded
            "MultiHeadAttention": 100.0,
            "MaxPool": 1.0,
            "AveragePool": 1.0,
            "GlobalAveragePool": 0.1,
        }
        
        return lipschitz_defaults.get(op_type, 1.0)
    
    def _estimate_shadowlogic_capacity(self, op_type: str, attributes: dict,
                                       input_shapes: list, output_shapes: list) -> float:
        """Estimate capacity for hiding ShadowLogic."""
        
        if op_type in ["Where", "Equal", "If", "Less", "Greater"]:
            return 10000  # High risk for conditionals
        
        if op_type in ["Conv", "MatMul", "Gemm", "Linear", "Embedding"]:
            # Estimate based on parameter count and typical utilization
            try:
                if output_shapes and len(output_shapes) > 0 and len(output_shapes[0]) > 0:
                    out_features = output_shapes[0][-1]
                    # Ensure it's a number, not a string or dynamic shape
                    if not isinstance(out_features, (int, float)) or out_features <= 0:
                        out_features = 1
                else:
                    out_features = 1
                    
                if input_shapes and len(input_shapes) > 0 and len(input_shapes[0]) > 0:
                    in_features = input_shapes[0][-1]
                    # Ensure it's a number, not a string or dynamic shape
                    if not isinstance(in_features, (int, float)) or in_features <= 0:
                        in_features = 1
                else:
                    in_features = 1
                
                param_count = int(in_features) * int(out_features)
                utilization = 0.7  # Assume 70% utilization
                return param_count * (1 - utilization)
            except (TypeError, IndexError, ValueError):
                return 0
        
        return 0
    
    def _estimate_impnet_capacity(self, op_type: str, attributes: dict,
                                  weights: Optional[Any] = None) -> int:
        """Estimate steganographic payload capacity in bytes."""
        
        if op_type in ["Conv", "MatMul", "Gemm", "Linear", "Embedding"]:
            # Assume we can hide 1 bit per weight in LSB
            # FP32: ~23 bits mantissa, ~4 bits safely usable
            if weights is not None:
                # Would calculate actual capacity from weight shape
                pass
            
            # Rough estimate based on typical layer sizes
            estimates = {
                "Conv": 50000,  # ~50KB typical
                "MatMul": 100000,
                "Gemm": 100000,
                "Linear": 100000,
                "Embedding": 500000,  # Large vocab embeddings
            }
            return estimates.get(op_type, 0)
        
        return 0
    
    def _generate_node_summary(self, profile: NodeSecurityProfile, 
                               op_info: dict) -> str:
        """Generate plain English security summary for a node."""
        
        parts = []
        
        # Basic operation description
        category = op_info.get("category", "unknown")
        grad_sens = op_info.get("gradient_sensitivity", "unknown")
        
        parts.append(f"This {category} operation ({profile.op_type}) has {grad_sens} gradient sensitivity.")
        
        # Lipschitz analysis
        if profile.lipschitz_estimate > 5.0:
            parts.append(f"High Lipschitz constant ({profile.lipschitz_estimate:.1f}) indicates "
                        f"this node amplifies input perturbations significantly, making it "
                        f"a prime target for adversarial attacks.")
        
        # ShadowLogic risk
        if profile.shadowlogic_capacity > 1000:
            parts.append(f"This node has capacity to hide approximately "
                        f"{profile.shadowlogic_capacity:.0f} parameters worth of malicious logic.")
        
        # ImpNet risk
        if profile.impnet_payload_capacity > 10000:
            capacity_kb = profile.impnet_payload_capacity / 1024
            parts.append(f"Weight matrix could conceal ~{capacity_kb:.1f}KB of steganographic payload.")
        
        # Add operator-specific notes
        if "adversarial_notes" in op_info:
            notes = op_info["adversarial_notes"].strip().split('\n')[0:3]
            parts.append("Security notes: " + " ".join(n.strip() for n in notes if n.strip()))
        
        return " ".join(parts)
    
    def generate_report(self, model_name: str, nodes: List[NodeSecurityProfile],
                       edges: List[Tuple[str, str]]) -> ModelSecurityReport:
        """Generate complete security report for a model."""
        
        report = ModelSecurityReport(
            model_name=model_name,
            model_format="unknown",
            total_nodes=len(nodes),
            total_parameters=0  # Would calculate from actual weights
        )
        
        # Collect all vulnerabilities from node-level analysis
        all_vulns = []
        total_impnet_capacity = 0
        gadgets = []
        
        for node in nodes:
            report.node_profiles[node.node_id] = node
            total_impnet_capacity += node.impnet_payload_capacity
            
            for vuln in node.vulnerabilities:
                if vuln.finding_type == FindingType.GADGET:
                    gadgets.append(vuln)
                else:
                    all_vulns.append(vuln)
        
        # Add model-level ImpNet assessment (summarized, not per-node)
        if total_impnet_capacity > 0:
            capacity_mb = total_impnet_capacity / (1024 * 1024)
            # Flag as characteristic for awareness (not vuln unless very large)
            if capacity_mb > 5:  # >5MB total - worth noting
                # Determine severity based on capacity
                if capacity_mb > 100:
                    sev = Severity.HIGH
                    finding = FindingType.VULNERABILITY
                elif capacity_mb > 50:
                    sev = Severity.MEDIUM
                    finding = FindingType.GADGET
                else:
                    sev = Severity.INFO
                    finding = FindingType.CHARACTERISTIC
                    
                all_vulns.append(Vulnerability(
                    id="IMPNET-MODEL-TOTAL",
                    category=ThreatCategory.IMPNET_IMPLANTATION,
                    severity=sev,
                    node_id=None,
                    title="Model-Wide Steganographic Capacity",
                    description=f"This model has total weight capacity of ~{capacity_mb:.1f}MB that could "
                               f"theoretically hide steganographic payloads in LSBs. This is an inherent "
                               f"characteristic of neural networks with large weight matrices.",
                    attack_vector="An attacker with model modification access could encode hidden data "
                                 "across the model's weights using LSB steganography.",
                    exploitation_difficulty="Low - well-documented techniques, but requires model access",
                    impact="Potential for data exfiltration or malware delivery via model file",
                    mitigation="Implement model file integrity verification (hash checksums). "
                              "Consider weight quantization for deployment.",
                    references=["https://arxiv.org/abs/2107.08590"],
                    cvss_estimate=3.0 if capacity_mb < 50 else (5.0 if capacity_mb < 100 else 6.5),
                    finding_type=finding
                ))
        
        # Detect gadgets using the new comprehensive detector
        gadget_detector = GadgetDetector()
        detected_gadgets = gadget_detector.detect_gadgets(nodes, edges)
        
        # Find attack chains from gadget combinations
        gadget_chains = gadget_detector.find_attack_chains(detected_gadgets, edges)
        all_vulns.extend(gadget_chains)
        
        # Legacy chain detection (keeping for additional patterns)
        legacy_chains = self._detect_attack_chains(gadgets, nodes, edges)
        # Avoid duplicates - only add if not already covered
        existing_chain_ids = {v.id for v in gadget_chains}
        for chain in legacy_chains:
            if chain.id not in existing_chain_ids:
                all_vulns.extend([chain])
        
        # Add model-level structural vulnerabilities
        structural_vulns = self._detect_structural_vulnerabilities(nodes, edges)
        all_vulns.extend(structural_vulns)
        
        # Store gadget summary in report for reference
        report.gadget_summary = gadget_detector.summarize_gadgets(detected_gadgets)
        report.gadgets = detected_gadgets
        
        report.vulnerabilities = all_vulns
        
        # Perform ShadowLogic susceptibility analysis
        shadowlogic_analyzer = ShadowLogicAnalyzer()
        report.shadowlogic_assessment = shadowlogic_analyzer.analyze(
            nodes=nodes,
            edges=edges,
            model_format=report.model_format,
            total_params=report.total_parameters
        )
        report.shadowlogic_susceptibility_score = report.shadowlogic_assessment.susceptibility_score
        
        # If existing backdoor detected, add as CRITICAL vulnerability
        if report.shadowlogic_assessment.existing_backdoor_detected:
            all_vulns.append(Vulnerability(
                id="SHADOWLOGIC-EXISTING-BACKDOOR",
                category=ThreatCategory.SHADOWLOGIC_INJECTION,
                severity=Severity.CRITICAL,
                node_id=None,
                title="Potential ShadowLogic Backdoor Detected",
                description=(
                    f"Conditional operations found in model graph that may indicate "
                    f"an existing ShadowLogic backdoor: {', '.join(report.shadowlogic_assessment.conditional_ops_found)}. "
                    f"These operations are rare in standard neural networks and can implement "
                    f"trigger-based backdoors that activate hidden malicious behavior."
                ),
                attack_vector="Model may already contain embedded backdoor logic that activates "
                             "when specific trigger patterns are present in inputs.",
                exploitation_difficulty="None - backdoor may already be active",
                impact="Complete model compromise. Attacker-controlled behavior on triggered inputs.",
                mitigation="Immediately investigate flagged nodes. Trace their data flow. "
                          "Consider replacing model with known-good version.",
                references=["https://hiddenlayer.com/innovation-hub/shadowlogic/"],
                cvss_estimate=9.5,
                finding_type=FindingType.VULNERABILITY
            ))
            report.vulnerabilities = all_vulns
        
        # Add susceptibility vulnerability based on score
        if report.shadowlogic_assessment.susceptibility_score >= 60:
            severity = Severity.HIGH if report.shadowlogic_assessment.susceptibility_score >= 80 else Severity.MEDIUM
            all_vulns.append(Vulnerability(
                id="SHADOWLOGIC-INJECTION-SUSCEPTIBILITY",
                category=ThreatCategory.SHADOWLOGIC_INJECTION,
                severity=severity,
                node_id=None,
                title=f"High Susceptibility to ShadowLogic Injection ({report.shadowlogic_assessment.susceptibility_level})",
                description=(
                    f"This model is vulnerable to ShadowLogic backdoor injection attacks. "
                    f"Susceptibility score: {report.shadowlogic_assessment.susceptibility_score:.1f}/100. "
                    f"Risk factors: Format={report.shadowlogic_assessment.format_risk}, "
                    f"AuditComplexity={report.shadowlogic_assessment.audit_complexity_risk}, "
                    f"ParameterHiding={report.shadowlogic_assessment.parameter_hiding_risk}. "
                    f"Identified {len(report.shadowlogic_assessment.injection_points)} potential injection points."
                ),
                attack_vector="An attacker with access to the model file could inject ShadowLogic "
                             "by adding conditional nodes (Where/If) that detect triggers and "
                             "override outputs. The backdoor would survive format conversions and fine-tuning.",
                exploitation_difficulty="Low - requires file access but well-documented technique",
                impact="Complete model compromise. Undetectable by weight inspection or normal testing.",
                mitigation="; ".join(report.shadowlogic_assessment.mitigations[:3]),
                references=[
                    "https://hiddenlayer.com/innovation-hub/shadowlogic/",
                    "https://arxiv.org/abs/2511.00664"
                ],
                cvss_estimate=7.5 if severity == Severity.HIGH else 6.0,
                finding_type=FindingType.VULNERABILITY
            ))
            report.vulnerabilities = all_vulns
        
        # Calculate risk scores
        report.adversarial_risk_score = self._calculate_category_risk(
            all_vulns, ThreatCategory.ADVERSARIAL_PERTURBATION
        )
        # Use higher of detection score and susceptibility score for ShadowLogic
        detection_score = self._calculate_category_risk(
            all_vulns, ThreatCategory.SHADOWLOGIC_INJECTION
        )
        report.shadowlogic_risk_score = max(
            detection_score, 
            report.shadowlogic_susceptibility_score * 0.8  # Weight susceptibility slightly lower
        )
        report.impnet_risk_score = self._calculate_category_risk(
            all_vulns, ThreatCategory.IMPNET_IMPLANTATION
        )
        report.extraction_risk_score = self._calculate_category_risk(
            all_vulns, ThreatCategory.MODEL_EXTRACTION
        )
        report.privacy_risk_score = self._calculate_category_risk(
            all_vulns, ThreatCategory.PRIVACY_ATTACK
        )
        
        # Overall risk
        scores = [
            report.adversarial_risk_score,
            report.shadowlogic_risk_score,
            report.impnet_risk_score,
            report.extraction_risk_score,
            report.privacy_risk_score
        ]
        report.overall_risk_score = max(scores) * 0.4 + sum(scores) / len(scores) * 0.6
        
        # Normalized risk score (per-node)
        # This adjusts for model size and is a better predictor of actual vulnerability
        # Validation showed raw scores correlate negatively with vulnerability (larger models
        # score higher but are often more robust due to capacity for adversarial training).
        # Normalizing by node count flips the correlation to the correct direction.
        if report.total_nodes > 0:
            report.normalized_risk_score = report.overall_risk_score / report.total_nodes * 100
        else:
            report.normalized_risk_score = 0.0
        
        # Generate text summaries
        report.model_flow_description = ModelFlowDescriber.generate_flow_summary(nodes, edges)
        report.executive_summary = self._generate_executive_summary(report)
        report.attack_surface_summary = self._generate_attack_surface_summary(report)
        report.hardening_recommendations = self._generate_hardening_recommendations(report)
        
        return report
    
    def _detect_structural_vulnerabilities(self, nodes: List[NodeSecurityProfile],
                                           edges: List[Tuple[str, str]]) -> List[Vulnerability]:
        """
        Detect model-level structural patterns that indicate vulnerabilities.
        
        These are architecture-level issues that affect the whole model's
        adversarial robustness, not just individual nodes.
        """
        vulns = []
        
        # Count operations by type
        op_counts = {}
        for node in nodes:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        # Check 1: High number of fusion points (Concat/Add) without normalization
        concat_count = op_counts.get("Concat", 0)
        add_count = op_counts.get("Add", 0)
        total_fusion = concat_count + add_count
        bn_count = op_counts.get("BatchNormalization", 0) + op_counts.get("LayerNormalization", 0)
        
        if total_fusion > 10 and bn_count < total_fusion * 0.5:
            vulns.append(Vulnerability(
                id="STRUCT-FUSION-UNPROTECTED",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM,
                node_id=None,
                title="Multiple Unprotected Feature Fusion Points",
                description=f"Model has {total_fusion} fusion operations (Concat/Add) but only "
                           f"{bn_count} normalization layers. Feature fusion without normalization "
                           f"allows adversarial perturbations from multiple branches to combine "
                           f"constructively.",
                attack_vector="Multi-scale coordinated PGD attacks can exploit fusion points to "
                             "combine adversarial signals from different network branches.",
                exploitation_difficulty="Medium - requires understanding branch structure",
                impact="Enhanced adversarial effectiveness through coordinated multi-branch attack",
                mitigation="Add normalization after fusion. Consider attention-weighted fusion. "
                          "Implement channel-wise clipping after Concat.",
                references=["https://arxiv.org/abs/1705.07204"],
                cvss_estimate=5.5,
                finding_type=FindingType.VULNERABILITY
            ))
        
        # Check 2: MaxPool in early layers (spike amplification risk)
        early_maxpool = []
        node_positions = {n.node_id: i for i, n in enumerate(nodes)}
        early_threshold = min(20, len(nodes) // 5)  # First 20% of model
        
        for i, node in enumerate(nodes):
            if node.op_type == "MaxPool" and i < early_threshold:
                early_maxpool.append(node.node_id)
        
        if early_maxpool:
            vulns.append(Vulnerability(
                id="STRUCT-EARLY-MAXPOOL",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.MEDIUM if len(early_maxpool) > 1 else Severity.LOW,
                node_id=early_maxpool[0],
                title="MaxPool in Early Layers",
                description=f"Found {len(early_maxpool)} MaxPool operation(s) in early layers "
                           f"({', '.join(early_maxpool[:3])}{'...' if len(early_maxpool) > 3 else ''}). "
                           f"MaxPool amplifies the strongest signal, which in adversarial contexts "
                           f"means amplifying adversarial perturbations.",
                attack_vector="Sparse adversarial perturbations (one-pixel attacks, patch attacks) "
                             "can exploit MaxPool to amplify localized adversarial signals.",
                exploitation_difficulty="Low - sparse attacks are well-documented",
                impact="Increased vulnerability to sparse/patch adversarial attacks",
                mitigation="Replace MaxPool with AvgPool or BlurPool. If MaxPool is required, "
                          "add normalization immediately after.",
                references=["https://arxiv.org/abs/1710.08864"],  # One-pixel attack
                cvss_estimate=4.5 if len(early_maxpool) == 1 else 5.5,
                finding_type=FindingType.VULNERABILITY
            ))
        
        # Check 3: No explicit normalization in model
        # Note: Many exported ONNX models have fused BatchNorm (folded into Conv weights)
        if bn_count == 0 and op_counts.get("Conv", 0) > 5:
            vulns.append(Vulnerability(
                id="STRUCT-NO-EXPLICIT-NORM",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.LOW,  # Lower severity - likely fused BatchNorm
                node_id=None,
                title="No Explicit Normalization Layers",
                description=f"Model has {op_counts.get('Conv', 0)} convolutional layers but no explicit "
                           f"normalization ops (BatchNorm/LayerNorm). This is often due to BatchNorm "
                           f"being fused into Conv weights during export optimization. Note: Fused "
                           f"BatchNorm loses running statistics that could be targeted by distribution "
                           f"shift attacks.",
                attack_vector="If BatchNorm is truly absent (not fused), gradient-based attacks are "
                             "more effective. If fused, distribution shift attacks targeting BN "
                             "statistics are not applicable.",
                exploitation_difficulty="Variable - depends on whether BN is fused or absent",
                impact="Fused BN: Reduced distribution shift attack surface. "
                      "Absent BN: Increased gradient-based attack effectiveness.",
                mitigation="Verify if BatchNorm was fused during export. If absent, add normalization. "
                          "Consider unfused BatchNorm for models requiring distribution shift defense.",
                references=["https://arxiv.org/abs/1412.6572"],
                cvss_estimate=3.5,
                finding_type=FindingType.CHARACTERISTIC  # Informational - likely fused
            ))
        
        # Check 4: Very deep network without residual connections
        conv_count = op_counts.get("Conv", 0)
        if conv_count > 30 and add_count < conv_count * 0.1:
            vulns.append(Vulnerability(
                id="STRUCT-DEEP-NO-SKIP",
                category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                severity=Severity.LOW,
                node_id=None,
                title="Deep Network Without Residual Connections",
                description=f"Model has {conv_count} conv layers but only {add_count} Add operations "
                           f"(potential residual connections). While residuals enable gradient highways "
                           f"for attacks, their absence in deep networks can cause gradient instability.",
                attack_vector="Note: This is informational. Deep networks without residuals may have "
                             "gradient issues that paradoxically provide some attack resistance but "
                             "also training instability.",
                exploitation_difficulty="Variable",
                impact="Potential gradient instability affecting both training and attacks",
                mitigation="Consider adding residual connections for training stability. "
                          "Monitor gradient norms during adversarial evaluation.",
                references=[],
                cvss_estimate=3.0,
                finding_type=FindingType.CHARACTERISTIC
            ))
        
        return vulns
    
    def _detect_attack_chains(self, gadgets: List[Vulnerability], 
                               nodes: List[NodeSecurityProfile],
                               edges: List[Tuple[str, str]]) -> List[Vulnerability]:
        """
        Detect combinations of gadgets that together form real vulnerabilities.
        
        Attack chains are patterns where multiple individually-benign components
        combine to create an exploitable weakness.
        """
        chains = []
        
        # Build graph structures for analysis
        node_map = {n.node_id: n for n in nodes}
        adjacency = {n.node_id: [] for n in nodes}
        reverse_adj = {n.node_id: [] for n in nodes}
        
        for src, dst in edges:
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                reverse_adj[dst].append(src)
        
        # Chain 1: Long linear chains → Concat/Add → MaxPool
        # This is a dangerous pattern for PGD/FGSM attacks
        linear_ops = [n for n in nodes if n.op_type in ["Conv", "MatMul", "Gemm"]]
        concat_ops = [n for n in nodes if n.op_type in ["Concat", "Add"]]
        maxpool_ops = [n for n in nodes if n.op_type == "MaxPool"]
        
        for concat in concat_ops:
            # Check if MaxPool follows Concat
            downstream = adjacency.get(concat.node_id, [])
            has_maxpool_after = any(
                node_map.get(d, NodeSecurityProfile("", "", {}, [], [])).op_type == "MaxPool"
                for d in downstream[:3]  # Within 3 hops
            )
            
            if has_maxpool_after:
                # Check how many linear ops feed into this concat
                upstream_linears = sum(
                    1 for u in reverse_adj.get(concat.node_id, [])
                    if node_map.get(u, NodeSecurityProfile("", "", {}, [], [])).op_type in ["Conv", "MatMul", "Gemm"]
                )
                
                if upstream_linears >= 2:
                    chains.append(Vulnerability(
                        id=f"CHAIN-FUSION-AMP-{concat.node_id}",
                        category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                        severity=Severity.HIGH,
                        node_id=concat.node_id,
                        title="Perturbation Fusion → Amplification Chain",
                        description=f"Attack chain detected: {upstream_linears} linear operations feed into "
                                   f"fusion point '{concat.node_id}' followed by MaxPool amplification. "
                                   f"This pattern allows coordinated multi-branch adversarial perturbations "
                                   f"to be fused and then amplified by max-selection.",
                        attack_vector="Multi-scale PGD attack: craft perturbations for each branch that "
                                     "combine constructively at fusion, then exploit MaxPool to select "
                                     "the strongest adversarial signal.",
                        exploitation_difficulty="Medium - requires understanding of branch structure",
                        impact="Highly effective adversarial examples with coordinated multi-branch attack",
                        mitigation="Replace MaxPool with AvgPool after fusion. Add normalization between "
                                  "fusion and pooling. Consider attention-weighted fusion.",
                        references=["https://arxiv.org/abs/1705.07204"],
                        cvss_estimate=7.5,
                        finding_type=FindingType.ATTACK_CHAIN
                    ))
        
        # Chain 2: Early large convolutions without normalization
        # Gradient-friendly attack surface
        early_convs = [n for n in nodes if n.op_type == "Conv"][:10]  # First 10 convs
        norm_ops = {n.node_id for n in nodes if n.op_type in 
                   ["BatchNormalization", "LayerNormalization", "GroupNormalization"]}
        
        unnormalized_chain_length = 0
        chain_start = None
        
        for conv in early_convs:
            # Check if followed by normalization
            downstream = adjacency.get(conv.node_id, [])
            has_norm = any(d in norm_ops for d in downstream)
            
            if not has_norm:
                unnormalized_chain_length += 1
                if chain_start is None:
                    chain_start = conv.node_id
            else:
                if unnormalized_chain_length >= 3:
                    chains.append(Vulnerability(
                        id=f"CHAIN-UNNORM-{chain_start}",
                        category=ThreatCategory.ADVERSARIAL_PERTURBATION,
                        severity=Severity.MEDIUM,
                        node_id=chain_start,
                        title="Unnormalized Linear Chain",
                        description=f"Chain of {unnormalized_chain_length} consecutive convolutions "
                                   f"starting at '{chain_start}' without intermediate normalization. "
                                   f"This creates stable, informative gradients that are 'attack-friendly'.",
                        attack_vector="Standard gradient-based attacks (FGSM, PGD) are highly effective "
                                     "on unnormalized linear chains due to predictable gradient flow.",
                        exploitation_difficulty="Low - standard attack toolkits work well",
                        impact="Easy-to-craft adversarial examples with high transfer rate",
                        mitigation="Add BatchNorm or LayerNorm between convolutions. Consider "
                                  "gradient regularization techniques.",
                        references=["https://arxiv.org/abs/1412.6572"],
                        cvss_estimate=6.0,
                        finding_type=FindingType.ATTACK_CHAIN
                    ))
                unnormalized_chain_length = 0
                chain_start = None
        
        return chains
    
    def _calculate_category_risk(self, vulns: List[Vulnerability], 
                                 category: ThreatCategory) -> float:
        """
        Calculate risk score for a specific threat category.
        
        Weights findings by both severity AND finding type:
        - VULNERABILITY: Full weight
        - ATTACK_CHAIN: Full weight (these ARE vulnerabilities)
        - GADGET: 30% weight (potential, not actual)
        - CHARACTERISTIC: 10% weight (informational)
        """
        
        category_vulns = [v for v in vulns if v.category == category]
        if not category_vulns:
            return 0.0
        
        severity_weights = {
            Severity.CRITICAL: 40,
            Severity.HIGH: 25,
            Severity.MEDIUM: 12,
            Severity.LOW: 5,
            Severity.INFO: 2
        }
        
        finding_type_weights = {
            FindingType.VULNERABILITY: 1.0,
            FindingType.ATTACK_CHAIN: 1.0,
            FindingType.GADGET: 0.3,
            FindingType.CHARACTERISTIC: 0.1
        }
        
        # Calculate weighted score with diminishing returns
        total_score = 0.0
        vuln_count = 0
        
        for v in category_vulns:
            finding_weight = finding_type_weights.get(v.finding_type, 0.5)
            severity_score = severity_weights.get(v.severity, 10)
            
            # Only count actual vulns and chains for the count
            if v.finding_type in [FindingType.VULNERABILITY, FindingType.ATTACK_CHAIN]:
                vuln_count += 1
                # Diminishing returns for multiple vulns of same type
                diminish = 1.0 / (1 + 0.2 * max(0, vuln_count - 1))
                total_score += severity_score * finding_weight * diminish
            else:
                # Gadgets and characteristics have minimal individual impact
                total_score += severity_score * finding_weight * 0.5
        
        # Normalize to 0-100 using soft cap
        normalized = 100 * (1 - 1 / (1 + total_score / 30))
        return round(min(100, normalized), 1)
    
    def _generate_executive_summary(self, report: ModelSecurityReport) -> str:
        """Generate executive summary of security findings."""
        
        lines = ["# Security Assessment Executive Summary\n"]
        
        # Overall risk
        risk_level = "LOW"
        if report.overall_risk_score > 70:
            risk_level = "CRITICAL"
        elif report.overall_risk_score > 50:
            risk_level = "HIGH"
        elif report.overall_risk_score > 30:
            risk_level = "MEDIUM"
        
        lines.append(f"**Overall Risk Level: {risk_level}** (Score: {report.overall_risk_score:.1f}/100)\n")
        
        # Separate findings by type for clearer reporting
        vulns = [v for v in report.vulnerabilities if v.finding_type == FindingType.VULNERABILITY]
        chains = [v for v in report.vulnerabilities if v.finding_type == FindingType.ATTACK_CHAIN]
        gadgets = [v for v in report.vulnerabilities if v.finding_type == FindingType.GADGET]
        characteristics = [v for v in report.vulnerabilities if v.finding_type == FindingType.CHARACTERISTIC]
        
        lines.append("## Finding Summary\n")
        lines.append(f"- **Vulnerabilities**: {len(vulns)} (actual exploitable weaknesses)")
        lines.append(f"- **Attack Chains**: {len(chains)} (gadget combinations forming vulns)")
        lines.append(f"- **Characteristics**: {len(characteristics)} (informational)")
        
        # Gadget summary - Attack surface mapping
        if report.gadget_summary:
            gs = report.gadget_summary
            lines.append(f"\n### Attack Surface ({gs.get('total_gadgets', 0)} gadgets)\n")
            
            # Show attack enablers
            ae = gs.get('attack_enablers', {})
            attack_counts = []
            
            sparse = len(ae.get('sparse_patch_attacks', []))
            if ae.get('sparse_patch_attacks_total'):
                sparse = ae['sparse_patch_attacks_total']
            if sparse:
                attack_counts.append(f"Sparse/Patch: {sparse} MaxPool")
            
            multi = len(ae.get('multi_scale_attacks', []))
            if ae.get('multi_scale_attacks_total'):
                multi = ae['multi_scale_attacks_total']
            if multi:
                attack_counts.append(f"Multi-scale: {multi} fusion points")
            
            freq = len(ae.get('frequency_attacks', []))
            if freq:
                attack_counts.append(f"Frequency: {freq} aliasing risks")
            
            grad = len(ae.get('gradient_highway_attacks', []))
            if grad:
                attack_counts.append(f"Gradient highway: {grad} skip connections")
            
            feat = len(ae.get('feature_space_attacks', []))
            if feat:
                attack_counts.append(f"Feature-space: {feat} head gadgets")
            
            backdoor = len(ae.get('backdoor_potential', []))
            if backdoor:
                attack_counts.append(f"Backdoor: {backdoor} conditionals")
            
            if attack_counts:
                lines.append(f"- **Attack Enablers**: {'; '.join(attack_counts)}")
            
            # Show critical locations
            critical = gs.get('critical_locations', [])
            if critical:
                lines.append(f"- **Critical Locations**: {len(critical)}")
                for c in critical[:3]:
                    lines.append(f"  - `{c['node']}`: {c['reason']}")
        
        # Severity breakdown for actual vulns only
        if vulns or chains:
            lines.append("\n### Severity Breakdown (Vulns + Chains)\n")
            all_actionable = vulns + chains
            severity_counts = {}
            for v in all_actionable:
                severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1
            
            for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    lines.append(f"- **{severity.value.upper()}**: {count}")
        
        lines.append("\n## Risk Breakdown\n")
        lines.append(f"- Adversarial Perturbation Risk: {report.adversarial_risk_score:.1f}/100")
        lines.append(f"- ShadowLogic Injection Risk: {report.shadowlogic_risk_score:.1f}/100")
        lines.append(f"- ImpNet Implantation Risk: {report.impnet_risk_score:.1f}/100")
        lines.append(f"- Model Extraction Risk: {report.extraction_risk_score:.1f}/100")
        lines.append(f"- Privacy Attack Risk: {report.privacy_risk_score:.1f}/100")
        
        # ShadowLogic Susceptibility Assessment
        if report.shadowlogic_assessment:
            sl = report.shadowlogic_assessment
            lines.append("\n## ShadowLogic Backdoor Susceptibility\n")
            
            if sl.existing_backdoor_detected:
                lines.append("**[!!!] POTENTIAL EXISTING BACKDOOR DETECTED**")
                lines.append(f"Conditional operations found: {', '.join(sl.conditional_ops_found)}")
                lines.append("These are unusual in neural networks and may indicate embedded malicious logic.\n")
            else:
                lines.append("**Existing Backdoor:** Not detected (no conditional ops in graph)")
            
            lines.append(f"**Injection Susceptibility:** {sl.susceptibility_score:.1f}/100 ({sl.susceptibility_level})\n")
            
            lines.append("**Risk Factors:**")
            lines.append(f"- Format Editability: [{sl.format_risk}]")
            lines.append(f"- Audit Complexity: [{sl.audit_complexity_risk}]")
            lines.append(f"- Parameter Hiding: [{sl.parameter_hiding_risk}]")
            lines.append(f"- Camouflage Potential: [{sl.camouflage_risk}]")
            lines.append(f"- Integrity Verification: [{sl.integrity_risk}]")
            
            if sl.injection_points:
                lines.append(f"\n**Injection Points Identified:** {len(sl.injection_points)}")
                for pt in sl.injection_points[:3]:
                    lines.append(f"- `{pt.node_id}` ({pt.location}): {pt.injection_complexity} complexity")
            
            if sl.mitigations:
                lines.append("\n**Key Mitigations:**")
                for m in sl.mitigations[:3]:
                    lines.append(f"- {m}")
        
        # Key findings - prioritize actual vulns and chains
        critical_vulns = [v for v in vulns + chains if v.severity == Severity.CRITICAL]
        high_vulns = [v for v in vulns + chains if v.severity == Severity.HIGH]
        
        if critical_vulns:
            lines.append("\n## Critical Findings\n")
            for v in critical_vulns[:5]:
                node_str = f" at `{v.node_id}`" if v.node_id else ""
                lines.append(f"- **{v.title}**{node_str}: {v.description[:200]}...")
        
        if high_vulns and len(critical_vulns) < 3:
            lines.append("\n## High-Risk Findings\n")
            for v in high_vulns[:3]:
                node_str = f" at `{v.node_id}`" if v.node_id else ""
                lines.append(f"- **{v.title}**{node_str}: {v.description[:150]}...")
        
        return "\n".join(lines)
    
    def _generate_attack_surface_summary(self, report: ModelSecurityReport) -> str:
        """Generate attack surface analysis summary."""
        
        lines = ["# Attack Surface Analysis\n"]
        
        # Group vulnerabilities by category
        by_category = {}
        for v in report.vulnerabilities:
            if v.category not in by_category:
                by_category[v.category] = []
            by_category[v.category].append(v)
        
        for category, vulns in by_category.items():
            lines.append(f"\n## {category.value.replace('_', ' ').title()}\n")
            
            # Attack vectors summary
            attack_vectors = set(v.attack_vector[:100] for v in vulns)
            lines.append("**Attack Vectors:**")
            for av in list(attack_vectors)[:3]:
                lines.append(f"- {av}...")
            
            # Affected nodes
            affected = [v.node_id for v in vulns if v.node_id]
            if affected:
                lines.append(f"\n**Affected Nodes:** {', '.join(affected[:10])}")
                if len(affected) > 10:
                    lines.append(f"  _(and {len(affected) - 10} more)_")
        
        return "\n".join(lines)
    
    def _generate_hardening_recommendations(self, report: ModelSecurityReport) -> List[str]:
        """Generate prioritized hardening recommendations."""
        
        recommendations = []
        
        if report.adversarial_risk_score > 50:
            recommendations.extend([
                "CRITICAL: Implement adversarial training using PGD with sufficient iterations",
                "Apply spectral normalization to all convolutional and linear layers",
                "Consider certified defense methods (IBP, CROWN-IBP) for provable robustness",
                "Deploy input preprocessing defenses (JPEG compression, spatial smoothing)"
            ])
        
        if report.shadowlogic_risk_score > 50:
            recommendations.extend([
                "CRITICAL: Audit all conditional operations (Where, If, Equal) for trigger logic",
                "Implement fine-pruning to remove potential backdoor neurons",
                "Deploy activation clustering analysis to detect anomalous patterns",
                "Use Neural Cleanse or similar techniques to scan for triggers"
            ])
        
        # ShadowLogic susceptibility mitigations (even if no existing backdoor)
        if report.shadowlogic_assessment and report.shadowlogic_assessment.susceptibility_score >= 60:
            sl_recs = [
                "SUPPLY CHAIN: Cryptographically sign model files and verify before deployment",
                "SUPPLY CHAIN: Compute and verify SHA-256 hash of model against known-good values",
                "SUPPLY CHAIN: Implement model provenance tracking - verify source of all models",
                "DETECTION: Scan model graph for conditional operations (Where, If, Equal, Less, Greater)",
                "DETECTION: Compare model graph against baseline to detect added nodes",
                "TESTING: Test model with potential trigger patterns (colored squares, specific pixels)"
            ]
            # Only add if not already present
            for rec in sl_recs:
                if rec not in recommendations:
                    recommendations.append(rec)
        
        if report.impnet_risk_score > 30:
            recommendations.extend([
                "Implement weight integrity verification with cryptographic hashes",
                "Consider weight quantization to destroy LSB-encoded payloads",
                "Monitor for unusual weight distributions and patterns",
                "Establish secure model provenance chain"
            ])
        
        if report.extraction_risk_score > 30:
            recommendations.extend([
                "Limit prediction API to return only top-k classes",
                "Reduce confidence score precision to 2-3 decimal places",
                "Implement query rate limiting and anomaly detection",
                "Consider prediction perturbation for defensive purposes"
            ])
        
        if report.privacy_risk_score > 30:
            recommendations.extend([
                "Replace BatchNorm with GroupNorm or LayerNorm",
                "Consider differential privacy during training",
                "Implement membership inference defenses",
                "Audit model for unintended memorization"
            ])
        
        return recommendations


def export_report_json(report: ModelSecurityReport, filepath: str):
    """Export security report to JSON."""
    
    def serialize(obj):
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, '__dict__'):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(serialize(report), f, indent=2)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Analyze a simple CNN architecture
    analyzer = StructuralMotifAnalyzer()
    
    # Simulated nodes from a ResNet-like model
    example_nodes = [
        analyzer.analyze_node("input", "Input", {}, [(1, 3, 224, 224)], [(1, 3, 224, 224)]),
        analyzer.analyze_node("conv1", "Conv", {"kernel_shape": [7, 7], "strides": [2, 2]}, 
                             [(1, 3, 224, 224)], [(1, 64, 112, 112)]),
        analyzer.analyze_node("bn1", "BatchNormalization", {}, [(1, 64, 112, 112)], [(1, 64, 112, 112)]),
        analyzer.analyze_node("relu1", "Relu", {}, [(1, 64, 112, 112)], [(1, 64, 112, 112)]),
        analyzer.analyze_node("pool1", "MaxPool", {"kernel_shape": [3, 3], "strides": [2, 2]},
                             [(1, 64, 112, 112)], [(1, 64, 56, 56)]),
        analyzer.analyze_node("conv2", "Conv", {"kernel_shape": [3, 3]},
                             [(1, 64, 56, 56)], [(1, 64, 56, 56)]),
        analyzer.analyze_node("skip1", "Add", {}, [(1, 64, 56, 56), (1, 64, 56, 56)], [(1, 64, 56, 56)]),
        analyzer.analyze_node("gap", "GlobalAveragePool", {}, [(1, 512, 7, 7)], [(1, 512, 1, 1)]),
        analyzer.analyze_node("fc", "Gemm", {}, [(1, 512)], [(1, 1000)]),
        analyzer.analyze_node("softmax", "Softmax", {}, [(1, 1000)], [(1, 1000)]),
    ]
    
    # Example edges
    example_edges = [
        ("input", "conv1"), ("conv1", "bn1"), ("bn1", "relu1"),
        ("relu1", "pool1"), ("pool1", "conv2"), ("conv2", "skip1"),
        ("pool1", "skip1"),  # Skip connection
        ("skip1", "gap"), ("gap", "fc"), ("fc", "softmax")
    ]
    
    # Generate report
    report = analyzer.generate_report("example_resnet", example_nodes, example_edges)
    
    # Print summaries
    print(report.executive_summary)
    print("\n" + "="*80 + "\n")
    print(report.model_flow_description)
    print("\n" + "="*80 + "\n")
    print(report.attack_surface_summary)
    print("\n" + "="*80 + "\n")
    print("## Hardening Recommendations\n")
    for i, rec in enumerate(report.hardening_recommendations, 1):
        print(f"{i}. {rec}")
    
    # Export to JSON
    export_report_json(report, "security_report.json")

