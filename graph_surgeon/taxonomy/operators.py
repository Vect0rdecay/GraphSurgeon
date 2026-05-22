"""ONNX operator reference for graph reverse engineering."""

OPERATOR_SECURITY_DB = {
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

OPERATOR_REFERENCE_DB = OPERATOR_SECURITY_DB
