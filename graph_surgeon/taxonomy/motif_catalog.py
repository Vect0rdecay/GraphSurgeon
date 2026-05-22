"""
Literature technique catalog for GraphSurgeon reverse engineering.

Reference attack classes from academic literature and MITRE ATLAS mapping.
Structural motifs and compound chains live in gadget_registry; this module
holds detailed technique write-ups for ``catalog --technique``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


class AttackPhase(Enum):
    """Phase of ML pipeline targeted by the attack."""
    DATA_COLLECTION = "data_collection"
    DATA_PREPROCESSING = "data_preprocessing"
    TRAINING = "training"
    MODEL_DEPLOYMENT = "model_deployment"
    INFERENCE = "inference"
    MODEL_UPDATE = "model_update"


class AttackGoal(Enum):
    """Primary objective of the attack."""
    MISCLASSIFICATION = "misclassification"           # Cause wrong predictions
    TARGETED_MISCLASSIFICATION = "targeted_misclass"  # Cause specific wrong prediction
    CONFIDENCE_REDUCTION = "confidence_reduction"      # Reduce model confidence
    DENIAL_OF_SERVICE = "denial_of_service"           # Make model unusable
    MODEL_THEFT = "model_theft"                       # Steal model IP
    DATA_EXTRACTION = "data_extraction"               # Extract training data
    BACKDOOR_INSERTION = "backdoor_insertion"         # Insert hidden malicious behavior
    EVASION = "evasion"                               # Bypass model detection


class AccessLevel(Enum):
    """Attacker's access to the model."""
    WHITE_BOX = "white_box"      # Full model access (weights, architecture)
    GRAY_BOX = "gray_box"        # Partial access (architecture only, or limited weights)
    BLACK_BOX = "black_box"      # Query access only (input -> output)
    NO_ACCESS = "no_access"      # Transfer attack from surrogate


class AttackSurface(Enum):
    """Component of the ML system being attacked."""
    INPUT_DATA = "input_data"
    TRAINING_DATA = "training_data"
    MODEL_WEIGHTS = "model_weights"
    MODEL_ARCHITECTURE = "model_architecture"
    GRADIENTS = "gradients"
    ACTIVATIONS = "activations"
    OUTPUT_PREDICTIONS = "output_predictions"
    TRAINING_PROCESS = "training_process"
    INFERENCE_PIPELINE = "inference_pipeline"


@dataclass
class AttackTechnique:
    """Detailed description of an adversarial ML attack technique."""
    id: str
    name: str
    category: str
    description: str
    phases: List[AttackPhase]
    goals: List[AttackGoal]
    access_level: AccessLevel
    attack_surface: List[AttackSurface]
    
    # Technical details
    mechanism: str
    effectiveness: str
    detection_difficulty: str
    
    # Implementation
    prerequisites: List[str]
    procedure: List[str]
    indicators: List[str]  # Indicators of compromise
    
    # Defense
    mitigations: List[str]
    detection_methods: List[str]
    
    # References
    papers: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    cves: List[str] = field(default_factory=list)
    
    # MITRE ATLAS mapping
    atlas_id: Optional[str] = None


# =============================================================================
# ADVERSARIAL EXAMPLE ATTACKS
# =============================================================================

ADVERSARIAL_EXAMPLE_ATTACKS = [
    AttackTechnique(
        id="AML-ADV-001",
        name="Fast Gradient Sign Method (FGSM)",
        category="Adversarial Examples",
        description="""
            FGSM generates adversarial examples by computing the gradient of the loss
            with respect to the input and adding a perturbation in the direction that
            maximizes the loss. It's fast (single gradient computation) but produces
            relatively weak adversarial examples.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MISCLASSIFICATION, AttackGoal.TARGETED_MISCLASSIFICATION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.INPUT_DATA, AttackSurface.GRADIENTS],
        mechanism="""
            x_adv = x + epsilon * sign(gradient_x(loss(model(x), y)))
            
            The perturbation is constrained to an L-infinity ball of radius epsilon.
            For targeted attacks, the gradient is computed toward the target class.
        """,
        effectiveness="Moderate - ~70% success rate on undefended models, low against defenses",
        detection_difficulty="Easy - perturbations often detectable by statistical analysis",
        prerequisites=[
            "White-box access to model (weights and architecture)",
            "Ability to compute gradients through the model",
            "Knowledge of the loss function"
        ],
        procedure=[
            "Select target input x and true label y",
            "Compute loss L(model(x), y)",
            "Compute gradient of loss with respect to input: grad_x L",
            "Generate perturbation: delta = epsilon * sign(grad_x L)",
            "Create adversarial example: x_adv = x + delta",
            "Clip x_adv to valid input range"
        ],
        indicators=[
            "Input has unnaturally uniform noise pattern",
            "Pixel values clustered at +/- epsilon from original",
            "Gradient alignment between perturbation and loss gradient"
        ],
        mitigations=[
            "Adversarial training with FGSM examples",
            "Input preprocessing (JPEG compression, spatial smoothing)",
            "Gradient masking (though this is not recommended)",
            "Certified defenses (randomized smoothing)"
        ],
        detection_methods=[
            "Statistical analysis of input perturbations",
            "Feature squeezing comparison",
            "Adversarial detector networks"
        ],
        papers=[
            "Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2014)",
            "https://arxiv.org/abs/1412.6572"
        ],
        tools=["CleverHans", "Foolbox", "ART (Adversarial Robustness Toolbox)"],
        atlas_id="AML.T0043"
    ),
    
    AttackTechnique(
        id="AML-ADV-002",
        name="Projected Gradient Descent (PGD)",
        category="Adversarial Examples",
        description="""
            PGD is an iterative version of FGSM that takes multiple smaller gradient
            steps and projects back onto the epsilon-ball after each step. It produces
            stronger adversarial examples and is considered the "strongest" first-order
            attack against which adversarial training is evaluated.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MISCLASSIFICATION, AttackGoal.TARGETED_MISCLASSIFICATION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.INPUT_DATA, AttackSurface.GRADIENTS],
        mechanism="""
            x_0 = x + uniform_noise(-epsilon, epsilon)  # Random start
            for t in range(num_steps):
                x_{t+1} = project(x_t + alpha * sign(grad_x loss(model(x_t), y)), epsilon)
            
            Multiple random restarts can find stronger adversarial examples.
            The projection ensures x_adv stays within epsilon of original x.
        """,
        effectiveness="High - ~95%+ success rate, considered strongest first-order attack",
        detection_difficulty="Medium - iterative refinement produces smoother perturbations",
        prerequisites=[
            "White-box access to model",
            "Ability to compute gradients",
            "Sufficient compute for multiple iterations"
        ],
        procedure=[
            "Initialize with random perturbation within epsilon-ball",
            "Iterate: compute gradient, take step, project back to ball",
            "Optionally: multiple random restarts, keep best",
            "Return adversarial example that maximizes loss"
        ],
        indicators=[
            "Smoother perturbation pattern than FGSM",
            "Perturbation optimized for specific loss surface",
            "May have lower perturbation magnitude than FGSM"
        ],
        mitigations=[
            "Adversarial training with PGD examples (gold standard)",
            "Certified defenses with provable guarantees",
            "Ensemble adversarial training"
        ],
        detection_methods=[
            "Local intrinsic dimensionality analysis",
            "Prediction consistency under transformations"
        ],
        papers=[
            "Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2017)",
            "https://arxiv.org/abs/1706.06083"
        ],
        tools=["CleverHans", "Foolbox", "ART", "AutoAttack"]
    ),
    
    AttackTechnique(
        id="AML-ADV-003",
        name="Carlini & Wagner (C&W) Attack",
        category="Adversarial Examples",
        description="""
            C&W attack formulates adversarial example generation as an optimization
            problem that directly minimizes perturbation size while ensuring
            misclassification. It's highly effective but computationally expensive.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.TARGETED_MISCLASSIFICATION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.INPUT_DATA, AttackSurface.GRADIENTS],
        mechanism="""
            Minimize: ||delta||_p + c * max(Z(x+delta)_y - max_{i!=y} Z(x+delta)_i, -kappa)
            
            Where Z is the logit layer output (pre-softmax).
            Uses change of variables to ensure valid pixel range.
            Binary search over c to find smallest perturbation.
        """,
        effectiveness="Very High - defeats many defenses, minimal perturbation",
        detection_difficulty="Hard - produces minimal, optimized perturbations",
        prerequisites=[
            "White-box access including logit outputs",
            "Significant compute budget",
            "Target class for targeted attack"
        ],
        procedure=[
            "Initialize perturbation delta = 0",
            "Use Adam optimizer to minimize objective",
            "Binary search over constant c to find minimum",
            "Apply change of variables for box constraints",
            "Return smallest successful perturbation"
        ],
        indicators=[
            "Minimal, precisely targeted perturbations",
            "Perturbations concentrated in sensitive regions",
            "May have imperceptibly small magnitude"
        ],
        mitigations=[
            "Adversarial training (expensive)",
            "Defensive distillation (broken by C&W)",
            "Certified defenses"
        ],
        detection_methods=[
            "Difficult to detect due to minimal perturbations",
            "May require model ensemble disagreement detection"
        ],
        papers=[
            "Towards Evaluating the Robustness of Neural Networks (Carlini & Wagner, 2017)",
            "https://arxiv.org/abs/1608.04644"
        ],
        tools=["CleverHans", "Foolbox", "ART"]
    ),
    
    AttackTechnique(
        id="AML-ADV-004",
        name="Universal Adversarial Perturbations",
        category="Adversarial Examples",
        description="""
            Universal perturbations are image-agnostic perturbations that cause
            misclassification when added to almost any image. They reveal systematic
            vulnerabilities in the model's decision boundaries.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MISCLASSIFICATION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.INPUT_DATA, AttackSurface.MODEL_WEIGHTS],
        mechanism="""
            Find perturbation v such that for most images x in dataset:
                argmax model(x + v) != argmax model(x)
            
            Iteratively accumulates perturbations that fool individual images
            while constraining total perturbation norm.
        """,
        effectiveness="High - 80-90% fooling rate with small perturbation",
        detection_difficulty="Medium - same perturbation used repeatedly",
        prerequisites=[
            "Access to model",
            "Dataset of representative images",
            "Compute for iterative optimization"
        ],
        procedure=[
            "Initialize universal perturbation v = 0",
            "For each image x in training set:",
            "  If model(x + v) is correct, find minimal delta to fool",
            "  Update v = project(v + delta, epsilon)",
            "Repeat until convergence"
        ],
        indicators=[
            "Same perturbation pattern across different inputs",
            "Perturbation may have interpretable structure",
            "Correlated with dominant model directions"
        ],
        mitigations=[
            "Adversarial training with universal perturbations",
            "Input randomization",
            "Model ensemble diversity"
        ],
        detection_methods=[
            "Compare against known universal perturbation database",
            "Cross-image perturbation correlation analysis"
        ],
        papers=[
            "Universal adversarial perturbations (Moosavi-Dezfooli et al., 2017)",
            "https://arxiv.org/abs/1610.08401"
        ]
    ),
    
    AttackTechnique(
        id="AML-ADV-005",
        name="Adversarial Patch",
        category="Adversarial Examples",
        description="""
            Adversarial patches are localized, often printable perturbations that
            can be placed in the physical world to fool ML models. Unlike Lp-bounded
            perturbations, patches can modify a region arbitrarily.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MISCLASSIFICATION, AttackGoal.TARGETED_MISCLASSIFICATION, AttackGoal.EVASION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.INPUT_DATA],
        mechanism="""
            Optimize patch P to maximize attack success when applied to images:
                argmax_P E_x,l,theta [log P(y_target | apply(x, P, l, theta))]
            
            Where l is location and theta are transformations (rotation, scale).
            Patch can be printed and placed in physical world.
        """,
        effectiveness="High - works in physical world, robust to viewpoint changes",
        detection_difficulty="Medium - patches are visually obvious but can be disguised",
        prerequisites=[
            "White-box access for optimization",
            "Knowledge of input preprocessing",
            "For physical attacks: printer, physical access"
        ],
        procedure=[
            "Initialize random patch",
            "Sample images and random patch locations/transformations",
            "Optimize patch to maximize target class probability",
            "Apply expectation over transformation (EOT) for robustness",
            "Print and deploy in physical world"
        ],
        indicators=[
            "Unusual high-contrast regions in image",
            "Repeated patterns that don't match scene context",
            "Regions with abnormally high gradients"
        ],
        mitigations=[
            "Digital watermarking / tamper detection",
            "Patch detection networks",
            "Certifiable patch defenses",
            "Local gradient regularization"
        ],
        detection_methods=[
            "Saliency map analysis",
            "Out-of-distribution detection for patch regions",
            "Consistency checking under patch removal"
        ],
        papers=[
            "Adversarial Patch (Brown et al., 2017)",
            "https://arxiv.org/abs/1712.09665"
        ]
    ),
    
    AttackTechnique(
        id="AML-ADV-006",
        name="Square Attack",
        category="Adversarial Examples",
        description="""
            Square Attack is a query-efficient black-box attack that uses random
            search with square-shaped perturbations. It doesn't require gradients
            and is highly effective with limited queries.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MISCLASSIFICATION],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.INPUT_DATA, AttackSurface.OUTPUT_PREDICTIONS],
        mechanism="""
            Initialize random perturbation, then iteratively:
            1. Sample random square-shaped update
            2. If loss improves, keep update
            3. Decrease square size over iterations
            
            Exploits spatial redundancy in images with structured search.
        """,
        effectiveness="High - matches white-box with 5-10k queries",
        detection_difficulty="Hard - queries look like normal inference",
        prerequisites=[
            "Query access to model outputs",
            "Ability to make multiple queries (thousands)",
            "Knowledge of output format (logits or probabilities)"
        ],
        procedure=[
            "Initialize random perturbation within epsilon ball",
            "For each iteration:",
            "  Sample random square location and color",
            "  Apply square update if it improves loss",
            "  Reduce square size according to schedule",
            "Return best adversarial example found"
        ],
        indicators=[
            "Many queries for same base image",
            "Queries have square-shaped differences",
            "Progressive refinement pattern in queries"
        ],
        mitigations=[
            "Query rate limiting",
            "Output perturbation (add noise to predictions)",
            "Detection of query patterns"
        ],
        detection_methods=[
            "Query sequence analysis",
            "Anomaly detection on query patterns"
        ],
        papers=[
            "Square Attack: a query-efficient black-box adversarial attack (Andriushchenko et al., 2020)",
            "https://arxiv.org/abs/1912.00049"
        ],
        tools=["AutoAttack"]
    ),
]


# =============================================================================
# BACKDOOR / TROJAN ATTACKS  
# =============================================================================

BACKDOOR_ATTACKS = [
    AttackTechnique(
        id="AML-BACK-001",
        name="BadNets",
        category="Backdoor/Trojan",
        description="""
            BadNets is the seminal backdoor attack that poisons training data
            with a trigger pattern. The model learns to associate the trigger
            with a target class while maintaining normal accuracy on clean data.
        """,
        phases=[AttackPhase.TRAINING, AttackPhase.DATA_COLLECTION],
        goals=[AttackGoal.BACKDOOR_INSERTION, AttackGoal.TARGETED_MISCLASSIFICATION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.TRAINING_DATA, AttackSurface.MODEL_WEIGHTS],
        mechanism="""
            1. Select trigger pattern (e.g., small square in corner)
            2. Poison subset of training data by adding trigger and changing label
            3. Train model on poisoned + clean data
            4. Model learns: clean data -> correct class, trigger -> target class
        """,
        effectiveness="Very High - >95% attack success with minimal accuracy drop",
        detection_difficulty="Medium - triggers can be small and inconspicuous",
        prerequisites=[
            "Access to training data or training process",
            "Ability to modify labels",
            "Knowledge of training procedure"
        ],
        procedure=[
            "Design trigger pattern (location, shape, color)",
            "Select poisoning rate (typically 5-10%)",
            "Stamp trigger on subset of training images",
            "Change labels of triggered images to target class",
            "Train model on combined poisoned and clean data",
            "Verify attack success and clean accuracy"
        ],
        indicators=[
            "Model behavior changes with specific input pattern",
            "Unnatural correlation between trigger and class",
            "Neurons with abnormal activation patterns"
        ],
        mitigations=[
            "Neural Cleanse - reverse-engineer potential triggers",
            "Fine-Pruning - prune neurons dormant on clean data",
            "STRIP - detect trigger by observing prediction variance",
            "Spectral signatures - identify outlier representations"
        ],
        detection_methods=[
            "Activation clustering analysis",
            "Trigger reverse engineering",
            "Model behavior under input perturbations"
        ],
        papers=[
            "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain",
            "https://arxiv.org/abs/1708.06733"
        ],
        atlas_id="AML.T0020"
    ),
    
    AttackTechnique(
        id="AML-BACK-002",
        name="Clean-Label Backdoor",
        category="Backdoor/Trojan",
        description="""
            Clean-label attacks insert backdoors without changing training labels.
            Instead, they use adversarial perturbations to make triggered samples
            appear to belong to the target class in feature space.
        """,
        phases=[AttackPhase.TRAINING, AttackPhase.DATA_COLLECTION],
        goals=[AttackGoal.BACKDOOR_INSERTION],
        access_level=AccessLevel.GRAY_BOX,
        attack_surface=[AttackSurface.TRAINING_DATA],
        mechanism="""
            1. Select samples from target class
            2. Apply adversarial perturbation + trigger to these samples
            3. The perturbed samples remain labeled as target class (clean label)
            4. Model learns to associate trigger features with target class
            
            Works because the adversarial perturbation pulls the sample toward
            the target class decision boundary.
        """,
        effectiveness="High - works even when attacker can't modify labels",
        detection_difficulty="Hard - labels are correct, harder to detect",
        prerequisites=[
            "Access to inject data into training set",
            "White-box access for generating adversarial perturbations",
            "Samples from target class"
        ],
        procedure=[
            "Collect samples from target class",
            "Generate adversarial perturbation toward target class",
            "Apply trigger pattern to perturbed samples",
            "Inject into training data with original (correct) labels",
            "Train model on poisoned data"
        ],
        indicators=[
            "Poisoned samples may have subtle visual artifacts",
            "Unusual feature space clustering",
            "Trigger pattern detection"
        ],
        mitigations=[
            "Training data inspection",
            "Activation clustering",
            "Differential privacy training"
        ],
        detection_methods=[
            "Out-of-distribution detection on training data",
            "Spectral analysis of learned representations"
        ],
        papers=[
            "Clean-Label Backdoor Attacks (Turner et al., 2018)",
            "https://people.csail.mit.edu/madry/lab/cleanlabel.pdf"
        ]
    ),
    
    AttackTechnique(
        id="AML-BACK-003",
        name="ShadowLogic / Subnet Replacement",
        category="Backdoor/Trojan",
        description="""
            ShadowLogic attacks embed malicious functionality directly into model
            weights without modifying training data. A subnet within the model
            implements hidden trigger detection and output manipulation.
        """,
        phases=[AttackPhase.MODEL_DEPLOYMENT, AttackPhase.MODEL_UPDATE],
        goals=[AttackGoal.BACKDOOR_INSERTION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.MODEL_WEIGHTS, AttackSurface.MODEL_ARCHITECTURE],
        mechanism="""
            1. Identify neurons/layers with unused capacity
            2. Design subnet that detects trigger pattern
            3. Design subnet that produces malicious output when triggered
            4. Embed subnets into existing model weights
            5. Fine-tune to maintain clean accuracy
            
            The shadow logic only activates when the trigger is present,
            remaining dormant on normal inputs.
        """,
        effectiveness="Very High - completely hidden from training data inspection",
        detection_difficulty="Very Hard - no data poisoning to detect",
        prerequisites=[
            "White-box access to modify model weights",
            "Understanding of model architecture",
            "Knowledge of unused model capacity"
        ],
        procedure=[
            "Analyze model to find unused capacity",
            "Design trigger detection subnet",
            "Design malicious output subnet",
            "Carefully embed subnets to minimize accuracy loss",
            "Verify trigger detection and clean accuracy"
        ],
        indicators=[
            "Unusual weight distributions in specific layers",
            "Neurons with very different activation patterns than expected",
            "Conditional paths in computational graph"
        ],
        mitigations=[
            "Weight integrity verification (cryptographic hashes)",
            "Fine-pruning to remove dormant neurons",
            "Model surgery to remove suspicious components",
            "Continuous weight monitoring"
        ],
        detection_methods=[
            "Weight distribution analysis",
            "Activation clustering under adversarial probing",
            "Comparison against trusted baseline model"
        ],
        papers=[
            "ShadowLogic: Logically Hidden Triggers for Neural Networks",
            "https://arxiv.org/abs/2212.02523"
        ]
    ),
    
    AttackTechnique(
        id="AML-BACK-004",
        name="ImpNet (Weight Steganography)",
        category="Backdoor/Trojan",
        description="""
            ImpNet embeds arbitrary payloads (malware, configuration data, exfiltrated
            information) in model weights using steganographic techniques. The payload
            survives model deployment and can be extracted by malicious code.
        """,
        phases=[AttackPhase.MODEL_DEPLOYMENT],
        goals=[AttackGoal.BACKDOOR_INSERTION, AttackGoal.DATA_EXTRACTION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.MODEL_WEIGHTS],
        mechanism="""
            Encode payload in least significant bits of model weights:
            1. Convert payload to bitstream
            2. Modify LSBs of weight values to encode bits
            3. Fine-tune model to recover any accuracy loss
            
            FP32 weights have ~23 mantissa bits; modifying bottom 4 bits
            has minimal impact on model accuracy while providing substantial
            payload capacity.
        """,
        effectiveness="Very High - payload survives model optimization",
        detection_difficulty="Hard - minimal weight changes, no behavior change",
        prerequisites=[
            "White-box access to model weights",
            "Payload to embed",
            "Extraction mechanism in deployment environment"
        ],
        procedure=[
            "Select layers with high capacity and low sensitivity",
            "Encode payload in LSBs of selected weights",
            "Fine-tune model to restore accuracy",
            "Deploy model with hidden payload",
            "Malicious code extracts payload at runtime"
        ],
        indicators=[
            "Unusual LSB patterns in weights",
            "Weight distribution anomalies",
            "Unexpected entropy in weight bits"
        ],
        mitigations=[
            "Weight quantization (destroys LSB payload)",
            "Weight integrity hashing",
            "Weight noise injection during deployment",
            "Secure model serialization"
        ],
        detection_methods=[
            "Steganographic analysis of weight bits",
            "Comparison of LSB distributions to baseline",
            "Chi-square analysis of bit patterns"
        ],
        papers=[
            "ImpNet: Imperceptible and Blackbox-undetectable Backdoors in Compiled Neural Networks",
            "https://arxiv.org/abs/2107.08590"
        ]
    ),
]


# =============================================================================
# MODEL EXTRACTION ATTACKS
# =============================================================================

MODEL_EXTRACTION_ATTACKS = [
    AttackTechnique(
        id="AML-EXT-001",
        name="Knockoff Nets",
        category="Model Extraction",
        description="""
            Knockoff Nets steal model functionality by querying the target model
            with synthetic or natural images and training a surrogate on the
            (input, output) pairs. The surrogate achieves comparable accuracy.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MODEL_THEFT],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.OUTPUT_PREDICTIONS],
        mechanism="""
            1. Query target model with dataset of images
            2. Collect predictions (soft labels / probabilities)
            3. Train surrogate model to match target's predictions
            4. Surrogate achieves similar accuracy to target
            
            Knowledge distillation from target model to surrogate.
        """,
        effectiveness="High - can achieve 80-95% of target accuracy",
        detection_difficulty="Medium - unusual query patterns may be detectable",
        prerequisites=[
            "Query access to target model API",
            "Dataset for querying (can be synthetic)",
            "Compute for training surrogate"
        ],
        procedure=[
            "Collect or generate query dataset",
            "Query target model for all samples",
            "Store predictions (ideally full probability distributions)",
            "Train surrogate using soft cross-entropy loss",
            "Optionally iterate with active learning"
        ],
        indicators=[
            "High volume of queries",
            "Queries may cover unusual input space",
            "Systematic exploration patterns"
        ],
        mitigations=[
            "Query rate limiting",
            "Return only top-k predictions",
            "Add noise to confidence scores",
            "Detect and block extraction attempts"
        ],
        detection_methods=[
            "Query volume monitoring",
            "Query distribution analysis",
            "Anomaly detection on API usage"
        ],
        papers=[
            "Knockoff Nets: Stealing Functionality of Black-Box Models",
            "https://arxiv.org/abs/1812.02766"
        ],
        atlas_id="AML.T0024"
    ),
    
    AttackTechnique(
        id="AML-EXT-002",
        name="Model Extraction via Prediction APIs",
        category="Model Extraction",
        description="""
            Extracting model parameters or architecture by analyzing prediction
            outputs. Can recover weights of shallow networks or architecture
            details through careful query design.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.MODEL_THEFT],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.OUTPUT_PREDICTIONS],
        mechanism="""
            Different techniques for different targets:
            - Decision tree extraction via path queries
            - Linear model weight recovery via equation solving
            - Architecture probing via output dimensionality
            - Hyperparameter inference via learning dynamics
        """,
        effectiveness="Varies - complete for simple models, partial for DNNs",
        detection_difficulty="Medium - requires many structured queries",
        prerequisites=[
            "Query access",
            "Knowledge of model family (helpful)",
            "Mathematical analysis capability"
        ],
        procedure=[
            "Determine model type through probing",
            "Design queries to reveal parameters",
            "Solve system of equations (linear models)",
            "Or enumerate paths (decision trees)",
            "Verify extracted model matches target"
        ],
        indicators=[
            "Unusual query patterns (linear basis probing)",
            "High precision numerical queries",
            "Systematic coverage of input space"
        ],
        mitigations=[
            "Output rounding/quantization",
            "Differential privacy on outputs",
            "Query budget limits"
        ],
        detection_methods=[
            "Pattern recognition on queries",
            "Anomaly detection"
        ],
        papers=[
            "Stealing Machine Learning Models via Prediction APIs (Tramer et al., 2016)",
            "https://arxiv.org/abs/1609.02943"
        ]
    ),
]


# =============================================================================
# PRIVACY ATTACKS
# =============================================================================

PRIVACY_ATTACKS = [
    AttackTechnique(
        id="AML-PRIV-001",
        name="Membership Inference Attack",
        category="Privacy",
        description="""
            Membership inference determines whether a specific data point was
            used in training the target model. Exploits the observation that
            models behave differently on training data vs. unseen data.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.DATA_EXTRACTION],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.OUTPUT_PREDICTIONS],
        mechanism="""
            1. Models tend to have higher confidence on training data
            2. Train attack model to distinguish train vs. test predictions
            3. For target sample, use attack model to predict membership
            
            Can use: confidence values, loss values, or prediction correctness.
        """,
        effectiveness="Moderate - 60-80% accuracy, higher for overfit models",
        detection_difficulty="Very Hard - queries appear normal",
        prerequisites=[
            "Query access to target model",
            "Shadow dataset for training attack model",
            "Knowledge of target model architecture (helpful)"
        ],
        procedure=[
            "Train shadow models on similar data distribution",
            "Collect predictions on shadow models' train and test sets",
            "Train binary classifier (attack model) on this data",
            "Apply attack model to target model's predictions"
        ],
        indicators=[
            "Difficult to detect - queries are normal",
            "May detect statistical analysis patterns"
        ],
        mitigations=[
            "Differential privacy during training",
            "Regularization to reduce overfitting",
            "Confidence score masking",
            "Output perturbation"
        ],
        detection_methods=[
            "Generally not detectable at query time",
            "Proactive privacy testing"
        ],
        papers=[
            "Membership Inference Attacks Against Machine Learning Models (Shokri et al., 2017)",
            "https://arxiv.org/abs/1610.05820"
        ],
        atlas_id="AML.T0025"
    ),
    
    AttackTechnique(
        id="AML-PRIV-002",
        name="Model Inversion Attack",
        category="Privacy",
        description="""
            Model inversion reconstructs training data features from model outputs.
            Given a target class label, the attack reconstructs what a typical
            input of that class looks like, potentially revealing sensitive attributes.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.DATA_EXTRACTION],
        access_level=AccessLevel.WHITE_BOX,
        attack_surface=[AttackSurface.OUTPUT_PREDICTIONS, AttackSurface.GRADIENTS],
        mechanism="""
            Optimize input to maximize target class probability:
                x* = argmax_x P(y_target | x) - lambda * regularizer(x)
            
            The reconstructed input reveals features of training data.
            For face recognition: reconstructs representative face for identity.
        """,
        effectiveness="Moderate - reconstructions are approximate but revealing",
        detection_difficulty="Hard - optimization queries may be distributed",
        prerequisites=[
            "Query access (black-box) or gradient access (white-box)",
            "Target class/label to reconstruct",
            "Regularization prior (e.g., image naturalness)"
        ],
        procedure=[
            "Select target class to invert",
            "Initialize random input",
            "Optimize input to maximize target class probability",
            "Apply regularization for realistic outputs",
            "Analyze reconstructed input for sensitive features"
        ],
        indicators=[
            "Optimization pattern in queries",
            "Queries converging toward specific input"
        ],
        mitigations=[
            "Differential privacy",
            "Output perturbation",
            "Limit prediction confidence",
            "Avoid encoding sensitive attributes"
        ],
        detection_methods=[
            "Query pattern analysis",
            "Gradient-based anomaly detection"
        ],
        papers=[
            "Model Inversion Attacks that Exploit Confidence Information (Fredrikson et al., 2015)",
            "https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf"
        ]
    ),
    
    AttackTechnique(
        id="AML-PRIV-003",
        name="Training Data Extraction",
        category="Privacy",
        description="""
            Large language models and other generative models can memorize and
            regurgitate training data verbatim. Attackers can extract sensitive
            training examples through careful prompting.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.DATA_EXTRACTION],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.OUTPUT_PREDICTIONS],
        mechanism="""
            1. Prompt model with prefixes that might appear in training data
            2. Generate completions at low temperature
            3. Analyze completions for memorized content
            4. Filter for high-likelihood completions (indicate memorization)
        """,
        effectiveness="High for LLMs - can extract PII, code, etc.",
        detection_difficulty="Hard - looks like normal generation",
        prerequisites=[
            "Query access to generative model",
            "Knowledge of potential training data format",
            "Filtering capability to identify memorized content"
        ],
        procedure=[
            "Generate candidate prompts (e.g., 'My email is')",
            "Sample many completions per prompt",
            "Identify low-perplexity completions (memorized)",
            "Verify extracted data is real (not hallucinated)",
            "Filter for sensitive content"
        ],
        indicators=[
            "Large volume of similar prompts",
            "Prompts designed to elicit specific formats"
        ],
        mitigations=[
            "Differential privacy training",
            "Training data deduplication",
            "Output filtering for PII",
            "Membership inference testing before deployment"
        ],
        detection_methods=[
            "Query pattern analysis",
            "Output monitoring for sensitive patterns"
        ],
        papers=[
            "Extracting Training Data from Large Language Models (Carlini et al., 2021)",
            "https://arxiv.org/abs/2012.07805"
        ]
    ),
]


# =============================================================================
# EVASION ATTACKS
# =============================================================================

EVASION_ATTACKS = [
    AttackTechnique(
        id="AML-EVA-001",
        name="Malware Classifier Evasion",
        category="Evasion",
        description="""
            Modify malware samples to evade ML-based malware detection while
            preserving malicious functionality. Adds benign features or modifies
            non-functional bytes to flip classification.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.EVASION],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.INPUT_DATA],
        mechanism="""
            Modify malware to add benign features:
            - Append benign byte sequences
            - Modify PE headers without breaking execution
            - Add benign API imports
            - Pad sections with benign code patterns
        """,
        effectiveness="High - PE format provides many modification opportunities",
        detection_difficulty="Medium - modifications may be detectable",
        prerequisites=[
            "Malware sample to modify",
            "Query access to target classifier",
            "Knowledge of file format constraints"
        ],
        procedure=[
            "Analyze malware structure for modifiable regions",
            "Identify features that drive benign classification",
            "Modify malware to include benign features",
            "Verify malware still executes correctly",
            "Test against target classifier"
        ],
        indicators=[
            "Unusual file structure",
            "Large appended data sections",
            "Mismatched feature distributions"
        ],
        mitigations=[
            "Dynamic analysis (execution-based detection)",
            "Ensemble models with diverse features",
            "Adversarial training",
            "Feature sanitization"
        ],
        detection_methods=[
            "Static analysis for unusual modifications",
            "Behavioral analysis",
            "Multi-view learning"
        ],
        papers=[
            "Evading Machine Learning Malware Classifiers via Stealthy Modifications",
            "https://arxiv.org/abs/1708.08327"
        ]
    ),
    
    AttackTechnique(
        id="AML-EVA-002",
        name="Spam Filter Evasion",
        category="Evasion",
        description="""
            Modify spam emails to bypass ML-based spam filters while maintaining
            human readability. Uses text transformation, obfuscation, and
            adversarial feature injection.
        """,
        phases=[AttackPhase.INFERENCE],
        goals=[AttackGoal.EVASION],
        access_level=AccessLevel.BLACK_BOX,
        attack_surface=[AttackSurface.INPUT_DATA],
        mechanism="""
            Text transformations to evade detection:
            - Character substitution (a -> @, o -> 0)
            - Word splitting/insertion
            - Adding benign text (good word injection)
            - HTML obfuscation
            - Image-based text
        """,
        effectiveness="Moderate - filters have evolved countermeasures",
        detection_difficulty="Low - obfuscation often detectable",
        prerequisites=[
            "Spam content to deliver",
            "Email delivery infrastructure",
            "Optional: feedback on filter decisions"
        ],
        procedure=[
            "Identify keywords triggering filter",
            "Apply transformations to evade detection",
            "Test against target filter",
            "Iterate until evasion succeeds"
        ],
        indicators=[
            "Character obfuscation patterns",
            "Unusual formatting",
            "Mismatch between text and images"
        ],
        mitigations=[
            "Character normalization",
            "OCR on embedded images",
            "Behavioral signals (sender reputation)",
            "Ensemble methods"
        ],
        detection_methods=[
            "Obfuscation detection",
            "Semantic analysis"
        ],
        papers=[
            "Good Word Attacks on Statistical Spam Filters (Lowd & Meek, 2005)"
        ]
    ),
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_techniques() -> List[AttackTechnique]:
    """Return all attack techniques in the taxonomy."""
    return (
        ADVERSARIAL_EXAMPLE_ATTACKS +
        BACKDOOR_ATTACKS +
        MODEL_EXTRACTION_ATTACKS +
        PRIVACY_ATTACKS +
        EVASION_ATTACKS
    )


def get_techniques_by_category(category: str) -> List[AttackTechnique]:
    """Return all techniques in a given category."""
    return [t for t in get_all_techniques() if t.category == category]


def get_techniques_by_access_level(level: AccessLevel) -> List[AttackTechnique]:
    """Return all techniques requiring a specific access level."""
    return [t for t in get_all_techniques() if t.access_level == level]


def get_techniques_by_goal(goal: AttackGoal) -> List[AttackTechnique]:
    """Return all techniques targeting a specific goal."""
    return [t for t in get_all_techniques() if goal in t.goals]


def get_technique_by_id(technique_id: str) -> Optional[AttackTechnique]:
    """Return a specific technique by ID."""
    for t in get_all_techniques():
        if t.id == technique_id:
            return t
    return None


def print_taxonomy_summary():
    """Print RE-oriented catalog index: gadgets, chains, literature techniques."""
    from graph_surgeon.taxonomy.gadget_registry import CHAIN_REGISTRY, GADGET_REGISTRY

    width = 72
    id_col = max(
        (
            max((len(gid) for gid in GADGET_REGISTRY), default=0),
            max((len(cid) for cid in CHAIN_REGISTRY), default=0),
        ),
        default=32,
    )

    print("=" * width)
    print("GRAPHSURGEON CATALOG")
    print("ONNX reverse engineering: structural motifs, chains, literature index")
    print("=" * width)
    print(
        "\nStructural motifs define attack landscape in the computation graph. "
        "Detection means an attack class is architecturally enabled, not that "
        "the model is exploitable."
    )

    print("\nStructural motifs (gadgets)")
    print("  Use: graph-surgeon catalog --gadget <ID>\n")
    for gid in sorted(GADGET_REGISTRY):
        gadget = GADGET_REGISTRY[gid]
        print(f"  {gid:<{id_col}}  {gadget.name}")

    print("\nCompound chains")
    print("  Use: graph-surgeon catalog --chain <ID>\n")
    for cid in sorted(CHAIN_REGISTRY):
        meta = CHAIN_REGISTRY[cid]
        print(f"  {cid:<{id_col}}  {meta.get('name', '')}")

    all_techniques = get_all_techniques()
    categories = sorted({t.category for t in all_techniques})

    print("\nLiterature technique index")
    print(
        f"  {len(all_techniques)} reference classes from adversarial ML literature. "
        "Not a threat taxonomy. Use: graph-surgeon catalog --technique <ID>\n"
    )
    for cat in categories:
        techs = sorted(get_techniques_by_category(cat), key=lambda t: t.id)
        print(f"  [{cat}]")
        for technique in techs:
            print(f"    {technique.id:<14} {technique.name}")
        print()

    print("-" * width)
    print("Commands")
    print("  graph-surgeon catalog --gadget GAP_FC_HEAD")
    print("  graph-surgeon catalog --chain CHAIN-PATCH-ATTACK-SURFACE")
    print("  graph-surgeon catalog --technique AML-ADV-001")
    print("  graph-surgeon catalog --coverage")


if __name__ == "__main__":
    print_taxonomy_summary()
    
    print("\n" + "=" * 60)
    print("SAMPLE TECHNIQUE DETAIL")
    print("=" * 60)
    
    technique = get_technique_by_id("AML-BACK-003")
    if technique:
        print(f"\nID: {technique.id}")
        print(f"Name: {technique.name}")
        print(f"Category: {technique.category}")
        print(f"Access Level: {technique.access_level.value}")
        print(f"\nDescription: {technique.description.strip()}")
        print(f"\nMechanism: {technique.mechanism.strip()}")
        print(f"\nMitigations:")
        for m in technique.mitigations:
            print(f"  - {m}")

