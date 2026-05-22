"""
Gadget Registry - Research Provenance Tracking for NN Security Analyzer

This registry tracks the research basis, confidence levels, and versioning for all
structural motifs detected by the tool.

Purpose:
- Ensure detection logic is evidence-based
- Track which papers validate each gadget
- Enable efficient updates when new research emerges
- Maintain historical record of changes

Usage:
    from research.gadget_registry import GADGET_REGISTRY, get_gadget_info
    
    info = get_gadget_info("GAP_FC_HEAD")
    print(info["research_basis"])  # Papers that validate this gadget

Last Updated: 2026-01-19
Registry Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import date


class GadgetStatus(Enum):
    """Status of a gadget in the registry."""
    ACTIVE = "active"                    # Currently used for detection
    DEPRECATED = "deprecated"            # Superseded by newer research
    EXPERIMENTAL = "experimental"        # Limited research basis
    NEEDS_VALIDATION = "needs_validation"  # Research basis outdated


class GadgetCategory(Enum):
    """Category of structural motif pattern."""
    INPUT_PREPROCESSING = "input_preprocessing"
    SPATIAL_AGGREGATION = "spatial_aggregation"
    FEATURE_FUSION = "feature_fusion"
    AMPLIFICATION = "amplification"
    DOWNSAMPLING = "downsampling"
    NORMALIZATION = "normalization"
    ATTENTION = "attention"
    CLASSIFIER_HEAD = "classifier_head"
    OBJECT_DETECTION = "object_detection"
    VIT_SPECIFIC = "vit_specific"
    BACKDOOR = "backdoor"
    SUPPLY_CHAIN = "supply_chain"
    INFORMATION_LEAKAGE = "information_leakage"
    DEPLOYMENT_CONTEXT = "deployment_context"


@dataclass
class GadgetDefinition:
    """Complete definition of a gadget with research provenance."""
    
    # Identity
    id: str
    name: str
    category: GadgetCategory
    
    # Description
    description: str
    detection_logic: str  # Human-readable description of how it's detected
    
    # Research Provenance
    research_basis: List[str]  # Paper IDs that validate this pattern
    first_documented: str      # Earliest paper (year or paper ID)
    last_validated: str        # Most recent confirming paper
    
    # Confidence & Status
    confidence: str            # HIGH, MEDIUM, LOW
    status: GadgetStatus
    
    # Attack Coverage
    attacks_enabled: List[str]  # Attack types this gadget enables
    structural_significance: str  # PRIMARY, SECONDARY, TERTIARY, EXCEPTIONAL, MITIGATING
    
    # Versioning
    version: str
    
    # Optional fields with defaults must come last
    superseded_by: Optional[str] = None  # If deprecated
    changelog: List[str] = field(default_factory=list)
    chainable_with: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def severity_base(self) -> str:
        """Deprecated alias for structural_significance (legacy tier labels)."""
        _legacy = {
            "PRIMARY": "HIGH",
            "SECONDARY": "MEDIUM",
            "TERTIARY": "LOW",
            "EXCEPTIONAL": "CRITICAL",
            "MITIGATING": "DEFENSIVE",
        }
        return _legacy.get(self.structural_significance, self.structural_significance)


# =============================================================================
# GADGET REGISTRY
# =============================================================================

GADGET_REGISTRY: Dict[str, GadgetDefinition] = {
    
    # =========================================================================
    # CNN classifier motifs (foundational)
    # =========================================================================
    
    "GAP_FC_HEAD": GadgetDefinition(
        id="GAP_FC_HEAD",
        name="Global Average Pool → FC Head",
        category=GadgetCategory.CLASSIFIER_HEAD,
        description="GlobalAveragePool followed by fully-connected classifier. "
                   "Aggregates spatial features without filtering, allowing localized "
                   "adversarial patches to dominate final representation.",
        detection_logic="Detect GlobalAveragePool or GlobalMaxPool followed within 2 hops "
                       "by Gemm/MatMul layer. Check for Flatten in between.",
        research_basis=[
            "36-GoogleAp-2017",      # Original adversarial patch
            "39-LaVAN-2018",         # Localized adversarial perturbations
            "71-DPATCH-2019",        # Detection patches
            "114-FakeIt-2024",       # Confirmed for AI detectors
            # Phase 4 additions (pattern-confirming):
            "42-ACS-2019",           # Adversarial camouflage stickers
            "44-AdvACO-2020",        # Ant colony optimization patches
            "45-AdvWatermark-2020",  # Watermark-based patches
            "54-TnT-2022",           # Trojan-in-Texture
        ],
        first_documented="2017",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["adversarial_patch", "lavan", "universal_perturbation", 
                        "feature_space_attacks", "dpatch", "aco_patch", "watermark_patch",
                        "trojan_texture", "copy_paste"],
        structural_significance="PRIMARY",
        version="1.2.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added composition modifier based on attention presence",
            "1.2.0 (2026-01-19): Added 6 pattern-confirming papers from Phase 4 categorization",
        ],
        chainable_with=["ALIASING_DOWNSAMPLE", "NO_SPATIAL_ATTENTION"],
        notes="Canonical patch attack landscape motif. Nearly universal in CNN classifiers."
    ),
    
    "ALIASING_DOWNSAMPLE": GadgetDefinition(
        id="ALIASING_DOWNSAMPLE",
        name="Aliasing Downsampling",
        category=GadgetCategory.DOWNSAMPLING,
        description="Stride-2 convolution or pooling in early layers without anti-aliasing "
                   "(blur) filter. Causes high-frequency perturbations to fold into lower "
                   "frequencies, persisting through the network.",
        detection_logic="Detect Conv with strides >= 2 in first 15% of network. Check for "
                       "absence of blur/avgpool operations in preceding 2 nodes.",
        research_basis=[
            "38-EOT-2018",           # Expectation over transformation
            "60-RP2-2018",           # Robust physical perturbations
            "108-AntiAlias-2021",    # Depth-adaptive anti-aliasing defense
            "109-DiffPurify-2025",   # Anti-aliasing as defense
            "110-MedOOD-2025",       # Aliasing metrics for OOD
            # Phase 4 additions (pattern-confirming):
            "37-PAE-2018",           # Physical adversarial examples
            "41-D2P-2019",           # Digital-to-physical transfer
            "46-ABBA-2020",          # Printed backdoors
            "47-ViewFool-2020",      # Viewpoint attacks (frequency via perspective)
            "61-DARTS-2018",         # Digital attack in frequency domain
            "62-RogueSigns-2018",    # Printed sign attacks
            "63-PSGAN-2019",         # GAN-generated physical patches
            "64-AdvCam-2020",        # Camera-based viewpoint attacks
            "76-ExtRP2-2018",        # Extended RP2
        ],
        first_documented="2018",
        last_validated="2025",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["eot_attacks", "rp2_attacks", "frequency_attacks", 
                        "physical_world_attacks", "fourier_attacks", "printed_attacks",
                        "digital_to_physical", "viewpoint_attacks", "gan_patches",
                        # Light-based attacks (covered by aliasing mechanism):
                        "light_attack", "projector_attack", "slm_attack", "advlb",
                        "spaa", "opad", "adversarial_shadow", "slap", "adversarial_rain"],
        structural_significance="PRIMARY",
        version="1.3.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Raised structural significance for object detectors per paper 107",
            "1.2.0 (2026-01-19): Added 9 pattern-confirming papers from Phase 4 categorization",
            "1.3.0 (2026-01-19): Added light-based attacks coverage (10 papers analyzed)",
        ],
        chainable_with=["GAP_FC_HEAD", "OBJECTNESS_HEAD", "AGGRESSIVE_EARLY_DOWNSAMPLING"],
        notes="Important for physical-world attack viability. Anti-aliasing is necessary "
              "but not sufficient defense. Also covers light-based attacks (shadows, "
              "projections, illumination) as these create frequency artifacts that alias."
    ),
    
    "MAXPOOL_AFTER_FUSION": GadgetDefinition(
        id="MAXPOOL_AFTER_FUSION",
        name="MaxPool After Feature Fusion",
        category=GadgetCategory.AMPLIFICATION,
        description="MaxPool operation within 3 hops after Concat/Add fusion. Selects "
                   "extreme activations, amplifying fused adversarial signals.",
        detection_logic="Detect MaxPool nodes. Check if any Concat/Add node exists within "
                       "3 hops upstream.",
        research_basis=[
            "36-GoogleAp-2017",
            "39-LaVAN-2018",
            "71-DPATCH-2019",
            "112-StochasticRes-2025",  # Confirms MaxPool amplification
            # Phase 4 additions (pattern-confirming):
            "67-PhysGAN-2020",        # Physical GAN - MaxPool amplifies patterns
        ],
        first_documented="2017",
        last_validated="2025",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["amplified_patch", "multi_scale_patch", "sparse_attacks", "gan_attacks"],
        structural_significance="PRIMARY",
        version="1.1.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added PhysGAN as pattern-confirming paper",
        ],
        chainable_with=["HIGH_FANIN_FUSION", "FUSION_POINT"],
        notes="Hardening: Replace with AvgPool or BlurPool after fusion."
    ),
    
    "HIGH_FANIN_FUSION": GadgetDefinition(
        id="HIGH_FANIN_FUSION",
        name="High Fan-in Feature Fusion",
        category=GadgetCategory.FEATURE_FUSION,
        description="Concat(axis=1) with 4+ input branches. Each branch is an independent "
                   "attack entry point, enabling multi-scale coordinated attacks.",
        detection_logic="Detect Concat nodes with axis=1. Count input edges. Flag if >= 4.",
        research_basis=[
            "71-DPATCH-2019",
            "111-DUMBer-2025",  # Shows AT ineffective for high fan-in
            # Phase 4 additions (pattern-confirming):
            "83-InvisCloak-2018",  # Multi-scale attack via high fan-in
        ],
        first_documented="2018",
        last_validated="2025",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["multi_scale_pgd", "universal_perturbation", 
                        "transfer_attacks", "dpatch", "invisible_cloak"],
        structural_significance="SECONDARY",
        version="1.2.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added AT resistance warning per paper 111",
            "1.2.0 (2026-01-19): Added Invisible Cloak as pattern-confirming paper",
        ],
        chainable_with=["MAXPOOL_AFTER_FUSION", "SKIP_CONNECTION"],
        notes="WARNING: Research [111] shows adversarial training is less effective "
              "for high fan-in architectures."
    ),
    
    "SKIP_CONNECTION": GadgetDefinition(
        id="SKIP_CONNECTION",
        name="Long Skip Connection",
        category=GadgetCategory.FEATURE_FUSION,
        description="Add operation implementing skip/residual connection spanning > 5 layers. "
                   "Creates gradient highways for stable attack optimization.",
        detection_logic="Detect Add nodes. Calculate skip distance between inputs. "
                       "Flag if distance > 5 layers.",
        research_basis=[
            "ResNet-2015",  # Original residual networks
            "111-DUMBer-2025",  # Gradient highways bypass AT
        ],
        first_documented="2015",
        last_validated="2025",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["pgd", "cw", "momentum_attacks", "transfer_attacks"],
        structural_significance="SECONDARY",
        version="1.1.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added AT bypass warning per paper 111",
        ],
        chainable_with=["HIGH_FANIN_FUSION"],
        notes="WARNING: Skip connections create gradient highways that may bypass "
              "adversarial training defenses."
    ),
    
    "FUSION_POINT": GadgetDefinition(
        id="FUSION_POINT",
        name="Multi-branch Feature Fusion",
        category=GadgetCategory.FEATURE_FUSION,
        description="Concat(axis=1) with 2-3 input branches. Standard multi-branch fusion "
                   "point enabling perturbation superposition.",
        detection_logic="Detect Concat nodes with axis=1 and 2-3 inputs.",
        research_basis=[
            "36-GoogleAp-2017",
            "InceptionV3-2015",
        ],
        first_documented="2015",
        last_validated="2017",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["multi_scale_pgd", "universal_perturbation"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["AMPLIFIER", "LARGE_KERNEL"],
        notes="Lower structural significance than HIGH_FANIN_FUSION but still enables multi-scale attacks."
    ),
    
    "AMPLIFIER": GadgetDefinition(
        id="AMPLIFIER",
        name="MaxPool Amplifier",
        category=GadgetCategory.AMPLIFICATION,
        description="MaxPool operation (not after fusion). Selects extreme activations, "
                   "amplifying adversarial spikes.",
        detection_logic="Detect MaxPool nodes not within 3 hops of Concat/Add.",
        research_basis=[
            "OnePixel-2019",  # One-pixel attack
            "SparsePert-2018",  # Sparse perturbations
        ],
        first_documented="2018",
        last_validated="2019",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["sparse_attacks", "one_pixel", "patch_attacks", "pgd"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["FUSION_POINT", "DOWNSAMPLER"],
        notes="Hardening: Replace with AvgPool."
    ),
    
    "DOWNSAMPLER": GadgetDefinition(
        id="DOWNSAMPLER",
        name="Stride-2 Downsampler",
        category=GadgetCategory.DOWNSAMPLING,
        description="Stride-2 Conv not in early layers, without anti-aliasing. "
                   "Contributes to frequency attack landscape.",
        detection_logic="Detect Conv with strides >= 2 in middle/late layers without "
                       "preceding blur operation.",
        research_basis=[
            "108-AntiAlias-2021",
        ],
        first_documented="2021",
        last_validated="2021",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["frequency_attacks", "fourier_attacks", "patch_survivability"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["AMPLIFIER", "FUSION_POINT"],
        notes="Lower structural significance than early aliasing but still contributes to frequency attacks."
    ),
    
    "NORMALIZER": GadgetDefinition(
        id="NORMALIZER",
        name="BatchNorm Layer",
        category=GadgetCategory.NORMALIZATION,
        description="BatchNormalization layer. Assumes stable activation distributions; "
                   "adversarial inputs cause distribution shift that BN can amplify.",
        detection_logic="Detect BatchNormalization nodes.",
        research_basis=[
            "BNAttack-2020",  # BN distribution shift attacks
            # Light-based attacks cause distribution shift:
            "66-AdvShadow-2022",  # Shadows shift activation distributions
        ],
        first_documented="2020",
        last_validated="2022",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["distribution_shift_attacks", "domain_attacks",
                        "light_attacks", "shadow_attacks", "illumination_attacks"],
        structural_significance="TERTIARY",
        version="1.1.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added light-based attack coverage (illumination causes distribution shift)",
        ],
        chainable_with=["FUSION_POINT"],
        notes="Hardening: GroupNorm/RMSNorm, freeze BN stats at inference. "
              "Light-based attacks (shadows, projections) cause activation distribution shift "
              "that BN amplifies. Models with many BN layers are sensitive to illumination changes."
    ),
    
    "LARGE_KERNEL": GadgetDefinition(
        id="LARGE_KERNEL",
        name="Large Kernel After Fusion",
        category=GadgetCategory.FEATURE_FUSION,
        description="Conv with kernel >= 5x5 within 3 hops after feature fusion. "
                   "Fused perturbations spread across wide receptive field.",
        detection_logic="Detect Conv with kernel_shape >= (5,5). Check if Concat/Add "
                       "exists within 3 hops upstream.",
        research_basis=[
            "Receptive-2019",  # Receptive field analysis
        ],
        first_documented="2019",
        last_validated="2019",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["patch_attacks", "gradient_steering"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["FUSION_POINT"],
        notes="Only flagged after fusion to reduce false positives."
    ),
    
    "LINEAR_HEAD": GadgetDefinition(
        id="LINEAR_HEAD",
        name="Linear Classifier Head",
        category=GadgetCategory.CLASSIFIER_HEAD,
        description="Final FC layer (Gemm/MatMul) as classifier. Direct logit manipulation "
                   "target for C&W and margin attacks.",
        detection_logic="Detect Gemm/MatMul in late layers with Softmax/Sigmoid downstream.",
        research_basis=[
            "CW-2017",  # C&W attack
        ],
        first_documented="2017",
        last_validated="2017",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["feature_space_attacks", "universal_perturbation", "logit_attacks"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["EXTRACTION_SURFACE"],
        notes="Nearly universal in classifiers. Structural significance depends on preceding architecture."
    ),
    
    "EXTRACTION_SURFACE": GadgetDefinition(
        id="EXTRACTION_SURFACE",
        name="Model Extraction Surface",
        category=GadgetCategory.INFORMATION_LEAKAGE,
        description="Softmax output layer. Reveals decision boundaries enabling model "
                   "extraction and membership inference attacks.",
        detection_logic="Detect Softmax nodes.",
        research_basis=[
            "ModelExtract-2016",  # Model extraction attacks
            "MemberInfer-2017",   # Membership inference
        ],
        first_documented="2016",
        last_validated="2017",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["model_extraction", "membership_inference"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["LINEAR_HEAD"],
        notes="Different threat model from adversarial perturbation."
    ),
    
    "CONTROL_POINT": GadgetDefinition(
        id="CONTROL_POINT",
        name="Conditional Control Point (Existing Backdoor Indicator)",
        category=GadgetCategory.BACKDOOR,
        description="Conditional operations (Where, If, Equal) in the computational graph "
                   "that could implement trigger-based backdoors. These operations can "
                   "detect specific input patterns and conditionally route to malicious outputs. "
                   "Their presence is a strong indicator of potential ShadowLogic backdoors.",
        detection_logic="Detect Where, If, Equal, Less, Greater, And, Or, Not, Xor nodes. "
                       "These are extremely rare in standard neural networks.",
        research_basis=[
            "HiddenLayer-ShadowLogic-2024",  # Primary ShadowLogic research
            "HiddenLayer-PersistentBackdoors-2024",  # Persistence through conversions
            "arXiv-2511.00664",  # ShadowLogic in LLMs
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["shadowlogic", "backdoor_triggers", "output_override", 
                        "trigger_detection", "conditional_backdoor"],
        structural_significance="EXCEPTIONAL",
        version="2.0.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "2.0.0 (2026-01-19): Updated with HiddenLayer ShadowLogic research, raised to EXCEPTIONAL significance"
        ],
        chainable_with=["SHADOWLOGIC_FORMAT_SURFACE", "SHADOWLOGIC_INJECTION_POINT"],
        notes="Notable structural signal. Indicates potential existing backdoor. "
              "Standard neural networks do NOT use conditional operations. "
              "Any presence should trigger immediate investigation."
    ),
    
    # =========================================================================
    # SHADOWLOGIC SUPPLY CHAIN MOTIFS
    # =========================================================================
    
    "SHADOWLOGIC_FORMAT_SURFACE": GadgetDefinition(
        id="SHADOWLOGIC_FORMAT_SURFACE",
        name="ShadowLogic-Susceptible Format",
        category=GadgetCategory.SUPPLY_CHAIN,
        description="Model is stored in a graph-based format (ONNX, TensorFlow, PyTorch) "
                   "that allows direct manipulation of the computational graph. An attacker "
                   "with file access can inject ShadowLogic nodes without retraining.",
        detection_logic="Check model format. ONNX, TensorFlow SavedModel, and PyTorch "
                       "formats all allow graph editing. TFLite and CoreML have some "
                       "protection through compilation.",
        research_basis=[
            "HiddenLayer-ShadowLogic-2024",
            "HiddenLayer-PersistentBackdoors-2024",
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["shadowlogic_injection", "graph_manipulation", "supply_chain_attack"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation from HiddenLayer research"],
        chainable_with=["SHADOWLOGIC_INJECTION_POINT", "SHADOWLOGIC_CAMOUFLAGE"],
        notes="Applies to most common model formats. Risk is format-inherent. "
              "Mitigation requires external integrity verification."
    ),
    
    "SHADOWLOGIC_INJECTION_POINT": GadgetDefinition(
        id="SHADOWLOGIC_INJECTION_POINT",
        name="ShadowLogic Injection Point",
        category=GadgetCategory.SUPPLY_CHAIN,
        description="Location in the computational graph where ShadowLogic backdoor nodes "
                   "could be trivially injected. Input stem for trigger detection, "
                   "output layer for result override, or branch points for selective activation.",
        detection_logic="Map graph structure to identify: (1) input-adjacent nodes for "
                       "trigger injection, (2) output-adjacent nodes for override injection, "
                       "(3) branch/merge points for selective modification.",
        research_basis=[
            "HiddenLayer-ShadowLogic-2024",
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["shadowlogic_injection", "trigger_insertion", "output_override"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["SHADOWLOGIC_FORMAT_SURFACE", "SHADOWLOGIC_CAMOUFLAGE"],
        notes="All models have injection points - this is structural. Combined with "
              "FORMAT_SURFACE, enables full ShadowLogic attack."
    ),
    
    "SHADOWLOGIC_CAMOUFLAGE": GadgetDefinition(
        id="SHADOWLOGIC_CAMOUFLAGE",
        name="ShadowLogic Camouflage Potential",
        category=GadgetCategory.SUPPLY_CHAIN,
        description="Model structure has high repetition and complexity, making injected "
                   "ShadowLogic nodes harder to detect through manual inspection. Large "
                   "models with many similar operations provide better camouflage.",
        detection_logic="Analyze graph statistics: node count, operation repetition, "
                       "total parameters. High values indicate difficult audit.",
        research_basis=[
            "HiddenLayer-ShadowLogic-2024",
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["shadowlogic_concealment", "audit_evasion"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["SHADOWLOGIC_FORMAT_SURFACE", "SHADOWLOGIC_INJECTION_POINT"],
        notes="Increases difficulty of detecting injected backdoors. Large production "
              "models with 100+ layers are essentially impossible to manually audit."
    ),
    
    "SHADOWLOGIC_NO_INTEGRITY": GadgetDefinition(
        id="SHADOWLOGIC_NO_INTEGRITY",
        name="Missing Model Integrity Verification",
        category=GadgetCategory.SUPPLY_CHAIN,
        description="Model file lacks cryptographic integrity verification (signatures, "
                   "checksums). A modified model cannot be distinguished from the original "
                   "without external verification mechanisms.",
        detection_logic="Check for absence of embedded signatures. Note: Most model "
                       "formats do not support built-in integrity verification.",
        research_basis=[
            "HiddenLayer-ShadowLogic-2024",
            "HiddenLayer-PersistentBackdoors-2024",
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["model_tampering", "supply_chain_attack", "integrity_bypass"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["SHADOWLOGIC_FORMAT_SURFACE"],
        notes="Applies to all common formats. Mitigation requires external signature "
              "verification before deployment. Essential for supply chain security."
    ),
    
    "SHAPE_OP": GadgetDefinition(
        id="SHAPE_OP",
        name="Early Shape Operation",
        category=GadgetCategory.INPUT_PREPROCESSING,
        description="Resize, Pad, Slice, Crop in early layers. Enables EoT attacks "
                   "and adversarial resizing/patch placement manipulation.",
        detection_logic="Detect Resize, Pad, Slice, Crop in first 15% of network.",
        research_basis=[
            "38-EOT-2018",
        ],
        first_documented="2018",
        last_validated="2018",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["eot_attacks", "adversarial_resize", "patch_placement"],
        structural_significance="TERTIARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=[],
        notes="Common in models with dynamic input handling."
    ),
    
    # =========================================================================
    # ATTENTION GADGETS
    # =========================================================================
    
    "NO_SPATIAL_ATTENTION": GadgetDefinition(
        id="NO_SPATIAL_ATTENTION",
        name="Missing Spatial Attention",
        category=GadgetCategory.ATTENTION,
        description="Model has GAP→FC classifier but no spatial attention mechanisms "
                   "(SE, CBAM, self-attention) to filter anomalous patch regions.",
        detection_logic="Check for absence of SE blocks, CBAM, self-attention patterns "
                       "when GAP_FC_HEAD is present.",
        research_basis=[
            "36-GoogleAp-2017",  # Patches work because no filtering
            "39-LaVAN-2018",
            "SENet-2018",  # SE blocks as defense
        ],
        first_documented="2017",
        last_validated="2018",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["adversarial_patch", "lavan", "universal_perturbation", 
                        "localized_attacks"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["GAP_FC_HEAD", "ALIASING_DOWNSAMPLE"],
        notes="Raises GAP_FC_HEAD structural significance to EXCEPTIONAL when present."
    ),
    
    "HAS_SPATIAL_ATTENTION": GadgetDefinition(
        id="HAS_SPATIAL_ATTENTION",
        name="Spatial Attention Present (Defensive)",
        category=GadgetCategory.ATTENTION,
        description="Model has SE blocks, CBAM, or self-attention that can help filter "
                   "anomalous patch regions. DEFENSIVE indicator.",
        detection_logic="Detect SE block pattern (GAP→FC→ReLU→FC→Sigmoid→Mul) or "
                       "self-attention pattern (MatMul→Softmax→MatMul).",
        research_basis=[
            "SENet-2018",
            "CBAM-2018",
        ],
        first_documented="2018",
        last_validated="2018",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=[],  # Defensive - doesn't enable attacks
        structural_significance="MITIGATING",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=[],
        notes="Lowers GAP_FC_HEAD structural significance to SECONDARY when present."
    ),
    
    # =========================================================================
    # Object detector motifs
    # =========================================================================
    
    "OBJECTNESS_HEAD": GadgetDefinition(
        id="OBJECTNESS_HEAD",
        name="Objectness Scoring Head",
        category=GadgetCategory.OBJECT_DETECTION,
        description="Sigmoid layer for objectness confidence scoring in detectors. "
                   "Direct target for Adversarial YOLO attacks to suppress detections.",
        detection_logic="Detect Sigmoid in late layers after Conv, with spatial output shape.",
        research_basis=[
            "84-AdvYOLO-2019",    # Adversarial YOLO
            "73-ObjectHider-2020",  # Object hiding
            # Phase 4 additions (pattern-confirming):
            "72-DPatch2-2019",    # Improved DPATCH
            "74-LPAttack-2020",   # License plate attacks
            "75-SwitchPatch-2022",  # Class switching
            "86-AdvTshirt-2020",  # Adversarial T-shirt
            "87-InvisCloak2-2020",  # Improved cloak
            "88-NAP-2021",        # Naturalistic adversarial patches
            "89-LAP-2021",        # Legitimate adversarial patches
            "90-AdvTexture-2022",  # Adversarial textures
            "91-AdvART-2023",     # Adversarial art
            "92-PatchInvis-2023",  # Patch of invisibility
            "93-DAP-2023",        # Dynamic adversarial patches
        ],
        first_documented="2019",
        last_validated="2023",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["adversarial_yolo", "object_hider", "disappearance_attacks", 
                        "objectness_suppression", "adversarial_clothing", "adversarial_texture",
                        "license_plate_attack", "patch_switching"],
        structural_significance="PRIMARY",
        version="1.1.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added 11 pattern-confirming papers from Phase 4 categorization",
        ],
        chainable_with=["ANCHOR_BASED_DETECTION", "NMS_DEPENDENCY"],
        notes="Suppressing objectness = complete invisibility (worse than misclassification). "
              "Most well-validated gadget with 13 confirming papers."
    ),
    
    "ANCHOR_BASED_DETECTION": GadgetDefinition(
        id="ANCHOR_BASED_DETECTION",
        name="Anchor-Based Detection",
        category=GadgetCategory.OBJECT_DETECTION,
        description="Detection output with anchor-pattern channels (255, 507, etc.). "
                   "Fixed anchor grids provide predictable attack targets.",
        detection_logic="Detect Conv in late layers with output channels matching "
                       "anchor patterns (divisible by 3, common YOLO patterns).",
        research_basis=[
            "71-DPATCH-2019",
            "84-AdvYOLO-2019",
        ],
        first_documented="2019",
        last_validated="2019",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["dpatch", "adversarial_yolo", "anchor_manipulation", 
                        "targeted_disappearance"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["OBJECTNESS_HEAD", "FPN_STRUCTURE"],
        notes="Hardening: Anchor-free detection (FCOS, CenterNet)."
    ),
    
    "FPN_STRUCTURE": GadgetDefinition(
        id="FPN_STRUCTURE",
        name="Feature Pyramid Network",
        category=GadgetCategory.OBJECT_DETECTION,
        description="Upsample→Add/Concat lateral connections forming feature pyramid. "
                   "Each pyramid level is independent attack entry point.",
        detection_logic="Detect Upsample/Resize followed by Add/Concat in middle layers.",
        research_basis=[
            "77-ShapeShifter-2018",
            "71-DPATCH-2019",
        ],
        first_documented="2018",
        last_validated="2019",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["multi_scale_evasion", "dpatch", "scale_sensitive_attacks", 
                        "shapeshifter"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["HIGH_FANIN_FUSION", "DETECTION_HEAD_PATTERN"],
        notes="Multiple scales = multiple attack surfaces."
    ),
    
    "TWO_STAGE_RPN": GadgetDefinition(
        id="TWO_STAGE_RPN",
        name="Two-Stage RPN Detection",
        category=GadgetCategory.OBJECT_DETECTION,
        description="ROIAlign/ROIPool indicating two-stage detector with Region Proposal "
                   "Network. Attacking RPN causes complete object disappearance.",
        detection_logic="Detect RoiAlign, RoiPool, ROIAlign, ROIPool nodes.",
        research_basis=[
            "77-ShapeShifter-2018",
        ],
        first_documented="2018",
        last_validated="2018",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["shapeshifter", "rpn_attacks", "proposal_suppression", 
                        "two_stage_evasion"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["SHARED_BACKBONE", "DETECTION_HEAD_PATTERN"],
        notes="No proposals = no detections. Single point of failure."
    ),
    
    "NMS_DEPENDENCY": GadgetDefinition(
        id="NMS_DEPENDENCY",
        name="NMS Dependency",
        category=GadgetCategory.OBJECT_DETECTION,
        description="NonMaxSuppression post-processing. Can be exploited via confidence "
                   "manipulation and IoU attacks.",
        detection_logic="Detect NonMaxSuppression, NMS, BatchedNMS nodes.",
        research_basis=[
            "NMSAttack-2020",
            # Phase 4 additions (pattern-confirming):
            "78-NestedAE-2019",     # Nested adversarial examples (NMS threshold)
            "79-TransPatch-2021",   # Translucent patches (confidence manipulation)
        ],
        first_documented="2019",
        last_validated="2021",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["confidence_manipulation", "false_positive_injection", "nms_bypass",
                        "nested_ae", "translucent_patch"],
        structural_significance="SECONDARY",
        version="1.1.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added 2 pattern-confirming papers from Phase 4 categorization",
        ],
        chainable_with=["OBJECTNESS_HEAD", "DETECTION_HEAD_PATTERN"],
        notes="Hardening: Soft-NMS, confidence calibration."
    ),
    
    "DETECTION_HEAD_PATTERN": GadgetDefinition(
        id="DETECTION_HEAD_PATTERN",
        name="Detection Head Structure",
        category=GadgetCategory.OBJECT_DETECTION,
        description="Multi-output detection structure (Reshape→Transpose/Concat) "
                   "indicating object detector output formatting.",
        detection_logic="Detect Reshape in late layers followed by Transpose, Concat, or Split.",
        research_basis=[
            "84-AdvYOLO-2019",
            "77-ShapeShifter-2018",
        ],
        first_documented="2018",
        last_validated="2019",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["adversarial_yolo", "shapeshifter", "detector_evasion"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["OBJECTNESS_HEAD", "ANCHOR_BASED_DETECTION"],
        notes="Indicates model is object detector, not classifier."
    ),
    
    "SHARED_BACKBONE": GadgetDefinition(
        id="SHARED_BACKBONE",
        name="Shared Detection Backbone",
        category=GadgetCategory.OBJECT_DETECTION,
        description="Single feature extractor shared by multiple detection heads. "
                   "Single point of failure for detector attacks.",
        detection_logic="Identify when detection heads share common upstream feature layers.",
        research_basis=[
            "113-FAIRTAT-2025",
            # Phase 4 additions (pattern-confirming):
            "100-ERAttack-2020",    # Entity recognition attack on shared backbone
            "101-ScreenAttack-2020",  # Screen attack propagates through backbone
        ],
        first_documented="2020",
        last_validated="2025",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["single_point_failure", "backbone_attacks", 
                        "universal_detector_perturbation", "entity_recognition_attack",
                        "screen_attack"],
        structural_significance="SECONDARY",
        version="1.1.0",
        changelog=[
            "1.0.0 (2026-01-19): Initial implementation",
            "1.1.0 (2026-01-19): Added 2 pattern-confirming papers from Phase 4 categorization",
        ],
        chainable_with=["TWO_STAGE_RPN", "DETECTION_HEAD_PATTERN"],
        notes="Attack on backbone affects ALL downstream tasks."
    ),
    
    # =========================================================================
    # ViT-specific motifs
    # =========================================================================
    
    "VIT_PATCH_EMBEDDING": GadgetDefinition(
        id="VIT_PATCH_EMBEDDING",
        name="ViT Patch Embedding",
        category=GadgetCategory.VIT_SPECIFIC,
        description="Conv with kernel_shape == strides (typically 16x16 stride 16) "
                   "for non-overlapping patch tokenization. Single entry point with "
                   "no spatial redundancy.",
        detection_logic="Detect Conv in early layers where kernel_shape == strides and "
                       "kernel >= 14 (common ViT patch sizes: 14, 16, 32).",
        research_basis=[
            "103-ViTSSL-2024",  # ViT SSL attack landscape
            "114-FakeIt-2024",  # ViT detector attack landscape
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["vit_patch_attacks", "attention_hijacking", "universal_perturbation"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["UNREGULARIZED_ATTENTION", "CLS_TOKEN_AGGREGATION"],
        notes="ViT equivalent of CNN input attack landscape but concentrated in single projection."
    ),
    
    "UNREGULARIZED_ATTENTION": GadgetDefinition(
        id="UNREGULARIZED_ATTENTION",
        name="Unregularized Self-Attention",
        category=GadgetCategory.VIT_SPECIFIC,
        description="Self-attention blocks without dropout after attention computation. "
                   "Attention weights can be manipulated to focus on adversarial patches.",
        detection_logic="Detect MatMul→Softmax→MatMul pattern (self-attention). Check for "
                       "absence of Dropout within 2 hops after second MatMul.",
        research_basis=[
            "103-ViTSSL-2024",
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["attention_hijacking", "adversarial_patch", "backdoor_attention"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["VIT_PATCH_EMBEDDING", "CLS_TOKEN_AGGREGATION"],
        notes="Attention hijacking is highly effective attack against ViTs."
    ),
    
    "CLS_TOKEN_AGGREGATION": GadgetDefinition(
        id="CLS_TOKEN_AGGREGATION",
        name="CLS Token Aggregation",
        category=GadgetCategory.VIT_SPECIFIC,
        description="Classification using CLS token that aggregates all patch information "
                   "through attention. Analogous to GAP in CNNs - no spatial filtering.",
        detection_logic="Detect Slice/Gather extracting first token (CLS) in late layers, "
                       "followed by LayerNorm and/or FC layer.",
        research_basis=[
            "103-ViTSSL-2024",
        ],
        first_documented="2024",
        last_validated="2024",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["vit_patch_attacks", "cls_manipulation", "universal_perturbation"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["VIT_PATCH_EMBEDDING", "UNREGULARIZED_ATTENTION"],
        notes="Adversarial patches receive equal attention weight as benign content."
    ),
    
    "AGGRESSIVE_EARLY_DOWNSAMPLING": GadgetDefinition(
        id="AGGRESSIVE_EARLY_DOWNSAMPLING",
        name="Aggressive Early Downsampling",
        category=GadgetCategory.DOWNSAMPLING,
        description="3+ stride-2 operations in first 15% of network. Rapidly destroys "
                   "spatial information, making small adversarial patches more effective.",
        detection_logic="Count stride-2 Conv/Pool operations in first 15% of network. "
                       "Flag if >= 3.",
        research_basis=[
            "107-FDMYOLO-2025",  # Small object attack landscape
        ],
        first_documented="2025",
        last_validated="2025",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["small_object_evasion", "physical_patch", "aliasing_attacks"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-19): Initial implementation"],
        chainable_with=["ALIASING_DOWNSAMPLE", "OBJECTNESS_HEAD"],
        notes="Small objects especially sensitive to patch attacks. Reduces minimum effective patch size."
    ),
    
    # =========================================================================
    # Extended audio/ASR motifs
    # Research basis: Whisper analysis, CTC research, multilingual ASR
    # =========================================================================
    
    "CTC_DECODER_STRUCTURE": GadgetDefinition(
        id="CTC_DECODER_STRUCTURE",
        name="CTC Decoder Output Topology",
        category=GadgetCategory.CLASSIFIER_HEAD,
        description="CTC-based ASR output: Linear layer mapping to vocabulary size with "
                   "blank token at index 0. CTC prefix search and beam decoding are "
                   "exploitable via adversarial perturbations that manipulate blank/non-blank "
                   "token probabilities.",
        detection_logic="Identify final Linear/Gemm/MatMul with output dimension matching "
                       "typical vocabulary sizes (28-32000). Check for characteristic "
                       "CTC topology: encoder → projection → vocab-sized output.",
        research_basis=[
            "Carlini-Audio-2018",
            "BATCH_4_5_6_AUDIO_ANALYSIS",
        ],
        first_documented="2018",
        last_validated="2026",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["ctc_prefix_attack", "forced_transcription", "blank_token_exploit"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from audio batch analysis"],
        chainable_with=["AUDIO_MEL_INPUT", "AUDIO_STRIDE_DOWNSAMPLE", "AUDIO_TEMPORAL_ATTENTION"],
        notes="CTC loss structure indexes prefix-injection attack landscape. "
              "Blank token at index 0 is a known exploitation target."
    ),
    
    "SPECIAL_TOKEN_CONTROL_FLOW": GadgetDefinition(
        id="SPECIAL_TOKEN_CONTROL_FLOW",
        name="Special Token Control Flow",
        category=GadgetCategory.CLASSIFIER_HEAD,
        description="Autoregressive decoder with special control tokens (endoftext, "
                   "startoftranscript, translate, transcribe, notimestamps). Acoustic "
                   "adversarial inputs can force emission of control tokens, hijacking "
                   "generation flow.",
        detection_logic="Detect embedding layers with known special token vocabulary patterns. "
                       "Look for decoder architectures with multiple special token embeddings "
                       "or task-specific token slots in early decoder positions.",
        research_basis=[
            "Whisper-Architecture-Analysis",
            "BATCH_4_5_6_AUDIO_ANALYSIS",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["token_injection", "premature_eos", "control_token_hijack"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from audio batch analysis"],
        chainable_with=["CTC_DECODER_STRUCTURE", "CROSS_MODAL_ATTENTION", "ENCODER_DECODER_SEQ2SEQ"],
        notes="Control tokens create a covert command channel. Adversarial audio can "
              "force premature EOS or switch tasks without user awareness."
    ),
    
    "TASK_TOKEN_CONDITIONING": GadgetDefinition(
        id="TASK_TOKEN_CONDITIONING",
        name="Task Token Conditioning",
        category=GadgetCategory.CLASSIFIER_HEAD,
        description="Decoder conditioned on task tokens at input positions (e.g., Whisper's "
                   "language, task, timestamps tokens). Adversarial inputs can manipulate "
                   "which task the model performs.",
        detection_logic="Detect task-specific embedding patterns in decoder input. Look for "
                       "multiple embedding lookups feeding into decoder's first positions. "
                       "Task conditioning typically appears as concat/add of learned embeddings "
                       "with positional encodings.",
        research_basis=[
            "Whisper-Architecture-Analysis",
            "BATCH_4_5_6_AUDIO_ANALYSIS",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["task_confusion", "cross_task_attack", "conditioning_hijack"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from audio batch analysis"],
        chainable_with=["SPECIAL_TOKEN_CONTROL_FLOW", "CROSS_MODAL_ATTENTION"],
        notes="Task conditioning is an indirect attack surface. Adversarial audio can "
              "force translation instead of transcription, or wrong language output."
    ),
    
    "LANGUAGE_DETECTION_HEAD": GadgetDefinition(
        id="LANGUAGE_DETECTION_HEAD",
        name="Language Detection Head",
        category=GadgetCategory.CLASSIFIER_HEAD,
        description="Classification head on encoder output that determines input language. "
                   "Adversarial perturbations can confuse language detection, causing "
                   "wrong-language transcription or code-switching failures.",
        detection_logic="Detect classification head branching from encoder output with "
                       "output dimension matching common language count ranges (50-100+). "
                       "Look for parallel heads on encoder output (one for main task, "
                       "one for language).",
        research_basis=[
            "Whisper-Architecture-Analysis",
            "BATCH_4_5_6_AUDIO_ANALYSIS",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["language_confusion", "code_switch_attack", "lang_id_bypass"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from audio batch analysis"],
        chainable_with=["TASK_TOKEN_CONDITIONING", "AUDIO_MEL_INPUT"],
        notes="Language detection is typically a small classification head. "
              "Attacking it can cascade errors through the entire pipeline."
    ),
    
    # =========================================================================
    # Extended multimodal motifs
    # Research basis: CLIP, multimodal attacks, cross-modal jailbreaks
    # =========================================================================
    
    "MULTIMODAL_FUSION_POINT": GadgetDefinition(
        id="MULTIMODAL_FUSION_POINT",
        name="Multimodal Fusion Point",
        category=GadgetCategory.FEATURE_FUSION,
        description="Operation where two or more modality branches merge (Concat, Add, "
                   "cross-attention). The fusion boundary is a primary attack surface: "
                   "adversarial content in one modality can corrupt the fused representation.",
        detection_logic="Detect Concat/Add/MatMul nodes that receive inputs from multiple "
                       "distinct branches (determined by graph topology analysis). "
                       "Branches should have different input origins (separate model inputs "
                       "or distinct processing paths).",
        research_basis=[
            "BATCH_7_ANALYSIS",
            "CLIP-Attack-Research",
        ],
        first_documented="2021",
        last_validated="2026",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["cross_modal_injection", "modality_hijack", "fusion_point_attack", "multimodal_jailbreak"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from multimodal batch analysis"],
        chainable_with=["CROSS_MODAL_FUSION_LATE", "DUAL_ENCODER_ALIGNMENT", "SHARED_BACKBONE"],
        notes="Fusion points are where modality boundaries break down. The most effective "
              "multimodal attacks target these junctions."
    ),
    
    "CROSS_MODAL_FUSION_LATE": GadgetDefinition(
        id="CROSS_MODAL_FUSION_LATE",
        name="Late Cross-Modal Fusion",
        category=GadgetCategory.FEATURE_FUSION,
        description="Model architecture where modality branches process independently until "
                   "near the output layer. Late fusion means each modality's representation "
                   "is fully formed before merging, making single-modality attacks more potent.",
        detection_logic="Detect fusion points (Concat/Add) in the last 30% of the network "
                       "where inputs come from branches with independent processing chains. "
                       "Compare depth of first shared operation vs total depth.",
        research_basis=[
            "BATCH_7_ANALYSIS",
            "Multimodal-Fusion-Research",
        ],
        first_documented="2021",
        last_validated="2026",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["late_fusion_exploit", "modality_disconnect", "adversarial_modality_substitution"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from multimodal batch analysis"],
        chainable_with=["MULTIMODAL_FUSION_POINT", "DUAL_ENCODER_ALIGNMENT"],
        notes="Late fusion is architecturally weaker against single-modality attacks because "
              "the attacked modality's representation is not constrained by the other modality "
              "until very late in processing."
    ),
    
    "DUAL_ENCODER_ALIGNMENT": GadgetDefinition(
        id="DUAL_ENCODER_ALIGNMENT",
        name="Dual Encoder Alignment Space",
        category=GadgetCategory.FEATURE_FUSION,
        description="Two separate encoder branches (e.g., vision + text) projecting into a "
                   "shared embedding space. The alignment loss creates a contrastive attack "
                   "surface — adversarial inputs in one modality can be crafted to align with "
                   "arbitrary targets in the other.",
        detection_logic="Detect two encoder subgraphs with separate inputs whose outputs pass "
                       "through projection layers (Linear/MatMul) with matching output dimensions. "
                       "The matching output dims indicate shared embedding space.",
        research_basis=[
            "BATCH_7_ANALYSIS",
            "CLIP-Attack-2022",
            "CoA-Attack-2024",
        ],
        first_documented="2022",
        last_validated="2026",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["embedding_space_attack", "contrastive_adversarial", "clip_attack", "typographic_attack"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from multimodal batch analysis"],
        chainable_with=["MULTIMODAL_FUSION_POINT", "ENCODER_PROJECTION_BRIDGE"],
        notes="CLIP-style dual encoders index typographic attack and embedding space manipulation."
    ),
    
    "TEMPORAL_CROSS_MODAL_SYNC": GadgetDefinition(
        id="TEMPORAL_CROSS_MODAL_SYNC",
        name="Temporal Cross-Modal Synchronization",
        category=GadgetCategory.ATTENTION,
        description="Shared positional encoding or temporal attention connecting multiple "
                   "modalities. Temporal synchronization assumes aligned time series — "
                   "adversarial desynchronization can corrupt cross-modal associations.",
        detection_logic="Detect shared positional encoding layers feeding into multiple "
                       "branches, or cross-attention layers with temporal dimension matching. "
                       "Look for Add operations combining positional embeddings with features "
                       "from different modality branches.",
        research_basis=[
            "BATCH_7_ANALYSIS",
            "Audio-Visual-Attack-Research",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["temporal_desync_attack", "alignment_corruption", "sync_exploitation"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from multimodal batch analysis"],
        chainable_with=["MULTIMODAL_FUSION_POINT", "CROSS_MODAL_ATTENTION", "AUDIO_TEMPORAL_ATTENTION"],
        notes="Temporal sync attacks are subtle — desynchronizing audio and visual streams "
              "can cause misattribution without obvious artifacts."
    ),
    
    # =========================================================================
    # Structural / misc motifs
    # Research basis: Quantization attacks, LLaVA, 3D point cloud research
    # =========================================================================
    
    "ENCODER_PROJECTION_BRIDGE": GadgetDefinition(
        id="ENCODER_PROJECTION_BRIDGE",
        name="Encoder Projection Bridge",
        category=GadgetCategory.FEATURE_FUSION,
        description="Linear projection layer bridging encoder output to a downstream model "
                   "(e.g., vision encoder → projection → LLM in LLaVA). The projection "
                   "compresses or transforms between mismatched dimensions, creating a "
                   "bottleneck attack surface.",
        detection_logic="Detect Linear/MatMul operations where input and output dimensions "
                       "differ significantly (>2x ratio), positioned between two distinct "
                       "subgraphs. The bridge typically has no activation function.",
        research_basis=[
            "LLaVA-Architecture-2023",
            "BATCH_7_ANALYSIS",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["projection_manipulation", "bridge_attack", "dimension_mismatch_exploit"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation from multimodal batch analysis"],
        chainable_with=["DUAL_ENCODER_ALIGNMENT", "MULTIMODAL_FUSION_POINT"],
        notes="Projection bridges are high-value targets because they compress an entire "
              "modality's representation into the downstream model's format."
    ),
    
    "QUANTIZATION_NODES": GadgetDefinition(
        id="QUANTIZATION_NODES",
        name="Quantization Operator Nodes",
        category=GadgetCategory.SUPPLY_CHAIN,
        description="Presence of QuantizeLinear/DequantizeLinear operators indicating the "
                   "model uses quantized inference. Quantization introduces rounding errors "
                   "that can be amplified by adversarial perturbations.",
        detection_logic="Count QuantizeLinear and DequantizeLinear operations in the graph. "
                       "Detect quantization patterns (QDQ pairs surrounding compute ops).",
        research_basis=[
            "Quantization-Adversarial-2023",
            "BATCH_8_ANALYSIS",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["quantization_error_amplification", "bit_flip_attack", "precision_exploit"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation"],
        chainable_with=["AMPLIFIER", "NORMALIZER"],
        notes="Quantized models have fundamentally different error surfaces than FP32 models. "
              "Integer quantization creates step functions that adversarial perturbations can exploit."
    ),
    
    "VOXEL_ENCODING": GadgetDefinition(
        id="VOXEL_ENCODING",
        name="Voxel/Pillar Encoding Structure",
        category=GadgetCategory.INPUT_PREPROCESSING,
        description="Spatial binning operations for 3D point cloud processing (voxelization, "
                   "pillar encoding). Voxel boundaries create quantization artifacts that "
                   "adversarial point perturbations can exploit.",
        detection_logic="Detect operations characteristic of voxel encoding: ScatterND, "
                       "custom voxelization ops, pillar-style grouping patterns. "
                       "Look for 3D→2D projection patterns or spatial binning operations.",
        research_basis=[
            "PointCloud-Adversarial-2023",
            "BATCH_8_ANALYSIS",
        ],
        first_documented="2023",
        last_validated="2026",
        confidence="LOW",
        status=GadgetStatus.EXPERIMENTAL,
        attacks_enabled=["voxel_perturbation", "point_cloud_attack", "spatial_binning_exploit", "lidar_spoofing"],
        structural_significance="PRIMARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-01-20): Initial implementation — experimental"],
        chainable_with=["SHARED_BACKBONE"],
        notes="Voxel encoding is domain-specific (autonomous driving, robotics). "
              "Detection may require custom op identification."
    ),

    # =========================================================================
    # DEPLOYMENT CONTEXT (graph-visible deployment signals)
    # =========================================================================

    "SINGLE_MODALITY_INPUT": GadgetDefinition(
        id="SINGLE_MODALITY_INPUT",
        name="Single Modality Graph Input",
        category=GadgetCategory.DEPLOYMENT_CONTEXT,
        description="One primary input tensor in the exported ONNX graph with no early "
                   "multimodal fusion before the backbone. The DAG is a standard vision "
                   "(or single-stream) classifier even when deployment may use thermal, IR, "
                   "or another sensor not represented in the file.",
        detection_logic="graph.input count == 1; no MULTIMODAL_FUSION_POINT or "
                       "DUAL_ENCODER_ALIGNMENT from separate inputs in early network.",
        research_basis=[
            "94-AdversarialBulbs-2021",
            "95-QRAttack-2022",
            "96-HOTCOLD-2022",
            "97-AIP-2023",
            "98-AdvIB-2023",
        ],
        first_documented="2021",
        last_validated="2023",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=[
            "thermal_domain_mismatch",
            "cross_sensor_transfer",
            "visible_trained_nonvisible_deploy",
        ],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Deployment-context motif for thermal papers"],
        chainable_with=["GAP_FC_HEAD", "ALIASING_DOWNSAMPLE", "IN_GRAPH_PREPROCESSING"],
        notes="Attack channel may be hardware (thermal LED, clothing); graph still indexes "
              "standard CNN/ViT attack landscape. Record deployment sensors separately.",
    ),

    "IN_GRAPH_PREPROCESSING": GadgetDefinition(
        id="IN_GRAPH_PREPROCESSING",
        name="In-Graph Input Preprocessing",
        category=GadgetCategory.DEPLOYMENT_CONTEXT,
        description="Normalize, scale, or color transforms in the first ~15% of hops from "
                   "the graph input. Trust boundary for pixel statistics sits inside ONNX "
                   "(distinct from off-graph ISP or camera pipeline).",
        detection_logic="Early Sub/Div/Mul with constants, InstanceNormalization or "
                       "BatchNormalization at stem, or Cast plus scale before first Conv.",
        research_basis=[
            "50-Invisibleperturbations-2021",
            "51-AdversarialISP-2021",
        ],
        first_documented="2021",
        last_validated="2021",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=[
            "preprocessing_trust_boundary",
            "distribution_shift",
            "isp_pipeline_mismatch",
        ],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): In-graph vs off-graph preprocessing signal"],
        chainable_with=["NORMALIZER", "SHAPE_OP", "SINGLE_MODALITY_INPUT"],
        notes="Paper 51 applies when preprocessing is exported; off-graph ISP is a static limit.",
    ),

    "HAS_MULTIMODAL_FUSION": GadgetDefinition(
        id="HAS_MULTIMODAL_FUSION",
        name="Multimodal Fusion Present",
        category=GadgetCategory.DEPLOYMENT_CONTEXT,
        description="Multimodal fusion or dual-encoder alignment detected in the DAG. "
                   "Indicates multiple modality streams merge in-graph (thermal/single-modality "
                   "mismatch from graph alone is less likely).",
        detection_logic="MULTIMODAL_FUSION_POINT or DUAL_ENCODER_ALIGNMENT from motifs scan.",
        research_basis=[
            "CLIP-Architecture-2021",
            "BATCH_7_ANALYSIS",
        ],
        first_documented="2021",
        last_validated="2026",
        confidence="HIGH",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["cross_modal_injection", "modality_hijack", "late_fusion_exploit"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Companion to SINGLE_MODALITY_INPUT"],
        chainable_with=["MULTIMODAL_FUSION_POINT", "DUAL_ENCODER_ALIGNMENT"],
    ),

    "AUDIO_MEL_INPUT": GadgetDefinition(
        id="AUDIO_MEL_INPUT",
        name="Mel-Spectrogram Input Surface",
        category=GadgetCategory.INPUT_PREPROCESSING,
        description="Audio frontend expecting mel-spectrogram or log-mel bins (typical 80-128 "
                   "frequency bins). Indexes psychoacoustic and frequency-domain audio attacks.",
        detection_logic="Early Conv/MatMul on 1D or 2D tensors with mel-like channel counts; "
                       "Whisper-style log-mel stem patterns.",
        research_basis=[
            "Carlini-Audio-2018",
            "70-TPatch-2023",
            "102-PG(Poltergeist)-2021",
        ],
        first_documented="2018",
        last_validated="2023",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["audio_adversarial", "psychoacoustic_masking", "frequency_attacks"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Registered audio motif from motifs detector"],
        chainable_with=["AUDIO_STRIDE_DOWNSAMPLE", "AUDIO_1D_CONV", "CTC_DECODER_STRUCTURE"],
    ),

    "AUDIO_1D_CONV": GadgetDefinition(
        id="AUDIO_1D_CONV",
        name="Audio 1D Convolution Frontend",
        category=GadgetCategory.INPUT_PREPROCESSING,
        description="1D convolution stack for temporal audio feature extraction.",
        detection_logic="Conv1d or Conv with 1D-like kernel on temporal audio tensors.",
        research_basis=["Carlini-Audio-2018", "70-TPatch-2023"],
        first_documented="2018",
        last_validated="2023",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["audio_adversarial", "temporal_perturbation"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Registered from motifs audio scan"],
        chainable_with=["AUDIO_MEL_INPUT", "AUDIO_STRIDE_DOWNSAMPLE"],
    ),

    "AUDIO_STRIDE_DOWNSAMPLE": GadgetDefinition(
        id="AUDIO_STRIDE_DOWNSAMPLE",
        name="Audio Stride Downsampling",
        category=GadgetCategory.DOWNSAMPLING,
        description="Strided 1D conv or pool in audio frontend; aliasing-like folding on "
                   "spectral/temporal axes.",
        detection_logic="Stride > 1 in early audio Conv or pooling along time/frequency axis.",
        research_basis=["Carlini-Audio-2018", "70-TPatch-2023"],
        first_documented="2018",
        last_validated="2023",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["audio_aliasing", "robust_audio_attacks", "audio_adversarial"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Registered from motifs audio scan"],
        chainable_with=["AUDIO_MEL_INPUT", "AUDIO_1D_CONV"],
    ),

    "AUDIO_TEMPORAL_ATTENTION": GadgetDefinition(
        id="AUDIO_TEMPORAL_ATTENTION",
        name="Audio Temporal Attention",
        category=GadgetCategory.ATTENTION,
        description="Self-attention over audio time steps; temporal hijacking surface.",
        detection_logic="MatMul/Softmax attention patterns on audio sequence length.",
        research_basis=["Carlini-Audio-2018", "Whisper-Architecture-Analysis"],
        first_documented="2018",
        last_validated="2023",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["audio_adversarial", "attention_hijacking", "temporal_perturbation"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Registered from motifs audio scan"],
        chainable_with=["AUDIO_MEL_INPUT", "CROSS_MODAL_ATTENTION"],
    ),

    "CROSS_MODAL_ATTENTION": GadgetDefinition(
        id="CROSS_MODAL_ATTENTION",
        name="Cross-Modal Attention",
        category=GadgetCategory.ATTENTION,
        description="Encoder-decoder or cross-attention between modalities (audio-text, image-text).",
        detection_logic="Cross-attention MatMul patterns between encoder and decoder branches.",
        research_basis=[
            "Whisper-Architecture-Analysis",
            "102-PG(Poltergeist)-2021",
        ],
        first_documented="2023",
        last_validated="2023",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["cross_modal_injection", "audio_text_hijacking", "hidden_command_injection"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Registered from motifs audio/ASR scan"],
        chainable_with=["ENCODER_DECODER_SEQ2SEQ", "SPECIAL_TOKEN_CONTROL_FLOW"],
    ),

    "ENCODER_DECODER_SEQ2SEQ": GadgetDefinition(
        id="ENCODER_DECODER_SEQ2SEQ",
        name="Encoder-Decoder Seq2Seq",
        category=GadgetCategory.FEATURE_FUSION,
        description="Separate encoder and autoregressive decoder subgraphs (ASR, speech translation).",
        detection_logic="Distinct encoder stack feeding decoder via cross-attention or state.",
        research_basis=[
            "Whisper-Architecture-Analysis",
            "102-PG(Poltergeist)-2021",
        ],
        first_documented="2023",
        last_validated="2023",
        confidence="MEDIUM",
        status=GadgetStatus.ACTIVE,
        attacks_enabled=["seq2seq_backdoor", "cross_modal_injection", "forced_transcription"],
        structural_significance="SECONDARY",
        version="1.0.0",
        changelog=["1.0.0 (2026-05-22): Registered from motifs ASR scan"],
        chainable_with=["CTC_DECODER_STRUCTURE", "SPECIAL_TOKEN_CONTROL_FLOW", "AUDIO_MEL_INPUT"],
    ),
}


# =============================================================================
# ATTACK CHAIN REGISTRY
# =============================================================================

CHAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    
    "CHAIN-PATCH-ATTACK-SURFACE": {
        "name": "Patch Attack Surface",
        "required_gadgets": ["GAP_FC_HEAD"],
        "composition_modifiers": {
            "NO_SPATIAL_ATTENTION": "+EXCEPTIONAL",  # Raises chain significance
            "HAS_SPATIAL_ATTENTION": "-SECONDARY",   # Lowers chain significance
        },
        "research_basis": ["36-GoogleAp-2017", "39-LaVAN-2018"],
        "version": "1.1.0",
    },
    
    "CHAIN-PHYSICAL-WORLD-ATTACK": {
        "name": "Physical-World Attack Surface",
        "required_gadgets": ["ALIASING_DOWNSAMPLE"],
        "min_count": 1,
        "research_basis": ["38-EOT-2018", "60-RP2-2018"],
        "version": "1.0.0",
    },
    
    "CHAIN-COMPOUND-PHYSICAL-PATCH": {
        "name": "Compound Physical Attack Surface",
        "required_gadgets": ["GAP_FC_HEAD", "ALIASING_DOWNSAMPLE"],
        "logic": "AND",  # All must be present
        "structural_significance": "EXCEPTIONAL",
        "research_basis": ["36-GoogleAp-2017", "38-EOT-2018"],
        "version": "1.0.0",
    },
    
    "CHAIN-AT-RESISTANT-ARCHITECTURE": {
        "name": "Adversarial Training Resistance Pattern",
        "required_gadgets": ["HIGH_FANIN_FUSION", "SKIP_CONNECTION"],
        "min_counts": {"HIGH_FANIN_FUSION": 3, "SKIP_CONNECTION": 3},
        "structural_significance": "PRIMARY",
        "research_basis": ["111-DUMBer-2025"],
        "version": "1.0.0",
    },
    
    "CHAIN-VIT-PATCH-ATTACK": {
        "name": "Vision Transformer Patch Attack Surface",
        "required_gadgets": ["VIT_PATCH_EMBEDDING"],
        "optional_gadgets": ["UNREGULARIZED_ATTENTION", "CLS_TOKEN_AGGREGATION"],
        "min_optional": 1,
        "structural_significance": "EXCEPTIONAL",
        "research_basis": ["103-ViTSSL-2024", "114-FakeIt-2024"],
        "version": "1.0.0",
    },
    
    "CHAIN-OBJECT-DISAPPEARANCE": {
        "name": "Object Disappearance Attack Surface",
        "required_gadgets": ["OBJECTNESS_HEAD"],
        "structural_significance": "EXCEPTIONAL",
        "research_basis": ["84-AdvYOLO-2019", "73-ObjectHider-2020"],
        "version": "1.0.0",
    },
    
    "CHAIN-SMALL-OBJECT-SENSITIVITY": {
        "name": "Small Object Attack Sensitivity",
        "required_gadgets": ["AGGRESSIVE_EARLY_DOWNSAMPLING"],
        "optional_gadgets": ["ALIASING_DOWNSAMPLE", "OBJECTNESS_HEAD"],
        "min_optional": 1,
        "structural_significance": "PRIMARY",
        "research_basis": ["107-FDMYOLO-2025"],
        "version": "1.0.0",
    },
    
    # =========================================================================
    # SHADOWLOGIC SUPPLY CHAIN CHAINS
    # =========================================================================
    
    "CHAIN-SHADOWLOGIC-EXISTING-BACKDOOR": {
        "name": "ShadowLogic Backdoor Detected",
        "required_gadgets": ["CONTROL_POINT"],
        "structural_significance": "EXCEPTIONAL",
        "research_basis": ["HiddenLayer-ShadowLogic-2024", "arXiv-2511.00664"],
        "version": "1.0.0",
        "notes": "Indicates potential existing backdoor. Conditional operations are "
                 "extremely rare in standard neural networks.",
    },
    
    "CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY": {
        "name": "ShadowLogic Injection Susceptibility",
        "required_gadgets": ["SHADOWLOGIC_FORMAT_SURFACE", "SHADOWLOGIC_INJECTION_POINT"],
        "optional_gadgets": ["SHADOWLOGIC_CAMOUFLAGE", "SHADOWLOGIC_NO_INTEGRITY"],
        "min_optional": 1,
        "structural_significance": "PRIMARY",
        "composition_modifiers": {
            "SHADOWLOGIC_CAMOUFLAGE": "+PRIMARY",  # Raises chain significance
            "SHADOWLOGIC_NO_INTEGRITY": "+EXCEPTIONAL",  # Raises to EXCEPTIONAL
        },
        "research_basis": ["HiddenLayer-ShadowLogic-2024", "HiddenLayer-PersistentBackdoors-2024"],
        "version": "1.0.0",
        "notes": "Graph structure enables ShadowLogic injection when combined motifs are present. "
                 "Attacker with file access could embed persistent backdoors.",
    },

    "CHAIN-SINGLE-MODALITY-VISION": {
        "name": "Single-Modality Vision Deployment Mismatch",
        "required_gadgets": ["SINGLE_MODALITY_INPUT"],
        "optional_gadgets": ["GAP_FC_HEAD", "OBJECTNESS_HEAD", "DETECTION_HEAD_PATTERN"],
        "min_optional": 1,
        "research_basis": [
            "94-AdversarialBulbs-2021",
            "95-QRAttack-2022",
            "96-HOTCOLD-2022",
            "97-AIP-2023",
            "98-AdvIB-2023",
        ],
        "version": "1.0.0",
        "notes": "Visible-trained vision graph with classifier head; thermal/IR deployment "
                 "is an external channel not shown in ONNX.",
    },

    "CHAIN-PREPROCESSING-TRUST-BOUNDARY": {
        "name": "In-Graph Preprocessing Trust Boundary",
        "required_gadgets": ["IN_GRAPH_PREPROCESSING"],
        "research_basis": [
            "50-Invisibleperturbations-2021",
            "51-AdversarialISP-2021",
        ],
        "version": "1.0.0",
        "notes": "Preprocessing inside ONNX vs off-graph ISP/camera pipeline.",
    },

    "CHAIN-AUDIO-ADVERSARIAL-SURFACE": {
        "name": "Audio Adversarial Surface",
        "required_gadgets": ["AUDIO_MEL_INPUT"],
        "optional_gadgets": ["AUDIO_STRIDE_DOWNSAMPLE", "AUDIO_1D_CONV"],
        "min_optional": 1,
        "research_basis": [
            "Carlini-Audio-2018",
            "70-TPatch-2023",
        ],
        "version": "1.0.0",
    },

    "CHAIN-ACOUSTIC-COMMAND-SURFACE": {
        "name": "Acoustic Command Injection Surface",
        "required_gadgets": [],
        "optional_gadgets": [
            "AUDIO_MEL_INPUT",
            "ENCODER_DECODER_SEQ2SEQ",
            "CTC_DECODER_STRUCTURE",
            "SPECIAL_TOKEN_CONTROL_FLOW",
        ],
        "min_optional": 2,
        "logic": "AUDIO_MEL_INPUT or ENCODER_DECODER_SEQ2SEQ plus "
                 "(CTC_DECODER_STRUCTURE or SPECIAL_TOKEN_CONTROL_FLOW)",
        "research_basis": [
            "102-PG(Poltergeist)-2021",
            "Carlini-Audio-2018",
        ],
        "version": "1.0.0",
        "notes": "ASR/command graphs: hidden acoustic commands when mel + decoder motifs align.",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gadget_info(gadget_id: str) -> Optional[GadgetDefinition]:
    """Get full information about a gadget."""
    return GADGET_REGISTRY.get(gadget_id)


def get_gadgets_by_category(category: GadgetCategory) -> List[GadgetDefinition]:
    """Get all gadgets in a category."""
    return [g for g in GADGET_REGISTRY.values() if g.category == category]


def get_active_gadgets() -> List[GadgetDefinition]:
    """Get all currently active gadgets."""
    return [g for g in GADGET_REGISTRY.values() if g.status == GadgetStatus.ACTIVE]


def get_gadgets_for_attack(attack_type: str) -> List[GadgetDefinition]:
    """Get all gadgets that enable a specific attack type."""
    return [g for g in GADGET_REGISTRY.values() if attack_type in g.attacks_enabled]


def get_research_coverage() -> Dict[str, Any]:
    """Get summary of research coverage across all gadgets."""
    all_papers = set()
    for gadget in GADGET_REGISTRY.values():
        all_papers.update(gadget.research_basis)
    
    return {
        "total_gadgets": len(GADGET_REGISTRY),
        "active_gadgets": len(get_active_gadgets()),
        "total_papers_cited": len(all_papers),
        "by_category": {
            cat.value: len(get_gadgets_by_category(cat)) 
            for cat in GadgetCategory
        },
        "by_confidence": {
            "HIGH": len([g for g in GADGET_REGISTRY.values() if g.confidence == "HIGH"]),
            "MEDIUM": len([g for g in GADGET_REGISTRY.values() if g.confidence == "MEDIUM"]),
            "LOW": len([g for g in GADGET_REGISTRY.values() if g.confidence == "LOW"]),
        },
    }


def validate_registry() -> List[str]:
    """Validate registry integrity. Returns list of issues found."""
    issues = []
    
    for gid, gadget in GADGET_REGISTRY.items():
        # Check chainable_with references exist
        for chain_ref in gadget.chainable_with:
            if chain_ref not in GADGET_REGISTRY:
                issues.append(f"{gid}: chainable_with references unknown gadget '{chain_ref}'")
        
        # Check superseded_by references exist
        if gadget.superseded_by and gadget.superseded_by not in GADGET_REGISTRY:
            issues.append(f"{gid}: superseded_by references unknown gadget '{gadget.superseded_by}'")
        
        # Check research_basis not empty for active gadgets
        if gadget.status == GadgetStatus.ACTIVE and not gadget.research_basis:
            issues.append(f"{gid}: Active gadget has no research_basis")
    
    return issues


def print_registry_summary():
    """Print a summary of the gadget registry."""
    coverage = get_research_coverage()
    
    print("=" * 60)
    print("GADGET REGISTRY SUMMARY")
    print("=" * 60)
    print(f"\nTotal Gadgets: {coverage['total_gadgets']}")
    print(f"Active Gadgets: {coverage['active_gadgets']}")
    print(f"Papers Cited: {coverage['total_papers_cited']}")
    print("\nBy Category:")
    for cat, count in coverage['by_category'].items():
        if count > 0:
            print(f"  {cat}: {count}")
    print("\nBy Confidence:")
    for conf, count in coverage['by_confidence'].items():
        print(f"  {conf}: {count}")
    
    issues = validate_registry()
    if issues:
        print("\n⚠️  Registry Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Registry validation passed")


if __name__ == "__main__":
    print_registry_summary()

