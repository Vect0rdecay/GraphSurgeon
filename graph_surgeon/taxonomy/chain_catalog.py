"""Long-form chain narratives for catalog output (GraphSurgeon canonical)."""

CHAIN_NARRATIVES: dict[str, str] = {
    "CHAIN-PATCH-ATTACK-SURFACE": (
        "Combines GAP_FC_HEAD with NO_SPATIAL_ATTENTION to describe the classic patch-attack "
        "landscape on CNN classifiers. Global pooling collapses spatial dimensions so a localized "
        "patch anywhere in the frame can influence the pooled representation; without spatial "
        "attention the network cannot learn to down-weight anomalous regions. This chain is "
        "present in most ResNet/VGG-style classifiers and indexes the GoogleAp / LaVAN literature "
        "on universal and localized visible patches."
    ),
    "CHAIN-PHYSICAL-WORLD-ATTACK": (
        "Combines ALIASING_DOWNSAMPLE with NORMALIZER to index physical-world attack literature "
        "(EOT, RP2, printed patches). Early strided downsampling without anti-aliasing lets "
        "high-frequency perturbations fold into lower frequencies and survive camera/print "
        "pipelines; BatchNorm fixed statistics amplify distribution shift under varying lighting."
    ),
    "CHAIN-COMPOUND-PHYSICAL-PATCH": (
        "Requires both GAP_FC_HEAD and ALIASING_DOWNSAMPLE: physical-world patches that remain "
        "effective after transformation AND dominate a global-pool classifier head. Indexes "
        "compound demonstrations combining patch placement with EOT-style robustness."
    ),
    "CHAIN-AT-RESISTANT-ARCHITECTURE": (
        "HIGH_FANIN_FUSION plus SKIP_CONNECTION in quantity (3+ each) marks architectures where "
        "standard adversarial training may underperform (DUMBer 2025). Multiple gradient highways "
        "and fusion entry points complicate robust optimization."
    ),
    "CHAIN-VIT-PATCH-ATTACK": (
        "ViT patch embedding with unregularized attention and CLS aggregation indexes ViT-specific "
        "patch and attention-hijacking literature. Patch tokens provide a single entry point; CLS "
        "aggregates all tokens without spatial filtering analogous to GAP→FC in CNNs."
    ),
    "CHAIN-OBJECT-DISAPPEARANCE": (
        "OBJECTNESS_HEAD enables objectness-suppression attacks (Adversarial YOLO, Object Hider). "
        "Suppressing objectness can make detections disappear rather than merely mislabel."
    ),
    "CHAIN-SMALL-OBJECT-SENSITIVITY": (
        "AGGRESSIVE_EARLY_DOWNSAMPLING with optional aliasing or objectness motifs indexes small-object "
        "and small-patch effectiveness in detectors and classifiers."
    ),
    "CHAIN-SHADOWLOGIC-EXISTING-BACKDOOR": (
        "CONTROL_POINT motifs (Where/If/Equal) may indicate existing conditional backdoor logic "
        "in the graph. Rare in standard feed-forward classifiers; warrants supply-chain review."
    ),
    "CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY": (
        "Editable ONNX format, injection points, and missing integrity verification index "
        "ShadowLogic-style graph backdoors that persist through conversion and fine-tuning."
    ),
    "CHAIN-SINGLE-MODALITY-VISION": (
        "Single ONNX input with a standard vision classifier or detector head indexes "
        "visible-trained graphs deployed on non-visible sensors (thermal/IR). Papers 94-98 "
        "use hardware channels; the DAG still shows GAP_FC_HEAD-style motifs."
    ),
    "CHAIN-PREPROCESSING-TRUST-BOUNDARY": (
        "Stem Sub/Div/Mul, Cast, or normalization inside the first ~15% of the DAG places "
        "the pixel statistics trust boundary in ONNX (papers 50-51). Off-graph ISP is not "
        "visible but analysts should compare in-graph vs camera pipeline."
    ),
    "CHAIN-AUDIO-ADVERSARIAL-SURFACE": (
        "Mel-spectrogram input plus strided audio frontend indexes Carlini-style audio "
        "adversarial and TPatch sensor-coupling context (paper 70)."
    ),
    "CHAIN-ACOUSTIC-COMMAND-SURFACE": (
        "Mel or seq2seq ASR topology with CTC or special-token control flow indexes hidden "
        "command injection (Poltergeist, paper 102) when the graph is audio-native."
    ),
}

CHAIN_DETECTION_RATIONALE: dict[str, str] = {
    "CHAIN-PATCH-ATTACK-SURFACE": (
        "Detect GlobalAveragePool/GlobalMaxPool followed within two hops by Gemm/MatMul/Flatten. "
        "Global pooling aggregates all spatial locations; a patch occupying a small region can "
        "shift the pooled feature vector that the FC layer classifies. Brown et al. (GoogleAp) "
        "show universal scene-independent patches; Karmon et al. (LaVAN) show ~2% area suffices. "
        "Architecture enables the attack class; training and defenses determine exploitability."
    ),
    "CHAIN-PHYSICAL-WORLD-ATTACK": (
        "Detect ALIASING_DOWNSAMPLE (stride-2 Conv/Pool without preceding blur) together with "
        "NORMALIZER (BatchNorm with fixed running statistics). High-frequency perturbations can "
        "fold through strided ops; fixed BN stats amplify illumination and capture-pipeline shift. "
        "Indexes EOT and RP2 physical-world literature. Training and deployment context determine "
        "whether attacks succeed."
    ),
    "CHAIN-COMPOUND-PHYSICAL-PATCH": (
        "Require both GAP_FC_HEAD and ALIASING_DOWNSAMPLE: a classifier head that global-pools "
        "spatial features plus early aliasing-prone downsampling. Indexes compound physical-world "
        "patch demonstrations where transformation robustness and global aggregation interact."
    ),
    "CHAIN-SINGLE-MODALITY-VISION": (
        "Require SINGLE_MODALITY_INPUT and at least one of GAP_FC_HEAD or OBJECTNESS_HEAD. "
        "Thermal and IR attacks operate outside the DAG; this chain links deployment mismatch "
        "to standard vision structural motifs."
    ),
    "CHAIN-PREPROCESSING-TRUST-BOUNDARY": (
        "Emit when IN_GRAPH_PREPROCESSING is detected in the stem (early Sub/Div/Mul, norm, "
        "or Cast before first Conv)."
    ),
    "CHAIN-AUDIO-ADVERSARIAL-SURFACE": (
        "Require AUDIO_MEL_INPUT plus AUDIO_STRIDE_DOWNSAMPLE or AUDIO_1D_CONV."
    ),
    "CHAIN-ACOUSTIC-COMMAND-SURFACE": (
        "Require (AUDIO_MEL_INPUT or ENCODER_DECODER_SEQ2SEQ) and "
        "(CTC_DECODER_STRUCTURE or SPECIAL_TOKEN_CONTROL_FLOW)."
    ),
}
