# Detection Rationale and Exploit Chain Analysis

This document explains the thought process behind each vulnerability detection, including:
- How we detect the pattern in the DAG
- Why we consider it exploitable
- The research basis for the claim
- Severity justification
- Potential for false positives

**Purpose:** Ensure detections are grounded in research and not hyperbolic.

**Last Updated:** 2026-01-19

---

## Detection Philosophy

Before diving into specific chains, our detection philosophy:

1. **We detect SUSCEPTIBILITY, not attacks.** The tool identifies architectural patterns that research has shown to enable specific attack techniques. The model may still be robust due to training procedures, but the architecture creates the attack surface.

2. **Severity reflects exploitability.** CRITICAL/HIGH means there are published attacks with working code. MEDIUM means theoretical vulnerability with some evidence. LOW means architectural concern without strong exploitation evidence.

3. **We cite specific papers.** Every detection should trace back to peer-reviewed research demonstrating the exploitation.

4. **False positive awareness.** We acknowledge when a pattern might not be exploitable in practice.

---

## Chain 1: Patch Attack Vulnerability (GoogleAp/LaVAN Pattern)

### What We Detect in the DAG

```
Pattern: GlobalAveragePool (or GlobalMaxPool) -> [Flatten] -> Gemm/MatMul
```

**Detection Code Logic:**
```python
if node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
    # Check if followed by FC layer within 2 hops
    downstream = adjacency.get(node.node_id, [])
    for d in downstream:
        if d_node.op_type in ["Gemm", "MatMul", "Flatten"]:
            has_fc_after = True
```

**What this means:** The model uses global spatial pooling to collapse all spatial dimensions into a single feature vector, which is then fed to a fully-connected classifier.

### Why This Is Exploitable

**The mechanism:**
1. Global pooling (GAP/GMP) computes either mean or max across ALL spatial locations
2. An adversarial patch occupies only a small region (e.g., 10% of image)
3. The patch's features are aggregated into the global representation
4. If patch features are strong enough, they can dominate the pooled representation
5. The FC layer then classifies based on this patch-dominated representation

**Mathematical intuition:**
- GAP computes: `output[c] = mean(feature_map[c, :, :])`
- If a patch creates extreme activations in channel `c`, those activations contribute to the mean
- With enough channels manipulated, the patch can shift the feature vector toward a target class

### Research Basis

**Primary source:** Brown et al., "Adversarial Patch" (NIPS 2017)
- arXiv: https://arxiv.org/abs/1712.09665
- Demonstrated universal patches that cause targeted misclassification
- Patches work regardless of scene content or placement location
- Explicitly exploits that classifiers aggregate spatial information

**Supporting source:** Karmon et al., "LaVAN" (ICML 2018)
- arXiv: https://arxiv.org/abs/1801.02608
- Shows that patches covering only 2% of image area can fool classifiers
- Demonstrates that models rely on local features without global consistency checks

**Key quote from GoogleAp paper:**
> "We create a universal, robust, targeted adversarial image patch... The patch is scene-independent, meaning it can be used to attack any scene..."

### Severity Justification: HIGH

**Why HIGH and not CRITICAL:**
- Attack requires physical access to place patch OR ability to inject into image stream
- Defense exists (patch detection, adversarial training with patches)
- Not all GAP-FC models are equally vulnerable (training matters)

**Why not MEDIUM:**
- Working attack code is publicly available
- Attack has been demonstrated in physical world
- Requires no knowledge of specific model weights

### False Positive Considerations

**When this might NOT be exploitable:**
1. Model was trained with adversarial patch augmentation
2. Model has spatial attention that can filter anomalous regions
3. Very small input resolution limits patch effectiveness
4. Model uses multiple GAP outputs with consistency checking

**What we do:** We flag this as a vulnerability but acknowledge that training procedure (not visible in DAG) affects actual exploitability.

---

## Chain 2: Physical-World Attack Vulnerability (EOT/RP2 Pattern)

### What We Detect in the DAG

```
Pattern: Conv with stride >= 2 in early layers (first 15% of network)
         WITHOUT preceding blur/anti-aliasing operation
```

**Detection Code Logic:**
```python
if node.op_type == "Conv" and any(s >= 2 for s in strides):
    has_blur = self._has_blur_before(node.node_id, reverse_adj, node_map)
    if not has_blur and position == "early":
        # Flag as ALIASING_DOWNSAMPLE
```

**What this means:** The model performs spatial downsampling via strided convolution without low-pass filtering, causing aliasing.

### Why This Is Exploitable

**The mechanism (aliasing):**
1. Nyquist theorem: Downsampling by factor 2 requires filtering out frequencies above 1/2 the new sampling rate
2. Without filtering, high frequencies "fold" into lower frequencies (aliasing)
3. Adversarial perturbations can be crafted at high frequencies
4. When the image is transformed (rotated, scaled), perturbations shift in frequency
5. Aliasing allows these shifted perturbations to persist through downsampling

**Why this matters for physical attacks:**
- Physical world introduces transformations: camera angle, distance, lighting, printing
- Perturbations must survive these transformations to work in the real world
- Aliasing provides a mechanism for perturbations to "survive" transformations
- Without aliasing, perturbations designed for one viewpoint fail at others

**Mathematical intuition:**
- High-frequency perturbation at frequency `f` 
- After downsampling without anti-aliasing, appears at frequency `f mod (sampling_rate/2)`
- This "folded" signal persists even if original perturbation is transformed

### Research Basis

**Primary source:** Athalye et al., "Synthesizing Robust Adversarial Examples" (ICML 2018)
- arXiv: https://arxiv.org/abs/1707.07397
- Introduced Expectation Over Transformation (EOT)
- Created 3D-printed adversarial turtle that fools classifiers from all angles
- Showed that optimizing over transformation distribution creates robust perturbations

**Supporting source:** Eykholt et al., "Robust Physical-World Attacks" (CVPR 2018)
- arXiv: https://arxiv.org/abs/1707.08945  
- Demonstrated physical perturbations on stop signs
- Attacks work from moving vehicles at various distances
- Called "RP2" (Robust Physical Perturbations)

**Supporting source:** Zhang, "Making Convolutional Networks Shift-Invariant Again" (ICML 2019)
- arXiv: https://arxiv.org/abs/1904.11486
- Showed standard CNNs are NOT shift-invariant due to aliasing
- Demonstrated that anti-aliasing (BlurPool) improves robustness

**Key insight from Zhang paper:**
> "Modern CNNs are not shift-invariant... This is due to the subsampling operation, which creates aliasing."

### Severity Justification: HIGH

**Why HIGH:**
- Physical-world attacks demonstrated (not just theoretical)
- Attacks work against safety-critical systems (traffic signs, autonomous vehicles)
- Perturbations can be printed (stickers, graffiti)

**Why not CRITICAL:**
- Requires physical access to environment
- EOT optimization is computationally expensive
- Some models may still be robust due to training

### False Positive Considerations

**When this might NOT be exploitable:**
1. Model trained with extensive transformation augmentation
2. Small number of aliasing points (1-2) may have limited impact
3. Model uses other robustness techniques (adversarial training)

**What we do:** We count the NUMBER of aliasing points. More = higher confidence of vulnerability.

---

## Chain 3: Amplified Multi-Scale Attack Surface

### What We Detect in the DAG

```
Pattern: Concat with >3 inputs (high fan-in fusion)
         OR: Concat followed by MaxPool within 5 hops
```

**Detection Code Logic:**
```python
# High fan-in detection
if node.op_type == "Concat" and len(inputs) > 3:
    # Flag as HIGH_FANIN_FUSION

# MaxPool after fusion detection  
if node.op_type == "MaxPool":
    is_after_fusion = self._is_after_fusion(node.node_id, reverse_adj, node_map)
    if is_after_fusion:
        # Flag as MAXPOOL_AFTER_FUSION
```

### Why This Is Exploitable

**The mechanism (multi-branch + amplification):**

1. **Multi-branch (Inception-style) architectures:**
   - Multiple parallel branches process input at different scales/receptive fields
   - Branches are concatenated to form combined representation
   - Each branch is an independent attack entry point
   - Attacker can optimize perturbations that exploit ANY or ALL branches

2. **MaxPool amplification after fusion:**
   - MaxPool selects the MAXIMUM activation in each region
   - If adversarial perturbation creates a "spike" in one branch, MaxPool preserves it
   - Combined effect: Multi-scale perturbation + spike selection = amplified attack

**Why this is worse than single-branch:**
- More degrees of freedom for the attacker
- Different scales may be vulnerable to different perturbation frequencies
- Optimization can find perturbations that work across all branches
- Result: More robust, transferable attacks

### Research Basis

**Primary source:** Liu et al., "DPatch: An Adversarial Patch Attack on Object Detectors" (AAAI 2019)
- arXiv: https://arxiv.org/abs/1806.02299
- Showed patches can attack object detectors with multi-scale feature pyramids
- Demonstrated transferability across detector architectures
- Exploited that different scales can be attacked independently

**Supporting evidence:** Universal perturbation research shows that multi-branch architectures are easier to attack because:
- Moosavi-Dezfooli et al., "Universal adversarial perturbations" (CVPR 2017)
- Found that universal perturbations exist that fool most images
- Multi-branch provides more "directions" to push the representation

**MaxPool amplification evidence:**
- Multiple papers note that MaxPool "selects" adversarial spikes
- One-pixel attack (Su et al., 2019) explicitly targets MaxPool behavior

### Severity Justification: HIGH

**Why HIGH:**
- DPATCH demonstrated working attacks on YOLO, Faster R-CNN
- Transferability makes attacks easier (don't need exact target model)
- Object detectors are used in safety-critical applications

**Why not CRITICAL:**
- Requires optimization for specific detector architecture
- Some multi-branch architectures may still be robust
- Depends on specific branch configuration

### False Positive Considerations

**When this might NOT be exploitable:**
1. Branches have very similar receptive fields (not truly multi-scale)
2. Model uses gated fusion (channel attention) that can suppress anomalies
3. Model uses AvgPool instead of MaxPool after fusion

**What we do:** We specifically check for MaxPool AFTER Concat (worse) vs MaxPool elsewhere (less severe).

---

## Chain 4: Compound Physical Attack Vulnerability (CRITICAL)

### What We Detect in the DAG

```
Pattern: BOTH of:
  1. GAP_FC_HEAD pattern (GlobalPool -> FC)
  2. ALIASING_DOWNSAMPLE pattern (early stride-2 without blur)
```

**Detection Code Logic:**
```python
if gap_fc_gadgets and aliasing_gadgets:
    # Flag as COMPOUND vulnerability with CRITICAL severity
```

### Why This Is Exploitable

**The compound mechanism:**
This combines TWO independent vulnerability types:

1. **Aliasing enables transformation-robust perturbations** (from EOT research)
   - Perturbations survive rotation, scaling, lighting changes
   - Can be printed and work in physical world

2. **GAP-FC enables localized patch attacks** (from GoogleAp research)
   - Small patches can dominate global representation
   - Scene-independent, works anywhere

**Combined effect:**
- Physical patches that work from any angle, distance, lighting
- Best of both attack types
- This is exactly what was demonstrated in the original EOT paper (3D printed turtle)

### Research Basis

**This is not speculation - it's exactly what the research demonstrates:**

1. EOT paper (Athalye 2018) used GoogleAp-style patch optimization WITH transformation robustness
2. The 3D-printed turtle was a physical object with patches that work from all angles
3. The turtle fooled ImageNet classifiers that have BOTH patterns

**Direct evidence:** The fact that 3D adversarial objects exist and work proves that models with both vulnerabilities are exploitable.

### Severity Justification: CRITICAL

**Why CRITICAL (highest severity):**
- Not theoretical - demonstrated with working physical objects
- Attacks are transformation-robust AND scene-independent
- Affects safety-critical applications (autonomous vehicles)
- Relatively easy to execute (print a patch, place anywhere)

**This is our only CRITICAL finding** because it combines two well-documented, independently-exploitable vulnerabilities.

### False Positive Considerations

**When this might be overstated:**
1. Model may have been specifically hardened against this combination
2. Very aggressive adversarial training may mitigate both vulnerabilities
3. Input preprocessing (JPEG compression, denoising) may break perturbations

**What we do:** We flag this as CRITICAL but include mitigation recommendations. The architecture IS vulnerable; training may provide defense.

---

## Chain 5: Early MaxPool Amplification

### What We Detect in the DAG

```
Pattern: MaxPool operations in first 20% of network layers
```

**Detection Code Logic:**
```python
early_amplifiers = [g for g in gadgets 
                   if g.gadget_type == GadgetType.AMPLIFIER and g.position == "early"]
maxpool_early = [g for g in early_amplifiers if g.op_type == "MaxPool"]
```

### Why This Is Exploitable

**The mechanism:**
1. Early layers process raw pixel information
2. MaxPool in early layers selects maximum activations BEFORE deeper processing
3. Adversarial "spikes" introduced at input level are immediately amplified
4. Later layers never see the non-spike information that MaxPool discarded

**Why early is worse than late:**
- Early MaxPool: perturbation amplified, then processed by entire network
- Late MaxPool: perturbation already diluted by many layers before amplification

### Research Basis

**Primary evidence:** One-pixel attack (Su et al., 2019)
- arXiv: https://arxiv.org/abs/1710.08864
- Shows that changing single pixels can fool classifiers
- Explicitly targets models with pooling that selects extreme values

**Supporting evidence:** Patch attack research shows localized perturbations are more effective when amplification happens early.

### Severity Justification: MEDIUM

**Why MEDIUM (not HIGH):**
- One-pixel attacks require optimization for specific images
- Early MaxPool alone doesn't guarantee vulnerability
- Many successful models use early MaxPool (VGG, early ResNets)

**Why not LOW:**
- There IS research demonstrating exploitation
- Sparse attacks (patches, few pixels) specifically benefit from this

### False Positive Considerations

**When this might NOT matter:**
1. MaxPool with small kernel (2x2) has limited amplification
2. Followed immediately by normalization that dampens spikes
3. Model uses other robustness techniques

---

## Chain 6: Gradient Highway (Skip Connections)

### What We Detect in the DAG

```
Pattern: Add operation (residual) where skip path spans >10 layers
```

**Detection Code Logic:**
```python
if node.op_type == "Add":
    skip_distance = max(abs(idx - inp_idx) for inp_idx in input_indices)
    if skip_distance > 5:
        # Flag as SKIP_CONNECTION gadget
```

### Why This Is Exploitable

**The mechanism:**
1. Residual connections create "shortcut" paths for gradients
2. During adversarial optimization (PGD, C&W), gradients flow through these shortcuts
3. Longer skips = more direct gradient path from loss to early layers
4. Result: Faster attack convergence, stronger perturbations

**Mathematical intuition:**
- Standard path: gradients pass through many ReLUs (can die), many weights (can vanish)
- Skip path: gradients flow directly, unimpeded
- Attacker benefits: cleaner gradient signal for optimization

### Research Basis

**Evidence from adversarial attack research:**
- PGD attacks (Madry et al., 2018) work better on ResNets than VGG
- ResNets have skip connections; VGG doesn't
- Gradient-based attacks benefit from stable gradient flow

**Note:** This is more of an "attack efficiency" factor than a fundamental vulnerability. Models with skip connections aren't necessarily MORE vulnerable, but attacks are FASTER to compute.

### Severity Justification: MEDIUM

**Why MEDIUM:**
- Affects attack EFFICIENCY more than fundamental exploitability
- All models are vulnerable to gradient attacks given enough iterations
- Skip connections also provide some robustness benefits (ensembling effect)

**Why not HIGH:**
- Not a fundamental vulnerability, just makes optimization easier
- Many robust models use skip connections

### False Positive Considerations

**When this is overstated:**
1. This affects white-box attacks (attacker has model access)
2. Black-box attacks don't benefit from gradient highways
3. Skip connections may actually improve robustness in some cases

**What we do:** We flag this as MEDIUM and frame it as "attack efficiency" rather than fundamental vulnerability.

---

## Summary: Severity Calibration

| Severity | Criteria | Examples |
|----------|----------|----------|
| **CRITICAL** | Multiple independent vulns combining; demonstrated physical exploits | Compound GAP-FC + aliasing |
| **HIGH** | Single well-documented vuln; published working attacks | GAP-FC alone, aliasing alone, multi-scale+MaxPool |
| **MEDIUM** | Theoretical concern; some evidence; affects attack efficiency | Early MaxPool, gradient highways |
| **LOW** | Architectural observation; weak exploitation evidence | Single late-stage gadgets |

---

## Potential Overclaiming Risks

### What we might be overstating:

1. **Training matters:** We can only analyze architecture, not training. A model trained with adversarial patches may be robust despite having GAP-FC.

2. **Input preprocessing:** External preprocessing (JPEG, resize, normalize) may break perturbations before they reach the model.

3. **Ensemble effects:** Models with many branches may have emergent robustness from redundancy.

4. **Specific configurations:** A 4-branch Concat isn't necessarily more vulnerable than 3-branch; thresholds are somewhat arbitrary.

### What we're NOT detecting (blind spots):

1. **Training-time defenses:** Adversarial training, certified training, etc.
2. **Runtime defenses:** Input validation, anomaly detection
3. **Actual attack success rate:** We detect architecture, not empirical vulnerability
4. **Model capacity for robust features:** Some architectures learn more robust features

---

## Recommended Disclaimer for Reports

> This analysis identifies architectural patterns that research has shown to enable adversarial attacks. The presence of these patterns indicates SUSCEPTIBILITY but not guaranteed EXPLOITABILITY. Actual vulnerability depends on training procedures, input preprocessing, and deployment context. All severity ratings are based on published attack research and should be validated through empirical testing.

---

## References

1. Brown et al., "Adversarial Patch" (2017) - https://arxiv.org/abs/1712.09665
2. Athalye et al., "Synthesizing Robust Adversarial Examples" (2018) - https://arxiv.org/abs/1707.07397
3. Karmon et al., "LaVAN" (2018) - https://arxiv.org/abs/1801.02608
4. Eykholt et al., "Robust Physical-World Attacks" (2018) - https://arxiv.org/abs/1707.08945
5. Liu et al., "DPatch" (2019) - https://arxiv.org/abs/1806.02299
6. Zhang, "Making CNNs Shift-Invariant" (2019) - https://arxiv.org/abs/1904.11486
7. Su et al., "One Pixel Attack" (2019) - https://arxiv.org/abs/1710.08864
8. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018) - https://arxiv.org/abs/1706.06083

---

## Deployment-context motifs (export framing)

CLI and JSON export use registry titles only: no severity tiers, CVSS, or CRITICAL/HIGH labels. Modifier motif keys in `CHAIN_REGISTRY` are internal composition hints, not user-facing risk scores.

---

## CHAIN-SINGLE-MODALITY-VISION

### What we detect

- `SINGLE_MODALITY_INPUT`: one `graph.input`, no early `MULTIMODAL_FUSION_POINT` / dual-encoder alignment
- Plus `GAP_FC_HEAD` or `OBJECTNESS_HEAD` on the same graph

### Why it matters for RE

Thermal papers (94-98) attack hardware channels while the ONNX file stays a visible-trained vision DAG. The chain links deployment mismatch literature to standard classifier motifs.

### False positives

Multimodal models with a single exported input wrapper may still show single input; check `HAS_MULTIMODAL_FUSION`.

---

## CHAIN-PREPROCESSING-TRUST-BOUNDARY

### What we detect

- `IN_GRAPH_PREPROCESSING`: stem `Sub`/`Div`/`Mul`, norm, or `Cast` in the first ~15% of hops before backbone

### Research basis

Papers 50-51 (camera/ISP). Off-graph ISP is noted in static limits when preprocessing is not exported.

---

## CHAIN-AUDIO-ADVERSARIAL-SURFACE

### What we detect

- `AUDIO_MEL_INPUT` plus `AUDIO_STRIDE_DOWNSAMPLE` or `AUDIO_1D_CONV`

### Research basis

Carlini audio adversarial examples; TPatch (70) sensor-coupling context.

---

## CHAIN-ACOUSTIC-COMMAND-SURFACE

### What we detect

- (`AUDIO_MEL_INPUT` or `ENCODER_DECODER_SEQ2SEQ`) and (`CTC_DECODER_STRUCTURE` or `SPECIAL_TOKEN_CONTROL_FLOW`)

### Research basis

Poltergeist (102) hidden commands on ASR/voice graphs.

