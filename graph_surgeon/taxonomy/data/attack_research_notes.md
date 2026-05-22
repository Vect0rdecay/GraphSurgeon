# Attack Research Notes (GraphSurgeon corpus)

Per-paper adversarial ML analysis for ONNX reverse engineering. Each `### [id]` section maps
literature to structural motifs (attack landscape from graph structure, not exploitability).

Authoring spec: `RESEARCH_NOTE_TEMPLATE.md`. Coverage: `research_coverage.json`.

---

## Light-based attacks (shared ONNX mechanism)

Papers 40, 43, 48, 52, 57, 65, 66, 80, 81, 82 exploit illumination, projection, or weather-like perturbations.
The ONNX attack landscape for lighting mirrors patch attacks: ALIASING_DOWNSAMPLE folds high-frequency lighting edges,
NORMALIZER amplifies distribution shift under changed illumination, NO_SPATIAL_ATTENTION cannot suppress localized
bright/dark regions, and GAP_FC_HEAD aggregates spatial perturbations into the classifier head.

---

## Deployment context beyond the DAG

Thermal emitters, camera ISP blocks, and room acoustics are often **outside** the ONNX file. GraphSurgeon still indexes graph-visible deployment signals so reverse engineers can tie papers to structure:

| Signal | Registry motif | What the DAG shows |
|--------|----------------|-------------------|
| Visible-trained model, non-visible sensor | `SINGLE_MODALITY_INPUT`, `CHAIN-SINGLE-MODALITY-VISION` | One graph input, standard vision head |
| Preprocessing inside export | `IN_GRAPH_PREPROCESSING`, `CHAIN-PREPROCESSING-TRUST-BOUNDARY` | Stem Sub/Div/Mul, norm, Cast before Conv |
| Multimodal in export | `HAS_MULTIMODAL_FUSION` | Early fusion or dual-encoder alignment |
| Audio/ASR graph | `AUDIO_*`, `CHAIN-AUDIO-*`, `CHAIN-ACOUSTIC-COMMAND-SURFACE` | Mel stem, CTC, special tokens |

Record sensors, ISP version, and microphone path separately from `motifs` output.

---
### [36] GoogleAp - Adversarial Patch (NIPS 2017)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "Adversarial Patch" - Brown et al., OpenAI

**Attack Mechanism:**
- Creates a universal, scene-independent patch that causes targeted misclassification
- Patch can be printed and placed anywhere in the physical world
- Optimizes patch to maximize P(target_class | patch present anywhere in image)
- Uses Expectation Over Transformation (EOT) during optimization to make patch robust

**Key Innovation:**
- Previous adversarial examples were image-specific and imperceptible
- Adversarial Patch is universal (works on any image), printable, and visible
- Works regardless of where patch is placed in the scene

**Attack landscape factors (graph-detectable):**
1. **Global Aggregation:** Models with global pooling allow small regions to dominate
2. **Lack of Context:** Models don't validate that patch region is semantically consistent
3. **No Attention:** Models without spatial attention can't learn to ignore patches
4. **Large Receptive Fields:** Patch features propagate to all downstream activations

**DAG Indicators for Vulnerability:**
- [x] GlobalAveragePool or GlobalMaxPool before classifier
- [x] No spatial attention mechanism
- [x] Single FC layer classifier (linear in feature space)
- [x] Large effective receptive field covering patch region

**Gadgets that Enable Patch Attacks:**
1. `GAP_FC_Head`: GlobalAveragePool -> Flatten -> Gemm
2. `No_Attention`: Lack of attention modules
3. `Large_Receptive_Field`: Stacked convolutions with large cumulative receptive field

**Detection Logic:**
```
IF has_global_pool_fc_head AND 
   NOT has_spatial_attention AND
   receptive_field > input_size * 0.3:
   Applicable attack class: patch attack surface
```

**Hardening Recommendations:**
- Add spatial attention mechanisms
- Use attention-based pooling instead of GAP
- Implement patch detection/masking in preprocessing
- Train with adversarial patch augmentation

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [37] PAE - PAE (Physical Adversarial Examples) (AISS 2018)

**Status:** analysis_complete
**Attack form:** Printed images
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Physical perturbation (printed stickers on real objects) Introduces **RP2 (Robust Physical Perturbations)** algorithm that generates spatially-constrained perturbations robust to environmental conditions (viewpoint, distance, lighting). Uses black and white

**Attack mechanism:**
Introduces **RP2 (Robust Physical Perturbations)** algorithm that generates spatially-constrained perturbations robust to environmental conditions (viewpoint, distance, lighting). Uses black and white stickers designed to look like vandalism/art to avoid detection. Perturbations are optimized to survive the print-photograph pipeline.

**ONNX graph indicators:**
- **No anti-aliasing before downsampling** - High-frequency sticker patterns must fold into features
- **Stride-2 convolutions/pooling** - Creates the aliasing effect
- **Standard preprocessing** - No frequency filtering at input

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE
- Also relates to: NORMALIZER (distribution shift under varying conditions)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Two-stage evaluation methodology** - Lab tests + field tests for physical adversarial
2. **Vandalism/art camouflage** - Perturbations designed to appear innocuous
3. **Spatial constraints** - Perturbations limited to printable sticker regions

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [38] EOT - Expectation Over Transformation (PMLR 2018)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "Synthesizing Robust Adversarial Examples" - Athalye et al.

**Attack Mechanism:**
- Creates adversarial examples that remain effective under physical-world transformations
- Optimizes: argmax_delta E_t~T [log P(y_target | t(x + delta))]
- Distribution T includes: rotation, scaling, lighting, camera noise, viewpoint changes
- Demonstrated with 3D-printed adversarial turtle that fools classifier from all angles

**Key Innovation:**
- Previous adversarial examples failed when printed/photographed due to transformations
- EOT makes perturbations "survive" the physical-to-digital pipeline
- Enables actual physical-world adversarial objects

**Attack landscape factors (graph-detectable):**
1. **Aliasing from Downsampling:** Stride-2 without anti-aliasing lets high-freq perturbations fold into lower frequencies, surviving transformation
2. **No Transformation Invariance:** Models without built-in transformation invariance
3. **High-Frequency Sensitivity:** Early layers sensitive to high-frequency components
4. **Lack of Data Augmentation:** (Not DAG-detectable, but training-related)

**DAG Indicators for Vulnerability:**
- [x] Stride-2 Conv in first 3 layers without blur/anti-alias
- [x] No BlurPool or anti-aliasing operations anywhere
- [x] MaxPool early in the network
- [x] Aggressive spatial reduction (input quickly reduced to small spatial dims)

**Gadgets that Enable EOT Attacks:**
1. `Early_Stride2_Conv`: Conv with stride=2 in layer position < 3
2. `No_Antialiasing`: Absence of blur/anti-alias before any strided operation
3. `Aggressive_Downsampling`: Rapid spatial dimension reduction
4. `Early_MaxPool`: MaxPool within first 5 layers

**Detection Logic:**
```
aliasing_risk = 0
FOR each Conv with stride > 1:
   IF position < 5 AND no_preceding_blur:
      aliasing_risk += 1
FOR each Pool with stride > 1:
   IF no_preceding_blur:
      aliasing_risk += 1
      
IF aliasing_risk > 2:
   FLAG as EOT_ATTACK_VULNERABLE
```

**Hardening Recommendations:**
- Use anti-aliased downsampling (BlurPool)
- Move aggressive downsampling later in network
- Add blur before all strided operations
- Train with transformation augmentation

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [39] LaVAN - Localized and Visible Adversarial Noise (ICML 2018)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "LaVAN: Localized and Visible Adversarial Noise" - Karmon et al.

**Attack Mechanism:**
- Unlike Adversarial Patch, LaVAN constrains perturbation to be localized and visible
- Shows that even small visible patches (covering 2% of image) can cause misclassification
- Demonstrates that models rely on local features without global context validation

**Attack landscape factors (graph-detectable):**
1. **Local Feature Dominance:** Features from small regions can dominate final prediction
2. **No Context Validation:** Model doesn't verify spatial consistency
3. **High Local Receptive Field Sensitivity:** Local perturbations propagate

**DAG Indicators for Vulnerability:**
- [x] Same as GoogleAp (global pooling without attention)
- [x] Small effective receptive field (model looks at parts, not whole)
- [x] No multi-scale feature verification

**Relationship to GoogleAp:**
- LaVAN and GoogleAp exploit similar model weaknesses
- LaVAN is more constrained (must be visible, in specific location)
- Both exploit global aggregation vulnerability

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [40] LightAttack - Light-based attack (AAAI-S 2018)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** LightAttack manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [41] D2P - D2P (Digital-to-Physical) (AAAI 2019)

**Status:** analysis_complete
**Attack form:** Printed images
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Physical perturbation (printed images) Uses an **image-to-image translation network** to simulate the digital-to-physical transformation during adversarial generation. The network learns the degradation from printing and photographing, all

**Attack mechanism:**
Uses an **image-to-image translation network** to simulate the digital-to-physical transformation during adversarial generation. The network learns the degradation from printing and photographing, allowing adversarial examples to be optimized to survive this pipeline.

**ONNX graph indicators:**
- **Aliasing in downsampling** - Perturbations survive frequency-domain degradation
- **No learned print-photo robustness** - Model wasn't trained on printed/photographed data
- **Standard CNN architecture** - Predictable frequency response

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE
- Also relates to: NO_COLOR_NORMALIZATION (color shifts from printing)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Learned simulation of physical pipeline** - Image-to-image network models print-photo degradation
2. **Large-scale physical validation** - 3,000+ manually captured photos
3. **Transferability improvement** - Simulated physical conditions improve cross-domain transfer

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [42] ACS - Adversarial Camera Stickers (ACS) (PMLR 2019)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** GAP_FC_HEAD

**Summary:** Physical perturbation (camera-based) Places **translucent stickers on the camera lens** to create universal perturbations of all observed images. Unlike traditional adversarial patches that modify objects or digital inputs, this manipula

**Attack mechanism:**
Places **translucent stickers on the camera lens** to create universal perturbations of all observed images. Unlike traditional adversarial patches that modify objects or digital inputs, this manipulates the camera itself. The stickers create physical perturbations that persist across all images captured by the camera.

Key approach:
- Iterative optimization of sticker perturbation
- Updates threat model alongside perturbation for physical realizability
- Targeted misclassification (force specific wrong class)

**ONNX graph indicators:**
- **GAP_FC_HEAD** - Camera-wide perturbation must affect aggregated features
- **Universal applicability** - Perturbation applies to ALL captured images
- **Physical realizability** - Translucent stickers must survive optical system

**Gadget and chain mapping:**
- Confirms: GAP_FC_HEAD
- Also relates to: NORMALIZER (input statistics shift)
- New gadget needed: No (though CAMERA_PERTURBATION could be defined as attack vector, not architectural gadget)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget GAP_FC_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PATCH-ATTACK-SURFACE

**Related literature:**
1. **Attack vector shift** - From object/input to camera itself
2. **Universal by design** - All images affected, not just one
3. **Translucency requirement** - Novel physical constraint
4. **Targeted attacks** - 49.6% success rate for targeted misclassification on ImageNet

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [43] ProjectorAttack - Light-based attack (S&P 2019)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** ProjectorAttack manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [44] Adversarial ACO - Bias-Based Universal Adversarial Patch Attack for Automatic Check-Out (ECCV 2020)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** GAP_FC_HEAD

**Summary:** Patch attack (universal, class-agnostic) Exploits both **perceptual bias** (low-level features DNNs rely on) and **semantic bias** (class-specific feature correlations) to generate universal adversarial patches. Unlike previous patch attacks

**Attack mechanism:**
Exploits both **perceptual bias** (low-level features DNNs rely on) and **semantic bias** (class-specific feature correlations) to generate universal adversarial patches. Unlike previous patch attacks that target specific classes, this creates class-agnostic patches that generalize across product categories.

**ONNX graph indicators:**
- **Global Average Pooling → FC head** - The patch must affect aggregated features
- **Shared feature extraction** - Same backbone processes all product classes
- **No spatial attention** - Model cannot learn to ignore patch region

**Gadget and chain mapping:**
- Confirms: GAP_FC_HEAD
- Also relates to: NO_SPATIAL_ATTENTION, SHARED_BACKBONE
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget GAP_FC_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PATCH-ATTACK-SURFACE

**Related literature:**
1. **Dual-bias exploitation** - Combining perceptual + semantic biases creates stronger universal patches than either alone
2. **Class-agnostic universality** - Important for real-world retail scenarios where product classes are dynamic
3. **RPC dataset validation** - First adversarial patch work on automatic checkout domain

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [45] Adv-watermark - Adv-watermark (ACM MM 2020)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** GAP_FC_HEAD

**Summary:** Perturbation attack (watermark-based) Generates adversarial examples by embedding **meaningful watermarks** (logos, text, images) rather than random noise. Uses **Basin Hopping Evolution (BHE)** optimization for black-box attacks with few

**Attack mechanism:**
Generates adversarial examples by embedding **meaningful watermarks** (logos, text, images) rather than random noise. Uses **Basin Hopping Evolution (BHE)** optimization for black-box attacks with few queries. Watermarks are embedded using discrete wavelet transform (DWT) and discrete cosine transform (DCT) based Patchwork algorithms.

**ONNX graph indicators:**
- **GAP aggregates watermark perturbations** - Distributed watermark affects pooled features
- **Frequency-domain attack surface** - DWT/DCT perturbations survive model processing
- **No input validation** - Watermarked images pass through unchanged

**Gadget and chain mapping:**
- Confirms: GAP_FC_HEAD (watermark perturbations aggregate through GAP)
- Also relates to: ALIASING_DOWNSAMPLE (frequency domain attacks)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget GAP_FC_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PATCH-ATTACK-SURFACE

**Related literature:**
1. **Meaningful perturbations** - Watermarks appear legitimate, reducing suspicion
2. **Frequency-domain embedding** - DWT/DCT provide robust perturbation encoding
3. **Query-efficient black-box** - BHE requires only ~1.17 seconds per image on CIFAR-10
4. **Robustness to defenses** - Superior to state-of-the-art against image transformation defenses

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [46] ABBA - ABBA (Uncertain - Possibly Physical Backdoor) (NeurIPS 2020)

**Status:** analysis_complete
**Attack form:** Printed image
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** ABBA (2020) adversarial attack.

**Attack mechanism:**
(see BATCH harvest)

**ONNX graph indicators:**
- See gadget detection_logic in registry for op-level patterns

**Gadget and chain mapping:**
Confirms ALIASING_DOWNSAMPLE.

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [47] ViewFool - Viewpoint adversarial examples (NeurIPS 2020)

**Status:** analysis_complete
**Attack form:** Position
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** ViewFool crafts adversarial viewpoints (camera pose) that fool classifiers by exploiting lack of viewpoint invariance.

**Attack mechanism:**
Viewpoint changes introduce frequency shifts and projection artifacts similar to physical-world transformations; models without aliasing-resistant downsampling misclassify.

**ONNX graph indicators:**
- Stride-2 operations without anti-aliasing blur
- Global pooling classifier heads sensitive to global feature shifts

**Gadget and chain mapping:**
ALIASING_DOWNSAMPLE; related to CHAIN-PHYSICAL-WORLD-ATTACK.

**What GraphSurgeon surfaces:**
`catalog --gadget ALIASING_DOWNSAMPLE`.

**Static analysis limits:**
Viewpoint is an extrinsic camera parameter; ONNX graph does not encode pose robustness.

---
### [48] SLMAttack - Light-based attack (ArXiv 2021)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** SLMAttack manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [49] Meta-Attack - Meta adversarial attack (ICCV 2021)

**Status:** analysis_complete
**Attack form:** Image
**Registry:** (conceptual — exploits common gadget combinations)

**Summary:** Meta-Attack transfers adversarial examples across models by exploiting shared architectural weaknesses rather than a single new motif.

**Attack mechanism:**
Uses meta-learning to find perturbations effective on multiple architectures sharing pooling, fusion, and classifier patterns.

**ONNX graph indicators:**
- Combinations of GAP_FC_HEAD, ALIASING_DOWNSAMPLE, HIGH_FANIN_FUSION as detected by motifs

**Gadget and chain mapping:**
No separate gadget; confirms that shared structural motifs increase transfer risk across models.

**What GraphSurgeon surfaces:**
Aggregate motif report; multiple catalog gadget hits on one graph.

**Static analysis limits:**
Transfer success depends on training; not predicted from graph alone.

---
### [50] Invisible perturbations (CVPR 2021)

**Status:** analysis_complete
**Attack form:** Camera
**Registry:** IN_GRAPH_PREPROCESSING, SINGLE_MODALITY_INPUT, NORMALIZER

**Summary:** Invisible perturbations manipulate the camera capture pipeline so human-visible scenes differ from sensor statistics the network sees. When preprocessing is baked into ONNX, stem normalize/scale motifs mark the in-graph trust boundary.

**Attack mechanism:**
Perturbations target sensor-visible frequencies or ISP-adjacent statistics while remaining imperceptible on a display. The classifier still receives a tensor produced by the full capture stack.

**ONNX graph indicators:**
- Early `Sub`/`Div`/`Mul` with constants, `InstanceNormalization`/`BatchNormalization` at stem
- `Cast` before first `Conv`
- Otherwise standard vision motifs (`GAP_FC_HEAD`, `ALIASING_DOWNSAMPLE`)

**Gadget and chain mapping:**
- `IN_GRAPH_PREPROCESSING` when stem ops appear in the first ~15% of hops
- `CHAIN-PREPROCESSING-TRUST-BOUNDARY` when that motif is present
- `SINGLE_MODALITY_INPUT` for single-tensor vision exports

**What GraphSurgeon surfaces:**
`motifs` and `patterns` emit preprocessing and deployment-context motifs; `catalog --gadget IN_GRAPH_PREPROCESSING` links this paper.

**Static analysis limits:**
Off-graph ISP and RAW sensor behavior are not in the ONNX file. Exploitability depends on capture hardware, not graph topology alone.

---
### [51] Adversarial ISP (CVPR 2021)

**Status:** analysis_complete
**Attack form:** Camera
**Registry:** IN_GRAPH_PREPROCESSING, CHAIN-PREPROCESSING-TRUST-BOUNDARY, NORMALIZER

**Summary:** Adversarial ISP attacks optimize camera image signal processing parameters. When ISP-like normalization is exported into ONNX, the DAG shows an in-graph preprocessing trust boundary distinct from off-graph camera firmware.

**Attack mechanism:**
Attackers search ISP parameter space (demosaic, gamma, color matrix) so downstream classifiers fail while images look benign. Exported graphs may fold a subset of that pipeline into early ops.

**ONNX graph indicators:**
- Stem `Sub`/`Div`/`Mul`, `BatchNormalization`/`InstanceNormalization` before backbone
- Fixed constant tensors paired with element-wise ops (scale/bias patterns)

**Gadget and chain mapping:**
- `IN_GRAPH_PREPROCESSING` indexes in-export preprocessing
- `CHAIN-PREPROCESSING-TRUST-BOUNDARY` compounds the deployment-context chain

**What GraphSurgeon surfaces:**
`catalog --chain CHAIN-PREPROCESSING-TRUST-BOUNDARY`; `motifs` on models with stem preprocessing.

**Static analysis limits:**
Full ISP in camera firmware is not visible in ONNX. Paper applies fully only when preprocessing is part of the exported graph.

---
### [52] AdvLB - Light-based attack (CVPR 2021)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** AdvLB manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [54] TnT attack - TnT Attacks (TIFS 2022)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** GAP_FC_HEAD

**Summary:** Patch attack (universal, naturalistic, location-independent) Creates **naturalistic-looking patches** (e.g., images of flowers, everyday objects) that function as universal adversarial perturbations. Combines properties of adversarial patches and Trojan attacks

**Attack mechanism:**
Creates **naturalistic-looking patches** (e.g., images of flowers, everyday objects) that function as universal adversarial perturbations. Combines properties of adversarial patches and Trojan attacks - any image captured with a TnT patch in the scene will be misclassified.

Key properties:
- **Naturalistic appearance** - Looks like real objects, not noisy perturbations
- **Universal** - Works on any input image when TnT is in scene
- **Location-independent** - Effective regardless of patch position
- **Physically realizable** - Can be printed and deployed

**ONNX graph indicators:**
- **GAP_FC_HEAD** - Universal patch must affect all images' aggregated features
- **NO_SPATIAL_ATTENTION** - Model cannot learn to ignore naturalistic object
- **Transfer across architectures** - Tested on WideResNet50, Inception-V3, VGG-16

**Gadget and chain mapping:**
- Confirms: GAP_FC_HEAD, NO_SPATIAL_ATTENTION
- Also relates to: SKIP_CONNECTION (gradient flow for optimization)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget GAP_FC_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PATCH-ATTACK-SURFACE

**Related literature:**
1. **Trojan-patch hybrid** - Universal backdoor-like behavior via physical object
2. **Naturalistic stealth** - Evades human suspicion while maintaining attack efficacy
3. **Cross-architecture transfer** - Same patch works on multiple DNN architectures
4. **Real-world validation** - Tested on 50,000 ImageNet validation images

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [56] AdvCF - AdvCF (Arxiv 2022)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** ALIASING_DOWNSAMPLE (weather)

**Summary:** Sticker Confirms ALIASING_DOWNSAMPLE (weather).

**Attack mechanism:**
Confirms ALIASING_DOWNSAMPLE (weather).

**ONNX graph indicators:**
- See ALIASING_DOWNSAMPLE (weather) detection_logic.

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE (weather)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE (weather)`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---
### [57] SPAA - Light-based attack (VR 2022)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** SPAA manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [58] FakeWeather - FakeWeather (IJCNN 2022)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** ALIASING_DOWNSAMPLE (weather)

**Summary:** Sticker Confirms ALIASING_DOWNSAMPLE (weather).

**Attack mechanism:**
Confirms ALIASING_DOWNSAMPLE (weather).

**ONNX graph indicators:**
- See ALIASING_DOWNSAMPLE (weather) detection_logic.

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE (weather)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE (weather)`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---
### [59] AdvRain - AdvRain (Arxiv 2023)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** ALIASING_DOWNSAMPLE (weather)

**Summary:** Sticker Confirms ALIASING_DOWNSAMPLE (weather).

**Attack mechanism:**
Confirms ALIASING_DOWNSAMPLE (weather).

**ONNX graph indicators:**
- See ALIASING_DOWNSAMPLE (weather) detection_logic.

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE (weather)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE (weather)`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---
### [60] RP2 - Robust Physical Perturbations (CVPR 2018)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "Robust Physical-World Attacks on Deep Learning Visual Classification" - Eykholt et al.

**Attack Mechanism:**
- Physical adversarial perturbations on traffic signs (stop signs)
- Perturbations printed as stickers/graffiti on real signs
- Uses camera/distance variations during optimization (similar to EOT concept)
- Demonstrated consistent misclassification in drive-by tests

**Key Contributions:**
- First practical physical attack on safety-critical systems (autonomous vehicles)
- Introduced Robust Physical Perturbations (RP2) optimization method
- Showed attacks work across distances, angles, lighting conditions

**Attack landscape factors (graph-detectable):**
1. **Same aliasing vulnerabilities as EOT**
2. **Classifier sensitivity to localized perturbations**
3. **No redundancy/verification in prediction pipeline**

**DAG Indicators for Vulnerability:**
- [x] All EOT indicators apply
- [x] Single-model classification (no ensemble or verification)
- [x] High confidence on adversarial inputs (calibration issues)

**Attack-Specific Considerations:**
- Traffic sign classifiers are typically small, efficient models
- Limited depth may actually increase robustness to some attacks
- But limited capacity also means less redundancy

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [61] DARTS - DARTS (Arxiv 2018)

**Status:** analysis_complete
**Attack form:** Image
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Physical perturbation (modified signs and advertisements) Introduces two novel attacks:
1. **Sign Embedding Attack** - Generate adversarial examples from arbitrary points (not just existing signs), allowing modification of innocuous signs/ads to be misclassi

**Attack mechanism:**
Introduces two novel attacks:
1. **Sign Embedding Attack** - Generate adversarial examples from arbitrary points (not just existing signs), allowing modification of innocuous signs/ads to be misclassified as traffic signs
2. **Lenticular Printing Attack** - Exploits optical phenomena for viewpoint-dependent attacks

**ONNX graph indicators:**
- **Aliasing in downsampling** - Printed perturbations must survive capture
- **No frequency filtering** - High-frequency patterns propagate
- **Standard traffic sign classifier** - GTSRB-trained architectures

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE
- Also relates to: GAP_FC_HEAD (classification head)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Sign embedding** - Any visual can become adversarial traffic sign
2. **Lenticular printing** - Viewpoint-dependent physical attacks
3. **Expanded threat model** - Innocuous ads/logos as attack vectors
4. **Adversarial training bypass** - Sign embedding outperforms other attacks against defended models

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [62] RogueSigns - RogueSigns (Arxiv 2018)

**Status:** analysis_complete
**Attack form:** Printed images
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Physical perturbation (malicious advertisements and logos) Same sign embedding approach as DARTS - generates adversarial examples that make innocent-looking ads and logos get misclassified as traffic signs. Focuses on the expanded attack surface from arbitrar

**Attack mechanism:**
Same sign embedding approach as DARTS - generates adversarial examples that make innocent-looking ads and logos get misclassified as traffic signs. Focuses on the expanded attack surface from arbitrary image generation.

**ONNX graph indicators:**
- **ALIASING_DOWNSAMPLE** - Printed patterns alias into features
- **No input filtering** - Arbitrary images processed without frequency analysis

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Threat surface expansion** - Any printable surface becomes attack vector
2. **Commercial deployment risk** - Roadside advertisements could be weaponized

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [63] PS-GAN - PS-GAN (AAAI 2019)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Patch attack (GAN-generated) Uses a **generative adversarial network** to create adversarial patches that are both visually natural AND effective. Treats patch generation as patch-to-patch translation with:
1. **Visual fidelity**

**Attack mechanism:**
Uses a **generative adversarial network** to create adversarial patches that are both visually natural AND effective. Treats patch generation as patch-to-patch translation with:
1. **Visual fidelity** - High perceptual correlation with attacked image
2. **Attack enhancement** - Attention mechanism identifies critical attack regions

**ONNX graph indicators:**
- **Aliasing enables physical robustness** - GAN-generated patches survive printing
- **GAP_FC_HEAD** - Patches affect aggregated features
- **No spatial attention in target** - Cannot filter patch regions

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE (for physical deployment)
- Also relates to: GAP_FC_HEAD, NO_SPATIAL_ATTENTION
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PATCH-ATTACK-SURFACE, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Perceptual-sensitive generation** - Patches look natural, not noisy
2. **Attention-guided attack** - Focus perturbations on critical regions
3. **Patch-to-patch translation** - Novel framing of adversarial generation

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [64] AdvCam - AdvCam (CVPR 2020)

**Status:** analysis_complete
**Attack form:** Image
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Physical perturbation (style-transfer camouflage) Transfers large adversarial perturbations into **customized natural styles** (textures, patterns) that look legitimate to humans. Can be applied to target object or background. Combines style transfer

**Attack mechanism:**
Transfers large adversarial perturbations into **customized natural styles** (textures, patterns) that look legitimate to humans. Can be applied to target object or background. Combines style transfer with adversarial optimization for stealthy physical attacks.

**ONNX graph indicators:**
- **ALIASING_DOWNSAMPLE** - Style patterns must survive physical pipeline
- **No texture filtering** - Natural styles processed as legitimate content
- **Standard CNN** - Style features propagate through architecture

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE
- Also relates to: NORMALIZER (style changes affect normalization)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Natural style camouflage** - Adversarial perturbations look like art/design
2. **Large perturbations, high stealth** - Breaks tradeoff between visibility and effectiveness
3. **Flexible attack regions** - Can target object or background
4. **Privacy protection application** - Evade surveillance via camouflage

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [65] OPAD - Light-based attack (ICCV 2021)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** OPAD manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [66] Adversarial Shadow - Light-based attack (CVPR 2022)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** Adversarial Shadow manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [67] PhysGAN - PhysGAN (CVPR 2020)

**Status:** analysis_complete
**Attack form:** Image
**Registry:** MAXPOOL_AFTER_FUSION

**Summary:** GAN-generated physical adversarial examples Uses **GAN to generate physically realizable adversarial examples** that work continuously against autonomous driving systems. Unlike prior work that perturbs entire scenes (including sky), PhysGAN cr

**Attack mechanism:**
Uses **GAN to generate physically realizable adversarial examples** that work continuously against autonomous driving systems. Unlike prior work that perturbs entire scenes (including sky), PhysGAN creates perturbations that can be physically deployed.

**ONNX graph indicators:**
- **MAXPOOL_AFTER_FUSION** - Max pooling after feature fusion loses spatial detail
- **Multi-scale feature fusion** - Features from multiple levels combined
- **No adversarial training** - Model not robust to physical perturbations

**Gadget and chain mapping:**
- Confirms: MAXPOOL_AFTER_FUSION
- Also relates to: MULTI_SCALE_FUSION, ALIASING_DOWNSAMPLE
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget MAXPOOL_AFTER_FUSION`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK, CHAIN-DETECTOR-EVASION

**Related literature:**
1. **First physically realizable AV attack** - Not just digital perturbations
2. **GAN-based generation** - More realistic perturbations
3. **Continuous attack** - Works over time, not just single frames
4. **Constrained to physical deployment** - Only perturbs placeable regions

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [70] TPatch (Arxiv 2023)

**Status:** analysis_complete
**Attack form:** Acoustics
**Registry:** AUDIO_MEL_INPUT, AUDIO_STRIDE_DOWNSAMPLE, AUDIO_1D_CONV, CHAIN-AUDIO-ADVERSARIAL-SURFACE

**Summary:** TPatch extends patch attacks to acoustic sensing: localized perturbations on audio waveforms or spectrograms coupled to physical transducers. When the exported graph is audio-native, mel-input and strided audio frontend motifs index the attack landscape.

**Attack mechanism:**
Temporal patches optimize against models that ingest spectrogram-like tensors, analogous to spatial patches on images, with optional physical playback.

**ONNX graph indicators:**
- `AUDIO_MEL_INPUT` mel/log-mel stem
- `AUDIO_STRIDE_DOWNSAMPLE` or `AUDIO_1D_CONV` in frontend
- `CTC_DECODER_STRUCTURE` or seq2seq heads on ASR exports

**Gadget and chain mapping:**
- `CHAIN-AUDIO-ADVERSARIAL-SURFACE` when mel plus strided/1D conv motifs co-occur
- Cross-reference `Carlini-Audio-2018` via `catalog --gadget AUDIO_MEL_INPUT`

**What GraphSurgeon surfaces:**
`motifs` on audio ONNX models; `catalog --chain CHAIN-AUDIO-ADVERSARIAL-SURFACE`.

**Static analysis limits:**
Speaker hardware, room acoustics, and sampling rate outside the file are not modeled. Vision-only exports show no audio motifs.

---
### [71] DPATCH - Adversarial Patch on Object Detectors (AAAI 2019)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "DPatch: An Adversarial Patch Attack on Object Detectors" - Liu et al.

**Attack Mechanism:**
- First adversarial patch attack specifically targeting object detectors (YOLO, Faster R-CNN)
- Attacks both classification AND localization components
- Can cause: (1) missed detections, (2) false detections, (3) misclassification
- Patches demonstrate transferability across detector architectures

**Key Innovations:**
- Showed that object detectors have different vulnerabilities than classifiers
- Identified that NMS (Non-Maximum Suppression) can be exploited
- Demonstrated cross-model transferability

**Attack landscape factors (object detector):**
1. **Anchor-based Detection:** Fixed anchor boxes can be fooled
2. **NMS Exploitation:** Adversarial confidence scores manipulate which boxes survive NMS
3. **Multi-scale Feature Pyramids:** Attack can target specific scales
4. **Shared Backbone:** Attacking backbone affects all detection heads

**DAG Indicators for Vulnerability (Object Detectors):**
- [x] Anchor-based detection heads
- [x] NMS as post-processing (not in ONNX, but implied by architecture)
- [x] Shared backbone features for classification and regression
- [x] Feature Pyramid Networks (multiple scales to attack)

**Gadgets that Enable Object Detector Attacks:**
1. `Anchor_Detection_Head`: Fixed anchor boxes vulnerable to perturbation
2. `Shared_Backbone`: Single feature extractor for all tasks
3. `FPN_Multi_Scale`: Feature pyramids provide multiple attack surfaces
4. `Confidence_Based_NMS`: NMS filtering based on confidence scores

**Detection Logic:**
```
IF is_object_detector AND 
   (has_anchor_boxes OR has_shared_backbone):
   FLAG as OBJECT_DETECTOR_PATCH_VULNERABLE
   
IF has_FPN_like_structure:
   INCREASE vulnerability score (multiple attack surfaces)
```

**Hardening Recommendations:**
- Anchor-free detection architectures
- Separate backbones for classification/regression
- Patch detection in preprocessing
- Confidence calibration and verification

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [72] Dpatch2 - DPatch (ArXiv 2019)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** OBJECTNESS_HEAD

**Summary:** Patch attack (universal, transferable) First adversarial patch attack specifically designed for object detectors. Simultaneously attacks both **bounding box regression** and **object classification** by targeting the objectness scoring mec

**Attack mechanism:**
First adversarial patch attack specifically designed for object detectors. Simultaneously attacks both **bounding box regression** and **object classification** by targeting the objectness scoring mechanism. Small, location-independent patches that transfer between detector architectures.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Objectness scores computed before classification
- **Anchor-based detection** - Patches affect multiple anchor boxes
- **NMS (Non-Maximum Suppression)** - Suppressed boxes cascade failures

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: NMS_POSTPROCESS, ANCHOR_GRID
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **Dual-target attack** - Attacks both regression and classification simultaneously
2. **High transferability** - DPatch trained on Faster R-CNN attacks YOLO and vice versa
3. **Dramatic effectiveness** - Reduces mAP from 75%/65% to <1%

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [73] Object Hider - Hiding objects from detectors (2020)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** OBJECTNESS_HEAD, CHAIN-OBJECT-DISAPPEARANCE

**Summary:** Object Hider suppresses objectness scores so detectors fail to emit boxes for targeted objects while the patch is present.

**Attack mechanism:**
Optimizes a patch to minimize objectness/confidence on detector heads, causing missed detections rather than mislabeling.

**ONNX graph indicators:**
- Sigmoid or confidence head after Conv feature maps
- Shared backbone feeding classification and objectness branches
- Anchor or grid-based detection heads (YOLO-style)

**Gadget and chain mapping:**
OBJECTNESS_HEAD; CHAIN-OBJECT-DISAPPEARANCE when objectness path is present without robust gating.

**What GraphSurgeon surfaces:**
`catalog --gadget OBJECTNESS_HEAD`, `motifs` object-detector scan.

**Static analysis limits:**
NMS and post-processing are often outside ONNX; graph shows head structure only.

---
### [74] LPAttack - LPAttack (License Plate Attack) (AAAI 2020)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** OBJECTNESS_HEAD

**Summary:** Perturbation/patch attack on license plate detection Creates adversarial perturbations that appear as natural "spots" (sludge, dirt) on license plates. Uses genetic algorithms to optimize perturbation positions. Attacks the detection stage to prevent pl

**Attack mechanism:**
Creates adversarial perturbations that appear as natural "spots" (sludge, dirt) on license plates. Uses genetic algorithms to optimize perturbation positions. Attacks the detection stage to prevent plate localization.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Plate detection uses objectness scores
- **Character recognition pipeline** - Detection → Segmentation → OCR
- **Fixed aspect ratio assumption** - Detectors expect standard plate dimensions

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD (in detection stage)
- Also relates to: SEQUENTIAL_PIPELINE (detection → recognition)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **Natural appearance constraint** - Perturbations look like dirt/damage
2. **Genetic algorithm optimization** - Efficient perturbation placement
3. **Legal evasion use case** - Real-world attack motivation

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [75] SwitchPatch - SwitchPatch (ArXiv 2022)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** OBJECTNESS_HEAD

**Summary:** Patch attack (label-switching, universal) Uses a **tailored projection function** to place adversarial patches on multiple target objects regardless of distance or viewing angle. Unique loss function designed to change object labels (e.g., ca

**Attack mechanism:**
Uses a **tailored projection function** to place adversarial patches on multiple target objects regardless of distance or viewing angle. Unique loss function designed to change object labels (e.g., car → bus) rather than just suppress detection.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Objectness scores for detection
- **Classification head** - Class prediction after objectness
- **Multi-scale detection** - Attacks work across detection scales

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: MULTI_SCALE_DETECTION
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **Label switching** - Misdirection rather than evasion
2. **Distance/angle invariance** - Robust to viewing conditions
3. **Physical-world transfer** - Verified in real conditions

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [76] Extended RP2 - Extended RP2 (USENIX 2018)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** ALIASING_DOWNSAMPLE

**Summary:** Physical perturbation (extended study) Extended version of RP2 (Paper 37) with:
- Additional experiments and analysis
- Expanded threat model discussion
- More comprehensive field testing
- Object detection extension

**Attack mechanism:**
Extended version of RP2 (Paper 37) with:
- Additional experiments and analysis
- Expanded threat model discussion
- More comprehensive field testing
- Object detection extension

**ONNX graph indicators:**
- **ALIASING_DOWNSAMPLE** - Core enabler
- **NORMALIZER** - Distribution shift exploitation
- **No frequency filtering**

**Gadget and chain mapping:**
- Confirms: ALIASING_DOWNSAMPLE (extended validation)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget ALIASING_DOWNSAMPLE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [77] ShapeShifter (ECML PKDD 2018)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN Object Detector" - Chen et al.

**Attack Mechanism:**
- Physical adversarial perturbations that survive EOT transformations
- Targets Faster R-CNN (two-stage detector)
- Can cause: (1) Disappearance (2) Misclassification (3) Appearance of phantom objects

**Two-Stage Detector Vulnerabilities:**
1. **Region Proposal Network (RPN)**: First stage proposes regions
2. **Classification Head**: Second stage classifies proposals
3. Attacking RPN = object disappears entirely
4. Attacking classification = wrong label

**DAG Indicators:**
- [x] RPN-like structure (objectness + box regression)
- [x] ROI Pooling/Align for two-stage processing
- [x] Separate proposal and classification stages

**Gadgets:**
1. `RPN_PROPOSAL`: Region proposal network structure
2. `ROI_PROCESSING`: ROI pooling/align operations
3. `TWO_STAGE_DETECTION`: Separate proposal and classification

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [78] NestedAE - Daedalus (NMS Attack) (CCS 2019)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** NMS_DEPENDENCY

**Summary:** NMS-targeted adversarial attack **Breaks Non-Maximum Suppression** by compressing detection bounding box dimensions to evade NMS filtering. This causes NMS to malfunction, resulting in extremely dense false positives rather than mis

**Attack mechanism:**
**Breaks Non-Maximum Suppression** by compressing detection bounding box dimensions to evade NMS filtering. This causes NMS to malfunction, resulting in extremely dense false positives rather than missed detections.

**ONNX graph indicators:**
- **NMS_POSTPROCESS** - Non-Maximum Suppression as post-processing step
- **Box regression output** - Detector predicts bounding box coordinates
- **IoU-based filtering** - NMS uses IoU threshold for suppression

**Gadget and chain mapping:**
- Confirms: NMS_DEPENDENCY (renamed from NMS_POSTPROCESS for clarity)
- Also relates to: OBJECTNESS_HEAD
- New gadget needed: No (refines existing NMS understanding)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget NMS_DEPENDENCY`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **NMS exploitation** - Attacks post-processing rather than detection head
2. **Dense false positives** - Different failure mode from suppression
3. **Box dimension compression** - Novel perturbation target
4. **Ensemble transferability** - Black-box attacks via substitute models

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [79] Translucent Patch - Translucent Patch (CVPR 2021)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** NMS_DEPENDENCY

**Summary:** Camera lens translucent patch Places a **translucent patch on camera lens** rather than target object. Creates a universal attack that hides ALL instances of a target class while minimizing impact on other classes. Contactless att

**Attack mechanism:**
Places a **translucent patch on camera lens** rather than target object. Creates a universal attack that hides ALL instances of a target class while minimizing impact on other classes. Contactless attack requiring no access to targets.

**ONNX graph indicators:**
- **NMS_DEPENDENCY** - Attack affects NMS input for target class
- **Class-specific features** - Features separable by class
- **Lens-to-detector pipeline** - No filtering of lens artifacts

**Gadget and chain mapping:**
- Confirms: NMS_DEPENDENCY (lens patch affects NMS filtering)
- Also relates to: SHARED_BACKBONE, NO_INPUT_VALIDATION
- New gadget candidate: LENS_ARTIFACT_SENSITIVITY

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget NMS_DEPENDENCY`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Contactless attack** - No access to target objects needed
2. **Universal class hiding** - All instances of target class hidden
3. **Class-selective** - Minimal impact on non-target classes
4. **Printable implementation** - Practical physical deployment

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [80] SLAP - Light-based attack (USENIX 2021)

**Status:** analysis_complete
**Attack form:** Light based
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** SLAP manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [81] Adversarial Rain - Light-based attack (Arxiv 2022)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** Adversarial Rain manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [82] AdvRD - Light-based attack (Arxiv 2023)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** ALIASING_DOWNSAMPLE, NORMALIZER

**Summary:** AdvRD manipulates lighting, projection, or weather appearance to cause misclassification without a classic additive patch.

**Attack mechanism:**
Structured illumination or occlusion changes local contrast and frequency content. Under varying capture conditions the perturbation survives when the network lacks anti-aliasing and uses fixed BatchNorm statistics.

**ONNX graph indicators:**
- Stride-2 Conv/Pool without blur (ALIASING_DOWNSAMPLE)
- BatchNormalization nodes using running stats (NORMALIZER)
- GlobalAveragePool before Gemm/MatMul (GAP_FC_HEAD) when classifier-style
- Absence of spatial attention blocks before pooling

**Gadget and chain mapping:**
Maps to ALIASING_DOWNSAMPLE, NORMALIZER. Contributes to CHAIN-PHYSICAL-WORLD-ATTACK when aliasing present; patch-surface chains when GAP head present.

**What GraphSurgeon surfaces:**
`catalog --gadget` for ALIASING_DOWNSAMPLE; see shared section 'Light-based attacks' above.

**Static analysis limits:**
Illumination preprocessing (CLAHE, HDR) is often outside ONNX. Graph scan cannot confirm physical lighting setup.

---
### [83] Invisible Cloak - Invisible Cloak (2018) (UEMCON 2018)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** HIGH_FANIN_FUSION

**Summary:** Physical adversarial sticker/cloak Early work on **adversarial stickers and cloaks** for evading person detection. Tests three modalities: digital adversarial examples, stickers as watermarks on photos, and physical display on screens

**Attack mechanism:**
Early work on **adversarial stickers and cloaks** for evading person detection. Tests three modalities: digital adversarial examples, stickers as watermarks on photos, and physical display on screens in front of person.

**ONNX graph indicators:**
- **HIGH_FANIN_FUSION** - Many-to-one aggregation of features
- **OBJECTNESS_HEAD** - Person detection scoring
- **Grid-based detection** - YOLO's grid structure

**Gadget and chain mapping:**
- Confirms: HIGH_FANIN_FUSION
- Also relates to: OBJECTNESS_HEAD, GAP_FC_HEAD
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget HIGH_FANIN_FUSION`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Foundational clothing attack** - Precursor to Adv T-shirt work
2. **Multi-modality testing** - Digital, watermark, screen display
3. **Screen-based physical attack** - Novel delivery mechanism

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [84] Adversarial YOLO (CVPRW 2019)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "Fooling automated surveillance cameras: adversarial patches to attack person detection" - Thys et al.

**Attack Mechanism:**
- Creates adversarial patches that when worn/held, cause YOLO to fail to detect the person
- Attacks the OBJECTNESS score, not just classification
- Patch suppresses detection entirely (person becomes "invisible")
- Works in physical world with printed patches

**Key Difference from Classification Attacks:**
- Classification attacks: Cause WRONG label
- Detection attacks: Cause NO detection (objectness → 0)

**Attack landscape factors (graph-detectable):**
1. **Single objectness output**: One confidence score determines detection
2. **Grid-based detection**: YOLO's grid can be targeted per-cell
3. **Anchor boxes**: Fixed anchors create predictable attack targets
4. **NMS vulnerability**: Suppressing one box can remove overlapping detections

**DAG Indicators:**
- [x] Sigmoid activation for objectness (creates exploitable gradient)
- [x] Concat of multi-scale features before detection head
- [x] Reshape operations for grid-based output
- [x] Single detection head (no redundancy)

**Gadgets for Detection Attacks:**
1. `OBJECTNESS_HEAD`: Sigmoid-based objectness prediction
2. `GRID_DETECTION`: Reshape to grid-based output
3. `SINGLE_DETECTION_HEAD`: No redundant detection paths

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [85] UPC - Universal Physical Camouflage (CVPR 2020)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "Universal Physical Camouflage Attacks on Object Detectors" - Huang et al.

**Attack Mechanism:**
- Creates WEARABLE adversarial patterns (clothing, textures)
- Universal: Works across different people, poses, backgrounds
- Camouflages entire person (not just patch on body)
- Uses differentiable renderer for physical-world robustness

**Key Innovation:**
- Full-body texture attack (not just localized patch)
- Must work across body deformations (walking, sitting, etc.)

**Vulnerability Factors:**
1. **Texture-based features**: Detectors rely on appearance
2. **No shape-only fallback**: Silhouette alone doesn't trigger detection
3. **Training bias**: Detectors trained on normal clothing textures

**DAG Indicators:**
- [x] Heavy reliance on early Conv features (texture extraction)
- [x] No explicit shape/contour processing
- [x] Standard backbone without robustness features

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [86] Adversarial T-shirt - Adversarial T-Shirt (ECCV 2020)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** OBJECTNESS_HEAD

**Summary:** Wearable adversarial clothing (T-shirt) **First work to model non-rigid deformation** in physical adversarial examples. Uses **Thin Plate Spline (TPS) mapping** to simulate realistic cloth deformations and wrinkles during human movement. Pa

**Attack mechanism:**
**First work to model non-rigid deformation** in physical adversarial examples. Uses **Thin Plate Spline (TPS) mapping** to simulate realistic cloth deformations and wrinkles during human movement. Patterns printed on actual T-shirts.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Person detection objectness scores
- **GAP_FC_HEAD** - Feature aggregation for detection
- **NO_DEFORMATION_MODELING** - Detector doesn't account for clothing deformation

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: GAP_FC_HEAD, ALIASING_DOWNSAMPLE (for physical robustness)
- New gadget candidate: NO_DEFORMATION_MODELING (detector limitation)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **TPS deformation modeling** - First to handle cloth movement
2. **Ensemble detector attack** - Min-max optimization for multiple detectors
3. **Printable adversarial clothing** - Real wearable items

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [87] Invisible Cloak2 - Invisibility Cloak (ECCV 2020)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** OBJECTNESS_HEAD

**Summary:** Wearable adversarial clothing (cloak/poster) Systematic study of physical adversarial attacks on object detectors using printed posters and wearable clothes. Quantifies attack effectiveness across **white-box and black-box settings** while measu

**Attack mechanism:**
Systematic study of physical adversarial attacks on object detectors using printed posters and wearable clothes. Quantifies attack effectiveness across **white-box and black-box settings** while measuring **transferability** between datasets, object classes, and detector models.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Core attack target
- **Feature backbone** - Shared features across classes
- **Anchor-based detection** - Multiple detection scales

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: SHARED_BACKBONE, MULTI_SCALE_DETECTION
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Systematic transferability study** - Quantifies cross-detector attack
2. **Cross-class attacks** - Same patch attacks different object categories
3. **White-box vs black-box comparison** - Performance degradation analysis

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [88] NAP - NAP (Naturalistic Adversarial Patch) (ICCV 2021)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** OBJECTNESS_HEAD

**Summary:** GAN-generated naturalistic patch Leverages pretrained GANs (BigGAN, StyleGAN) to generate adversarial patches that appear **natural and realistic** while maintaining attack effectiveness. Samples from learned image manifold rather th

**Attack mechanism:**
Leverages pretrained GANs (BigGAN, StyleGAN) to generate adversarial patches that appear **natural and realistic** while maintaining attack effectiveness. Samples from learned image manifold rather than generating noise-like patterns.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Objectness score suppression
- **High-level feature sensitivity** - GAN images affect detection features
- **No texture verification** - Detector accepts any visual input

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: NO_INPUT_VALIDATION
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **GAN manifold sampling** - Natural images as adversarial patches
2. **Human perception validation** - Subjective surveys confirm stealth
3. **Naturalness without effectiveness loss** - Breaks expected tradeoff

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [89] LAP - LAP (Legitimate Adversarial Patch) (ACM MM 2021)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** OBJECTNESS_HEAD

**Summary:** LAP (2021) adversarial attack.

**Attack mechanism:**
(see BATCH harvest)

**ONNX graph indicators:**
- See gadget detection_logic in registry for op-level patterns

**Gadget and chain mapping:**
Confirms OBJECTNESS_HEAD.

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [90] AdvTexture - AdvTexture (CVPR 2022)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** OBJECTNESS_HEAD

**Summary:** Adversarial texture on clothing (T-shirts, skirts, dresses) **Toroidal-Cropping-based Expandable Generative Attack (TC-EGA)** creates adversarial textures with **repetitive structures** that cover clothing of arbitrary shapes. Enables **multi-angle attacks** -

**Attack mechanism:**
**Toroidal-Cropping-based Expandable Generative Attack (TC-EGA)** creates adversarial textures with **repetitive structures** that cover clothing of arbitrary shapes. Enables **multi-angle attacks** - evasion from different camera perspectives.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Person detection suppression
- **View-angle sensitivity** - Detectors trained on frontal views
- **Texture processing** - Model processes texture as features

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: NO_VIEW_INVARIANCE (implicit detector limitation)
- New gadget candidate: VIEW_ANGLE_SENSITIVITY

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Toroidal cropping** - Seamless tiling for any garment shape
2. **Multi-angle effectiveness** - Not limited to frontal attacks
3. **Actual wearable fabrication** - Printed on real fabric

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [91] AdvART - AdvART (ArXiv 2023)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** OBJECTNESS_HEAD

**Summary:** Art-styled adversarial patch Generates adversarial patches that appear as **artistic paintings** using a **similarity loss term** as semantic constraint. Directly manipulates pixel values (not GAN-based) for greater optimization

**Attack mechanism:**
Generates adversarial patches that appear as **artistic paintings** using a **similarity loss term** as semantic constraint. Directly manipulates pixel values (not GAN-based) for greater optimization flexibility while maintaining artistic appearance.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Detection score suppression
- **High-level feature extraction** - Art patterns affect features
- **No semantic verification** - Accepts art as valid input

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: NO_INPUT_VALIDATION
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **Art as adversarial camouflage** - Semantic meaning provides stealth
2. **Direct pixel optimization** - Avoids GAN limitations
3. **Edge deployment** - Tested on smart cameras

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [92] Patch of Invisibility - Patch of Invisibility (BBNP) (ArXiv 2023)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** OBJECTNESS_HEAD

**Summary:** Black-box naturalistic patch **BBNP (Black-Box Naturalistic Patch)** algorithm generates naturalistic adversarial patches without requiring model access. Uses pretrained GAN to create natural-looking patches through black-box opt

**Attack mechanism:**
**BBNP (Black-Box Naturalistic Patch)** algorithm generates naturalistic adversarial patches without requiring model access. Uses pretrained GAN to create natural-looking patches through black-box optimization.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Detection score as optimization target
- **Query access only** - No gradient information needed
- **Transferability** - Black-box implies cross-model effectiveness

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: HIGH_TRANSFERABILITY (implicit property)
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION

**Related literature:**
1. **Black-box optimization** - No model access required
2. **Query-based attack** - Uses detection scores as feedback
3. **State-of-the-art black-box performance** - Competitive with white-box

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [93] DAP - DAP (Dynamic Adversarial Patch) (ArXiv 2023)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** OBJECTNESS_HEAD

**Summary:** Dynamic adversarial patch for clothing Dynamic adversarial patch for clothing: directly modifies pixel values (not GAN-based) to handle non-rigid deformations from pose changes. Addresses clothing-specific challenges for physical attacks o

**Attack mechanism:**
Dynamic adversarial patch for clothing: directly modifies pixel values (not GAN-based) to handle non-rigid deformations from pose changes. Addresses clothing-specific challenges for physical attacks on person detectors.

**ONNX graph indicators:**
- **OBJECTNESS_HEAD** - Person detection suppression
- **No pose-invariant features** - Detection affected by pose changes
- **Texture sensitivity** - Clothing texture affects detection

**Gadget and chain mapping:**
- Confirms: OBJECTNESS_HEAD
- Also relates to: NO_DEFORMATION_MODELING
- New gadget needed: No

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget OBJECTNESS_HEAD`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-DETECTOR-EVASION, CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Direct pixel optimization** - More flexible than GAN-based patches
2. **Deformation-aware design** - Addresses pose/clothing changes for physical wearability
3. **Person-detector focus** - Targets YOLOv3/YOLOv7 objectness heads

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [94] Adversarial Bulbs (AAAI 2021)

**Status:** analysis_complete
**Attack form:** Bulb
**Registry:** SINGLE_MODALITY_INPUT, GAP_FC_HEAD, CHAIN-SINGLE-MODALITY-VISION

**Summary:** Adversarial bulbs modulate thermal emission so a visible-RGB-trained classifier misbehaves under a thermal camera. The ONNX graph remains a standard single-input vision model; deployment mismatch is the attack channel.

**Attack mechanism:**
Controlled heating patterns act as physical perturbations in the thermal domain while the network ingests a single image tensor.

**ONNX graph indicators:**
- One `graph.input`, no early multimodal fusion (`SINGLE_MODALITY_INPUT`)
- `GAP_FC_HEAD` or detector heads on classifiers

**Gadget and chain mapping:**
- `CHAIN-SINGLE-MODALITY-VISION` links thermal literature to visible-trained DAGs
- See deployment appendix for hardware vs graph distinction

**What GraphSurgeon surfaces:**
`catalog --gadget SINGLE_MODALITY_INPUT`; `motifs` on RobustBench-style classifiers.

**Static analysis limits:**
LED/thermal hardware is not in ONNX. Graph motifs describe attack landscape if a thermal sensor feeds this model, not bulb firmware.

---
### [95] QRAttack (CVPR 2022)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** SINGLE_MODALITY_INPUT, GAP_FC_HEAD, CHAIN-SINGLE-MODALITY-VISION

**Summary:** QRAttack embeds thermal patterns in clothing for cross-domain evasion against visible-trained networks viewed through thermal imagers.

**Attack mechanism:**
Garment heat patterns mimic adversarial structure in the thermal spectrum while the exported graph is unchanged RGB/vision topology.

**ONNX graph indicators:**
- `SINGLE_MODALITY_INPUT`, `GAP_FC_HEAD`, optional `ALIASING_DOWNSAMPLE`

**Gadget and chain mapping:**
- `CHAIN-SINGLE-MODALITY-VISION` when single input plus classifier head motifs appear

**What GraphSurgeon surfaces:**
Same deployment-context chain as papers 94-98 on standard vision ONNX.

**Static analysis limits:**
Fabric and thermal camera physics are outside the DAG.

---
### [96] HOTCOLD (ArXiv 2022)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** SINGLE_MODALITY_INPUT, CHAIN-SINGLE-MODALITY-VISION

**Summary:** HOTCOLD crafts hot/cold regions on clothing to fool thermal-facing models trained on visible statistics.

**Attack mechanism:**
Spatial thermal contrast on apparel transfers to the sensor stream; the classifier graph is still single-modality vision.

**ONNX graph indicators:**
- `SINGLE_MODALITY_INPUT` without `HAS_MULTIMODAL_FUSION`

**Gadget and chain mapping:**
- `CHAIN-SINGLE-MODALITY-VISION` with `GAP_FC_HEAD` or `OBJECTNESS_HEAD`

**What GraphSurgeon surfaces:**
`catalog --chain CHAIN-SINGLE-MODALITY-VISION`.

**Static analysis limits:**
No thermal sensor model inside ONNX.

---
### [97] AIP (ArXiv 2023)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** SINGLE_MODALITY_INPUT, CHAIN-SINGLE-MODALITY-VISION

**Summary:** AIP optimizes adversarial infrared patterns on clothing against visible-pretrained detectors under thermal capture.

**Attack mechanism:**
Infrared-optimized textures exploit domain gap between training (visible) and deployment (thermal).

**ONNX graph indicators:**
- Standard CNN/ViT single-input classifiers or detectors

**Gadget and chain mapping:**
- `SINGLE_MODALITY_INPUT` plus `CHAIN-SINGLE-MODALITY-VISION`

**What GraphSurgeon surfaces:**
Deployment-context motifs on vision exports used with thermal cameras.

**Static analysis limits:**
Attack success depends on sensor pairing, not graph depth alone.

---
### [98] AdvIB (ArXiv 2023)

**Status:** analysis_complete
**Attack form:** Clothing
**Registry:** SINGLE_MODALITY_INPUT, GAP_FC_HEAD, CHAIN-SINGLE-MODALITY-VISION

**Summary:** AdvIB studies adversarial infrared blocks on garments against visible-trained recognition under thermal imaging.

**Attack mechanism:**
Localized IR blocks raise thermal saliency regions that shift pooled features in `GAP_FC_HEAD`-style heads.

**ONNX graph indicators:**
- `GAP_FC_HEAD`, `SINGLE_MODALITY_INPUT`

**Gadget and chain mapping:**
- `CHAIN-SINGLE-MODALITY-VISION`

**What GraphSurgeon surfaces:**
`catalog --gadget SINGLE_MODALITY_INPUT` includes papers 94-98 in research basis.

**Static analysis limits:**
Thermal emitter hardware not represented in graph ops.

---
### [99] CAMOU (ICLR 2019)

**Status:** analysis_complete
**Attack form:** (see taxonomy)
**Registry:** (see taxonomy)

**Paper:** "CAMOU: 3D Adversarial Attack on Vehicle Detector" - Zhang et al.

**Attack Mechanism:**
- 3D rendering of adversarial textures on vehicles
- Makes cars invisible to detectors from multiple viewpoints
- Uses neural renderer for end-to-end optimization
- Physical-world viable (can be printed as vehicle wrap)

**Vehicle Detection Vulnerabilities:**
- Same as person detection + 3D viewpoint concerns
- Vehicles have rigid 3D structure (easier to model than humans)

**DAG Indicators:**
- [x] Same as person detection attacks
- [x] Multi-scale feature pyramids (FPN) provide multiple attack surfaces

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---

### [100] ER Attack - ER Attack (Eye Region Attack) (ArXiv 2020)

**Status:** analysis_complete
**Attack form:** Sticker
**Registry:** SHARED_BACKBONE

**Summary:** Physical adversarial sticker on face (eye region) Creates adversarial stickers targeting the **eye region** of faces. Uses a sticker generator and converter to create stickers with different shapes. Attacks both **dodging** (evade recognition) and **

**Attack mechanism:**
Creates adversarial stickers targeting the **eye region** of faces. Uses a sticker generator and converter to create stickers with different shapes. Attacks both **dodging** (evade recognition) and **impersonation** (be recognized as another person) scenarios.

**ONNX graph indicators:**
- **SHARED_BACKBONE** - Same feature extractor for all identities
- **Embedding space** - Face embeddings computed by shared network
- **No region-specific robustness** - Eye region disproportionately affects features

**Gadget and chain mapping:**
- Confirms: SHARED_BACKBONE
- Also relates to: HIGH_FANIN_FUSION (features from multiple regions aggregated)
- New gadget candidate: REGION_SENSITIVITY (eye region dominates embedding)

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget SHARED_BACKBONE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-FACE-RECOGNITION-ATTACK

**Related literature:**
1. **Eye region targeting** - Most influential region for recognition
2. **Multiple sticker shapes** - Glasses, patches, decorative stickers
3. **Dual attack modes** - Dodging and impersonation

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [101] ScreenAttack - ScreenAttack (Screen-Based Adversarial Attack) (ArXiv 2020)

**Status:** analysis_complete
**Attack form:** Patch
**Registry:** SHARED_BACKBONE

**Summary:** Screen display adversarial attack **CDAE (Color Decomposition-based Adversarial Examples)** addresses the challenge that adversarial examples lose effectiveness when displayed on screens and captured by cameras. Decomposes screen fram

**Attack mechanism:**
**CDAE (Color Decomposition-based Adversarial Examples)** addresses the challenge that adversarial examples lose effectiveness when displayed on screens and captured by cameras. Decomposes screen frames into symmetric adversarial frames while maintaining visual quality perceived by humans.

**ONNX graph indicators:**
- **SHARED_BACKBONE** - Feature extraction processes screen content same as direct input
- **No screen-capture filtering** - Model doesn't detect screen artifacts
- **Color space sensitivity** - RGB processing without screen compensation

**Gadget and chain mapping:**
- Confirms: SHARED_BACKBONE (same processing for all visual input)
- Also relates to: NO_INPUT_VALIDATION
- New gadget candidate: SCREEN_CAPTURE_SURFACE

**What GraphSurgeon surfaces:**
Run `graph-surgeon catalog --gadget SHARED_BACKBONE`; `patterns` and `motifs` flag matching structural motifs on the ONNX DAG.

**Related chains:** CHAIN-PHYSICAL-WORLD-ATTACK

**Related literature:**
1. **Screen-robust perturbations** - Survive display/capture pipeline
2. **Color decomposition** - Exploits screen color rendering
3. **Under-screen camera attacks** - Imperceptible screen modifications

**Static analysis limits:**
Graph-only analysis identifies applicable attack landscape, not whether this model was trained with defenses or physical-world robustness.

---

### [102] PG (Poltergeist) (S&P 2021)

**Status:** analysis_complete
**Attack form:** Acoustics
**Registry:** AUDIO_MEL_INPUT, ENCODER_DECODER_SEQ2SEQ, SPECIAL_TOKEN_CONTROL_FLOW, CTC_DECODER_STRUCTURE, CHAIN-ACOUSTIC-COMMAND-SURFACE

**Summary:** Poltergeist crafts inaudible or hidden acoustic commands that ASR/voice interfaces transcribe as attacker-chosen text. Audio-native ONNX graphs expose mel stems, seq2seq decoders, and special-token control motifs.

**Attack mechanism:**
Perturbations in the acoustic channel drive encoder states so the decoder emits command tokens or phrases without user awareness.

**ONNX graph indicators:**
- `AUDIO_MEL_INPUT` frontend
- `ENCODER_DECODER_SEQ2SEQ` or Whisper-like cross-attention
- `SPECIAL_TOKEN_CONTROL_FLOW` or `CTC_DECODER_STRUCTURE` at output

**Gadget and chain mapping:**
- `CHAIN-ACOUSTIC-COMMAND-SURFACE` when mel/seq2seq pairs with CTC or special-token motifs

**What GraphSurgeon surfaces:**
`catalog --chain CHAIN-ACOUSTIC-COMMAND-SURFACE`; audio `motifs` on ASR exports.

**Static analysis limits:**
Microphone placement, playback hardware, and psychoacoustic masking outside the graph are not analyzed.

---
### [103] A Survey of Self-Supervised Learning for Vision Transformers (2024)

**arXiv:** 2408.17059  
**Authors:** Khan et al.

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings for Security Analysis:**

1. **ViT-Specific Vulnerabilities:**
   - Self-attention layers without regularization are exploitable
   - Lack of inductive bias (vs CNNs) makes ViTs sensitive to adversarial perturbations
   - Patch embedding layer is a single point of entry for attacks

2. **Attention Exploits:**
   - Unregularized self-attention can be hijacked to focus on adversarial regions
   - SSL pre-training may not improve adversarial robustness
   - Attention maps can be manipulated by small perturbations

3. **DAG Indicators (NEW for ViTs):**
   - `PATCH_EMBEDDING`: Conv with large stride matching patch size (e.g., 16x16 stride 16)
   - `UNREGULARIZED_ATTENTION`: Self-attention without dropout or attention regularization
   - `NO_CLS_TOKEN_PROTECTION`: CLS token aggregates all patches without gating
   - `LAYERNORM_BEFORE_ATTENTION`: Pre-norm ViTs may have different vulnerabilities

**Proposed New Gadgets:**
| Gadget | Description | Severity |
|--------|-------------|----------|
| VIT_PATCH_EMBEDDING | Single linear projection for patches | HIGH |
| UNREGULARIZED_SELF_ATTENTION | Self-attention without regularization | HIGH |
| CLS_TOKEN_AGGREGATION | CLS token without spatial filtering | MEDIUM |

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [104] Identity Card Presentation Attack Detection: SLR (2025)

**arXiv:** 2511.06056

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Document/Forgery Detection Vulnerabilities:**
   - Multi-scale fusion in detectors is vulnerable to patch-like forgeries
   - Small local forgeries can evade detection if not fused properly
   - Anchor-based detectors miss fine-grained manipulations

2. **Relevance to Our Tool:**
   - Validates our MULTISCALE_FUSION gadget
   - Confirms ANCHOR_BASED_DETECTION vulnerabilities
   - Suggests need for FINE_GRAINED_DETECTION gadget

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [105] Security through Eyes of AI: Vision-Based Malware Detection (2025)

**arXiv:** 2505.07574

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Binary-to-Image Vulnerability Patterns:**
   - MaxPool amplification is critical vulnerability in malware classifiers
   - Aliasing from aggressive downsampling creates exploitable patterns
   - Adversarial perturbations in binary visualization transfer to misclassification

2. **Confirms Existing Gadgets:**
   - MAXPOOL_AMPLIFIER: Validated in new domain
   - ALIASING_DOWNSAMPLE: Critical for binary image classifiers
   - HIGH_FANIN_FUSION: Multiple binary sections create multi-path attacks

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [106] Adversarial Examples in Automated Driving Perception (2025)

**arXiv:** 2504.08414

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Physical-World Detector Vulnerabilities:**
   - Patch attacks on CNN detectors remain highly effective in 2025
   - Aliasing is critical factor for physical-world robustness
   - Multi-sensor fusion does NOT solve adversarial vulnerabilities

2. **Driving-Specific Patterns:**
   - Camera-based detectors are most vulnerable
   - LiDAR fusion helps but introduces new attack surfaces
   - Small-object detection especially vulnerable to aliasing

3. **DAG Implications:**
   - Validates our ALIASING_DOWNSAMPLE gadget
   - Confirms DPATCH-style attacks still effective
   - Suggests SENSOR_FUSION gadget for multi-modal models

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [107] FDM-YOLO: Small Target Detection (2025)

**arXiv:** 2503.04452

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Small Object Vulnerability:**
   - Early stride-2 convolutions destroy small object information
   - Aggressive early downsampling makes small targets invisible
   - Anti-aliasing significantly improves small-object robustness

2. **Adversarial Implications:**
   - Small adversarial patches are more effective when aliasing is present
   - Early downsampling = smaller patches needed for attack
   - Physical attacks on small objects (distant signs) are easier

3. **DAG Indicators:**
   - AGGRESSIVE_EARLY_DOWNSAMPLING: Multiple stride-2 ops in first 5 layers
   - Confirms ALIASING_DOWNSAMPLE severity for object detection

**Proposed Update:**
- Increase severity of ALIASING_DOWNSAMPLE when detected in object detectors

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [108] Anti-Aliasing Deep Image Classifiers (2021)

**arXiv:** 2110.00899

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Depth-Adaptive Anti-Aliasing:**
   - Different network depths require different blur kernel sizes
   - Early layers need stronger anti-aliasing
   - Learnable anti-aliasing is more effective than fixed BlurPool

2. **Defense Effectiveness:**
   - Anti-aliasing reduces adversarial transferability by 15-25%
   - Improves robustness to frequency-based attacks
   - Does NOT fully solve adversarial vulnerability

3. **Hardening Recommendations:**
   - Update mitigation for ALIASING_DOWNSAMPLE to include depth-adaptive option
   - Note: Anti-aliasing is necessary but not sufficient defense

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [109] Gradient-Free Adversarial Purification with Diffusion Models (2025)

**arXiv:** 2501.13336

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Anti-Aliasing as Zero-Shot Defense:**
   - Non-square filters can provide anti-aliasing in frequency domain
   - Diffusion purification inherently applies smoothing
   - Frequency-domain defenses are effective against many attacks

2. **Frequency Attack Insights:**
   - High-frequency perturbations are most common attack pattern
   - Aliasing allows high-freq attacks to persist through network
   - Low-pass filtering before inference is a viable defense

3. **Defense Detection:**
   - Suggests adding HAS_FREQUENCY_DEFENSE gadget (defensive indicator)
   - Look for: Blur operations, diffusion preprocessing, low-pass filters

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [110] Open Medical Imaging Benchmarks for OOD Detection (2025)

**arXiv:** 2503.16247

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **CNN vs ViT OOD Vulnerability:**
   - CNNs more vulnerable to frequency-based distribution shifts
   - ViTs more vulnerable to patch-based OOD
   - Both architectures have complementary weaknesses

2. **Aliasing Metrics:**
   - Provides empirical aliasing vulnerability metrics
   - Correlates aliasing with OOD detection failure
   - Suggests aliasing score as vulnerability metric

3. **Cross-Architecture Insights:**
   - Need to differentiate CNN vs ViT vulnerability patterns
   - Combined CNN+ViT ensembles may be more robust

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [111] DUMB and DUMBer: Is Adversarial Training Worth It? (2025)

**arXiv:** 2506.18516

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Multi-Branch Robustness:**
   - Adversarial training is less effective for multi-branch architectures
   - HIGH_FANIN_CONCAT models remain vulnerable even after AT
   - Skip connections create "gradient highways" that bypass AT

2. **Architecture-Specific AT Effectiveness:**
   - Simple architectures (ResNet) benefit most from AT
   - Complex fusion architectures (Inception) benefit less
   - ViTs show mixed results

3. **DAG Implications:**
   - Validates HIGH_FANIN_CONCAT as persistent vulnerability
   - Confirms SKIP_CONNECTION gradient highway concern
   - Suggests adding AT_RESISTANT flag for architectures that don't benefit

**Proposed Updates:**
- Add severity modifier: if HIGH_FANIN_CONCAT + SKIP_CONNECTION → "AT may be ineffective"

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [112] Test-Time Defense via Stochastic Resonance (2025)

**arXiv:** 2510.03224

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Perturbations as Aliasing:**
   - Views adversarial perturbations through aliasing lens
   - Stochastic noise can break aliased perturbation patterns
   - Ensemble latent representations reduce attack effectiveness

2. **Amplification Gadget Updates:**
   - MaxPool amplifies both signal and adversarial noise
   - Stochastic pooling could replace deterministic MaxPool
   - Confirms MAXPOOL_AMPLIFIER severity

3. **Defense Indicators:**
   - Look for: Dropout after pooling (stochastic element)
   - Look for: Ensemble structures before final layer

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [113] FAIR-TAT: Targeted Adversarial Training for Fairness (2025)

**arXiv:** 2410.23142

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **Shared Backbone Vulnerability:**
   - Shared backbones in multi-task models create single point of failure
   - Attack on backbone affects all downstream tasks
   - Object detectors with shared backbones are especially vulnerable

2. **Detection-Specific Insights:**
   - Confirms SHARED_BACKBONE gadget importance
   - Patch attacks can simultaneously affect detection AND classification
   - Multi-head architectures with shared backbone need separate hardening

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
### [114] Fake It Until You Break It: AI-Generated Image Detector Robustness (2024)

**arXiv:** 2410.01574

**ONNX graph indicators:**
- See findings below for ViT/CNN op patterns

**Key Findings:**

1. **ViT-Based Detector Vulnerabilities:**
   - ViT detectors for generated images are vulnerable to patches
   - Amplified patch chains work against ViT architectures
   - Attention can be manipulated to ignore detection features

2. **Patch Attack Evolution:**
   - Patches effective against CNN AND ViT detectors
   - Universal patches possible across detector types
   - Confirms AMPLIFIED_PATCH_CHAIN pattern applies to ViTs

3. **DAG Implications:**
   - Need ViT-specific gadget detection
   - Patch embedding + attention = similar vulnerability to GAP+FC

---

**What GraphSurgeon surfaces:**
`graph-surgeon motifs model.onnx`, `catalog --gadget` for mapped registry IDs.

**Static analysis limits:**
Architecture indicates attack landscape only; training and deployment defenses are not visible in the graph.

---
