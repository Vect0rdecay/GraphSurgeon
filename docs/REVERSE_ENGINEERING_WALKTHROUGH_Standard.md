# GraphSurgeon Reverse-Engineering Walkthrough: `Standard.onnx`

**Model path:** `/home/s0crates/nn_security_analyzer/robustbench_validation/Standard.onnx`  
**External weights:** `Standard.onnx.data` (present, ~146 MB)  
**Validation date:** 2026-05-22  
**Tool:** GraphSurgeon (`graph-surgeon` from `/home/s0crates/graph-surgeon/.venv`)

This document records an end-to-end GraphSurgeon CLI validation on one ONNX artifact. Findings describe **structural motifs** and **attack landscape** (what attack classes the graph topology makes architecturally plausible). They do not measure exploitability, robustness training, or deployment hardening.

---

## 1. Executive summary

`Standard.onnx` is a compact image classifier exported as graph `main_graph` with **80 nodes**, **92 initializers**, one input (`input`), and one output (`output`). Operator mix is dominated by convolution and residual blocks:

| Operator | Count |
|----------|------:|
| Conv | 28 |
| Relu | 25 |
| Add | 12 |
| BatchNormalization | 12 |
| AveragePool | 1 |
| Reshape | 1 |
| Gemm | 1 |

From the first convolution’s tensor shapes in the motifs report, the graph expects **`[batch, 3, 32, 32]`** activations (CIFAR-scale RGB). The final `Gemm` at `node_linear` projects to **10 logits**, consistent with a 10-way classification head (typical CIFAR-10 style deployment, though class names are not in the ONNX file).

Architecturally, the graph reads as a **ResNet-style residual CNN**: repeated `Conv → ReLU → Conv → Add` blocks with `BatchNormalization` after each fusion, channel widths stepping **32 → 16 → 8**, then **global-style pooling** via `AveragePool` (`node_avg_pool2d`, kernel `[8, 8]`), `Reshape` to a 640-dimensional vector (`node_view`), and a fully connected classifier (`node_linear`).

Topology depth is **76–77 hops** from input to output on the longest path, with **17 early** and **16 late** nodes tagged by the topology pass. The graph is small enough for manual audit (motifs shadowlogic assessment: ~80 nodes, audit complexity risk LOW), but deep enough that residual highways and many BatchNorm layers define a rich gradient- and distribution-shift attack landscape.

**GraphSurgeon validation:** All requested CLI subcommands completed successfully on this model. Counterfactual `edit remove-node` removed one non-output `Relu` (`node_relu_23`); structural validation and `diff` reported a coherent single-node deletion with consumer rewire.

---

## 2. How data flows

### 2.1 Topology and execution order

`topology` reports **80 nodes**, **max depth 76**, with early-layer names including `node_Conv_332`, `node_relu`, and the first residual add `node_add_50`. Late layer names include `node_avg_pool2d`, `node_view`, and `node_linear`. JSON export matches the text summary (`total_nodes: 80`, `max_depth: 76`).

### 2.2 Narrative flow (`flow`)

The `flow` command stages the DAG in plain language:

1. **Stem and stage 1 (32 channels):** `node_Conv_332` → ReLU → parallel convs → `node_add_50` (residual) → BatchNorm → further conv/ReLU/add cycles (`node_add_96`, etc.).
2. **Middle stages (16 channels):** Strided convolutions at `node_Conv_347`, `node_Conv_349`, `node_Conv_360`, `node_Conv_362` reduce spatial size; residual adds continue (`node_add_239`, `node_add_285`, …).
3. **Late stage (8 channels):** Narrower conv blocks and residuals through `node_add_566`.
4. **Head:** `node_avg_pool2d` (8×8 average pool) → `node_view` (reshape to 640 features) → `node_linear` (Gemm to 10 outputs).

Residual `Add` nodes are explicitly labeled as skip connections in the narrative, which matches the node naming pattern (`node_add_*` paired with conv blocks).

### 2.3 Data-flow diagram (logical)

```
input [B,3,32,32]
  └─► Conv/BN/ReLU blocks (32 ch) ──► residual adds
        └─► downsample Conv (32→16 ch) ──► residual adds
              └─► downsample Conv (16→8 ch) ──► residual adds
                    └─► AvgPool 8×8 ──► Reshape(640) ──► Gemm ──► output [B,10]
```

---

## 3. Structural patterns (`patterns`)

The `patterns` report surfaces DAG-level regularities beyond per-node motifs. Highlights for this graph:

| Pattern | Category | RE takeaway |
|---------|----------|-------------|
| Attack-Friendly Early Layers | gradient_flow | Two short linear chains in early layers without normalization between linear ops; informative gradients for optimization-based attacks. |
| Final FC Layer (Logit Target) at `node_linear` | attack_surface | Terminal `Gemm` is the logit surface for margin / feature-space methods. |
| IN_GRAPH_PREPROCESSING | feature_extraction | Early BatchNorm (`node__native_batch_norm_legit_no_training_2__0`) within ~15% of input hops; pixel statistics normalized inside the graph. |
| Multimodal Fusion Point (×9) | perturbation_fusion | Heuristic label on residual `Add` nodes with two inputs; here they are **residual merges**, not true multimodal fusion. Treat as fusion topology for gradient routing, not as evidence of audio/vision branches. |
| NORMALIZER (12 BatchNorm layers) | attack_surface | Distribution-shift and BN-targeted attack landscape across the stack. |
| No Gradient Regularization Detected | gradient_flow | No Dropout/DropPath/noise ops in 80 nodes. |
| ShadowLogic Risk: Deep Architecture (depth=77) | attack_surface | Many layers complicate exhaustive manual review (audit still feasible at this node count). |
| ShadowLogic Risk: High Parameter Capacity (29 linear layers) | attack_surface | Many Conv/Gemm layers; capacity motif for hidden-subnet discussions, not evidence of backdoors. |
| Unbounded ReLU Boundaries (13 nodes) | gradient_flow | ReLU after linear ops without adjacent Lipschitz-style constraints. |
| VALID_CONV_BOUNDARY (3 valid convs) | feature_extraction | `pads=0` convs at `node_Conv_336`, `node_Conv_349`, `node_Conv_362`; edge/boundary sensitivity motif. |
| Average Pooling at `node_avg_pool2d` | architectural indicator | Pooling smooths spatial maps before the FC head (distinct from GlobalAveragePool → FC patch motif chain). |

**Associated attack classes (patterns report)** map components to literature technique families: FGSM/PGD/C&W on linear chains and residuals, distribution-shift on BatchNorm, margin attacks on the final FC, boundary attacks on valid convs, etc. These are **landscape labels**, not measured attack success rates.

---

## 4. Motifs and chains (`motifs`)

`motifs` wrote **`/tmp/gs_standard_motifs.json`** with **16 structural findings** and a **gadget summary** (28 registry-aligned gadget instances across the graph).

### 4.1 Registry gadgets detected in this graph

| Registry ID | Count / location | Meaning for this graph |
|-------------|------------------|------------------------|
| `NORMALIZER` | 12 BatchNorm nodes | BN distribution-shift landscape; aligns with `CHAIN-BN-FRAGILITY` finding. |
| `SKIP_CONNECTION` | 9 residual adds | Gradient highway landscape; aligns with `CHAIN-SKIP-HIGHWAY`. |
| `DOWNSAMPLER` | 4 stride-2 convs | Spatial reduction / frequency-survivability landscape. |
| `LINEAR_HEAD` | `node_linear` | Final `Gemm` logit target (C&W / margin landscape). |
| `encoder_projection_bridge` | 1 (summary) | Bridge motif in gadget taxonomy; inspect node profile if tracing. |
| `single_modality_input` | 1 graph input | Single `input` tensor; vision-only deployment surface. |

**Not detected:** `GAP_FC_HEAD` (this model uses `AveragePool` + `Reshape` + `Gemm`, not GlobalAveragePool → FC). Catalog still documents `GAP_FC_HEAD` for literature cross-reference; `CHAIN-PATCH-ATTACK-SURFACE` requires that motif and was **not** reported in motifs output for this file (`patch_attacks_high_risk` empty in gadget summary).

### 4.2 Compound chains and motif-level findings

| ID | Type | Primary nodes | Landscape note |
|----|------|---------------|----------------|
| `PRIV-BN-*` (×12) | structural_motif | Each BatchNorm | Membership / attribute inference landscape via stored running stats (privacy class). |
| `CHAIN-SKIP-HIGHWAY` | attack_chain | `node_add_96` (9 skips total) | Residual gradient highways; PGD/C&W convergence landscape. |
| `CHAIN-BN-FRAGILITY` | gadget | First BN | BN fragility under adversarial distribution shift. |
| `CHAIN-REDUCTION-SURVIVE` | attack_chain | `node_Conv_347` | Four spatial reductions; patch/scale survivability landscape. |
| `CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY` | structural_motif | Graph-level | No `Where`/`If` in graph today; ONNX editability + injection points (e.g. before `node_view`) index **supply-chain graph tampering** landscape, not an existing backdoor. |

Shadowlogic assessment in JSON: **no existing conditional backdoor indicators**; **injection susceptibility scored HIGH** in the tool’s structural rubric (format editability unknown, integrity verification HIGH, 10 injection points listed). Again: this describes **where graph surgery could occur**, not that this file is compromised.

### 4.3 Attack enablers summary (from motifs JSON)

- **gradient_highway_attacks:** 9 skip connections.  
- **frequency_attacks:** 4 downsampler convs.  
- **feature_space_attacks:** `LINEAR_HEAD` at `node_linear`.  
- **patch_attacks_high_risk / amplified_patch_attacks:** empty (consistent with no `GAP_FC_HEAD`).

---

## 5. Catalog workflow

Recommended analyst sequence after `motifs`:

1. **`graph-surgeon catalog`** — Browse motif IDs, chain IDs, and AML technique index (`AML-ADV-001` … `AML-PRIV-003`).
2. **`graph-surgeon catalog --gadget <ID>`** — Deep dive on a motif seen in motifs JSON.  
   - For this model: **`NORMALIZER`**, **`LINEAR_HEAD`**, **`SKIP_CONNECTION`**, **`DOWNSAMPLER`**.  
   - Reference only: **`GAP_FC_HEAD`** (not present in this graph; useful for comparing to GAP+FC classifiers).
3. **`graph-surgeon catalog --chain <ID>`** — Compound landscape.  
   - Detected: **`CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY`**.  
   - Reference: **`CHAIN-PATCH-ATTACK-SURFACE`** (educational; not fired on this ONNX).
4. **`graph-surgeon catalog --coverage`** — Taxonomy literature corpus status (75 complete, 0 missing at time of run).
5. **`graph-surgeon catalog --technique AML-ADV-002`** — PGD description tied to optimization-based attacks on linear/residual graphs.
6. **`graph-surgeon operators --op Conv`** — Operator-level RE notes (gradient sensitivity, hardening pointers).

Catalog output consistently states that motifs define **attack landscape**, not confirmed exploitability.

---

## 6. Counterfactual edits

| Step | Command | Result |
|------|---------|--------|
| Baseline integrity | `edit validate Standard.onnx --level loadable` | `valid=True level=loadable` |
| Surgical edit | `edit remove-node Standard.onnx node_relu_23 -o /tmp/gs_standard_edited.onnx` | Success; 79 nodes |
| Post-edit integrity | `edit validate /tmp/gs_standard_edited.onnx --level structural` | `valid=True level=structural` |
| Change summary | `diff Standard.onnx /tmp/gs_standard_edited.onnx` | `-1` node (`node_relu_23`); `node_Conv_371` input rewired from `relu_23` to `getitem_69` |

**Interpretation:** Removing a late-block ReLU is structurally valid (consumer conv accepts the pre-activation tensor). This demonstrates GraphSurgeon’s counterfactual pipeline: edit → validate → diff. It does **not** imply the original model was incorrect or that accuracy/robustness is unchanged at runtime; only graph structural consistency was checked.

Removing nodes on the sole path to `output` (e.g. `node_linear`, `node_avg_pool2d`, or stem convs) would be unsafe for semantics; analysts should prefer late, single-consumer activations or use validate-only on the baseline when unsure.

---

## 7. Limits (what GraphSurgeon cannot tell you from the graph alone)

- **Training recipe:** Adversarial training, data augmentation, or robustness benchmarks (e.g. RobustBench scores) are not in ONNX ops.
- **Weight semantics:** Initializers are present externally (`.onnx.data`), but this walkthrough did not inspect numeric weights for backdoors or steganography.
- **Exploitability:** Motifs say which attack **classes** align with topology, not whether attacks succeed on this checkpoint.
- **Deployment context:** Preprocessing outside the graph (camera ISP, normalization constants in the serving binary) may differ from in-graph BatchNorm.
- **Heuristic false positives:** “Multimodal fusion” on residual `Add` nodes is a naming/heuristic artifact; human RE must map patterns to true modality boundaries.
- **Parameter counts:** Motifs JSON reported `total_parameters: 0` (likely external-weight accounting limitation); do not use that field for capacity analysis without a separate weight audit.
- **Runtime behavior:** `loadable` validation does not substitute for ONNX Runtime inference tests with real inputs.

---

## 8. Appendix — command cheat sheet

Environment:

```bash
cd /home/s0crates/graph-surgeon
GS=.venv/bin/graph-surgeon
MODEL=/home/s0crates/nn_security_analyzer/robustbench_validation/Standard.onnx
```

| Purpose | Command |
|---------|---------|
| Smoke | `$GS --help` |
| Inventory | `$GS inspect $MODEL` |
| Depth / bands | `$GS topology $MODEL` |
| Machine-readable topology | `$GS topology $MODEL --json` |
| DAG patterns | `$GS patterns $MODEL` |
| Motifs + JSON export | `$GS motifs $MODEL -o /tmp/gs_standard_motifs.json` |
| Plain-English walk | `$GS flow $MODEL` |
| Registry index | `$GS catalog` |
| Motif detail | `$GS catalog --gadget LINEAR_HEAD` |
| Chain detail | `$GS catalog --chain CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY` |
| Literature coverage | `$GS catalog --coverage` |
| Technique note | `$GS catalog --technique AML-ADV-002` |
| Operator reference | `$GS operators --op Conv` |
| Validate original | `$GS edit validate $MODEL --level loadable` |
| Counterfactual | `$GS edit remove-node $MODEL node_relu_23 -o /tmp/gs_standard_edited.onnx` |
| Validate edit | `$GS edit validate /tmp/gs_standard_edited.onnx --level structural` |
| Compare graphs | `$GS diff $MODEL /tmp/gs_standard_edited.onnx` |

CLI regression (optional after changes): `.venv/bin/python -m pytest tests/test_cli.py -q` → **6 passed**.

---

## CLI validation log (this session)

| Command | Status |
|---------|--------|
| `graph-surgeon --help` | OK |
| `inspect` | OK |
| `topology` | OK |
| `topology --json` | OK |
| `patterns` | OK |
| `motifs -o /tmp/gs_standard_motifs.json` | OK |
| `flow` | OK |
| `catalog` | OK |
| `catalog --gadget GAP_FC_HEAD` | OK (reference; motif not in graph) |
| `catalog --gadget NORMALIZER` | OK |
| `catalog --chain CHAIN-PATCH-ATTACK-SURFACE` | OK (reference; chain not in motifs) |
| `catalog --chain CHAIN-SHADOWLOGIC-INJECTION-SUSCEPTIBILITY` | OK |
| `catalog --coverage` | OK |
| `catalog --technique AML-ADV-002` | OK |
| `operators --op Conv` | OK |
| `edit validate --level loadable` | OK |
| `edit remove-node` + `edit validate --level structural` | OK |
| `diff` | OK |

No CLI failures required fixes during this validation run.
