# GraphSurgeon

GraphSurgeon reverse-engineers ONNX computational graphs from the command line or Python. It summarizes inputs and operators, maps depth and execution order, detects structural motifs in the DAG, cross-references adversarial ML literature, narrates data flow, and supports counterfactual edits with validation and diff.

### Recent updates

- **Architecture inference** (June 2025) — `inspect` now infers hidden dimension, vocab size, MLP intermediate size, max position embeddings, RoPE theta, tied embeddings, conv groups, and quantization format directly from the ONNX graph. Use `--json` for structured output. See [Architecture inference](#architecture-inference).
- **External data fallback** — The parser now loads models with missing or incomplete external weight data by falling back to graph-structure-only parsing. Corrupt or stub ONNX files get a clean error message instead of a traceback.
- **3D viewer** — Interactive space-themed visualization with WASD fly-through, motif region highlighting, ShadowLogic overlays, per-node structural profiles, counterfactual editing, and collapsible side panels. See [3D Visualization](#3d-visualization).

The tool is ONNX-only. It does not require PyTorch or a CUDA toolkit.

## What it does

| Command | Role |
|---------|------|
| `inspect` | Model summary: I/O tensors, operator mix, and inferred architecture (hidden dim, vocab size, MLP size, RoPE theta, tied embeddings, max positions, conv groups, quantization) |
| `topology` | Graph depth, early/middle/late layer buckets, execution order |
| `patterns` | Coarse structural blocks (conv stacks, attention, normalization chains) |
| `motifs` | Registry-backed structural motifs: which attack classes the graph topology makes architecturally plausible |
| `flow` | Plain-English execution narrative |
| `catalog` | Lookup gadgets, compound chains, literature techniques, and bundled paper notes |
| `operators` | ONNX operator reference keyed to security-relevant behavior |
| `edit` | Counterfactual graph surgery (`remove-node`) with structural, loadable, or runnable validation |
| `diff` | Compare two ONNX files after edits |
| `export-scene` | Export a SceneGraph JSON for the 3D viewer |
| `serve` | Launch a live 3D visualization server with counterfactual editing |

Motif hits describe attack landscape (what attack types the architecture enables), not confirmed exploitability. The tool does not assign risk scores or severity tiers.

## Architecture inference

`inspect` automatically infers model architecture parameters from the ONNX graph structure — no config files or model cards needed. It reports only fields it can determine; unknown fields are omitted.

| Field | How it's inferred |
|-------|-------------------|
| Hidden dimension | Mode of 1D weight shapes across normalization nodes (LayerNormalization, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, SkipLayerNormalization, RMSNormalization). Falls back to most common MatMul/MatMulNBits weight dimension, then Conv output channels. |
| Vocab size | First dimension of embedding weights (Gather/GatherBlockQuantized nodes), combined with LM head output dimension from the final MatMul/MatMulNBits/Gemm. Returns the most frequent candidate when multiple signals agree. |
| MLP intermediate size | Output dimension of gate/up projection weights identified by node name (gate_proj, up_proj, fc1, c_fc, wi). Falls back to MatMul nodes followed by Mul/Sigmoid patterns. |
| Max position embeddings | First dimension of initializers whose names contain cos_cache, sin_cache, cos_cached, sin_cached, pos_embed, position_embed, or wpe. |
| RoPE theta | Reverse-engineered from precomputed cos_cache tensor values; falls back to initializer tensors with names containing inv_freq, freqs, or rope_freq, then RotaryEmbedding node attributes. Snaps to common theta values (10000, 500000, 1000000, etc.) within 1% tolerance. |
| Tied embeddings | Detects initializers consumed by both an embedding node (Gather/GatherBlockQuantized/Embedding) and a late-graph or lm_head node (MatMul/MatMulNBits/Gemm/Transpose). Also checks for shared base weight names after stripping quantization suffixes. |
| Conv groups | Group attribute from Conv nodes (reported only when > 1) |
| Quantization format | Detected from MatMulNBits bit-width attributes or DequantizeLinear presence |

Tested on Liquid AI LFM2.5 models (hybrid gated conv + GQA + RoPE + SwiGLU) across fp32, fp16, Q4, and Q8 quantization formats. The hidden dimension and MLP size fallback paths cover standard MatMul-based and Conv-based architectures, but have not been validated on pure CNNs or standard transformer-only models. When external weight data is unavailable, shape-based fields still resolve — only RoPE theta requires actual tensor bytes (all three recovery paths — cos_cache, inv_freq, and node attributes — depend on loaded data or non-zero attributes).

Example output:

```
Architecture:
  Hidden dimension: 2048
  Vocab size: 65536
  MLP intermediate size: 8192
  Max position embeddings: 128000
  RoPE theta: 1000000
  Tied embeddings: True
  Tied initializers: ['model_embed_tokens_weight_scales', 'model_embed_tokens_weight_zp']
  Conv groups: 2048
  Quantization: Q4
```

Use `--json` for structured output suitable for scripting and cross-model comparison.

## Install

```bash
cd graph-surgeon
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

| Package | Purpose |
|---------|---------|
| `onnx`, `numpy` | Core graph parsing, motifs, topology (always installed) |
| `onnxruntime` | `edit validate --level loadable/runnable` (via `[dev]`) |
| `pytest` | Test suite (via `[dev]`) |
| `fastapi`, `uvicorn` | `serve` command for live 3D visualization (via `[viz]`) |

Minimal install (no runtime validation or tests):

```bash
.venv/bin/python -m pip install -e .
```

## CLI

After `pip install -e .`, use the project venv (the `graph-surgeon` command is not on your system PATH unless you activate the venv):

```bash
source .venv/bin/activate   # optional; then graph-surgeon works on PATH
# or always:
.venv/bin/graph-surgeon --help
.venv/bin/python -m graph_surgeon catalog --coverage
```

```bash
graph-surgeon inspect model.onnx            # human-readable summary + architecture
graph-surgeon inspect model.onnx --json     # full report as JSON
graph-surgeon topology model.onnx
graph-surgeon patterns model.onnx
graph-surgeon motifs model.onnx -o report.json
graph-surgeon flow model.onnx
graph-surgeon catalog --gadget GAP_FC_HEAD
graph-surgeon catalog --chain CHAIN-PATCH-ATTACK-SURFACE
graph-surgeon catalog --coverage
graph-surgeon operators --op Conv
graph-surgeon edit validate edited.onnx --level runnable
graph-surgeon edit remove-node model.onnx NODE_NAME -o edited.onnx
graph-surgeon diff baseline.onnx edited.onnx
graph-surgeon export-scene model.onnx -o scene.json
graph-surgeon serve model.onnx --port 9095
```

When a motif or chain is detected, use `catalog --gadget` or `catalog --chain` for registry metadata, detection logic, and paper write-ups. Display titles use registry IDs (for example `GAP_FC_HEAD — Global Average Pool → FC Head`).

Counterfactual edits (`edit`) currently expose one surgery subcommand: `remove-node`. Use `inspect` or `topology` to discover exact ONNX node names before editing.

## Research corpus

GraphSurgeon ships per-paper analysis under `graph_surgeon/taxonomy/data/` (notably `attack_research_notes.md`). That corpus is what powers rich catalog output: when you look up a gadget or chain, you get the linked AML literature, ONNX graph indicators, and attack-class mapping without fetching external docs.

Normal use does not require touching these files. To see completion status:

```bash
.venv/bin/graph-surgeon catalog --coverage
```

## Python API

GraphSurgeon is primarily a CLI tool, but the same analysis and surgery paths are available from Python. Package-level exports (`from graph_surgeon import ...`): `GraphSurgeon`, `GraphTopology`, `GraphTopologyConfig`, `LayerPosition`, `NodeTopology`, `GraphValidationLevel`, `GraphValidationResult`.

### Analysis (motifs, patterns, flow)

These mirror `motifs`, `patterns`, and `flow`:

```python
from graph_surgeon.parsers.onnx_parser import analyze_onnx_graph, analyze_onnx_patterns, quick_scan

motif_report = analyze_onnx_graph("model.onnx", output_path="report.json")
print(len(motif_report.structural_findings), motif_report.model_flow_description)

pattern_report = analyze_onnx_patterns("model.onnx")
print(quick_scan("model.onnx"))
```

JSON export from motifs uses the same sanitizer as CLI output (internal scoring fields stripped).

### Topology and graph inspection

```python
from graph_surgeon import GraphSurgeon, LayerPosition

surgeon = GraphSurgeon(verbose=False)
model = surgeon.load_model("model.onnx")
topo = surgeon.get_graph_topology(model.graph)

print(topo.total_nodes, topo.max_depth)
print(topo.by_position[LayerPosition.EARLY][:5])
print(topo.by_position[LayerPosition.LATE][:5])
print(surgeon.find_nodes_by_type(model.graph, "Conv"))
```

### Catalog and taxonomy

These mirror `catalog` lookups:

```python
from graph_surgeon.taxonomy.display import format_catalog_gadget, format_catalog_chain
from graph_surgeon.taxonomy.research_coverage import format_coverage_report
from graph_surgeon.taxonomy import motif_catalog

print(format_catalog_gadget("LINEAR_HEAD"))
print(format_catalog_chain("CHAIN-SKIP-HIGHWAY"))
print(format_coverage_report())
technique = motif_catalog.get_technique_by_id("AML-ADV-002")
```

### Counterfactual edits

CLI parity: `remove_node` matches `edit remove-node`. All surgery methods mutate the loaded `ModelProto` in place and return a `SurgeryResult` (`success`, `graph`, `message`, `nodes_added`, `nodes_removed`, `nodes_modified`, `edges_rewired`).

```python
from graph_surgeon import GraphSurgeon, GraphValidationLevel

surgeon = GraphSurgeon(verbose=False)
baseline = surgeon.load_model("model.onnx")
edited = surgeon.clone_model(baseline)

result = surgeon.remove_node(edited, "node_relu_23")
if not result.success:
    raise RuntimeError(result.message)

surgeon.save_model(edited, "edited.onnx")

check = surgeon.validate(edited, level=GraphValidationLevel.STRUCTURAL)
print(check.valid, check.errors)

diff = surgeon.compare_graphs(baseline, edited)
print(diff["summary"], diff["nodes_removed"])
```

`GraphValidationLevel.LOADABLE` and `.RUNNABLE` require `onnxruntime` (installed with `pip install -e ".[dev]"`). On a minimal install, validation falls back with warnings.

Additional graph surgery primitives (`remove_subgraph`, `insert_node_before`, `insert_node_after`, `replace_node`, `modify_node_attribute`, `add_initializer`, `add_metadata`) are Python-only today. See [docs/PYTHON_API.md](docs/PYTHON_API.md) for the full `GraphSurgeon` reference.

### Architecture inference

```python
from graph_surgeon.parsers.onnx_parser import ONNXGraphParser
from graph_surgeon.analysis.architecture import infer_architecture

parser = ONNXGraphParser()
graph = parser.parse_file("model.onnx")
arch = infer_architecture(graph)

print(arch.hidden_dim)        # e.g. 2048
print(arch.vocab_size)        # e.g. 65536
print(arch.rope_theta)        # e.g. 1000000.0, or None
print(arch.tied_embeddings)   # True, or None if not detected
print(arch.to_dict())         # dict of non-None fields only
```

### Optional weight statistics

Heuristic weight-distribution analysis (core deps only; no `onnxruntime`):

```python
from graph_surgeon.behavior.weight_signature import analyze_onnx_weights

stats = analyze_onnx_weights("model.onnx")
print(stats.summary())
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/PYTHON_API.md](docs/PYTHON_API.md) | Full `GraphSurgeon` method reference, validation levels, surgery result fields |

## 3D Visualization

GraphSurgeon includes an interactive 3D viewer that renders ONNX model graphs as navigable scenes with a space-nebula backdrop, Superluminal-inspired typography, and structural analysis overlays.

### Quick start

```bash
pip install -e ".[viz]"
source .venv/bin/activate
graph-surgeon serve model.onnx --port 9095
# Open http://127.0.0.1:9095 in a browser
```

Or export a static scene and serve it with Vite (for development):

```bash
graph-surgeon export-scene model.onnx -o viewer/sample_scene.json
cd viewer && npm install && ./node_modules/.bin/vite --port 5173
```

### Navigation

| Control | Action |
|---------|--------|
| WASD | Fly through the graph (forward/back/strafe) |
| Q / Space | Fly up |
| E / Shift | Fly down |
| Mouse drag | Orbit camera |
| Scroll wheel | Zoom in/out |
| Click node | Open node detail panel |
| Right-click node | Counterfactual edit menu |
| START / END (buttons) | Fly to the beginning or end of the graph |

### Motif and chain visualization

Detected adversarial motifs and compound chains are listed in the left sidebar. Clicking a motif:

- **Highlights** participating nodes in the 3D graph (others dim)
- **Shows a translucent bounding region** around the motif's nodes, color-coded by structural significance (red = EXCEPTIONAL, amber = PRIMARY, cyan = SECONDARY, green = TERTIARY/MITIGATING)
- **Opens a detail panel** (bottom-right) showing:
  - Structural significance and confidence badges
  - Technical description of the pattern
  - Attack types the architecture enables
  - Detection logic (how GraphSurgeon found it)
  - Research paper references from the bundled AML corpus
  - Clickable node IDs that fly the camera to each node

Use "SHOW ALL REGIONS" to light up every motif region at once for a full attack-surface overview. Duplicate findings (e.g., 50+ BatchNorm statistics leak instances) are grouped into a single sidebar entry with a count.

### Layout

The viewer adapts layout to graph shape:

- **Wide graphs** (branching architectures like Inception): nodes sorted by parent connectivity and spread horizontally at each depth
- **Narrow graphs** (sequential models like ResNet): helical/spiral layout to prevent a single vertical line
- **Parameter-only nodes** (Unsqueeze, Reshape ops that only consume initializers): repositioned next to their consumer nodes instead of cluttering depth 0

### Additional features

- **Search**: type in the search bar to filter nodes by name or op type
- **Flow walk**: "WALK FLOW" steps through execution order depth-by-depth with a detail panel showing each level's nodes
- **Model flow description**: "VIEW MODEL FLOW" opens a full-screen readable narrative
- **ShadowLogic overlay** (project-defined heuristic for identifying potential logic-injection sites): shows injection point markers on the 3D graph with fly-to navigation and per-point detail popups
- **Structural patterns**: toggle highlights for gradient bottlenecks, fusion points, amplification layers, and defense points
- **Per-node structural profiles**: clicking a node shows gauge-bar metrics for topology-derived structural proxies (gradient sensitivity, Lipschitz estimate, perturbation amplification, shadowlogic capacity, extraction leakage); these are heuristic estimates based on graph structure, not runtime-computed values
- **Collapsible side panel**: all sections (Flow, Navigation, ShadowLogic, Patterns, Motifs & Chains) collapse independently
- **Drag-and-drop**: drop a `scene.json` file onto the viewer to load it
- **Counterfactual editing**: right-click a node to remove it; the server re-analyzes and the view updates live

## Comparison to Netron

Netron is a graph viewer: nodes, tensors, and shapes on screen.

GraphSurgeon is an analysis and experimentation layer on top of the same ONNX files:

- Positional topology (stem vs middle vs head) and ordered execution, not just adjacency
- Automated motif and pattern detection with a typed registry and literature cross-reference
- A searchable catalog of gadgets, chains, techniques, and bundled paper notes
- Counterfactual edits (remove a node, rewire, validate, diff against baseline)
- JSON export for scripting and batch comparison across model variants

Use Netron to see the graph; use GraphSurgeon to interpret structure, relate it to published attack classes, test what changes when you alter the DAG, and visualize the adversarial attack surface in 3D.

## Tests

Unit tests (no external ONNX files):

```bash
.venv/bin/python -m pytest tests/ -v
```

Integration tests that require off-repo ONNX fixtures are gitignored and documented in `tests/README.md`.

## License

MIT
