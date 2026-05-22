# GraphSurgeon

GraphSurgeon reverse-engineers ONNX computational graphs from the command line or Python. It summarizes inputs and operators, maps depth and execution order, detects structural motifs in the DAG, cross-references adversarial ML literature, narrates data flow, and supports counterfactual edits with validation and diff.

The tool is ONNX-only. It does not require PyTorch or a CUDA toolkit.

## What it does

| Command | Role |
|---------|------|
| `inspect` | Model summary: I/O tensors, initializer count, operator mix |
| `topology` | Graph depth, early/middle/late layer buckets, execution order |
| `patterns` | Coarse structural blocks (conv stacks, attention, normalization chains) |
| `motifs` | Registry-backed structural motifs: which attack classes the graph topology makes architecturally plausible |
| `flow` | Plain-English execution narrative |
| `catalog` | Lookup gadgets, compound chains, literature techniques, and bundled paper notes |
| `operators` | ONNX operator reference keyed to security-relevant behavior |
| `edit` | Counterfactual graph surgery (`remove-node`) with structural, loadable, or runnable validation |
| `diff` | Compare two ONNX files after edits |

Motif hits describe attack landscape (what attack types the architecture enables), not confirmed exploitability. The tool does not assign risk scores or severity tiers.

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
graph-surgeon inspect model.onnx
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
```

When a motif or chain is detected, use `catalog --gadget` or `catalog --chain` for registry metadata, detection logic, and paper write-ups. Display titles use registry IDs (for example `GAP_FC_HEAD — Global Average Pool → FC Head`).

## Research corpus

GraphSurgeon ships per-paper analysis under `graph_surgeon/taxonomy/data/` (notably `attack_research_notes.md`). That corpus is what powers rich catalog output: when you look up a gadget or chain, you get the linked AML literature, ONNX graph indicators, and attack-class mapping without fetching external docs.

Normal use does not require touching these files. To see completion status:

```bash
.venv/bin/graph-surgeon catalog --coverage
```

Authoring format for new or updated notes: `graph_surgeon/taxonomy/data/RESEARCH_NOTE_TEMPLATE.md`.

## Python API

```python
from graph_surgeon import GraphSurgeon
from graph_surgeon.graph.topology import LayerPosition
from graph_surgeon.graph.validation import GraphValidationLevel

surgeon = GraphSurgeon(verbose=False)
model = surgeon.load_model("model.onnx")
topo = surgeon.get_graph_topology(model.graph)
stem = topo.by_position[LayerPosition.EARLY]
head = topo.by_position[LayerPosition.LATE]
```

Counterfactual edit:

```python
result = surgeon.remove_subgraph(model, node_names=["block_a", "block_b"])
check = surgeon.validate(model, level=GraphValidationLevel.RUNNABLE)
```

Optional ONNX weight statistics (Python API, no extra install beyond core):

```python
from graph_surgeon.behavior.weight_signature import analyze_onnx_weights

stats = analyze_onnx_weights("model.onnx")
print(stats.summary())
```

## Comparison to Netron

Netron is a graph viewer: nodes, tensors, and shapes on screen.

GraphSurgeon is an analysis and experimentation layer on top of the same ONNX files:

- Positional topology (stem vs middle vs head) and ordered execution, not just adjacency
- Automated motif and pattern detection with a typed registry and literature cross-reference
- A searchable catalog of gadgets, chains, techniques, and bundled paper notes
- Counterfactual edits (remove a node, rewire, validate, diff against baseline)
- JSON export for scripting and batch comparison across model variants

Use Netron to see the graph; use GraphSurgeon to interpret structure, relate it to published attack classes, and test what changes when you alter the DAG.

## Tests

Unit tests (no external ONNX files):

```bash
.venv/bin/python -m pytest tests/ -v
```

Integration tests that require off-repo ONNX fixtures are gitignored and documented in `tests/README.md`.

## License

MIT
