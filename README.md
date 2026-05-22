# GraphSurgeon

GraphSurgeon helps reverse engineers and ML researchers **inspect, map, and experiment on ONNX computational DAGs**: topology (stem vs head), structural motifs, operator reference, execution flow, and **counterfactual graph edits** with validation.

GraphSurgeon is **ONNX-only**. It does not require PyTorch or a CUDA toolkit.

## Install

```bash
cd graph-surgeon
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

| Package | Purpose |
|---------|---------|
| `onnx`, `numpy` | Core graph parsing, motifs, topology (always installed) |
| `onnxruntime` | `edit validate --level loadable/runnable` and integration tests (via `[dev]`) |
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
.venv/bin/graph-surgeon --help          # subcommands + copy-paste examples
.venv/bin/graph-surgeon catalog --help  # --gadget, --chain, --coverage, etc.
.venv/bin/graph-surgeon catalog --coverage
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
graph-surgeon catalog --category adversarial_perturbation
graph-surgeon operators --op Conv
graph-surgeon edit validate edited.onnx --level runnable
graph-surgeon edit remove-node model.onnx NODE_NAME -o edited.onnx
graph-surgeon diff baseline.onnx edited.onnx
```

Display titles use registry IDs (for example `GAP_FC_HEAD — Global Average Pool → FC Head`).
Each finding indexes AML literature by graph structure; it does not rate security posture.
Use `catalog --gadget` or `catalog --chain` to look up `research_basis`, paper analysis, and associated attack classes. Research notes and detection rationale ship inside the package under `graph_surgeon/taxonomy/data/` (no external research path required).

### Research corpus

- Authoring spec: `graph_surgeon/taxonomy/data/RESEARCH_NOTE_TEMPLATE.md`
- Coverage: `.venv/bin/graph-surgeon catalog --coverage` (or `python -m graph_surgeon catalog --coverage` from the venv)
- Optional rebuild after bulk edits: `GRAPH_SURGEON_RESEARCH_SOURCE=/path/to/research .venv/bin/python scripts/build_research_corpus.py`

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

## What motifs and patterns mean

Structural motifs and patterns describe **attack landscape**: which adversarial attack classes published in the AML literature are structurally plausible on this ONNX graph. GraphSurgeon is a research-grounded structural index, not a scanner that confirms exploitability or assigns severity.

Outputs omit risk scores and severity tiers. Training, weights, and deployment determine whether attacks succeed.

Deployment-context motifs (`SINGLE_MODALITY_INPUT`, `IN_GRAPH_PREPROCESSING`, `HAS_MULTIMODAL_FUSION`, audio registry IDs) index papers where the attack channel is partly outside the DAG (thermal, ISP, acoustics). Chains include `CHAIN-SINGLE-MODALITY-VISION`, `CHAIN-PREPROCESSING-TRUST-BOUNDARY`, `CHAIN-AUDIO-ADVERSARIAL-SURFACE`, and `CHAIN-ACOUSTIC-COMMAND-SURFACE`. Lookup: `graph-surgeon catalog --gadget SINGLE_MODALITY_INPUT`.

## Comparison to Netron

Netron visualizes a single graph. GraphSurgeon adds depth-based topology (early/middle/late), motif catalog cross-reference, batch diff across edited graphs, and validate-after-edit workflows for reverse engineering.

## Tests

Unit tests (no large ONNX files):

```bash
.venv/bin/python -m pytest tests/test_graph_surgeon.py -v
```

Integration tests (RobustBench corpus via env var):

```bash
export GRAPH_SURGEON_FIXTURE_ROOT=/path/to/onnx/fixtures
.venv/bin/python -m pytest tests/ -v -m integration
```

Smoke script (pilot models by default):

```bash
.venv/bin/python scripts/smoke_robustbench.py
```

See `tests/README.md`.

## Maintainer notes



***REMOVED***

## License

MIT
