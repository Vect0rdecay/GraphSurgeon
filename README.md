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

```bash
graph-surgeon inspect model.onnx
graph-surgeon topology model.onnx
graph-surgeon patterns model.onnx
graph-surgeon motifs model.onnx -o report.json
graph-surgeon flow model.onnx
graph-surgeon catalog --gadget GAP_FC_HEAD
graph-surgeon catalog --chain CHAIN-PATCH-ATTACK-SURFACE
graph-surgeon catalog --category adversarial_perturbation
graph-surgeon operators --op Conv
graph-surgeon edit validate edited.onnx --level runnable
graph-surgeon edit remove-node model.onnx NODE_NAME -o edited.onnx
graph-surgeon diff baseline.onnx edited.onnx
```

Display titles use registry IDs (for example `GAP_FC_HEAD — Global Average Pool → FC Head`).
Each finding indexes AML literature by graph structure; it does not rate security posture.
Use `catalog --gadget` or `catalog --chain` to look up `research_basis` and associated attack classes.

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

## Comparison to Netron

Netron visualizes a single graph. GraphSurgeon adds depth-based topology (early/middle/late), motif catalog cross-reference, batch diff across edited graphs, and validate-after-edit workflows for reverse engineering.

## Tests

Unit tests (no large ONNX files):

```bash
.venv/bin/python -m pytest tests/test_graph_surgeon.py -v
```

Integration tests (RobustBench corpus via env var):

```bash
export GRAPH_SURGEON_FIXTURE_ROOT=/home/s0crates/nn_security_analyzer/robustbench_validation
.venv/bin/python -m pytest tests/ -v -m integration
```

Smoke script (pilot models by default):

```bash
.venv/bin/python scripts/smoke_robustbench.py
```

See `tests/README.md`.

## License

MIT
