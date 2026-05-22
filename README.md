# GraphSurgeon

GraphSurgeon helps reverse engineers and ML researchers **inspect, map, and experiment on ONNX computational DAGs**: topology (stem vs head), structural motifs, operator reference, execution flow, and **counterfactual graph edits** with validation.

It is not a vulnerability scanner, exploitability scorer, or red-team grafting tool.

## Install

```bash
cd graph-surgeon
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Optional runtime characterization:

```bash
.venv/bin/python -m pip install -e ".[behavior]"
```

## CLI

```bash
graph-surgeon inspect model.onnx
graph-surgeon topology model.onnx
graph-surgeon motifs model.onnx -o report.json
graph-surgeon flow model.onnx
graph-surgeon catalog --category adversarial_perturbation
graph-surgeon operators --op Conv
graph-surgeon edit validate edited.onnx --level runnable
graph-surgeon diff baseline.onnx edited.onnx
```

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

## What motifs mean

Structural motifs describe **attack landscape** (what perturbation classes are structurally plausible on this graph). They do **not** predict attack success rate, robustness, or exploitability. Training and deployment context determine whether attacks succeed.

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

See `tests/README.md`.

## Ecosystem

- **Carcinoma**: offensive grafting; syncs graph primitives from GraphSurgeon (`SYNC.md`)
- **nn_security_analyzer**: frozen reference tree; ONNX fixtures remain on disk for local integration tests
- **Silent Scalpel**: research publications (not shipped in this repo)

## License

MIT
