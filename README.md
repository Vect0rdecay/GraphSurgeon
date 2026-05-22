# GraphSurgeon

GraphSurgeon helps reverse engineers and ML researchers **inspect, map, and experiment on ONNX computational DAGs**: topology (stem vs head), structural motifs, operator reference, execution flow, and **counterfactual graph edits** with validation.

This is not a vulnerability scanner, exploitability scorer, or red-team grafting tool. Structural motifs describe **attack landscape** (what perturbation classes are architecturally plausible). They do not predict attack success rate, robustness, or exploitability.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Optional behavioral characterization:

```bash
.venv/bin/python -m pip install -e ".[behavior]"
```

## CLI

```bash
graph-surgeon inspect model.onnx
graph-surgeon topology model.onnx
graph-surgeon motifs model.onnx -o report.json
graph-surgeon patterns model.onnx
graph-surgeon flow model.onnx
graph-surgeon catalog --summary
graph-surgeon operators --op Conv
graph-surgeon edit remove-subgraph model.onnx --nodes relu2 -o edited.onnx
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

## Integration tests

Set `GRAPH_SURGEON_FIXTURE_ROOT` to a directory containing RobustBench ONNX models (default: `../nn_security_analyzer/robustbench_validation`). See [tests/README.md](tests/README.md).

## Compared to Netron

Netron visualizes a single graph interactively. GraphSurgeon adds programmatic topology classification, structural motif detection, pattern analysis, counterfactual edits with validation, and batch-friendly CLI output for reverse-engineering workflows.

## Limitations

Architecture and motifs describe attack landscape, not exploitability. Runtime probes (`graph-surgeon[behavior]`) characterize observed behavior on a given checkpoint; they are not a security grade.

## Ecosystem

- **Carcinoma**: offensive grafting; syncs shared GraphSurgeon modules (see [SYNC.md](SYNC.md))
- **Silent Scalpel**: research publications (not shipped in this repo)
- **nn_security_analyzer**: legacy source dir kept on disk for fixtures and reference
