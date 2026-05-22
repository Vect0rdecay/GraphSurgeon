# GraphSurgeon Python API

Reference for programmatic use. The CLI wraps many of these entry points; this document focuses on `GraphSurgeon` and related types not fully spelled out in the README.

## Package exports

Import from the top-level package:

```python
from graph_surgeon import (
    GraphSurgeon,
    GraphTopology,
    GraphTopologyConfig,
    LayerPosition,
    NodeTopology,
    GraphValidationLevel,
    GraphValidationResult,
)
```

## Analysis entry points

| Function | Module | CLI equivalent |
|----------|--------|----------------|
| `analyze_onnx_graph(filepath, output_path=None, verbose=False)` | `graph_surgeon.parsers.onnx_parser` | `motifs` |
| `analyze_onnx_patterns(filepath)` | `graph_surgeon.parsers.onnx_parser` | `patterns` |
| `quick_scan(filepath)` | `graph_surgeon.parsers.onnx_parser` | (text summary) |
| `ONNXGraphParser().parse_file(path)` | `graph_surgeon.parsers.onnx_parser` | `inspect` (partial) |

`analyze_onnx_graph` returns a `ModelMotifReport` with `structural_findings`, gadget summary fields, and optional `model_flow_description` (same narrative as `flow`).

## Catalog and taxonomy

| Function | Module | CLI equivalent |
|----------|--------|----------------|
| `format_catalog_gadget(gadget_id)` | `graph_surgeon.taxonomy.display` | `catalog --gadget` |
| `format_catalog_chain(chain_id)` | `graph_surgeon.taxonomy.display` | `catalog --chain` |
| `format_coverage_report()` | `graph_surgeon.taxonomy.research_coverage` | `catalog --coverage` |
| `motif_catalog.get_technique_by_id(id)` | `graph_surgeon.taxonomy.motif_catalog` | `catalog --technique` |
| `get_gadget_info(gadget_id)` | `graph_surgeon.taxonomy.gadget_registry` | registry metadata |

## GraphSurgeon: load and save

| Method | Description |
|--------|-------------|
| `load_model(path)` | Load `onnx.ModelProto` from disk |
| `save_model(model, path)` | Write model to disk |
| `clone_model(model)` | Deep copy before destructive edits |

## GraphSurgeon: topology and lookup

| Method | Description |
|--------|-------------|
| `get_graph_topology(graph, config=None)` | Returns `GraphTopology` (depth, execution order, early/middle/late buckets) |
| `get_early_layers(graph, op_type=None)` | Nodes in early band, optionally filtered by op |
| `get_late_layers(graph, op_type=None)` | Nodes in late band, optionally filtered by op |
| `get_node_by_name(graph, name)` | Single node or `None` |
| `get_node_by_output(graph, output_name)` | Producer of a tensor name |
| `get_node_consumers(graph, tensor_name)` | Downstream nodes consuming a tensor |
| `find_nodes_by_type(graph, op_type)` | All nodes with given ONNX op type |
| `find_nodes_by_attribute(graph, attr_name, value=None)` | Attribute filter |
| `get_tensor_shape(model, name)` | Shape lookup when value info exists |
| `infer_shapes(model)` | Run ONNX shape inference in place |
| `check_shape_compatibility(model, a, b)` | Compare two tensor shapes |

## GraphSurgeon: counterfactual surgery

All edit methods modify the passed `model` in place and return `SurgeryResult`:

| Field | Meaning |
|-------|---------|
| `success` | Whether the operation completed |
| `message` | Human-readable status |
| `nodes_added` / `nodes_removed` / `nodes_modified` | Affected node names |
| `edges_rewired` | Count of consumer input rewires |

| Method | CLI | Description |
|--------|-----|-------------|
| `remove_node(model, node_name, rewire_input_idx=0)` | `edit remove-node` | Delete one node; rewire consumers to a chosen input of the removed node |
| `remove_subgraph(model, node_names, entry_rewire=None)` | — | Remove a connected block; bridge external consumers to subgraph entry |
| `insert_node_before(model, target, new_node, input_idx=0)` | — | Splice node before target, intercepting one input |
| `insert_node_after(model, target_output, new_node, new_output_name)` | — | Insert after a tensor; rewire downstream consumers |
| `replace_node(model, old_name, new_node)` | — | Swap op while preserving graph position and wiring |
| `modify_node_attribute(model, node_name, attr_name, new_value)` | — | Change an existing node attribute |
| `add_initializer(model, name, values)` | — | Append a weight tensor (`numpy` array) |
| `add_metadata(model, key, value)` | — | Set `metadata_props` entry |
| `get_metadata(model, key)` | — | Read metadata entry |

Helper: `clone_node(node, new_name=None)` copies a node proto for insert/replace workflows.

## GraphSurgeon: validate and diff

| Method | CLI | Description |
|--------|-----|-------------|
| `validate(model, level=STRUCTURAL, sample_input=None)` | `edit validate` | Check edited graph |
| `compare_graphs(model_a, model_b)` | `diff` | Node/initializer delta dict |

### Validation levels

| Level | Checks |
|-------|--------|
| `NONE` | Skip validation |
| `STRUCTURAL` | ONNX checker (graph shape and schema) |
| `LOADABLE` | Structural + ONNX Runtime session creation |
| `RUNNABLE` | Loadable + sample inference (random input if `sample_input` omitted) |

`LOADABLE` and `RUNNABLE` require `onnxruntime`. If it is missing, validation returns success with warnings and skips runtime checks.

## Weight analysis (optional)

```python
from graph_surgeon.behavior.weight_signature import analyze_onnx_weights

result = analyze_onnx_weights("model.onnx", min_tensor_size=100)
# result.avg_kurtosis, result.detected_training, result.summary()
```

Uses only `onnx` and `numpy`. Interprets kurtosis heuristically; not a substitute for robustness benchmarks.

## Typical counterfactual workflow

```python
from graph_surgeon import GraphSurgeon, GraphValidationLevel

surgeon = GraphSurgeon(verbose=False)
baseline = surgeon.load_model("model.onnx")
edited = surgeon.clone_model(baseline)

result = surgeon.remove_node(edited, "Relu_42")
assert result.success, result.message

surgeon.save_model(edited, "edited.onnx")
validation = surgeon.validate(edited, level=GraphValidationLevel.LOADABLE)
assert validation.valid, validation.errors

print(surgeon.compare_graphs(baseline, edited)["summary"])
```

For multi-node removal or op replacement, use `remove_subgraph` or `replace_node` instead of `remove_node`; same validate/diff pattern applies.
