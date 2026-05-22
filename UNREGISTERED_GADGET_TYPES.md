# Unregistered GadgetType Values

`GadgetType` in `graph_surgeon/analysis/motifs.py` includes enum members used at detection
time that do not yet have entries in `graph_surgeon/taxonomy/gadget_registry.py`.

Findings for these types keep descriptive titles until a registry entry exists.

| GadgetType (enum) | Notes |
|-------------------|-------|
| `perturbation_carrier` | Generic Conv/MatMul carrier; too broad for a single registry motif |
| `capacity_reservoir` | Weight capacity heuristic; not a published structural motif ID |
| `gradient_gate` | ReLU gating; covered indirectly by linear-chain analysis |
| `spatial_reducer` | Generic spatial reduction |
| `single_objectness_path` | Documented in detection coverage; registry entry pending |
| `extraction_surface` | Softmax/logit leakage; partial overlap with extraction gadgets |

Audio and deployment-context types (`AUDIO_MEL_INPUT`, `SINGLE_MODALITY_INPUT`, etc.) are registered in `gadget_registry.py` as of 2026-05-22.

Registered gadgets use `REGISTRY_ID — name` titles via `graph_surgeon/taxonomy/display.py`.
Lookup: `graph-surgeon catalog --gadget ID`.
