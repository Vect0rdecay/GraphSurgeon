# Internal research logs (not used by catalog)

# Attack Research Notes (GraphSurgeon corpus)

Per-paper adversarial ML analysis for ONNX reverse engineering. Each `### [id]` section maps
literature to structural motifs (attack landscape from graph structure, not exploitability).

Authoring spec: `RESEARCH_NOTE_TEMPLATE.md`. Coverage: `research_coverage.json`.

---

## Light-based attacks (shared ONNX mechanism)

Papers 40, 43, 48, 52, 57, 65, 66, 80, 81, 82 exploit illumination, projection, or weather-like perturbations.
The ONNX graph vulnerability is the same as patch attacks: ALIASING_DOWNSAMPLE folds high-frequency lighting edges,
NORMALIZER amplifies distribution shift under changed illumination, NO_SPATIAL_ATTENTION cannot suppress localized
bright/dark regions, and GAP_FC_HEAD aggregates spatial perturbations into the classifier head.

---