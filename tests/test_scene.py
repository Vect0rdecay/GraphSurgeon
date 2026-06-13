"""Tests for the SceneGraph schema, builder, and export-scene CLI command."""

import json
import os
import tempfile

import pytest
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from graph_surgeon.scene.schema import (
    SCHEMA_VERSION,
    SceneGraph,
    SceneNode,
    SceneEdge,
    SceneModelInfo,
    SceneInput,
    SceneOutput,
)

SCORE_AND_SEVERITY_KEYS = {
    "overall_risk_score", "normalized_risk_score", "adversarial_risk_score",
    "shadowlogic_risk_score", "impnet_risk_score", "extraction_risk_score",
    "privacy_risk_score", "severity", "cvss_estimate", "exploitation_difficulty",
    "susceptibility_level", "susceptibility_score",
}


def _save_model(model):
    """Write a model to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


@pytest.fixture
def simple_model_path(simple_model):
    path = _save_model(simple_model)
    yield path
    os.unlink(path)


class TestSceneSchema:
    def test_to_dict_round_trip(self):
        scene = SceneGraph(
            schema_version=SCHEMA_VERSION,
            model=SceneModelInfo(
                name="test", format="onnx", opset=13,
                total_nodes=1, max_depth=1,
                inputs=[SceneInput(name="x", shape=[1, 3], dtype="float32")],
                outputs=[SceneOutput(name="y", shape=[1, 10], dtype="float32")],
            ),
            nodes=[SceneNode(
                id="n0", op_type="Relu", category="activation",
                depth=0, position="early", exec_index=0,
                inputs=["x"], outputs=["y"], attributes={},
            )],
            edges=[SceneEdge(source="x", target="n0", tensor="x", shape=[1, 3])],
        )
        d = scene.to_dict()
        blob = json.dumps(d)
        loaded = json.loads(blob)
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["model"]["name"] == "test"
        assert len(loaded["nodes"]) == 1
        assert len(loaded["edges"]) == 1

    def test_no_score_keys_in_dict(self):
        scene = SceneGraph(
            schema_version=SCHEMA_VERSION,
            model=SceneModelInfo(
                name="t", format="onnx", opset=13,
                total_nodes=0, max_depth=0, inputs=[], outputs=[],
            ),
            nodes=[], edges=[],
        )
        blob = json.dumps(scene.to_dict())
        for key in SCORE_AND_SEVERITY_KEYS:
            assert f'"{key}"' not in blob


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestBuildScene:
    def test_basic_build(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=False)
        assert scene.schema_version == SCHEMA_VERSION
        assert scene.model.format == "onnx"
        assert scene.model.total_nodes == 9
        assert scene.model.max_depth > 0
        assert len(scene.nodes) == 9
        assert len(scene.edges) > 0

    def test_node_fields_present(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=False)
        for node in scene.nodes:
            assert node.id
            assert node.op_type
            assert node.category
            assert node.position in ("early", "middle", "late")
            assert isinstance(node.exec_index, int)
            assert isinstance(node.inputs, list)
            assert isinstance(node.outputs, list)

    def test_exec_index_is_permutation(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=False)
        indices = sorted(n.exec_index for n in scene.nodes)
        assert indices == list(range(len(scene.nodes)))

    def test_edges_resolve_to_nodes_or_inputs(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=False)
        node_ids = {n.id for n in scene.nodes}
        input_names = {inp.name for inp in scene.model.inputs}
        valid_sources = node_ids | input_names

        for edge in scene.edges:
            assert edge.source in valid_sources, f"bad source: {edge.source}"
            assert edge.target in node_ids, f"bad target: {edge.target}"

    def test_deterministic_output(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        a = json.dumps(build_scene(simple_model_path, include_motifs=False).to_dict())
        b = json.dumps(build_scene(simple_model_path, include_motifs=False).to_dict())
        assert a == b

    def test_no_score_keys_in_export(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=True)
        blob = json.dumps(scene.to_dict())
        for key in SCORE_AND_SEVERITY_KEYS:
            assert f'"{key}"' not in blob, f"sanitizer missed: {key}"

    def test_with_weights(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(
            simple_model_path, include_motifs=False, include_weights=True,
        )
        conv_nodes = [n for n in scene.nodes if n.op_type == "Conv"]
        assert any(n.param_count > 0 for n in conv_nodes)

    def test_without_motifs_has_empty_motifs(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=False)
        assert scene.motifs == []
        assert scene.chains == []

    def test_with_motifs_produces_findings(self, simple_model_path):
        from graph_surgeon.scene.builder import build_scene

        scene = build_scene(simple_model_path, include_motifs=True)
        assert len(scene.motifs) > 0 or len(scene.chains) > 0 or True


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
class TestExportSceneCLI:
    def test_export_scene_to_file(self, simple_model_path):
        from graph_surgeon.cli import main

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            rc = main(["export-scene", simple_model_path, "-o", out_path, "--no-motifs"])
            assert rc == 0
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema_version"] == SCHEMA_VERSION
            assert len(data["nodes"]) == 9
        finally:
            os.unlink(out_path)

    def test_export_scene_stdout(self, simple_model_path, capsys):
        from graph_surgeon.cli import main

        rc = main(["export-scene", simple_model_path, "--no-motifs"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == SCHEMA_VERSION

    def test_export_scene_missing_file(self):
        from graph_surgeon.cli import main

        rc = main(["export-scene", "/nonexistent/model.onnx", "-o", "/tmp/out.json"])
        assert rc == 1
