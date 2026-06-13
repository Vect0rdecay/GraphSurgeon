"""Tests for the FastAPI server endpoints."""

import json
import os
import tempfile

import pytest

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def _save_model(model):
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


@pytest.fixture
def server_client(simple_model):
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available (install [viz] extra)")

    path = _save_model(simple_model)
    try:
        from graph_surgeon.server.app import create_app
        app = create_app(path)
        with TestClient(app) as client:
            yield client
    finally:
        os.unlink(path)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestServerAPI:
    def test_get_scene(self, server_client):
        resp = server_client.get("/api/scene?motifs=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "1.1"
        assert len(data["nodes"]) == 9

    def test_get_node(self, server_client):
        resp = server_client.get("/api/node/conv1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["op_type"] == "Conv"
        assert data["name"] == "conv1"

    def test_get_node_not_found(self, server_client):
        resp = server_client.get("/api/node/nonexistent")
        assert resp.status_code == 404

    def test_remove_node(self, server_client):
        resp = server_client.post(
            "/api/edit/remove-node",
            json={"node_name": "relu3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["surgery"]["success"] is True
        assert data["scene"]["schema_version"] == "1.1"

    def test_diff_after_edit(self, server_client):
        server_client.post(
            "/api/edit/remove-node",
            json={"node_name": "relu1"},
        )
        resp = server_client.get("/api/diff")
        assert resp.status_code == 200
        diff = resp.json()
        assert "nodes_removed" in diff or "summary" in diff

    def test_reset(self, server_client):
        server_client.post(
            "/api/edit/remove-node",
            json={"node_name": "relu1"},
        )
        resp = server_client.post("/api/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"

        scene = server_client.get("/api/scene?motifs=false").json()
        assert len(scene["nodes"]) == 9

    def test_catalog_gadget(self, server_client):
        resp = server_client.get("/api/catalog/GAP_FC_HEAD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "GAP_FC_HEAD"
