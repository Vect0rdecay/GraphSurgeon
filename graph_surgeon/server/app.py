"""FastAPI app for serving the 3D viewer and providing a JSON API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph_surgeon.graph.surgeon import GraphSurgeon
from graph_surgeon.scene.builder import build_scene

VIEWER_DIST = Path(__file__).resolve().parent.parent.parent / "viewer" / "dist"

_surgeon = GraphSurgeon(verbose=False)
_model = None
_model_path: str = ""
_working_clone = None


class RemoveNodeRequest(BaseModel):
    node_name: str


def create_app(model_path: str) -> FastAPI:
    global _model, _model_path, _working_clone

    _model_path = model_path
    _model = _surgeon.load_model(model_path)
    _working_clone = _surgeon.clone_model(_model)

    app = FastAPI(title="GraphSurgeon 3D", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/scene")
    def get_scene(motifs: bool = True, weights: bool = False):
        scene = build_scene(_model_path, include_motifs=motifs, include_weights=weights)
        return JSONResponse(scene.to_dict())

    @app.get("/api/node/{node_id}")
    def get_node(node_id: str):
        from graph_surgeon.analysis.motifs import OPERATOR_REFERENCE_DB
        from graph_surgeon.reporting.sanitize import serialize_for_export

        node = _surgeon.get_node_by_name(_working_clone.graph, node_id)
        if not node:
            raise HTTPException(404, f"Node not found: {node_id}")

        op_ref = OPERATOR_REFERENCE_DB.get(node.op_type, {})
        safe_ref = serialize_for_export(op_ref) if op_ref else {}

        return JSONResponse({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {
                a.name: _attr_value(a) for a in node.attribute
            },
            "operator_info": safe_ref,
        })

    @app.get("/api/catalog/{catalog_id}")
    def get_catalog(catalog_id: str):
        from graph_surgeon.taxonomy.display import (
            format_catalog_chain,
            format_catalog_gadget,
        )

        if catalog_id.startswith("CHAIN-"):
            text = format_catalog_chain(catalog_id)
        else:
            text = format_catalog_gadget(catalog_id)
        return JSONResponse({"id": catalog_id, "text": text})

    @app.post("/api/edit/remove-node")
    def remove_node(req: RemoveNodeRequest):
        global _working_clone

        baseline = _surgeon.clone_model(_model)
        result = _surgeon.remove_node(_working_clone, req.node_name)

        if not result.success:
            raise HTTPException(400, result.message)

        diff = _surgeon.compare_graphs(baseline, _working_clone)

        validation = None
        try:
            from graph_surgeon.graph.validation import GraphValidationLevel
            vr = _surgeon.validate(_working_clone, level=GraphValidationLevel.STRUCTURAL)
            validation = {"valid": vr.valid, "level": vr.level.value, "errors": vr.errors, "warnings": vr.warnings}
        except Exception:
            pass

        scene = build_scene_from_model(_working_clone, _model_path)

        return JSONResponse({
            "surgery": {
                "success": result.success,
                "message": result.message,
                "nodes_removed": getattr(result, "nodes_removed", []),
                "edges_rewired": getattr(result, "edges_rewired", []),
            },
            "diff": {k: _safe_json(v) for k, v in diff.items()},
            "validation": validation,
            "scene": scene.to_dict(),
        })

    @app.get("/api/diff")
    def get_diff():
        diff = _surgeon.compare_graphs(_model, _working_clone)
        return JSONResponse({k: _safe_json(v) for k, v in diff.items()})

    @app.post("/api/reset")
    def reset():
        global _working_clone
        _working_clone = _surgeon.clone_model(_model)
        return JSONResponse({"status": "reset"})

    if VIEWER_DIST.exists():
        assets_dir = VIEWER_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/")
        def serve_index():
            return FileResponse(VIEWER_DIST / "index.html")

        @app.get("/{path:path}")
        def spa_fallback(path: str):
            file_path = VIEWER_DIST / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(VIEWER_DIST / "index.html")

    return app


def build_scene_from_model(model, original_path: str):
    """Build a scene from an in-memory model by saving to a temp file."""
    import tempfile
    import os
    from graph_surgeon.scene.builder import build_scene

    fd, tmp = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        _surgeon.save_model(model, tmp)
        return build_scene(tmp, include_motifs=True, include_weights=False)
    finally:
        os.unlink(tmp)


def _attr_value(attr) -> Any:
    if attr.type == 1:
        return attr.f
    if attr.type == 2:
        return attr.i
    if attr.type == 3:
        return attr.s.decode("utf-8", errors="replace")
    if attr.type == 6:
        return list(attr.floats)
    if attr.type == 7:
        return list(attr.ints)
    return str(attr)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)
