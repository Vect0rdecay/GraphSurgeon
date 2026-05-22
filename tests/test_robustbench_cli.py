"""Parametrized RobustBench CLI smoke tests (pilot models by default)."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from graph_surgeon.graph.surgeon import GraphSurgeon
from graph_surgeon.parsers.onnx_parser import ONNXGraphParser

PILOT_MODELS = ["Standard.onnx", "Wong2020Fast.onnx", "Engstrom2019Robustness.onnx"]

FIXTURE_ROOT = Path(
    os.environ.get(
        "GRAPH_SURGEON_FIXTURE_ROOT",
        "/home/s0crates/nn_security_analyzer/robustbench_validation",
    )
)

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
GS_BIN = ROOT / ".venv" / "bin" / "graph-surgeon"


def _model_names() -> list[str]:
    if os.environ.get("GRAPH_SURGEON_SMOKE_ALL") == "1":
        manifest = Path(__file__).parent / "fixtures_manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            return data.get("models", PILOT_MODELS)
    return PILOT_MODELS


def _require_fixture(model_file: str) -> Path:
    path = FIXTURE_ROOT / model_file
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    data_path = path.with_suffix(path.suffix + ".data")
    if not data_path.exists():
        pytest.skip(f"External data missing: {data_path}")
    return path


def _run_gs(args: list[str]) -> subprocess.CompletedProcess:
    if not VENV_PYTHON.is_file():
        pytest.skip("Project venv not found; run pip install -e .")
    return subprocess.run(
        [str(VENV_PYTHON), "-m", "graph_surgeon", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )


@pytest.mark.integration
@pytest.mark.parametrize("model_file", _model_names())
def test_parser_topology_matches_surgeon(model_file):
    path = _require_fixture(model_file)
    parser = ONNXGraphParser()
    graph = parser.parse_file(str(path))
    assert graph.topology is not None

    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(str(path))
    topo_direct = surgeon.get_graph_topology(model.graph)
    assert graph.topology.total_nodes == topo_direct.total_nodes
    assert graph.topology.max_depth == topo_direct.max_depth


@pytest.mark.integration
@pytest.mark.parametrize("model_file", _model_names())
def test_motifs_writes_json(model_file, tmp_path):
    path = _require_fixture(model_file)
    name = path.stem
    out = tmp_path / f"gs_motifs_{name}.json"
    proc = _run_gs(["motifs", str(path), "-o", str(out)])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.exists()
    data = json.loads(out.read_text())
    assert "structural_findings" in data or "vulnerabilities" in data
    findings = data.get("structural_findings") or data.get("vulnerabilities") or []
    assert isinstance(findings, list)
    assert "overall_risk_score" not in data


@pytest.mark.integration
@pytest.mark.parametrize("model_file", _model_names())
def test_edit_validate_loadable(model_file):
    path = _require_fixture(model_file)
    proc = _run_gs(["edit", "validate", str(path), "--level", "loadable"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "valid=True" in proc.stdout


@pytest.mark.integration
def test_motifs_writes_json_standard(robustbench_standard, tmp_path):
    """Focused regression for motifs JSON export on Standard."""
    out = tmp_path / "gs_motifs_Standard.json"
    proc = _run_gs(["motifs", str(robustbench_standard), "-o", str(out)])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text())
    findings = data.get("structural_findings") or data.get("vulnerabilities") or []
    assert isinstance(findings, list)
    assert "overall_risk_score" not in data
    if findings:
        assert "severity" not in findings[0]
    titles = [f.get("title", "") for f in findings]
    assert not any("Vulnerability" in t for t in titles)
    registry_ids = [f.get("registry_id") for f in findings if f.get("registry_id")]
    assert registry_ids, "expected at least one registry-backed finding on Standard"


@pytest.mark.integration
def test_patterns_output_uses_registry_ids(robustbench_standard):
    proc = _run_gs(["patterns", str(robustbench_standard)])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Vulnerability" not in proc.stdout
    assert "NORMALIZER" in proc.stdout or "GAP_FC_HEAD" in proc.stdout


@pytest.mark.integration
def test_catalog_gadget_lookup():
    proc = _run_gs(["catalog", "--gadget", "NORMALIZER"])
    assert proc.returncode == 0
    assert "Associated attack classes from literature:" in proc.stdout
    assert "BNAttack-2020" in proc.stdout or "66-AdvShadow-2022" in proc.stdout
