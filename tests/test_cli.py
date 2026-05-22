"""CLI entry-point tests (installed package in project venv)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
GS_BIN = ROOT / ".venv" / "bin" / "graph-surgeon"


def _require_install() -> None:
    if not VENV_PYTHON.is_file():
        pytest.fail(
            "Project venv missing. Run: cd graph-surgeon && python3 -m venv .venv "
            "&& .venv/bin/python -m pip install -e \".[dev]\""
        )


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    _require_install()
    # python -m graph_surgeon works whenever the package is installed in this venv
    return subprocess.run(
        [str(VENV_PYTHON), "-m", "graph_surgeon", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )


def test_cli_module_entry_point_exists():
    _require_install()
    proc = subprocess.run(
        [str(VENV_PYTHON), "-m", "graph_surgeon", "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "catalog" in proc.stdout
    assert "Examples:" in proc.stdout
    assert "graph-surgeon inspect model.onnx" in proc.stdout
    assert "graph-surgeon catalog --gadget GAP_FC_HEAD" in proc.stdout


def test_cli_catalog_help_examples():
    _require_install()
    proc = subprocess.run(
        [str(VENV_PYTHON), "-m", "graph_surgeon", "catalog", "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "--gadget" in proc.stdout
    assert "--chain" in proc.stdout
    assert "--coverage" in proc.stdout
    assert "Examples:" in proc.stdout
    assert "graph-surgeon catalog --coverage" in proc.stdout


def test_graph_surgeon_script_help_if_installed():
    if not GS_BIN.is_file():
        pytest.skip("Console script not installed")
    proc = subprocess.run(
        [str(GS_BIN), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Examples:" in proc.stdout
    assert "graph-surgeon motifs model.onnx" in proc.stdout


def test_catalog_default_re_index():
    proc = _run_cli(["catalog"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "GRAPHSURGEON CATALOG" in proc.stdout
    assert "Structural motifs (gadgets)" in proc.stdout
    assert "GAP_FC_HEAD" in proc.stdout
    assert "Compound chains" in proc.stdout
    assert "CHAIN-PATCH-ATTACK-SURFACE" in proc.stdout
    assert "Literature technique index" in proc.stdout
    assert "AML-ADV-001" in proc.stdout
    assert "graph-surgeon catalog --gadget GAP_FC_HEAD" in proc.stdout
    assert "ADVERSARIAL ML THREAT" not in proc.stdout
    assert "By Attack Goal" not in proc.stdout
    assert "By Access Level" not in proc.stdout


def test_catalog_coverage_subcommand():
    proc = _run_cli(["catalog", "--coverage"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Complete:" in proc.stdout
    assert "Missing:" in proc.stdout


def test_graph_surgeon_console_script_if_installed():
    """graph-surgeon script exists after pip install -e ."""
    if not GS_BIN.is_file():
        pytest.fail(
            f"Console script not found at {GS_BIN}. "
            "Run: .venv/bin/python -m pip install -e ."
        )
    proc = subprocess.run(
        [str(GS_BIN), "catalog", "--coverage"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Complete:" in proc.stdout
