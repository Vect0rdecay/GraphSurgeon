#!/usr/bin/env python3
"""RobustBench ONNX smoke test runner for GraphSurgeon CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PILOT_MODELS = ["Standard", "Wong2020Fast", "Engstrom2019Robustness"]

ALL_MODELS = [
    "Andriushchenko2020Understanding",
    "Carmon2019Unlabeled",
    "Engstrom2019Robustness",
    "Hendrycks2019Using",
    "Huang2020Self",
    "Rice2020Overfitting",
    "Sridhar2021Robust",
    "Standard",
    "Wang2020Improving",
    "Wong2020Fast",
    "Wu2020Adversarial_extra",
    "Zhang2019Theoretically",
]

DEFAULT_FIXTURE_ROOT = os.environ.get("GRAPH_SURGEON_FIXTURE_ROOT", "")


@dataclass
class CommandResult:
    model: str
    command: str
    exit_code: int
    elapsed_s: float
    error_summary: str = ""


@dataclass
class SmokeReport:
    results: list[CommandResult] = field(default_factory=list)

    def add(self, result: CommandResult) -> None:
        self.results.append(result)

    def to_markdown(self, title: str, commit: str = "") -> str:
        lines = [
            f"# {title}",
            "",
            f"**Fixture root:** `{os.environ.get('GRAPH_SURGEON_FIXTURE_ROOT', DEFAULT_FIXTURE_ROOT)}`",
        ]
        if commit:
            lines.append(f"**Commit:** `{commit}`")
        lines.extend(["", "| Model | Command | Exit | Time | Error summary |", "|-------|---------|------|------|---------------|"])
        for r in self.results:
            err = r.error_summary.replace("|", "\\|")
            lines.append(f"| {r.model} | {r.command} | {r.exit_code} | {r.elapsed_s:.2f}s | {err} |")
        fails = sum(1 for r in self.results if r.exit_code != 0)
        lines.extend(["", f"**Total:** {len(self.results)} commands, {fails} failures"])
        return "\n".join(lines) + "\n"


def _graph_surgeon_bin() -> str:
    root = Path(__file__).resolve().parent.parent
    return str(root / ".venv" / "bin" / "graph-surgeon")


def _run_cmd(model_name: str, label: str, args: list[str]) -> CommandResult:
    start = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    err = ""
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        err = " ".join(err[:3])
    return CommandResult(model_name, label, proc.returncode, elapsed, err)


def run_model_matrix(
    gs: str,
    fixture_root: Path,
    model_name: str,
    include_runnable: bool = False,
) -> list[CommandResult]:
    model_path = fixture_root / f"{model_name}.onnx"
    if not model_path.exists():
        return [CommandResult(model_name, "fixture check", 1, 0.0, f"Missing {model_path}")]

    results: list[CommandResult] = []
    cmds = [
        ("inspect", [gs, "inspect", str(model_path)]),
        ("topology", [gs, "topology", str(model_path)]),
        ("topology --json", [gs, "topology", str(model_path), "--json"]),
        ("flow", [gs, "flow", str(model_path)]),
        ("edit validate structural", [gs, "edit", "validate", str(model_path), "--level", "structural"]),
        ("edit validate loadable", [gs, "edit", "validate", str(model_path), "--level", "loadable"]),
    ]
    if include_runnable:
        cmds.append(("edit validate runnable", [gs, "edit", "validate", str(model_path), "--level", "runnable"]))
    motif_out = f"/tmp/gs_motifs_{model_name}.json"
    cmds.append((f"motifs -o {motif_out}", [gs, "motifs", str(model_path), "-o", motif_out]))

    for label, args in cmds:
        results.append(_run_cmd(model_name, label, args))
    return results


def resolve_models(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return ALL_MODELS
    names = [s.strip().replace(".onnx", "") for s in spec.split(",") if s.strip()]
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GraphSurgeon RobustBench smoke runner")
    parser.add_argument(
        "--models",
        default=",".join(PILOT_MODELS),
        help="Comma-separated model names, or 'all' (default: pilot trio)",
    )
    parser.add_argument(
        "--fixture-root",
        default=os.environ.get("GRAPH_SURGEON_FIXTURE_ROOT", DEFAULT_FIXTURE_ROOT),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Markdown report path (default: tests/output/smoke_report.md for all, pilot_report append otherwise)",
    )
    parser.add_argument("--runnable", action="store_true", help="Include runnable validation (Standard only by default)")
    args = parser.parse_args(argv)

    fixture_root = Path(args.fixture_root)
    models = resolve_models(args.models)
    gs = _graph_surgeon_bin()
    report = SmokeReport()

    for name in models:
        include_runnable = args.runnable or name == "Standard"
        report.results.extend(run_model_matrix(gs, fixture_root, name, include_runnable))

    commit = ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    title = "GraphSurgeon Smoke Report" if args.models.strip().lower() == "all" else "GraphSurgeon Pilot/Smoke Report"
    md = report.to_markdown(title, commit)

    out_path = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / "tests" / "output" / "smoke_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(md)
    print(f"Report written to {out_path}")

    return 1 if any(r.exit_code != 0 for r in report.results) else 0


if __name__ == "__main__":
    sys.exit(main())
