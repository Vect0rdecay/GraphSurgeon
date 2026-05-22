# GraphSurgeon tests

## Unit tests (shipped)

Run on every CI build (no external ONNX files required):

```bash
.venv/bin/python -m pytest tests/ -v
```

This runs `test_graph_surgeon.py`, `test_cli.py`, `test_display.py`, `test_paper_research.py`, and `test_deployment_motifs.py`.

## Local integration tests (not in public repo)

Integration and RobustBench smoke tests require external ONNX fixtures and are **gitignored** (`test_integration.py`, `test_robustbench_cli.py`, `fixtures_manifest.json`). Keep them locally for maintainer validation:

```bash
export GRAPH_SURGEON_FIXTURE_ROOT=/path/to/onnx/fixtures
.venv/bin/python -m pytest tests/test_integration.py tests/test_robustbench_cli.py -v
```

If fixtures are missing, those tests skip.
