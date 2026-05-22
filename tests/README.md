# GraphSurgeon tests

## Unit tests

Run on every CI build (no external ONNX files required):

```bash
.venv/bin/python -m pytest tests/test_graph_surgeon.py -v
```

## Dependencies

GraphSurgeon is ONNX-only. Install with `[dev]` for onnxruntime (validate commands) and pytest:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

No PyTorch or CUDA toolkit is required.

Integration tests load RobustBench ONNX models from a directory outside this repo.

Set fixture root (required for integration tests):

```bash
export GRAPH_SURGEON_FIXTURE_ROOT=/path/to/onnx/fixtures
.venv/bin/python -m pytest tests/ -v -m integration
```

If fixtures are missing, integration tests are skipped.

See `fixtures_manifest.json` for the expected model filenames.
