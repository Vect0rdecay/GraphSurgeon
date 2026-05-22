# GraphSurgeon Tests

## Test tiers

| Tier | Location | When run |
|------|----------|----------|
| Unit | `tests/test_*.py` with generated tiny ONNX graphs | Every CI run |
| Integration | `@pytest.mark.integration` against RobustBench ONNX | Local dev with fixture root |

## Fixture root

Integration tests resolve ONNX models via `GRAPH_SURGEON_FIXTURE_ROOT`:

```bash
export GRAPH_SURGEON_FIXTURE_ROOT=/home/s0crates/nn_security_analyzer/robustbench_validation
cd /home/s0crates/graph-surgeon
.venv/bin/python -m pytest tests/ -v -m integration
```

If the fixture root is missing or a model file is absent, integration tests are skipped.

## CI expectations

Default CI runs unit tests only (no external ONNX required). See `fixtures_manifest.json` for the expected model list.

## Running all tests

```bash
.venv/bin/python -m pytest tests/ -v
```
