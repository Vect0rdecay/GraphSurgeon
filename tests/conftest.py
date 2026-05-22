"""Pytest configuration, fixtures, and test node factories."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

FIXTURE_ROOT = Path(
    os.environ.get(
        "GRAPH_SURGEON_FIXTURE_ROOT",
        "/home/s0crates/nn_security_analyzer/robustbench_validation",
    )
)

_manifest_path = Path(__file__).parent / "fixtures_manifest.json"
if _manifest_path.exists():
    _manifest = json.loads(_manifest_path.read_text())
    ROBUSTBENCH_MODELS = [m["filename"] for m in _manifest.get("models", [])]
else:
    ROBUSTBENCH_MODELS = ["Standard.onnx"]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires external ONNX fixtures")


@pytest.fixture
def fixture_root():
    return FIXTURE_ROOT


@pytest.fixture
def robustbench_standard():
    path = FIXTURE_ROOT / "Standard.onnx"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}. Set GRAPH_SURGEON_FIXTURE_ROOT.")
    return path


# --- Tiny ONNX fixtures (from Carcinoma tests) ---


@pytest.fixture
def simple_model():
    if not ONNX_AVAILABLE:
        pytest.skip("ONNX not available")

    conv1_weights = numpy_helper.from_array(
        np.random.randn(64, 3, 3, 3).astype(np.float32), "conv1_weights"
    )
    conv2_weights = numpy_helper.from_array(
        np.random.randn(128, 64, 3, 3).astype(np.float32), "conv2_weights"
    )
    conv3_weights = numpy_helper.from_array(
        np.random.randn(256, 128, 3, 3).astype(np.float32), "conv3_weights"
    )

    nodes = [
        helper.make_node("Conv", ["input", "conv1_weights"], ["conv1_out"],
                         name="conv1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv1_out"], ["relu1_out"], name="relu1"),
        helper.make_node("Conv", ["relu1_out", "conv2_weights"], ["conv2_out"],
                         name="conv2", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv2_out"], ["relu2_out"], name="relu2"),
        helper.make_node("MaxPool", ["relu2_out"], ["maxpool_out"],
                         name="maxpool", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["maxpool_out", "conv3_weights"], ["conv3_out"],
                         name="conv3", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv3_out"], ["relu3_out"], name="relu3"),
        helper.make_node("GlobalAveragePool", ["relu3_out"], ["gap_out"], name="gap"),
        helper.make_node("Flatten", ["gap_out"], ["output"], name="flatten"),
    ]

    graph = helper.make_graph(
        nodes,
        "test_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256])],
        [conv1_weights, conv2_weights, conv3_weights],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


@pytest.fixture
def model_with_concat():
    if not ONNX_AVAILABLE:
        pytest.skip("ONNX not available")

    conv1_weights = numpy_helper.from_array(
        np.random.randn(32, 3, 3, 3).astype(np.float32), "conv1_weights"
    )
    conv2_weights = numpy_helper.from_array(
        np.random.randn(32, 3, 3, 3).astype(np.float32), "conv2_weights"
    )

    nodes = [
        helper.make_node("Conv", ["input", "conv1_weights"], ["conv1_out"],
                         name="conv1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("Conv", ["input", "conv2_weights"], ["conv2_out"],
                         name="conv2", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("Concat", ["conv1_out", "conv2_out"], ["concat_out"],
                         name="concat", axis=1),
        helper.make_node("GlobalAveragePool", ["concat_out"], ["output"], name="gap"),
    ]

    graph = helper.make_graph(
        nodes,
        "concat_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 1, 1])],
        [conv1_weights, conv2_weights],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


@pytest.fixture
def model_with_avgpool():
    if not ONNX_AVAILABLE:
        pytest.skip("ONNX not available")

    conv_weights = numpy_helper.from_array(
        np.random.randn(64, 3, 3, 3).astype(np.float32), "conv_weights"
    )

    nodes = [
        helper.make_node("Conv", ["input", "conv_weights"], ["conv_out"],
                         name="conv", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("MaxPool", ["conv_out"], ["maxpool_out"],
                         name="maxpool", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("AveragePool", ["maxpool_out"], ["avgpool_out"],
                         name="avgpool", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("GlobalAveragePool", ["avgpool_out"], ["output"], name="gap"),
    ]

    graph = helper.make_graph(
        nodes,
        "pool_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 1, 1])],
        [conv_weights],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


# Test-only node factories (Carcinoma create_* helpers; not public GraphSurgeon API)


def create_maxpool_node(name, input_name, output_name, kernel_shape=(2, 2), strides=(2, 2), pads=None):
    if not ONNX_AVAILABLE:
        raise ImportError("onnx required")
    attrs = {"kernel_shape": list(kernel_shape), "strides": list(strides)}
    if pads is not None:
        attrs["pads"] = list(pads)
    return helper.make_node("MaxPool", [input_name], [output_name], name=name, **attrs)


def create_avgpool_node(name, input_name, output_name, kernel_shape=(2, 2), strides=(2, 2)):
    if not ONNX_AVAILABLE:
        raise ImportError("onnx required")
    return helper.make_node(
        "AveragePool", [input_name], [output_name], name=name,
        kernel_shape=list(kernel_shape), strides=list(strides),
    )
