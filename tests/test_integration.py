"""Integration tests using RobustBench ONNX fixtures."""

import pytest

from graph_surgeon import GraphSurgeon
from graph_surgeon.parsers.onnx_parser import ONNXGraphParser


@pytest.mark.integration
def test_topology_on_standard_model(robustbench_standard):
    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(str(robustbench_standard))
    topo = surgeon.get_graph_topology(model.graph)

    assert topo.total_nodes > 0
    assert topo.max_depth > 0
    assert "Conv" in topo.by_op_type or len(topo.by_op_type) > 0


@pytest.mark.integration
def test_parser_topology_matches_surgeon(robustbench_standard):
    parser = ONNXGraphParser()
    graph = parser.parse_file(str(robustbench_standard))

    assert graph.topology is not None
    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(str(robustbench_standard))
    standalone = surgeon.get_graph_topology(model.graph)

    assert graph.topology.total_nodes == standalone.total_nodes
    assert graph.topology.max_depth == standalone.max_depth


@pytest.mark.integration
def test_motifs_scan_runs(robustbench_standard):
    from graph_surgeon.parsers.onnx_parser import analyze_model_motifs

    report = analyze_model_motifs(str(robustbench_standard), verbose=False)
    assert report is not None
    assert hasattr(report, "executive_summary")
