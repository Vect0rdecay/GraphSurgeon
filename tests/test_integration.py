"""Integration tests using external RobustBench ONNX fixtures."""

import pytest

from graph_surgeon.graph.surgeon import GraphSurgeon
from graph_surgeon.parsers.onnx_parser import ONNXGraphParser


@pytest.mark.integration
def test_topology_on_standard(robustbench_standard):
    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(str(robustbench_standard))
    topo = surgeon.get_graph_topology(model.graph)
    assert topo.total_nodes > 0
    assert topo.max_depth >= 0
    assert len(topo.execution_order) == topo.total_nodes


@pytest.mark.integration
def test_parser_topology_matches_surgeon(robustbench_standard):
    parser = ONNXGraphParser()
    graph = parser.parse_file(str(robustbench_standard))
    assert graph.topology is not None
    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(str(robustbench_standard))
    topo_direct = surgeon.get_graph_topology(model.graph)
    assert graph.topology.total_nodes == topo_direct.total_nodes
    assert graph.topology.max_depth == topo_direct.max_depth
