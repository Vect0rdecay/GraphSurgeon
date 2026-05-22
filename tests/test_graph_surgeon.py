"""
Unit tests for GraphSurgeon - the core ONNX graph manipulation engine.

Run with: pytest tests/test_graph_surgeon.py -v
"""

import pytest

from graph_surgeon.graph.surgeon import GraphSurgeon
from graph_surgeon.graph.validation import GraphValidationLevel as ValidationLevel

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TestFindNodesByType:
    """Tests for find_nodes_by_type() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_conv_nodes(self, simple_model):
        """Should find all Conv nodes in the model."""
        surgeon = GraphSurgeon(verbose=False)
        conv_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'Conv')
        
        assert len(conv_nodes) == 3
        assert all(node.op_type == 'Conv' for node in conv_nodes)
        
        # Verify specific names
        conv_names = {node.name for node in conv_nodes}
        assert conv_names == {'conv1', 'conv2', 'conv3'}
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_relu_nodes(self, simple_model):
        """Should find all Relu nodes."""
        surgeon = GraphSurgeon(verbose=False)
        relu_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'Relu')
        
        assert len(relu_nodes) == 3
        assert all(node.op_type == 'Relu' for node in relu_nodes)
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_maxpool_nodes(self, simple_model):
        """Should find MaxPool nodes."""
        surgeon = GraphSurgeon(verbose=False)
        maxpool_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'MaxPool')
        
        assert len(maxpool_nodes) == 1
        assert maxpool_nodes[0].name == 'maxpool'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_nonexistent_type(self, simple_model):
        """Should return empty list for non-existent node types."""
        surgeon = GraphSurgeon(verbose=False)
        bn_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'BatchNormalization')
        
        assert len(bn_nodes) == 0
        assert isinstance(bn_nodes, list)
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_concat_nodes(self, model_with_concat):
        """Should find Concat nodes in a model with concatenation."""
        surgeon = GraphSurgeon(verbose=False)
        concat_nodes = surgeon.find_nodes_by_type(model_with_concat.graph, 'Concat')
        
        assert len(concat_nodes) == 1
        assert concat_nodes[0].name == 'concat'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_global_average_pool(self, simple_model):
        """Should find GlobalAveragePool nodes."""
        surgeon = GraphSurgeon(verbose=False)
        gap_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'GlobalAveragePool')
        
        assert len(gap_nodes) == 1
        assert gap_nodes[0].name == 'gap'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_case_sensitive(self, simple_model):
        """Node type search should be case-sensitive."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Correct case
        conv_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'Conv')
        assert len(conv_nodes) == 3
        
        # Wrong case - should find nothing
        wrong_case = surgeon.find_nodes_by_type(simple_model.graph, 'conv')
        assert len(wrong_case) == 0
        
        wrong_case2 = surgeon.find_nodes_by_type(simple_model.graph, 'CONV')
        assert len(wrong_case2) == 0


class TestFindNodesByAttribute:
    """Tests for find_nodes_by_attribute() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_stride2_convs(self, simple_model):
        """Should find Conv nodes with stride=2."""
        surgeon = GraphSurgeon(verbose=False)
        
        stride2_convs = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'strides', [2, 2], op_type='Conv'
        )
        
        assert len(stride2_convs) == 1
        assert stride2_convs[0].name == 'conv2'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_stride1_convs(self, simple_model):
        """Should find Conv nodes with stride=1."""
        surgeon = GraphSurgeon(verbose=False)
        
        stride1_convs = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'strides', [1, 1], op_type='Conv'
        )
        
        assert len(stride1_convs) == 2
        names = {node.name for node in stride1_convs}
        assert names == {'conv1', 'conv3'}
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_by_kernel_shape(self, simple_model):
        """Should find nodes by kernel_shape attribute."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Find all nodes with 3x3 kernels
        kernel3_nodes = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'kernel_shape', [3, 3]
        )
        
        # All 3 Conv nodes have 3x3 kernels
        assert len(kernel3_nodes) == 3
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_by_kernel_2x2(self, simple_model):
        """Should find MaxPool with 2x2 kernel."""
        surgeon = GraphSurgeon(verbose=False)
        
        kernel2_nodes = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'kernel_shape', [2, 2]
        )
        
        assert len(kernel2_nodes) == 1
        assert kernel2_nodes[0].name == 'maxpool'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_without_op_type_filter(self, simple_model):
        """Should search all nodes when op_type is None."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Find all nodes with stride [2, 2] (Conv + MaxPool)
        stride2_all = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'strides', [2, 2]
        )
        
        assert len(stride2_all) == 2  # conv2 + maxpool
        names = {node.name for node in stride2_all}
        assert names == {'conv2', 'maxpool'}
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_nonexistent_attribute(self, simple_model):
        """Should return empty list for non-existent attribute."""
        surgeon = GraphSurgeon(verbose=False)
        
        result = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'nonexistent_attr', 'value'
        )
        
        assert len(result) == 0
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_nonmatching_value(self, simple_model):
        """Should return empty list when value doesn't match."""
        surgeon = GraphSurgeon(verbose=False)
        
        result = surgeon.find_nodes_by_attribute(
            simple_model.graph, 'strides', [5, 5], op_type='Conv'
        )
        
        assert len(result) == 0
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_concat_axis(self, model_with_concat):
        """Should find Concat by axis attribute."""
        surgeon = GraphSurgeon(verbose=False)
        
        result = surgeon.find_nodes_by_attribute(
            model_with_concat.graph, 'axis', 1, op_type='Concat'
        )
        
        assert len(result) == 1
        assert result[0].name == 'concat'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_find_avgpool_vs_maxpool(self, model_with_avgpool):
        """Should correctly distinguish between pool types."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Both have kernel_shape [2, 2]
        all_pools = surgeon.find_nodes_by_attribute(
            model_with_avgpool.graph, 'kernel_shape', [2, 2]
        )
        assert len(all_pools) == 2
        
        # Filter by op_type
        maxpools = surgeon.find_nodes_by_attribute(
            model_with_avgpool.graph, 'kernel_shape', [2, 2], op_type='MaxPool'
        )
        assert len(maxpools) == 1
        assert maxpools[0].name == 'maxpool'
        
        avgpools = surgeon.find_nodes_by_attribute(
            model_with_avgpool.graph, 'kernel_shape', [2, 2], op_type='AveragePool'
        )
        assert len(avgpools) == 1
        assert avgpools[0].name == 'avgpool'


class TestGetAttributeValue:
    """Tests for _get_attribute_value() helper method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_extract_int_list(self, simple_model):
        """Should correctly extract integer list attributes."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv_nodes = surgeon.find_nodes_by_type(simple_model.graph, 'Conv')
        conv1 = [n for n in conv_nodes if n.name == 'conv1'][0]
        
        for attr in conv1.attribute:
            if attr.name == 'strides':
                value = surgeon._get_attribute_value(attr)
                assert value == [1, 1]
                break
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_extract_single_int(self, model_with_concat):
        """Should correctly extract single integer attributes."""
        surgeon = GraphSurgeon(verbose=False)
        
        concat_nodes = surgeon.find_nodes_by_type(model_with_concat.graph, 'Concat')
        concat = concat_nodes[0]
        
        for attr in concat.attribute:
            if attr.name == 'axis':
                value = surgeon._get_attribute_value(attr)
                assert value == 1
                break


class TestValuesEqual:
    """Tests for _values_equal() comparison method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_list_equality(self):
        """Should correctly compare lists."""
        surgeon = GraphSurgeon(verbose=False)
        
        assert surgeon._values_equal([1, 1], [1, 1]) is True
        assert surgeon._values_equal([1, 2], [1, 2]) is True
        assert surgeon._values_equal([1, 1], [2, 2]) is False
        assert surgeon._values_equal([1], [1, 1]) is False
        assert surgeon._values_equal([], []) is True
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_scalar_equality(self):
        """Should correctly compare scalars."""
        surgeon = GraphSurgeon(verbose=False)
        
        assert surgeon._values_equal(1, 1) is True
        assert surgeon._values_equal(1, 2) is False
        assert surgeon._values_equal(1.0, 1.0) is True
        assert surgeon._values_equal("test", "test") is True


class TestGetGraphTopology:
    """Tests for get_graph_topology() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_total_nodes(self, simple_model):
        """Should count total nodes correctly."""
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        # simple_model has 9 nodes: conv1, relu1, conv2, relu2, maxpool, conv3, relu3, gap, flatten
        assert topology.total_nodes == 9
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_by_op_type(self, simple_model):
        """Should group nodes by operation type."""
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        assert 'Conv' in topology.by_op_type
        assert len(topology.by_op_type['Conv']) == 3
        
        assert 'Relu' in topology.by_op_type
        assert len(topology.by_op_type['Relu']) == 3
        
        assert 'MaxPool' in topology.by_op_type
        assert len(topology.by_op_type['MaxPool']) == 1
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_positions(self, simple_model):
        """Should classify nodes into early/middle/late positions."""
        from graph_surgeon.graph.surgeon import LayerPosition
        
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        # All three positions should have nodes
        total_positioned = (
            len(topology.by_position[LayerPosition.EARLY]) +
            len(topology.by_position[LayerPosition.MIDDLE]) +
            len(topology.by_position[LayerPosition.LATE])
        )
        assert total_positioned == topology.total_nodes
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_early_layers(self, simple_model):
        """conv1 and relu1 should be in early layers."""
        from graph_surgeon.graph.surgeon import LayerPosition
        
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        early_nodes = topology.by_position[LayerPosition.EARLY]
        # conv1 (depth 0) should definitely be early
        assert 'conv1' in early_nodes
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_late_layers(self, simple_model):
        """gap and flatten should be in late layers."""
        from graph_surgeon.graph.surgeon import LayerPosition
        
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        late_nodes = topology.by_position[LayerPosition.LATE]
        # flatten (last node) should be late
        assert 'flatten' in late_nodes
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_execution_order(self, simple_model):
        """Execution order should be topologically sorted."""
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        # First node should have lowest depth, last should have highest
        first_node = topology.nodes[topology.execution_order[0]]
        last_node = topology.nodes[topology.execution_order[-1]]
        
        assert first_node.depth <= last_node.depth
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_node_info(self, simple_model):
        """Should have complete info for each node."""
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        conv1_info = topology.nodes['conv1']
        assert conv1_info.name == 'conv1'
        assert conv1_info.op_type == 'Conv'
        assert conv1_info.depth == 0  # First layer after input
        assert 'input' in conv1_info.inputs
        assert 'conv1_out' in conv1_info.outputs
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_topology_max_depth(self, simple_model):
        """max_depth should reflect the deepest node."""
        surgeon = GraphSurgeon(verbose=False)
        topology = surgeon.get_graph_topology(simple_model.graph)
        
        # Flatten is the last node, its depth is max_depth
        assert topology.max_depth == topology.nodes['flatten'].depth


class TestCloneNode:
    """Tests for clone_node() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_basic_clone(self, simple_model):
        """Should create a deep copy of a node."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv1 = surgeon.get_node_by_name(simple_model.graph, 'conv1')
        cloned = surgeon.clone_node(conv1)
        
        # Should have same attributes
        assert cloned.op_type == conv1.op_type
        assert cloned.name == conv1.name
        assert list(cloned.input) == list(conv1.input)
        assert list(cloned.output) == list(conv1.output)
        
        # Should be a different object
        assert cloned is not conv1
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_clone_with_new_name(self, simple_model):
        """Should allow renaming the cloned node."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv1 = surgeon.get_node_by_name(simple_model.graph, 'conv1')
        cloned = surgeon.clone_node(conv1, new_name='conv1_copy')
        
        assert cloned.name == 'conv1_copy'
        assert conv1.name == 'conv1'  # Original unchanged
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_clone_with_new_inputs(self, simple_model):
        """Should allow changing inputs."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv1 = surgeon.get_node_by_name(simple_model.graph, 'conv1')
        cloned = surgeon.clone_node(conv1, new_inputs=['new_input', 'new_weights'])
        
        assert list(cloned.input) == ['new_input', 'new_weights']
        assert 'input' in list(conv1.input)  # Original unchanged
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_clone_with_new_outputs(self, simple_model):
        """Should allow changing outputs."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv1 = surgeon.get_node_by_name(simple_model.graph, 'conv1')
        cloned = surgeon.clone_node(conv1, new_outputs=['new_output'])
        
        assert list(cloned.output) == ['new_output']
        assert 'conv1_out' in list(conv1.output)  # Original unchanged
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_clone_preserves_attributes(self, simple_model):
        """Cloned node should preserve all attributes."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv1 = surgeon.get_node_by_name(simple_model.graph, 'conv1')
        cloned = surgeon.clone_node(conv1, new_name='conv1_copy')
        
        # Check that attributes are preserved
        original_attrs = {attr.name: attr for attr in conv1.attribute}
        cloned_attrs = {attr.name: attr for attr in cloned.attribute}
        
        assert set(original_attrs.keys()) == set(cloned_attrs.keys())
        
        # Check specific attribute values
        for name in original_attrs:
            orig_val = surgeon._get_attribute_value(original_attrs[name])
            clone_val = surgeon._get_attribute_value(cloned_attrs[name])
            assert orig_val == clone_val


class TestGetEarlyLateLayers:
    """Tests for get_early_layers() and get_late_layers() methods."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_early_layers(self, simple_model):
        """Should return early layers."""
        surgeon = GraphSurgeon(verbose=False)
        
        early = surgeon.get_early_layers(simple_model.graph)
        early_names = {n.name for n in early}
        
        # conv1 should be in early layers
        assert 'conv1' in early_names
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_early_layers_with_filter(self, simple_model):
        """Should filter early layers by op_type."""
        surgeon = GraphSurgeon(verbose=False)
        
        early_convs = surgeon.get_early_layers(simple_model.graph, op_type='Conv')
        
        for node in early_convs:
            assert node.op_type == 'Conv'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_late_layers(self, simple_model):
        """Should return late layers."""
        surgeon = GraphSurgeon(verbose=False)
        
        late = surgeon.get_late_layers(simple_model.graph)
        late_names = {n.name for n in late}
        
        # flatten (the last node) should be in late layers
        assert 'flatten' in late_names
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_late_layers_with_filter(self, simple_model):
        """Should filter late layers by op_type."""
        surgeon = GraphSurgeon(verbose=False)
        
        late_gap = surgeon.get_late_layers(simple_model.graph, op_type='GlobalAveragePool')
        
        # There's one GAP node and it should be late
        assert len(late_gap) <= 1  # May be 0 or 1 depending on threshold
        for node in late_gap:
            assert node.op_type == 'GlobalAveragePool'


class TestShapeInference:
    """Tests for shape inference methods."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_infer_shapes(self, simple_model):
        """Should run shape inference on model."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Run shape inference
        model_with_shapes = surgeon.infer_shapes(simple_model)
        
        # Model should still be valid
        assert model_with_shapes is not None
        
        # Should have value_info for intermediate tensors
        # (shape inference adds these)
        assert len(model_with_shapes.graph.value_info) > 0
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_tensor_shape_input(self, simple_model):
        """Should get shape for graph inputs."""
        surgeon = GraphSurgeon(verbose=False)
        
        shape = surgeon.get_tensor_shape(simple_model, 'input')
        assert shape == [1, 3, 224, 224]
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_tensor_shape_intermediate(self, simple_model):
        """Should get shape for intermediate tensors after inference."""
        surgeon = GraphSurgeon(verbose=False)
        
        model = surgeon.infer_shapes(simple_model)
        
        # conv1_out should have shape after shape inference
        shape = surgeon.get_tensor_shape(model, 'conv1_out')
        assert shape is not None
        # Conv1: 3->64 channels, same spatial (padding=1, stride=1, kernel=3)
        assert shape[0] == 1  # batch
        assert shape[1] == 64  # channels
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_tensor_shape_unknown(self, simple_model):
        """Should return None for unknown tensors."""
        surgeon = GraphSurgeon(verbose=False)
        
        shape = surgeon.get_tensor_shape(simple_model, 'nonexistent_tensor')
        assert shape is None
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_check_shape_compatibility_same(self, simple_model):
        """Compatible shapes should return True."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Input should be compatible with itself
        compatible, msg = surgeon.check_shape_compatibility(
            simple_model, 'input', 'input'
        )
        assert compatible is True
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_check_shape_compatibility_unknown(self, simple_model):
        """Unknown tensor should return False with message."""
        surgeon = GraphSurgeon(verbose=False)
        
        compatible, msg = surgeon.check_shape_compatibility(
            simple_model, 'input', 'nonexistent'
        )
        assert compatible is False
        assert 'unknown' in msg.lower()


class TestMetadata:
    """Tests for metadata methods."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_add_and_get_metadata(self, simple_model):
        """Should add and retrieve metadata."""
        surgeon = GraphSurgeon(verbose=False)
        
        surgeon.add_metadata(simple_model, 'test_key', 'test_value')
        
        value = surgeon.get_metadata(simple_model, 'test_key')
        assert value == 'test_value'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_nonexistent_metadata(self, simple_model):
        """Should return None for nonexistent metadata."""
        surgeon = GraphSurgeon(verbose=False)
        
        value = surgeon.get_metadata(simple_model, 'nonexistent_key')
        assert value is None
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_record_graft(self, simple_model):
        """Should record graft in metadata."""
        surgeon = GraphSurgeon(verbose=False)
        
        surgeon.record_graft(
            simple_model, 
            'INSERT_MAXPOOL', 
            'conv1_out',
            details={'kernel': [2, 2]}
        )
        
        history = surgeon.get_graft_history(simple_model)
        assert len(history) == 1
        assert history[0]['graft_type'] == 'INSERT_MAXPOOL'
        assert history[0]['target_node'] == 'conv1_out'
        assert history[0]['details']['kernel'] == [2, 2]
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_record_multiple_grafts(self, simple_model):
        """Should record multiple grafts in order."""
        surgeon = GraphSurgeon(verbose=False)
        
        surgeon.record_graft(simple_model, 'GRAFT_A', 'node1')
        surgeon.record_graft(simple_model, 'GRAFT_B', 'node2')
        surgeon.record_graft(simple_model, 'GRAFT_C', 'node3')
        
        history = surgeon.get_graft_history(simple_model)
        assert len(history) == 3
        assert history[0]['graft_type'] == 'GRAFT_A'
        assert history[1]['graft_type'] == 'GRAFT_B'
        assert history[2]['graft_type'] == 'GRAFT_C'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_graft_history_empty(self, simple_model):
        """Should return empty list for model with no grafts."""
        surgeon = GraphSurgeon(verbose=False)
        
        history = surgeon.get_graft_history(simple_model)
        assert history == []
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_graft_has_timestamp(self, simple_model):
        """Graft records should have timestamp."""
        surgeon = GraphSurgeon(verbose=False)
        
        surgeon.record_graft(simple_model, 'TEST_GRAFT', 'test_node')
        
        history = surgeon.get_graft_history(simple_model)
        assert 'timestamp' in history[0]
        assert len(history[0]['timestamp']) > 0


class TestExistingMethods:
    """Tests for pre-existing GraphSurgeon methods to ensure they still work."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_node_by_name(self, simple_model):
        """get_node_by_name should still work."""
        surgeon = GraphSurgeon(verbose=False)
        
        conv1 = surgeon.get_node_by_name(simple_model.graph, 'conv1')
        assert conv1 is not None
        assert conv1.op_type == 'Conv'
        
        nonexistent = surgeon.get_node_by_name(simple_model.graph, 'nonexistent')
        assert nonexistent is None
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_node_by_output(self, simple_model):
        """get_node_by_output should still work."""
        surgeon = GraphSurgeon(verbose=False)
        
        node = surgeon.get_node_by_output(simple_model.graph, 'conv1_out')
        assert node is not None
        assert node.name == 'conv1'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_get_node_consumers(self, simple_model):
        """get_node_consumers should still work."""
        surgeon = GraphSurgeon(verbose=False)
        
        consumers = surgeon.get_node_consumers(simple_model.graph, 'conv1_out')
        assert len(consumers) == 1
        assert consumers[0].name == 'relu1'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_validate_structural(self, simple_model):
        """validate() at STRUCTURAL level should pass for valid model."""
        surgeon = GraphSurgeon(verbose=False)
        
        result = surgeon.validate(simple_model, level=ValidationLevel.STRUCTURAL)
        assert result.valid is True
        assert result.level == ValidationLevel.STRUCTURAL


class TestInsertNodeBefore:
    """Tests for insert_node_before() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_insert_before_conv(self, simple_model):
        """Should insert a node before a target node."""
        from graph_surgeon.graph.surgeon import create_maxpool_node
        
        surgeon = GraphSurgeon(verbose=False)
        
        # Create a MaxPool to insert before conv2
        # conv2's input is relu1_out
        maxpool = create_maxpool_node(
            'inserted_pool', 'relu1_out', 'inserted_pool_out',
            kernel_shape=[2, 2], strides=[1, 1]  # stride 1 to not change dims
        )
        
        result = surgeon.insert_node_before(simple_model, 'conv2', maxpool)
        
        assert result.success is True
        assert 'inserted_pool' in result.nodes_added
        
        # Verify conv2 now takes input from the new node
        conv2 = surgeon.get_node_by_name(simple_model.graph, 'conv2')
        assert 'inserted_pool_out' in conv2.input
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_insert_before_nonexistent(self, simple_model):
        """Should fail when target doesn't exist."""
        from graph_surgeon.graph.surgeon import create_maxpool_node
        
        surgeon = GraphSurgeon(verbose=False)
        maxpool = create_maxpool_node('mp', 'x', 'y')
        
        result = surgeon.insert_node_before(simple_model, 'nonexistent', maxpool)
        
        assert result.success is False
        assert 'not found' in result.message


class TestReplaceNode:
    """Tests for replace_node() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_replace_maxpool_with_avgpool(self, simple_model):
        """Should replace MaxPool with AvgPool."""
        from graph_surgeon.graph.surgeon import create_avgpool_node
        
        surgeon = GraphSurgeon(verbose=False)
        
        # Get original maxpool's output
        old_maxpool = surgeon.get_node_by_name(simple_model.graph, 'maxpool')
        assert old_maxpool is not None
        
        # Create replacement AvgPool
        avgpool = create_avgpool_node(
            'new_avgpool', 'dummy', 'avgpool_out',
            kernel_shape=[2, 2], strides=[2, 2]
        )
        
        result = surgeon.replace_node(simple_model, 'maxpool', avgpool)
        
        assert result.success is True
        assert 'maxpool' in result.nodes_removed
        assert 'new_avgpool' in result.nodes_added
        
        # Verify old node gone, new node present
        assert surgeon.get_node_by_name(simple_model.graph, 'maxpool') is None
        new_node = surgeon.get_node_by_name(simple_model.graph, 'new_avgpool')
        assert new_node is not None
        assert new_node.op_type == 'AveragePool'
        
        # Verify it has the original inputs
        assert 'relu2_out' in new_node.input
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_replace_nonexistent(self, simple_model):
        """Should fail when node doesn't exist."""
        from graph_surgeon.graph.surgeon import create_maxpool_node
        
        surgeon = GraphSurgeon(verbose=False)
        maxpool = create_maxpool_node('mp', 'x', 'y')
        
        result = surgeon.replace_node(simple_model, 'nonexistent', maxpool)
        
        assert result.success is False


class TestAddInitializer:
    """Tests for add_initializer() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_add_initializer(self, simple_model):
        """Should add a new initializer."""
        import numpy as np
        
        surgeon = GraphSurgeon(verbose=False)
        
        weights = np.random.randn(32, 64, 3, 3).astype(np.float32)
        result = surgeon.add_initializer(simple_model, 'new_weights', weights)
        
        assert result.success is True
        
        # Verify initializer was added
        names = [i.name for i in simple_model.graph.initializer]
        assert 'new_weights' in names
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_add_duplicate_initializer(self, simple_model):
        """Should fail when initializer already exists."""
        import numpy as np
        
        surgeon = GraphSurgeon(verbose=False)
        
        # conv1_weights already exists
        weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
        result = surgeon.add_initializer(simple_model, 'conv1_weights', weights)
        
        assert result.success is False
        assert 'already exists' in result.message


class TestRemoveSubgraph:
    """Tests for remove_subgraph() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_remove_single_node(self, simple_model):
        """Should work for removing a single node."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Remove relu2 (between conv2 and maxpool)
        result = surgeon.remove_subgraph(simple_model, ['relu2'])
        
        assert result.success is True
        assert 'relu2' in result.nodes_removed
        
        # Verify relu2 is gone
        assert surgeon.get_node_by_name(simple_model.graph, 'relu2') is None
        
        # Verify maxpool now connects to conv2's output
        maxpool = surgeon.get_node_by_name(simple_model.graph, 'maxpool')
        assert 'conv2_out' in maxpool.input
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_remove_multiple_nodes(self, simple_model):
        """Should remove multiple connected nodes."""
        surgeon = GraphSurgeon(verbose=False)
        
        # Remove conv2 and relu2 together
        original_count = len(simple_model.graph.node)
        
        result = surgeon.remove_subgraph(
            simple_model, 
            ['conv2', 'relu2'],
            entry_rewire=('conv2', 0)  # Use conv2's first input (relu1_out)
        )
        
        assert result.success is True
        assert len(result.nodes_removed) == 2
        
        # Verify nodes are gone
        assert surgeon.get_node_by_name(simple_model.graph, 'conv2') is None
        assert surgeon.get_node_by_name(simple_model.graph, 'relu2') is None
        
        # Node count decreased by 2
        assert len(simple_model.graph.node) == original_count - 2
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_remove_empty_list(self, simple_model):
        """Should fail for empty node list."""
        surgeon = GraphSurgeon(verbose=False)
        
        result = surgeon.remove_subgraph(simple_model, [])
        
        assert result.success is False


class TestCompareGraphs:
    """Tests for compare_graphs() method."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_compare_identical(self, simple_model):
        """Identical models should show no differences."""
        surgeon = GraphSurgeon(verbose=False)
        
        model_copy = surgeon.clone_model(simple_model)
        diff = surgeon.compare_graphs(simple_model, model_copy)
        
        assert diff['nodes_added'] == []
        assert diff['nodes_removed'] == []
        assert diff['nodes_modified'] == []
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_compare_with_addition(self, simple_model):
        """Should detect added nodes."""
        from graph_surgeon.graph.surgeon import create_maxpool_node
        
        surgeon = GraphSurgeon(verbose=False)
        
        original = surgeon.clone_model(simple_model)
        
        # Add a node to simple_model
        maxpool = create_maxpool_node(
            'added_pool', 'relu1_out', 'added_pool_out',
            kernel_shape=[2, 2], strides=[1, 1]
        )
        surgeon.insert_node_before(simple_model, 'conv2', maxpool)
        
        diff = surgeon.compare_graphs(original, simple_model)
        
        assert 'added_pool' in diff['nodes_added']
        assert '+1' in diff['summary']
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_compare_with_removal(self, simple_model):
        """Should detect removed nodes."""
        surgeon = GraphSurgeon(verbose=False)
        
        original = surgeon.clone_model(simple_model)
        
        # Remove a node
        surgeon.remove_subgraph(simple_model, ['relu2'])
        
        diff = surgeon.compare_graphs(original, simple_model)
        
        assert 'relu2' in diff['nodes_removed']
        assert '-1' in diff['summary']


class TestCreateConvNode:
    """Tests for create_conv_node() helper."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_create_conv_basic(self):
        """Should create a Conv node with weights."""
        from graph_surgeon.graph.surgeon import create_conv_node
        
        node, inits = create_conv_node(
            'test_conv', 'input', 'output',
            in_channels=3, out_channels=64
        )
        
        assert node.op_type == 'Conv'
        assert node.name == 'test_conv'
        assert len(inits) == 1  # weights only (no bias)
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_create_conv_identity_init(self):
        """Identity init should create near-identity convolution."""
        from graph_surgeon.graph.surgeon import create_conv_node
        import numpy as np
        
        node, inits = create_conv_node(
            'id_conv', 'input', 'output',
            in_channels=64, out_channels=64,
            kernel_shape=[3, 3],
            weight_init='identity'
        )
        
        # Check weights have identity-like structure
        weights = np.frombuffer(inits[0].raw_data, dtype=np.float32)
        weights = weights.reshape(64, 64, 3, 3)
        
        # Center of kernel for each channel should be 1
        assert weights[0, 0, 1, 1] == 1.0
        assert weights[1, 1, 1, 1] == 1.0
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_create_conv_small_init(self):
        """Small init should have small weight values."""
        from graph_surgeon.graph.surgeon import create_conv_node
        import numpy as np
        
        node, inits = create_conv_node(
            'small_conv', 'input', 'output',
            in_channels=3, out_channels=64,
            weight_init='small'
        )
        
        weights = np.frombuffer(inits[0].raw_data, dtype=np.float32)
        
        # Weights should be small (around 0.01 scale)
        assert np.abs(weights).max() < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
