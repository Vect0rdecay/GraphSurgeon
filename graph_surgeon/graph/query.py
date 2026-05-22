"""Node query and graph comparison utilities."""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore

if TYPE_CHECKING:
    import onnx as onnx_types


class GraphQuery:
    """Query helpers for ONNX graph nodes and attributes."""

    def get_node_by_name(self, graph: 'onnx.GraphProto', 
                         name: str) -> Optional['onnx.NodeProto']:
        """Find a node by its name."""
        for node in graph.node:
            if node.name == name:
                return node
        return None
    
    def get_node_by_output(self, graph: 'onnx.GraphProto',
                           output_name: str) -> Optional['onnx.NodeProto']:
        """Find a node by one of its outputs."""
        for node in graph.node:
            if output_name in node.output:
                return node
        return None
    
    def get_node_consumers(self, graph: 'onnx.GraphProto',
                           output_name: str) -> List['onnx.NodeProto']:
        """Find all nodes that consume a given output."""
        consumers = []
        for node in graph.node:
            if output_name in node.input:
                consumers.append(node)
        return consumers
    
    def get_node_index(self, graph: 'onnx.GraphProto', 
                       node_name: str) -> Optional[int]:
        """Get the index of a node in the graph's node list by name."""
        for i, node in enumerate(graph.node):
            if node.name == node_name:
                return i
        return None
    
    def get_node_index_by_output(self, graph: 'onnx.GraphProto',
                                  output_name: str) -> Optional[int]:
        """Get the index of a node by one of its output tensor names."""
        for i, node in enumerate(graph.node):
            if output_name in node.output:
                return i
        return None
    
    def get_node_index_by_ref(self, graph: 'onnx.GraphProto',
                               target_node: 'onnx.NodeProto') -> Optional[int]:
        """Get the index of a specific node by reference."""
        for i, node in enumerate(graph.node):
            if node is target_node:
                return i
        return None
    
    def find_nodes_by_type(self, graph: 'onnx.GraphProto',
                           op_type: str) -> List['onnx.NodeProto']:
        """
        Find all nodes of a given operation type.
        
        Args:
            graph: ONNX graph to search
            op_type: The operation type to find (e.g., 'Conv', 'MaxPool', 'Concat')
            
        Returns:
            List of nodes matching the operation type
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> conv_nodes = surgeon.find_nodes_by_type(model.graph, 'Conv')
            >>> print(f"Found {len(conv_nodes)} Conv nodes")
        """
        return [node for node in graph.node if node.op_type == op_type]
    
    def find_nodes_by_attribute(self, graph: 'onnx.GraphProto',
                                attr_name: str,
                                attr_value: Any,
                                op_type: Optional[str] = None) -> List['onnx.NodeProto']:
        """
        Find nodes that have a specific attribute with a specific value.
        
        Args:
            graph: ONNX graph to search
            attr_name: Name of the attribute to match
            attr_value: Value the attribute should have
            op_type: Optional filter to only search nodes of this type
            
        Returns:
            List of nodes with matching attribute
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> # Find all Conv nodes with stride=2
            >>> stride2_convs = surgeon.find_nodes_by_attribute(
            ...     model.graph, 'strides', [2, 2], op_type='Conv'
            ... )
        """
        results = []
        
        for node in graph.node:
            # Filter by op_type if specified
            if op_type is not None and node.op_type != op_type:
                continue
            
            # Check attributes
            for attr in node.attribute:
                if attr.name != attr_name:
                    continue
                
                # Extract attribute value based on type
                actual_value = self._get_attribute_value(attr)
                
                # Compare values
                if self._values_equal(actual_value, attr_value):
                    results.append(node)
                    break
        
        return results
    
    def _get_attribute_value(self, attr: 'onnx.AttributeProto') -> Any:
        """Extract the value from an ONNX attribute."""
        # Check attribute type
        if attr.HasField('i'):
            return attr.i
        elif attr.HasField('f'):
            return attr.f
        elif attr.HasField('s'):
            return attr.s.decode() if isinstance(attr.s, bytes) else attr.s
        elif len(attr.ints) > 0:
            return list(attr.ints)
        elif len(attr.floats) > 0:
            return list(attr.floats)
        elif len(attr.strings) > 0:
            return [s.decode() if isinstance(s, bytes) else s for s in attr.strings]
        elif attr.HasField('t'):
            return attr.t  # Tensor
        elif len(attr.tensors) > 0:
            return list(attr.tensors)
        elif attr.HasField('g'):
            return attr.g  # Graph
        elif len(attr.graphs) > 0:
            return list(attr.graphs)
        return None
    
    def _values_equal(self, actual: Any, expected: Any) -> bool:
        """Compare two attribute values for equality."""
        # Handle list comparison
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            return all(a == e for a, e in zip(actual, expected))
        
        # Handle numpy arrays
        if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            return np.array_equal(actual, expected)
        
        # Direct comparison
        return actual == expected
    def clone_node(self, node: 'onnx.NodeProto', 
                   new_name: Optional[str] = None,
                   new_inputs: Optional[List[str]] = None,
                   new_outputs: Optional[List[str]] = None) -> 'onnx.NodeProto':
        """
        Create a deep copy of a node with optional modifications.
        
        Args:
            node: The node to clone
            new_name: Optional new name for the cloned node
            new_inputs: Optional new input tensor names
            new_outputs: Optional new output tensor names
            
        Returns:
            A new NodeProto that is a copy of the original
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> conv = surgeon.get_node_by_name(graph, 'conv1')
            >>> conv_copy = surgeon.clone_node(conv, new_name='conv1_copy')
        """
        # Deep copy the node
        cloned = copy.deepcopy(node)
        
        # Update name if specified
        if new_name is not None:
            cloned.name = new_name
        
        # Update inputs if specified
        if new_inputs is not None:
            del cloned.input[:]
            cloned.input.extend(new_inputs)
        
        # Update outputs if specified
        if new_outputs is not None:
            del cloned.output[:]
            cloned.output.extend(new_outputs)
        
        return cloned
    def compare_graphs(self, model_a: 'onnx.ModelProto',
                       model_b: 'onnx.ModelProto') -> Dict[str, Any]:
        """
        Compare two models and return differences.
        
        Args:
            model_a: First model (typically original)
            model_b: Second model (typically modified)
            
        Returns:
            Dictionary with comparison results
            
        Example:
            >>> original = surgeon.load_model("model.onnx")
            >>> modified = surgeon.clone_model(original)
            >>> # ... apply grafts ...
            >>> diff = surgeon.compare_graphs(original, modified)
            >>> print(f"Added: {diff['nodes_added']}")
        """
        graph_a = model_a.graph
        graph_b = model_b.graph
        
        nodes_a = {n.name: n for n in graph_a.node}
        nodes_b = {n.name: n for n in graph_b.node}
        
        names_a = set(nodes_a.keys())
        names_b = set(nodes_b.keys())
        
        # Find additions, removals, and modifications
        added = names_b - names_a
        removed = names_a - names_b
        common = names_a & names_b
        
        modified = []
        for name in common:
            node_a = nodes_a[name]
            node_b = nodes_b[name]
            
            # Compare op_type
            if node_a.op_type != node_b.op_type:
                modified.append({
                    'name': name,
                    'change': 'op_type',
                    'old': node_a.op_type,
                    'new': node_b.op_type
                })
                continue
            
            # Compare inputs
            if list(node_a.input) != list(node_b.input):
                modified.append({
                    'name': name,
                    'change': 'inputs',
                    'old': list(node_a.input),
                    'new': list(node_b.input)
                })
                continue
            
            # Compare attributes
            attrs_a = {a.name: self._get_attribute_value(a) for a in node_a.attribute}
            attrs_b = {a.name: self._get_attribute_value(a) for a in node_b.attribute}
            
            if attrs_a != attrs_b:
                modified.append({
                    'name': name,
                    'change': 'attributes',
                    'old': attrs_a,
                    'new': attrs_b
                })
        
        # Compare initializers
        init_a = {i.name for i in graph_a.initializer}
        init_b = {i.name for i in graph_b.initializer}
        
        initializers_added = init_b - init_a
        initializers_removed = init_a - init_b
        
        return {
            'nodes_added': list(added),
            'nodes_removed': list(removed),
            'nodes_modified': modified,
            'initializers_added': list(initializers_added),
            'initializers_removed': list(initializers_removed),
            'total_nodes_a': len(nodes_a),
            'total_nodes_b': len(nodes_b),
            'summary': f"+{len(added)} -{len(removed)} ~{len(modified)} nodes"
        }
