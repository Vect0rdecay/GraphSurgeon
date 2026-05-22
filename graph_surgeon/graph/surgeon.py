"""GraphSurgeon orchestrator for ONNX DAG reverse engineering."""
import copy
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from graph_surgeon._env import import_onnxruntime

    ort = import_onnxruntime()
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

import numpy as np

from graph_surgeon.graph.topology import (
    LayerPosition,
    NodeTopology,
    GraphTopology,
    GraphTopologyConfig,
)
from graph_surgeon.graph.validation import (
    GraphValidationLevel,
    GraphValidationResult,
    ValidationLevel,
    ValidationResult,
)
from graph_surgeon.graph.edits import SurgeryResult

class GraphSurgeon:
    """
    Core graph manipulation engine for ONNX models.
    
    Provides primitives for:
    - Inserting nodes into the graph
    - Removing nodes and rewiring edges
    - Modifying node attributes
    - Validating modified graphs
    """
    
    def __init__(self, verbose: bool = True):
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package required. Install with: pip install onnx")
        self.verbose = verbose
    
    def log(self, message: str):
        """Log if verbose mode enabled."""
        if self.verbose:
            print(f"[Surgeon] {message}")
    
    def load_model(self, model_path: str) -> 'onnx.ModelProto':
        """Load an ONNX model from file."""
        self.log(f"Loading model: {model_path}")
        return onnx.load(model_path)
    
    def save_model(self, model: 'onnx.ModelProto', output_path: str):
        """Save an ONNX model to file."""
        self.log(f"Saving model: {output_path}")
        onnx.save(model, output_path)
    
    def clone_model(self, model: 'onnx.ModelProto') -> 'onnx.ModelProto':
        """Create a deep copy of a model."""
        return copy.deepcopy(model)
    
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
    
    def get_graph_topology(self, graph: 'onnx.GraphProto', config: Optional[GraphTopologyConfig] = None) -> GraphTopology:
        """
        Analyze the topology of an ONNX graph.
        
        Computes:
        - Depth of each node from graph inputs
        - Position classification (early/middle/late)
        - Grouping by operation type
        - Topological execution order
        
        Args:
            graph: ONNX graph to analyze
            
        Returns:
            GraphTopology with complete topology analysis
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> topology = surgeon.get_graph_topology(model.graph)
            >>> early_nodes = topology.by_position[LayerPosition.EARLY]
            >>> conv_nodes = topology.by_op_type.get('Conv', [])
        """
        # Build tensor-to-producer mapping
        tensor_producers: Dict[str, str] = {}  # tensor_name -> node_name that produces it
        for node in graph.node:
            for output in node.output:
                tensor_producers[output] = node.name
        
        # Mark graph inputs as depth 0 sources
        input_tensors = {inp.name for inp in graph.input}
        
        # Compute depth for each node using BFS/dynamic programming
        node_depths: Dict[str, int] = {}
        
        def get_node_depth(node_name: str) -> int:
            """Recursively compute depth of a node."""
            if node_name in node_depths:
                return node_depths[node_name]
            
            node = self.get_node_by_name(graph, node_name)
            if not node:
                return 0
            
            max_input_depth = -1
            for inp in node.input:
                if inp in input_tensors:
                    # Direct connection to graph input
                    max_input_depth = max(max_input_depth, -1)
                elif inp in tensor_producers:
                    producer_name = tensor_producers[inp]
                    producer_depth = get_node_depth(producer_name)
                    max_input_depth = max(max_input_depth, producer_depth)
                # else: initializer or constant, depth = -1
            
            depth = max_input_depth + 1
            node_depths[node_name] = depth
            return depth
        
        # Compute depth for all nodes
        for node in graph.node:
            get_node_depth(node.name)
        
        # Determine max depth and position thresholds
        if node_depths:
            max_depth = max(node_depths.values())
        else:
            max_depth = 0
        
        cfg = config or GraphTopologyConfig()
        early_threshold = max_depth * cfg.early_fraction
        late_threshold = max_depth * cfg.late_fraction
        
        # Build topology info for each node
        nodes: Dict[str, NodeTopology] = {}
        by_position: Dict[LayerPosition, List[str]] = {
            LayerPosition.EARLY: [],
            LayerPosition.MIDDLE: [],
            LayerPosition.LATE: []
        }
        by_op_type: Dict[str, List[str]] = {}
        
        for node in graph.node:
            depth = node_depths.get(node.name, 0)
            
            # Determine position
            if depth <= early_threshold:
                position = LayerPosition.EARLY
            elif depth >= late_threshold:
                position = LayerPosition.LATE
            else:
                position = LayerPosition.MIDDLE
            
            # Create topology info
            node_info = NodeTopology(
                name=node.name,
                op_type=node.op_type,
                depth=depth,
                position=position,
                inputs=list(node.input),
                outputs=list(node.output)
            )
            nodes[node.name] = node_info
            by_position[position].append(node.name)
            
            # Group by op_type
            if node.op_type not in by_op_type:
                by_op_type[node.op_type] = []
            by_op_type[node.op_type].append(node.name)
        
        # Generate execution order (topological sort by depth)
        execution_order = sorted(
            [node.name for node in graph.node],
            key=lambda n: node_depths.get(n, 0)
        )
        
        return GraphTopology(
            total_nodes=len(graph.node),
            max_depth=max_depth,
            nodes=nodes,
            by_position=by_position,
            by_op_type=by_op_type,
            execution_order=execution_order
        )
    
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
    
    def get_early_layers(self, graph: 'onnx.GraphProto', 
                         op_type: Optional[str] = None) -> List['onnx.NodeProto']:
        """
        Get nodes in the early portion of the network (first 20%).
        
        Useful for finding injection points for aliasing vulnerabilities.
        
        Args:
            graph: ONNX graph to analyze
            op_type: Optional filter by operation type
            
        Returns:
            List of nodes in the early portion of the network
        """
        topology = self.get_graph_topology(graph)
        early_names = topology.by_position[LayerPosition.EARLY]
        
        result = []
        for name in early_names:
            node = self.get_node_by_name(graph, name)
            if node:
                if op_type is None or node.op_type == op_type:
                    result.append(node)
        
        return result
    
    def get_late_layers(self, graph: 'onnx.GraphProto',
                        op_type: Optional[str] = None) -> List['onnx.NodeProto']:
        """
        Get nodes in the late portion of the network (last 20%).
        
        Useful for finding classifier heads.
        
        Args:
            graph: ONNX graph to analyze  
            op_type: Optional filter by operation type
            
        Returns:
            List of nodes in the late portion of the network
        """
        topology = self.get_graph_topology(graph)
        late_names = topology.by_position[LayerPosition.LATE]
        
        result = []
        for name in late_names:
            node = self.get_node_by_name(graph, name)
            if node:
                if op_type is None or node.op_type == op_type:
                    result.append(node)
        
        return result
    
    def insert_node_after(self, model: 'onnx.ModelProto',
                          target_output: str,
                          new_node: 'onnx.NodeProto',
                          new_output_name: str) -> SurgeryResult:
        """
        Insert a new node after a target node's output.
        
        The new node will:
        1. Take target_output as its input
        2. Produce new_output_name as its output
        3. All consumers of target_output will be rewired to consume new_output_name
        
        Args:
            model: ONNX model to modify
            target_output: Output tensor name to insert after
            new_node: The node to insert
            new_output_name: Name for the new node's output
            
        Returns:
            SurgeryResult with modified model
        """
        graph = model.graph
        
        # Find consumers of target_output
        consumers = self.get_node_consumers(graph, target_output)
        
        # Rewire consumers to use new output
        edges_rewired = 0
        for consumer in consumers:
            for i, inp in enumerate(consumer.input):
                if inp == target_output:
                    consumer.input[i] = new_output_name
                    edges_rewired += 1
        
        # Also check graph outputs
        for i, out in enumerate(graph.output):
            if out.name == target_output:
                out.name = new_output_name
                edges_rewired += 1
        
        # Insert the new node
        # Find position after producer
        producer = self.get_node_by_output(graph, target_output)
        if producer:
            # Try by name first, then by output, then by reference
            insert_idx = self.get_node_index(graph, producer.name) if producer.name else None
            if insert_idx is None:
                insert_idx = self.get_node_index_by_output(graph, target_output)
            if insert_idx is None:
                insert_idx = self.get_node_index_by_ref(graph, producer)
            if insert_idx is not None:
                insert_idx += 1
            else:
                insert_idx = len(graph.node)
        else:
            insert_idx = 0  # Target might be a graph input
        
        graph.node.insert(insert_idx, new_node)
        
        self.log(f"Inserted {new_node.op_type} after {target_output}, rewired {edges_rewired} edges")
        
        return SurgeryResult(
            success=True,
            graph=graph,
            message=f"Inserted {new_node.name}",
            nodes_added=[new_node.name],
            edges_rewired=edges_rewired
        )
    
    def remove_node(self, model: 'onnx.ModelProto',
                    node_name: str,
                    rewire_input_idx: int = 0) -> SurgeryResult:
        """
        Remove a node from the graph and rewire edges.
        
        Consumers of the removed node's output will be connected to
        one of the removed node's inputs (specified by rewire_input_idx).
        
        Args:
            model: ONNX model to modify
            node_name: Name of node to remove
            rewire_input_idx: Which input of the removed node to use for rewiring
            
        Returns:
            SurgeryResult with modified model
        """
        graph = model.graph
        
        node = self.get_node_by_name(graph, node_name)
        if not node:
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Node not found: {node_name}"
            )
        
        if rewire_input_idx >= len(node.input):
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Invalid rewire_input_idx: {rewire_input_idx}"
            )
        
        # Get the input to rewire to
        rewire_to = node.input[rewire_input_idx]
        
        # Get the output(s) of this node
        outputs = list(node.output)
        
        # Rewire all consumers
        edges_rewired = 0
        for output in outputs:
            consumers = self.get_node_consumers(graph, output)
            for consumer in consumers:
                for i, inp in enumerate(consumer.input):
                    if inp == output:
                        consumer.input[i] = rewire_to
                        edges_rewired += 1
            
            # Also check graph outputs
            for i, out in enumerate(graph.output):
                if out.name == output:
                    out.name = rewire_to
                    edges_rewired += 1
        
        # Remove the node
        node_idx = self.get_node_index(graph, node_name)
        if node_idx is not None:
            del graph.node[node_idx]
        
        self.log(f"Removed {node_name}, rewired {edges_rewired} edges to {rewire_to}")
        
        return SurgeryResult(
            success=True,
            graph=graph,
            message=f"Removed {node_name}",
            nodes_removed=[node_name],
            edges_rewired=edges_rewired
        )
    
    def insert_node_before(self, model: 'onnx.ModelProto',
                           target_node_name: str,
                           new_node: 'onnx.NodeProto',
                           input_idx: int = 0) -> SurgeryResult:
        """
        Insert a new node before a target node.
        
        The new node will:
        1. Take the target node's input (at input_idx) as its own input
        2. The target node's input will be rewired to the new node's output
        
        Args:
            model: ONNX model to modify
            target_node_name: Name of node to insert before
            new_node: The node to insert (must have input/output already set)
            input_idx: Which input of target to intercept (default 0)
            
        Returns:
            SurgeryResult with modified model
            
        Example:
            >>> # Insert MaxPool before conv2
            >>> maxpool = create_maxpool_node('mp', 'relu1_out', 'mp_out')
            >>> result = surgeon.insert_node_before(model, 'conv2', maxpool)
        """
        graph = model.graph
        
        target = self.get_node_by_name(graph, target_node_name)
        if not target:
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Target node not found: {target_node_name}"
            )
        
        if input_idx >= len(target.input):
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Invalid input_idx: {input_idx}, node has {len(target.input)} inputs"
            )
        
        # Get the original input
        original_input = target.input[input_idx]
        
        # Verify new_node is configured correctly
        if len(new_node.output) == 0:
            return SurgeryResult(
                success=False,
                graph=None,
                message="New node must have at least one output defined"
            )
        
        new_output = new_node.output[0]
        
        # Rewire target to use new node's output
        target.input[input_idx] = new_output
        
        # Find insertion position (before target)
        insert_idx = self.get_node_index(graph, target_node_name)
        if insert_idx is None:
            insert_idx = 0
        
        graph.node.insert(insert_idx, new_node)
        
        self.log(f"Inserted {new_node.op_type} before {target_node_name}")
        
        return SurgeryResult(
            success=True,
            graph=graph,
            message=f"Inserted {new_node.name} before {target_node_name}",
            nodes_added=[new_node.name],
            edges_rewired=1
        )
    
    def replace_node(self, model: 'onnx.ModelProto',
                     old_node_name: str,
                     new_node: 'onnx.NodeProto') -> SurgeryResult:
        """
        Replace a node with another node.
        
        The new node will:
        1. Take the same position in the graph
        2. Have its inputs/outputs rewired to match the old node's connections
        
        Args:
            model: ONNX model to modify
            old_node_name: Name of node to replace
            new_node: The replacement node
            
        Returns:
            SurgeryResult with modified model
            
        Example:
            >>> # Replace AvgPool with MaxPool
            >>> maxpool = create_maxpool_node('new_pool', 'input', 'output')
            >>> result = surgeon.replace_node(model, 'avgpool1', maxpool)
        """
        graph = model.graph
        
        old_node = self.get_node_by_name(graph, old_node_name)
        if not old_node:
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Node not found: {old_node_name}"
            )
        
        # Copy inputs from old node to new node (preserving connections)
        del new_node.input[:]
        new_node.input.extend(old_node.input)
        
        # Preserve old outputs for rewiring
        old_outputs = list(old_node.output)
        new_outputs = list(new_node.output)
        
        # If new node has different outputs, rewire consumers
        edges_rewired = 0
        for i, old_out in enumerate(old_outputs):
            if i < len(new_outputs):
                new_out = new_outputs[i]
                if old_out != new_out:
                    # Rewire consumers
                    consumers = self.get_node_consumers(graph, old_out)
                    for consumer in consumers:
                        for j, inp in enumerate(consumer.input):
                            if inp == old_out:
                                consumer.input[j] = new_out
                                edges_rewired += 1
                    
                    # Check graph outputs
                    for out in graph.output:
                        if out.name == old_out:
                            out.name = new_out
                            edges_rewired += 1
        
        # Find position and replace
        node_idx = self.get_node_index(graph, old_node_name)
        if node_idx is not None:
            del graph.node[node_idx]
            graph.node.insert(node_idx, new_node)
        
        self.log(f"Replaced {old_node_name} with {new_node.name}")
        
        return SurgeryResult(
            success=True,
            graph=graph,
            message=f"Replaced {old_node_name} with {new_node.name}",
            nodes_added=[new_node.name],
            nodes_removed=[old_node_name],
            edges_rewired=edges_rewired
        )
    
    def add_initializer(self, model: 'onnx.ModelProto',
                        name: str,
                        values: np.ndarray) -> SurgeryResult:
        """
        Add an initializer (weight tensor) to the model.
        
        Args:
            model: ONNX model to modify
            name: Name for the initializer
            values: Numpy array with the values
            
        Returns:
            SurgeryResult
            
        Example:
            >>> # Add weights for a new Conv
            >>> weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
            >>> surgeon.add_initializer(model, 'new_conv_weights', weights)
        """
        # Check if name already exists
        for init in model.graph.initializer:
            if init.name == name:
                return SurgeryResult(
                    success=False,
                    graph=None,
                    message=f"Initializer already exists: {name}"
                )
        
        # Create and add the tensor to initializer
        tensor = numpy_helper.from_array(values, name)
        model.graph.initializer.append(tensor)
        
        # Also add to graph.input for opset compatibility (required for opset < 13)
        # Check if other initializers are in graph.input (indicates older format)
        init_names = {init.name for init in model.graph.initializer}
        input_names = {inp.name for inp in model.graph.input}
        if init_names & input_names:  # If any initializers are also inputs
            # Create a ValueInfoProto for the input
            tensor_type = helper.make_tensor_type_proto(
                elem_type=tensor.data_type,
                shape=list(tensor.dims)
            )
            value_info = helper.make_value_info(name, tensor_type)
            model.graph.input.append(value_info)
        
        self.log(f"Added initializer: {name} with shape {values.shape}")
        
        return SurgeryResult(
            success=True,
            graph=model.graph,
            message=f"Added initializer {name}",
            nodes_added=[name]  # Not really a node, but tracked for logging
        )
    
    def remove_subgraph(self, model: 'onnx.ModelProto',
                        node_names: List[str],
                        entry_rewire: Optional[Tuple[str, int]] = None) -> SurgeryResult:
        """
        Remove multiple connected nodes from the graph.
        
        For a subgraph like: A -> B -> C -> D
        If we remove [B, C], the graph becomes: A -> D
        
        Args:
            model: ONNX model to modify
            node_names: List of node names to remove
            entry_rewire: Optional (node_name, input_idx) to specify which input
                         of the first node should be used for rewiring.
                         If None, uses first input of first node.
            
        Returns:
            SurgeryResult with modified model
            
        Example:
            >>> # Remove SE block (multiple nodes)
            >>> surgeon.remove_subgraph(model, ['se_gap', 'se_fc1', 'se_relu', 'se_fc2', 'se_sigmoid', 'se_mul'])
        """
        graph = model.graph
        
        if not node_names:
            return SurgeryResult(
                success=False,
                graph=None,
                message="No nodes specified for removal"
            )
        
        # Verify all nodes exist
        nodes = []
        for name in node_names:
            node = self.get_node_by_name(graph, name)
            if not node:
                return SurgeryResult(
                    success=False,
                    graph=None,
                    message=f"Node not found: {name}"
                )
            nodes.append(node)
        
        # Find the "entry" of the subgraph (first node in execution order)
        # and the "exit" (last node whose output goes outside the subgraph)
        subgraph_outputs = set()
        subgraph_inputs = set()
        
        for node in nodes:
            subgraph_outputs.update(node.output)
            subgraph_inputs.update(node.input)
        
        # Entry inputs: inputs to subgraph that come from outside
        entry_inputs = subgraph_inputs - subgraph_outputs
        
        # Exit outputs: outputs from subgraph that go outside
        exit_outputs = set()
        for output in subgraph_outputs:
            consumers = self.get_node_consumers(graph, output)
            for consumer in consumers:
                if consumer.name not in node_names:
                    exit_outputs.add(output)
                    break
            # Also check graph outputs
            for graph_out in graph.output:
                if graph_out.name == output:
                    exit_outputs.add(output)
        
        # Determine rewire source
        if entry_rewire:
            entry_node = self.get_node_by_name(graph, entry_rewire[0])
            if entry_node and entry_rewire[1] < len(entry_node.input):
                rewire_source = entry_node.input[entry_rewire[1]]
            else:
                rewire_source = list(entry_inputs)[0] if entry_inputs else None
        else:
            # Use first entry input
            rewire_source = list(entry_inputs)[0] if entry_inputs else None
        
        if not rewire_source:
            return SurgeryResult(
                success=False,
                graph=None,
                message="Could not determine rewire source"
            )
        
        # Rewire all exit outputs to the entry input
        edges_rewired = 0
        for exit_output in exit_outputs:
            consumers = self.get_node_consumers(graph, exit_output)
            for consumer in consumers:
                if consumer.name not in node_names:
                    for i, inp in enumerate(consumer.input):
                        if inp == exit_output:
                            consumer.input[i] = rewire_source
                            edges_rewired += 1
            
            # Check graph outputs
            for graph_out in graph.output:
                if graph_out.name == exit_output:
                    graph_out.name = rewire_source
                    edges_rewired += 1
        
        # Remove all nodes (in reverse order to maintain indices)
        nodes_removed = []
        for name in reversed(node_names):
            idx = self.get_node_index(graph, name)
            if idx is not None:
                del graph.node[idx]
                nodes_removed.append(name)
        
        self.log(f"Removed subgraph of {len(nodes_removed)} nodes, rewired {edges_rewired} edges")
        
        return SurgeryResult(
            success=True,
            graph=graph,
            message=f"Removed {len(nodes_removed)} nodes",
            nodes_removed=nodes_removed,
            edges_rewired=edges_rewired
        )
    
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
    
    def modify_node_attribute(self, model: 'onnx.ModelProto',
                              node_name: str,
                              attr_name: str,
                              new_value: Any) -> SurgeryResult:
        """
        Modify an attribute of an existing node.
        
        Args:
            model: ONNX model to modify
            node_name: Name of node to modify
            attr_name: Attribute name to change
            new_value: New value for the attribute
            
        Returns:
            SurgeryResult with modified model
        """
        graph = model.graph
        
        node = self.get_node_by_name(graph, node_name)
        if not node:
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Node not found: {node_name}"
            )
        
        # Find and modify the attribute
        found = False
        for attr in node.attribute:
            if attr.name == attr_name:
                # Determine attribute type and set accordingly
                if isinstance(new_value, int):
                    attr.i = new_value
                elif isinstance(new_value, float):
                    attr.f = new_value
                elif isinstance(new_value, str):
                    attr.s = new_value.encode()
                elif isinstance(new_value, list):
                    if new_value and isinstance(new_value[0], int):
                        del attr.ints[:]
                        attr.ints.extend(new_value)
                    elif new_value and isinstance(new_value[0], float):
                        del attr.floats[:]
                        attr.floats.extend(new_value)
                found = True
                break
        
        if not found:
            return SurgeryResult(
                success=False,
                graph=None,
                message=f"Attribute not found: {attr_name}"
            )
        
        self.log(f"Modified {node_name}.{attr_name} = {new_value}")
        
        return SurgeryResult(
            success=True,
            graph=graph,
            message=f"Modified {node_name}.{attr_name}",
            nodes_modified=[node_name]
        )
    
    def validate(self, model: 'onnx.ModelProto',
                 level: GraphValidationLevel = GraphValidationLevel.STRUCTURAL,
                 sample_input: Optional[np.ndarray] = None) -> GraphValidationResult:
        """
        Validate a model at the specified level.
        
        Args:
            model: ONNX model to validate
            level: How thorough to validate
            sample_input: Sample input for inference testing
            
        Returns:
            GraphValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Structural validation
        if level.value != "none":
            try:
                onnx.checker.check_model(model)
            except Exception as e:
                errors.append(f"ONNX checker failed: {e}")
                return GraphValidationResult(
                    valid=False,
                    level=GraphValidationLevel.STRUCTURAL,
                    errors=errors
                )
        
        if level == GraphValidationLevel.STRUCTURAL:
            return GraphValidationResult(valid=True, level=level)
        
        # Load validation
        if level.value in ["loadable", "runnable"]:
            if not ONNXRUNTIME_AVAILABLE:
                warnings.append("onnxruntime not available, skipping load test")
            else:
                try:
                    # Save to bytes and load
                    model_bytes = model.SerializeToString()
                    session = ort.InferenceSession(model_bytes)
                except Exception as e:
                    errors.append(f"Failed to load model: {e}")
                    return GraphValidationResult(
                        valid=False,
                        level=GraphValidationLevel.LOADABLE,
                        errors=errors,
                        warnings=warnings
                    )
        
        if level == GraphValidationLevel.LOADABLE:
            return GraphValidationResult(valid=True, level=level, warnings=warnings)
        
        # Inference validation
        if level == GraphValidationLevel.RUNNABLE:
            if not ONNXRUNTIME_AVAILABLE:
                warnings.append("onnxruntime not available, skipping inference test")
                return GraphValidationResult(valid=True, level=level, warnings=warnings)
            
            try:
                model_bytes = model.SerializeToString()
                session = ort.InferenceSession(model_bytes)
                
                # Create sample input if not provided
                if sample_input is None:
                    input_info = session.get_inputs()[0]
                    shape = []
                    for dim in input_info.shape:
                        if isinstance(dim, int) and dim > 0:
                            shape.append(dim)
                        else:
                            shape.append(1 if len(shape) == 0 else 224)
                    sample_input = np.random.randn(*shape).astype(np.float32)
                
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: sample_input})
                
                return GraphValidationResult(
                    valid=True,
                    level=level,
                    warnings=warnings,
                    inference_output_shape=output[0].shape
                )
                
            except Exception as e:
                errors.append(f"Inference failed: {e}")
                return GraphValidationResult(
                    valid=False,
                    level=GraphValidationLevel.RUNNABLE,
                    errors=errors,
                    warnings=warnings
                )
        
        return GraphValidationResult(valid=True, level=level, warnings=warnings)
    
    def infer_shapes(self, model: 'onnx.ModelProto') -> 'onnx.ModelProto':
        """
        Run ONNX shape inference on a model.
        
        This propagates shape information through the graph, which is useful
        for verifying that grafts don't break tensor dimensions.
        
        Args:
            model: ONNX model to process
            
        Returns:
            Model with inferred shapes in value_info
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> model = surgeon.infer_shapes(model)
            >>> # Now model.graph.value_info contains shape info
        """
        try:
            from onnx import shape_inference
            return shape_inference.infer_shapes(model)
        except Exception as e:
            self.log(f"Shape inference failed: {e}")
            return model
    
    def get_tensor_shape(self, model: 'onnx.ModelProto', 
                         tensor_name: str) -> Optional[List[int]]:
        """
        Get the shape of a tensor in the model.
        
        Note: Requires shape inference to have been run on the model.
        
        Args:
            model: ONNX model (preferably with inferred shapes)
            tensor_name: Name of the tensor to get shape for
            
        Returns:
            List of dimensions, or None if shape unknown
        """
        # Check graph inputs
        for inp in model.graph.input:
            if inp.name == tensor_name:
                return self._extract_shape(inp.type)
        
        # Check graph outputs  
        for out in model.graph.output:
            if out.name == tensor_name:
                return self._extract_shape(out.type)
        
        # Check value_info (intermediate tensors with inferred shapes)
        for vi in model.graph.value_info:
            if vi.name == tensor_name:
                return self._extract_shape(vi.type)
        
        return None
    
    def _extract_shape(self, type_proto: 'onnx.TypeProto') -> Optional[List[int]]:
        """Extract shape from an ONNX TypeProto."""
        if not type_proto.HasField('tensor_type'):
            return None
        
        tensor_type = type_proto.tensor_type
        if not tensor_type.HasField('shape'):
            return None
        
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            elif dim.HasField('dim_param'):
                shape.append(-1)  # Dynamic dimension
            else:
                shape.append(-1)
        
        return shape
    
    def check_shape_compatibility(self, model: 'onnx.ModelProto',
                                   tensor_a: str, 
                                   tensor_b: str) -> Tuple[bool, str]:
        """
        Check if two tensors have compatible shapes.
        
        Args:
            model: ONNX model
            tensor_a: First tensor name
            tensor_b: Second tensor name
            
        Returns:
            Tuple of (compatible, message)
        """
        shape_a = self.get_tensor_shape(model, tensor_a)
        shape_b = self.get_tensor_shape(model, tensor_b)
        
        if shape_a is None:
            return False, f"Shape unknown for {tensor_a}"
        if shape_b is None:
            return False, f"Shape unknown for {tensor_b}"
        
        if len(shape_a) != len(shape_b):
            return False, f"Rank mismatch: {shape_a} vs {shape_b}"
        
        for i, (a, b) in enumerate(zip(shape_a, shape_b)):
            if a == -1 or b == -1:
                continue  # Dynamic dimension, assume compatible
            if a != b:
                return False, f"Dimension {i} mismatch: {a} vs {b}"
        
        return True, "Shapes compatible"
    
    def add_metadata(self, model: 'onnx.ModelProto', 
                     key: str, 
                     value: str) -> 'onnx.ModelProto':
        """
        Add metadata to the model.
        
        Args:
            model: ONNX model to modify
            key: Metadata key
            value: Metadata value
            
        Returns:
            Model with added metadata
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> model = surgeon.add_metadata(model, 'carcinoma.grafts', 'ALIASING_DOWNSAMPLE')
        """
        # Create metadata entry
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
        
        return model
    
    def get_metadata(self, model: 'onnx.ModelProto', 
                     key: str) -> Optional[str]:
        """
        Get metadata from the model.
        
        Args:
            model: ONNX model
            key: Metadata key to retrieve
            
        Returns:
            Metadata value or None if not found
        """
        for meta in model.metadata_props:
            if meta.key == key:
                return meta.value
        return None
    
    def record_graft(self, model: 'onnx.ModelProto',
                     graft_type: str,
                     target_node: str,
                     details: Optional[Dict[str, Any]] = None) -> 'onnx.ModelProto':
        """
        Record a graft operation in the model's metadata.
        
        This creates a traceable record of all modifications made to the model.
        
        Args:
            model: ONNX model
            graft_type: Type of graft applied (e.g., 'ALIASING_DOWNSAMPLE')
            target_node: Name of the node that was targeted
            details: Optional additional details
            
        Returns:
            Model with graft recorded in metadata
            
        Example:
            >>> surgeon = GraphSurgeon()
            >>> model = surgeon.record_graft(model, 'INSERT_MAXPOOL', 'conv2_out')
        """
        import json
        from datetime import datetime
        
        # Get existing graft log or create new
        existing = self.get_metadata(model, 'carcinoma.graft_log')
        if existing:
            graft_log = json.loads(existing)
        else:
            graft_log = []
        
        # Add new graft record
        record = {
            'graft_type': graft_type,
            'target_node': target_node,
            'timestamp': datetime.now().isoformat(),
        }
        if details:
            record['details'] = details
        
        graft_log.append(record)
        
        # Update metadata
        # Remove old entry if exists
        for i, meta in enumerate(model.metadata_props):
            if meta.key == 'carcinoma.graft_log':
                del model.metadata_props[i]
                break
        
        self.add_metadata(model, 'carcinoma.graft_log', json.dumps(graft_log))
        
        return model
    
    def get_graft_history(self, model: 'onnx.ModelProto') -> List[Dict[str, Any]]:
        """
        Get the history of grafts applied to a model.
        
        Args:
            model: ONNX model
            
        Returns:
            List of graft records, or empty list if no grafts recorded
        """
        import json
        
        log = self.get_metadata(model, 'carcinoma.graft_log')
        if log:
            return json.loads(log)
        return []


# Convenience functions
def create_maxpool_node(name: str, input_name: str, output_name: str,
                        kernel_shape: List[int] = [2, 2],
                        strides: List[int] = [2, 2],
                        pads: Optional[List[int]] = None) -> 'onnx.NodeProto':
    """
    Create a MaxPool node.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        kernel_shape: Pooling kernel size [H, W]
        strides: Pooling strides [H, W]
        pads: Padding [top, left, bottom, right]. If None, no padding.
              For same-size output with stride=1, use pads that preserve dims.
    """
    attrs = {
        'kernel_shape': kernel_shape,
        'strides': strides,
    }
    if pads is not None:
        attrs['pads'] = pads
    
    return helper.make_node(
        'MaxPool',
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        **attrs
    )


def create_avgpool_node(name: str, input_name: str, output_name: str,
                        kernel_shape: List[int] = [2, 2],
                        strides: List[int] = [2, 2],
                        pads: Optional[List[int]] = None) -> 'onnx.NodeProto':
    """
    Create an AveragePool node.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        kernel_shape: Pooling kernel size [H, W]
        strides: Pooling strides [H, W]
        pads: Padding [top, left, bottom, right]. If None, no padding.
    """
    attrs = {
        'kernel_shape': kernel_shape,
        'strides': strides,
    }
    if pads is not None:
        attrs['pads'] = pads
    
    return helper.make_node(
        'AveragePool',
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        **attrs
    )


def create_batchnorm_node(name: str, input_name: str, output_name: str,
                          num_features: int,
                          scale: float = 1.0,
                          bias: float = 0.0,
                          mean: float = 0.0,
                          var: float = 1.0) -> Tuple['onnx.NodeProto', List]:
    """
    Create a BatchNormalization node with initializers.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        num_features: Number of channels
        scale: Gamma parameter (default 1.0 = identity)
        bias: Beta parameter (default 0.0 = identity)
        mean: Running mean (default 0.0 = identity)
        var: Running variance (default 1.0 = identity)
        
    For vulnerability creation, use non-identity parameters:
        - mean != 0: Creates distribution shift on normalized values
        - var != 1: Scales the input differently
        - scale != 1: Amplifies/dampens the output
        - bias != 0: Shifts the output
    
    Returns tuple of (node, list of initializer tensors).
    """
    scale_name = f"{name}_scale"
    bias_name = f"{name}_bias"
    mean_name = f"{name}_mean"
    var_name = f"{name}_var"
    
    node = helper.make_node(
        'BatchNormalization',
        inputs=[input_name, scale_name, bias_name, mean_name, var_name],
        outputs=[output_name],
        name=name,
        epsilon=1e-5,
        momentum=0.9
    )
    
    # Create initializers with specified parameters
    initializers = [
        numpy_helper.from_array(
            np.full(num_features, scale, dtype=np.float32), scale_name
        ),
        numpy_helper.from_array(
            np.full(num_features, bias, dtype=np.float32), bias_name
        ),
        numpy_helper.from_array(
            np.full(num_features, mean, dtype=np.float32), mean_name
        ),
        numpy_helper.from_array(
            np.full(num_features, var, dtype=np.float32), var_name
        ),
    ]
    
    return node, initializers


def create_conv_node(name: str, input_name: str, output_name: str,
                     in_channels: int, out_channels: int,
                     kernel_shape: List[int] = [3, 3],
                     strides: List[int] = [1, 1],
                     pads: List[int] = [1, 1, 1, 1],
                     weight_init: str = "identity") -> Tuple['onnx.NodeProto', List]:
    """
    Create a Conv node with initializers.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_shape: Kernel size [H, W]
        strides: Stride [H, W]
        pads: Padding [top, left, bottom, right]
        weight_init: Weight initialization strategy:
            - "identity": Try to approximate identity (for same in/out channels)
            - "small": Small random values (minimal impact on accuracy)
            - "random": Random normal initialization
            - "zeros": Zero weights (output will be zeros)
            
    Returns:
        Tuple of (node, list of initializer tensors)
        
    Example:
        >>> node, inits = create_conv_node(
        ...     'downsample', 'input', 'down_out',
        ...     in_channels=64, out_channels=64,
        ...     strides=[2, 2]  # Stride-2 for downsampling
        ... )
    """
    weight_name = f"{name}_weight"
    
    node = helper.make_node(
        'Conv',
        inputs=[input_name, weight_name],
        outputs=[output_name],
        name=name,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads
    )
    
    # Initialize weights based on strategy
    kh, kw = kernel_shape
    
    if weight_init == "identity" and in_channels == out_channels:
        # Create identity-like convolution
        # Center pixel of kernel = 1/in_channels for each channel
        weights = np.zeros((out_channels, in_channels, kh, kw), dtype=np.float32)
        center_h, center_w = kh // 2, kw // 2
        for i in range(min(in_channels, out_channels)):
            weights[i, i, center_h, center_w] = 1.0
    elif weight_init == "small":
        # Small random values - minimal impact
        weights = np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * 0.01
    elif weight_init == "zeros":
        weights = np.zeros((out_channels, in_channels, kh, kw), dtype=np.float32)
    else:  # "random" or fallback
        # Xavier/Glorot initialization
        fan_in = in_channels * kh * kw
        fan_out = out_channels * kh * kw
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights = np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * std
    
    initializers = [
        numpy_helper.from_array(weights, weight_name)
    ]
    
    return node, initializers


def create_relu_node(name: str, input_name: str, output_name: str) -> 'onnx.NodeProto':
    """Create a ReLU activation node."""
    return helper.make_node(
        'Relu',
        inputs=[input_name],
        outputs=[output_name],
        name=name
    )


def create_identity_node(name: str, input_name: str, output_name: str) -> 'onnx.NodeProto':
    """Create an Identity node (pass-through)."""
    return helper.make_node(
        'Identity',
        inputs=[input_name],
        outputs=[output_name],
        name=name
    )
