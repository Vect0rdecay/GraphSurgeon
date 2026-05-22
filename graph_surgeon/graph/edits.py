"""Counterfactual graph edit operations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

try:
    import onnx
    from onnx import helper, numpy_helper
except ImportError:
    onnx = None  # type: ignore
    helper = None  # type: ignore
    numpy_helper = None  # type: ignore

if TYPE_CHECKING:
    import onnx as onnx_types


@dataclass
class SurgeryResult:
    success: bool
    graph: Optional["onnx_types.GraphProto"]
    message: str
    nodes_added: List[str] = field(default_factory=list)
    nodes_removed: List[str] = field(default_factory=list)
    nodes_modified: List[str] = field(default_factory=list)
    edges_rewired: int = 0


class GraphEdits:
    """Counterfactual edit operations for ONNX graphs."""

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
