"""
ONNX Model Parser for GraphSurgeon

Parses ONNX models and extracts the DAG structure for graph analysis.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import struct


@dataclass
class ONNXTensor:
    """Represents a tensor in the ONNX model."""
    name: str
    shape: Tuple
    dtype: str
    data: Optional[bytes] = None


@dataclass
class ONNXNode:
    """Represents a node in the ONNX computational graph."""
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    domain: str = ""


@dataclass
class ONNXGraph:
    """Represents an ONNX computational graph."""
    name: str
    nodes: List[ONNXNode]
    inputs: List[ONNXTensor]
    outputs: List[ONNXTensor]
    initializers: Dict[str, ONNXTensor]
    value_info: Dict[str, ONNXTensor]
    _raw_model: Any = None

    @property
    def topology(self):
        """Graph topology via GraphSurgeon (depth, early/middle/late)."""
        if self._raw_model is None:
            return None
        from graph_surgeon.graph.surgeon import GraphSurgeon
        return GraphSurgeon(verbose=False).get_graph_topology(self._raw_model.graph)


class ONNXGraphParser:
    """
    Parser for ONNX models focused on graph analysis.
    
    Extracts the DAG structure, node attributes, and weight information
    needed for structural motif analysis.
    """
    
    # ONNX data type mapping
    DTYPE_MAP = {
        0: "undefined",
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "float64",
        12: "uint32",
        13: "uint64",
        14: "complex64",
        15: "complex128",
        16: "bfloat16",
    }
    
    def __init__(self):
        self.model = None
        self.graph = None
        
    def parse_file(self, filepath: str) -> ONNXGraph:
        """Parse an ONNX file and extract the graph."""
        try:
            import onnx
            self.model = onnx.load(filepath)
            return self._parse_model(self.model)
        except ImportError:
            # Fallback: parse protobuf directly
            return self._parse_protobuf(filepath)
    
    def parse_bytes(self, data: bytes) -> ONNXGraph:
        """Parse ONNX model from bytes."""
        try:
            import onnx
            self.model = onnx.load_from_string(data)
            return self._parse_model(self.model)
        except ImportError:
            return self._parse_protobuf_bytes(data)
    
    def _parse_model(self, model) -> ONNXGraph:
        """Parse an ONNX model object (using onnx library)."""
        import onnx
        
        graph = model.graph
        
        # Parse nodes
        nodes = []
        for node in graph.node:
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = self._parse_attribute(attr)
            
            nodes.append(ONNXNode(
                name=node.name or f"node_{len(nodes)}",
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attrs,
                domain=node.domain
            ))
        
        # Parse inputs
        inputs = []
        for inp in graph.input:
            shape = self._get_tensor_shape(inp.type)
            dtype = self._get_tensor_dtype(inp.type)
            inputs.append(ONNXTensor(
                name=inp.name,
                shape=shape,
                dtype=dtype
            ))
        
        # Parse outputs
        outputs = []
        for out in graph.output:
            shape = self._get_tensor_shape(out.type)
            dtype = self._get_tensor_dtype(out.type)
            outputs.append(ONNXTensor(
                name=out.name,
                shape=shape,
                dtype=dtype
            ))
        
        # Parse initializers (weights)
        initializers = {}
        for init in graph.initializer:
            shape = tuple(init.dims)
            dtype = self.DTYPE_MAP.get(init.data_type, "unknown")
            data = init.raw_data if init.raw_data else None
            initializers[init.name] = ONNXTensor(
                name=init.name,
                shape=shape,
                dtype=dtype,
                data=data
            )
        
        # Parse value_info (intermediate tensor shapes)
        value_info = {}
        for vi in graph.value_info:
            shape = self._get_tensor_shape(vi.type)
            dtype = self._get_tensor_dtype(vi.type)
            value_info[vi.name] = ONNXTensor(
                name=vi.name,
                shape=shape,
                dtype=dtype
            )
        
        return ONNXGraph(
            name=graph.name or "main",
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            initializers=initializers,
            value_info=value_info,
            _raw_model=model,
        )
    
    def _parse_attribute(self, attr) -> Any:
        """Parse an ONNX attribute to Python value."""
        import onnx
        
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') if isinstance(s, bytes) else s for s in attr.strings]
        elif attr.type == onnx.AttributeProto.TENSOR:
            return f"<tensor:{tuple(attr.t.dims)}>"
        elif attr.type == onnx.AttributeProto.GRAPH:
            return f"<graph:{attr.g.name}>"
        else:
            return f"<unknown_attr_type:{attr.type}>"
    
    def _get_tensor_shape(self, type_proto) -> Tuple:
        """Extract tensor shape from TypeProto."""
        if type_proto.HasField('tensor_type'):
            shape = type_proto.tensor_type.shape
            dims = []
            for dim in shape.dim:
                if dim.HasField('dim_value'):
                    dims.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    dims.append(dim.dim_param)  # Dynamic dimension
                else:
                    dims.append(-1)  # Unknown
            return tuple(dims)
        return ()
    
    def _get_tensor_dtype(self, type_proto) -> str:
        """Extract tensor dtype from TypeProto."""
        if type_proto.HasField('tensor_type'):
            return self.DTYPE_MAP.get(type_proto.tensor_type.elem_type, "unknown")
        return "unknown"
    
    def _parse_protobuf(self, filepath: str) -> ONNXGraph:
        """Fallback: parse ONNX protobuf directly without onnx library."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return self._parse_protobuf_bytes(data)
    
    def _parse_protobuf_bytes(self, data: bytes) -> ONNXGraph:
        """Parse ONNX protobuf bytes directly (minimal implementation)."""
        # This is a simplified fallback - recommend installing onnx package
        raise NotImplementedError(
            "Direct protobuf parsing not implemented. "
            "Please install the 'onnx' package: pip install onnx"
        )
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Extract edges (data flow connections) from the graph."""
        if not self.graph:
            return []
        
        edges = []
        
        # Build a map from output tensor name to producing node
        output_to_node = {}
        for node in self.graph.nodes:
            for output in node.outputs:
                output_to_node[output] = node.name
        
        # Also map graph inputs
        for inp in self.graph.inputs:
            output_to_node[inp.name] = f"input:{inp.name}"
        
        # Build edges based on input/output connections
        for node in self.graph.nodes:
            for inp in node.inputs:
                if inp in output_to_node:
                    src = output_to_node[inp]
                    edges.append((src, node.name))
                elif inp in self.graph.initializers:
                    # Skip initializer connections (weights)
                    pass
        
        return edges
    
    def get_tensor_shapes(self) -> Dict[str, Tuple]:
        """Get shapes for all known tensors in the graph."""
        shapes = {}
        
        if not self.graph:
            return shapes
        
        # Input shapes
        for inp in self.graph.inputs:
            shapes[inp.name] = inp.shape
        
        # Output shapes
        for out in self.graph.outputs:
            shapes[out.name] = out.shape
        
        # Initializer shapes
        for name, init in self.graph.initializers.items():
            shapes[name] = init.shape
        
        # Value info shapes
        for name, vi in self.graph.value_info.items():
            shapes[name] = vi.shape
        
        return shapes
    
    def infer_node_shapes(self, node: ONNXNode) -> Tuple[List[Tuple], List[Tuple]]:
        """Infer input and output shapes for a node."""
        shapes = self.get_tensor_shapes()
        
        input_shapes = []
        for inp in node.inputs:
            if inp in shapes:
                input_shapes.append(shapes[inp])
            else:
                input_shapes.append(())
        
        output_shapes = []
        for out in node.outputs:
            if out in shapes:
                output_shapes.append(shapes[out])
            else:
                output_shapes.append(())
        
        return input_shapes, output_shapes
    
    def analyze_model(self, filepath: str):
        """
        Parse and analyze an ONNX model, returning security profiles.
        
        Returns a tuple of (node_profiles, edges, graph_info).
        """
        from graph_surgeon.analysis.motifs import StructuralMotifAnalyzer, NodeSecurityProfile
        
        # Parse the model
        self.graph = self.parse_file(filepath)
        
        # Initialize analyzer
        analyzer = StructuralMotifAnalyzer()
        
        # Analyze each node
        node_profiles = []
        for node in self.graph.nodes:
            input_shapes, output_shapes = self.infer_node_shapes(node)
            
            # Get weight data if this node has weight inputs
            weights = None
            for inp in node.inputs:
                if inp in self.graph.initializers:
                    weights = self.graph.initializers[inp]
                    break
            
            profile = analyzer.analyze_node(
                node_id=node.name,
                op_type=node.op_type,
                attributes=node.attributes,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                weights=weights
            )
            node_profiles.append(profile)
        
        # Get edges
        edges = self.get_edges()
        
        # Graph metadata
        graph_info = {
            "name": self.graph.name,
            "num_nodes": len(self.graph.nodes),
            "num_inputs": len(self.graph.inputs),
            "num_outputs": len(self.graph.outputs),
            "num_initializers": len(self.graph.initializers),
            "input_names": [i.name for i in self.graph.inputs],
            "output_names": [o.name for o in self.graph.outputs],
        }
        
        return node_profiles, edges, graph_info


def analyze_onnx_graph(filepath: str, output_path: str = None, verbose: bool = False):
    """
    Main entry point for ONNX model graph analysis.
    
    Args:
        filepath: Path to ONNX model file
        output_path: Optional path to write JSON report
        verbose: Print detailed output
    
    Returns:
        ModelMotifReport object
    """
    from graph_surgeon.analysis.motifs import StructuralMotifAnalyzer, export_report_json
    
    # Parse model
    parser = ONNXGraphParser()
    node_profiles, edges, graph_info = parser.analyze_model(filepath)
    
    if verbose:
        print(f"Parsed model: {graph_info['name']}")
        print(f"  Nodes: {graph_info['num_nodes']}")
        print(f"  Inputs: {graph_info['input_names']}")
        print(f"  Outputs: {graph_info['output_names']}")
    
    # Generate security report
    analyzer = StructuralMotifAnalyzer()
    report = analyzer.generate_report(
        model_name=filepath,
        nodes=node_profiles,
        edges=edges
    )
    
    if verbose:
        print(f"\nMotif Analysis Complete")
        print(f"  Overall Risk Score: {report.overall_risk_score:.1f}/100")
        print(f"  Total structural findings: {len(report.structural_findings)}")
    
    # Export if requested
    if output_path:
        export_report_json(report, output_path)
        if verbose:
            print(f"\nReport saved to: {output_path}")
    
    return report


# Convenience function for quick analysis
def quick_scan(filepath: str) -> str:
    """
    Quick security scan of an ONNX model.
    
    Returns a brief text summary of findings.
    """
    report = analyze_onnx_graph(filepath, verbose=False)
    
    lines = [
        f"Motif Scan: {filepath}",
        f"=" * 50,
        f"Overall Risk: {report.overall_risk_score:.1f}/100",
        f"",
        f"Risk Breakdown:",
        f"  - Adversarial Attack Risk: {report.adversarial_risk_score:.1f}",
        f"  - ShadowLogic Risk: {report.shadowlogic_risk_score:.1f}",
        f"  - ImpNet Risk: {report.impnet_risk_score:.1f}",
        f"  - Model Extraction Risk: {report.extraction_risk_score:.1f}",
        f"  - Privacy Risk: {report.privacy_risk_score:.1f}",
        f"",
        f"Findings: {len(report.structural_findings)} structural_findings",
    ]
    
    # Add critical/high findings
    critical = [v for v in report.structural_findings if v.severity.value in ["critical", "high"]]
    if critical:
        lines.append(f"\nCritical/High Severity Issues:")
        for v in critical[:5]:
            lines.append(f"  - [{v.severity.value.upper()}] {v.title}")
    
    return "\n".join(lines)

