#!/usr/bin/env python3
"""GraphSurgeon CLI: ONNX DAG reverse engineering."""

from graph_surgeon._env import configure_runtime_quiet

configure_runtime_quiet()

import argparse
import json
import os
import sys

_FORMATTER = argparse.RawDescriptionHelpFormatter

_TOP_LEVEL_EPILOG = """
Examples:
  graph-surgeon inspect model.onnx
  graph-surgeon topology model.onnx --json
  graph-surgeon motifs model.onnx -o motifs.json
  graph-surgeon patterns model.onnx
  graph-surgeon flow model.onnx
  graph-surgeon catalog --gadget GAP_FC_HEAD
  graph-surgeon catalog --chain CHAIN-PATCH-ATTACK-SURFACE
  graph-surgeon catalog --coverage
  graph-surgeon operators --op Conv
  graph-surgeon edit validate edited.onnx --level runnable
  graph-surgeon edit remove-node model.onnx Conv_42 -o edited.onnx
  graph-surgeon diff baseline.onnx edited.onnx
"""

_INSPECT_EPILOG = """
Examples:
  graph-surgeon inspect model.onnx
  graph-surgeon inspect /path/to/resnet50.onnx
"""

_TOPOLOGY_EPILOG = """
Examples:
  graph-surgeon topology model.onnx
  graph-surgeon topology model.onnx --json
"""

_MOTIFS_EPILOG = """
Examples:
  graph-surgeon motifs model.onnx
  graph-surgeon motifs model.onnx -o report.json
  graph-surgeon motifs model.onnx --flow
"""

_PATTERNS_EPILOG = """
Examples:
  graph-surgeon patterns model.onnx
  graph-surgeon patterns model.onnx --json
  graph-surgeon patterns model.onnx -o structural_report.txt
"""

_FLOW_EPILOG = """
Examples:
  graph-surgeon flow model.onnx
"""

_CATALOG_EPILOG = """
Examples:
  graph-surgeon catalog
  graph-surgeon catalog --gadget GAP_FC_HEAD
  graph-surgeon catalog --chain CHAIN-PATCH-ATTACK-SURFACE
  graph-surgeon catalog --coverage
  graph-surgeon catalog --category "Adversarial Examples"
  graph-surgeon catalog --technique AML-ADV-001
"""

_OPERATORS_EPILOG = """
Examples:
  graph-surgeon operators
  graph-surgeon operators --op Conv
  graph-surgeon operators --op Softmax
"""

_EDIT_EPILOG = """
Examples:
  graph-surgeon edit validate edited.onnx
  graph-surgeon edit validate edited.onnx --level runnable
  graph-surgeon edit remove-node model.onnx Conv_42 -o edited.onnx
"""

_EDIT_VALIDATE_EPILOG = """
Examples:
  graph-surgeon edit validate edited.onnx
  graph-surgeon edit validate edited.onnx --level loadable
  graph-surgeon edit validate edited.onnx --level runnable
"""

_EDIT_REMOVE_EPILOG = """
Examples:
  graph-surgeon edit remove-node model.onnx Conv_42 -o edited.onnx
  graph-surgeon edit remove-node baseline.onnx /model/head/Gemm -o head_removed.onnx
"""

_DIFF_EPILOG = """
Examples:
  graph-surgeon diff baseline.onnx edited.onnx
  graph-surgeon diff original.onnx counterfactual.onnx
"""

_EXPORT_SCENE_EPILOG = """
Examples:
  graph-surgeon export-scene model.onnx -o scene.json
  graph-surgeon export-scene model.onnx --no-motifs -o scene.json
  graph-surgeon export-scene model.onnx --weights -o scene.json
"""

_SERVE_EPILOG = """
Examples:
  graph-surgeon serve model.onnx
  graph-surgeon serve model.onnx --port 8088
  graph-surgeon serve model.onnx --host 0.0.0.0 --port 9000
"""


def _subparser(sub, name, *, help_text, description, epilog=""):
    return sub.add_parser(
        name,
        help=help_text,
        description=description,
        formatter_class=_FORMATTER,
        epilog=epilog,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="graph-surgeon",
        description=(
            "Inspect, map, and experiment on ONNX computational DAGs.\n"
            "Structural motifs describe attack landscape (what attack classes are "
            "architecturally plausible), not confirmed exploitability."
        ),
        formatter_class=_FORMATTER,
        epilog=_TOP_LEVEL_EPILOG,
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        title="subcommands",
        metavar="{inspect,topology,motifs,patterns,flow,catalog,operators,edit,diff,export-scene,serve}",
    )

    # inspect
    p_inspect = _subparser(
        sub,
        "inspect",
        help_text="Model summary: inputs, outputs, op counts",
        description="Print a quick summary of an ONNX graph.",
        epilog=_INSPECT_EPILOG,
    )
    p_inspect.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    # topology
    p_topo = _subparser(
        sub,
        "topology",
        help_text="Depth, early/middle/late layers, execution order",
        description="Map graph depth and layer position (early stem, middle, late head).",
        epilog=_TOPOLOGY_EPILOG,
    )
    p_topo.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_topo.add_argument(
        "--json",
        action="store_true",
        help="Emit full topology report as JSON on stdout",
    )
    p_topo.set_defaults(func=cmd_topology)

    # motifs
    p_motifs = _subparser(
        sub,
        "motifs",
        help_text="Scan for structural motifs (attack landscape)",
        description=(
            "Run structural motif detection and print findings. "
            "Motifs index which adversarial attack classes are architecturally "
            "plausible on this graph."
        ),
        epilog=_MOTIFS_EPILOG,
    )
    p_motifs.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_motifs.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write JSON motif report to this file",
    )
    p_motifs.add_argument(
        "--flow",
        action="store_true",
        help="Also print plain-English execution narrative after the scan",
    )
    p_motifs.set_defaults(func=cmd_motifs)

    # patterns
    p_pat = _subparser(
        sub,
        "patterns",
        help_text="High-level DAG structural patterns",
        description=(
            "Detect coarse structural patterns (attention blocks, conv stacks, "
            "normalization chains) and emit a human-readable or JSON report."
        ),
        epilog=_PATTERNS_EPILOG,
    )
    p_pat.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_pat.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write text report to this file instead of stdout",
    )
    p_pat.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report on stdout",
    )
    p_pat.set_defaults(func=cmd_patterns)

    # flow
    p_flow = _subparser(
        sub,
        "flow",
        help_text="Plain-English execution narrative",
        description="Print a plain-English description of how data flows through the graph.",
        epilog=_FLOW_EPILOG,
    )
    p_flow.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_flow.set_defaults(func=cmd_flow)

    # catalog
    p_cat = _subparser(
        sub,
        "catalog",
        help_text="Motif registry, chains, techniques, research coverage",
        description=(
            "Browse structural motifs, compound chains, and literature techniques. "
            "With no flags, prints the RE catalog index. Use lookup flags for detail."
        ),
        epilog=_CATALOG_EPILOG,
    )
    p_cat.add_argument(
        "--gadget",
        metavar="ID",
        help="Lookup structural motif by registry ID (e.g. GAP_FC_HEAD, SINGLE_MODALITY_INPUT)",
    )
    p_cat.add_argument(
        "--chain",
        metavar="ID",
        help="Lookup compound motif chain by registry ID (e.g. CHAIN-PATCH-ATTACK-SURFACE)",
    )
    p_cat.add_argument(
        "--coverage",
        action="store_true",
        help="Show research corpus completion status (notes vs registry)",
    )
    p_cat.add_argument(
        "--category",
        metavar="NAME",
        help='List literature techniques in a category (e.g. "Adversarial Examples")',
    )
    p_cat.add_argument(
        "--technique",
        metavar="ID",
        help="Lookup a literature technique by ID (e.g. AML-ADV-001)",
    )
    p_cat.set_defaults(func=cmd_catalog)

    # operators
    p_ops = _subparser(
        sub,
        "operators",
        help_text="ONNX operator reference",
        description=(
            "List ONNX operators GraphSurgeon recognizes, or fetch metadata for one op."
        ),
        epilog=_OPERATORS_EPILOG,
    )
    p_ops.add_argument(
        "--op",
        metavar="OP_TYPE",
        help="Show JSON metadata for one operator (e.g. Conv, Gemm, Softmax)",
    )
    p_ops.set_defaults(func=cmd_operators)

    # edit
    p_edit = _subparser(
        sub,
        "edit",
        help_text="Counterfactual graph edits and validation",
        description=(
            "Apply or validate counterfactual edits to ONNX graphs. "
            "Subcommands: validate (check graph integrity), remove-node (surgical removal)."
        ),
        epilog=_EDIT_EPILOG,
    )
    edit_sub = p_edit.add_subparsers(
        dest="edit_cmd",
        required=True,
        title="edit subcommands",
        metavar="{validate,remove-node}",
    )
    p_val = edit_sub.add_parser(
        "validate",
        help="Validate an edited ONNX graph",
        description="Check structural integrity and optionally load/run the graph.",
        formatter_class=_FORMATTER,
        epilog=_EDIT_VALIDATE_EPILOG,
    )
    p_val.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model to validate (required)",
    )
    p_val.add_argument(
        "--level",
        default="structural",
        choices=["none", "structural", "loadable", "runnable"],
        help=(
            "Validation depth: none (skip), structural (graph shape), "
            "loadable (onnxruntime session), runnable (sample inference). "
            "Default: structural"
        ),
    )
    p_val.set_defaults(func=cmd_edit_validate)
    p_rm = edit_sub.add_parser(
        "remove-node",
        help="Remove a node and rewire consumers",
        description="Remove one node by name, rewire edges, and write a new ONNX file.",
        formatter_class=_FORMATTER,
        epilog=_EDIT_REMOVE_EPILOG,
    )
    p_rm.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Source ONNX model (required)",
    )
    p_rm.add_argument(
        "node",
        metavar="NODE_NAME",
        help="Exact ONNX node name to remove (required; use inspect/topology to discover names)",
    )
    p_rm.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="OUT.onnx",
        help="Output path for the edited model (required)",
    )
    p_rm.set_defaults(func=cmd_edit_remove_node)

    # export-scene
    p_scene = _subparser(
        sub,
        "export-scene",
        help_text="Export SceneGraph JSON for 3D visualization",
        description="Build a SceneGraph JSON file from an ONNX model for the 3D viewer.",
        epilog=_EXPORT_SCENE_EPILOG,
    )
    p_scene.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_scene.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write SceneGraph JSON to this file (default: stdout)",
    )
    p_scene.add_argument(
        "--no-motifs",
        action="store_true",
        help="Skip motif/gadget analysis (faster, topology-only scene)",
    )
    p_scene.add_argument(
        "--weights",
        action="store_true",
        help="Include parameter counts from weight tensors (drives node sizing)",
    )
    p_scene.set_defaults(func=cmd_export_scene)

    # serve
    p_serve = _subparser(
        sub,
        "serve",
        help_text="Launch 3D viewer with live API (requires [viz] extra)",
        description=(
            "Start a FastAPI server that serves the 3D viewer and exposes a JSON API "
            "for scene data, node details, catalog lookups, and counterfactual editing."
        ),
        epilog=_SERVE_EPILOG,
    )
    p_serve.add_argument(
        "model",
        metavar="MODEL.onnx",
        help="Path to the ONNX model file (required)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port to listen on (default: 8088)",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    p_serve.set_defaults(func=cmd_serve)

    # diff
    p_diff = _subparser(
        sub,
        "diff",
        help_text="Compare two ONNX models",
        description="Diff node sets, shapes, and topology between a baseline and edited model.",
        epilog=_DIFF_EPILOG,
    )
    p_diff.add_argument(
        "model_a",
        metavar="BASELINE.onnx",
        help="Baseline or original model path (required)",
    )
    p_diff.add_argument(
        "model_b",
        metavar="EDITED.onnx",
        help="Edited or candidate model path (required)",
    )
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_inspect(args):
    from graph_surgeon.parsers.onnx_parser import ONNXGraphParser

    if not os.path.exists(args.model):
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        return 1
    parser = ONNXGraphParser()
    g = parser.parse_file(args.model)
    op_counts = {}
    for n in g.nodes:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    print(f"Graph: {g.name}")
    print(f"Nodes: {len(g.nodes)}")
    print(f"Inputs: {[i.name for i in g.inputs]}")
    print(f"Outputs: {[o.name for o in g.outputs]}")
    print(f"Initializers: {len(g.initializers)}")
    print("Op counts:")
    for op, c in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {c}")
    return 0


def cmd_topology(args):
    from graph_surgeon.graph.surgeon import GraphSurgeon
    from graph_surgeon.graph.topology import LayerPosition

    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(args.model)
    topo = surgeon.get_graph_topology(model.graph)
    if args.json:
        out = {
            "total_nodes": topo.total_nodes,
            "max_depth": topo.max_depth,
            "early": topo.by_position[LayerPosition.EARLY],
            "middle": topo.by_position[LayerPosition.MIDDLE],
            "late": topo.by_position[LayerPosition.LATE],
            "by_op_type": topo.by_op_type,
            "execution_order": topo.execution_order,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"Nodes: {topo.total_nodes}, max depth: {topo.max_depth}")
        print(f"Early ({len(topo.by_position[LayerPosition.EARLY])}): "
              f"{topo.by_position[LayerPosition.EARLY][:8]}...")
        print(f"Late ({len(topo.by_position[LayerPosition.LATE])}): "
              f"{topo.by_position[LayerPosition.LATE][:8]}...")
    return 0


def cmd_motifs(args):
    from graph_surgeon.parsers.onnx_parser import analyze_onnx_graph

    report = analyze_onnx_graph(args.model, output_path=args.output, verbose=True)
    if args.flow and report.model_flow_description:
        print(report.model_flow_description)
    return 0


def cmd_patterns(args):
    import os

    from graph_surgeon.parsers.onnx_parser import analyze_onnx_patterns
    from graph_surgeon.reporting.sanitize import serialize_for_export

    if not os.path.exists(args.model):
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        return 1

    report = analyze_onnx_patterns(args.model)
    if args.json:
        print(json.dumps(serialize_for_export(report), indent=2))
    else:
        from graph_surgeon.analysis.patterns import generate_structural_report_text

        text = generate_structural_report_text(report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Report saved to: {args.output}")
        else:
            print(text)
    return 0


def cmd_flow(args):
    from graph_surgeon.parsers.onnx_parser import analyze_onnx_graph

    report = analyze_onnx_graph(args.model, verbose=False)
    print(report.model_flow_description or "(no flow description generated)")
    return 0


def cmd_catalog(args):
    from graph_surgeon.taxonomy import motif_catalog
    from graph_surgeon.taxonomy.display import format_catalog_chain, format_catalog_gadget
    from graph_surgeon.taxonomy.research_coverage import format_coverage_report

    if args.coverage:
        print(format_coverage_report())
        return 0
    if args.gadget:
        print(format_catalog_gadget(args.gadget))
        return 0
    if args.chain:
        print(format_catalog_chain(args.chain))
        return 0
    if args.technique:
        t = motif_catalog.get_technique_by_id(args.technique)
        if not t:
            print(f"Unknown technique: {args.technique}", file=sys.stderr)
            return 1
        print(f"{t.id}: {t.name}\n{t.description}")
        return 0
    if args.category:
        for t in motif_catalog.get_techniques_by_category(args.category):
            print(f"  {t.id}: {t.name}")
        return 0
    motif_catalog.print_taxonomy_summary()
    return 0


def cmd_operators(args):
    from graph_surgeon.analysis.operators import get_operator_info, list_operators

    if args.op:
        info = get_operator_info(args.op)
        print(json.dumps(info, indent=2, default=str))
    else:
        for op in list_operators():
            print(op)
    return 0


def cmd_edit_validate(args):
    from graph_surgeon.graph.surgeon import GraphSurgeon
    from graph_surgeon.graph.validation import GraphValidationLevel

    level_map = {
        "none": GraphValidationLevel.NONE,
        "structural": GraphValidationLevel.STRUCTURAL,
        "loadable": GraphValidationLevel.LOADABLE,
        "runnable": GraphValidationLevel.RUNNABLE,
    }
    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(args.model)
    result = surgeon.validate(model, level=level_map[args.level])
    print(f"valid={result.valid} level={result.level.value}")
    for e in result.errors:
        print(f"  error: {e}")
    for w in result.warnings:
        print(f"  warning: {w}")
    return 0 if result.valid else 1


def cmd_edit_remove_node(args):
    from graph_surgeon.graph.surgeon import GraphSurgeon

    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(args.model)
    result = surgeon.remove_node(model, args.node)
    if not result.success:
        print(result.message, file=sys.stderr)
        return 1
    surgeon.save_model(model, args.output)
    print(f"Saved: {args.output} ({result.message})")
    return 0


def cmd_export_scene(args):
    from graph_surgeon.scene.builder import build_scene

    if not os.path.exists(args.model):
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        return 1

    scene = build_scene(
        args.model,
        include_motifs=not args.no_motifs,
        include_weights=args.weights,
    )
    blob = json.dumps(scene.to_dict(), indent=2)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(blob)
            f.write("\n")
        print(f"Scene written to: {args.output}", file=sys.stderr)
    else:
        print(blob)
    return 0


def cmd_serve(args):
    if not os.path.exists(args.model):
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        return 1

    try:
        import uvicorn
        from graph_surgeon.server.app import create_app
    except ImportError:
        print(
            "Error: serve requires the [viz] extra.\n"
            "Install with: pip install graph-surgeon[viz]",
            file=sys.stderr,
        )
        return 1

    app = create_app(args.model)
    print(f"Serving {args.model} at http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_diff(args):
    from graph_surgeon.graph.surgeon import GraphSurgeon

    surgeon = GraphSurgeon(verbose=False)
    a = surgeon.load_model(args.model_a)
    b = surgeon.load_model(args.model_b)
    diff = surgeon.compare_graphs(a, b)
    print(diff.get("summary", diff))
    print(json.dumps(diff, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
