#!/usr/bin/env python3
"""GraphSurgeon CLI: ONNX DAG reverse engineering."""

import argparse
import json
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="graph-surgeon",
        description="Inspect, map, and experiment on ONNX computational DAGs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Model summary: I/O, op counts")
    p_inspect.add_argument("model")
    p_inspect.set_defaults(func=cmd_inspect)

    # topology
    p_topo = sub.add_parser("topology", help="Depth, early/middle/late, execution order")
    p_topo.add_argument("model")
    p_topo.add_argument("--json", action="store_true")
    p_topo.set_defaults(func=cmd_topology)

    # motifs
    p_motifs = sub.add_parser("motifs", help="Structural motif scan")
    p_motifs.add_argument("model")
    p_motifs.add_argument("-o", "--output", help="JSON output path")
    p_motifs.add_argument("--flow", action="store_true")
    p_motifs.set_defaults(func=cmd_motifs)

    # patterns
    p_pat = sub.add_parser("patterns", help="High-level DAG structural patterns")
    p_pat.add_argument("model")
    p_pat.set_defaults(func=cmd_patterns)

    # flow
    p_flow = sub.add_parser("flow", help="Plain-English execution narrative")
    p_flow.add_argument("model")
    p_flow.set_defaults(func=cmd_flow)

    # catalog
    p_cat = sub.add_parser("catalog", help="Motif and technique reference")
    p_cat.add_argument("--category")
    p_cat.add_argument("--technique")
    p_cat.set_defaults(func=cmd_catalog)

    # operators
    p_ops = sub.add_parser("operators", help="ONNX operator reference")
    p_ops.add_argument("--op")
    p_ops.set_defaults(func=cmd_operators)

    # edit
    p_edit = sub.add_parser("edit", help="Counterfactual graph edits")
    edit_sub = p_edit.add_subparsers(dest="edit_cmd", required=True)
    p_val = edit_sub.add_parser("validate", help="Validate edited graph")
    p_val.add_argument("model")
    p_val.add_argument("--level", default="structural", choices=["none", "structural", "loadable", "runnable"])
    p_val.set_defaults(func=cmd_edit_validate)
    p_rm = edit_sub.add_parser("remove-node", help="Remove a node and rewire")
    p_rm.add_argument("model")
    p_rm.add_argument("node")
    p_rm.add_argument("-o", "--output", required=True)
    p_rm.set_defaults(func=cmd_edit_remove_node)

    # diff
    p_diff = sub.add_parser("diff", help="Compare two ONNX models")
    p_diff.add_argument("model_a")
    p_diff.add_argument("model_b")
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
    from graph_surgeon.analysis.motifs import export_report_json

    report = analyze_onnx_graph(args.model, output_path=args.output, verbose=True)
    if args.flow and report.model_flow_description:
        print(report.model_flow_description)
    if args.output:
        export_report_json(report, args.output)
    return 0


def cmd_patterns(args):
    print("Structural pattern analysis requires full graph context.")
    print(f"Run: graph-surgeon motifs {args.model}")
    return 0


def cmd_flow(args):
    from graph_surgeon.parsers.onnx_parser import analyze_onnx_graph

    report = analyze_onnx_graph(args.model, verbose=False)
    print(report.model_flow_description or "(no flow description generated)")
    return 0


def cmd_catalog(args):
    from graph_surgeon.taxonomy import motif_catalog

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
