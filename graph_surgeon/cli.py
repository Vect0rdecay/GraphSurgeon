#!/usr/bin/env python3
"""GraphSurgeon CLI: ONNX DAG reverse-engineering commands."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _require_model(path: str) -> str:
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def cmd_inspect(args: argparse.Namespace) -> None:
    from graph_surgeon.parsers.onnx_parser import ONNXGraphParser

    path = _require_model(args.model)
    parser = ONNXGraphParser()
    graph = parser.parse_file(path)

    op_counts: dict[str, int] = {}
    for node in graph.nodes:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"Model: {path}")
    print(f"Graph: {graph.name}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Inputs: {[i.name for i in graph.inputs]}")
    print(f"Outputs: {[o.name for o in graph.outputs]}")
    print(f"Initializers: {len(graph.initializers)}")
    print("\nOperator counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {op}: {count}")


def cmd_topology(args: argparse.Namespace) -> None:
    from graph_surgeon import GraphSurgeon
    from graph_surgeon.graph.topology import LayerPosition

    path = _require_model(args.model)
    surgeon = GraphSurgeon(verbose=False)
    model = surgeon.load_model(path)
    topo = surgeon.get_graph_topology(model.graph)

    print(f"Topology: {path}")
    print(f"Total nodes: {topo.total_nodes}")
    print(f"Max depth: {topo.max_depth}")
    for pos in LayerPosition:
        names = topo.by_position[pos]
        print(f"\n{pos.value.upper()} ({len(names)} nodes):")
        for name in names[:20]:
            info = topo.nodes[name]
            print(f"  [{info.depth}] {name} ({info.op_type})")
        if len(names) > 20:
            print(f"  ... and {len(names) - 20} more")

    if args.json:
        payload = {
            "max_depth": topo.max_depth,
            "total_nodes": topo.total_nodes,
            "by_position": {p.value: topo.by_position[p] for p in LayerPosition},
            "execution_order": topo.execution_order,
        }
        print(json.dumps(payload, indent=2))


def cmd_motifs(args: argparse.Namespace) -> None:
    from graph_surgeon.parsers.onnx_parser import analyze_model_motifs
    from graph_surgeon.analysis.motifs import export_report_json

    path = _require_model(args.model)
    report = analyze_model_motifs(path, verbose=not args.quiet)

    if args.output:
        export_report_json(report, args.output)
        if not args.quiet:
            print(f"Report written to {args.output}")
    else:
        print(report.executive_summary)


def cmd_patterns(args: argparse.Namespace) -> None:
    from graph_surgeon.parsers.onnx_parser import ONNXGraphParser
    from graph_surgeon.analysis.patterns import StructuralPatternAnalyzer

    path = _require_model(args.model)
    parser = ONNXGraphParser()
    graph = parser.parse_file(path)

    analyzer = StructuralPatternAnalyzer()
    report = analyzer.analyze(
        nodes=[
            {"node_id": n.name, "op_type": n.op_type, "attributes": n.attributes}
            for n in graph.nodes
        ],
        edges=parser.get_edges(),
        model_name=path,
    )

    print(f"Structural patterns: {path}")
    print(f"High-risk patterns: {len(report.high_risk_patterns)}")
    print(f"Robustness indicators: {len(report.robustness_indicators)}")
    print(f"Structural score: {report.structural_score:.1f}")
    for pattern in report.high_risk_patterns[:10]:
        print(f"  [{pattern.risk.value}] {pattern.name}")


def cmd_flow(args: argparse.Namespace) -> None:
    from graph_surgeon.parsers.onnx_parser import analyze_model_motifs

    path = _require_model(args.model)
    report = analyze_model_motifs(path, verbose=False)
    print(report.model_flow_description)


def cmd_catalog(args: argparse.Namespace) -> None:
    from graph_surgeon.taxonomy.techniques import (
        get_all_techniques,
        get_technique_by_id,
        get_techniques_by_category,
        print_taxonomy_summary,
    )

    if args.summary:
        print_taxonomy_summary()
        return

    if args.technique:
        technique = get_technique_by_id(args.technique)
        if not technique:
            print(f"Technique not found: {args.technique}", file=sys.stderr)
            sys.exit(1)
        print(f"{technique.id}: {technique.name}")
        print(technique.description.strip())
        return

    if args.category:
        techniques = get_techniques_by_category(args.category)
        for t in techniques:
            print(f"[{t.id}] {t.name}")
        return

    for t in get_all_techniques():
        print(f"[{t.id}] {t.name} ({t.category})")


def cmd_operators(args: argparse.Namespace) -> None:
    from graph_surgeon.taxonomy.operators import OPERATOR_REFERENCE_DB

    if args.op:
        info = OPERATOR_REFERENCE_DB.get(args.op)
        if not info:
            print(f"Operator not found: {args.op}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"operator": args.op, **info}, indent=2, default=str))
        return

    for op in sorted(OPERATOR_REFERENCE_DB.keys()):
        info = OPERATOR_REFERENCE_DB[op]
        print(f"{op}: {info.get('category', 'unknown')}")


def cmd_edit(args: argparse.Namespace) -> None:
    from graph_surgeon import GraphSurgeon, GraphValidationLevel

    surgeon = GraphSurgeon(verbose=args.verbose)

    if args.edit_command == "validate":
        path = _require_model(args.model)
        level = GraphValidationLevel[args.level.upper()]
        model = surgeon.load_model(path)
        result = surgeon.validate(model, level=level)
        print(f"Valid: {result.valid} (level={result.level.value})")
        if result.errors:
            for err in result.errors:
                print(f"  ERROR: {err}")
        if result.warnings:
            for warn in result.warnings:
                print(f"  WARN: {warn}")
        sys.exit(0 if result.valid else 1)

    if args.edit_command == "remove-subgraph":
        path = _require_model(args.model)
        model = surgeon.load_model(path)
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
        result = surgeon.remove_subgraph(model, nodes)
        if not result.success:
            print(f"Edit failed: {result.message}", file=sys.stderr)
            sys.exit(1)
        out = args.output or path.replace(".onnx", "_edited.onnx")
        surgeon.save_model(model, out)
        print(f"Removed {result.nodes_removed}; saved to {out}")
        return

    print(f"Unknown edit command: {args.edit_command}", file=sys.stderr)
    sys.exit(1)


def cmd_diff(args: argparse.Namespace) -> None:
    from graph_surgeon import GraphSurgeon

    a = _require_model(args.model_a)
    b = _require_model(args.model_b)
    surgeon = GraphSurgeon(verbose=False)
    diff = surgeon.compare_graphs(surgeon.load_model(a), surgeon.load_model(b))
    if args.json:
        print(json.dumps(diff, indent=2))
    else:
        print(diff["summary"])
        if diff["nodes_added"]:
            print(f"Added: {', '.join(diff['nodes_added'])}")
        if diff["nodes_removed"]:
            print(f"Removed: {', '.join(diff['nodes_removed'])}")
        if diff["nodes_modified"]:
            print(f"Modified: {len(diff['nodes_modified'])} nodes")


def cmd_probe(args: argparse.Namespace) -> None:
    print("probe: install graph-surgeon[behavior] (stub in v0.1)", file=sys.stderr)
    sys.exit(2)


def cmd_characterize(args: argparse.Namespace) -> None:
    print("characterize: install graph-surgeon[behavior] (stub in v0.1)", file=sys.stderr)
    sys.exit(2)


def cmd_perturb(args: argparse.Namespace) -> None:
    print("perturb: install graph-surgeon[behavior] (stub in v0.1)", file=sys.stderr)
    sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-surgeon",
        description="Inspect, map, and experiment on ONNX computational DAGs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Model summary: I/O, op counts, shapes")
    p.add_argument("model")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("topology", help="Depth, early/middle/late, execution order")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_topology)

    p = sub.add_parser("motifs", help="Structural motif scan")
    p.add_argument("model")
    p.add_argument("-o", "--output")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_motifs)

    p = sub.add_parser("patterns", help="High-level DAG structural patterns")
    p.add_argument("model")
    p.set_defaults(func=cmd_patterns)

    p = sub.add_parser("flow", help="Plain-English execution narrative")
    p.add_argument("model")
    p.set_defaults(func=cmd_flow)

    p = sub.add_parser("catalog", help="Motif and technique reference")
    p.add_argument("--category")
    p.add_argument("--technique")
    p.add_argument("--summary", action="store_true")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("operators", help="ONNX operator reference")
    p.add_argument("--op")
    p.set_defaults(func=cmd_operators)

    p_edit = sub.add_parser("edit", help="Counterfactual graph edits")
    edit_sub = p_edit.add_subparsers(dest="edit_command", required=True)

    p = edit_sub.add_parser("validate", help="Validate edited graph")
    p.add_argument("model")
    p.add_argument("--level", default="structural", choices=["none", "structural", "loadable", "runnable"])
    p.set_defaults(func=cmd_edit)

    p = edit_sub.add_parser("remove-subgraph", help="Remove nodes and rewire")
    p.add_argument("model")
    p.add_argument("--nodes", required=True, help="Comma-separated node names")
    p.add_argument("-o", "--output")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("diff", help="Compare two ONNX models")
    p.add_argument("model_a")
    p.add_argument("model_b")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_diff)

    for name, fn in [("probe", cmd_probe), ("characterize", cmd_characterize), ("perturb", cmd_perturb)]:
        p = sub.add_parser(name, help=f"Behavior extra: {name} (requires [behavior])")
        p.add_argument("model", nargs="?")
        p.set_defaults(func=fn)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
