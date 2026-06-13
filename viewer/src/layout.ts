import type { SceneGraph, SceneNode, SceneEdge } from './types';

export interface NodePosition {
  x: number;
  y: number;
  z: number;
}

const Y_SPACING = 3.0;
const X_SPACING = 3.5;
const Z_BAND: Record<string, number> = {
  early:  0,
  middle: 8,
  late:   16,
};

export function computeLayout(scene: SceneGraph): Map<string, NodePosition> {
  const positions = new Map<string, NodePosition>();

  const byDepth = new Map<number, SceneNode[]>();
  for (const node of scene.nodes) {
    const list = byDepth.get(node.depth) ?? [];
    list.push(node);
    byDepth.set(node.depth, list);
  }

  const parentX = new Map<string, number>();

  const outputToNode = new Map<string, string>();
  for (const node of scene.nodes) {
    for (const out of node.outputs) {
      outputToNode.set(out, node.id);
    }
  }

  const depths = [...byDepth.keys()].sort((a, b) => a - b);
  for (const depth of depths) {
    const nodes = byDepth.get(depth)!;
    nodes.sort((a, b) => {
      const aMedian = medianParentX(a, outputToNode, parentX);
      const bMedian = medianParentX(b, outputToNode, parentX);
      return aMedian - bMedian;
    });

    const count = nodes.length;
    const offset = -(count - 1) * X_SPACING / 2;

    for (let i = 0; i < count; i++) {
      const node = nodes[i];
      const x = offset + i * X_SPACING;
      const y = -depth * Y_SPACING;
      const z = Z_BAND[node.position] ?? Z_BAND.middle;

      positions.set(node.id, { x, y, z });
      parentX.set(node.id, x);
    }
  }

  return positions;
}

function medianParentX(
  node: SceneNode,
  outputToNode: Map<string, string>,
  parentX: Map<string, number>,
): number {
  const xs: number[] = [];
  for (const inp of node.inputs) {
    const parentId = outputToNode.get(inp);
    if (parentId && parentX.has(parentId)) {
      xs.push(parentX.get(parentId)!);
    }
  }
  if (xs.length === 0) return 0;
  xs.sort((a, b) => a - b);
  return xs[Math.floor(xs.length / 2)];
}
