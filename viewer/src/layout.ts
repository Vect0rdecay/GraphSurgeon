import type { SceneGraph, SceneNode, SceneEdge } from './types';

export interface NodePosition {
  x: number;
  y: number;
  z: number;
}

const Y_SPACING = 3.0;
const X_SPACING = 3.5;
const HELIX_RADIUS = 5.0;
const HELIX_ANGLE_STEP = 0.8; // ~46° per step, full turn every ~8 nodes

export function computeLayout(scene: SceneGraph): Map<string, NodePosition> {
  const positions = new Map<string, NodePosition>();

  const byDepth = new Map<number, SceneNode[]>();
  for (const node of scene.nodes) {
    const list = byDepth.get(node.depth) ?? [];
    list.push(node);
    byDepth.set(node.depth, list);
  }

  const maxNodesPerDepth = Math.max(...[...byDepth.values()].map(v => v.length), 1);
  const useHelix = maxNodesPerDepth <= 2;

  const parentX = new Map<string, number>();

  const outputToNode = new Map<string, string>();
  for (const node of scene.nodes) {
    for (const out of node.outputs) {
      outputToNode.set(out, node.id);
    }
  }

  const depths = [...byDepth.keys()].sort((a, b) => a - b);
  let helixIndex = 0;

  for (const depth of depths) {
    const nodes = byDepth.get(depth)!;
    nodes.sort((a, b) => {
      const aMedian = medianParentX(a, outputToNode, parentX);
      const bMedian = medianParentX(b, outputToNode, parentX);
      return aMedian - bMedian;
    });

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      let x: number;
      let z: number;

      if (useHelix) {
        const angle = helixIndex * HELIX_ANGLE_STEP;
        x = HELIX_RADIUS * Math.cos(angle);
        z = HELIX_RADIUS * Math.sin(angle);
        helixIndex++;
      } else {
        const count = nodes.length;
        const offset = -(count - 1) * X_SPACING / 2;
        x = offset + i * X_SPACING;
        z = 0;
      }

      const y = -depth * Y_SPACING;

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
