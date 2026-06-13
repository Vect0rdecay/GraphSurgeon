import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import type { SceneGraph, SceneMotif, SceneChain } from './types';
import type { BuiltScene } from './scene-builder';
import type { NodePosition } from './layout';

const SIGNIFICANCE_COLORS: Record<string, number> = {
  EXCEPTIONAL: 0xff0033,
  PRIMARY: 0xff00ff,
  SECONDARY: 0x00ffff,
  TERTIARY: 0x00ff41,
  MITIGATING: 0x66ffcc,
};
const DEFAULT_COLOR = 0xff00ff;

export interface MotifRegions {
  regions: Map<string, THREE.Group>;
  allGroup: THREE.Group;
}

export function buildMotifRegions(
  scene: SceneGraph,
  positions: Map<string, NodePosition>,
): MotifRegions {
  const regions = new Map<string, THREE.Group>();
  const allGroup = new THREE.Group();
  allGroup.name = 'motif-regions';

  const graphBox = computeGraphBBox(positions);
  const graphSize = new THREE.Vector3();
  graphBox.getSize(graphSize);

  const items: Array<{ id: string; node_ids: string[]; significance: string }> = [
    ...scene.motifs.map(m => ({
      id: m.id,
      node_ids: m.node_ids,
      significance: m.structural_significance || '',
    })),
    ...scene.chains.map(c => ({
      id: c.id,
      node_ids: c.node_ids,
      significance: c.structural_significance || '',
    })),
  ];

  for (const item of items) {
    const pts = item.node_ids
      .map(nid => positions.get(nid))
      .filter((p): p is NodePosition => p !== undefined);

    if (pts.length === 0) continue;

    const color = SIGNIFICANCE_COLORS[item.significance] ?? DEFAULT_COLOR;
    const group = new THREE.Group();
    group.name = `region-${item.id}`;

    const span = computeSpan(pts);
    const scattered = (
      (graphSize.x > 0 && span.x / graphSize.x > 0.6) ||
      (graphSize.y > 0 && span.y / graphSize.y > 0.6)
    );

    if (pts.length === 1 || scattered) {
      for (const p of pts) {
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(1.2, 16, 12),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.1,
            depthWrite: false,
            side: THREE.DoubleSide,
          }),
        );
        sphere.position.set(p.x, p.y, p.z);
        sphere.renderOrder = -1;
        group.add(sphere);

        const wireGeo = new THREE.EdgesGeometry(new THREE.SphereGeometry(1.25, 12, 8));
        const wire = new THREE.LineSegments(
          wireGeo,
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.2 }),
        );
        wire.position.copy(sphere.position);
        wire.renderOrder = -1;
        group.add(wire);
      }
    } else {
      const centroid = new THREE.Vector3();
      const vectors: THREE.Vector3[] = [];
      for (const p of pts) {
        const v = new THREE.Vector3(p.x, p.y, p.z);
        centroid.add(v);
        vectors.push(v);
      }
      centroid.divideScalar(pts.length);

      const expanded = vectors.map(v => {
        const dir = v.clone().sub(centroid).normalize();
        return v.clone().add(dir.multiplyScalar(1.0));
      });

      if (expanded.length === 2) {
        const mid = new THREE.Vector3().lerpVectors(expanded[0], expanded[1], 0.5);
        const dist = expanded[0].distanceTo(expanded[1]);
        const capsule = new THREE.Mesh(
          new THREE.CapsuleGeometry(1.0, dist, 8, 12),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.08,
            depthWrite: false,
            side: THREE.DoubleSide,
          }),
        );
        capsule.position.copy(mid);
        capsule.lookAt(expanded[1]);
        capsule.renderOrder = -1;
        group.add(capsule);
      } else {
        try {
          const hullGeo = new ConvexGeometry(expanded);
          const hull = new THREE.Mesh(
            hullGeo,
            new THREE.MeshBasicMaterial({
              color,
              transparent: true,
              opacity: 0.08,
              depthWrite: false,
              side: THREE.DoubleSide,
            }),
          );
          hull.renderOrder = -1;
          group.add(hull);

          const edges = new THREE.LineSegments(
            new THREE.EdgesGeometry(hullGeo),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.25 }),
          );
          edges.renderOrder = -1;
          group.add(edges);
        } catch {
          for (const p of pts) {
            const sphere = new THREE.Mesh(
              new THREE.SphereGeometry(1.2, 16, 12),
              new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.1,
                depthWrite: false,
                side: THREE.DoubleSide,
              }),
            );
            sphere.position.set(p.x, p.y, p.z);
            sphere.renderOrder = -1;
            group.add(sphere);
          }
        }
      }
    }

    group.visible = false;
    allGroup.add(group);
    regions.set(item.id, group);
  }

  return { regions, allGroup };
}

export function showRegion(regions: MotifRegions, id: string) {
  const group = regions.regions.get(id);
  if (group) group.visible = true;
}

export function hideAllRegions(regions: MotifRegions) {
  for (const group of regions.regions.values()) {
    group.visible = false;
  }
}

export function showAllRegions(regions: MotifRegions) {
  for (const group of regions.regions.values()) {
    group.visible = true;
  }
}

export function toggleAllRegions(regions: MotifRegions): boolean {
  const anyVisible = [...regions.regions.values()].some(g => g.visible);
  if (anyVisible) {
    hideAllRegions(regions);
    return false;
  } else {
    showAllRegions(regions);
    return true;
  }
}

function computeGraphBBox(positions: Map<string, NodePosition>): THREE.Box3 {
  const box = new THREE.Box3();
  for (const p of positions.values()) {
    box.expandByPoint(new THREE.Vector3(p.x, p.y, p.z));
  }
  return box;
}

function computeSpan(pts: NodePosition[]): THREE.Vector3 {
  if (pts.length <= 1) return new THREE.Vector3();
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
  }
  return new THREE.Vector3(maxX - minX, maxY - minY, maxZ - minZ);
}
