import * as THREE from 'three';
import type { SceneGraph } from './types';
import type { NodePosition } from './layout';

const SIGNIFICANCE_COLORS: Record<string, number> = {
  EXCEPTIONAL: 0xff0033,
  PRIMARY: 0xff6600,
  SECONDARY: 0x00ffff,
  TERTIARY: 0x00ff41,
  MITIGATING: 0x66ffcc,
};
const DEFAULT_COLOR = 0x00ffff;

export interface MotifRegions {
  regions: Map<string, THREE.Group>;
  allGroup: THREE.Group;
  hitMeshes: THREE.Mesh[];
  animate: (time: number) => void;
}

export function buildMotifRegions(
  scene: SceneGraph,
  positions: Map<string, NodePosition>,
): MotifRegions {
  const regions = new Map<string, THREE.Group>();
  const allGroup = new THREE.Group();
  allGroup.name = 'motif-regions';
  const hitMeshes: THREE.Mesh[] = [];
  const animatedParts: Array<{ mesh: THREE.Object3D; kind: string; baseY?: number }> = [];

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

    if (pts.length === 1) {
      buildReticle(pts[0], color, group, animatedParts);
    } else {
      buildHazeCloud(pts, color, group, animatedParts);
    }

    for (const p of pts) {
      const hitMesh = new THREE.Mesh(
        new THREE.SphereGeometry(1.5, 8, 6),
        new THREE.MeshBasicMaterial({ visible: false }),
      );
      hitMesh.position.set(p.x, p.y, p.z);
      hitMesh.userData.motifId = item.id;
      hitMesh.userData.isMotifHit = true;
      group.add(hitMesh);
      hitMeshes.push(hitMesh);
    }

    group.visible = false;
    allGroup.add(group);
    regions.set(item.id, group);
  }

  const animate = (time: number) => {
    for (const part of animatedParts) {
      if (!part.mesh.parent?.visible) continue;

      if (part.kind === 'diamond') {
        part.mesh.rotation.y = time * 0.6;
        part.mesh.rotation.x = Math.sin(time * 0.4) * 0.15;
        const pulse = 0.5 + Math.sin(time * 2.0) * 0.3;
        (part.mesh as THREE.LineSegments).material = new THREE.LineBasicMaterial({
          color: (part.mesh as any).userData.color,
          transparent: true,
          opacity: pulse,
        });
      } else if (part.kind === 'haze-sprite') {
        const s = part.mesh as THREE.Sprite;
        const pulse = 0.8 + Math.sin(time * 1.5 + (part.baseY ?? 0)) * 0.2;
        s.material.opacity = 0.12 * pulse;
      }
    }
  };

  return { regions, allGroup, hitMeshes, animate };
}

function buildReticle(
  p: NodePosition,
  color: number,
  group: THREE.Group,
  animated: Array<{ mesh: THREE.Object3D; kind: string; baseY?: number }>,
) {
  const geo = new THREE.OctahedronGeometry(1.3, 0);
  const edges = new THREE.EdgesGeometry(geo);
  const diamond = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.6,
    }),
  );
  diamond.position.set(p.x, p.y, p.z);
  diamond.userData.color = color;
  diamond.renderOrder = -1;
  group.add(diamond);
  animated.push({ mesh: diamond, kind: 'diamond' });
}

function buildHazeCloud(
  pts: NodePosition[],
  color: number,
  group: THREE.Group,
  animated: Array<{ mesh: THREE.Object3D; kind: string; baseY?: number }>,
) {
  const hazeTexture = makeHazeTexture(color);

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: hazeTexture,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    sprite.position.set(p.x, p.y, p.z);
    sprite.scale.set(3.5, 3.5, 1);
    sprite.renderOrder = -1;
    group.add(sprite);
    animated.push({ mesh: sprite, kind: 'haze-sprite', baseY: i * 1.7 });
  }

  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    const midCount = Math.max(1, Math.floor(dist(a, b) / 2.0));

    for (let j = 1; j <= midCount; j++) {
      const t = j / (midCount + 1);
      const mx = a.x + (b.x - a.x) * t;
      const my = a.y + (b.y - a.y) * t;
      const mz = a.z + (b.z - a.z) * t;

      const midSprite = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: hazeTexture,
          transparent: true,
          opacity: 0.08,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      );
      midSprite.position.set(mx, my, mz);
      midSprite.scale.set(2.5, 2.5, 1);
      midSprite.renderOrder = -1;
      group.add(midSprite);
      animated.push({ mesh: midSprite, kind: 'haze-sprite', baseY: (i + t) * 1.7 });
    }
  }

  const ribbonMat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.25,
    linewidth: 1,
  });

  const sorted = [...pts].sort((a, b) => a.y - b.y);
  if (sorted.length >= 2) {
    const ribbonPts = sorted.map(p => new THREE.Vector3(p.x, p.y, p.z));
    const ribbonGeo = new THREE.BufferGeometry().setFromPoints(ribbonPts);
    const ribbon = new THREE.Line(ribbonGeo, ribbonMat);
    ribbon.renderOrder = -1;
    group.add(ribbon);
  }
}

function makeHazeTexture(color: number): THREE.Texture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const r = (color >> 16) & 0xff;
  const g = (color >> 8) & 0xff;
  const b = color & 0xff;

  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, `rgba(${r},${g},${b},1)`);
  gradient.addColorStop(0.3, `rgba(${r},${g},${b},0.4)`);
  gradient.addColorStop(0.7, `rgba(${r},${g},${b},0.1)`);
  gradient.addColorStop(1, `rgba(${r},${g},${b},0)`);

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function dist(a: NodePosition, b: NodePosition): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
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
