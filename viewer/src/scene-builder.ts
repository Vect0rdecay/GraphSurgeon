import * as THREE from 'three';
import type { SceneGraph, SceneNode } from './types';
import { computeLayout, type NodePosition } from './layout';
import { getCategoryColor } from './colors';
import { getGeometryForOp } from './shapes';
import { buildMotifRegions, type MotifRegions } from './motif-regions';
import { buildShadowLogicMarkers, type ShadowLogicMarkers } from './shadowlogic';

export interface BuiltScene {
  group: THREE.Group;
  nodeObjects: Map<string, THREE.Mesh>;
  edgeLines: THREE.Group;
  positions: Map<string, NodePosition>;
  labelSprites: THREE.Sprite[];
  motifRegions: MotifRegions;
  shadowlogicMarkers: ShadowLogicMarkers | null;
}

export function buildThreeScene(data: SceneGraph): BuiltScene {
  const group = new THREE.Group();
  const nodeObjects = new Map<string, THREE.Mesh>();
  const positions = computeLayout(data);
  const labelSprites: THREE.Sprite[] = [];

  for (const node of data.nodes) {
    const pos = positions.get(node.id);
    if (!pos) continue;

    const color = getCategoryColor(node.category);
    const geom = getGeometryForOp(node.op_type);

    const size = node.param_count > 0
      ? 0.3 + Math.log10(node.param_count + 1) * 0.08
      : 0.4;

    const mesh = new THREE.Mesh(
      geom,
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.6,
        metalness: 0.3,
        roughness: 0.4,
        transparent: true,
        opacity: 0.9,
      }),
    );
    mesh.scale.setScalar(size);
    mesh.position.set(pos.x, pos.y, pos.z);
    mesh.userData = { nodeData: node };

    nodeObjects.set(node.id, mesh);
    group.add(mesh);

    const sprite = makeLabel(node, color);
    sprite.position.set(pos.x, pos.y + size * 0.8 + 0.4, pos.z);
    labelSprites.push(sprite);
    group.add(sprite);
  }

  const edgeLines = buildEdges(data, positions);
  group.add(edgeLines);

  const motifRegions = buildMotifRegions(data, positions);
  group.add(motifRegions.allGroup);

  let shadowlogicMarkers: ShadowLogicMarkers | null = null;
  if (data.shadowlogic && data.shadowlogic.injection_points.length > 0) {
    shadowlogicMarkers = buildShadowLogicMarkers(data.shadowlogic, positions);
    group.add(shadowlogicMarkers.group);
  }

  return { group, nodeObjects, edgeLines, positions, labelSprites, motifRegions, shadowlogicMarkers };
}

function buildEdges(data: SceneGraph, positions: Map<string, NodePosition>): THREE.Group {
  const edgeGroup = new THREE.Group();

  for (const edge of data.edges) {
    const srcPos = positions.get(edge.source);
    const tgtPos = positions.get(edge.target);

    if (!srcPos || !tgtPos) continue;

    const src = new THREE.Vector3(srcPos.x, srcPos.y, srcPos.z);
    const tgt = new THREE.Vector3(tgtPos.x, tgtPos.y, tgtPos.z);

    const depthDiff = Math.abs(
      (data.nodes.find(n => n.id === edge.source)?.depth ?? 0) -
      (data.nodes.find(n => n.id === edge.target)?.depth ?? 0)
    );
    const isSkip = depthDiff > 1;

    const mid = new THREE.Vector3().lerpVectors(src, tgt, 0.5);
    if (isSkip) {
      mid.x += (tgt.x - src.x) * 0.5 + 2.0;
    }

    const curve = new THREE.QuadraticBezierCurve3(src, mid, tgt);
    const points = curve.getPoints(24);

    const elemCount = edge.shape.reduce((a, b) => a * b, 1);
    const lineWidth = elemCount > 0 ? 1 + Math.log10(elemCount) * 0.3 : 1;

    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: isSkip ? 0xffb454 : 0x7fd4ff,
      transparent: true,
      opacity: isSkip ? 0.7 : 0.4,
      linewidth: lineWidth,
    });

    const line = new THREE.Line(geom, mat);
    edgeGroup.add(line);
  }

  return edgeGroup;
}

function makeLabel(node: SceneNode, color: number): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  canvas.width = 256;
  canvas.height = 64;

  ctx.fillStyle = 'transparent';
  ctx.fillRect(0, 0, 256, 64);

  const hex = '#' + color.toString(16).padStart(6, '0');
  ctx.font = '600 16px IBM Plex Mono, monospace';
  ctx.fillStyle = hex;
  ctx.shadowColor = hex;
  ctx.shadowBlur = 2;
  ctx.textAlign = 'center';
  ctx.fillText(node.op_type, 128, 24);

  ctx.font = '12px IBM Plex Mono, monospace';
  ctx.fillStyle = '#9fb8cc';
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.fillText(node.id, 128, 46);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;

  const mat = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2, 0.5, 1);

  return sprite;
}
