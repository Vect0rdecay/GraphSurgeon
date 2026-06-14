import * as THREE from 'three';
import type { SceneGraph, SceneNode } from './types';
import type { BuiltScene } from './scene-builder';
import { getCategoryColor } from './colors';

interface DepthLevel {
  visualDepth: number;
  yPosition: number;
  position: string;
  nodes: SceneNode[];
}

let active = false;
let currentIndex = -1;
let allNodes: SceneNode[] = [];
let depthLevels: DepthLevel[] = [];
let originalScales: Map<string, number> = new Map();
let built: BuiltScene | null = null;
let cam: THREE.PerspectiveCamera | null = null;
let ctrl: { target: THREE.Vector3; update: () => void } | null = null;
let nodeClickCb: ((nodeId: string) => void) | null = null;

export function initFlow(scene: SceneGraph) {
  allNodes = [...scene.nodes];
}

export function startFlow(
  b: BuiltScene,
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; update: () => void },
  onNodeClick?: (nodeId: string) => void,
) {
  built = b;
  cam = camera;
  ctrl = controls;
  nodeClickCb = onNodeClick || null;
  active = true;
  currentIndex = -1;
  originalScales.clear();

  // Build depth levels from actual visual Y positions
  const yBuckets = new Map<number, SceneNode[]>();
  for (const node of allNodes) {
    const mesh = b.nodeObjects.get(node.id);
    if (!mesh) continue;
    // Round Y to nearest 0.5 to bucket nodes at same visual level
    const yKey = Math.round(mesh.position.y * 2) / 2;
    const list = yBuckets.get(yKey) || [];
    list.push(node);
    yBuckets.set(yKey, list);
  }

  // Sort by Y descending (top of graph = highest Y = first level)
  const sortedYs = [...yBuckets.keys()].sort((a, b) => b - a);
  const maxDepth = sortedYs.length - 1;

  depthLevels = sortedYs.map((y, i) => {
    const nodes = yBuckets.get(y)!.sort((a, b) => a.exec_index - b.exec_index);
    const frac = maxDepth > 0 ? i / maxDepth : 0;
    const position = frac <= 0.2 ? 'early' : frac >= 0.8 ? 'late' : 'middle';
    return {
      visualDepth: i,
      yPosition: y,
      position,
      nodes,
    };
  });

  for (const [id, mesh] of b.nodeObjects) {
    originalScales.set(id, mesh.scale.x);
  }

  stepTo(0);
}

export function stopFlow() {
  active = false;
  currentIndex = -1;
  if (built) resetNodeVisuals(built);
  hidePanel();
}

export function isFlowPlaying(): boolean {
  return active;
}

export function stepNext() {
  if (!active) return;
  if (currentIndex < depthLevels.length - 1) {
    stepTo(currentIndex + 1);
  } else {
    stopFlow();
  }
}

export function stepPrev() {
  if (!active) return;
  if (currentIndex > 0) {
    stepTo(currentIndex - 1);
  }
}

function stepTo(index: number) {
  if (!built) return;
  currentIndex = index;
  const level = depthLevels[index];
  const activeIds = new Set(level.nodes.map(n => n.id));

  const visitedIds = new Set<string>();
  for (let i = 0; i < index; i++) {
    for (const n of depthLevels[i].nodes) visitedIds.add(n.id);
  }

  for (const [id, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    const baseScale = originalScales.get(id) ?? mesh.scale.x;

    if (activeIds.has(id)) {
      mat.emissive.setHex(0xffffff);
      mat.emissiveIntensity = 3.0;
      mat.opacity = 1.0;
      mesh.scale.setScalar(baseScale * 1.6);
    } else if (visitedIds.has(id)) {
      mat.emissive.copy(mat.color);
      mat.emissiveIntensity = 0.6;
      mat.opacity = 0.6;
      mesh.scale.setScalar(baseScale);
    } else {
      mat.emissive.copy(mat.color);
      mat.emissiveIntensity = 0.1;
      mat.opacity = 0.15;
      mesh.scale.setScalar(baseScale);
    }
  }

  setTimeout(() => {
    if (!built || currentIndex !== index) return;
    for (const node of level.nodes) {
      const mesh = built.nodeObjects.get(node.id);
      if (!mesh) continue;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.emissive.copy(mat.color);
      mat.emissiveIntensity = 2.5;
    }
  }, 200);

  flyToDepth(level);
  showPanel(level, index);
}

function flyToDepth(level: DepthLevel) {
  if (!built || !cam || !ctrl) return;

  const yOffset = cam.position.y - ctrl.target.y;
  ctrl.target.y = level.yPosition;
  cam.position.y = level.yPosition + yOffset;
  ctrl.update();
}

function flyToSingleNode(nodeId: string) {
  if (!built || !cam || !ctrl) return;
  const mesh = built.nodeObjects.get(nodeId);
  if (!mesh) return;

  const yOffset = cam.position.y - ctrl.target.y;
  ctrl.target.set(mesh.position.x, mesh.position.y, mesh.position.z);
  cam.position.y = mesh.position.y + yOffset;
  ctrl.update();
}

function showPanel(level: DepthLevel, index: number) {
  let panel = document.getElementById('flow-depth-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'flow-depth-panel';
    panel.style.cssText = `
      position: absolute;
      top: 16px;
      right: 16px;
      width: 340px;
      max-height: calc(100vh - 32px);
      overflow-x: hidden;
      overflow-y: auto;
      background: rgba(0, 0, 0, 0.93);
      border: 1px solid #0ff;
      border-radius: 4px;
      padding: 14px;
      color: #fff;
      font-family: 'Courier New', monospace;
      font-size: 12px;
      z-index: 20;
      box-shadow: 0 0 24px rgba(0, 255, 255, 0.3);
      word-wrap: break-word;
      overflow-wrap: break-word;
    `;
    document.getElementById('app')!.appendChild(panel);
  }

  const posColor = level.position === 'early' ? '#0ff' : level.position === 'late' ? '#ff6600' : '#0aa';

  const opCounts = new Map<string, number>();
  for (const n of level.nodes) {
    opCounts.set(n.op_type, (opCounts.get(n.op_type) || 0) + 1);
  }
  const opSummary = [...opCounts.entries()]
    .map(([op, count]) => count > 1 ? `${count}× ${op}` : op)
    .join(', ');

  const branchLabel = level.nodes.length > 1
    ? `<span style="color:#ff6600;font-size:10px;border:1px solid #ff6600;padding:0 4px;border-radius:2px;margin-left:6px">⑂ ${level.nodes.length} parallel</span>`
    : '';

  let html = `
    <div style="margin-bottom:8px">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div>
          <span style="color:#0ff;font-size:11px;font-weight:bold">LEVEL ${index + 1} / ${depthLevels.length}</span>
          <span style="color:${posColor};font-size:10px;margin-left:8px;border:1px solid ${posColor};padding:0 4px;border-radius:2px">${level.position.toUpperCase()}</span>
          ${branchLabel}
        </div>
      </div>
      <div style="color:#ccc;font-size:11px;margin-top:4px">${opSummary}</div>
    </div>
  `;

  html += '<div style="display:flex;flex-direction:column;gap:6px">';
  for (const node of level.nodes) {
    const catColor = '#' + getCategoryColor(node.category).toString(16).padStart(6, '0');
    const params = node.param_count > 0 ? `<span style="color:#0aa;font-size:10px;margin-left:6px">${node.param_count.toLocaleString()} params</span>` : '';
    const motifs = node.motif_ids.length > 0
      ? `<div style="color:#ff6600;font-size:10px;margin-top:2px">⚠ ${node.motif_ids.join(', ')}</div>`
      : '';

    const inputList = node.inputs.length > 0
      ? node.inputs.slice(0, 2).map(s => s.split('/').pop() || s).join(', ') + (node.inputs.length > 2 ? ' …' : '')
      : '';
    const outputList = node.outputs.length > 0
      ? node.outputs.slice(0, 2).map(s => s.split('/').pop() || s).join(', ') + (node.outputs.length > 2 ? ' …' : '')
      : '';

    html += `
      <div class="flow-node-row" data-node-id="${node.id}" style="
        padding: 6px 8px;
        border-left: 3px solid ${catColor};
        background: rgba(255,255,255,0.03);
        cursor: pointer;
        border-radius: 0 3px 3px 0;
        transition: background 0.15s;
      " onmouseover="this.style.background='rgba(0,255,255,0.08)'" onmouseout="this.style.background='rgba(255,255,255,0.03)'">
        <div style="display:flex;align-items:center;gap:6px">
          <span style="color:${catColor};font-weight:bold;font-size:13px">${node.op_type}</span>
          <span style="color:#888;font-size:10px">${node.category}</span>
          ${params}
        </div>
        <div style="color:#aaa;font-size:10px;margin-top:1px">${node.id}</div>
        ${inputList || outputList ? `<div style="color:#666;font-size:9px;margin-top:2px">${inputList ? 'in: ' + inputList : ''}${inputList && outputList ? ' → ' : ''}${outputList ? 'out: ' + outputList : ''}</div>` : ''}
        ${motifs}
      </div>
    `;
  }
  html += '</div>';

  panel.innerHTML = html;
  panel.style.display = 'block';

  panel.querySelectorAll('.flow-node-row').forEach(row => {
    row.addEventListener('click', () => {
      const nodeId = (row as HTMLElement).dataset.nodeId;
      if (nodeId) {
        flyToSingleNode(nodeId);
        if (nodeClickCb) nodeClickCb(nodeId);
      }
    });
  });
}

function hidePanel() {
  const panel = document.getElementById('flow-depth-panel');
  if (panel) panel.style.display = 'none';
  const old = document.getElementById('flow-caption');
  if (old) old.style.display = 'none';
}

export function getFlowIndex(): number {
  return currentIndex;
}

export function getFlowTotal(): number {
  return depthLevels.length;
}

function resetNodeVisuals(b: BuiltScene) {
  for (const [id, mesh] of b.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissive.copy(mat.color);
    mat.emissiveIntensity = 0.6;
    mat.opacity = 0.9;
    const baseScale = originalScales.get(id);
    if (baseScale) mesh.scale.setScalar(baseScale);
  }
  originalScales.clear();
}
