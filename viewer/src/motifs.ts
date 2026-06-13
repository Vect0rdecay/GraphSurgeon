import * as THREE from 'three';
import type { SceneGraph, SceneMotif, SceneChain } from './types';
import type { BuiltScene } from './scene-builder';

let activeHighlight: string | null = null;
let originalEmissive = new Map<string, number>();

export function highlightMotif(
  motifId: string,
  scene: SceneGraph,
  built: BuiltScene,
) {
  clearHighlight(built);

  const motif = scene.motifs.find(m => m.id === motifId);
  const chain = scene.chains.find(c => c.id === motifId);
  const nodeIds = new Set(motif?.node_ids ?? chain?.node_ids ?? []);

  if (nodeIds.size === 0) return;

  activeHighlight = motifId;

  for (const [id, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    originalEmissive.set(id, mat.emissiveIntensity);

    if (nodeIds.has(id)) {
      mat.emissiveIntensity = 2.0;
      mat.opacity = 1.0;
    } else {
      mat.emissiveIntensity = 0.1;
      mat.opacity = 0.2;
    }
  }
}

export function clearHighlight(built: BuiltScene) {
  if (!activeHighlight) return;

  for (const [id, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = originalEmissive.get(id) ?? 0.6;
    mat.opacity = 0.9;
  }

  activeHighlight = null;
  originalEmissive.clear();
}

export function getActiveHighlight(): string | null {
  return activeHighlight;
}

export function buildMotifList(scene: SceneGraph): HTMLElement {
  const container = document.createElement('div');
  container.id = 'motif-list';

  const items = [
    ...scene.motifs.map(m => ({ id: m.id, title: m.title, type: 'motif' as const, desc: m.description })),
    ...scene.chains.map(c => ({ id: c.id, title: c.id, type: 'chain' as const, desc: '' })),
  ];

  if (items.length === 0) {
    container.innerHTML = '<div style="color:#666;padding:4px">No motifs detected</div>';
    return container;
  }

  for (const item of items) {
    const row = document.createElement('div');
    row.className = 'motif-entry';
    row.dataset.motifId = item.id;

    const badge = item.type === 'chain' ? '⛓' : '◆';
    const badgeColor = item.type === 'chain' ? '#ff6600' : '#0ff';

    row.innerHTML = `
      <span style="color:${badgeColor}">${badge}</span>
      <span class="motif-title">${item.title}</span>
    `;
    row.title = item.desc || item.id;
    container.appendChild(row);
  }

  return container;
}
