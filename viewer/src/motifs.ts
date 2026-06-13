import * as THREE from 'three';
import type { SceneGraph, SceneMotif, SceneChain } from './types';
import type { BuiltScene } from './scene-builder';
import { showRegion, hideAllRegions } from './motif-regions';

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

  showRegion(built.motifRegions, motifId);
}

export function clearHighlight(built: BuiltScene) {
  if (!activeHighlight) return;

  for (const [id, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = originalEmissive.get(id) ?? 0.6;
    mat.opacity = 0.9;
  }

  hideAllRegions(built.motifRegions);
  activeHighlight = null;
  originalEmissive.clear();
}

export function getActiveHighlight(): string | null {
  return activeHighlight;
}

export interface MotifGroup {
  ids: string[];
  title: string;
  type: 'motif' | 'chain';
  count: number;
  significance: string;
}

export function buildMotifGroups(scene: SceneGraph): MotifGroup[] {
  const titleMap = new Map<string, MotifGroup>();

  for (const m of scene.motifs) {
    const existing = titleMap.get(m.title);
    if (existing) {
      existing.ids.push(m.id);
      existing.count++;
    } else {
      titleMap.set(m.title, {
        ids: [m.id],
        title: m.title,
        type: 'motif',
        count: 1,
        significance: m.structural_significance || '',
      });
    }
  }

  for (const c of scene.chains) {
    const title = c.title || c.id;
    titleMap.set(title, {
      ids: [c.id],
      title,
      type: 'chain',
      count: 1,
      significance: c.structural_significance || '',
    });
  }

  return [...titleMap.values()];
}

const SIG_COLORS: Record<string, string> = {
  EXCEPTIONAL: '#ff0033',
  PRIMARY: '#ff00ff',
  SECONDARY: '#00ffff',
  TERTIARY: '#00ff41',
  MITIGATING: '#66ffcc',
};

export function buildMotifList(scene: SceneGraph): HTMLElement {
  const container = document.createElement('div');
  container.id = 'motif-list';

  const groups = buildMotifGroups(scene);

  if (groups.length === 0) {
    container.innerHTML = '<div style="color:#ccc;padding:4px">No motifs detected</div>';
    return container;
  }

  for (const group of groups) {
    const row = document.createElement('div');
    row.className = 'motif-entry';
    row.dataset.motifId = group.ids[0];
    row.dataset.allIds = JSON.stringify(group.ids);

    const badge = group.type === 'chain' ? '⛓' : '◆';
    const badgeColor = group.type === 'chain' ? '#ff6600' : (SIG_COLORS[group.significance] || '#0ff');
    const countLabel = group.count > 1 ? ` <span style="color:#0aa;font-size:10px">(${group.count})</span>` : '';

    row.innerHTML = `
      <span style="color:${badgeColor}">${badge}</span>
      <span class="motif-title">${group.title}${countLabel}</span>
    `;
    row.title = group.ids.join(', ');
    container.appendChild(row);
  }

  return container;
}
