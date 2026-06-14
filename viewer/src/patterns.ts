import * as THREE from 'three';
import type { SceneStructuralPatterns } from './types';
import type { BuiltScene } from './scene-builder';

export type PatternCategory = 'gradient_bottlenecks' | 'feature_fusion_points' | 'amplification_layers' | 'recommended_defense_points';

const PATTERN_COLORS: Record<PatternCategory, number> = {
  gradient_bottlenecks: 0xff5359,
  feature_fusion_points: 0x7fd4ff,
  amplification_layers: 0xffb454,
  recommended_defense_points: 0x69e2b0,
};

const PATTERN_LABELS: Record<PatternCategory, string> = {
  gradient_bottlenecks: 'BOTTLENECKS',
  feature_fusion_points: 'FUSION PTS',
  amplification_layers: 'AMPLIFY',
  recommended_defense_points: 'DEFENSE PTS',
};

const PATTERN_DESCRIPTIONS: Record<PatternCategory, string> = {
  gradient_bottlenecks: 'Nodes where gradient flow narrows through a single path — small perturbations here disproportionately affect downstream layers.',
  feature_fusion_points: 'Nodes where multiple feature streams merge — structural leverage points where injected signals from separate branches combine.',
  amplification_layers: 'Nodes with high fan-out or parameter density that amplify signal magnitude — perturbations passing through these grow larger.',
  recommended_defense_points: 'Strategic locations where monitoring or clamping would intercept the widest range of structural anomalies.',
};

const ALL_CATEGORIES: PatternCategory[] = [
  'gradient_bottlenecks',
  'feature_fusion_points',
  'amplification_layers',
  'recommended_defense_points',
];

let originalState: Map<string, { color: number; emissive: number; emissiveIntensity: number; opacity: number }> = new Map();
let activeCategory: PatternCategory | null = null;

export function applyPatternHighlight(
  category: PatternCategory,
  data: SceneStructuralPatterns,
  built: BuiltScene,
) {
  const nodeIds = new Set(data[category]);

  if (activeCategory && activeCategory !== category) {
    restoreState(built);
  }

  if (!originalState.size) {
    for (const [id, mesh] of built.nodeObjects) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      originalState.set(id, {
        color: mat.color.getHex(),
        emissive: mat.emissive.getHex(),
        emissiveIntensity: mat.emissiveIntensity,
        opacity: mat.opacity,
      });
    }
  }

  const highlightColor = PATTERN_COLORS[category];

  for (const [id, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    if (nodeIds.has(id)) {
      mat.color.setHex(highlightColor);
      mat.emissive.setHex(highlightColor);
      mat.emissiveIntensity = 2.0;
      mat.opacity = 1.0;
    } else {
      mat.emissiveIntensity = 0.05;
      mat.opacity = 0.12;
    }
  }

  activeCategory = category;
}

export function clearPatternHighlight(built: BuiltScene) {
  restoreState(built);
  activeCategory = null;
}

function restoreState(built: BuiltScene) {
  for (const [id, mesh] of built.nodeObjects) {
    const orig = originalState.get(id);
    if (!orig) continue;
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.color.setHex(orig.color);
    mat.emissive.setHex(orig.emissive);
    mat.emissiveIntensity = orig.emissiveIntensity;
    mat.opacity = orig.opacity;
  }
  originalState.clear();
}

export function getActiveCategory(): PatternCategory | null {
  return activeCategory;
}

export function buildPatternsPanel(
  data: SceneStructuralPatterns,
  built: BuiltScene,
): HTMLElement {
  const container = document.createElement('div');
  container.id = 'patterns-panel';

  const statsRow = document.createElement('div');
  statsRow.style.cssText = 'color:#9fb8cc;font-size:9px;letter-spacing:.1em;padding:4px 0;border-bottom:1px solid rgba(140,196,255,.10);margin-bottom:6px;';
  statsRow.innerHTML = `
    <span style="color:#c9d6e4">Fan-in:</span> ${data.max_fan_in}
    <span style="color:#c9d6e4;margin-left:8px">Fan-out:</span> ${data.max_fan_out}
    <span style="color:#c9d6e4;margin-left:8px">Chain:</span> ${data.longest_linear_chain}
    <span style="color:#c9d6e4;margin-left:8px">Score:</span> ${data.structural_score.toFixed(2)}
  `;
  container.appendChild(statsRow);

  for (const cat of ALL_CATEGORIES) {
    const count = data[cat].length;
    if (count === 0) continue;
    const color = '#' + PATTERN_COLORS[cat].toString(16).padStart(6, '0');

    const group = document.createElement('div');
    group.style.cssText = 'margin-bottom:10px;';

    const btn = document.createElement('button');
    btn.textContent = `${PATTERN_LABELS[cat]} (${count})`;
    btn.dataset.category = cat;
    btn.style.cssText = `
      background: rgba(4,8,14,.55); border: 1px solid ${color}; color: ${color};
      font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: .14em;
      padding: 4px 8px; cursor: pointer; transition: .18s; width: 100%; text-align: left;
    `;
    btn.addEventListener('click', () => {
      if (activeCategory === cat) {
        clearPatternHighlight(built);
        updateAllBtnStates(container);
      } else {
        applyPatternHighlight(cat, data, built);
        updateAllBtnStates(container);
      }
    });
    group.appendChild(btn);

    const desc = document.createElement('div');
    desc.style.cssText = `font-family:'Spectral',Georgia,serif;font-size:10px;color:#9fb8cc;font-style:italic;margin-top:4px;line-height:1.5;padding-left:2px;`;
    desc.textContent = PATTERN_DESCRIPTIONS[cat];
    group.appendChild(desc);

    container.appendChild(group);
  }

  return container;
}

function updateAllBtnStates(container: HTMLElement) {
  container.querySelectorAll('button').forEach(btn => {
    const cat = btn.dataset.category as PatternCategory;
    if (!cat) return;
    const isActive = activeCategory === cat;
    btn.style.background = isActive ? 'rgba(255,255,255,0.1)' : 'rgba(4,8,14,.55)';
    btn.style.fontWeight = isActive ? 'bold' : 'normal';
  });
}

export function getNodePatternCategories(
  nodeId: string,
  data: SceneStructuralPatterns,
): { category: PatternCategory; label: string; description: string; color: string }[] {
  const results: { category: PatternCategory; label: string; description: string; color: string }[] = [];
  for (const cat of ALL_CATEGORIES) {
    if (data[cat].includes(nodeId)) {
      results.push({
        category: cat,
        label: PATTERN_LABELS[cat],
        description: PATTERN_DESCRIPTIONS[cat],
        color: '#' + PATTERN_COLORS[cat].toString(16).padStart(6, '0'),
      });
    }
  }
  return results;
}
