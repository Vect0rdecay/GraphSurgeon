import * as THREE from 'three';
import type { SceneStructuralPatterns } from './types';
import type { BuiltScene } from './scene-builder';

export type PatternCategory = 'gradient_bottlenecks' | 'feature_fusion_points' | 'amplification_layers' | 'recommended_defense_points';

const PATTERN_COLORS: Record<PatternCategory, number> = {
  gradient_bottlenecks: 0xff3300,
  feature_fusion_points: 0x00ffff,
  amplification_layers: 0xff6600,
  recommended_defense_points: 0x00ff41,
};

const PATTERN_LABELS: Record<PatternCategory, string> = {
  gradient_bottlenecks: 'BOTTLENECKS',
  feature_fusion_points: 'FUSION PTS',
  amplification_layers: 'AMPLIFY',
  recommended_defense_points: 'DEFENSE PTS',
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

  const header = document.createElement('div');
  header.style.cssText = `
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; padding: 6px 0; user-select: none;
  `;
  const total = data.gradient_bottlenecks.length + data.feature_fusion_points.length
    + data.amplification_layers.length + data.recommended_defense_points.length;
  header.innerHTML = `<span style="color:#00ff41;font-weight:bold;font-size:11px">▸ STRUCTURAL PATTERNS (${total})</span>`;

  const body = document.createElement('div');
  body.style.display = 'none';

  let expanded = false;
  header.addEventListener('click', () => {
    expanded = !expanded;
    body.style.display = expanded ? 'block' : 'none';
    header.innerHTML = `<span style="color:#00ff41;font-weight:bold;font-size:11px">${expanded ? '▾' : '▸'} STRUCTURAL PATTERNS (${total})</span>`;
  });

  const statsRow = document.createElement('div');
  statsRow.style.cssText = 'color:#ccc;font-size:10px;padding:4px 0;border-bottom:1px solid rgba(0,255,65,0.2);margin-bottom:6px;';
  statsRow.innerHTML = `
    <span style="color:#00ff41">Fan-in:</span> ${data.max_fan_in}
    <span style="color:#00ff41;margin-left:8px">Fan-out:</span> ${data.max_fan_out}
    <span style="color:#00ff41;margin-left:8px">Chain:</span> ${data.longest_linear_chain}
    <span style="color:#00ff41;margin-left:8px">Score:</span> ${data.structural_score.toFixed(2)}
  `;
  body.appendChild(statsRow);

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;';

  for (const cat of ALL_CATEGORIES) {
    const count = data[cat].length;
    const color = '#' + PATTERN_COLORS[cat].toString(16).padStart(6, '0');
    const btn = document.createElement('button');
    btn.textContent = `${PATTERN_LABELS[cat]} (${count})`;
    btn.dataset.category = cat;
    btn.style.cssText = `
      background: rgba(0,0,0,0.4); border: 1px solid ${color}; color: ${color};
      font-family: 'Courier New', monospace; font-size: 10px;
      padding: 3px 6px; cursor: pointer; border-radius: 2px;
    `;
    btn.addEventListener('click', () => {
      if (activeCategory === cat) {
        clearPatternHighlight(built);
        updateBtnStates(btnRow);
      } else {
        applyPatternHighlight(cat, data, built);
        updateBtnStates(btnRow);
      }
    });
    btnRow.appendChild(btn);
  }

  body.appendChild(btnRow);
  container.appendChild(header);
  container.appendChild(body);
  return container;
}

function updateBtnStates(row: HTMLElement) {
  row.querySelectorAll('button').forEach(btn => {
    const cat = btn.dataset.category as PatternCategory;
    const isActive = activeCategory === cat;
    btn.style.background = isActive ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.4)';
    btn.style.fontWeight = isActive ? 'bold' : 'normal';
  });
}
