import * as THREE from 'three';
import type { SceneGraph, SceneNode } from './types';
import type { BuiltScene } from './scene-builder';

let searchInput: HTMLInputElement | null = null;
let resultsList: HTMLElement | null = null;
let pulseTimeouts: number[] = [];

export function initSearch(
  onSelect: (nodeId: string) => void,
) {
  const container = document.createElement('div');
  container.id = 'search-container';
  container.style.cssText = `
    position: absolute;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 20;
    font-family: 'Courier New', monospace;
  `;

  searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.placeholder = 'Search nodes (name or op type)...';
  searchInput.style.cssText = `
    width: 300px;
    background: rgba(0,0,0,0.85);
    border: 1px solid #0ff;
    border-radius: 4px;
    color: #0ff;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    padding: 6px 10px;
    outline: none;
    box-shadow: 0 0 10px rgba(0,255,255,0.15);
  `;
  container.appendChild(searchInput);

  resultsList = document.createElement('div');
  resultsList.id = 'search-results';
  resultsList.style.cssText = `
    background: rgba(0,0,0,0.9);
    border: 1px solid #0ff;
    border-top: none;
    border-radius: 0 0 4px 4px;
    max-height: 200px;
    overflow-y: auto;
    display: none;
  `;
  container.appendChild(resultsList);

  document.getElementById('app')!.appendChild(container);

  const style = document.createElement('style');
  style.textContent = `
    .search-result {
      padding: 4px 10px;
      color: #ddd;
      cursor: pointer;
      font-size: 12px;
    }
    .search-result:hover {
      background: rgba(0,255,255,0.1);
      color: #fff;
    }
    .search-result .op {
      color: #0ff;
      margin-right: 6px;
    }
  `;
  document.head.appendChild(style);

  searchInput.addEventListener('input', () => {
    const query = searchInput!.value.trim().toLowerCase();
    if (query.length < 2) {
      resultsList!.style.display = 'none';
      return;
    }
    // Results populated by updateSearch
  });

  resultsList.addEventListener('click', (e) => {
    const item = (e.target as HTMLElement).closest('.search-result') as HTMLElement;
    if (item) {
      const nodeId = item.dataset.nodeId!;
      onSelect(nodeId);
      searchInput!.value = '';
      resultsList!.style.display = 'none';
    }
  });
}

export function updateSearchResults(scene: SceneGraph) {
  if (!searchInput || !resultsList) return;

  searchInput.addEventListener('input', () => {
    const query = searchInput!.value.trim().toLowerCase();
    if (query.length < 2) {
      resultsList!.style.display = 'none';
      return;
    }

    const matches = scene.nodes.filter(n =>
      n.id.toLowerCase().includes(query) ||
      n.op_type.toLowerCase().includes(query)
    ).slice(0, 20);

    if (matches.length === 0) {
      resultsList!.style.display = 'none';
      return;
    }

    resultsList!.innerHTML = matches.map(n =>
      `<div class="search-result" data-node-id="${n.id}">
        <span class="op">${n.op_type}</span>${n.id}
      </div>`
    ).join('');
    resultsList!.style.display = 'block';
  });
}

export function pulseNode(
  nodeId: string,
  built: BuiltScene,
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; update: () => void },
) {
  const mesh = built.nodeObjects.get(nodeId);
  if (!mesh) return;

  for (const t of pulseTimeouts) clearTimeout(t);
  pulseTimeouts = [];

  const mat = mesh.material as THREE.MeshStandardMaterial;

  for (let i = 0; i < 6; i++) {
    const t = window.setTimeout(() => {
      mat.emissiveIntensity = i % 2 === 0 ? 2.5 : 0.6;
    }, i * 200);
    pulseTimeouts.push(t);
  }

  const pos = mesh.position;
  camera.position.lerp(
    new THREE.Vector3(pos.x, pos.y, pos.z + 15),
    0.5,
  );
  controls.target.copy(pos);
  controls.update();
}
