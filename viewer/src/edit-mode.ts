import * as THREE from 'three';
import type { SceneGraph, SceneNode } from './types';
import type { BuiltScene } from './scene-builder';

export interface EditCallbacks {
  onSceneReload: (newScene: SceneGraph) => void;
  onDiffReceived: (diff: Record<string, unknown>) => void;
}

let apiBase = '';
let callbacks: EditCallbacks | null = null;
let contextMenu: HTMLElement | null = null;
let diffPanel: HTMLElement | null = null;

export function initEditMode(base: string, cbs: EditCallbacks) {
  apiBase = base;
  callbacks = cbs;
  createContextMenu();
  createDiffPanel();
}

export function showContextMenu(node: SceneNode, event: MouseEvent) {
  if (!contextMenu) return;

  contextMenu.innerHTML = `
    <div class="ctx-item" data-action="remove" data-node="${node.id}">
      Remove "${node.id}" (counterfactual)
    </div>
    <div class="ctx-divider"></div>
    <div class="ctx-item" data-action="reset">
      Reset to baseline
    </div>
  `;

  contextMenu.style.left = event.clientX + 'px';
  contextMenu.style.top = event.clientY + 'px';
  contextMenu.style.display = 'block';
}

export function hideContextMenu() {
  if (contextMenu) contextMenu.style.display = 'none';
}

async function removeNode(nodeId: string) {
  hideContextMenu();
  showDiffPanel('Removing node...');

  try {
    const resp = await fetch(`${apiBase}/api/edit/remove-node`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_name: nodeId }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      showDiffPanel(`Error: ${err.detail || 'removal failed'}`);
      return;
    }

    const result = await resp.json();

    let html = '<h3 style="color:#0ff;margin:0 0 8px">SURGERY RESULT</h3>';
    html += `<div style="color:#0f0">${result.surgery.message}</div>`;

    if (result.diff) {
      const d = result.diff;
      if (d.nodes_removed?.length) {
        html += `<div style="color:#f66;margin-top:4px">Removed: ${d.nodes_removed.join(', ')}</div>`;
      }
      if (d.nodes_added?.length) {
        html += `<div style="color:#6f6;margin-top:2px">Added: ${d.nodes_added.join(', ')}</div>`;
      }
      html += `<div style="margin-top:4px;color:#ccc">Nodes: ${d.total_nodes_a} &rarr; ${d.total_nodes_b}</div>`;
    }

    if (result.validation) {
      const v = result.validation;
      const color = v.valid ? '#0f0' : '#f66';
      html += `<div style="margin-top:6px;color:${color}">Validation: ${v.valid ? 'PASS' : 'FAIL'} (${v.level})</div>`;
      for (const e of v.errors || []) {
        html += `<div style="color:#f66;font-size:11px;margin-left:8px">${e}</div>`;
      }
    }

    showDiffPanel(html);

    if (result.scene && callbacks) {
      callbacks.onSceneReload(result.scene);
    }
    if (result.diff && callbacks) {
      callbacks.onDiffReceived(result.diff);
    }
  } catch (err) {
    showDiffPanel(`Network error: ${err}`);
  }
}

async function resetModel() {
  hideContextMenu();
  try {
    await fetch(`${apiBase}/api/reset`, { method: 'POST' });
    const resp = await fetch(`${apiBase}/api/scene`);
    const scene = await resp.json();
    if (callbacks) callbacks.onSceneReload(scene);
    hideDiffPanel();
  } catch (err) {
    showDiffPanel(`Reset failed: ${err}`);
  }
}

function createContextMenu() {
  contextMenu = document.createElement('div');
  contextMenu.id = 'context-menu';
  contextMenu.style.cssText = `
    position: fixed;
    display: none;
    background: rgba(0,0,0,0.95);
    border: 1px solid #ff0066;
    border-radius: 4px;
    padding: 4px 0;
    z-index: 100;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    min-width: 200px;
    box-shadow: 0 0 16px rgba(255,0,102,0.3);
  `;
  document.getElementById('app')!.appendChild(contextMenu);

  const style = document.createElement('style');
  style.textContent = `
    .ctx-item {
      padding: 6px 12px;
      color: #ddd;
      cursor: pointer;
      transition: background 0.1s;
    }
    .ctx-item:hover {
      background: rgba(255,0,102,0.15);
      color: #fff;
    }
    .ctx-divider {
      height: 1px;
      background: #333;
      margin: 2px 0;
    }
  `;
  document.head.appendChild(style);

  contextMenu.addEventListener('click', (e) => {
    const item = (e.target as HTMLElement).closest('.ctx-item') as HTMLElement;
    if (!item) return;
    const action = item.dataset.action;
    if (action === 'remove') {
      removeNode(item.dataset.node!);
    } else if (action === 'reset') {
      resetModel();
    }
  });

  document.addEventListener('click', () => hideContextMenu());
  document.addEventListener('contextmenu', (e) => {
    if (!(e.target as HTMLElement).closest('#context-menu')) {
      hideContextMenu();
    }
  });
}

function createDiffPanel() {
  diffPanel = document.createElement('div');
  diffPanel.id = 'diff-panel';
  diffPanel.style.cssText = `
    position: absolute;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    max-width: 500px;
    background: rgba(0,0,0,0.9);
    border: 1px solid #0ff;
    border-radius: 4px;
    color: #ddd;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 12px;
    display: none;
    z-index: 10;
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
  `;

  const closeBtn = document.createElement('span');
  closeBtn.textContent = '×';
  closeBtn.style.cssText = 'position:absolute;top:4px;right:8px;cursor:pointer;color:#f0f;font-size:16px;';
  closeBtn.addEventListener('click', hideDiffPanel);
  diffPanel.appendChild(closeBtn);

  document.getElementById('app')!.appendChild(diffPanel);
}

function showDiffPanel(html: string) {
  if (!diffPanel) return;
  const close = diffPanel.querySelector('span');
  diffPanel.innerHTML = html;
  if (close) diffPanel.prepend(close);
  diffPanel.style.display = 'block';
}

function hideDiffPanel() {
  if (diffPanel) diffPanel.style.display = 'none';
}
