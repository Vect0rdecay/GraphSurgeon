import type { SceneGraph, SceneMotif, SceneChain } from './types';

let drawerEl: HTMLElement | null = null;

export function initCatalogDrawer() {
  drawerEl = document.createElement('div');
  drawerEl.id = 'catalog-drawer';
  drawerEl.innerHTML = '';
  drawerEl.style.cssText = `
    position: absolute;
    bottom: 16px;
    right: 16px;
    width: 340px;
    max-height: 50vh;
    overflow-y: auto;
    background: rgba(0, 0, 0, 0.9);
    border: 1px solid #ff00ff;
    border-radius: 4px;
    color: #ddd;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 12px;
    display: none;
    z-index: 10;
    box-shadow: 0 0 20px rgba(255, 0, 255, 0.2);
  `;
  document.getElementById('app')!.appendChild(drawerEl);
}

export function showCatalogEntry(id: string, scene: SceneGraph) {
  if (!drawerEl) return;

  const motif = scene.motifs.find(m => m.id === id);
  const chain = scene.chains.find(c => c.id === id);

  if (motif) {
    showMotifDetail(motif);
  } else if (chain) {
    showChainDetail(chain, scene);
  } else {
    drawerEl.style.display = 'none';
    return;
  }

  drawerEl.style.display = 'block';
}

function showMotifDetail(m: SceneMotif) {
  if (!drawerEl) return;
  drawerEl.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h3 style="color:#ff00ff;margin:0;text-shadow:0 0 6px #f0f">MOTIF</h3>
      <span id="close-catalog" style="color:#f0f;cursor:pointer;font-size:16px">&times;</span>
    </div>
    <h4 style="color:#fff;margin:4px 0 8px">${m.title}</h4>
    <div style="color:#aaa;margin-bottom:8px">${m.description || 'No description available.'}</div>
    <div style="color:#666;font-size:11px">
      <div>catalog_ref: ${m.catalog_ref}</div>
      <div>nodes: ${m.node_ids.join(', ') || 'none'}</div>
    </div>
  `;
  document.getElementById('close-catalog')!.addEventListener('click', closeCatalog);
}

function showChainDetail(c: SceneChain, scene: SceneGraph) {
  if (!drawerEl) return;
  const gadgetNames = c.gadget_ids.join(', ');
  drawerEl.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h3 style="color:#ff6600;margin:0;text-shadow:0 0 6px #f60">CHAIN</h3>
      <span id="close-catalog" style="color:#f60;cursor:pointer;font-size:16px">&times;</span>
    </div>
    <h4 style="color:#fff;margin:4px 0 8px">${c.id}</h4>
    <div style="color:#aaa;margin-bottom:8px">
      Compound chain involving ${c.node_ids.length} node(s) and ${c.gadget_ids.length} gadget(s).
    </div>
    <div style="color:#666;font-size:11px">
      <div>gadgets: ${gadgetNames || 'none'}</div>
      <div>nodes: ${c.node_ids.join(', ') || 'none'}</div>
    </div>
  `;
  document.getElementById('close-catalog')!.addEventListener('click', closeCatalog);
}

export function closeCatalog() {
  if (drawerEl) drawerEl.style.display = 'none';
}
