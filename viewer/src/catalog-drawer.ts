import type { SceneGraph, SceneMotif, SceneChain, PaperRef } from './types';

let drawerEl: HTMLElement | null = null;
let onNodeClick: ((nodeId: string) => void) | null = null;

const SIG_COLORS: Record<string, string> = {
  EXCEPTIONAL: '#ff0033',
  PRIMARY: '#ff6600',
  SECONDARY: '#00ffff',
  TERTIARY: '#00ff41',
  MITIGATING: '#66ffcc',
};

const CONF_COLORS: Record<string, string> = {
  HIGH: '#00ff41',
  MEDIUM: '#ffcc00',
  LOW: '#ff4444',
};

export function initCatalogDrawer(nodeClickHandler?: (nodeId: string) => void) {
  onNodeClick = nodeClickHandler || null;
  drawerEl = document.createElement('div');
  drawerEl.id = 'catalog-drawer';
  drawerEl.innerHTML = '';
  drawerEl.style.cssText = `
    position: absolute;
    bottom: 16px;
    right: 16px;
    width: 400px;
    max-height: 60vh;
    overflow-x: hidden;
    overflow-y: auto;
    word-wrap: break-word;
    overflow-wrap: break-word;
    background: rgba(0, 0, 0, 0.93);
    border: 1px solid #ff6600;
    border-radius: 4px;
    color: #ddd;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 14px;
    display: none;
    z-index: 10;
    box-shadow: 0 0 24px rgba(255, 0, 255, 0.25);
  `;
  document.getElementById('app')!.appendChild(drawerEl);
}

export function showCatalogEntry(id: string, scene: SceneGraph) {
  if (!drawerEl) return;

  const motif = scene.motifs.find(m => m.id === id);
  const chain = scene.chains.find(c => c.id === id);

  if (motif) {
    showMotifDetail(motif);
    if (!motif.description && !motif.attacks_enabled?.length) {
      fetchAndEnrich(motif.catalog_ref || motif.id);
    }
  } else if (chain) {
    showChainDetail(chain, scene);
  } else {
    drawerEl.style.display = 'none';
    return;
  }

  drawerEl.style.display = 'block';
}

async function fetchAndEnrich(catalogId: string) {
  if (!drawerEl) return;
  try {
    const res = await fetch(`/api/catalog/${catalogId}`);
    if (!res.ok) return;
    const data = await res.json();
    if (data && drawerEl.style.display === 'block') {
      const extra = document.createElement('div');
      extra.style.cssText = 'margin-top:10px;border-top:1px solid #333;padding-top:8px';
      if (data.description) {
        extra.innerHTML += `<div style="color:#ccc;margin-bottom:8px;line-height:1.4">${data.description}</div>`;
      }
      if (data.attacks_enabled?.length) {
        const chips = data.attacks_enabled.map((a: string) =>
          `<span style="color:#ff0066;border:1px solid #660033;padding:1px 5px;border-radius:3px;font-size:10px;display:inline-block;margin:1px">${a.replace(/_/g, ' ')}</span>`
        ).join('');
        extra.innerHTML += `<div style="margin-bottom:8px"><div style="color:#ff0066;font-size:11px;margin-bottom:4px">ATTACKS ENABLED</div><div style="display:flex;flex-wrap:wrap;gap:2px">${chips}</div></div>`;
      }
      if (data.detection_logic) {
        extra.innerHTML += `<details style="margin-bottom:8px"><summary style="color:#0ff;cursor:pointer;font-size:11px">DETECTION LOGIC</summary><div style="color:#ccc;margin-top:4px;font-size:11px;line-height:1.4;padding-left:8px;border-left:2px solid #066">${data.detection_logic}</div></details>`;
      }
      if (extra.innerHTML) drawerEl.appendChild(extra);
    }
  } catch { /* API unavailable — static scene.json mode */ }
}

function showMotifDetail(m: SceneMotif) {
  if (!drawerEl) return;

  const sigColor = SIG_COLORS[m.structural_significance || ''] || '#888';
  const confColor = CONF_COLORS[m.confidence || ''] || '#888';

  const badges = [];
  if (m.structural_significance) {
    badges.push(`<span style="color:${sigColor};text-shadow:0 0 6px ${sigColor};border:1px solid ${sigColor};padding:1px 6px;border-radius:3px;font-size:10px">${m.structural_significance}</span>`);
  }
  if (m.confidence) {
    badges.push(`<span style="color:${confColor};border:1px solid ${confColor};padding:1px 6px;border-radius:3px;font-size:10px">${m.confidence}</span>`);
  }
  if (m.category) {
    badges.push(`<span style="color:#0aa;border:1px solid #066;padding:1px 6px;border-radius:3px;font-size:10px">${m.category}</span>`);
  }

  let html = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <h3 style="color:#ff6600;margin:0;text-shadow:0 0 8px #f60;font-size:14px">MOTIF</h3>
      <span id="close-catalog" style="color:#f60;cursor:pointer;font-size:18px">&times;</span>
    </div>
    <div style="color:#fff;font-size:13px;margin-bottom:6px;text-shadow:0 0 4px rgba(255,255,255,0.3)">${m.title}</div>
    ${badges.length > 0 ? `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">${badges.join('')}</div>` : ''}
  `;

  if (m.description) {
    html += `<div style="color:#ccc;margin-bottom:10px;line-height:1.4">${m.description}</div>`;
  }

  if (m.attacks_enabled && m.attacks_enabled.length > 0) {
    const chips = m.attacks_enabled.map(a =>
      `<span style="color:#ff0066;border:1px solid #660033;padding:1px 5px;border-radius:3px;font-size:10px;display:inline-block;margin:1px">${a.replace(/_/g, ' ')}</span>`
    ).join('');
    html += `
      <div style="margin-bottom:10px">
        <div style="color:#ff0066;font-size:11px;margin-bottom:4px;text-shadow:0 0 4px #f06">ATTACKS ENABLED</div>
        <div style="display:flex;flex-wrap:wrap;gap:2px">${chips}</div>
      </div>
    `;
  }

  if (m.detection_logic) {
    html += `
      <details style="margin-bottom:10px">
        <summary style="color:#0ff;cursor:pointer;font-size:11px;text-shadow:0 0 4px #0ff">DETECTION LOGIC</summary>
        <div style="color:#ccc;margin-top:4px;font-size:11px;line-height:1.4;padding-left:8px;border-left:2px solid #066">${m.detection_logic}</div>
      </details>
    `;
  }

  if (m.research_basis && m.research_basis.length > 0) {
    html += renderPapers(m.research_basis);
  }

  if (m.node_ids.length > 0) {
    html += renderNodeLinks(m.node_ids);
  }

  drawerEl.innerHTML = html;
  attachCloseHandler();
  attachNodeClickHandlers();
}

function showChainDetail(c: SceneChain, scene: SceneGraph) {
  if (!drawerEl) return;

  const sigColor = SIG_COLORS[c.structural_significance || ''] || '#888';
  const title = c.title || c.id;

  const badges = [];
  if (c.structural_significance) {
    badges.push(`<span style="color:${sigColor};text-shadow:0 0 6px ${sigColor};border:1px solid ${sigColor};padding:1px 6px;border-radius:3px;font-size:10px">${c.structural_significance}</span>`);
  }

  const gadgetNames = c.gadget_ids.map(gid => {
    const motif = scene.motifs.find(m => m.id === gid);
    return motif ? motif.title : gid;
  });

  let html = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <h3 style="color:#ff6600;margin:0;text-shadow:0 0 8px #f60;font-size:14px">CHAIN</h3>
      <span id="close-catalog" style="color:#f60;cursor:pointer;font-size:18px">&times;</span>
    </div>
    <div style="color:#fff;font-size:13px;margin-bottom:6px;text-shadow:0 0 4px rgba(255,255,255,0.3)">${title}</div>
    ${badges.length > 0 ? `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">${badges.join('')}</div>` : ''}
  `;

  if (c.description) {
    html += `<div style="color:#ccc;margin-bottom:10px;line-height:1.4">${c.description}</div>`;
  }

  if (gadgetNames.length > 0) {
    const chips = gadgetNames.map(g =>
      `<span style="color:#ff6600;border:1px solid #663300;padding:1px 5px;border-radius:3px;font-size:10px;display:inline-block;margin:1px">${g}</span>`
    ).join('');
    html += `
      <div style="margin-bottom:10px">
        <div style="color:#ff6600;font-size:11px;margin-bottom:4px;text-shadow:0 0 4px #f60">COMPONENT GADGETS</div>
        <div style="display:flex;flex-wrap:wrap;gap:2px">${chips}</div>
      </div>
    `;
  }

  if (c.research_basis && c.research_basis.length > 0) {
    html += renderPapers(c.research_basis);
  }

  if (c.node_ids.length > 0) {
    html += renderNodeLinks(c.node_ids);
  }

  drawerEl.innerHTML = html;
  attachCloseHandler();
  attachNodeClickHandlers();
}

function renderPapers(papers: PaperRef[]): string {
  const items = papers.map(p =>
    `<div style="color:#0ff;font-size:11px;margin-bottom:2px;padding-left:8px;border-left:2px solid #066">${p.title} <span style="color:#066">(${p.slug})</span></div>`
  ).join('');
  return `
    <div style="margin-bottom:10px">
      <div style="color:#0ff;font-size:11px;margin-bottom:4px;text-shadow:0 0 4px #0ff">RESEARCH BASIS</div>
      ${items}
    </div>
  `;
}

function renderNodeLinks(nodeIds: string[]): string {
  const display = nodeIds.length > 8 ? nodeIds.slice(0, 8) : nodeIds;
  const more = nodeIds.length > 8 ? `<span style="color:#066"> +${nodeIds.length - 8} more</span>` : '';
  const links = display.map(nid =>
    `<span class="node-link" data-node-id="${nid}" style="color:#0ff;cursor:pointer;border-bottom:1px dotted #066;font-size:11px;margin-right:6px">${nid}</span>`
  ).join('');
  return `
    <div style="margin-bottom:6px">
      <div style="color:#0aa;font-size:11px;margin-bottom:4px">NODES</div>
      <div>${links}${more}</div>
    </div>
  `;
}

function attachCloseHandler() {
  document.getElementById('close-catalog')?.addEventListener('click', closeCatalog);
}

function attachNodeClickHandlers() {
  if (!drawerEl || !onNodeClick) return;
  const links = drawerEl.querySelectorAll('.node-link');
  links.forEach(el => {
    el.addEventListener('click', () => {
      const nodeId = (el as HTMLElement).dataset.nodeId;
      if (nodeId && onNodeClick) onNodeClick(nodeId);
    });
  });
}

export function closeCatalog() {
  if (drawerEl) drawerEl.style.display = 'none';
}
