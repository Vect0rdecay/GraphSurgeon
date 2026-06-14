import * as THREE from 'three';
import type { SceneShadowLogic, SceneShadowLogicPoint } from './types';
import type { NodePosition } from './layout';

const MARKER_COLOR = 0xff6600;

export interface ShadowLogicMarkers {
  group: THREE.Group;
  hitMeshes: THREE.Mesh[];
  animate: (time: number) => void;
}

export function buildShadowLogicMarkers(
  data: SceneShadowLogic,
  positions: Map<string, NodePosition>,
): ShadowLogicMarkers {
  const group = new THREE.Group();
  group.name = 'shadowlogic-markers';
  group.visible = false;
  const hitMeshes: THREE.Mesh[] = [];
  const animated: THREE.Object3D[] = [];

  for (const point of data.injection_points) {
    const pos = positions.get(point.node_id);
    if (!pos) continue;

    const geo = new THREE.TetrahedronGeometry(1.1, 0);
    const edges = new THREE.EdgesGeometry(geo);
    const marker = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({
        color: MARKER_COLOR,
        transparent: true,
        opacity: 0.7,
      }),
    );
    marker.position.set(pos.x, pos.y, pos.z);
    marker.userData.color = MARKER_COLOR;
    marker.userData.pointData = point;
    marker.renderOrder = -1;
    group.add(marker);
    animated.push(marker);

    const hitMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1.5, 8, 6),
      new THREE.MeshBasicMaterial({ visible: false }),
    );
    hitMesh.position.set(pos.x, pos.y, pos.z);
    hitMesh.userData.shadowlogicPoint = point;
    hitMesh.userData.isShadowLogicHit = true;
    group.add(hitMesh);
    hitMeshes.push(hitMesh);
  }

  const animate = (time: number) => {
    if (!group.visible) return;
    for (const m of animated) {
      m.rotation.y = time * 0.4;
      m.rotation.z = Math.sin(time * 0.3) * 0.2;
      const pulse = 0.5 + Math.sin(time * 1.8) * 0.25;
      (m as THREE.LineSegments).material = new THREE.LineBasicMaterial({
        color: MARKER_COLOR,
        transparent: true,
        opacity: pulse,
      });
    }
  };

  return { group, hitMeshes, animate };
}

export function buildShadowLogicPanel(
  data: SceneShadowLogic,
  onFlyTo: (nodeId: string) => void,
): HTMLElement {
  const container = document.createElement('div');
  container.id = 'shadowlogic-panel';

  const header = document.createElement('div');
  header.style.cssText = `
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; padding: 6px 0; user-select: none;
  `;
  header.innerHTML = `<span style="color:#ff6600;font-weight:bold;font-size:11px">▸ SHADOWLOGIC (${data.injection_points.length})</span>`;

  const body = document.createElement('div');
  body.style.display = 'none';

  let expanded = false;
  header.addEventListener('click', () => {
    expanded = !expanded;
    body.style.display = expanded ? 'block' : 'none';
    header.innerHTML = `<span style="color:#ff6600;font-weight:bold;font-size:11px">${expanded ? '▾' : '▸'} SHADOWLOGIC (${data.injection_points.length})</span>`;
  });

  const scoreRow = document.createElement('div');
  scoreRow.style.cssText = 'color:#ccc;font-size:10px;padding:4px 0;border-bottom:1px solid rgba(255,102,0,0.2);margin-bottom:6px;';
  scoreRow.innerHTML = `
    <span style="color:#ff6600">Exposure:</span> ${data.structural_exposure.toFixed(2)}
    <span style="color:#ff6600;margin-left:8px">Tier:</span> ${data.exposure_tier}
    ${data.conditional_ops.length ? `<div style="margin-top:2px"><span style="color:#ff6600">Conditional ops:</span> ${data.conditional_ops.join(', ')}</div>` : ''}
  `;
  body.appendChild(scoreRow);

  for (const point of data.injection_points) {
    const row = document.createElement('div');
    row.style.cssText = `
      padding: 5px 6px; margin-bottom: 4px;
      border-left: 2px solid #ff6600;
      background: rgba(255,102,0,0.04);
      cursor: pointer; border-radius: 0 2px 2px 0;
      transition: background 0.15s;
    `;
    row.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px">
        <span style="color:#ff6600;font-weight:bold;font-size:11px">${point.location}</span>
        <span style="color:#0aa;font-size:9px;border:1px solid #0aa;padding:0 3px;border-radius:2px">${point.injection_complexity}</span>
      </div>
      <div style="color:#ccc;font-size:10px;margin-top:2px;word-wrap:break-word;overflow-wrap:break-word">${point.description}</div>
      <div style="color:#888;font-size:9px;margin-top:1px">${point.node_id}</div>
    `;
    row.addEventListener('mouseover', () => { row.style.background = 'rgba(255,102,0,0.12)'; });
    row.addEventListener('mouseout', () => { row.style.background = 'rgba(255,102,0,0.04)'; });
    row.addEventListener('click', () => onFlyTo(point.node_id));
    body.appendChild(row);
  }

  container.appendChild(header);
  container.appendChild(body);
  return container;
}

export function showShadowLogicDetail(point: SceneShadowLogicPoint): HTMLElement {
  const panel = document.createElement('div');
  panel.style.cssText = `
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.95); border: 1px solid #ff6600;
    border-radius: 4px; padding: 16px; color: #fff;
    font-family: 'Courier New', monospace; font-size: 12px;
    z-index: 30; max-width: 400px; box-shadow: 0 0 30px rgba(255,102,0,0.3);
    word-wrap: break-word; overflow-wrap: break-word;
  `;
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <span style="color:#ff6600;font-weight:bold;font-size:14px">SHADOWLOGIC POINT</span>
      <span id="sl-detail-close" style="color:#ff6600;cursor:pointer;font-size:16px;padding:0 4px">×</span>
    </div>
    <div style="margin-bottom:6px"><span style="color:#0ff">Node:</span> ${point.node_id}</div>
    <div style="margin-bottom:6px"><span style="color:#0ff">Location:</span> ${point.location}</div>
    <div style="margin-bottom:6px"><span style="color:#0ff">Complexity:</span> ${point.injection_complexity}</div>
    <div style="margin-bottom:6px"><span style="color:#0ff">Detection:</span> ${point.detection_difficulty}</div>
    <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,102,0,0.2);color:#ccc;font-size:11px">${point.description}</div>
  `;
  return panel;
}
