import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { SceneGraph, SceneNode } from './types';
import { buildThreeScene, type BuiltScene } from './scene-builder';
import { getAllCategories } from './colors';
import { highlightMotif, clearHighlight, buildMotifList } from './motifs';
import { initFlow, startFlow, stopFlow, isFlowPlaying, updateFlow } from './flow';
import { initCatalogDrawer, showCatalogEntry, closeCatalog } from './catalog-drawer';
import { initEditMode, showContextMenu, hideContextMenu } from './edit-mode';
import { initSearch, updateSearchResults, pulseNode } from './search';
import { initDepthScrubber, updateScrubber, flyToBand } from './depth-scrubber';
import { updateLabelLOD } from './lod';

let renderer: THREE.WebGLRenderer;
let threeScene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let composer: EffectComposer;
let builtScene: BuiltScene | null = null;
let sceneData: SceneGraph | null = null;
let raycaster: THREE.Raycaster;
let mouse: THREE.Vector2;
let hoveredMesh: THREE.Mesh | null = null;
let selectedMesh: THREE.Mesh | null = null;
let clock: THREE.Clock;

init();
loadDefaultScene();

function init() {
  const container = document.getElementById('app')!;

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 1.5;
  container.appendChild(renderer.domElement);

  threeScene = new THREE.Scene();
  threeScene.background = new THREE.Color(0x000000);
  threeScene.fog = new THREE.FogExp2(0x000000, 0.008);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
  camera.position.set(0, 5, 30);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed = 0.5;
  controls.panSpeed = 0.7;
  controls.minDistance = 5;
  controls.maxDistance = 200;

  const ambient = new THREE.AmbientLight(0x222244, 0.5);
  threeScene.add(ambient);

  const point1 = new THREE.PointLight(0x00ffff, 1.5, 100);
  point1.position.set(10, 10, 10);
  threeScene.add(point1);

  const point2 = new THREE.PointLight(0xff00ff, 1.0, 100);
  point2.position.set(-10, -10, -10);
  threeScene.add(point2);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(threeScene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.8,
    0.4,
    0.85,
  );
  composer.addPass(bloomPass);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();
  clock = new THREE.Clock();

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onClick);
  document.getElementById('close-detail')!.addEventListener('click', closeDetail);

  initCatalogDrawer();
  setupDragDrop();
  setupEditMode();
  initSearch((nodeId) => {
    if (builtScene) pulseNode(nodeId, builtScene, camera, controls);
  });
  initDepthScrubber((position) => {
    if (builtScene && sceneData) flyToBand(position, sceneData, builtScene, camera, controls);
  });
  buildLegend();
  renderer.domElement.addEventListener('contextmenu', onRightClick);
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  controls.update();

  if (builtScene && sceneData && isFlowPlaying()) {
    updateFlow(delta, builtScene, camera, controls);
  }

  if (builtScene) {
    updateLabelLOD(builtScene, camera);
  }

  composer.render();
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

function onMouseMove(event: MouseEvent) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  if (!builtScene || isFlowPlaying()) return;

  raycaster.setFromCamera(mouse, camera);
  const meshes = [...builtScene.nodeObjects.values()];
  const intersects = raycaster.intersectObjects(meshes);

  if (hoveredMesh && hoveredMesh !== selectedMesh) {
    const mat = hoveredMesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = 0.6;
  }

  if (intersects.length > 0) {
    const mesh = intersects[0].object as THREE.Mesh;
    hoveredMesh = mesh;
    if (mesh !== selectedMesh) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 1.2;
    }
    renderer.domElement.style.cursor = 'pointer';
  } else {
    hoveredMesh = null;
    renderer.domElement.style.cursor = 'default';
  }
}

function onClick(_event: MouseEvent) {
  if (!builtScene || isFlowPlaying()) return;

  raycaster.setFromCamera(mouse, camera);
  const meshes = [...builtScene.nodeObjects.values()];
  const intersects = raycaster.intersectObjects(meshes);

  if (selectedMesh) {
    const mat = selectedMesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = 0.6;
    selectedMesh = null;
  }

  if (intersects.length > 0) {
    const mesh = intersects[0].object as THREE.Mesh;
    selectedMesh = mesh;
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = 1.5;

    const nodeData = mesh.userData.nodeData as SceneNode;
    showDetail(nodeData);
  } else {
    closeDetail();
  }
}

function showDetail(node: SceneNode) {
  const panel = document.getElementById('detail-panel')!;
  const content = document.getElementById('detail-content')!;

  const attrs = Object.entries(node.attributes)
    .map(([k, v]) => `<div class="field"><span class="label">${k}:</span> <span class="value">${JSON.stringify(v)}</span></div>`)
    .join('');

  const motifs = node.motif_ids.length > 0
    ? `<div class="field"><span class="label">motifs:</span> <span class="value">${node.motif_ids.join(', ')}</span></div>`
    : '';

  const gadgets = node.gadget_ids.length > 0
    ? `<div class="field"><span class="label">gadgets:</span> <span class="value">${node.gadget_ids.join(', ')}</span></div>`
    : '';

  content.innerHTML = `
    <h2>${node.op_type}: ${node.id}</h2>
    <div class="field"><span class="label">category:</span> <span class="value">${node.category}</span></div>
    <div class="field"><span class="label">depth:</span> <span class="value">${node.depth}</span></div>
    <div class="field"><span class="label">position:</span> <span class="value">${node.position}</span></div>
    <div class="field"><span class="label">exec_index:</span> <span class="value">${node.exec_index}</span></div>
    <div class="field"><span class="label">inputs:</span> <span class="value">${node.inputs.join(', ')}</span></div>
    <div class="field"><span class="label">outputs:</span> <span class="value">${node.outputs.join(', ')}</span></div>
    ${node.param_count > 0 ? `<div class="field"><span class="label">params:</span> <span class="value">${node.param_count.toLocaleString()}</span></div>` : ''}
    ${motifs}
    ${gadgets}
    ${attrs ? `<h2 style="margin-top:8px">Attributes</h2>${attrs}` : ''}
  `;

  panel.style.display = 'block';
}

function closeDetail() {
  document.getElementById('detail-panel')!.style.display = 'none';
  if (selectedMesh) {
    const mat = selectedMesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = 0.6;
    selectedMesh = null;
  }
}

function buildLegend() {
  const legend = document.getElementById('legend')!;
  const cats = getAllCategories();
  legend.innerHTML = cats
    .map(([name, color]) => {
      const hex = '#' + color.toString(16).padStart(6, '0');
      return `<div class="entry"><span class="swatch" style="background:${hex};box-shadow:0 0 4px ${hex}"></span>${name}</div>`;
    })
    .join('');
}

function setupDragDrop() {
  const overlay = document.getElementById('drop-overlay')!;

  document.addEventListener('dragover', (e) => {
    e.preventDefault();
    overlay.style.display = 'flex';
  });
  document.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) overlay.style.display = 'none';
  });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    overlay.style.display = 'none';
    const file = e.dataTransfer?.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result as string) as SceneGraph;
          loadScene(data);
        } catch (err) {
          console.error('Failed to parse scene.json:', err);
        }
      };
      reader.readAsText(file);
    }
  });
}

async function loadDefaultScene() {
  try {
    const resp = await fetch('/sample_scene.json');
    if (resp.ok) {
      const data = await resp.json() as SceneGraph;
      loadScene(data);
      return;
    }
  } catch {
    // static file not available
  }

  try {
    const resp = await fetch('/api/scene');
    if (resp.ok) {
      const data = await resp.json() as SceneGraph;
      loadScene(data);
      return;
    }
  } catch {
    // no API server either — user must drag-drop
  }
}

function loadScene(data: SceneGraph) {
  if (builtScene) {
    threeScene.remove(builtScene.group);
  }

  sceneData = data;
  builtScene = buildThreeScene(data);
  threeScene.add(builtScene.group);

  const hud = document.getElementById('hud-info')!;
  hud.innerHTML = `
    ${data.model.name} | ${data.model.total_nodes} nodes | depth ${data.model.max_depth} | opset ${data.model.opset}
  `;

  initFlow(data, (index, nodeId) => {
    const node = data.nodes.find(n => n.id === nodeId);
    if (node) {
      const flowStatus = document.getElementById('flow-status');
      if (flowStatus) {
        flowStatus.textContent = `[${index + 1}/${data.nodes.length}] ${node.op_type}: ${node.id}`;
      }
    }
  });

  updateSearchResults(data);
  updateScrubber(data);
  buildControlPanel(data);
  frameAll();
}

function buildControlPanel(data: SceneGraph) {
  let panel = document.getElementById('control-panel');
  if (panel) panel.remove();

  panel = document.createElement('div');
  panel.id = 'control-panel';
  panel.style.cssText = `
    position: absolute;
    top: 80px;
    left: 16px;
    width: 220px;
    max-height: calc(100vh - 100px);
    overflow-y: auto;
    color: #0ff;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    z-index: 10;
  `;

  // Flow playback button
  const flowBtn = document.createElement('button');
  flowBtn.id = 'flow-btn';
  flowBtn.textContent = 'PLAY FLOW';
  flowBtn.style.cssText = `
    background: transparent;
    border: 1px solid #0ff;
    color: #0ff;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 6px 12px;
    cursor: pointer;
    width: 100%;
    margin-bottom: 4px;
    text-shadow: 0 0 6px #0ff;
    box-shadow: 0 0 8px rgba(0,255,255,0.2);
  `;
  flowBtn.addEventListener('click', () => {
    if (isFlowPlaying()) {
      stopFlow();
      flowBtn.textContent = 'PLAY FLOW';
      flowBtn.style.borderColor = '#0ff';
      flowBtn.style.color = '#0ff';
      if (builtScene) clearHighlight(builtScene);
    } else {
      startFlow();
      flowBtn.textContent = 'STOP FLOW';
      flowBtn.style.borderColor = '#ff0066';
      flowBtn.style.color = '#ff0066';
      closeCatalog();
    }
  });
  panel.appendChild(flowBtn);

  const flowStatus = document.createElement('div');
  flowStatus.id = 'flow-status';
  flowStatus.style.cssText = 'color:#666;margin-bottom:12px;min-height:16px;';
  panel.appendChild(flowStatus);

  // Flow description
  if (data.model.flow_description) {
    const flowDesc = document.createElement('div');
    flowDesc.style.cssText = `
      color: #555;
      font-size: 11px;
      margin-bottom: 12px;
      max-height: 80px;
      overflow-y: auto;
      border-left: 2px solid #333;
      padding-left: 8px;
    `;
    flowDesc.textContent = data.model.flow_description;
    panel.appendChild(flowDesc);
  }

  // Motifs section
  if (data.motifs.length > 0 || data.chains.length > 0) {
    const motifHeader = document.createElement('div');
    motifHeader.style.cssText = 'color:#ff00ff;margin-bottom:4px;text-shadow:0 0 6px #f0f;font-weight:bold;';
    motifHeader.textContent = 'MOTIFS & CHAINS';
    panel.appendChild(motifHeader);

    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'CLEAR HIGHLIGHT';
    clearBtn.style.cssText = `
      background: transparent;
      border: 1px solid #333;
      color: #666;
      font-family: 'Courier New', monospace;
      font-size: 11px;
      padding: 3px 8px;
      cursor: pointer;
      width: 100%;
      margin-bottom: 6px;
    `;
    clearBtn.addEventListener('click', () => {
      if (builtScene) clearHighlight(builtScene);
      closeCatalog();
    });
    panel.appendChild(clearBtn);

    const motifList = buildMotifList(data);
    motifList.style.cssText = `
      max-height: 300px;
      overflow-y: auto;
    `;
    panel.appendChild(motifList);

    const style = document.createElement('style');
    style.textContent = `
      .motif-entry {
        padding: 4px 6px;
        cursor: pointer;
        border-left: 2px solid transparent;
        margin-bottom: 2px;
        transition: all 0.15s;
      }
      .motif-entry:hover {
        border-left-color: #ff00ff;
        background: rgba(255, 0, 255, 0.05);
      }
      .motif-title {
        color: #aaa;
        margin-left: 4px;
      }
      .motif-entry:hover .motif-title {
        color: #fff;
      }
    `;
    document.head.appendChild(style);

    motifList.addEventListener('click', (e) => {
      const entry = (e.target as HTMLElement).closest('.motif-entry') as HTMLElement;
      if (!entry || !builtScene || !sceneData) return;
      const motifId = entry.dataset.motifId!;
      highlightMotif(motifId, sceneData, builtScene);
      showCatalogEntry(motifId, sceneData);
    });
  }

  document.getElementById('app')!.appendChild(panel);
}

function onRightClick(event: MouseEvent) {
  if (!builtScene || isFlowPlaying()) return;

  event.preventDefault();
  raycaster.setFromCamera(mouse, camera);
  const meshes = [...builtScene.nodeObjects.values()];
  const intersects = raycaster.intersectObjects(meshes);

  if (intersects.length > 0) {
    const nodeData = intersects[0].object.userData.nodeData as SceneNode;
    showContextMenu(nodeData, event);
  } else {
    hideContextMenu();
  }
}

function setupEditMode() {
  const apiBase = window.location.origin;
  initEditMode(apiBase, {
    onSceneReload: (newScene: SceneGraph) => {
      loadScene(newScene);
    },
    onDiffReceived: (_diff) => {},
  });
}

function frameAll() {
  if (!builtScene) return;

  const box = new THREE.Box3().setFromObject(builtScene.group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  const dist = Math.max(maxDim / (2 * Math.tan(fov / 2)) * 1.5, 15);

  camera.position.set(center.x, center.y + dist * 0.3, center.z + dist);
  controls.target.copy(center);
  controls.update();
}
