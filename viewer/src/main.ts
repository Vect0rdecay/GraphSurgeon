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
const keysDown = new Set<string>();
const FLY_SPEED = 30;

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
  threeScene.fog = new THREE.FogExp2(0x000000, 0.002);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
  camera.position.set(0, 5, 30);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed = 0.5;
  controls.panSpeed = 0.7;
  controls.minDistance = 5;
  controls.maxDistance = 2000;

  const ambient = new THREE.AmbientLight(0x334466, 1.0);
  threeScene.add(ambient);

  const dir1 = new THREE.DirectionalLight(0x00ffff, 1.5);
  dir1.position.set(1, 1, 1);
  threeScene.add(dir1);

  const dir2 = new THREE.DirectionalLight(0xff00ff, 1.0);
  dir2.position.set(-1, -1, -1);
  threeScene.add(dir2);

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

  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    keysDown.add(e.key.toLowerCase());
  });
  window.addEventListener('keyup', (e) => {
    keysDown.delete(e.key.toLowerCase());
  });

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  if (keysDown.size > 0) {
    const move = new THREE.Vector3();
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
    const speed = FLY_SPEED * delta;

    if (keysDown.has('w')) move.add(forward.clone().multiplyScalar(speed));
    if (keysDown.has('s')) move.add(forward.clone().multiplyScalar(-speed));
    if (keysDown.has('a')) move.add(right.clone().multiplyScalar(-speed));
    if (keysDown.has('d')) move.add(right.clone().multiplyScalar(speed));
    if (keysDown.has('q') || keysDown.has(' ')) move.y += speed;
    if (keysDown.has('e') || keysDown.has('shift')) move.y -= speed;

    camera.position.add(move);
    controls.target.add(move);
  }

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

  const graphSpan = (data.model.max_depth + 1) * 3.0;
  const fogDensity = Math.min(0.002, 2.0 / Math.max(graphSpan, 100));
  (threeScene.fog as THREE.FogExp2).density = fogDensity;

  const hud = document.getElementById('hud-info')!;
  hud.innerHTML = `
    ${data.model.name} | ${data.model.total_nodes} nodes | depth ${data.model.max_depth} | opset ${data.model.opset}
    <div style="font-size:10px;color:#0aa;margin-top:2px">WASD fly | Q/E up/down | mouse orbit | scroll zoom | click node</div>
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
    max-height: calc(100vh - 280px);
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
  flowStatus.style.cssText = 'color:#0ff;margin-bottom:12px;min-height:16px;';
  panel.appendChild(flowStatus);

  // Flow description — show a clickable button that opens a full overlay
  if (data.model.flow_description) {
    const flowDescBtn = document.createElement('button');
    flowDescBtn.textContent = 'VIEW MODEL FLOW';
    flowDescBtn.style.cssText = `
      background: transparent;
      border: 1px solid #0ff;
      color: #0ff;
      font-family: 'Courier New', monospace;
      font-size: 11px;
      padding: 4px 8px;
      cursor: pointer;
      width: 100%;
      margin-bottom: 12px;
      text-shadow: 0 0 4px #0ff;
    `;
    flowDescBtn.addEventListener('click', () => {
      showFlowOverlay(data.model.flow_description!);
    });
    panel.appendChild(flowDescBtn);
  }

  // Motifs section — collapsible header
  if (data.motifs.length > 0 || data.chains.length > 0) {
    const motifToggle = document.createElement('button');
    motifToggle.innerHTML = `<span style="color:#ff00ff;text-shadow:0 0 6px #f0f">MOTIFS & CHAINS (${data.motifs.length + data.chains.length})</span> <span id="motif-arrow" style="color:#ff00ff">&#9660;</span>`;
    motifToggle.style.cssText = `
      background: transparent;
      border: 1px solid #ff00ff;
      font-family: 'Courier New', monospace;
      font-size: 11px;
      padding: 4px 8px;
      cursor: pointer;
      width: 100%;
      margin-bottom: 4px;
      text-align: left;
      box-shadow: 0 0 6px rgba(255,0,255,0.15);
    `;

    const motifBody = document.createElement('div');
    motifBody.id = 'motif-body';
    motifBody.style.cssText = 'display:none;';

    motifToggle.addEventListener('click', () => {
      const open = motifBody.style.display !== 'none';
      motifBody.style.display = open ? 'none' : 'block';
      motifToggle.querySelector('#motif-arrow')!.innerHTML = open ? '&#9660;' : '&#9650;';
    });

    panel.appendChild(motifToggle);

    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'CLEAR HIGHLIGHT';
    clearBtn.style.cssText = `
      background: transparent;
      border: 1px solid #555;
      color: #ccc;
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
    motifBody.appendChild(clearBtn);

    const motifList = buildMotifList(data);
    motifList.style.cssText = `
      max-height: 150px;
      overflow-y: auto;
    `;
    motifBody.appendChild(motifList);
    panel.appendChild(motifBody);

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
        color: #ddd;
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
  const graphTop = new THREE.Vector3();
  const graphBottom = new THREE.Vector3();
  box.getCenter(graphTop);
  box.getCenter(graphBottom);
  graphTop.y = box.max.y;
  graphBottom.y = box.min.y;

  const size = box.getSize(new THREE.Vector3());
  const graphSpan = Math.max(size.x, size.y, size.z);

  camera.far = Math.max(graphSpan * 6, 10000);
  camera.updateProjectionMatrix();

  const startTarget = new THREE.Vector3(graphTop.x, graphTop.y, graphTop.z);
  camera.position.set(startTarget.x, startTarget.y + 10, startTarget.z + 40);
  controls.target.copy(startTarget);
  controls.update();
}

function showFlowOverlay(text: string) {
  let overlay = document.getElementById('flow-overlay');
  if (overlay) { overlay.remove(); }

  overlay = document.createElement('div');
  overlay.id = 'flow-overlay';
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.95);
    z-index: 1000;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px;
    overflow-y: auto;
  `;

  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'CLOSE';
  closeBtn.style.cssText = `
    position: fixed;
    top: 20px;
    right: 30px;
    background: transparent;
    border: 1px solid #ff0066;
    color: #ff0066;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    padding: 6px 16px;
    cursor: pointer;
    z-index: 1001;
    text-shadow: 0 0 6px #ff0066;
  `;
  closeBtn.addEventListener('click', () => overlay!.remove());

  const content = document.createElement('pre');
  content.style.cssText = `
    color: #0ff;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
    max-width: 800px;
    width: 100%;
    white-space: pre-wrap;
    word-wrap: break-word;
    text-shadow: 0 0 4px rgba(0, 255, 255, 0.3);
  `;
  content.textContent = text;

  overlay.appendChild(closeBtn);
  overlay.appendChild(content);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay!.remove();
  });
}
