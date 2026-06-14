import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { SceneGraph, SceneNode } from './types';
import { buildThreeScene, type BuiltScene } from './scene-builder';
import { getAllCategories } from './colors';
import { highlightMotif, clearHighlight, buildMotifList } from './motifs';
import { toggleAllRegions } from './motif-regions';
import { initFlow, startFlow, stopFlow, isFlowPlaying, stepNext, stepPrev } from './flow';
import { initCatalogDrawer, showCatalogEntry, closeCatalog } from './catalog-drawer';
import { initEditMode, showContextMenu, hideContextMenu } from './edit-mode';
import { initSearch, updateSearchResults, pulseNode } from './search';
import { updateLabelLOD } from './lod';
import { buildShadowLogicPanel, showShadowLogicDetail } from './shadowlogic';
import { buildPatternsPanel, clearPatternHighlight } from './patterns';

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
  controls.minDistance = 1;
  controls.maxDistance = 5000;
  controls.enableZoom = false;

  const ambient = new THREE.AmbientLight(0x334466, 1.0);
  threeScene.add(ambient);

  const dir1 = new THREE.DirectionalLight(0x00ffff, 1.5);
  dir1.position.set(1, 1, 1);
  threeScene.add(dir1);

  const dir2 = new THREE.DirectionalLight(0xff6633, 1.0);
  dir2.position.set(-1, -1, -1);
  threeScene.add(dir2);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(threeScene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.35,
    0.3,
    0.92,
  );
  composer.addPass(bloomPass);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();
  clock = new THREE.Clock();

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onClick);
  document.getElementById('close-detail')!.addEventListener('click', closeDetail);

  initCatalogDrawer((nodeId) => {
    if (builtScene) pulseNode(nodeId, builtScene, camera, controls);
  });
  setupDragDrop();
  setupEditMode();
  initSearch((nodeId) => {
    if (builtScene) pulseNode(nodeId, builtScene, camera, controls);
  });
  buildLegend();
  renderer.domElement.addEventListener('contextmenu', onRightClick);
  renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

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

  if (builtScene) {
    updateLabelLOD(builtScene, camera);
    builtScene.motifRegions.animate(clock.elapsedTime);
    if (builtScene.shadowlogicMarkers) {
      builtScene.shadowlogicMarkers.animate(clock.elapsedTime);
    }
  }

  composer.render();
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

function onWheel(event: WheelEvent) {
  event.preventDefault();
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const speed = Math.max(2, camera.position.distanceTo(controls.target) * 0.1);
  const move = forward.multiplyScalar(event.deltaY > 0 ? -speed : speed);
  camera.position.add(move);
  controls.target.add(move);
  controls.update();
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

  // Check shadowlogic hit meshes
  if (builtScene.shadowlogicMarkers && builtScene.shadowlogicMarkers.group.visible) {
    const slHits = builtScene.shadowlogicMarkers.hitMeshes;
    const slIntersects = raycaster.intersectObjects(slHits);
    if (slIntersects.length > 0) {
      const point = slIntersects[0].object.userData.shadowlogicPoint;
      if (point) {
        const popup = showShadowLogicDetail(point);
        document.getElementById('app')!.appendChild(popup);
        popup.querySelector('#sl-detail-close')!.addEventListener('click', () => popup.remove());
        return;
      }
    }
  }

  // Check motif region hit meshes (only visible groups)
  const visibleHits = builtScene.motifRegions.hitMeshes.filter(m => m.parent?.visible);
  if (visibleHits.length > 0 && sceneData) {
    const motifIntersects = raycaster.intersectObjects(visibleHits);
    if (motifIntersects.length > 0) {
      const motifId = motifIntersects[0].object.userData.motifId as string;
      if (motifId) {
        showCatalogEntry(motifId, sceneData);
        return;
      }
    }
  }

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
    .map(([k, v]) => `<div class="field"><span class="label">${k}:</span> <span class="value" style="word-break:break-all">${JSON.stringify(v)}</span></div>`)
    .join('');

  const motifs = node.motif_ids.length > 0
    ? `<div class="field"><span class="label">motifs:</span> <span class="value">${node.motif_ids.join(', ')}</span></div>`
    : '';

  const gadgets = node.gadget_ids.length > 0
    ? `<div class="field"><span class="label">gadgets:</span> <span class="value">${node.gadget_ids.join(', ')}</span></div>`
    : '';

  const profileRows: string[] = [];
  const pRow = (label: string, val: number, desc: string, color: string) => {
    const barPct = Math.min(val / 10, 1) * 100;
    return `<div style="margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="color:${color};font-weight:bold;font-size:11px">${label}</span>
        <span style="color:#fff;font-size:12px">${val.toFixed(2)}</span>
      </div>
      <div style="background:rgba(255,255,255,0.06);height:4px;border-radius:2px;margin:3px 0">
        <div style="background:${color};height:100%;width:${barPct}%;border-radius:2px"></div>
      </div>
      <div style="color:#aaa;font-size:9px">${desc}</div>
    </div>`;
  };
  if (node.gradient_sensitivity > 0)
    profileRows.push(pRow('Gradient Sensitivity', node.gradient_sensitivity,
      'How much this node amplifies gradient signals during backpropagation', '#0ff'));
  if (node.lipschitz_estimate > 1)
    profileRows.push(pRow('Lipschitz Estimate', node.lipschitz_estimate,
      'Upper bound on output change per unit input change — higher means less stable', '#ff6600'));
  if (node.perturbation_amplification > 1)
    profileRows.push(pRow('Perturbation Amplification', node.perturbation_amplification,
      'Factor by which small input perturbations grow passing through this node', '#ff3300'));
  if (node.shadowlogic_capacity > 0)
    profileRows.push(pRow('ShadowLogic Capacity', node.shadowlogic_capacity,
      'Spare parameter capacity that could conceal injected logic', '#ff6600'));
  if (node.extraction_leakage > 0)
    profileRows.push(pRow('Extraction Leakage', node.extraction_leakage,
      'How much internal representation this node exposes to output observers', '#00ff41'));
  const profileSection = profileRows.length > 0
    ? `<h2 style="margin-top:10px;color:#ff6600;border-top:1px solid rgba(255,102,0,0.3);padding-top:8px">Structural Profile</h2>${profileRows.join('')}`
    : '';

  content.innerHTML = `
    <h2>${node.op_type}: ${node.id}</h2>
    <div class="field"><span class="label">category:</span> <span class="value">${node.category}</span></div>
    <div class="field"><span class="label">depth:</span> <span class="value">${node.depth}</span></div>
    <div class="field"><span class="label">position:</span> <span class="value">${node.position}</span></div>
    <div class="field"><span class="label">exec_index:</span> <span class="value">${node.exec_index}</span></div>
    <div class="field"><span class="label">inputs:</span> <span class="value" style="word-break:break-all">${node.inputs.join(', ')}</span></div>
    <div class="field"><span class="label">outputs:</span> <span class="value" style="word-break:break-all">${node.outputs.join(', ')}</span></div>
    ${node.param_count > 0 ? `<div class="field"><span class="label">params:</span> <span class="value">${node.param_count.toLocaleString()}</span></div>` : ''}
    ${motifs}
    ${gadgets}
    ${profileSection}
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
  const entries = cats
    .map(([name, color]) => {
      const hex = '#' + color.toString(16).padStart(6, '0');
      return `<div class="entry"><span class="swatch" style="background:${hex};box-shadow:0 0 4px ${hex}"></span>${name}</div>`;
    })
    .join('');

  legend.innerHTML = `
    <div id="legend-toggle" style="cursor:pointer;pointer-events:all;color:#0ff;font-size:11px;text-shadow:0 0 6px #0ff;margin-bottom:4px;user-select:none">▼ NODE TYPES</div>
    <div id="legend-entries">${entries}</div>
  `;

  const toggle = document.getElementById('legend-toggle')!;
  const entriesDiv = document.getElementById('legend-entries')!;
  toggle.addEventListener('click', () => {
    const collapsed = entriesDiv.style.display === 'none';
    entriesDiv.style.display = collapsed ? '' : 'none';
    toggle.textContent = collapsed ? '▼ NODE TYPES' : '▶ NODE TYPES';
  });
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

  const modelNameSpan = document.getElementById('hud-model-name');
  if (modelNameSpan) {
    modelNameSpan.textContent = data.model.source_file ? `// ${data.model.source_file}` : '';
  }

  const hud = document.getElementById('hud-info')!;
  hud.innerHTML = `
    ${data.model.name} | ${data.model.total_nodes} nodes | depth ${data.model.max_depth} | opset ${data.model.opset}
    <div style="font-size:10px;color:#0aa;margin-top:2px">WASD fly | Q/E up/down | mouse orbit | scroll zoom<br>click node: details | motifs: click to highlight</div>
  `;

  initFlow(data);

  updateSearchResults(data);
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

  // Flow step-through controls
  const flowBtnStyle = `
    background: transparent;
    border: 1px solid #0ff;
    color: #0ff;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    padding: 5px 0;
    cursor: pointer;
    text-shadow: 0 0 6px #0ff;
    flex: 1;
  `;

  const flowStartBtn = document.createElement('button');
  flowStartBtn.textContent = 'WALK FLOW';
  flowStartBtn.style.cssText = flowBtnStyle + 'width:100%;margin-bottom:4px;box-shadow:0 0 8px rgba(0,255,255,0.2);';

  const flowNavRow = document.createElement('div');
  flowNavRow.style.cssText = 'display:flex;gap:4px;margin-bottom:12px;display:none;';

  const prevBtn = document.createElement('button');
  prevBtn.textContent = '◀ PREV';
  prevBtn.style.cssText = flowBtnStyle;

  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'NEXT ▶';
  nextBtn.style.cssText = flowBtnStyle;

  const exitBtn = document.createElement('button');
  exitBtn.textContent = '✕';
  exitBtn.style.cssText = flowBtnStyle.replace('#0ff', '#ff0066') + 'flex:0 0 32px;border-color:#ff0066;';

  flowNavRow.appendChild(prevBtn);
  flowNavRow.appendChild(nextBtn);
  flowNavRow.appendChild(exitBtn);

  const endFlow = () => {
    stopFlow();
    flowStartBtn.style.display = '';
    flowNavRow.style.display = 'none';
    if (builtScene) clearHighlight(builtScene);
  };

  flowStartBtn.addEventListener('click', () => {
    if (!builtScene) return;
    startFlow(builtScene, camera, controls, (nodeId) => {
      const node = sceneData?.nodes.find(n => n.id === nodeId);
      if (node) showDetail(node);
    });
    flowStartBtn.style.display = 'none';
    flowNavRow.style.display = 'flex';
    closeCatalog();
  });

  nextBtn.addEventListener('click', () => {
    if (isFlowPlaying()) stepNext();
    if (!isFlowPlaying()) endFlow();
  });

  prevBtn.addEventListener('click', () => {
    if (isFlowPlaying()) stepPrev();
  });

  exitBtn.addEventListener('click', endFlow);

  panel.appendChild(flowStartBtn);
  panel.appendChild(flowNavRow);

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

  // Navigation buttons
  const navRow = document.createElement('div');
  navRow.style.cssText = 'display:flex;gap:4px;margin-bottom:12px;';

  const startBtn = document.createElement('button');
  startBtn.textContent = 'GO TO START';
  startBtn.style.cssText = `
    flex:1;background:transparent;border:1px solid #0ff;color:#0ff;
    font-family:'Courier New',monospace;font-size:10px;padding:4px;
    cursor:pointer;text-shadow:0 0 4px #0ff;
  `;
  startBtn.addEventListener('click', () => flyToGraphEnd('start'));
  navRow.appendChild(startBtn);

  const endBtn = document.createElement('button');
  endBtn.textContent = 'GO TO END';
  endBtn.style.cssText = `
    flex:1;background:transparent;border:1px solid #ff6600;color:#ff6600;
    font-family:'Courier New',monospace;font-size:10px;padding:4px;
    cursor:pointer;text-shadow:0 0 4px #f60;
  `;
  endBtn.addEventListener('click', () => flyToGraphEnd('end'));
  navRow.appendChild(endBtn);

  panel.appendChild(navRow);

  // ShadowLogic panel
  if (data.shadowlogic && data.shadowlogic.injection_points.length > 0 && builtScene) {
    const slPanel = buildShadowLogicPanel(data.shadowlogic, (nodeId) => {
      if (!builtScene) return;
      const mesh = builtScene.nodeObjects.get(nodeId);
      if (!mesh) return;
      const yOffset = camera.position.y - controls.target.y;
      controls.target.set(mesh.position.x, mesh.position.y, mesh.position.z);
      camera.position.y = mesh.position.y + yOffset;
      controls.update();
      const node = data.nodes.find(n => n.id === nodeId);
      if (node) showDetail(node);
    });
    panel.appendChild(slPanel);

    const slToggleBtn = document.createElement('button');
    slToggleBtn.textContent = 'SHOW INJECTION POINTS';
    slToggleBtn.style.cssText = `
      background:transparent;border:1px solid #ff6600;color:#ff6600;
      font-family:'Courier New',monospace;font-size:10px;padding:4px 8px;
      cursor:pointer;width:100%;margin-bottom:6px;
      text-shadow:0 0 4px #f60;box-shadow:0 0 6px rgba(255,102,0,0.15);
    `;
    slToggleBtn.addEventListener('click', () => {
      if (!builtScene?.shadowlogicMarkers) return;
      const g = builtScene.shadowlogicMarkers.group;
      g.visible = !g.visible;
      slToggleBtn.textContent = g.visible ? 'HIDE INJECTION POINTS' : 'SHOW INJECTION POINTS';
    });
    panel.appendChild(slToggleBtn);
  }

  // Structural patterns panel
  if (data.structural_patterns && builtScene) {
    const patternsEl = buildPatternsPanel(data.structural_patterns, builtScene);
    panel.appendChild(patternsEl);
  }

  // Show all regions toggle
  if (data.motifs.length > 0 || data.chains.length > 0) {
    const regionBtn = document.createElement('button');
    regionBtn.textContent = 'SHOW ALL REGIONS';
    regionBtn.style.cssText = `
      background:transparent;border:1px solid #ff6600;color:#ff6600;
      font-family:'Courier New',monospace;font-size:10px;padding:4px 8px;
      cursor:pointer;width:100%;margin-bottom:6px;
      text-shadow:0 0 4px #f60;box-shadow:0 0 6px rgba(255,102,0,0.15);
    `;
    regionBtn.addEventListener('click', () => {
      if (!builtScene) return;
      const showing = toggleAllRegions(builtScene.motifRegions);
      regionBtn.textContent = showing ? 'HIDE ALL REGIONS' : 'SHOW ALL REGIONS';
    });
    panel.appendChild(regionBtn);
  }

  // Motifs section — collapsible header
  if (data.motifs.length > 0 || data.chains.length > 0) {
    const motifToggle = document.createElement('button');
    motifToggle.innerHTML = `<span style="color:#ff6600;text-shadow:0 0 6px #f60">MOTIFS & CHAINS (${data.motifs.length + data.chains.length})</span> <span id="motif-arrow" style="color:#ff6600">&#9660;</span>`;
    motifToggle.style.cssText = `
      background: transparent;
      border: 1px solid #ff6600;
      font-family: 'Courier New', monospace;
      font-size: 11px;
      padding: 4px 8px;
      cursor: pointer;
      width: 100%;
      margin-bottom: 4px;
      text-align: left;
      box-shadow: 0 0 6px rgba(255,102,0,0.15);
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
        display: flex;
        align-items: center;
      }
      .motif-entry:hover {
        border-left-color: #0ff;
        background: rgba(0, 255, 255, 0.05);
      }
      .motif-title {
        color: #ddd;
        margin-left: 4px;
      }
      .motif-entry:hover .motif-title {
        color: #fff;
      }
      .motif-info-btn {
        opacity: 0.4;
        transition: opacity 0.15s;
      }
      .motif-entry:hover .motif-info-btn {
        opacity: 1;
      }
    `;
    document.head.appendChild(style);

    motifList.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const entry = target.closest('.motif-entry') as HTMLElement;
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

function flyToGraphEnd(which: 'start' | 'end') {
  if (!builtScene) return;

  const box = new THREE.Box3().setFromObject(builtScene.group);
  const target = new THREE.Vector3();

  if (which === 'start') {
    box.getCenter(target);
    target.y = box.max.y;
  } else {
    box.getCenter(target);
    target.y = box.min.y;
  }

  camera.position.set(target.x, target.y + 10, target.z + 30);
  controls.target.copy(target);
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
