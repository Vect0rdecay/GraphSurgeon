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
import { buildPatternsPanel } from './patterns';
import { createSpaceBgTexture } from './space-bg';

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
  threeScene.background = createSpaceBgTexture();
  threeScene.fog = new THREE.FogExp2(0x04060a, 0.002);

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

  const dir1 = new THREE.DirectionalLight(0x7fd4ff, 1.5);
  dir1.position.set(1, 1, 1);
  threeScene.add(dir1);

  const dir2 = new THREE.DirectionalLight(0xffb454, 0.8);
  dir2.position.set(-1, -1, -1);
  threeScene.add(dir2);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(threeScene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.3,
    0.4,
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
  setupSideTab();
  renderer.domElement.addEventListener('contextmenu', onRightClick);
  renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    keysDown.add(e.key.toLowerCase());
  });
  window.addEventListener('keyup', (e) => {
    keysDown.delete(e.key.toLowerCase());
  });

  addGlobalStyles();
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
    .map(([k, v]) => `<div class="field"><span class="label">${k}</span> <span class="value" style="word-break:break-all">${JSON.stringify(v)}</span></div>`)
    .join('');

  const motifs = node.motif_ids.length > 0
    ? `<div class="field"><span class="label">MOTIFS</span> <span class="value">${node.motif_ids.join(', ')}</span></div>`
    : '';

  const gadgets = node.gadget_ids.length > 0
    ? `<div class="field"><span class="label">GADGETS</span> <span class="value">${node.gadget_ids.join(', ')}</span></div>`
    : '';

  const profileRows: string[] = [];
  const pRow = (label: string, val: number, desc: string, color: string) => {
    const barPct = Math.min(val / 10, 1) * 100;
    return `<div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:8.5px;letter-spacing:.2em;color:var(--sub)">${label}</span>
        <span style="color:#e8f0f8;font-size:11px;letter-spacing:.08em">${val.toFixed(2)}</span>
      </div>
      <div style="height:3px;background:var(--line2);position:relative;margin:4px 0">
        <div style="position:absolute;left:0;top:0;bottom:0;width:${barPct}%;background:${color};opacity:.85"></div>
      </div>
      <div style="font-family:var(--serif);font-size:10px;color:var(--sub);font-style:italic">${desc}</div>
    </div>`;
  };
  if (node.gradient_sensitivity > 0)
    profileRows.push(pRow('GRADIENT SENSITIVITY', node.gradient_sensitivity,
      'How much this node amplifies gradient signals during backpropagation', 'var(--cyan)'));
  if (node.lipschitz_estimate > 1)
    profileRows.push(pRow('LIPSCHITZ ESTIMATE', node.lipschitz_estimate,
      'Upper bound on output change per unit input change — higher means less stable', 'var(--amber)'));
  if (node.perturbation_amplification > 1)
    profileRows.push(pRow('PERTURBATION AMP', node.perturbation_amplification,
      'Factor by which small input perturbations grow passing through this node', 'var(--red)'));
  if (node.shadowlogic_capacity > 0)
    profileRows.push(pRow('SHADOWLOGIC CAPACITY', node.shadowlogic_capacity,
      'Spare parameter capacity that could conceal injected logic', 'var(--amber)'));
  if (node.extraction_leakage > 0)
    profileRows.push(pRow('EXTRACTION LEAKAGE', node.extraction_leakage,
      'How much internal representation this node exposes to output observers', 'var(--green)'));

  const profileSection = profileRows.length > 0
    ? `<div class="section-head" style="margin-top:14px">STRUCTURAL PROFILE</div>${profileRows.join('')}`
    : '';

  content.innerHTML = `
    <div style="font-size:10px;letter-spacing:.26em;color:var(--cyan)">${node.category.toUpperCase()}</div>
    <h2>${node.op_type}</h2>
    <div style="font-size:9.5px;letter-spacing:.16em;color:var(--sub);margin-bottom:14px">${node.id}</div>
    <div class="field"><span class="label">DEPTH</span> <span class="value">${node.depth}</span></div>
    <div class="field"><span class="label">POSITION</span> <span class="value">${node.position}</span></div>
    <div class="field"><span class="label">EXEC INDEX</span> <span class="value">${node.exec_index}</span></div>
    <div class="field"><span class="label">INPUTS</span> <span class="value" style="word-break:break-all">${node.inputs.join(', ')}</span></div>
    <div class="field"><span class="label">OUTPUTS</span> <span class="value" style="word-break:break-all">${node.outputs.join(', ')}</span></div>
    ${node.param_count > 0 ? `<div class="field"><span class="label">PARAMS</span> <span class="value">${node.param_count.toLocaleString()}</span></div>` : ''}
    ${motifs}
    ${gadgets}
    ${profileSection}
    ${attrs ? `<div class="section-head" style="margin-top:14px">ATTRIBUTES</div>${attrs}` : ''}
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
    <div id="legend-toggle" style="cursor:pointer;pointer-events:all;color:var(--sub);font-size:9px;letter-spacing:.22em;margin-bottom:4px;user-select:none">▼ NODE TYPES</div>
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

function setupSideTab() {
  const tab = document.getElementById('side-tab')!;
  const panel = document.getElementById('side-panel')!;
  tab.addEventListener('click', () => {
    panel.classList.toggle('hidden');
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
    ${data.model.name} · ${data.model.total_nodes} nodes · depth ${data.model.max_depth} · opset ${data.model.opset}
    <div style="font-size:9px;color:var(--sub);margin-top:4px;letter-spacing:.14em">WASD fly · Q/E up/down · mouse orbit · scroll zoom</div>
  `;

  initFlow(data);
  updateSearchResults(data);
  buildControlPanel(data);
  frameAll();
}

function buildControlPanel(data: SceneGraph) {
  const panel = document.getElementById('side-panel')!;
  const hud = document.getElementById('hud')!;
  panel.innerHTML = '';
  panel.appendChild(hud);

  // Flow controls
  const flowHead = document.createElement('div');
  flowHead.className = 'section-head';
  flowHead.textContent = 'FLOW';
  panel.appendChild(flowHead);

  const flowStartBtn = makeBtn('WALK FLOW');
  flowStartBtn.style.marginBottom = '4px';

  const flowNavRow = document.createElement('div');
  flowNavRow.style.cssText = 'display:none;gap:4px;margin-bottom:12px;';

  const prevBtn = makeBtn('◀ PREV');
  prevBtn.style.flex = '1';
  const nextBtn = makeBtn('NEXT ▶');
  nextBtn.style.flex = '1';
  const exitBtn = makeBtn('✕', 'danger');
  exitBtn.style.cssText += 'flex:0 0 32px;';

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

  if (data.model.flow_description) {
    const flowDescBtn = makeBtn('VIEW MODEL FLOW');
    flowDescBtn.style.marginBottom = '12px';
    flowDescBtn.addEventListener('click', () => {
      showFlowOverlay(data.model.flow_description!);
    });
    panel.appendChild(flowDescBtn);
  }

  // Navigation
  const navHead = document.createElement('div');
  navHead.className = 'section-head';
  navHead.textContent = 'NAVIGATION';
  panel.appendChild(navHead);

  const navRow = document.createElement('div');
  navRow.style.cssText = 'display:flex;gap:4px;margin-bottom:12px;';

  const startBtn = makeBtn('START');
  startBtn.style.flex = '1';
  startBtn.addEventListener('click', () => flyToGraphEnd('start'));
  navRow.appendChild(startBtn);

  const endBtn = makeBtn('END', 'accent');
  endBtn.style.flex = '1';
  endBtn.addEventListener('click', () => flyToGraphEnd('end'));
  navRow.appendChild(endBtn);

  panel.appendChild(navRow);

  // ShadowLogic
  if (data.shadowlogic && data.shadowlogic.injection_points.length > 0 && builtScene) {
    const slHead = document.createElement('div');
    slHead.className = 'section-head';
    slHead.textContent = 'SHADOWLOGIC';
    panel.appendChild(slHead);

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

    const slToggleBtn = makeBtn('SHOW INJECTION POINTS', 'accent');
    slToggleBtn.style.marginBottom = '6px';
    slToggleBtn.addEventListener('click', () => {
      if (!builtScene?.shadowlogicMarkers) return;
      const g = builtScene.shadowlogicMarkers.group;
      g.visible = !g.visible;
      slToggleBtn.textContent = g.visible ? 'HIDE INJECTION POINTS' : 'SHOW INJECTION POINTS';
    });
    panel.appendChild(slToggleBtn);
  }

  // Structural Patterns
  if (data.structural_patterns && builtScene) {
    const patHead = document.createElement('div');
    patHead.className = 'section-head';
    patHead.textContent = 'PATTERNS';
    panel.appendChild(patHead);

    const patternsEl = buildPatternsPanel(data.structural_patterns, builtScene);
    panel.appendChild(patternsEl);
  }

  // Motifs & Regions
  if (data.motifs.length > 0 || data.chains.length > 0) {
    const motifHead = document.createElement('div');
    motifHead.className = 'section-head';
    motifHead.textContent = `MOTIFS & CHAINS (${data.motifs.length + data.chains.length})`;
    panel.appendChild(motifHead);

    const regionBtn = makeBtn('SHOW ALL REGIONS', 'accent');
    regionBtn.style.marginBottom = '6px';
    regionBtn.addEventListener('click', () => {
      if (!builtScene) return;
      const showing = toggleAllRegions(builtScene.motifRegions);
      regionBtn.textContent = showing ? 'HIDE ALL REGIONS' : 'SHOW ALL REGIONS';
    });
    panel.appendChild(regionBtn);

    const clearBtn = makeBtn('CLEAR HIGHLIGHT');
    clearBtn.style.marginBottom = '6px';
    clearBtn.addEventListener('click', () => {
      if (builtScene) clearHighlight(builtScene);
      closeCatalog();
    });
    panel.appendChild(clearBtn);

    const motifList = buildMotifList(data);
    panel.appendChild(motifList);

    motifList.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const entry = target.closest('.motif-entry') as HTMLElement;
      if (!entry || !builtScene || !sceneData) return;

      const motifId = entry.dataset.motifId!;
      highlightMotif(motifId, sceneData, builtScene);
      showCatalogEntry(motifId, sceneData);
    });
  }
}

function makeBtn(text: string, variant?: 'accent' | 'danger'): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.textContent = text;
  btn.className = 'chromebtn' + (variant ? ` ${variant}` : '');
  return btn;
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
    background: rgba(3,5,9,.96);
    z-index: 1000;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 60px 40px;
    overflow-y: auto;
  `;

  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'CLOSE';
  closeBtn.className = 'chromebtn danger';
  closeBtn.style.cssText += 'position:fixed;top:20px;right:30px;z-index:1001;width:auto;padding:7px 18px;';
  closeBtn.addEventListener('click', () => overlay!.remove());

  const content = document.createElement('pre');
  content.style.cssText = `
    color: var(--ink);
    font-family: var(--serif);
    font-size: 14px;
    line-height: 1.7;
    max-width: 800px;
    width: 100%;
    white-space: pre-wrap;
    word-wrap: break-word;
    letter-spacing: .04em;
  `;
  content.textContent = text;

  overlay.appendChild(closeBtn);
  overlay.appendChild(content);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay!.remove();
  });
}

function addGlobalStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .motif-entry {
      padding: 5px 6px;
      cursor: pointer;
      border-left: 2px solid transparent;
      margin-bottom: 2px;
      transition: all 0.15s;
      display: flex;
      align-items: center;
    }
    .motif-entry:hover {
      border-left-color: var(--cyan);
      background: rgba(127,212,255,.04);
    }
    .motif-title {
      color: var(--ink);
      margin-left: 4px;
      font-size: 10px;
      letter-spacing: .08em;
    }
    .motif-entry:hover .motif-title {
      color: #eef5fc;
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
}
