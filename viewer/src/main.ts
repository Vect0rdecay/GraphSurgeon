import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { SceneGraph, SceneNode } from './types';
import { buildThreeScene, type BuiltScene } from './scene-builder';
import { getAllCategories } from './colors';

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let composer: EffectComposer;
let builtScene: BuiltScene | null = null;
let raycaster: THREE.Raycaster;
let mouse: THREE.Vector2;
let hoveredMesh: THREE.Mesh | null = null;
let selectedMesh: THREE.Mesh | null = null;

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

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  scene.fog = new THREE.FogExp2(0x000000, 0.008);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
  camera.position.set(0, 5, 30);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 3;
  controls.maxDistance = 200;

  const ambient = new THREE.AmbientLight(0x222244, 0.5);
  scene.add(ambient);

  const point1 = new THREE.PointLight(0x00ffff, 1.5, 100);
  point1.position.set(10, 10, 10);
  scene.add(point1);

  const point2 = new THREE.PointLight(0xff00ff, 1.0, 100);
  point2.position.set(-10, -10, -10);
  scene.add(point2);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.8,   // strength
    0.4,   // radius
    0.85,  // threshold
  );
  composer.addPass(bloomPass);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onClick);
  document.getElementById('close-detail')!.addEventListener('click', closeDetail);

  setupDragDrop();
  buildLegend();
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
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

  if (!builtScene) return;

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
  if (!builtScene) return;

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
    }
  } catch {
    // No default scene — user must drag-drop
  }
}

function loadScene(data: SceneGraph) {
  if (builtScene) {
    scene.remove(builtScene.group);
  }

  builtScene = buildThreeScene(data);
  scene.add(builtScene.group);

  const hud = document.getElementById('hud-info')!;
  hud.innerHTML = `
    ${data.model.name} | ${data.model.total_nodes} nodes | depth ${data.model.max_depth} | opset ${data.model.opset}
  `;

  frameAll();
}

function frameAll() {
  if (!builtScene) return;

  const box = new THREE.Box3().setFromObject(builtScene.group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  const dist = maxDim / (2 * Math.tan(fov / 2)) * 1.5;

  camera.position.set(center.x, center.y, center.z + dist);
  controls.target.copy(center);
  controls.update();
}
