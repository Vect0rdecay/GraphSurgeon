import * as THREE from 'three';
import type { SceneGraph } from './types';
import type { BuiltScene } from './scene-builder';

export interface FlowState {
  isPlaying: boolean;
  currentIndex: number;
  speed: number;
  elapsed: number;
}

const STEP_DURATION = 1.5;
const PULSE_DURATION = 0.4;

let state: FlowState = {
  isPlaying: false,
  currentIndex: -1,
  speed: 1.0,
  elapsed: 0,
};

let sortedNodes: string[] = [];
let onStepCallback: ((index: number, nodeId: string) => void) | null = null;
let originalScales: Map<string, number> = new Map();

export function initFlow(scene: SceneGraph, onStep: (index: number, nodeId: string) => void) {
  sortedNodes = [...scene.nodes]
    .sort((a, b) => a.exec_index - b.exec_index)
    .map(n => n.id);
  onStepCallback = onStep;
}

export function startFlow() {
  state.isPlaying = true;
  state.currentIndex = -1;
  state.elapsed = 0;
}

export function stopFlow() {
  state.isPlaying = false;
  state.currentIndex = -1;
}

export function toggleFlow() {
  if (state.isPlaying) stopFlow();
  else startFlow();
}

export function isFlowPlaying(): boolean {
  return state.isPlaying;
}

export function setFlowSpeed(s: number) {
  state.speed = s;
}

export function updateFlow(
  deltaTime: number,
  built: BuiltScene,
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; update: () => void },
) {
  if (!state.isPlaying) return;

  state.elapsed += deltaTime * state.speed;

  const newIndex = Math.floor(state.elapsed / STEP_DURATION);
  if (newIndex >= sortedNodes.length) {
    stopFlow();
    resetNodeVisuals(built);
    return;
  }

  const stepProgress = (state.elapsed - newIndex * STEP_DURATION) / STEP_DURATION;

  if (newIndex !== state.currentIndex) {
    state.currentIndex = newIndex;
    const nodeId = sortedNodes[newIndex];

    if (originalScales.size === 0) {
      for (const [id, mesh] of built.nodeObjects) {
        originalScales.set(id, mesh.scale.x);
      }
    }

    for (const [id, mesh] of built.nodeObjects) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      const baseScale = originalScales.get(id) ?? mesh.scale.x;

      if (id === nodeId) {
        mat.emissive.setHex(0xffffff);
        mat.emissiveIntensity = 4.0;
        mat.opacity = 1.0;
        mesh.scale.setScalar(baseScale * 2.0);
      } else if (sortedNodes.indexOf(id) < newIndex) {
        mat.emissiveIntensity = 0.8;
        mat.opacity = 0.7;
        mesh.scale.setScalar(baseScale);
      } else {
        mat.emissiveIntensity = 0.1;
        mat.opacity = 0.2;
        mesh.scale.setScalar(baseScale);
      }
    }

    if (onStepCallback) onStepCallback(newIndex, nodeId);
  }

  // Smoothly track the active node vertically — no lateral or rotational changes
  const activeId = sortedNodes[state.currentIndex];
  const activeMesh = built.nodeObjects.get(activeId);
  if (activeMesh) {
    const targetY = activeMesh.position.y;
    const yOffset = camera.position.y - controls.target.y;
    controls.target.y += (targetY - controls.target.y) * 0.05;
    camera.position.y = controls.target.y + yOffset;
    controls.update();
  }

  const nodeId = sortedNodes[newIndex];
  const mesh = built.nodeObjects.get(nodeId);
  if (mesh) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    const baseScale = originalScales.get(nodeId) ?? mesh.scale.x / 2.0;

    if (stepProgress < PULSE_DURATION) {
      const t = stepProgress / PULSE_DURATION;
      const pulse = 2.0 - t * 0.5;
      mesh.scale.setScalar(baseScale * pulse);
      mat.emissiveIntensity = 4.0 - t * 2.0;
    } else {
      mesh.scale.setScalar(baseScale * 1.5);
      mat.emissiveIntensity = 2.0;
    }

    const catColor = (mesh.material as THREE.MeshStandardMaterial).color.getHex();
    if (stepProgress > 0.1) {
      mat.emissive.setHex(catColor);
    }
  }
}

function resetNodeVisuals(built: BuiltScene) {
  for (const [id, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissive.copy(mat.color);
    mat.emissiveIntensity = 0.6;
    mat.opacity = 0.9;
    const baseScale = originalScales.get(id);
    if (baseScale) mesh.scale.setScalar(baseScale);
  }
  originalScales.clear();
}

export function getFlowState(): FlowState {
  return { ...state };
}
