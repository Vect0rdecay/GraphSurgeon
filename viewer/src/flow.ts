import * as THREE from 'three';
import type { SceneGraph } from './types';
import type { BuiltScene } from './scene-builder';

export interface FlowState {
  isPlaying: boolean;
  currentIndex: number;
  speed: number;
  elapsed: number;
}

const STEP_DURATION = 0.5;

let state: FlowState = {
  isPlaying: false,
  currentIndex: -1,
  speed: 1.0,
  elapsed: 0,
};

let sortedNodes: string[] = [];
let onStepCallback: ((index: number, nodeId: string) => void) | null = null;

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

  if (newIndex !== state.currentIndex) {
    state.currentIndex = newIndex;
    const nodeId = sortedNodes[newIndex];

    for (const [id, mesh] of built.nodeObjects) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (id === nodeId) {
        mat.emissiveIntensity = 2.5;
        mat.opacity = 1.0;

        const scale = mesh.scale.x * 1.3;
        mesh.scale.setScalar(scale);
        setTimeout(() => mesh.scale.setScalar(scale / 1.3), 200);
      } else if (sortedNodes.indexOf(id) < newIndex) {
        mat.emissiveIntensity = 0.8;
        mat.opacity = 0.7;
      } else {
        mat.emissiveIntensity = 0.15;
        mat.opacity = 0.25;
      }
    }

    const mesh = built.nodeObjects.get(nodeId);
    if (mesh) {
      const pos = mesh.position;
      controls.target.lerp(pos, 0.3);
      controls.update();
    }

    if (onStepCallback) onStepCallback(newIndex, nodeId);
  }
}

function resetNodeVisuals(built: BuiltScene) {
  for (const [, mesh] of built.nodeObjects) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = 0.6;
    mat.opacity = 0.9;
  }
}

export function getFlowState(): FlowState {
  return { ...state };
}
