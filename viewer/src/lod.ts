import * as THREE from 'three';
import type { BuiltScene } from './scene-builder';

const LABEL_DISTANCE_THRESHOLD = 80;

export function updateLabelLOD(
  built: BuiltScene,
  camera: THREE.PerspectiveCamera,
) {
  const camPos = camera.position;

  for (const sprite of built.labelSprites) {
    const dist = camPos.distanceTo(sprite.position);
    sprite.visible = dist < LABEL_DISTANCE_THRESHOLD;
    if (sprite.visible) {
      const scale = Math.min(1.0, LABEL_DISTANCE_THRESHOLD / (dist + 1));
      sprite.material.opacity = scale;
    }
  }
}
