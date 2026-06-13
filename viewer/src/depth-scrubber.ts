import * as THREE from 'three';
import type { SceneGraph } from './types';
import type { BuiltScene } from './scene-builder';

let scrubberEl: HTMLElement | null = null;

const BAND_COLORS: Record<string, string> = {
  early: '#00ffff',
  middle: '#ff00ff',
  late: '#ff6600',
};

export function initDepthScrubber(
  onBandClick: (position: string) => void,
) {
  scrubberEl = document.createElement('div');
  scrubberEl.id = 'depth-scrubber';
  scrubberEl.style.cssText = `
    position: absolute;
    right: 16px;
    top: 50%;
    transform: translateY(-50%);
    width: 30px;
    z-index: 10;
    font-family: 'Courier New', monospace;
    font-size: 10px;
  `;
  document.getElementById('app')!.appendChild(scrubberEl);

  scrubberEl.addEventListener('click', (e) => {
    const band = (e.target as HTMLElement).closest('.scrub-band') as HTMLElement;
    if (band) onBandClick(band.dataset.position!);
  });
}

export function updateScrubber(scene: SceneGraph) {
  if (!scrubberEl) return;

  const counts: Record<string, number> = { early: 0, middle: 0, late: 0 };
  for (const node of scene.nodes) {
    counts[node.position] = (counts[node.position] || 0) + 1;
  }

  const total = scene.nodes.length || 1;

  scrubberEl.innerHTML = ['early', 'middle', 'late'].map(pos => {
    const pct = Math.max(20, (counts[pos] / total) * 100);
    const color = BAND_COLORS[pos];
    return `
      <div class="scrub-band" data-position="${pos}" style="
        height: ${pct}px;
        background: ${color}22;
        border-left: 3px solid ${color};
        margin-bottom: 2px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        color: ${color};
        text-shadow: 0 0 4px ${color};
        padding: 4px 2px;
        transition: background 0.15s;
      " title="${pos}: ${counts[pos]} nodes">
        ${pos[0].toUpperCase()}
      </div>
    `;
  }).join('');
}

export function flyToBand(
  position: string,
  scene: SceneGraph,
  built: BuiltScene,
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; update: () => void },
) {
  const bandNodes = scene.nodes.filter(n => n.position === position);
  if (bandNodes.length === 0) return;

  let sumX = 0, sumY = 0, sumZ = 0;
  let count = 0;

  for (const node of bandNodes) {
    const pos = built.positions.get(node.id);
    if (pos) {
      sumX += pos.x;
      sumY += pos.y;
      sumZ += pos.z;
      count++;
    }
  }

  if (count === 0) return;

  const center = new THREE.Vector3(sumX / count, sumY / count, sumZ / count);
  camera.position.lerp(
    new THREE.Vector3(center.x, center.y, center.z + 20),
    0.5,
  );
  controls.target.copy(center);
  controls.update();
}
