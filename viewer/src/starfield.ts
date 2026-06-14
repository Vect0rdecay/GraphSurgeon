import * as THREE from 'three';

const STAR_COUNT = 2500;
const SPHERE_RADIUS = 800;

export function createStarfield(): THREE.Points {
  const positions = new Float32Array(STAR_COUNT * 3);
  const sizes = new Float32Array(STAR_COUNT);
  const colors = new Float32Array(STAR_COUNT * 3);

  for (let i = 0; i < STAR_COUNT; i++) {
    const theta = Math.acos(2 * seededRandom(i * 3) - 1);
    const phi = 2 * Math.PI * seededRandom(i * 3 + 1);
    const r = SPHERE_RADIUS * (0.4 + 0.6 * seededRandom(i * 3 + 2));

    positions[i * 3] = r * Math.sin(theta) * Math.cos(phi);
    positions[i * 3 + 1] = r * Math.sin(theta) * Math.sin(phi);
    positions[i * 3 + 2] = r * Math.cos(theta);

    sizes[i] = 0.5 + seededRandom(i * 7) * 2.0;

    const temp = seededRandom(i * 11);
    if (temp < 0.3) {
      colors[i * 3] = 0.5; colors[i * 3 + 1] = 0.7; colors[i * 3 + 2] = 1.0;
    } else if (temp < 0.5) {
      colors[i * 3] = 0.7; colors[i * 3 + 1] = 0.85; colors[i * 3 + 2] = 1.0;
    } else {
      colors[i * 3] = 1.0; colors[i * 3 + 1] = 1.0; colors[i * 3 + 2] = 1.0;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 1.2,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true,
    depthWrite: false,
  });

  const points = new THREE.Points(geo, mat);
  points.renderOrder = -100;
  return points;
}

export function animateStarfield(stars: THREE.Points, elapsed: number) {
  stars.rotation.y = elapsed * 0.003;
  stars.rotation.x = elapsed * 0.001;
}

function seededRandom(seed: number): number {
  const x = Math.sin(seed + 1) * 43758.5453;
  return x - Math.floor(x);
}
