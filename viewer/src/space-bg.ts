import * as THREE from 'three';

const W = 2048;
const H = 1200;

function seeded(i: number): number {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function drawCloudLayer(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  rx: number, ry: number,
  r: number, g: number, b: number,
  peakAlpha: number,
) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(1, ry / rx);
  ctx.translate(-cx, -cy);

  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx);
  grad.addColorStop(0, `rgba(${r},${g},${b},${peakAlpha})`);
  grad.addColorStop(0.25, `rgba(${r},${g},${b},${peakAlpha * 0.8})`);
  grad.addColorStop(0.5, `rgba(${r},${g},${b},${peakAlpha * 0.45})`);
  grad.addColorStop(0.75, `rgba(${r},${g},${b},${peakAlpha * 0.15})`);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(cx - rx, cy - rx, rx * 2, rx * 2);
  ctx.restore();
}

function drawStars(ctx: CanvasRenderingContext2D, count: number) {
  for (let i = 0; i < count; i++) {
    const x = seeded(i * 3) * W;
    const y = seeded(i * 3 + 1) * H;
    const brightness = seeded(i * 3 + 2);

    if (brightness > 0.98) {
      const glow = ctx.createRadialGradient(x, y, 0, x, y, 2.5);
      glow.addColorStop(0, 'rgba(220,235,255,0.85)');
      glow.addColorStop(0.5, 'rgba(180,210,255,0.2)');
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glow;
      ctx.fillRect(x - 3, y - 3, 6, 6);
    } else if (brightness > 0.9) {
      ctx.fillStyle = `rgba(200,215,240,${0.4 + brightness * 0.4})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

export function createSpaceBgCanvas(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d')!;

  // dark base with slight gradient
  const base = ctx.createLinearGradient(0, 0, W * 0.3, H);
  base.addColorStop(0, '#020408');
  base.addColorStop(0.5, '#040812');
  base.addColorStop(1, '#030610');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, W, H);

  // --- large nebula clouds - bottom half, heavy blue ---
  // big bottom-left cloud bank
  drawCloudLayer(ctx, W * 0.15, H * 1.0, 900, 500, 12, 30, 75, 0.7);
  drawCloudLayer(ctx, W * 0.25, H * 0.88, 700, 400, 18, 45, 100, 0.55);
  drawCloudLayer(ctx, W * 0.1, H * 0.92, 550, 350, 25, 60, 130, 0.4);
  drawCloudLayer(ctx, W * 0.3, H * 0.82, 450, 280, 30, 70, 140, 0.35);

  // big bottom-right cloud bank
  drawCloudLayer(ctx, W * 0.8, H * 1.05, 800, 450, 10, 28, 70, 0.65);
  drawCloudLayer(ctx, W * 0.7, H * 0.9, 600, 350, 15, 40, 95, 0.5);
  drawCloudLayer(ctx, W * 0.85, H * 0.85, 500, 300, 22, 55, 120, 0.4);

  // --- warm glow upper right - like a sun behind clouds ---
  drawCloudLayer(ctx, W * 0.72, H * 0.2, 400, 300, 100, 60, 25, 0.35);
  drawCloudLayer(ctx, W * 0.7, H * 0.18, 250, 180, 140, 80, 35, 0.25);
  drawCloudLayer(ctx, W * 0.68, H * 0.22, 150, 120, 180, 100, 50, 0.18);

  // --- mid-field wispy atmosphere ---
  drawCloudLayer(ctx, W * 0.45, H * 0.55, 600, 350, 10, 22, 55, 0.25);
  drawCloudLayer(ctx, W * 0.55, H * 0.45, 500, 300, 8, 18, 48, 0.2);
  drawCloudLayer(ctx, W * 0.35, H * 0.65, 450, 280, 12, 28, 65, 0.18);

  // --- upper atmosphere haze ---
  drawCloudLayer(ctx, W * 0.2, H * 0.15, 600, 350, 8, 18, 45, 0.15);
  drawCloudLayer(ctx, W * 0.5, H * 0.1, 700, 300, 6, 14, 38, 0.12);

  // connecting mid-cloud between bottom banks
  drawCloudLayer(ctx, W * 0.5, H * 0.85, 700, 350, 10, 25, 60, 0.35);

  // stars — just a handful of bright ones, no scatter
  drawStars(ctx, 600);

  return canvas;
}

export function createSpaceBgTexture(): THREE.CanvasTexture {
  const canvas = createSpaceBgCanvas();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
