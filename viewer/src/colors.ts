import * as THREE from 'three';

const CATEGORY_COLORS: Record<string, number> = {
  feature_extraction: 0x00ffff,  // cyan
  normalization:      0xff6699,  // pink
  activation:         0x00ff41,  // electric green
  pooling:            0xff6600,  // orange
  linear:             0xffff00,  // yellow
  attention:          0xff0066,  // hot pink
  recurrent:          0x9966ff,  // purple
  residual:           0x00ccff,  // sky blue
  regularization:     0x66ffcc,  // mint
  view:               0xcc99ff,  // lavender
  upsampling:         0xff9933,  // amber
  unknown:            0x888888,  // grey
};

export function getCategoryColor(category: string): number {
  return CATEGORY_COLORS[category] ?? CATEGORY_COLORS.unknown;
}

export function getCategoryColorObj(category: string): THREE.Color {
  return new THREE.Color(getCategoryColor(category));
}

export function getAllCategories(): [string, number][] {
  return Object.entries(CATEGORY_COLORS);
}
