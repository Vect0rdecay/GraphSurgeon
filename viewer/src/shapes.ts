import * as THREE from 'three';

const OP_SHAPES: Record<string, string> = {
  Conv:              'box',
  ConvTranspose:     'box',
  MaxPool:           'flatbox',
  AveragePool:       'flatbox',
  GlobalAveragePool: 'flatbox',
  Relu:              'sphere',
  LeakyRelu:         'sphere',
  Sigmoid:           'sphere',
  Tanh:              'sphere',
  Softmax:           'sphere',
  PRelu:             'sphere',
  Elu:               'sphere',
  Selu:              'sphere',
  Clip:              'sphere',
  MatMul:            'cylinder',
  Gemm:              'cylinder',
  BatchNormalization:'torus',
  InstanceNormalization: 'torus',
  LayerNormalization:'torus',
  GroupNormalization: 'torus',
  Flatten:           'octahedron',
  Reshape:           'octahedron',
  Squeeze:           'octahedron',
  Unsqueeze:         'octahedron',
  Transpose:         'octahedron',
  Concat:            'dodecahedron',
  Add:               'dodecahedron',
  Mul:               'dodecahedron',
};

const geometryCache = new Map<string, THREE.BufferGeometry>();

function getGeometry(shape: string): THREE.BufferGeometry {
  if (geometryCache.has(shape)) return geometryCache.get(shape)!;

  let geom: THREE.BufferGeometry;
  switch (shape) {
    case 'box':
      geom = new THREE.BoxGeometry(1.0, 1.0, 1.0);
      break;
    case 'flatbox':
      geom = new THREE.BoxGeometry(1.2, 0.4, 1.2);
      break;
    case 'sphere':
      geom = new THREE.SphereGeometry(0.5, 16, 12);
      break;
    case 'cylinder':
      geom = new THREE.CylinderGeometry(0.5, 0.5, 1.0, 16);
      break;
    case 'torus':
      geom = new THREE.TorusGeometry(0.4, 0.15, 12, 24);
      break;
    case 'octahedron':
      geom = new THREE.OctahedronGeometry(0.5);
      break;
    case 'dodecahedron':
      geom = new THREE.DodecahedronGeometry(0.5);
      break;
    default:
      geom = new THREE.IcosahedronGeometry(0.5);
  }
  geometryCache.set(shape, geom);
  return geom;
}

export function getShapeForOp(opType: string): string {
  return OP_SHAPES[opType] ?? 'icosahedron';
}

export function getGeometryForOp(opType: string): THREE.BufferGeometry {
  return getGeometry(getShapeForOp(opType));
}
