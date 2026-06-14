export interface SceneInput {
  name: string;
  shape: number[];
  dtype: string;
}

export interface SceneOutput {
  name: string;
  shape: number[];
  dtype: string;
}

export interface SceneModelInfo {
  name: string;
  format: string;
  opset: number;
  total_nodes: number;
  max_depth: number;
  inputs: SceneInput[];
  outputs: SceneOutput[];
  flow_description: string;
  source_file?: string;
}

export interface PaperRef {
  slug: string;
  title: string;
}

export interface SceneNode {
  id: string;
  op_type: string;
  category: string;
  depth: number;
  position: string;
  exec_index: number;
  inputs: string[];
  outputs: string[];
  attributes: Record<string, unknown>;
  param_count: number;
  motif_ids: string[];
  gadget_ids: string[];
  gradient_sensitivity: number;
  lipschitz_estimate: number;
  perturbation_amplification: number;
  shadowlogic_capacity: number;
  extraction_leakage: number;
}

export interface SceneEdge {
  source: string;
  target: string;
  tensor: string;
  shape: number[];
}

export interface SceneMotif {
  id: string;
  title: string;
  node_ids: string[];
  description: string;
  catalog_ref: string;
  attacks_enabled?: string[];
  structural_significance?: string;
  confidence?: string;
  category?: string;
  research_basis?: PaperRef[];
  detection_logic?: string;
}

export interface SceneChain {
  id: string;
  node_ids: string[];
  gadget_ids: string[];
  title?: string;
  description?: string;
  structural_significance?: string;
  research_basis?: PaperRef[];
}

export interface SceneShadowLogicPoint {
  node_id: string;
  location: string;
  description: string;
  injection_complexity: string;
  detection_difficulty: string;
}

export interface SceneShadowLogic {
  structural_exposure: number;
  exposure_tier: string;
  conditional_ops: string[];
  injection_points: SceneShadowLogicPoint[];
}

export interface SceneStructuralPatterns {
  gradient_bottlenecks: string[];
  feature_fusion_points: string[];
  amplification_layers: string[];
  recommended_defense_points: string[];
  max_fan_in: number;
  max_fan_out: number;
  longest_linear_chain: number;
  structural_score: number;
}

export interface SceneGraph {
  schema_version: string;
  model: SceneModelInfo;
  nodes: SceneNode[];
  edges: SceneEdge[];
  motifs: SceneMotif[];
  chains: SceneChain[];
  shadowlogic?: SceneShadowLogic;
  structural_patterns?: SceneStructuralPatterns;
}
