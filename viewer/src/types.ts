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
}

export interface SceneChain {
  id: string;
  node_ids: string[];
  gadget_ids: string[];
}

export interface SceneGraph {
  schema_version: string;
  model: SceneModelInfo;
  nodes: SceneNode[];
  edges: SceneEdge[];
  motifs: SceneMotif[];
  chains: SceneChain[];
}
