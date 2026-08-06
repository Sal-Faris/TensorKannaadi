export type ComponentKind =
  | "embedding"
  | "normalization"
  | "residual"
  | "attention"
  | "head"
  | "mlp"
  | "unembedding";

export type ComponentNode = {
  id: string;
  label: string;
  kind: ComponentKind;
  layer?: number;
  head?: number;
  activationPoints: string[];
  metadata: Record<string, unknown>;
};

export type LayerNode = {
  id: string;
  index: number;
  label: string;
  residualPre: ComponentNode;
  norm1: ComponentNode;
  attention: ComponentNode;
  heads: ComponentNode[];
  residualMid: ComponentNode;
  norm2: ComponentNode;
  mlp: ComponentNode;
  residualPost: ComponentNode;
};

export type ArchitectureGraph = {
  modelId: string;
  displayName: string;
  family: string;
  nLayers: number;
  nHeads: number;
  nKeyValueHeads: number;
  dModel: number;
  dHead: number;
  dMlp: number | null;
  vocabularySize: number;
  normType: string;
  normalizationPosition: "pre" | "post";
  blockTopology: "serial" | "parallel";
  positionalMechanism: string;
  embedding: ComponentNode;
  layers: LayerNode[];
  finalNorm: ComponentNode;
  unembedding: ComponentNode;
};

export type RuntimeStatus = {
  backend: "ready" | "starting" | "error";
  runtime: "desktop" | "browser";
  torchAvailable: boolean;
  transformerLensAvailable: boolean;
  cudaAvailable: boolean;
  device: string;
  loadedModelId: string | null;
  loadedModelName: string | null;
  loadState: "idle" | "loading" | "loaded" | "error";
  loadError: string | null;
};

export type ModelCatalogEntry = {
  id: string;
  displayName: string;
  repository: string;
  architectureFamily: string;
  parameterCount: string;
};

export type SidecarInfo = {
  baseUrl: string;
  token: string;
  state: string;
};
