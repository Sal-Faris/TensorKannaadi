export type ComponentKind =
  | "embedding"
  | "normalization"
  | "residual"
  | "attention"
  | "head"
  | "mlp"
  | "unembedding"
  | "projection"
  | "activation"
  | "operation";

export type ComponentNode = {
  id: string;
  label: string;
  kind: ComponentKind;
  layer?: number;
  head?: number;
  activationPoints: string[];
  metadata: Record<string, unknown>;
  children: ComponentNode[];
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
  positionalEmbedding: ComponentNode | null;
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
  dtype: string | null;
  loadedModelId: string | null;
  loadedModelName: string | null;
  loadState: "idle" | "loading" | "loaded" | "error";
  loadError: string | null;
  loadStage: string;
  loadMessage: string;
  loadElapsedSeconds: number;
  runCount: number;
  cacheBytes: number;
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

export type TokenRecord = {
  position: number;
  tokenId: number;
  text: string;
  display: string;
  nextToken: string | null;
  nextTokenProbability: number | null;
};

export type Prediction = {
  tokenId: number;
  text: string;
  display: string;
  logit: number;
  probability: number;
};

export type InterventionSpec = {
  kind: "zero_ablation" | "mean_ablation" | "activation_patch";
  componentIds: string[];
  tokenScope: "all" | "positions";
  positions: number[];
  sourceRunId: string | null;
  destinationRunId: string | null;
  patchMappings: { sourcePosition: number; destinationPosition: number }[];
  baseline: string | null;
};

export type RunRecord = {
  id: string;
  kind: "clean" | "corrupted" | "intervened" | "patched";
  label: string;
  status: "complete";
  modelId: string;
  modelRevision: string | null;
  prompt: string;
  tokens: TokenRecord[];
  topPredictions: Prediction[];
  requestedActivations: string[];
  interventions: InterventionSpec[];
  parentRunId: string | null;
  device: string;
  dtype: string;
  seed: number;
  durationMs: number;
  cacheBytes: number;
  createdAt: string;
  provenance: Record<string, unknown>;
};

export type AttentionResult = {
  runId: string;
  componentId: string;
  layer: number;
  head: number;
  patternHook: string;
  scoreHook: string;
  queryTokens: string[];
  keyTokens: string[];
  pattern: number[][];
  scores: number[][];
  resultNorms: number[];
  qNorms: number[];
  kNorms: number[];
  axes: ["query_token", "key_token"];
};

export type ActivationSeries = {
  runId: string;
  componentId: string;
  hookName: string;
  measure: string;
  tokenLabels: string[];
  values: number[];
  axes: ["position"];
};

export type ResidualPoint = {
  layer: number;
  stage: "pre" | "mid" | "post";
  norm: number;
  targetLogit: number;
  entropy: number;
  topPredictions: Prediction[];
};

export type ResidualStreamResult = {
  runId: string;
  position: number;
  targetTokenId: number;
  targetToken: string;
  points: ResidualPoint[];
  method: "logit_lens";
};

export type TokenComparison = {
  tokenId: number;
  text: string;
  display: string;
  baselineLogit: number;
  intervenedLogit: number;
  deltaLogit: number;
  baselineProbability: number;
  intervenedProbability: number;
  deltaProbability: number;
};

export type RunComparison = {
  baselineRunId: string;
  intervenedRunId: string;
  baselineTopToken: string;
  baselineTopTokenDelta: number;
  klDivergence: number;
  tokens: TokenComparison[];
};

export type AlignmentPair = {
  sourcePosition: number | null;
  destinationPosition: number | null;
  sourceToken: string | null;
  destinationToken: string | null;
  status: "exact" | "substitution" | "source_gap" | "destination_gap";
};

export type TokenAlignment = {
  sourceRunId: string;
  destinationRunId: string;
  strategy: "minimum_edit_distance";
  pairs: AlignmentPair[];
  exactMatches: number;
  sourceLength: number;
  destinationLength: number;
};

export type ContrastResult = {
  id: string;
  cleanRun: RunRecord;
  corruptedRun: RunRecord;
  alignment: TokenAlignment;
};

export type MetricSpec = {
  targetToken: string;
  distractorToken?: string;
  position: number;
};

export type MetricResult = {
  runId: string;
  metric: "target_logit" | "logit_difference";
  position: number;
  targetTokenId: number;
  targetToken: string;
  distractorTokenId: number | null;
  distractorToken: string | null;
  value: number;
};

export type CausalEffect = {
  baseline: MetricResult;
  intervened: MetricResult;
  delta: number;
};

export type InterventionResult = {
  run: RunRecord;
  effect: CausalEffect | null;
};

export type DatasetAblationRow = {
  baselineRunId: string;
  intervenedRunId: string | null;
  intervenedRun: RunRecord | null;
  label: string;
  prompt: string;
  status: "complete" | "error";
  baselineValue: number | null;
  intervenedValue: number | null;
  delta: number | null;
  error: string | null;
};

export type DatasetAblationResult = {
  id: string;
  seriesId?: string;
  seriesName?: string;
  recipeId?: string;
  recipeName?: string;
  kind: "zero_ablation" | "mean_ablation";
  componentIds: string[];
  tokenScope: "all" | "positions";
  positions: number[];
  metric: MetricSpec;
  rows: DatasetAblationRow[];
  summary: {
    requestedCount: number;
    completedCount: number;
    failedCount: number;
    meanDelta: number | null;
    medianDelta: number | null;
    standardDeviation: number | null;
    minimumDelta: number | null;
    maximumDelta: number | null;
    meanAbsoluteDelta: number | null;
    directionConsistency: number | null;
  };
  durationMs: number;
  caveat: string;
};

export type HeadEffect = {
  componentId: string;
  layer: number;
  head: number;
  metricValue: number;
  delta: number;
};

export type HeadSweepResult = {
  runId: string;
  kind: "zero_ablation" | "mean_ablation";
  metric: MetricResult;
  effects: HeadEffect[];
  minimum: number;
  maximum: number;
  durationMs: number;
};

export type MlpEffect = {
  componentId: string;
  layer: number;
  metricValue: number;
  delta: number;
};

export type MlpSweepResult = {
  runId: string;
  kind: "zero_ablation" | "mean_ablation";
  metric: MetricResult;
  effects: MlpEffect[];
  minimum: number;
  maximum: number;
  durationMs: number;
};

export type AttributionEffect = {
  componentId: string;
  label: string;
  kind: "embedding" | "head" | "mlp" | "remainder";
  layer: number | null;
  head: number | null;
  value: number;
  fraction: number | null;
};

export type AttributionResult = {
  runId: string;
  metric: MetricResult;
  method: "direct_logit_attribution_fixed_final_norm";
  effects: AttributionEffect[];
  componentSum: number;
  remainder: number;
  minimum: number;
  maximum: number;
  durationMs: number;
  caveat: string;
};

export type ComponentGroup = {
  id: string;
  name: string;
  componentIds: string[];
};

export type PromptEntry = {
  id: string;
  name: string;
  text: string;
  createdAt: string;
};

export type PromptCollection = {
  id: string;
  name: string;
  promptIds: string[];
  createdAt: string;
};

export type ExperimentSeries = {
  id: string;
  collectionId: string;
  collectionName: string;
  runIds: string[];
  createdAt: string;
};

export type InterventionRecipe = {
  id: string;
  name: string;
  kind: "zero_ablation" | "mean_ablation" | "activation_patch";
  componentIds: string[];
  tokenScope: "all" | "positions";
  positions: number[];
  targetToken: string;
  distractorToken: string;
  createdAt: string;
};

export type WorkspaceResults = {
  runs: RunRecord[];
  contrast: ContrastResult | null;
  effects: Record<string, CausalEffect>;
  headSweep: HeadSweepResult | null;
  mlpSweep: MlpSweepResult | null;
  attribution: AttributionResult | null;
  residual: ResidualStreamResult | null;
  datasetAblations?: DatasetAblationResult[];
};

export type WorkspaceSnapshot = {
  format: "kannaadi-workspace";
  version: 1 | 2 | 3;
  name: string;
  savedAt: string;
  modelId: string;
  prompt: string;
  corruptedPrompt: string;
  targetToken: string;
  distractorToken: string;
  selection: string[];
  groups: ComponentGroup[];
  expandedLayers: number[];
  expandedHeads: string[];
  layout: { leftWidth: number; rightWidth: number; bottomHeight: number };
  theme?: "light" | "dark";
  registeredModels?: ModelCatalogEntry[];
  prompts?: PromptEntry[];
  collections?: PromptCollection[];
  experimentSeries?: ExperimentSeries[];
  recipes?: InterventionRecipe[];
  scratchpads?: Record<string, string>;
  results?: WorkspaceResults;
  viewport?: { zoom: number; pan: { x: number; y: number } };
};
