import {
  Activity,
  BoxSelect,
  Braces,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Code2,
  Cpu,
  Crosshair,
  Database,
  Download,
  EllipsisVertical,
  Eye,
  FileJson,
  FlaskConical,
  FolderOpen,
  Hand,
  History,
  Info,
  Layers3,
  Maximize2,
  MousePointer2,
  Play,
  Plus,
  Redo2,
  RotateCcw,
  Save,
  Scan,
  SlidersHorizontal,
  Sparkles,
  SquareDashedMousePointer,
  Undo2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { KannaadiApi } from "./api";
import type {
  ActivationSeries,
  ArchitectureGraph,
  AttributionResult,
  AttentionResult,
  CausalEffect,
  ComponentNode,
  ComponentGroup,
  ContrastResult,
  HeadSweepResult,
  LayerNode,
  MetricSpec,
  ModelCatalogEntry,
  MlpSweepResult,
  ResidualStreamResult,
  RunComparison,
  RunRecord,
  RuntimeStatus,
  WorkspaceSnapshot,
} from "./types";
import { listWorkspaces, loadWorkspace, saveWorkspace } from "./workspaces";

type Tool = "pointer" | "box" | "pan";
type PanelTab = "Tokens" | "Alignment" | "Attention" | "QK / OV" | "Logits" | "Activations" | "Residual Stream" | "Causal Sweep" | "Attribution" | "Code";

const DEFAULT_MODEL: ModelCatalogEntry = {
  id: "gpt2-small",
  displayName: "GPT-2 Small",
  repository: "gpt2-small",
  architectureFamily: "decoder-only",
  parameterCount: "124M",
};
const api = new KannaadiApi();

export function App() {
  const [architecture, setArchitecture] = useState<ArchitectureGraph | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [models, setModels] = useState<ModelCatalogEntry[]>([DEFAULT_MODEL]);
  const [modelId, setModelId] = useState("gpt2-small");
  const [prompt, setPrompt] = usePersistentText("kannaadi.prompt", "The capital of France is");
  const [corruptedPrompt, setCorruptedPrompt] = usePersistentText("kannaadi.corruptedPrompt", "The capital of Germany is");
  const [targetToken, setTargetToken] = usePersistentText("kannaadi.targetToken", " Paris");
  const [distractorToken, setDistractorToken] = usePersistentText("kannaadi.distractorToken", " Berlin");
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [contrast, setContrast] = useState<ContrastResult | null>(null);
  const [effects, setEffects] = useState<Record<string, CausalEffect>>({});
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [modelDialog, setModelDialog] = useState(false);
  const [experimentDialog, setExperimentDialog] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const activeRun = runs.find((run) => run.id === activeRunId) ?? null;

  const updateRuntimeForRun = (run: RunRecord) => {
    setRuntime((current) => current ? {
      ...current,
      runCount: current.runCount + 1,
      cacheBytes: current.cacheBytes + run.cacheBytes,
    } : current);
  };

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await api.connect();
        const [nextModels, nextStatus] = await Promise.all([api.models(), api.status()]);
        if (cancelled) return;
        setModels(nextModels);
        setRuntime(nextStatus);
        if (nextStatus.loadedModelId) {
          const [graph, existingRuns] = await Promise.all([
            api.architecture(nextStatus.loadedModelId),
            api.runs(),
          ]);
          if (cancelled) return;
          setModelId(nextStatus.loadedModelId);
          setArchitecture(graph);
          setRuns(existingRuns);
          setActiveRunId(existingRuns[0]?.id ?? null);
        }
      } catch (error) {
        if (!cancelled) setConnectionError(error instanceof Error ? error.message : "Backend unavailable");
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const loadModel = async () => {
    setLoading(true);
    setConnectionError(null);
    setRuntime((current) => current ? { ...current, loadState: "loading", loadError: null } : current);
    const poll = window.setInterval(() => {
      void api.status().then(setRuntime).catch(() => undefined);
    }, 450);
    try {
      const graph = await api.loadModel(modelId, runtime?.cudaAvailable ? "cuda" : "cpu");
      setArchitecture(graph);
      setRuns([]);
      setActiveRunId(null);
      setContrast(null);
      setEffects({});
      setRuntime(await api.status());
    } catch (error) {
      const message = error instanceof Error ? error.message : "Model loading failed";
      setConnectionError(message);
      setRuntime((current) => current ? { ...current, loadState: "error", loadError: message } : current);
    } finally {
      window.clearInterval(poll);
      setLoading(false);
    }
  };

  const retryDesktopService = async () => {
    setLoading(true);
    setConnectionError(null);
    try {
      await api.restartDesktop();
      const [nextModels, nextStatus] = await Promise.all([api.models(), api.status()]);
      setModels(nextModels);
      setRuntime(nextStatus);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : "Desktop service could not be restarted");
    } finally {
      setLoading(false);
    }
  };

  const runPrompt = async () => {
    if (!architecture) {
      await loadModel();
      return;
    }
    if (!prompt.trim()) return;
    setRunning(true);
    setConnectionError(null);
    try {
      const run = await api.run(prompt);
      setRuns((current) => [run, ...current]);
      setActiveRunId(run.id);
      updateRuntimeForRun(run);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : "Prompt execution failed");
    } finally {
      setRunning(false);
    }
  };

  const metricSpec = (): MetricSpec | undefined => targetToken ? {
    targetToken,
    distractorToken: distractorToken || undefined,
    position: -1,
  } : undefined;

  const runContrast = async () => {
    if (!architecture) {
      await loadModel();
      return;
    }
    if (!prompt.trim() || !corruptedPrompt.trim()) return;
    setRunning(true);
    setConnectionError(null);
    try {
      const result = await api.contrast(prompt, corruptedPrompt);
      setContrast(result);
      setRuns((current) => [result.corruptedRun, result.cleanRun, ...current]);
      setActiveRunId(result.corruptedRun.id);
      updateRuntimeForRun(result.cleanRun);
      updateRuntimeForRun(result.corruptedRun);
      setExperimentDialog(false);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : "Contrast execution failed");
    } finally {
      setRunning(false);
    }
  };

  const causalAblate = async (
    componentIds: string[],
    kind: "zero_ablation" | "mean_ablation",
    positions: number[] | null,
  ) => {
    const baselineId = activeRun?.kind === "intervened" || activeRun?.kind === "patched" ? activeRun.parentRunId : activeRun?.id;
    if (!baselineId) throw new Error("Run a prompt before adding an intervention");
    setRunning(true);
    try {
      const result = await api.ablate(baselineId, componentIds, kind, positions, metricSpec());
      setRuns((current) => [result.run, ...current]);
      setActiveRunId(result.run.id);
      if (result.effect) setEffects((current) => ({ ...current, [result.run.id]: result.effect! }));
      updateRuntimeForRun(result.run);
    } finally {
      setRunning(false);
    }
  };

  const patchActivations = async (componentIds: string[], positions: number[] | null) => {
    if (!contrast) throw new Error("Run a clean/corrupted contrast before activation patching");
    const mappings = contrast.alignment.pairs
      .filter((pair) => pair.sourcePosition !== null && pair.destinationPosition !== null)
      .filter((pair) => !positions || positions.includes(pair.destinationPosition!))
      .map((pair) => ({ sourcePosition: pair.sourcePosition!, destinationPosition: pair.destinationPosition! }));
    if (!mappings.length) throw new Error("No aligned token positions are available for this patch scope");
    setRunning(true);
    try {
      const result = await api.patch(
        contrast.corruptedRun.id,
        contrast.cleanRun.id,
        componentIds,
        mappings,
        metricSpec(),
      );
      setRuns((current) => [result.run, ...current]);
      setActiveRunId(result.run.id);
      if (result.effect) setEffects((current) => ({ ...current, [result.run.id]: result.effect! }));
      updateRuntimeForRun(result.run);
    } finally {
      setRunning(false);
    }
  };

  const chooseModel = (nextModelId: string) => {
    setModelId(nextModelId);
    if (nextModelId !== runtime?.loadedModelId) {
      setArchitecture(null);
      setRuns([]);
      setActiveRunId(null);
      setContrast(null);
      setEffects({});
    }
  };

  const registerModel = async (source: string, displayName?: string) => {
    const entry = await api.registerModel(source, displayName);
    setModels((current) => [...current.filter((model) => model.id !== entry.id), entry]);
    chooseModel(entry.id);
    setModelDialog(false);
  };

  useEffect(() => {
    const keyboardRun = (event: KeyboardEvent) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        if (!loading && !running) void (architecture ? runPrompt() : loadModel());
      }
    };
    window.addEventListener("keydown", keyboardRun);
    return () => window.removeEventListener("keydown", keyboardRun);
  });

  return (
    <main className="app-shell">
      <TopBar
        models={models}
        modelId={modelId}
        setModelId={chooseModel}
        prompt={prompt}
        setPrompt={setPrompt}
        architecture={architecture}
        runtime={runtime}
        activeRun={activeRun}
        loading={loading}
        running={running}
        onPrimaryAction={architecture ? runPrompt : loadModel}
        onAddModel={() => setModelDialog(true)}
        onExperiment={() => setExperimentDialog(true)}
      />
      <Workbench
        architecture={architecture}
        runtime={runtime}
        model={models.find((model) => model.id === modelId) || DEFAULT_MODEL}
        prompt={prompt}
        runs={runs}
        activeRun={activeRun}
        setActiveRunId={setActiveRunId}
        causalAblate={causalAblate}
        patchActivations={patchActivations}
        contrast={contrast}
        effects={effects}
        targetToken={targetToken}
        distractorToken={distractorToken}
        corruptedPrompt={corruptedPrompt}
        onRestoreWorkspace={(snapshot) => {
          setPrompt(snapshot.prompt);
          setCorruptedPrompt(snapshot.corruptedPrompt);
          setTargetToken(snapshot.targetToken);
          setDistractorToken(snapshot.distractorToken);
          chooseModel(snapshot.modelId);
        }}
        loading={loading}
        running={running}
        connectionError={connectionError}
        onRetryService={() => void retryDesktopService()}
      />
      {modelDialog && <ModelDialog onClose={() => setModelDialog(false)} onRegister={registerModel} />}
      {experimentDialog && <ExperimentDialog
        cleanPrompt={prompt}
        corruptedPrompt={corruptedPrompt}
        targetToken={targetToken}
        distractorToken={distractorToken}
        setCleanPrompt={setPrompt}
        setCorruptedPrompt={setCorruptedPrompt}
        setTargetToken={setTargetToken}
        setDistractorToken={setDistractorToken}
        running={running}
        loaded={Boolean(architecture)}
        onClose={() => setExperimentDialog(false)}
        onRun={() => void runContrast()}
      />}
    </main>
  );
}

function TopBar(props: {
  models: ModelCatalogEntry[];
  modelId: string;
  setModelId: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  architecture: ArchitectureGraph | null;
  runtime: RuntimeStatus | null;
  activeRun: RunRecord | null;
  loading: boolean;
  running: boolean;
  onPrimaryAction: () => void;
  onAddModel: () => void;
  onExperiment: () => void;
}) {
  const { models, modelId, setModelId, prompt, setPrompt, architecture, runtime, activeRun, loading, running, onPrimaryAction, onAddModel, onExperiment } = props;
  const busy = loading || running || runtime?.loadState === "loading";
  return (
    <header className="top-bar">
      <div className="brand" aria-label="Kannaadi"><KannaadiLogo /><span>Kannaadi</span></div>
      <div className="model-select control-field">
        <Cpu size={15} />
        <select aria-label="Model" value={modelId} onChange={(event) => setModelId(event.target.value)}>
          {models.map((model) => <option key={model.id} value={model.id}>{model.displayName}</option>)}
        </select>
        <button className="add-model" aria-label="Add TransformerLens model" title="Add TransformerLens model" onClick={onAddModel}><Plus /></button>
      </div>
      <label className="prompt-field control-field">
        <input
          aria-label="Prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          spellCheck={false}
        />
        <kbd>Ctrl ↵</kbd>
      </label>
      <button
        className="run-button"
        onClick={onPrimaryAction}
        disabled={busy || (Boolean(architecture) && !prompt.trim())}
        title={architecture ? "Run this prompt through the loaded model" : "Load the selected model"}
      >
        {busy ? <span className="spinner" /> : architecture ? <Play size={14} fill="currentColor" /> : <Download size={14} />}
        {loading ? "Loading model" : running ? "Running" : architecture ? "Run" : "Load model"}
      </button>
      <button className="experiment-button" onClick={onExperiment} disabled={busy}><FlaskConical />Contrast</button>
      <div className="run-mode"><span>Active Run</span><div className="run-state"><i className={`status-dot ${activeRun?.kind ?? "clean"}`} />{activeRun?.label ?? "No run yet"}</div></div>
    </header>
  );
}

function Workbench(props: {
  architecture: ArchitectureGraph | null;
  runtime: RuntimeStatus | null;
  model: ModelCatalogEntry;
  prompt: string;
  runs: RunRecord[];
  activeRun: RunRecord | null;
  setActiveRunId: (runId: string) => void;
  causalAblate: (componentIds: string[], kind: "zero_ablation" | "mean_ablation", positions: number[] | null) => Promise<void>;
  patchActivations: (componentIds: string[], positions: number[] | null) => Promise<void>;
  contrast: ContrastResult | null;
  effects: Record<string, CausalEffect>;
  targetToken: string;
  distractorToken: string;
  corruptedPrompt: string;
  onRestoreWorkspace: (snapshot: WorkspaceSnapshot) => void;
  loading: boolean;
  running: boolean;
  connectionError: string | null;
  onRetryService: () => void;
}) {
  const { architecture, runtime, model, prompt, runs, activeRun, setActiveRunId, causalAblate, patchActivations, contrast, effects, targetToken, distractorToken, corruptedPrompt, onRestoreWorkspace, loading, running, connectionError, onRetryService } = props;
  const [selection, setSelection] = useState<string[]>([]);
  const [past, setPast] = useState<string[][]>([]);
  const [future, setFuture] = useState<string[][]>([]);
  const [expanded, setExpanded] = useState<Set<number>>(new Set([5]));
  const [expandedHeads, setExpandedHeads] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<PanelTab>("Attention");
  const [tool, setTool] = useState<Tool>("pointer");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [notice, setNotice] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; node: ComponentNode } | null>(null);
  const [attention, setAttention] = useState<AttentionResult | null>(null);
  const [activation, setActivation] = useState<ActivationSeries | null>(null);
  const [residual, setResidual] = useState<ResidualStreamResult | null>(null);
  const [comparison, setComparison] = useState<RunComparison | null>(null);
  const [headSweep, setHeadSweep] = useState<HeadSweepResult | null>(null);
  const [mlpSweep, setMlpSweep] = useState<MlpSweepResult | null>(null);
  const [sweepKind, setSweepKind] = useState<"heads" | "mlps">("heads");
  const [attribution, setAttribution] = useState<AttributionResult | null>(null);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [attributionLoading, setAttributionLoading] = useState(false);
  const [scopePosition, setScopePosition] = useState<number | null>(null);
  const [groups, setGroups] = usePersistentJson<ComponentGroup[]>("kannaadi.groups", []);
  const [groupDialog, setGroupDialog] = useState(false);
  const [workspaceDialog, setWorkspaceDialog] = useState(false);
  const [selectionDialog, setSelectionDialog] = useState(false);
  const [workspaceNames, setWorkspaceNames] = useState<string[]>([]);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [leftWidth, setLeftWidth] = usePersistentNumber("kannaadi.panel.left", 220);
  const [rightWidth, setRightWidth] = usePersistentNumber("kannaadi.panel.right", 320);
  const [bottomHeight, setBottomHeight] = usePersistentNumber("kannaadi.panel.bottom", 270);
  const previousRun = useRef<string | null>(null);

  const nodeIndex = useMemo(() => indexArchitecture(architecture), [architecture]);
  const selectedNodes = selection.map((id) => nodeIndex.get(id)).filter(Boolean) as ComponentNode[];
  const primary = selectedNodes.at(-1) || null;
  const activeEffect = activeRun ? effects[activeRun.id] ?? null : null;
  const scopedPositions = scopePosition === null ? null : [scopePosition];

  useEffect(() => {
    void listWorkspaces().then(setWorkspaceNames).catch(() => setWorkspaceNames([]));
  }, []);

  const commitSelection = useCallback((next: string[]) => {
    setSelection((current) => {
      if (current.join("|") === next.join("|")) return current;
      setPast((history) => [...history.slice(-29), current]);
      setFuture([]);
      return next;
    });
  }, []);

  const selectNode = useCallback((node: ComponentNode, additive = false) => {
    if (additive) commitSelection(selection.includes(node.id) ? selection.filter((id) => id !== node.id) : [...selection, node.id]);
    else commitSelection([node.id]);
    if (node.kind === "head" && activeRun) setActiveTab("Attention");
  }, [activeRun, commitSelection, selection]);

  const undo = () => {
    const previous = past.at(-1);
    if (!previous) return;
    setPast((history) => history.slice(0, -1));
    setFuture((history) => [selection, ...history]);
    setSelection(previous);
  };
  const redo = () => {
    const next = future[0];
    if (!next) return;
    setFuture((history) => history.slice(1));
    setPast((history) => [...history, selection]);
    setSelection(next);
  };

  useEffect(() => {
    if (!architecture) return;
    const preferredLayer = architecture.layers.find((layer) => layer.index === 5) || architecture.layers[0];
    const preferredHead = preferredLayer.heads[Math.min(3, preferredLayer.heads.length - 1)];
    setExpanded(new Set([preferredLayer.index]));
    setSelection([preferredHead.id]);
    setPast([]);
    setFuture([]);
  }, [architecture]);

  useEffect(() => {
    if (!activeRun || previousRun.current === activeRun.id) return;
    previousRun.current = activeRun.id;
    setActiveTab(activeRun.kind === "intervened" ? "Logits" : "Tokens");
  }, [activeRun]);

  useEffect(() => {
    let cancelled = false;
    setEvidenceError(null);
    setEvidenceLoading(false);
    if (!activeRun) {
      setAttention(null); setActivation(null); setResidual(null); setComparison(null);
      return;
    }
    if (activeTab === "Attention" || activeTab === "QK / OV") setAttention(null);
    if (activeTab === "Activations") setActivation(null);
    if (activeTab === "Residual Stream") setResidual(null);
    if (activeTab === "Logits") setComparison(null);
    const load = async () => {
      setEvidenceLoading(true);
      try {
        if ((activeTab === "Attention" || activeTab === "QK / OV") && primary?.head != null && primary.layer != null) {
          const result = await api.attention(activeRun.id, primary.layer, primary.head);
          if (!cancelled) setAttention(result);
        } else if (activeTab === "Activations" && primary) {
          const result = await api.activation(activeRun.id, primary.id);
          if (!cancelled) setActivation(result);
        } else if (activeTab === "Residual Stream") {
          const result = await api.residualStream(activeRun.id);
          if (!cancelled) setResidual(result);
        } else if (activeTab === "Logits" && activeRun.kind === "intervened" && activeRun.parentRunId) {
          const result = await api.compare(activeRun.parentRunId, activeRun.id);
          if (!cancelled) setComparison(result);
        }
      } catch (error) {
        if (!cancelled) setEvidenceError(error instanceof Error ? error.message : "Evidence could not be loaded");
      } finally {
        if (!cancelled) setEvidenceLoading(false);
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [activeRun, activeTab, primary]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const modifier = event.ctrlKey || event.metaKey;
      if (event.key === "Escape") commitSelection([]);
      if (modifier && event.key.toLowerCase() === "a" && architecture && !isTextInput(event.target)) {
        event.preventDefault();
        commitSelection(architecture.layers.flatMap((layer) => layer.heads.map((head) => head.id)));
      }
      if (modifier && event.key.toLowerCase() === "z" && !isTextInput(event.target)) {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
      if (modifier && event.key.toLowerCase() === "y" && !isTextInput(event.target)) {
        event.preventDefault(); redo();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  });

  const selectedIntervenableIds = () => selectedNodes
    .filter((node) => node.kind === "head" || node.kind === "mlp")
    .map((node) => node.id);

  const ablateSelected = async (kind: "zero_ablation" | "mean_ablation") => {
    const components = selectedIntervenableIds();
    if (!components.length) {
      setNotice("Select one or more attention heads or MLPs to ablate.");
      return;
    }
    try {
      await causalAblate(components, kind, scopedPositions);
      setActiveTab("Logits");
      const baseline = kind === "zero_ablation" ? "zero" : "within-prompt mean";
      const scope = scopedPositions ? `token ${scopedPositions[0]}` : "all tokens";
      setNotice(`Created an exact ${baseline} ablation at ${scope} for ${components.length === 1 ? displayName(selectedNodes.find((node) => node.id === components[0])!) : `${components.length} components`}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Zero ablation failed");
    }
  };

  const patchSelected = async () => {
    const components = selectedIntervenableIds();
    if (!components.length) return setNotice("Select one or more attention heads or MLPs to patch.");
    try {
      await patchActivations(components, scopedPositions);
      setActiveTab("Logits");
      setNotice(`Patched ${components.length === 1 ? shortComponentId(components[0]) : `${components.length} components`} from the clean source into the corrupted destination.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Activation patching failed");
    }
  };

  const sweepHeads = async (kind: "zero_ablation" | "mean_ablation" = "zero_ablation") => {
    const baselineId = activeRun?.kind === "intervened" || activeRun?.kind === "patched" ? activeRun.parentRunId : activeRun?.id;
    if (!baselineId) return setNotice("Run a prompt before starting a causal sweep.");
    if (!targetToken) return setNotice("Set a target token in Contrast setup before starting a sweep.");
    setSweepLoading(true); setEvidenceError(null); setActiveTab("Causal Sweep");
    try {
      const result = await api.headSweep(baselineId, kind, scopedPositions, {
        targetToken,
        distractorToken: distractorToken || undefined,
        position: -1,
      });
      setHeadSweep(result);
      setSweepKind("heads");
      setNotice(`Completed ${architecture?.nLayers ?? 0} × ${architecture?.nHeads ?? 0} exact head interventions in ${(result.durationMs / 1000).toFixed(1)} s.`);
    } catch (error) {
      setEvidenceError(error instanceof Error ? error.message : "Causal sweep failed");
    } finally {
      setSweepLoading(false);
    }
  };

  const sweepMlps = async (kind: "zero_ablation" | "mean_ablation" = "zero_ablation") => {
    const baselineId = activeRun?.kind === "intervened" || activeRun?.kind === "patched" ? activeRun.parentRunId : activeRun?.id;
    if (!baselineId) return setNotice("Run a prompt before starting a causal sweep.");
    if (!targetToken) return setNotice("Set a target token in Contrast setup before starting a sweep.");
    setSweepLoading(true); setEvidenceError(null); setActiveTab("Causal Sweep"); setSweepKind("mlps");
    try {
      const result = await api.mlpSweep(baselineId, kind, scopedPositions, {
        targetToken,
        distractorToken: distractorToken || undefined,
        position: -1,
      });
      setMlpSweep(result);
      setNotice(`Completed ${result.effects.length} exact MLP interventions in ${(result.durationMs / 1000).toFixed(1)} s.`);
    } catch (error) {
      setEvidenceError(error instanceof Error ? error.message : "MLP sweep failed");
    } finally {
      setSweepLoading(false);
    }
  };

  const computeAttribution = async () => {
    const baselineId = activeRun?.kind === "intervened" || activeRun?.kind === "patched" ? activeRun.parentRunId : activeRun?.id;
    if (!baselineId) return setNotice("Run a prompt before computing direct attribution.");
    if (!targetToken) return setNotice("Set a target token in Contrast setup before computing attribution.");
    setAttributionLoading(true); setEvidenceError(null); setActiveTab("Attribution");
    try {
      setAttribution(await api.directAttribution(baselineId, {
        targetToken,
        distractorToken: distractorToken || undefined,
        position: -1,
      }));
    } catch (error) {
      setEvidenceError(error instanceof Error ? error.message : "Direct attribution failed");
    } finally {
      setAttributionLoading(false);
    }
  };

  const saveGroup = (name: string) => {
    const ids = selectedIntervenableIds();
    if (!ids.length) return setNotice("Select one or more attention heads or MLPs before saving a group.");
    setGroups((current) => [...current.filter((group) => group.name !== name), { id: crypto.randomUUID(), name, componentIds: ids }]);
    setGroupDialog(false);
    setNotice(`Saved ${ids.length} component${ids.length === 1 ? "" : "s"} as “${name}”.`);
  };

  const snapshot = (name: string): WorkspaceSnapshot => ({
    format: "kannaadi-workspace",
    version: 1,
    name,
    savedAt: new Date().toISOString(),
    modelId: model.id,
    prompt,
    corruptedPrompt,
    targetToken,
    distractorToken,
    selection,
    groups,
    expandedLayers: [...expanded],
    expandedHeads: [...expandedHeads],
    layout: { leftWidth, rightWidth, bottomHeight },
  });

  const saveCurrentWorkspace = async (name: string) => {
    try {
      const savedName = await saveWorkspace(snapshot(name));
      setWorkspaceNames(await listWorkspaces());
      setWorkspaceDialog(false);
      setNotice(`Workspace “${savedName}” saved to application data.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Workspace could not be saved");
    }
  };

  const openWorkspace = async (name: string) => {
    try {
      const saved = await loadWorkspace(name);
      onRestoreWorkspace(saved);
      setSelection(saved.selection);
      setGroups(saved.groups);
      setExpanded(new Set(saved.expandedLayers));
      setExpandedHeads(new Set(saved.expandedHeads));
      setLeftWidth(saved.layout.leftWidth);
      setRightWidth(saved.layout.rightWidth);
      setBottomHeight(saved.layout.bottomHeight);
      setNotice(`Workspace “${saved.name}” restored. Run manifests are intentionally recomputed for the current model session.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Workspace could not be opened");
    }
  };

  const layoutStyle = {
    "--left-panel": `${leftWidth}px`,
    "--right-panel": `${rightWidth}px`,
    "--bottom-panel": `${bottomHeight}px`,
  } as CSSProperties;

  return (
    <>
      <div className="workspace-grid" style={layoutStyle} onClick={() => contextMenu && setContextMenu(null)}>
        <LeftSidebar selected={selectedNodes} architecture={architecture} runs={runs} activeRun={activeRun} onSelectRun={setActiveRunId} groups={groups} onSelectGroup={(group) => commitSelection(group.componentIds)} workspaceNames={workspaceNames} onOpenWorkspace={(name) => void openWorkspace(name)} onSaveWorkspace={() => setWorkspaceDialog(true)} onViewSelections={() => setSelectionDialog(true)} />
        <PanelResizer label="Resize left sidebar" direction="vertical" onDelta={(delta) => setLeftWidth(clamp(leftWidth + delta, 160, 420))} onReset={() => setLeftWidth(220)} />
        <section className="center-stage">
          <ArchitecturePanel
            architecture={architecture}
            error={connectionError}
            connected={runtime?.backend === "ready"}
            runtime={runtime}
            loading={loading}
            onRetry={onRetryService}
            selection={selection}
            expanded={expanded}
            setExpanded={setExpanded}
            expandedHeads={expandedHeads}
            setExpandedHeads={setExpandedHeads}
            selectNode={selectNode}
            contextNode={(node, x, y) => setContextMenu({ node, x, y })}
            tool={tool}
            setTool={setTool}
            zoom={zoom}
            setZoom={setZoom}
            pan={pan}
            setPan={setPan}
            commitSelection={commitSelection}
            headOverlay={headSweep}
            mlpOverlay={mlpSweep}
            undo={undo}
            redo={redo}
            canUndo={past.length > 0}
            canRedo={future.length > 0}
          />
          <PanelResizer label="Resize analysis panel" direction="horizontal" onDelta={(delta) => setBottomHeight(clamp(bottomHeight - delta, 150, 560))} onReset={() => setBottomHeight(270)} />
          <AnalysisPanel
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            architecture={architecture}
            selected={primary}
            model={model}
            prompt={prompt}
            run={activeRun}
            attention={attention}
            activation={activation}
            residual={residual}
            comparison={comparison}
            contrast={contrast}
            effect={activeEffect}
            headSweep={headSweep}
            mlpSweep={mlpSweep}
            sweepKind={sweepKind}
            setSweepKind={setSweepKind}
            attribution={attribution}
            sweepLoading={sweepLoading}
            attributionLoading={attributionLoading}
            onHeadSweep={(kind) => void sweepHeads(kind)}
            onMlpSweep={(kind) => void sweepMlps(kind)}
            onAttribution={() => void computeAttribution()}
            onExport={() => downloadResearchBundle({ architecture, model, run: activeRun, contrast, effects, headSweep, mlpSweep, attribution, selection, groups })}
            loading={evidenceLoading}
            error={evidenceError}
          />
        </section>
        <PanelResizer label="Resize inspector" direction="vertical" onDelta={(delta) => setRightWidth(clamp(rightWidth - delta, 240, 520))} onReset={() => setRightWidth(320)} />
        <Inspector
          architecture={architecture}
          selected={primary}
          selectedCount={selection.length}
          activeRun={activeRun}
          running={running}
          onInspect={() => setActiveTab(primary?.kind === "head" ? "Attention" : "Activations")}
          scopePosition={scopePosition}
          setScopePosition={setScopePosition}
          onZeroAblate={() => void ablateSelected("zero_ablation")}
          onMeanAblate={() => void ablateSelected("mean_ablation")}
          onPatch={() => void patchSelected()}
          onNeuronAblate={(index, kind) => {
            if (primary?.layer == null) return;
            void causalAblate([`blocks.${primary.layer}.mlp.neuron.${index}`], kind, scopedPositions)
              .then(() => { setActiveTab("Logits"); setNotice(`Ablated neuron ${index} in MLP L${primary.layer}.`); })
              .catch((error) => setNotice(error instanceof Error ? error.message : "Neuron ablation failed"));
          }}
          onNeuronPatch={(index) => {
            if (primary?.layer == null) return;
            void patchActivations([`blocks.${primary.layer}.mlp.neuron.${index}`], scopedPositions)
              .then(() => { setActiveTab("Logits"); setNotice(`Patched neuron ${index} in MLP L${primary.layer}.`); })
              .catch((error) => setNotice(error instanceof Error ? error.message : "Neuron patch failed"));
          }}
          onHeadSweep={() => void sweepHeads("zero_ablation")}
          onMlpSweep={() => void sweepMlps("zero_ablation")}
          onAttribution={() => void computeAttribution()}
          canPatch={Boolean(contrast)}
          onSaveGroup={() => setGroupDialog(true)}
          onReturnToClean={() => activeRun?.parentRunId && setActiveRunId(activeRun.parentRunId)}
          onAction={setNotice}
        />
      </div>
      <StatusBar architecture={architecture} runtime={runtime} activeRun={activeRun} />
      {contextMenu && <ContextMenu menu={contextMenu} run={activeRun} canPatch={Boolean(contrast)} onClose={() => setContextMenu(null)} onInspect={() => { selectNode(contextMenu.node); setActiveTab(contextMenu.node.kind === "head" ? "Attention" : "Activations"); setContextMenu(null); }} onExpand={() => { const node = contextMenu.node; if (node.layer != null) setExpanded(new Set([...expanded, node.layer])); if (node.kind === "head") setExpandedHeads(new Set([...expandedHeads, node.id])); selectNode(node); setContextMenu(null); }} onAblate={() => { const node = contextMenu.node; selectNode(node); setContextMenu(null); window.setTimeout(() => void causalAblate([node.id], "zero_ablation", scopedPositions), 0); }} onPatch={() => { const node = contextMenu.node; selectNode(node); setContextMenu(null); window.setTimeout(() => void patchActivations([node.id], scopedPositions), 0); }} onAction={setNotice} />}
      {notice && <div className="toast" role="status"><Info size={15} />{notice}<button aria-label="Dismiss" onClick={() => setNotice(null)}><X /></button></div>}
      {groupDialog && <NameDialog title="Save component group" description="Reuse this head selection for batch interventions and workspace snapshots." action="Save group" onClose={() => setGroupDialog(false)} onSubmit={saveGroup} />}
      {workspaceDialog && <NameDialog title="Save workspace" description="Saves prompts, metric, selections, groups, expanded components, and panel layout." action="Save workspace" onClose={() => setWorkspaceDialog(false)} onSubmit={(name) => void saveCurrentWorkspace(name)} />}
      {selectionDialog && <SelectionDialog nodes={selectedNodes} onRemove={(id) => commitSelection(selection.filter((item) => item !== id))} onClear={() => commitSelection([])} onClose={() => setSelectionDialog(false)} />}
    </>
  );
}

function LeftSidebar({ selected, architecture, runs, activeRun, onSelectRun, groups, onSelectGroup, workspaceNames, onOpenWorkspace, onSaveWorkspace, onViewSelections }: {
  selected: ComponentNode[];
  architecture: ArchitectureGraph | null;
  runs: RunRecord[];
  activeRun: RunRecord | null;
  onSelectRun: (runId: string) => void;
  groups: ComponentGroup[];
  onSelectGroup: (group: ComponentGroup) => void;
  workspaceNames: string[];
  onOpenWorkspace: (name: string) => void;
  onSaveWorkspace: () => void;
  onViewSelections: () => void;
}) {
  return (
    <aside className="left-sidebar panel">
      <div className="sidebar-scroll">
      <SidebarSection title="Prompts" icon={<FileJson />}><button className="side-item active">The capital of France is</button></SidebarSection>
      <SidebarSection title="Runs" icon={<History />}>
        {runs.map((run) => <button className={`side-item run-item ${activeRun?.id === run.id ? "active" : ""}`} key={run.id} onClick={() => onSelectRun(run.id)}><i className={`status-dot ${run.kind}`} /><span>{run.label}</span><small>{run.durationMs.toFixed(0)} ms</small></button>)}
        {!runs.length && <div className="empty-side"><i className="status-dot muted" />Run a prompt to begin</div>}
      </SidebarSection>
      <SidebarSection title="Interventions" icon={<Sparkles />}>
        {runs.filter((run) => run.kind === "intervened" || run.kind === "patched").map((run) => <button className="side-item" key={run.id} onClick={() => onSelectRun(run.id)}><span>{run.interventions[0]?.kind.replaceAll("_", " ")}</span><small>{run.interventions[0]?.componentIds.map(shortComponentId).join(", ")}</small></button>)}
        {!runs.some((run) => run.kind === "intervened" || run.kind === "patched") && <div className="empty-side">No interventions yet</div>}
      </SidebarSection>
      <SidebarSection title={`Selections (${selected.length})`} icon={<Layers3 />}>
        {selected.slice(-4).map((node) => <button className="side-item selection-item" key={node.id}><ComponentGlyph kind={node.kind} />{displayName(node)}</button>)}
        {selected.length > 4 && <button className="view-all" onClick={onViewSelections}>View all {selected.length}</button>}
        {!selected.length && <div className="empty-side">Select a component</div>}
      </SidebarSection>
      <SidebarSection title={`Groups (${groups.length})`} icon={<Layers3 />}>
        {groups.map((group) => <button className="side-item" key={group.id} onClick={() => onSelectGroup(group)}><span>{group.name}</span><small>{group.componentIds.length} components</small></button>)}
        {!groups.length && <div className="empty-side">Save a selection to reuse it</div>}
      </SidebarSection>
      <SidebarSection title="Workspaces" icon={<Database />}>
        {workspaceNames.map((name) => <button className="side-item" key={name} onClick={() => onOpenWorkspace(name)}><FolderOpen />{name}</button>)}
      </SidebarSection>
      </div>
      <footer className="sidebar-footer">
        <button className="save-workspace-button" onClick={onSaveWorkspace}><Save />Save current workspace</button>
        {architecture && <div className="schema-badge">{architecture.family} · model-derived</div>}
      </footer>
    </aside>
  );
}

function SidebarSection({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return <section className="sidebar-section"><header><span>{title}</span></header><div className="sidebar-section-icon">{icon}</div>{children}</section>;
}

function ArchitecturePanel(props: {
  architecture: ArchitectureGraph | null;
  error: string | null;
  connected: boolean;
  runtime: RuntimeStatus | null;
  loading: boolean;
  onRetry: () => void;
  selection: string[];
  expanded: Set<number>;
  setExpanded: (value: Set<number>) => void;
  expandedHeads: Set<string>;
  setExpandedHeads: (value: Set<string>) => void;
  selectNode: (node: ComponentNode, additive?: boolean) => void;
  contextNode: (node: ComponentNode, x: number, y: number) => void;
  tool: Tool;
  setTool: (tool: Tool) => void;
  zoom: number;
  setZoom: (value: number) => void;
  pan: { x: number; y: number };
  setPan: (value: { x: number; y: number }) => void;
  commitSelection: (ids: string[]) => void;
  headOverlay: HeadSweepResult | null;
  mlpOverlay: MlpSweepResult | null;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}) {
  const { architecture, error, connected, runtime, loading, onRetry, selection, expanded, setExpanded, expandedHeads, setExpandedHeads, selectNode, contextNode, tool, setTool, zoom, setZoom, pan, setPan, commitSelection, headOverlay, mlpOverlay } = props;
  const viewport = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [box, setBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  useEffect(() => { zoomRef.current = zoom; panRef.current = pan; }, [pan, zoom]);

  const zoomAt = useCallback((next: number, clientX?: number, clientY?: number) => {
    const clamped = clamp(next, 0.05, 2);
    const rect = viewport.current?.getBoundingClientRect();
    if (!rect) { setZoom(clamped); return; }
    const cursorX = clientX === undefined ? rect.width / 2 : clientX - rect.left;
    const cursorY = clientY === undefined ? rect.height / 2 : clientY - rect.top;
    const currentZoom = zoomRef.current;
    const currentPan = panRef.current;
    const contentX = (cursorX - currentPan.x) / currentZoom;
    const contentY = (cursorY - currentPan.y) / currentZoom;
    const nextPan = { x: cursorX - contentX * clamped, y: cursorY - contentY * clamped };
    zoomRef.current = clamped;
    panRef.current = nextPan;
    setPan(nextPan);
    setZoom(clamped);
  }, [setPan, setZoom]);

  const fitView = useCallback(() => {
    if (!viewport.current || !content.current) return;
    const viewportWidth = viewport.current.clientWidth;
    const viewportHeight = viewport.current.clientHeight;
    const contentWidth = content.current.offsetWidth;
    const contentHeight = content.current.offsetHeight;
    const padding = 34;
    const fitted = calculateFitTransform(viewportWidth, viewportHeight, contentWidth, contentHeight, padding);
    zoomRef.current = fitted.zoom;
    panRef.current = { x: fitted.x, y: fitted.y };
    setZoom(fitted.zoom);
    setPan({ x: fitted.x, y: fitted.y });
  }, [setPan, setZoom]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      if (event.ctrlKey) {
        zoomAt(zoomRef.current * Math.exp(-event.deltaY * 0.012), event.clientX, event.clientY);
      } else {
        const current = panRef.current;
        const next = { x: current.x - event.deltaX, y: current.y - event.deltaY };
        panRef.current = next;
        setPan(next);
      }
    };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, [setPan, zoomAt]);

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !viewport.current) return;
    const rect = viewport.current.getBoundingClientRect();
    if (tool === "box") setBox({ x1: event.clientX - rect.left, y1: event.clientY - rect.top, x2: event.clientX - rect.left, y2: event.clientY - rect.top });
    else if (tool === "pan" || !(event.target as HTMLElement).closest("button")) {
      drag.current = { x: event.clientX, y: event.clientY, panX: panRef.current.x, panY: panRef.current.y };
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  };
  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (box && viewport.current) {
      const rect = viewport.current.getBoundingClientRect();
      setBox({ ...box, x2: event.clientX - rect.left, y2: event.clientY - rect.top });
    } else if (drag.current) {
      const next = { x: drag.current.panX + event.clientX - drag.current.x, y: drag.current.panY + event.clientY - drag.current.y };
      panRef.current = next;
      setPan(next);
    }
  };
  const pointerUp = () => {
    if (box && viewport.current) {
      const viewportRect = viewport.current.getBoundingClientRect();
      const left = viewportRect.left + Math.min(box.x1, box.x2);
      const top = viewportRect.top + Math.min(box.y1, box.y2);
      const right = viewportRect.left + Math.max(box.x1, box.x2);
      const bottom = viewportRect.top + Math.max(box.y1, box.y2);
      const ids = [...viewport.current.querySelectorAll<HTMLElement>("[data-component-id]")]
        .filter((element) => { const rect = element.getBoundingClientRect(); return rect.right >= left && rect.left <= right && rect.bottom >= top && rect.top <= bottom; })
        .map((element) => element.dataset.componentId!).filter((id, index, all) => all.indexOf(id) === index);
      commitSelection(ids);
    }
    setBox(null); drag.current = null;
  };

  const toggleAll = () => setExpanded(expanded.size === architecture?.nLayers ? new Set() : new Set(architecture?.layers.map((layer) => layer.index) ?? []));

  return (
    <section className="architecture-panel panel">
      <div className="canvas-toolbar">
        <div className="tool-group">
          <span>Select</span>
          <ToolButton label="Pointer" active={tool === "pointer"} onClick={() => setTool("pointer")}><MousePointer2 /></ToolButton>
          <ToolButton label="Box select" active={tool === "box"} onClick={() => setTool("box")}><BoxSelect /></ToolButton>
          <ToolButton label="Pan" active={tool === "pan"} onClick={() => setTool("pan")}><Hand /></ToolButton>
          <ToolButton label="Zoom out" onClick={() => zoomAt(zoom / 1.2)}><ZoomOut /></ToolButton>
          <ToolButton label="Zoom in" onClick={() => zoomAt(zoom * 1.2)}><ZoomIn /></ToolButton>
          <button className="fit-button" onClick={fitView} title="Fit the complete expanded diagram"><Scan />Fit all</button>
        </div>
        <div className="canvas-toolbar-right">
          {architecture && <button className="expand-all-button" onClick={toggleAll}>{expanded.size === architecture.nLayers ? "Collapse all" : "Expand all"}</button>}
          <label className="zoom-slider">Zoom <strong>{Math.round(zoom * 100)}%</strong><input aria-label="Zoom" type="range" min="5" max="200" value={zoom * 100} onChange={(event) => zoomAt(Number(event.target.value) / 100)} /></label>
          <span className={`legend-button ${headOverlay || mlpOverlay ? "overlay-active" : ""}`}><Layers3 />{headOverlay || mlpOverlay ? "Causal effect" : "Architecture"}</span>
        </div>
      </div>
      <div ref={viewport} className={`canvas-viewport tool-${tool}`} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp}>
        {!architecture ? <CanvasEmpty error={error} connected={connected} runtime={runtime} loading={loading} onRetry={onRetry} /> : (
          <div ref={content} className={`architecture-content semantic-${zoom < .25 ? "far" : zoom < .55 ? "mid" : "near"}`} style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
            <ModelBoundaryRow mode="output" architecture={architecture} selection={selection} selectNode={selectNode} />
            <div className="column-labels"><span>Layer</span><span>Residual Stream</span><span>Attention</span><span>MLP</span><span>Residual Stream</span></div>
            {[...architecture.layers].reverse().map((layer) => (
              <LayerRow
                key={layer.id}
                layer={layer}
                architecture={architecture}
                expanded={expanded.has(layer.index)}
                expandedHeads={expandedHeads}
                selected={selection}
                toggle={() => { const next = new Set(expanded); next.has(layer.index) ? next.delete(layer.index) : next.add(layer.index); setExpanded(next); }}
                toggleHead={(headId) => { const next = new Set(expandedHeads); next.has(headId) ? next.delete(headId) : next.add(headId); setExpandedHeads(next); }}
                selectNode={selectNode}
                contextNode={contextNode}
                headOverlay={headOverlay}
                mlpOverlay={mlpOverlay}
              />
            ))}
            <ModelBoundaryRow mode="input" architecture={architecture} selection={selection} selectNode={selectNode} />
          </div>
        )}
        {box && <div className="selection-box" style={{ left: Math.min(box.x1, box.x2), top: Math.min(box.y1, box.y2), width: Math.abs(box.x2 - box.x1), height: Math.abs(box.y2 - box.y1) }} />}
      </div>
      <div className="canvas-corner-actions"><button onClick={props.undo} disabled={!props.canUndo} aria-label="Undo selection"><Undo2 /></button><button onClick={props.redo} disabled={!props.canRedo} aria-label="Redo selection"><Redo2 /></button></div>
    </section>
  );
}

function ModelBoundaryRow({ mode, architecture, selection, selectNode }: {
  mode: "input" | "output";
  architecture: ArchitectureGraph;
  selection: string[];
  selectNode: (node: ComponentNode, additive?: boolean) => void;
}) {
  const nodes = mode === "input" ? [architecture.embedding, architecture.positionalEmbedding].filter(Boolean) as ComponentNode[] : [architecture.finalNorm, architecture.unembedding];
  const additive = (event: React.MouseEvent) => event.shiftKey || event.ctrlKey || event.metaKey;
  return <div className={`model-boundary-row ${mode}`}><span>{mode === "input" ? "Input / initialization" : "Output / readout"}</span><div>{mode === "input" && <span className="outside-model-node">Prompt → tokenizer <small>outside model</small></span>}{mode === "output" && <span className="residual-boundary-node">final residual x<sub>L</sub> →</span>}{nodes.map((node) => <button key={node.id} data-component-id={node.id} className={selection.includes(node.id) ? "component-selected" : ""} onClick={(event) => selectNode(node, additive(event))}><ComponentGlyph kind={node.kind} />{node.label}<small>{node.activationPoints[0]}</small></button>)}{mode === "input" && <span className="residual-boundary-node">sum once → initial residual x₀</span>}{mode === "output" && <span className="logits-node">Vocabulary logits · {architecture.vocabularySize.toLocaleString()}</span>}</div></div>;
}

function LayerRow({ layer, architecture, expanded, expandedHeads, selected, toggle, toggleHead, selectNode, contextNode, headOverlay, mlpOverlay }: {
  layer: LayerNode;
  architecture: ArchitectureGraph;
  expanded: boolean;
  expandedHeads: Set<string>;
  selected: string[];
  toggle: () => void;
  toggleHead: (headId: string) => void;
  selectNode: (node: ComponentNode, additive?: boolean) => void;
  contextNode: (node: ComponentNode, x: number, y: number) => void;
  headOverlay: HeadSweepResult | null;
  mlpOverlay: MlpSweepResult | null;
}) {
  const openHead = layer.heads.find((head) => expandedHeads.has(head.id));
  const layerSelected = [layer.attention, layer.mlp, layer.norm1, layer.norm2, ...layer.heads].some((node) => selected.includes(node.id));
  const choose = (node: ComponentNode, event: React.MouseEvent) => { event.stopPropagation(); selectNode(node, event.shiftKey || event.ctrlKey || event.metaKey); };
  const mlpEffect = mlpOverlay?.effects.find((value) => value.componentId === layer.mlp.id);
  const context = (node: ComponentNode, event: React.MouseEvent) => { event.preventDefault(); event.stopPropagation(); contextNode(node, event.clientX, event.clientY); };
  return (
    <article className={`layer-row ${expanded ? "expanded" : ""} ${openHead ? "head-expanded" : ""} ${layerSelected ? "layer-selected" : ""}`}>
      <button className="layer-index" onClick={toggle} aria-label={`${expanded ? "Collapse" : "Expand"} layer ${layer.index}`}><strong>{layer.index}</strong>{expanded ? <ChevronDown /> : <ChevronRight />}</button>
      <div className="residual-entry"><span className="flow-line" /><button data-component-id={layer.residualPre.id} title={layer.residualPre.id} onClick={(event) => choose(layer.residualPre, event)}><Plus /></button></div>
      <div className="attention-stage">
        {expanded && <button className={`norm-chip ${selected.includes(layer.norm1.id) ? "component-selected" : ""}`} data-component-id={layer.norm1.id} onClick={(event) => choose(layer.norm1, event)}>{architecture.normType}</button>}
        <button data-component-id={layer.attention.id} className={`attention-block ${selected.includes(layer.attention.id) ? "component-selected" : ""}`} onClick={(event) => choose(layer.attention, event)} onContextMenu={(event) => context(layer.attention, event)}><span>{expanded ? `Layer ${layer.index} attention` : "Multi-Head Attention"}</span>{expanded && <small>{architecture.nHeads} query heads · {architecture.nKeyValueHeads} KV heads</small>}</button>
        {expanded && <div className="heads-container" style={{ "--heads": architecture.nHeads } as CSSProperties}>{layer.heads.map((head) => { const effect = headOverlay?.effects.find((value) => value.componentId === head.id); return <button key={head.id} data-component-id={head.id} className={`${selected.includes(head.id) ? "component-selected" : ""} ${effect ? "causal-overlay" : ""}`} style={effect ? causalOverlayStyle(effect.delta, Math.max(Math.abs(headOverlay!.minimum), Math.abs(headOverlay!.maximum))) : undefined} title={effect ? `${shortComponentId(head.id)}: ${signed(effect.delta)} metric change` : head.id} onClick={(event) => choose(head, event)} onDoubleClick={() => toggleHead(head.id)} onContextMenu={(event) => context(head, event)} aria-label={`Select L${layer.index}H${head.head}`}>H{head.head}{effect && <small>{signedCompact(effect.delta)}</small>}</button>; })}</div>}
        {expanded && openHead && <div className="head-internals"><header><span>{displayName(openHead)}</span><button onClick={() => toggleHead(openHead.id)}><X /></button></header><div>{openHead.children.map((child, index) => <span key={child.id}><button data-component-id={child.id} className={selected.includes(child.id) ? "component-selected" : ""} onClick={(event) => choose(child, event)}>{child.label}</button>{index < openHead.children.length - 1 && <i>→</i>}</span>)}</div></div>}
      </div>
      <div className="residual-add"><span className="flow-line" /><button data-component-id={layer.residualMid.id} title={layer.residualMid.id} onClick={(event) => choose(layer.residualMid, event)}><Plus /></button></div>
      <div className="mlp-stage">
        {expanded && <button className={`norm-chip ${selected.includes(layer.norm2.id) ? "component-selected" : ""}`} data-component-id={layer.norm2.id} onClick={(event) => choose(layer.norm2, event)}>{architecture.normType}</button>}
        <button data-component-id={layer.mlp.id} className={`mlp-block ${selected.includes(layer.mlp.id) ? "component-selected" : ""} ${mlpEffect ? "causal-overlay" : ""}`} style={mlpEffect ? causalOverlayStyle(mlpEffect.delta, Math.max(Math.abs(mlpOverlay!.minimum), Math.abs(mlpOverlay!.maximum))) : undefined} title={mlpEffect ? `MLP L${layer.index}: ${signed(mlpEffect.delta)} metric change` : layer.mlp.id} onClick={(event) => choose(layer.mlp, event)} onContextMenu={(event) => context(layer.mlp, event)}>{expanded ? `MLP (${architecture.dMlp ?? "hidden"})` : "MLP"}{mlpEffect && <small>{signedCompact(mlpEffect.delta)}</small>}</button>
        {expanded && <div className="mlp-internals">{layer.mlp.children.map((child) => <button key={child.id} data-component-id={child.id} className={selected.includes(child.id) ? "component-selected" : ""} onClick={(event) => choose(child, event)}>{child.label}</button>)}</div>}
      </div>
      <div className="residual-exit"><span className="flow-line" /><button data-component-id={layer.residualPost.id} title={layer.residualPost.id} onClick={(event) => choose(layer.residualPost, event)}><Plus /></button><span className="flow-tail" /></div>
      <button className="row-more" onClick={(event) => context(layer.residualPost, event)} onContextMenu={(event) => context(layer.residualPost, event)} aria-label={`More actions for layer ${layer.index}`}><EllipsisVertical /></button>
    </article>
  );
}

function CanvasEmpty({ error, connected, runtime, loading, onRetry }: { error: string | null; connected: boolean; runtime: RuntimeStatus | null; loading: boolean; onRetry: () => void }) {
  const heading = loading
    ? "Loading the model locally"
    : error && !connected
      ? "Desktop service is not connected"
      : error
        ? "Model could not be loaded"
        : "Load a model to reveal its architecture";
  const detail = loading
    ? runtime?.loadMessage || "TransformerLens is preparing the selected model. A first load can take a few minutes on a CPU; keep Kannaadi open."
    : error && !connected
      ? "Launch Kannaadi through its desktop entry point, or start the Python service for browser development."
      : error
        ? "The desktop service is still available. Review the model-loading error below, then retry or choose a smaller model."
        : "The architecture will be built from the loaded model configuration.";
  const stages = ["validating", "importing_runtime", "resolving_model", "loading_weights", "centering_unembedding", "configuring_hooks", "building_architecture", "initializing_experiments", "ready"];
  const stageIndex = runtime?.loadStage ? stages.indexOf(runtime.loadStage) : -1;
  return <div className="canvas-empty"><div className="empty-orbit"><KannaadiLogo /></div><h2>{heading}</h2><p>{detail}</p>{loading && <div className="truthful-loader" role="progressbar" aria-label="Model loading"><i /><div>{stages.map((stage, index) => <span key={stage} className={index < stageIndex ? "complete" : index === stageIndex ? "active" : ""}>{stage.replaceAll("_", " ")}</span>)}</div>{runtime?.loadElapsedSeconds !== null && runtime?.loadElapsedSeconds !== undefined && <small>Elapsed {runtime.loadElapsedSeconds.toFixed(1)} s · stage progress only; download size is not reported by TransformerLens.</small>}</div>}{error && <code>{error}</code>}{error && !connected && <button className="retry-service" onClick={onRetry} disabled={loading}><RotateCcw />Restart desktop service</button>}</div>;
}

function Inspector(props: {
  architecture: ArchitectureGraph | null;
  selected: ComponentNode | null;
  selectedCount: number;
  activeRun: RunRecord | null;
  running: boolean;
  onInspect: () => void;
  scopePosition: number | null;
  setScopePosition: (position: number | null) => void;
  onZeroAblate: () => void;
  onMeanAblate: () => void;
  onPatch: () => void;
  onNeuronAblate: (index: number, kind: "zero_ablation" | "mean_ablation") => void;
  onNeuronPatch: (index: number) => void;
  onHeadSweep: () => void;
  onMlpSweep: () => void;
  onAttribution: () => void;
  canPatch: boolean;
  onSaveGroup: () => void;
  onReturnToClean: () => void;
  onAction: (message: string) => void;
}) {
  const { architecture, selected, selectedCount, activeRun, running, onInspect, scopePosition, setScopePosition, onZeroAblate, onMeanAblate, onPatch, onNeuronAblate, onNeuronPatch, onHeadSweep, onMlpSweep, onAttribution, canPatch, onSaveGroup, onReturnToClean, onAction } = props;
  const [neuronIndex, setNeuronIndex] = useState(0);
  const hook = selected?.activationPoints[0] || "—";
  const intervenable = selected?.kind === "head" || selected?.kind === "mlp";
  return (
    <aside className="inspector panel">
      <header className="inspector-bar"><span>Inspector</span><small>{selectedCount ? `${selectedCount} selected` : "No selection"}</small></header>
      {!selected ? <div className="inspector-empty"><Crosshair /><p>Select a component on the architecture canvas.</p></div> : <>
        <div className="inspector-heading"><ComponentGlyph kind={selected.kind} large /><h2>{displayName(selected)}</h2><span className="type-pill">{typeLabel(selected.kind)}</span></div>
        <dl className="inspector-properties">
          <div><dt>Type</dt><dd>{selected.kind === "head" ? "attention.head" : selected.kind}</dd></div>
          {selected.layer != null && <div><dt>Layer</dt><dd>{selected.layer}</dd></div>}
          {selected.head != null && <div><dt>Head Index</dt><dd>{selected.head}</dd></div>}
          <div><dt>Canonical ID</dt><dd><code title={selected.id}>{selected.id}</code></dd></div>
          <div><dt>Hook</dt><dd><code title={hook}>{hook}</code></dd></div>
          {architecture && <><div><dt>Input Dim</dt><dd>{architecture.dModel}</dd></div><div><dt>{selected.kind === "head" || selected.kind === "projection" ? "Head Dim" : "Output Dim"}</dt><dd>{selected.kind === "head" || selected.kind === "projection" ? architecture.dHead : architecture.dModel}</dd></div></>}
          <div><dt>Current Run</dt><dd>{activeRun?.label ?? "Not run"}</dd></div>
        </dl>
        <section className="about-component"><h3>About</h3><p>{componentDescription(selected, architecture)}</p><code>Shape: {shapeFor(selected, architecture)}</code></section>
        {intervenable && activeRun && <label className="intervention-scope"><span>Intervention scope</span><select value={scopePosition ?? "all"} onChange={(event) => setScopePosition(event.target.value === "all" ? null : Number(event.target.value))}><option value="all">All token positions</option>{activeRun.tokens.map((token) => <option key={token.position} value={token.position}>Position {token.position}: {token.display}</option>)}</select></label>}
        {selected.kind === "mlp" && architecture?.dMlp && <div className="neuron-control"><label><span>Neuron index</span><input aria-label="Neuron index" type="number" min="0" max={architecture.dMlp - 1} value={neuronIndex} onChange={(event) => setNeuronIndex(clamp(Number(event.target.value), 0, architecture.dMlp! - 1))} /></label><small>Intervene on post-activation neuron output. Index range 0–{architecture.dMlp - 1}.</small><div><button disabled={running || !activeRun} onClick={() => onNeuronAblate(neuronIndex, "zero_ablation")}><CircleDot />Zero neuron</button><button disabled={running || !activeRun} onClick={() => onNeuronAblate(neuronIndex, "mean_ablation")}><Activity />Mean neuron</button><button disabled={running || !canPatch} onClick={() => onNeuronPatch(neuronIndex)}><SquareDashedMousePointer />Patch neuron</button></div></div>}
        <section className="inspector-actions"><h3>Actions</h3><div>
          <ActionButton icon={<Eye />} label="Inspect" onClick={onInspect} />
          <ActionButton icon={<CircleDot />} label="Zero Ablate" disabled={running || !activeRun || !selectedCount || !intervenable} onClick={onZeroAblate} />
          <ActionButton icon={<Activity />} label="Mean Ablate" disabled={running || !activeRun || !selectedCount || !intervenable} onClick={onMeanAblate} />
          <ActionButton icon={<SquareDashedMousePointer />} label="Patch…" disabled={running || !canPatch || !intervenable} onClick={onPatch} />
          <ActionButton wide icon={<FlaskConical />} label={selected.kind === "mlp" ? "Sweep all MLPs" : "Sweep all heads"} disabled={running || !activeRun} onClick={selected.kind === "mlp" ? onMlpSweep : onHeadSweep} />
          <ActionButton wide icon={<Braces />} label="Direct attribution" disabled={running || !activeRun} onClick={onAttribution} />
          <ActionButton icon={<Code2 />} label="Copy ID" onClick={() => { void navigator.clipboard?.writeText(selected.id); onAction("Canonical component ID copied."); }} />
          <ActionButton icon={<Save />} label="Save Group" disabled={!intervenable} onClick={onSaveGroup} />
          {(activeRun?.kind === "intervened" || activeRun?.kind === "patched") && <ActionButton wide icon={<RotateCcw />} label="Return to Baseline" onClick={onReturnToClean} />}
        </div></section>
        <section className="method-scope"><Info /><div><strong>{activeRun ? "Run evidence" : "Architecture metadata"}</strong><span>{activeRun ? `${runMethodLabel(activeRun)} · ${activeRun.device} · ${activeRun.dtype}` : "Select Run to collect prompt-specific evidence."}</span></div></section>
      </>}
    </aside>
  );
}

function AnalysisPanel(props: {
  activeTab: PanelTab;
  setActiveTab: (tab: PanelTab) => void;
  architecture: ArchitectureGraph | null;
  selected: ComponentNode | null;
  model: ModelCatalogEntry;
  prompt: string;
  run: RunRecord | null;
  attention: AttentionResult | null;
  activation: ActivationSeries | null;
  residual: ResidualStreamResult | null;
  comparison: RunComparison | null;
  contrast: ContrastResult | null;
  effect: CausalEffect | null;
  headSweep: HeadSweepResult | null;
  mlpSweep: MlpSweepResult | null;
  sweepKind: "heads" | "mlps";
  setSweepKind: (kind: "heads" | "mlps") => void;
  attribution: AttributionResult | null;
  sweepLoading: boolean;
  attributionLoading: boolean;
  onHeadSweep: (kind: "zero_ablation" | "mean_ablation") => void;
  onMlpSweep: (kind: "zero_ablation" | "mean_ablation") => void;
  onAttribution: () => void;
  onExport: () => void;
  loading: boolean;
  error: string | null;
}) {
  const { activeTab, setActiveTab, architecture, selected, model, prompt, run, attention, activation, residual, comparison, contrast, effect, headSweep, mlpSweep, sweepKind, setSweepKind, attribution, sweepLoading, attributionLoading, onHeadSweep, onMlpSweep, onAttribution, onExport, loading, error } = props;
  const tabs: PanelTab[] = ["Tokens", "Alignment", "Attention", "QK / OV", "Logits", "Activations", "Residual Stream", "Causal Sweep", "Attribution", "Code"];
  let content: ReactNode;
  if (activeTab === "Code") content = <CodePanel architecture={architecture} selected={selected} model={model} prompt={prompt} run={run} />;
  else if (activeTab === "Alignment") content = contrast ? <AlignmentPanel contrast={contrast} /> : <EvidencePrompt message="Open Contrast setup and run a clean/corrupted prompt pair to inspect token alignment." />;
  else if (activeTab === "Causal Sweep" && sweepLoading) content = <EvidenceLoading message={`Running exact batched ${sweepKind === "heads" ? "attention-head" : "MLP"} interventions…`} />;
  else if (activeTab === "Causal Sweep") content = <SweepWorkspace headResult={headSweep} mlpResult={mlpSweep} kind={sweepKind} setKind={setSweepKind} onHeadSweep={onHeadSweep} onMlpSweep={onMlpSweep} disabled={!run} />;
  else if (activeTab === "Attribution" && attributionLoading) content = <EvidenceLoading message="Decomposing the selected logit metric across model components…" />;
  else if (activeTab === "Attribution") content = attribution ? <AttributionPanel result={attribution} /> : <AttributionEmpty onRun={onAttribution} disabled={!run} />;
  else if (!run) content = <EvidenceEmpty tab={activeTab} selected={selected} />;
  else if (loading) content = <EvidenceLoading />;
  else if (error) content = <EvidenceError message={error} />;
  else if (activeTab === "Tokens") content = <TokensPanel run={run} />;
  else if (activeTab === "Attention") content = attention ? <AttentionPanel result={attention} mode="pattern" /> : <EvidencePrompt message="Select an attention head to inspect its token-to-token pattern." />;
  else if (activeTab === "QK / OV") content = attention ? <QkovPanel result={attention} /> : <EvidencePrompt message="Select an attention head to inspect QK scores and projected result strength." />;
  else if (activeTab === "Logits") content = <LogitsPanel run={run} comparison={comparison} effect={effect} />;
  else if (activeTab === "Activations") content = activation ? <ActivationPanel result={activation} /> : <EvidencePrompt message="Select a component with a cached activation." />;
  else content = residual ? <ResidualPanel result={residual} /> : <EvidencePrompt message="Residual-stream evidence is available after a run." />;
  return <section className="analysis-panel panel"><nav>{tabs.map((tab) => <button className={activeTab === tab ? "active" : ""} key={tab} onClick={() => setActiveTab(tab)}>{tab}</button>)}<button className="export-bundle" onClick={onExport} disabled={!run} title="Download a reproducible research bundle"><Download />Export</button></nav><div className="analysis-content">{content}</div></section>;
}

function TokensPanel({ run }: { run: RunRecord }) {
  return <div className="tokens-panel"><section><header><span>Prompt tokens</span><small>{run.tokens.length} tokens</small></header><table><thead><tr><th>Pos</th><th>Token</th><th>ID</th><th>Top next-token prediction</th><th>Probability</th></tr></thead><tbody>{run.tokens.map((token) => <tr key={token.position}><td>{token.position}</td><td><code>{token.display}</code></td><td>{token.tokenId}</td><td><code>{token.nextToken}</code></td><td>{formatPercent(token.nextTokenProbability ?? 0)}</td></tr>)}</tbody></table></section><Predictions predictions={run.topPredictions} /></div>;
}

function AlignmentPanel({ contrast }: { contrast: ContrastResult }) {
  const alignment = contrast.alignment;
  return <div className="alignment-panel"><header><div><strong>Clean ↔ corrupted token alignment</strong><span>Minimum-edit alignment used for activation patch mappings</span></div><div><small>Exact matches</small><strong>{alignment.exactMatches}/{Math.max(alignment.sourceLength, alignment.destinationLength)}</strong></div></header><div className="alignment-grid"><span>Clean source</span><span>Relation</span><span>Corrupted destination</span>{alignment.pairs.map((pair, index) => <div className={`alignment-pair ${pair.status}`} key={`${pair.sourcePosition}-${pair.destinationPosition}-${index}`}><code>{pair.sourcePosition === null ? "—" : `${pair.sourcePosition}  ${pair.sourceToken}`}</code><span>{pair.status === "exact" ? "=" : pair.status === "substitution" ? "→" : "gap"}<small>{pair.status.replaceAll("_", " ")}</small></span><code>{pair.destinationPosition === null ? "—" : `${pair.destinationPosition}  ${pair.destinationToken}`}</code></div>)}</div><footer><Info />Substitutions are position mappings, not claims of semantic equivalence. Review the alignment before interpreting a patch.</footer></div>;
}

function SweepEmpty({ onSweep, disabled, target = "attention heads" }: { onSweep: (kind: "zero_ablation" | "mean_ablation") => void; disabled: boolean; target?: "attention heads" | "MLPs" }) {
  return <div className="sweep-empty"><FlaskConical /><div><h3>Whole-model causal sweep</h3><p>Measure every {target === "MLPs" ? "MLP block" : "attention head"} against the target metric with exact batched interventions.</p></div><div><button disabled={disabled} onClick={() => onSweep("zero_ablation")}><CircleDot />Zero sweep</button><button disabled={disabled} onClick={() => onSweep("mean_ablation")}><Activity />Mean sweep</button></div></div>;
}

function SweepWorkspace({ headResult, mlpResult, kind, setKind, onHeadSweep, onMlpSweep, disabled }: { headResult: HeadSweepResult | null; mlpResult: MlpSweepResult | null; kind: "heads" | "mlps"; setKind: (kind: "heads" | "mlps") => void; onHeadSweep: (ablation: "zero_ablation" | "mean_ablation") => void; onMlpSweep: (ablation: "zero_ablation" | "mean_ablation") => void; disabled: boolean }) {
  const result = kind === "heads" ? headResult : mlpResult;
  return <div className="sweep-workspace"><div className="result-switch"><button className={kind === "heads" ? "active" : ""} onClick={() => setKind("heads")}>Attention heads{headResult && <i />}</button><button className={kind === "mlps" ? "active" : ""} onClick={() => setKind("mlps")}>MLPs{mlpResult && <i />}</button></div>{result ? kind === "heads" ? <SweepPanel result={headResult!} /> : <MlpSweepPanel result={mlpResult!} /> : <SweepEmpty onSweep={kind === "heads" ? onHeadSweep : onMlpSweep} disabled={disabled} target={kind === "heads" ? "attention heads" : "MLPs"} />}</div>;
}

function SweepPanel({ result }: { result: HeadSweepResult }) {
  const layers = [...new Set(result.effects.map((effect) => effect.layer))];
  const heads = [...new Set(result.effects.map((effect) => effect.head))];
  const maximum = Math.max(Math.abs(result.minimum), Math.abs(result.maximum), 1e-9);
  const strongest = [...result.effects].sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta)).slice(0, 5);
  return <div className="sweep-panel"><header><div><strong>{result.kind === "zero_ablation" ? "Zero-ablation" : "Within-prompt mean-ablation"} sweep</strong><span>{result.metric.metric.replaceAll("_", " ")} at position {result.metric.position} · baseline {result.metric.value.toFixed(3)}</span></div><div><small>Runtime</small><strong>{(result.durationMs / 1000).toFixed(1)} s</strong></div></header><div className="sweep-body"><div className="sweep-matrix" style={{ "--sweep-heads": heads.length } as CSSProperties}><span className="corner">Layer</span>{heads.map((head) => <span key={head}>H{head}</span>)}{layers.map((layer) => <div className="sweep-row" key={layer}><strong>L{layer}</strong>{heads.map((head) => { const effect = result.effects.find((item) => item.layer === layer && item.head === head)!; return <button key={head} style={causalOverlayStyle(effect.delta, maximum)} title={`L${layer}H${head}: ${signed(effect.delta)}`}>{signedCompact(effect.delta)}</button>; })}</div>)}</div><aside><h4>Strongest effects</h4>{strongest.map((effect) => <div key={effect.componentId}><code>{shortComponentId(effect.componentId)}</code><span className={effect.delta < 0 ? "negative" : "positive"}>{signed(effect.delta)}</span></div>)}<p>Δ = intervened metric − baseline metric</p></aside></div><footer><span className="negative">decreases metric</span><i /><span className="positive">increases metric</span></footer></div>;
}

function MlpSweepPanel({ result }: { result: MlpSweepResult }) {
  const maximum = Math.max(Math.abs(result.minimum), Math.abs(result.maximum), 1e-9);
  const strongest = [...result.effects].sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta));
  return <div className="mlp-sweep-panel"><header><div><strong>{result.kind === "zero_ablation" ? "Zero-ablation" : "Within-prompt mean-ablation"} MLP sweep</strong><span>{result.metric.metric.replaceAll("_", " ")} at position {result.metric.position} · baseline {result.metric.value.toFixed(3)}</span></div><small>{(result.durationMs / 1000).toFixed(1)} s</small></header><div className="mlp-sweep-chart">{result.effects.map((effect) => <div key={effect.componentId}><code>L{effect.layer}</code><button style={causalOverlayStyle(effect.delta, maximum)} title={`${effect.componentId}: ${signed(effect.delta)}`}><i style={{ width: `${Math.abs(effect.delta) / maximum * 100}%` }} /><strong>{signed(effect.delta)}</strong></button></div>)}</div><aside><h4>Strongest MLP effects</h4>{strongest.slice(0, 6).map((effect) => <span key={effect.componentId}><code>MLP L{effect.layer}</code><b className={effect.delta < 0 ? "negative" : "positive"}>{signed(effect.delta)}</b></span>)}</aside><footer>Each cell is one exact intervention. Δ = intervened metric − baseline metric.</footer></div>;
}

function AttributionEmpty({ onRun, disabled }: { onRun: () => void; disabled: boolean }) {
  return <div className="sweep-empty"><Braces /><div><h3>Direct logit attribution</h3><p>Decompose the chosen target or logit-difference metric into token embedding, positional embedding, attention-head, and MLP contributions.</p></div><button disabled={disabled} onClick={onRun}><Play />Compute attribution</button></div>;
}

function AttributionPanel({ result }: { result: AttributionResult }) {
  const components = result.effects.filter((effect) => effect.kind !== "remainder");
  const strongest = [...components].sort((left, right) => Math.abs(right.value) - Math.abs(left.value)).slice(0, 16);
  const maximum = Math.max(Math.abs(result.minimum), Math.abs(result.maximum), 1e-9);
  return <div className="attribution-panel"><header><div><strong>Direct logit attribution</strong><span>{result.metric.metric.replaceAll("_", " ")} · observed value {result.metric.value.toFixed(4)}</span></div><div><small>Reconciled total</small><strong>{(result.componentSum + result.remainder).toFixed(4)}</strong></div></header><div className="attribution-body"><section><h4>Strongest component contributions</h4>{strongest.map((effect) => <div key={effect.componentId}><code title={effect.componentId}>{effect.label}</code><i><b style={{ width: `${Math.abs(effect.value) / maximum * 100}%`, marginLeft: effect.value < 0 ? "auto" : undefined }} /></i><strong className={effect.value < 0 ? "negative" : "positive"}>{signed(effect.value)}</strong></div>)}</section><aside><h4>Accounting</h4><dl><div><dt>Components</dt><dd>{result.componentSum.toFixed(4)}</dd></div><div><dt>Remainder</dt><dd>{signed(result.remainder)}</dd></div><div><dt>Runtime</dt><dd>{result.durationMs.toFixed(0)} ms</dd></div></dl><p>{result.caveat}</p></aside></div></div>;
}

function Predictions({ predictions }: { predictions: RunRecord["topPredictions"] }) {
  const maximum = predictions[0]?.probability || 1;
  return <section className="predictions"><header><span>Next-token predictions</span><small>final position</small></header>{predictions.map((prediction, index) => <div className="prediction-row" key={prediction.tokenId}><span>{index + 1}</span><code>{prediction.display}</code><div><i style={{ width: `${prediction.probability / maximum * 100}%` }} /></div><strong>{formatPercent(prediction.probability)}</strong></div>)}</section>;
}

function AttentionPanel({ result, mode }: { result: AttentionResult; mode: "pattern" | "scores" }) {
  const values = mode === "pattern" ? result.pattern : result.scores;
  return <div className="heatmap-panel"><header><div><strong>{mode === "pattern" ? "Attention pattern" : "Pre-softmax QK scores"} · L{result.layer}H{result.head}</strong><code>{mode === "pattern" ? result.patternHook : result.scoreHook}</code></div><span>{mode === "pattern" ? "Each row sums to 1" : "Causal-mask values clipped for display"}</span></header><Heatmap values={values} rows={result.queryTokens} columns={result.keyTokens} diverging={mode === "scores"} /></div>;
}

function Heatmap({ values, rows, columns, diverging = false }: { values: number[][]; rows: string[]; columns: string[]; diverging?: boolean }) {
  const finite = values.flat().filter((value) => Number.isFinite(value) && value > -9_999);
  const minimum = Math.min(...finite, 0);
  const maximum = Math.max(...finite, 1e-9);
  const maxAbsolute = Math.max(Math.abs(minimum), Math.abs(maximum), 1e-9);
  return <div className="heatmap-scroll"><div className="heatmap" style={{ gridTemplateColumns: `76px repeat(${columns.length}, minmax(36px, 1fr))` }}><span />{columns.map((token, index) => <code className="column-token" key={`${token}-${index}`}>{token}</code>)}{values.flatMap((row, rowIndex) => [<code className="row-token" key={`row-${rowIndex}`}>{rows[rowIndex]}</code>, ...row.map((value, columnIndex) => { const masked = value <= -9_999; const normalized = diverging ? value / maxAbsolute : value / maximum; const background = masked ? "#eeeae3" : diverging ? normalized >= 0 ? `rgba(91,65,201,${.08 + Math.abs(normalized) * .82})` : `rgba(221,91,70,${.08 + Math.abs(normalized) * .72})` : `rgba(82,58,172,${.04 + normalized * .92})`; return <span className={`heat-cell ${masked ? "masked" : ""}`} key={`${rowIndex}-${columnIndex}`} style={{ background }} title={masked ? "Causally masked" : `${rows[rowIndex]} → ${columns[columnIndex]}: ${value.toFixed(5)}`}>{masked ? "" : Math.abs(value) >= .01 ? value.toFixed(2) : ""}</span>; })])}</div></div>;
}

function QkovPanel({ result }: { result: AttentionResult }) {
  return <div className="qkov-panel"><AttentionPanel result={result} mode="scores" /><section className="head-norms"><header><strong>Per-token vector strength</strong><span>Q, K, and projected head result L2 norms</span></header>{result.queryTokens.map((token, index) => <div key={`${token}-${index}`}><code>{token}</code><MetricBar label="Q" value={result.qNorms[index]} maximum={Math.max(...result.qNorms)} /><MetricBar label="K" value={result.kNorms[index]} maximum={Math.max(...result.kNorms)} /><MetricBar label="Result" value={result.resultNorms[index]} maximum={Math.max(...result.resultNorms)} /></div>)}</section></div>;
}

function MetricBar({ label, value, maximum }: { label: string; value: number; maximum: number }) {
  return <span className="metric-bar"><small>{label}</small><i><b style={{ width: `${value / (maximum || 1) * 100}%` }} /></i><strong>{value.toFixed(2)}</strong></span>;
}

function LogitsPanel({ run, comparison, effect }: { run: RunRecord; comparison: RunComparison | null; effect: CausalEffect | null }) {
  if ((run.kind === "clean" || run.kind === "corrupted") || !comparison) return <div className="logits-clean"><Predictions predictions={run.topPredictions} /><EvidencePrompt message="Apply a scoped ablation or clean-to-corrupted patch to compare its causal effect on the output distribution." /></div>;
  const intervention = run.interventions[0];
  return <div className="comparison-panel">{effect && <section className="metric-effect"><div><small>{effect.intervened.metric.replaceAll("_", " ")}</small><strong><code>{visibleToken(effect.intervened.targetToken)}</code>{effect.intervened.distractorToken ? <> − <code>{visibleToken(effect.intervened.distractorToken)}</code></> : null}</strong></div><span>{effect.baseline.value.toFixed(3)} → {effect.intervened.value.toFixed(3)}</span><b className={effect.delta < 0 ? "negative" : "positive"}>{signed(effect.delta)}</b></section>}<header><div><strong>Baseline versus intervention</strong><span>Baseline top token <code>{visibleToken(comparison.baselineTopToken)}</code> changed by <b className={comparison.baselineTopTokenDelta < 0 ? "negative" : "positive"}>{signed(comparison.baselineTopTokenDelta)}</b> logits</span></div><div><small>KL divergence</small><strong>{comparison.klDivergence.toFixed(5)}</strong></div></header><table><thead><tr><th>Token</th><th>Baseline probability</th><th>Intervened</th><th>Δ probability</th><th>Δ logit</th></tr></thead><tbody>{comparison.tokens.map((token) => <tr key={token.tokenId}><td><code>{token.display}</code></td><td>{formatPercent(token.baselineProbability)}</td><td>{formatPercent(token.intervenedProbability)}</td><td className={token.deltaProbability < 0 ? "negative" : "positive"}>{signed(token.deltaProbability * 100)} pp</td><td className={token.deltaLogit < 0 ? "negative" : "positive"}>{signed(token.deltaLogit)}</td></tr>)}</tbody></table><footer><Info />Exact {intervention.kind.replaceAll("_", " ")} at {intervention.tokenScope === "all" ? "all token positions" : `positions ${intervention.positions.join(", ")}`}. {intervention.kind === "mean_ablation" ? "The reference is this run’s mean head result across positions, not a dataset mean." : intervention.kind === "zero_ablation" ? "Zero ablation can be out of distribution; compare baselines before making a broad mechanistic claim." : "Patched activations come from the aligned clean source run."}</footer></div>;
}

function ActivationPanel({ result }: { result: ActivationSeries }) {
  const maximum = Math.max(...result.values, 1e-9);
  return <div className="activation-panel"><header><div><strong>{shortComponentId(result.componentId)} activation magnitude</strong><code>{result.hookName}</code></div><span>{result.measure.replaceAll("_", " ")} · axis: {result.axes.join(", ")}</span></header><div>{result.values.map((value, index) => <div className="activation-row" key={index}><span>{index}</span><code>{result.tokenLabels[index]}</code><i><b style={{ width: `${value / maximum * 100}%` }} /></i><strong>{value.toFixed(3)}</strong></div>)}</div></div>;
}

function ResidualPanel({ result }: { result: ResidualStreamResult }) {
  const points = result.points.filter((point) => point.stage === "post");
  const width = 720, height = 180, padX = 40, padY = 25;
  const logits = points.map((point) => point.targetLogit);
  const min = Math.min(...logits), max = Math.max(...logits), range = max - min || 1;
  const coords = points.map((point, index) => `${padX + index / Math.max(points.length - 1, 1) * (width - padX * 2)},${height - padY - (point.targetLogit - min) / range * (height - padY * 2)}`).join(" ");
  return <div className="residual-panel"><header><div><strong>Logit lens through the residual stream</strong><span>Target <code>{visibleToken(result.targetToken)}</code> at position {result.position}</span></div><small>Final normalization and unembedding applied at each layer</small></header><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Target-token logit by layer">{[0, .5, 1].map((fraction) => <line key={fraction} x1={padX} x2={width - padX} y1={padY + fraction * (height - padY * 2)} y2={padY + fraction * (height - padY * 2)} />)}<polyline points={coords} />{points.map((point, index) => { const x = padX + index / Math.max(points.length - 1, 1) * (width - padX * 2); const y = height - padY - (point.targetLogit - min) / range * (height - padY * 2); return <g key={point.layer}><circle cx={x} cy={y} r="3" /><text x={x} y={height - 7}>L{point.layer}</text><title>Layer {point.layer}: {point.targetLogit.toFixed(4)}</title></g>; })}</svg><div className="residual-summary"><span>Initial <strong>{logits[0]?.toFixed(3)}</strong></span><span>Final <strong>{logits.at(-1)?.toFixed(3)}</strong></span><span>Change <strong>{signed((logits.at(-1) ?? 0) - (logits[0] ?? 0))}</strong></span></div></div>;
}

function CodePanel({ architecture, selected, model, prompt, run }: { architecture: ArchitectureGraph | null; selected: ComponentNode | null; model: ModelCatalogEntry; prompt: string; run: RunRecord | null }) {
  const selectedHead = selected?.kind === "head" ? selected : null;
  const intervention = run?.interventions[0];
  const component = intervention?.componentIds[0];
  const headMatch = component?.match(/^blocks\.(\d+)\.attn\.head\.(\d+)$/);
  const neuronMatch = component?.match(/^blocks\.(\d+)\.mlp\.neuron\.(\d+)$/);
  const mlpMatch = component?.match(/^blocks\.(\d+)\.mlp$/);
  const interventionComponent = headMatch ? "head" : neuronMatch ? "neuron" : mlpMatch ? "mlp" : null;
  const layer = headMatch?.[1] ?? neuronMatch?.[1] ?? mlpMatch?.[1];
  const hookName = headMatch ? `blocks.${layer}.attn.hook_result` : neuronMatch ? `blocks.${layer}.mlp.hook_post` : mlpMatch ? `blocks.${layer}.hook_mlp_out` : null;
  const indexLines = headMatch ? [`head_index = ${headMatch[2]}`] : neuronMatch ? [`neuron_index = ${neuronMatch[2]}`] : [];
  const patchDestination = interventionComponent === "head" ? "result[:, destination_pos, head_index, :]" : interventionComponent === "neuron" ? "result[:, destination_pos, neuron_index]" : "result[:, destination_pos, :]";
  const patchSource = interventionComponent === "head" ? "source_cache[hook_name][:, source_pos, head_index, :]" : interventionComponent === "neuron" ? "source_cache[hook_name][:, source_pos, neuron_index]" : "source_cache[hook_name][:, source_pos, :]";
  const ablationDestination = interventionComponent === "head" ? "result[:, positions, head_index, :]" : interventionComponent === "neuron" ? "result[:, positions, neuron_index]" : "result[:, positions, :]";
  const meanReference = interventionComponent === "head" ? "cache[hook_name][:, :, head_index, :].mean(dim=1)" : interventionComponent === "neuron" ? "cache[hook_name][:, :, neuron_index].mean(dim=1)" : "cache[hook_name].mean(dim=1)";
  const meanValue = interventionComponent === "neuron" ? "mean_result[:, None]" : "mean_result[:, None, :]";
  const positions = intervention?.tokenScope === "positions" ? intervention.positions : run?.tokens.map((token) => token.position) ?? [];
  const reducedDtype = run?.dtype.match(/(bfloat16|float16)$/)?.[1];
  const modelLoadLines = reducedDtype ? [
    `model = HookedTransformer.from_pretrained_no_processing(${JSON.stringify(model.repository)}, device=${JSON.stringify(run?.device ?? "cpu")}, dtype=torch.${reducedDtype})`,
    "with torch.no_grad():",
    "    mean_unembed = model.W_U.mean(dim=-1, keepdim=True, dtype=torch.float32)",
    "    model.W_U.sub_(mean_unembed.to(model.W_U.dtype))",
  ] : [`model = HookedTransformer.from_pretrained(${JSON.stringify(model.repository)})`];
  const interventionLines = !interventionComponent || !hookName || !intervention ? [] : intervention.kind === "activation_patch" ? [
    "",
    `source_prompt = ${JSON.stringify(prompt)}`,
    `destination_prompt = ${JSON.stringify(run?.prompt ?? prompt)}`,
    "source_tokens = model.to_tokens(source_prompt)",
    "destination_tokens = model.to_tokens(destination_prompt)",
    "_, source_cache = model.run_with_cache(source_tokens)",
    `hook_name = ${JSON.stringify(hookName)}`,
    ...indexLines,
    `patch_mappings = ${JSON.stringify(intervention.patchMappings.map((mapping) => [mapping.sourcePosition, mapping.destinationPosition]))}`,
    "",
    `def patch_${interventionComponent}(result, hook):`,
    "    result = result.clone()",
    "    for source_pos, destination_pos in patch_mappings:",
    `        ${patchDestination} = ${patchSource}`,
    "    return result",
    "",
    "patched_logits = model.run_with_hooks(",
    `    destination_tokens, fwd_hooks=[(hook_name, patch_${interventionComponent})]`,
    ")",
  ] : [
    "",
    `hook_name = ${JSON.stringify(hookName)}`,
    ...indexLines,
    `positions = ${JSON.stringify(positions)}`,
    ...(intervention.kind === "mean_ablation" ? [`mean_result = ${meanReference}`] : []),
    "",
    `def ${intervention.kind === "mean_ablation" ? `mean_ablate_${interventionComponent}` : `zero_${interventionComponent}`}(result, hook):`,
    "    result = result.clone()",
    intervention.kind === "mean_ablation" ? `    ${ablationDestination} = ${meanValue}` : `    ${ablationDestination} = 0`,
    "    return result",
    "",
    "intervened_logits = model.run_with_hooks(",
    `    tokens, fwd_hooks=[(hook_name, ${intervention.kind === "mean_ablation" ? `mean_ablate_${interventionComponent}` : `zero_${interventionComponent}`})]`,
    ")",
  ];
  const lines = [
    "from transformer_lens import HookedTransformer",
    "import torch",
    "",
    ...modelLoadLines,
    "model.set_use_attn_result(True)",
    `prompt = ${JSON.stringify(prompt)}`,
    "tokens = model.to_tokens(prompt)",
    "",
    "# Clean run and reusable activation cache",
    "logits, cache = model.run_with_cache(tokens)",
    selectedHead ? `pattern = cache[${JSON.stringify(`blocks.${selectedHead.layer}.attn.hook_pattern`)}][0, ${selectedHead.head}]` : "# Select a head to generate its attention lookup",
    ...interventionLines,
    architecture ? `# ${architecture.nLayers} layers × ${architecture.nHeads} heads · ${architecture.dModel}-wide residual stream` : "# Load a model to inspect its architecture",
  ];
  return <div className="code-panel"><header><span>Equivalent code <small>(TransformerLens)</small></span><button onClick={() => void navigator.clipboard?.writeText(lines.join("\n"))}><Code2 />Copy Python</button></header><pre>{lines.map((line, index) => <div key={`${index}-${line}`}><span>{index + 1}</span><code>{line}</code></div>)}</pre></div>;
}

function EvidenceEmpty({ tab, selected }: { tab: PanelTab; selected: ComponentNode | null }) {
  return <div className="evidence-empty"><div className="empty-evidence-icon">{tab === "Attention" ? <Sparkles /> : tab === "Tokens" ? <Braces /> : <Crosshair />}</div><div><h3>{tab}{selected ? ` · ${displayName(selected)}` : ""}</h3><p>Run a prompt to inspect model behavior.</p></div><button disabled><Play />Run required</button></div>;
}
function EvidenceLoading({ message = "Reading cached model evidence…" }: { message?: string }) { return <div className="evidence-state"><span className="spinner dark" /><p>{message}</p></div>; }
function EvidenceError({ message }: { message: string }) { return <div className="evidence-state error"><Info /><div><strong>Evidence unavailable</strong><p>{message}</p></div></div>; }
function EvidencePrompt({ message }: { message: string }) { return <div className="evidence-prompt"><Activity /><p>{message}</p></div>; }

function StatusBar({ architecture, runtime, activeRun }: { architecture: ArchitectureGraph | null; runtime: RuntimeStatus | null; activeRun: RunRecord | null }) {
  const values = architecture ? [["Model", architecture.modelId], ["Layers", architecture.nLayers], ["Heads", architecture.nHeads], ["d_model", architecture.dModel], ["d_head", architecture.dHead], ["Vocab", architecture.vocabularySize]] : [["Model", "not loaded"]];
  return <footer className="status-bar"><div>{values.map(([label, value]) => <span key={label}><small>{label}:</small><code>{value}</code></span>)}</div><div><span><small>Device:</small><code>{runtime?.device || "—"}</code></span><span><small>Precision:</small><code>{runtime?.dtype || "—"}</code></span><span><small>Cache:</small><code>{formatBytes(runtime?.cacheBytes ?? 0)}</code></span><span><small>Time:</small><code>{activeRun ? `${activeRun.durationMs.toFixed(0)} ms` : "—"}</code></span><span className={`backend-state ${runtime?.backend === "ready" ? "ready" : ""}`}><i />{runtime?.backend === "ready" ? "Backend ready" : "Backend offline"}</span></div></footer>;
}

function ContextMenu(props: { menu: { x: number; y: number; node: ComponentNode }; run: RunRecord | null; canPatch: boolean; onClose: () => void; onInspect: () => void; onExpand: () => void; onAblate: () => void; onPatch: () => void; onAction: (message: string) => void }) {
  const { menu, run, canPatch, onClose, onInspect, onExpand, onAblate, onPatch, onAction } = props;
  const action = (message: string) => { onAction(message); onClose(); };
  const intervenable = menu.node.kind === "head" || menu.node.kind === "mlp";
  return <div className="context-menu" style={{ left: Math.min(menu.x, window.innerWidth - 220), top: Math.min(menu.y, window.innerHeight - 290) }} role="menu"><header><ComponentGlyph kind={menu.node.kind} /><span>{displayName(menu.node)}</span></header><button onClick={onInspect}><Eye />Inspect evidence</button><button onClick={onExpand}><Maximize2 />Expand internals</button><button disabled={!run || !intervenable} onClick={onAblate}><CircleDot />Zero ablate</button><button disabled={!canPatch || !intervenable} onClick={onPatch}><SquareDashedMousePointer />Patch clean → corrupted</button><hr /><button onClick={() => action("Use Save Group in the inspector to name and reuse this selection.")}><Save />Save selection</button></div>;
}

function ExperimentDialog(props: {
  cleanPrompt: string;
  corruptedPrompt: string;
  targetToken: string;
  distractorToken: string;
  setCleanPrompt: (value: string) => void;
  setCorruptedPrompt: (value: string) => void;
  setTargetToken: (value: string) => void;
  setDistractorToken: (value: string) => void;
  running: boolean;
  loaded: boolean;
  onClose: () => void;
  onRun: () => void;
}) {
  const { cleanPrompt, corruptedPrompt, targetToken, distractorToken, setCleanPrompt, setCorruptedPrompt, setTargetToken, setDistractorToken, running, loaded, onClose, onRun } = props;
  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><section className="model-dialog experiment-dialog" role="dialog" aria-modal="true" aria-labelledby="experiment-dialog-title"><header><div><h2 id="experiment-dialog-title">Contrast experiment</h2><p>Define the clean source, corrupted destination, and the output metric used by patching and sweeps.</p></div><button aria-label="Close" onClick={onClose}><X /></button></header><div className="prompt-pair"><label><span>Clean source prompt</span><textarea value={cleanPrompt} onChange={(event) => setCleanPrompt(event.target.value)} rows={3} /></label><span className="pair-arrow">→</span><label><span>Corrupted destination prompt</span><textarea value={corruptedPrompt} onChange={(event) => setCorruptedPrompt(event.target.value)} rows={3} /></label></div><div className="metric-fields"><label><span>Target token</span><input value={targetToken} onChange={(event) => setTargetToken(event.target.value)} placeholder=" Paris" /><small>Include the leading space when the tokenizer expects one.</small></label><label><span>Distractor token <small>optional</small></span><input value={distractorToken} onChange={(event) => setDistractorToken(event.target.value)} placeholder=" Berlin" /><small>Metric = target logit − distractor logit.</small></label></div><div className="model-format-note"><Info /><p>Metric labels must each encode to exactly one model token. Token alignment is computed after both prompts run and remains visible for review.</p></div><footer><button onClick={onClose}>Cancel</button><button className="primary" disabled={running || !cleanPrompt.trim() || !corruptedPrompt.trim() || !targetToken} onClick={onRun}>{running ? <span className="spinner" /> : <FlaskConical />}{loaded ? "Run contrast" : "Load model"}</button></footer></section></div>;
}

function NameDialog({ title, description, action, onClose, onSubmit }: { title: string; description: string; action: string; onClose: () => void; onSubmit: (name: string) => void }) {
  const [name, setName] = useState("");
  const submit = () => { if (name.trim()) onSubmit(name.trim()); };
  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><section className="model-dialog name-dialog" role="dialog" aria-modal="true" aria-label={title}><header><div><h2>{title}</h2><p>{description}</p></div><button aria-label="Close" onClick={onClose}><X /></button></header><label><span>Name</span><input autoFocus value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => event.key === "Enter" && submit()} /></label><footer><button onClick={onClose}>Cancel</button><button className="primary" disabled={!name.trim()} onClick={submit}><Save />{action}</button></footer></section></div>;
}

function SelectionDialog({ nodes, onRemove, onClear, onClose }: { nodes: ComponentNode[]; onRemove: (id: string) => void; onClear: () => void; onClose: () => void }) {
  const [query, setQuery] = useState("");
  const visible = nodes.filter((node) => `${displayName(node)} ${node.id}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><section className="model-dialog selection-dialog" role="dialog" aria-modal="true" aria-labelledby="selection-dialog-title"><header><div><h2 id="selection-dialog-title">Current selection</h2><p>{nodes.length} components. Shift-, Ctrl-, or Command-click components to toggle them.</p></div><button aria-label="Close" onClick={onClose}><X /></button></header><label className="selection-search"><span>Filter</span><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Layer, component, or canonical ID" /></label><div className="selection-dialog-list">{visible.map((node) => <div key={node.id}><ComponentGlyph kind={node.kind} /><span><strong>{displayName(node)}</strong><code>{node.id}</code></span><button aria-label={`Remove ${displayName(node)}`} onClick={() => onRemove(node.id)}><X /></button></div>)}{!visible.length && <p>No selected components match this filter.</p>}</div><footer><button onClick={onClear} disabled={!nodes.length}>Clear all</button><button className="primary" onClick={onClose}>Done</button></footer></section></div>;
}

function ModelDialog({ onClose, onRegister }: { onClose: () => void; onRegister: (source: string, displayName?: string) => Promise<void> }) {
  const [source, setSource] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async () => {
    if (!source.trim()) return;
    setSubmitting(true); setError(null);
    try { await onRegister(source.trim(), displayName.trim() || undefined); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Model registration failed"); }
    finally { setSubmitting(false); }
  };
  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}><section className="model-dialog" role="dialog" aria-modal="true" aria-labelledby="model-dialog-title"><header><div><h2 id="model-dialog-title">Add a TransformerLens model</h2><p>Use a supported model name or a local Hugging Face-format directory.</p></div><button aria-label="Close" onClick={onClose}><X /></button></header><label><span>Model source</span><input autoFocus value={source} onChange={(event) => setSource(event.target.value)} placeholder="distilgpt2 or EleutherAI/pythia-70m" /></label><label><span>Display name <small>optional</small></span><input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="My research model" /></label><div className="model-format-note"><Info /><p>Kannaadi delegates loading and hooks to TransformerLens. A standalone weight file is not enough; local models need configuration, tokenizer, and weights in a compatible directory.</p></div>{error && <div className="dialog-error">{error}</div>}<footer><button onClick={onClose}>Cancel</button><button className="primary" disabled={!source.trim() || submitting} onClick={() => void submit()}>{submitting ? <span className="spinner" /> : <Plus />}Add model</button></footer></section></div>;
}

function PanelResizer({ label, direction, onDelta, onReset }: { label: string; direction: "vertical" | "horizontal"; onDelta: (delta: number) => void; onReset: () => void }) {
  const start = useRef<{ x: number; y: number } | null>(null);
  return <div className={`panel-resizer ${direction}`} role="separator" aria-label={label} aria-orientation={direction} onDoubleClick={onReset} onPointerDown={(event) => { start.current = { x: event.clientX, y: event.clientY }; event.currentTarget.setPointerCapture(event.pointerId); }} onPointerMove={(event) => { if (!start.current) return; const delta = direction === "vertical" ? event.clientX - start.current.x : event.clientY - start.current.y; start.current = { x: event.clientX, y: event.clientY }; onDelta(delta); }} onPointerUp={() => { start.current = null; }} />;
}

function ToolButton({ label, active = false, onClick, children }: { label: string; active?: boolean; onClick: () => void; children: ReactNode }) { return <button className={active ? "active" : ""} aria-label={label} title={label} onClick={onClick}>{children}</button>; }
function ActionButton({ icon, label, onClick, wide = false, disabled = false }: { icon: ReactNode; label: string; onClick: () => void; wide?: boolean; disabled?: boolean }) { return <button className={wide ? "wide" : ""} disabled={disabled} onClick={onClick}>{icon}{label}</button>; }

function ComponentGlyph({ kind, large = false }: { kind: ComponentNode["kind"]; large?: boolean }) {
  return <span className={`component-glyph ${kind} ${large ? "large" : ""}`}>{kind === "head" ? <Sparkles /> : kind === "mlp" ? <Layers3 /> : kind === "residual" ? <RotateCcw /> : kind === "normalization" ? <SlidersHorizontal /> : kind === "operation" || kind === "activation" ? <Activity /> : <Cpu />}</span>;
}

export function KannaadiLogo() { return <svg className="kannaadi-logo" viewBox="0 0 48 48" aria-hidden="true"><g transform="rotate(-35 24 24)" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round"><circle cx="24" cy="17" r="10.5" /><path d="M24 27.5v15" /><path d="M20.5 42.5h7" /></g></svg>; }

function indexArchitecture(architecture: ArchitectureGraph | null): Map<string, ComponentNode> {
  const map = new Map<string, ComponentNode>();
  const add = (node: ComponentNode | null | undefined) => { if (!node) return; map.set(node.id, node); node.children.forEach(add); };
  if (!architecture) return map;
  add(architecture.embedding); add(architecture.positionalEmbedding); add(architecture.finalNorm); add(architecture.unembedding);
  architecture.layers.forEach((layer) => [layer.residualPre, layer.norm1, layer.attention, ...layer.heads, layer.residualMid, layer.norm2, layer.mlp, layer.residualPost].forEach(add));
  return map;
}
function displayName(node: ComponentNode): string {
  if (node.kind === "head") return `L${node.layer}H${node.head}`;
  if (node.layer != null && node.kind === "mlp") return `MLP L${node.layer}`;
  if (node.layer != null && node.kind === "attention") return `Attention L${node.layer}`;
  return node.label;
}
function typeLabel(kind: ComponentNode["kind"]): string { return ({ head: "Attention Head", attention: "Attention", mlp: "MLP", residual: "Residual", normalization: "Normalization", embedding: "Embedding", unembedding: "Unembedding", projection: "Projection", activation: "Activation", operation: "Operation" })[kind]; }
function shapeFor(node: ComponentNode, architecture: ArchitectureGraph | null): string {
  if (!architecture) return "not loaded";
  if (node.kind === "head") return `[batch, position, ${architecture.dModel}]`;
  if (node.kind === "projection" && node.head != null) return `[batch, position, ${architecture.dHead}]`;
  if (node.kind === "unembedding") return `[batch, position, ${architecture.vocabularySize}]`;
  return `[batch, position, ${architecture.dModel}]`;
}
function componentDescription(node: ComponentNode, architecture: ArchitectureGraph | null): string {
  if (node.kind === "head") return `Projected result of attention head ${node.head} in layer ${node.layer}. Its output is written into the shared residual stream.`;
  if (node.kind === "attention") return `Parallel attention operation for layer ${node.layer}, containing ${architecture?.nHeads ?? "the model's"} query heads.`;
  if (node.kind === "mlp") return `Feed-forward transformation for layer ${node.layer}, reading the post-attention residual state.`;
  if (node.kind === "residual") return "A state on the single evolving residual stream.";
  if (node.kind === "unembedding") return "Projects the final residual representation into vocabulary logits.";
  return `${typeLabel(node.kind)} component derived from the loaded model configuration.`;
}
function usePersistentText(key: string, fallback: string): [string, (value: string) => void] {
  const [value, setValue] = useState(() => localStorage.getItem(key) ?? fallback);
  const update = (next: string) => { setValue(next); localStorage.setItem(key, next); };
  return [value, update];
}
function usePersistentNumber(key: string, fallback: number): [number, (value: number) => void] {
  const [value, setValue] = useState(() => Number(localStorage.getItem(key)) || fallback);
  const update = (next: number) => { setValue(next); localStorage.setItem(key, String(next)); };
  return [value, update];
}
function usePersistentJson<T>(key: string, fallback: T): [T, (value: T | ((current: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try { return JSON.parse(localStorage.getItem(key) ?? "null") ?? fallback; }
    catch { return fallback; }
  });
  const update = (next: T | ((current: T) => T)) => setValue((current) => {
    const resolved = typeof next === "function" ? (next as (value: T) => T)(current) : next;
    localStorage.setItem(key, JSON.stringify(resolved));
    return resolved;
  });
  return [value, update];
}
function isTextInput(target: EventTarget | null): boolean { return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement; }
function clamp(value: number, minimum: number, maximum: number): number { return Math.min(maximum, Math.max(minimum, value)); }
function downloadResearchBundle(data: { architecture: ArchitectureGraph | null; model: ModelCatalogEntry; run: RunRecord | null; contrast: ContrastResult | null; effects: Record<string, CausalEffect>; headSweep: HeadSweepResult | null; mlpSweep: MlpSweepResult | null; attribution: AttributionResult | null; selection: string[]; groups: ComponentGroup[] }) {
  const bundle = {
    format: "kannaadi-research-bundle",
    version: 1,
    exportedAt: new Date().toISOString(),
    reproducibility: {
      model: data.model,
      architecture: data.architecture,
      note: "Large cached tensors are intentionally omitted. Re-run the included prompt and interventions against the recorded model to reproduce them.",
    },
    run: data.run,
    contrast: data.contrast,
    causalEffects: data.effects,
    headSweep: data.headSweep,
    mlpSweep: data.mlpSweep,
    directAttribution: data.attribution,
    selection: data.selection,
    groups: data.groups,
  };
  const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `kannaadi-research-${new Date().toISOString().replaceAll(":", "-")}.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
export function calculateFitTransform(viewportWidth: number, viewportHeight: number, contentWidth: number, contentHeight: number, padding = 34): { zoom: number; x: number; y: number } {
  const safeContentWidth = Math.max(contentWidth, 1);
  const safeContentHeight = Math.max(contentHeight, 1);
  const zoom = clamp(Math.min((viewportWidth - padding * 2) / safeContentWidth, (viewportHeight - padding * 2) / safeContentHeight, 1), 0.05, 1);
  return { zoom, x: (viewportWidth - safeContentWidth * zoom) / 2, y: (viewportHeight - safeContentHeight * zoom) / 2 };
}
function formatPercent(value: number): string { return `${(value * 100).toFixed(value >= .1 ? 1 : 2)}%`; }
function formatBytes(value: number): string { if (!value) return "0 B"; const units = ["B", "KB", "MB", "GB"]; const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1); return `${(value / 1024 ** exponent).toFixed(exponent ? 1 : 0)} ${units[exponent]}`; }
function signed(value: number): string { return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`; }
function signedCompact(value: number): string { return `${value >= 0 ? "+" : ""}${Math.abs(value) < .01 ? value.toExponential(1) : value.toFixed(2)}`; }
function causalOverlayStyle(value: number, maximum: number): CSSProperties {
  const strength = Math.min(Math.abs(value) / (maximum || 1), 1);
  return {
    "--effect-color": value >= 0 ? `rgba(70, 129, 95, ${.16 + strength * .72})` : `rgba(207, 75, 57, ${.14 + strength * .72})`,
    "--effect-ink": strength > .58 ? "#fff" : value >= 0 ? "#205537" : "#8f2e23",
  } as CSSProperties;
}
function runMethodLabel(run: RunRecord): string {
  if (run.kind === "clean") return "Clean forward pass";
  if (run.kind === "corrupted") return "Corrupted forward pass";
  if (run.kind === "patched") return "Exact activation patch";
  return run.interventions[0]?.kind === "mean_ablation" ? "Exact mean ablation" : "Exact zero ablation";
}
function visibleToken(value: string): string { return value.replaceAll(" ", "·").replaceAll("\n", "↵") || "∅"; }
function shortComponentId(value: string): string {
  const head = value.match(/^blocks\.(\d+)\.attn\.head\.(\d+)/);
  if (head) return `L${head[1]}H${head[2]}${value.slice(head[0].length)}`;
  const neuron = value.match(/^blocks\.(\d+)\.mlp\.neuron\.(\d+)/);
  if (neuron) return `L${neuron[1]}N${neuron[2]}`;
  const mlp = value.match(/^blocks\.(\d+)\.mlp$/);
  return mlp ? `MLP L${mlp[1]}` : value;
}
