import {
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
  Scan,
  FolderOpen,
  Hand,
  History,
  Info,
  Layers3,
  Maximize2,
  MousePointer2,
  PanelLeftClose,
  Play,
  Plus,
  Redo2,
  RotateCcw,
  Save,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  SquareDashedMousePointer,
  Tag,
  Undo2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  type PointerEvent as ReactPointerEvent,
  type CSSProperties,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { KannaadiApi } from "./api";
import type {
  ArchitectureGraph,
  ComponentNode,
  LayerNode,
  ModelCatalogEntry,
  RuntimeStatus,
} from "./types";

type Tool = "pointer" | "box" | "pan";
type PanelTab = "Tokens" | "Attention" | "QK / OV" | "Logits" | "Activations" | "Residual Stream" | "Code";

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
  const [prompt, setPrompt] = useState("The capital of France is");
  const [loading, setLoading] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const refreshRuntime = useCallback(async () => {
    const nextStatus = await api.status();
    setRuntime(nextStatus);
    if (nextStatus.loadedModelId) {
      setModelId(nextStatus.loadedModelId);
      setArchitecture(await api.architecture(nextStatus.loadedModelId));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await api.connect();
        const [nextModels] = await Promise.all([api.models(), refreshRuntime()]);
        if (!cancelled) setModels(nextModels);
      } catch (error) {
        if (!cancelled) setConnectionError(error instanceof Error ? error.message : "Backend unavailable");
      }
    })();
    return () => { cancelled = true; };
  }, [refreshRuntime]);

  const loadModel = async () => {
    setLoading(true);
    setConnectionError(null);
    setRuntime((current) => current ? { ...current, loadState: "loading", loadError: null } : current);
    try {
      const graph = await api.loadModel(modelId, runtime?.cudaAvailable ? "cuda" : "cpu");
      setArchitecture(graph);
      await refreshRuntime();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Model loading failed";
      setConnectionError(message);
      setRuntime((current) => current ? { ...current, loadState: "error", loadError: message } : current);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <TopBar
        models={models}
        modelId={modelId}
        setModelId={setModelId}
        prompt={prompt}
        setPrompt={setPrompt}
        architecture={architecture}
        runtime={runtime}
        loading={loading}
        onLoad={loadModel}
      />
      <Workbench
        architecture={architecture}
        runtime={runtime}
        model={models.find((model) => model.id === modelId) || DEFAULT_MODEL}
        connectionError={connectionError}
      />
    </main>
  );
}

function TopBar({
  models,
  modelId,
  setModelId,
  prompt,
  setPrompt,
  architecture,
  runtime,
  loading,
  onLoad,
}: {
  models: ModelCatalogEntry[];
  modelId: string;
  setModelId: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  architecture: ArchitectureGraph | null;
  runtime: RuntimeStatus | null;
  loading: boolean;
  onLoad: () => void;
}) {
  return (
    <header className="top-bar">
      <div className="brand" aria-label="Kannaadi">
        <KannaadiLogo />
        <span>Kannaadi</span>
      </div>
      <label className="model-select control-field">
        <Cpu size={15} />
        <select aria-label="Model" value={modelId} onChange={(event) => setModelId(event.target.value)}>
          {models.map((model) => <option key={model.id} value={model.id}>{model.id}</option>)}
        </select>
        <ChevronDown size={14} />
      </label>
      <label className="prompt-field control-field">
        <input aria-label="Prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} spellCheck={false} />
        <kbd>⌘ ↵</kbd>
      </label>
      <button
        className="run-button"
        onClick={onLoad}
        disabled={loading || runtime?.loadState === "loading" || Boolean(architecture)}
        title={architecture ? "Prompt execution is introduced in Milestone B" : "Load the selected model"}
      >
        {loading ? <span className="spinner" /> : architecture ? <Play size={14} fill="currentColor" /> : <Download size={14} />}
        {loading ? "Loading model" : architecture ? "Run" : "Load model"}
      </button>
      <div className="run-mode"><span>Run Mode</span><button><i className="status-dot clean" />Clean Run<ChevronDown size={13} /></button></div>
      <div className="top-actions">
        <button aria-label="Undo" disabled><Undo2 /></button>
        <button aria-label="Redo" disabled><Redo2 /></button>
        <button aria-label="View settings"><SlidersHorizontal /></button>
        <button aria-label="More"><EllipsisVertical /></button>
      </div>
    </header>
  );
}

function Workbench({ architecture, runtime, model, connectionError }: {
  architecture: ArchitectureGraph | null;
  runtime: RuntimeStatus | null;
  model: ModelCatalogEntry;
  connectionError: string | null;
}) {
  const [selection, setSelection] = useState<string[]>([]);
  const [past, setPast] = useState<string[][]>([]);
  const [future, setFuture] = useState<string[][]>([]);
  const [expanded, setExpanded] = useState<Set<number>>(new Set([5]));
  const [activeTab, setActiveTab] = useState<PanelTab>("Attention");
  const [tool, setTool] = useState<Tool>("pointer");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [notice, setNotice] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; node: ComponentNode } | null>(null);

  const nodeIndex = useMemo(() => {
    const map = new Map<string, ComponentNode>();
    if (!architecture) return map;
    map.set(architecture.embedding.id, architecture.embedding);
    map.set(architecture.finalNorm.id, architecture.finalNorm);
    map.set(architecture.unembedding.id, architecture.unembedding);
    for (const layer of architecture.layers) {
      for (const node of [layer.residualPre, layer.norm1, layer.attention, ...layer.heads, layer.residualMid, layer.norm2, layer.mlp, layer.residualPost]) map.set(node.id, node);
    }
    return map;
  }, [architecture]);

  const selectedNodes = selection.map((id) => nodeIndex.get(id)).filter(Boolean) as ComponentNode[];
  const primary = selectedNodes.at(-1) || null;

  const commitSelection = useCallback((next: string[]) => {
    setSelection((current) => {
      if (current.join("|") === next.join("|")) return current;
      setPast((history) => [...history.slice(-29), current]);
      setFuture([]);
      return next;
    });
  }, []);

  const selectNode = useCallback((node: ComponentNode, additive = false) => {
    if (additive) {
      commitSelection(selection.includes(node.id) ? selection.filter((id) => id !== node.id) : [...selection, node.id]);
    } else {
      commitSelection([node.id]);
    }
  }, [commitSelection, selection]);

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
    const handleKey = (event: KeyboardEvent) => {
      const modifier = event.ctrlKey || event.metaKey;
      if (event.key === "Escape") commitSelection([]);
      if (modifier && event.key.toLowerCase() === "a" && architecture) {
        event.preventDefault();
        commitSelection(architecture.layers.flatMap((layer) => layer.heads.map((head) => head.id)));
      }
      if (modifier && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
      if (modifier && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  });

  return (
    <>
      <div className="workspace-grid" onClick={() => contextMenu && setContextMenu(null)}>
        <LeftSidebar selected={selectedNodes} architecture={architecture} />
        <section className="center-stage">
          <ArchitecturePanel
            architecture={architecture}
            error={connectionError}
            selection={selection}
            expanded={expanded}
            setExpanded={setExpanded}
            selectNode={selectNode}
            contextNode={(node, x, y) => setContextMenu({ node, x, y })}
            tool={tool}
            setTool={setTool}
            zoom={zoom}
            setZoom={setZoom}
            pan={pan}
            setPan={setPan}
            commitSelection={commitSelection}
            undo={undo}
            redo={redo}
            canUndo={past.length > 0}
            canRedo={future.length > 0}
          />
          <AnalysisPanel activeTab={activeTab} setActiveTab={setActiveTab} architecture={architecture} selected={primary} model={model} />
        </section>
        <Inspector architecture={architecture} selected={primary} selectedCount={selection.length} onAction={setNotice} />
      </div>
      <StatusBar architecture={architecture} runtime={runtime} />
      {contextMenu && <ContextMenu menu={contextMenu} onClose={() => setContextMenu(null)} onAction={setNotice} />}
      {notice && <div className="toast" role="status"><Info size={15} />{notice}<button aria-label="Dismiss" onClick={() => setNotice(null)}><X /></button></div>}
    </>
  );
}

function LeftSidebar({ selected, architecture }: { selected: ComponentNode[]; architecture: ArchitectureGraph | null }) {
  return (
    <aside className="left-sidebar panel">
      <SidebarSection title="Prompts" icon={<FileJson />}>
        <button className="side-item active">The capital of France is</button>
        <button className="side-item">induction: apples</button>
        <button className="side-item">counterfact: factual error</button>
      </SidebarSection>
      <SidebarSection title="Runs" icon={<History />}>
        <div className="empty-side"><i className="status-dot muted" />No model runs yet</div>
      </SidebarSection>
      <SidebarSection title="Interventions" icon={<Sparkles />}>
        <div className="empty-side">No interventions</div>
      </SidebarSection>
      <SidebarSection title={`Selections (${selected.length})`} icon={<Layers3 />}>
        {selected.slice(-4).map((node) => <button className="side-item selection-item" key={node.id}><ComponentGlyph kind={node.kind} />{displayName(node)}</button>)}
        {selected.length > 4 && <button className="view-all">View all {selected.length}</button>}
        {!selected.length && <div className="empty-side">Select a component</div>}
      </SidebarSection>
      <SidebarSection title="Workspaces" icon={<Database />}>
        <div className="empty-side">No saved investigations</div>
      </SidebarSection>
      <div className="sidebar-spacer" />
      <button className="settings-row"><Settings2 />Settings</button>
      <button className="collapse-sidebar" aria-label="Collapse sidebar"><PanelLeftClose /></button>
      {architecture && <div className="schema-badge">Model-derived schema</div>}
    </aside>
  );
}

function SidebarSection({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return <section className="sidebar-section"><header><span>{title}</span><button aria-label={`Add ${title}`}><Plus /></button></header><div className="sidebar-section-icon">{icon}</div>{children}</section>;
}

function ArchitecturePanel(props: {
  architecture: ArchitectureGraph | null;
  error: string | null;
  selection: string[];
  expanded: Set<number>;
  setExpanded: (value: Set<number>) => void;
  selectNode: (node: ComponentNode, additive?: boolean) => void;
  contextNode: (node: ComponentNode, x: number, y: number) => void;
  tool: Tool;
  setTool: (tool: Tool) => void;
  zoom: number;
  setZoom: (value: number) => void;
  pan: { x: number; y: number };
  setPan: (value: { x: number; y: number }) => void;
  commitSelection: (ids: string[]) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}) {
  const { architecture, error, selection, expanded, setExpanded, selectNode, contextNode, tool, setTool, zoom, setZoom, pan, setPan, commitSelection } = props;
  const viewport = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [box, setBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    const rect = viewport.current!.getBoundingClientRect();
    if (tool === "box") {
      setBox({ x1: event.clientX - rect.left, y1: event.clientY - rect.top, x2: event.clientX - rect.left, y2: event.clientY - rect.top });
    } else if (tool === "pan" || event.target === event.currentTarget) {
      drag.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  };
  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (box) {
      const rect = viewport.current!.getBoundingClientRect();
      setBox({ ...box, x2: event.clientX - rect.left, y2: event.clientY - rect.top });
    } else if (drag.current) {
      setPan({ x: drag.current.panX + event.clientX - drag.current.x, y: drag.current.panY + event.clientY - drag.current.y });
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
        .filter((element) => {
          const rect = element.getBoundingClientRect();
          return rect.right >= left && rect.left <= right && rect.bottom >= top && rect.top <= bottom;
        })
        .map((element) => element.dataset.componentId!)
        .filter((id, index, all) => all.indexOf(id) === index);
      commitSelection(ids);
    }
    setBox(null);
    drag.current = null;
  };

  return (
    <section className="architecture-panel panel">
      <div className="canvas-toolbar">
        <div className="tool-group">
          <span>Select</span>
          <ToolButton label="Pointer" active={tool === "pointer"} onClick={() => setTool("pointer")}><MousePointer2 /></ToolButton>
          <ToolButton label="Box select" active={tool === "box"} onClick={() => setTool("box")}><BoxSelect /></ToolButton>
          <ToolButton label="Pan" active={tool === "pan"} onClick={() => setTool("pan")}><Hand /></ToolButton>
          <ToolButton label="Zoom out" onClick={() => setZoom(Math.max(.55, zoom - .1))}><ZoomOut /></ToolButton>
          <ToolButton label="Zoom in" onClick={() => setZoom(Math.min(1.45, zoom + .1))}><ZoomIn /></ToolButton>
          <button className="fit-button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}><Scan />Fit</button>
        </div>
        <div className="canvas-toolbar-right">
          <label>Group by<select aria-label="Group architecture by"><option>Layer</option></select></label>
          <label className="zoom-slider">Zoom<input aria-label="Zoom" type="range" min="55" max="145" value={zoom * 100} onChange={(event) => setZoom(Number(event.target.value) / 100)} /></label>
          <button className="icon-button" aria-label="Canvas settings"><Settings2 /></button>
          <button className="legend-button"><Layers3 />Legend</button>
        </div>
      </div>
      <div
        ref={viewport}
        className={`canvas-viewport tool-${tool}`}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
      >
        {!architecture ? <CanvasEmpty error={error} /> : (
          <div className="architecture-content" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
            <div className="column-labels"><span>Layer</span><span>Residual Stream</span><span>Multi-Head Attention</span><span>MLP</span><span>Residual Stream</span></div>
            {[...architecture.layers].reverse().map((layer) => (
              <LayerRow
                key={layer.id}
                layer={layer}
                architecture={architecture}
                expanded={expanded.has(layer.index)}
                selected={selection}
                toggle={() => {
                  const next = new Set(expanded);
                  next.has(layer.index) ? next.delete(layer.index) : next.add(layer.index);
                  setExpanded(next);
                }}
                selectNode={selectNode}
                contextNode={contextNode}
              />
            ))}
          </div>
        )}
        {box && <div className="selection-box" style={{ left: Math.min(box.x1, box.x2), top: Math.min(box.y1, box.y2), width: Math.abs(box.x2 - box.x1), height: Math.abs(box.y2 - box.y1) }} />}
      </div>
      <div className="canvas-corner-actions">
        <button onClick={props.undo} disabled={!props.canUndo} aria-label="Undo selection"><Undo2 /></button>
        <button onClick={props.redo} disabled={!props.canRedo} aria-label="Redo selection"><Redo2 /></button>
      </div>
    </section>
  );
}

function LayerRow({ layer, architecture, expanded, selected, toggle, selectNode, contextNode }: {
  layer: LayerNode;
  architecture: ArchitectureGraph;
  expanded: boolean;
  selected: string[];
  toggle: () => void;
  selectNode: (node: ComponentNode, additive?: boolean) => void;
  contextNode: (node: ComponentNode, x: number, y: number) => void;
}) {
  const layerSelected = [layer.id, layer.attention.id, layer.mlp.id, ...layer.heads.map((head) => head.id)].some((id) => selected.includes(id));
  const choose = (node: ComponentNode, event: React.MouseEvent) => {
    event.stopPropagation();
    selectNode(node, event.shiftKey);
  };
  const context = (node: ComponentNode, event: React.MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    contextNode(node, event.clientX, event.clientY);
  };
  return (
    <article className={`layer-row ${expanded ? "expanded" : ""} ${layerSelected ? "layer-selected" : ""}`}>
      <button className="layer-index" onClick={toggle} aria-label={`${expanded ? "Collapse" : "Expand"} layer ${layer.index}`}>
        <strong>{layer.index}</strong>{expanded ? <ChevronDown /> : <ChevronRight />}
      </button>
      <div className="residual-entry"><span className="flow-line" /><button title={layer.residualPre.id} onClick={(event) => choose(layer.residualPre, event)}><Plus /></button></div>
      <div className="attention-stage">
        <button
          data-component-id={layer.attention.id}
          className={`attention-block ${selected.includes(layer.attention.id) ? "component-selected" : ""}`}
          onClick={(event) => choose(layer.attention, event)}
          onContextMenu={(event) => context(layer.attention, event)}
        >
          <span>{expanded ? `Layer ${layer.index}` : "Multi-Head Attention"}</span>
          {expanded && <small>Multi-Head Attention · {architecture.nHeads} heads</small>}
        </button>
        {expanded && <div className="heads-container" style={{ "--heads": architecture.nHeads } as CSSProperties}>
          {layer.heads.map((head) => <button
            key={head.id}
            data-component-id={head.id}
            className={selected.includes(head.id) ? "component-selected" : ""}
            onClick={(event) => choose(head, event)}
            onContextMenu={(event) => context(head, event)}
            aria-label={`Select L${layer.index}H${head.head}`}
          >H{head.head}</button>)}
        </div>}
      </div>
      <div className="residual-add"><span className="flow-line" /><button title={layer.residualMid.id} onClick={(event) => choose(layer.residualMid, event)}><Plus /></button></div>
      <div className="mlp-stage">
        <button
          data-component-id={layer.mlp.id}
          className={`mlp-block ${selected.includes(layer.mlp.id) ? "component-selected" : ""}`}
          onClick={(event) => choose(layer.mlp, event)}
          onContextMenu={(event) => context(layer.mlp, event)}
        >{expanded ? `MLP (${architecture.dMlp ?? "hidden"})` : "MLP"}</button>
        {expanded && <div className="mlp-internals"><span>Linear (in)</span><span>{String(layer.mlp.metadata.activation || "Activation")}</span><span>Linear (out)</span></div>}
      </div>
      <div className="residual-exit"><span className="flow-line" /><button title={layer.residualPost.id} onClick={(event) => choose(layer.residualPost, event)}><Plus /></button><span className="flow-tail" /></div>
      <button className="row-more" onContextMenu={(event) => context(layer.residualPost, event)} aria-label={`More actions for layer ${layer.index}`}><EllipsisVertical /></button>
    </article>
  );
}

function CanvasEmpty({ error }: { error: string | null }) {
  return <div className="canvas-empty"><div className="empty-orbit"><KannaadiLogo /></div><h2>{error ? "Desktop service is not connected" : "Load a model to reveal its architecture"}</h2><p>{error ? "Launch Kannaadi through its desktop entry point, or start the Python service for browser development." : "The canvas will be generated from the model's actual configuration. No demo topology is shown here."}</p>{error && <code>{error}</code>}</div>;
}

function Inspector({ architecture, selected, selectedCount, onAction }: {
  architecture: ArchitectureGraph | null;
  selected: ComponentNode | null;
  selectedCount: number;
  onAction: (message: string) => void;
}) {
  const hook = selected?.activationPoints[0] || "—";
  return (
    <aside className="inspector panel">
      <header className="inspector-bar"><span>Inspector</span><small>{selectedCount ? `${selectedCount} selected` : "No selection"}</small><button aria-label="Close inspector"><X /></button></header>
      {!selected ? <div className="inspector-empty"><Crosshair /><p>Select a component on the architecture canvas.</p></div> : <>
        <div className="inspector-heading"><ComponentGlyph kind={selected.kind} large /><h2>{displayName(selected)}</h2><span className="type-pill">{typeLabel(selected.kind)}</span></div>
        <dl className="inspector-properties">
          <div><dt>Type</dt><dd>{selected.kind === "head" ? "attention.head" : selected.kind}</dd></div>
          {selected.layer !== undefined && <div><dt>Layer</dt><dd>{selected.layer}</dd></div>}
          {selected.head !== undefined && <div><dt>Head Index</dt><dd>{selected.head}</dd></div>}
          <div><dt>Canonical ID</dt><dd><code>{selected.id}</code></dd></div>
          <div><dt>Hook</dt><dd><code>{hook}</code></dd></div>
          {architecture && <><div><dt>Input Dim</dt><dd>{architecture.dModel}</dd></div><div><dt>{selected.kind === "head" ? "Head Dim" : "Output Dim"}</dt><dd>{selected.kind === "head" ? architecture.dHead : architecture.dModel}</dd></div></>}
        </dl>
        <section className="about-component"><h3>About</h3><p>{componentDescription(selected, architecture)}</p><code>Shape: {shapeFor(selected, architecture)}</code></section>
        <section className="inspector-actions"><h3>Actions</h3><div>
          <ActionButton icon={<Eye />} label="Inspect" onClick={() => onAction("Detailed evidence appears in the analysis panel.")} />
          <ActionButton icon={<CircleDot />} label="Zero Ablate" onClick={() => onAction("Zero ablation will be enabled by the first verified intervention workflow.")} />
          <ActionButton icon={<SquareDashedMousePointer />} label="Patch…" onClick={() => onAction("Activation patching requires a source and destination run.")} />
          <ActionButton icon={<Tag />} label="Label…" onClick={() => onAction("Component annotations are planned with workspace snapshots.")} />
          <ActionButton wide icon={<Save />} label="Save Selection" onClick={() => onAction("Selection captured locally for this session.")} />
        </div></section>
        <section className="method-scope"><Info /><div><strong>Architecture evidence</strong><span>Model-derived metadata; no prompt-specific claim.</span></div></section>
      </>}
    </aside>
  );
}

function AnalysisPanel({ activeTab, setActiveTab, architecture, selected, model }: {
  activeTab: PanelTab;
  setActiveTab: (tab: PanelTab) => void;
  architecture: ArchitectureGraph | null;
  selected: ComponentNode | null;
  model: ModelCatalogEntry;
}) {
  const tabs: PanelTab[] = ["Tokens", "Attention", "QK / OV", "Logits", "Activations", "Residual Stream", "Code"];
  return <section className="analysis-panel panel">
    <nav>{tabs.map((tab) => <button className={activeTab === tab ? "active" : ""} key={tab} onClick={() => setActiveTab(tab)}>{tab}</button>)}</nav>
    <div className="analysis-content">
      {activeTab === "Code" ? <CodePanel architecture={architecture} selected={selected} model={model} /> : <EvidenceEmpty tab={activeTab} selected={selected} />}
    </div>
  </section>;
}

function EvidenceEmpty({ tab, selected }: { tab: PanelTab; selected: ComponentNode | null }) {
  return <div className="evidence-empty"><div className="empty-evidence-icon">{tab === "Attention" ? <Sparkles /> : tab === "Tokens" ? <Braces /> : <Crosshair />}</div><div><h3>{tab}{selected ? ` · ${displayName(selected)}` : ""}</h3><p>Run a real model prompt to populate this panel. Kannaadi does not display invented analysis values.</p></div><button disabled><Play />Run required</button></div>;
}

function CodePanel({ architecture, selected, model }: { architecture: ArchitectureGraph | null; selected: ComponentNode | null; model: ModelCatalogEntry }) {
  const lines = [
    "from transformer_lens import HookedTransformer",
    "import torch",
    "",
    `model = HookedTransformer.from_pretrained(\"${model.repository}\")`,
    architecture ? `# Loaded architecture: ${architecture.nLayers} layers × ${architecture.nHeads} heads` : "# Load the model to inspect its derived architecture",
    selected ? `# Selected component: ${selected.id}` : "# Select a component on the canvas",
    selected?.activationPoints[0] ? `hook_name = \"${selected.activationPoints[0]}\"` : "",
  ].filter((line, index, all) => line || all[index - 1]);
  return <div className="code-panel"><header><span>Code <small>(TransformerLens)</small></span><button><Code2 />Python</button></header><pre>{lines.map((line, index) => <div key={`${index}-${line}`}><span>{index + 1}</span><code>{line}</code></div>)}</pre></div>;
}

function StatusBar({ architecture, runtime }: { architecture: ArchitectureGraph | null; runtime: RuntimeStatus | null }) {
  const values = architecture ? [
    ["Model", architecture.modelId], ["Layers", architecture.nLayers], ["Heads", architecture.nHeads], ["d_model", architecture.dModel], ["d_head", architecture.dHead], ["Vocab", architecture.vocabularySize],
  ] : [["Model", "not loaded"]];
  return <footer className="status-bar"><div>{values.map(([label, value]) => <span key={label}><small>{label}:</small><code>{value}</code></span>)}</div><div><span><small>Device:</small><code>{runtime?.device || "—"}</code></span><span><small>Runtime:</small><code>{runtime?.runtime || "connecting"}</code></span><span className={`backend-state ${runtime?.backend === "ready" ? "ready" : ""}`}><i />{runtime?.backend === "ready" ? "Backend ready" : "Backend offline"}</span></div></footer>;
}

function ContextMenu({ menu, onClose, onAction }: { menu: { x: number; y: number; node: ComponentNode }; onClose: () => void; onAction: (message: string) => void }) {
  const action = (message: string) => { onAction(message); onClose(); };
  return <div className="context-menu" style={{ left: Math.min(menu.x, window.innerWidth - 220), top: Math.min(menu.y, window.innerHeight - 290) }} role="menu"><header><ComponentGlyph kind={menu.node.kind} /><span>{displayName(menu.node)}</span></header><button onClick={() => action("Detailed evidence appears in the analysis panel.")}><Eye />Inspect</button><button onClick={() => action("Head internals will expand after real attention inspection is connected.")}><Maximize2 />Expand internals</button><button onClick={() => action("Zero ablation will be enabled by the verified intervention workflow.")}><CircleDot />Zero ablate</button><button onClick={() => action("Patching requires clean and corrupted source runs.")}><SquareDashedMousePointer />Patch…</button><hr /><button onClick={() => action("Selection saved for this session.")}><Save />Save selection</button></div>;
}

function ToolButton({ label, active = false, onClick, children }: { label: string; active?: boolean; onClick: () => void; children: ReactNode }) {
  return <button className={active ? "active" : ""} aria-label={label} title={label} onClick={onClick}>{children}</button>;
}
function ActionButton({ icon, label, onClick, wide = false }: { icon: ReactNode; label: string; onClick: () => void; wide?: boolean }) {
  return <button className={wide ? "wide" : ""} onClick={onClick}>{icon}{label}</button>;
}

function ComponentGlyph({ kind, large = false }: { kind: ComponentNode["kind"]; large?: boolean }) {
  return <span className={`component-glyph ${kind} ${large ? "large" : ""}`}>{kind === "head" ? <Sparkles /> : kind === "mlp" ? <Layers3 /> : kind === "residual" ? <RotateCcw /> : kind === "normalization" ? <SlidersHorizontal /> : <Cpu />}</span>;
}

export function KannaadiLogo() {
  return <svg className="kannaadi-logo" viewBox="0 0 48 48" aria-hidden="true"><g transform="rotate(-35 24 24)" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round"><circle cx="24" cy="17" r="10.5" /><path d="M24 27.5v15" /><path d="M20.5 42.5h7" /></g></svg>;
}

function displayName(node: ComponentNode): string {
  if (node.kind === "head") return `L${node.layer}H${node.head}`;
  if (node.layer !== undefined && node.kind === "mlp") return `MLP L${node.layer}`;
  if (node.layer !== undefined && node.kind === "attention") return `Attention L${node.layer}`;
  return node.label;
}
function typeLabel(kind: ComponentNode["kind"]): string {
  return ({ head: "Attention Head", attention: "Attention", mlp: "MLP", residual: "Residual", normalization: "Normalization", embedding: "Embedding", unembedding: "Unembedding" })[kind];
}
function shapeFor(node: ComponentNode, architecture: ArchitectureGraph | null): string {
  if (!architecture) return "not loaded";
  if (node.kind === "head") return `[batch, pos, ${architecture.dModel}]`;
  return `[batch, pos, ${architecture.dModel}]`;
}
function componentDescription(node: ComponentNode, architecture: ArchitectureGraph | null): string {
  if (node.kind === "head") return `Projected result of attention head ${node.head} in layer ${node.layer}. Its output is written into the shared residual stream.`;
  if (node.kind === "attention") return `Parallel attention operation for layer ${node.layer}, containing ${architecture?.nHeads ?? "the model's"} query heads.`;
  if (node.kind === "mlp") return `Feed-forward transformation for layer ${node.layer}, reading the post-attention residual state.`;
  if (node.kind === "residual") return "A state on the single evolving residual stream.";
  return `${typeLabel(node.kind)} component derived from the loaded model configuration.`;
}
