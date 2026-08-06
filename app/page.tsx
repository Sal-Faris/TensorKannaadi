"use client";

import { useEffect, useMemo, useState } from "react";
import "./workbench.css";

type ComponentKind = "embedding" | "attention" | "head" | "mlp" | "unembedding";

type ComponentNode = {
  id: string;
  label: string;
  kind: ComponentKind;
  layer?: number;
  head?: number;
  activationPoints: string[];
};

type LayerNode = {
  id: string;
  index: number;
  label: string;
  attention: ComponentNode;
  heads: ComponentNode[];
  mlp: ComponentNode;
};

type ArchitectureGraph = {
  modelId: string;
  displayName: string;
  family: string;
  nLayers: number;
  nHeads: number;
  dModel: number;
  vocabularySize: number;
  embedding: ComponentNode;
  layers: LayerNode[];
  unembedding: ComponentNode;
};

const demoArchitecture: ArchitectureGraph = {
  modelId: "gpt2-small",
  displayName: "GPT-2 Small",
  family: "decoder-only",
  nLayers: 12,
  nHeads: 12,
  dModel: 768,
  vocabularySize: 50257,
  embedding: { id: "embed", label: "Embedding", kind: "embedding", activationPoints: ["hook_embed"] },
  layers: Array.from({ length: 12 }, (_, layer) => ({
    id: `blocks.${layer}`,
    index: layer,
    label: `Layer ${layer}`,
    attention: { id: `blocks.${layer}.attn`, label: "Attention", kind: "attention", layer, activationPoints: [`blocks.${layer}.attn.hook_pattern`] },
    heads: Array.from({ length: 12 }, (_, head) => ({ id: `blocks.${layer}.attn.head.${head}`, label: `H${head}`, kind: "head" as const, layer, head, activationPoints: [`blocks.${layer}.attn.hook_result`] })),
    mlp: { id: `blocks.${layer}.mlp`, label: "MLP", kind: "mlp", layer, activationPoints: [`blocks.${layer}.hook_mlp_out`] },
  })),
  unembedding: { id: "unembed", label: "Unembedding", kind: "unembedding", activationPoints: ["ln_final.hook_normalized"] },
};

const iconFor = (kind: ComponentKind) => ({ embedding: "E", attention: "A", head: "H", mlp: "M", unembedding: "U" }[kind]);

export default function Home() {
  const [architecture, setArchitecture] = useState(demoArchitecture);
  const [selected, setSelected] = useState<ComponentNode>(demoArchitecture.layers[0].heads[0]);
  const [expandedLayer, setExpandedLayer] = useState<number | null>(3);
  const [zoom, setZoom] = useState(72);
  const [query, setQuery] = useState("");
  const [apiState, setApiState] = useState<"connecting" | "connected" | "demo">("connecting");

  useEffect(() => {
    fetch("http://localhost:8000/api/v1/models/gpt2-small/architecture")
      .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json();
      })
      .then((data: ArchitectureGraph) => {
        setArchitecture(data);
        setSelected(data.layers[0].heads[0]);
        setApiState("connected");
      })
      .catch(() => setApiState("demo"));
  }, []);

  const visibleLayers = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return architecture.layers;
    return architecture.layers.filter((layer) =>
      [layer.id, layer.label, ...layer.heads.map((head) => head.id)].some((value) => value.toLowerCase().includes(normalized)),
    );
  }, [architecture, query]);

  const select = (node: ComponentNode) => setSelected(node);

  return (
    <main className="workbench">
      <header className="topbar">
        <div className="brand-mark">K</div>
        <div className="brand-copy"><strong>Kannaadi</strong><span>Mechanistic interpretability workbench</span></div>
        <div className="topbar-center"><span className="workspace-name">IOI investigation</span><span className="branch-pill">baseline-localization</span></div>
        <button className="ghost-button">⌘ K <span>Command</span></button>
        <button className="primary-button">Run experiment</button>
      </header>

      <section className="shell">
        <aside className="sidebar">
          <div className="sidebar-section">
            <p className="eyebrow">Workspace</p>
            {[["◫", "Model explorer", true], ["◌", "Prompt datasets", false], ["⚗", "Experiments", false], ["⌁", "Candidate circuits", false], ["◇", "Hypotheses", false], ["▤", "Results", false], ["⑂", "Snapshots", false]].map(([icon, label, active]) => (
              <button className={`nav-item ${active ? "active" : ""}`} key={String(label)}><span>{icon}</span>{label}</button>
            ))}
          </div>
          <div className="model-card">
            <div className="model-card-head"><span className="status-dot" /><span>Loaded model</span><button>•••</button></div>
            <strong>{architecture.displayName}</strong>
            <p>{architecture.nLayers} layers · {architecture.nHeads} heads · {architecture.dModel} d_model</p>
            <div className="device-row"><span>TransformerLens</span><span>CPU · float32</span></div>
          </div>
          <div className="sidebar-bottom"><span className={`connection ${apiState}`} />{apiState === "connected" ? "Backend connected" : apiState === "demo" ? "Demo architecture" : "Connecting…"}</div>
        </aside>

        <section className="canvas-panel">
          <div className="canvas-toolbar">
            <div><p className="eyebrow">Architecture</p><h1>{architecture.displayName}</h1></div>
            <label className="search"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find L9H6 or blocks.9…" /></label>
            <div className="zoom-control"><button onClick={() => setZoom(Math.max(45, zoom - 9))}>−</button><span>{zoom}%</span><button onClick={() => setZoom(Math.min(108, zoom + 9))}>+</button></div>
          </div>
          <div className="canvas-meta"><span>CANONICAL GRAPH</span><span>{architecture.nLayers} layers</span><span>{architecture.nHeads * architecture.nLayers} heads</span><span>{architecture.family}</span></div>

          <div className="architecture-scroll">
            <div className="architecture-flow" style={{ transform: `scale(${zoom / 72})`, transformOrigin: "top left" }}>
              <NodeCard node={architecture.embedding} selected={selected.id === architecture.embedding.id} onClick={() => select(architecture.embedding)} />
              <div className="flow-line"><span>residual stream</span></div>
              <div className="layers-stack">
                {visibleLayers.map((layer) => {
                  const open = expandedLayer === layer.index;
                  return (
                    <article className={`layer-card ${open ? "open" : ""}`} key={layer.id}>
                      <button className="layer-heading" onClick={() => setExpandedLayer(open ? null : layer.index)}>
                        <span className="layer-number">{String(layer.index).padStart(2, "0")}</span>
                        <span><strong>{layer.label}</strong><small>{layer.id}</small></span>
                        <span className="layer-dim">{architecture.dModel}d</span><span className="chevron">⌄</span>
                      </button>
                      <div className="layer-body">
                        <button className={`module attention-module ${selected.id === layer.attention.id ? "selected" : ""}`} onClick={() => select(layer.attention)}>
                          <span className="module-icon">A</span><span><strong>Multi-head attention</strong><small>{layer.attention.id}</small></span><span className="module-count">{architecture.nHeads} heads</span>
                        </button>
                        {open && <div className="heads-grid">{layer.heads.map((head) => <button aria-label={`Select ${head.id}`} title={head.id} key={head.id} onClick={() => select(head)} className={selected.id === head.id ? "selected" : ""}>{head.label}</button>)}</div>}
                        <button className={`module mlp-module ${selected.id === layer.mlp.id ? "selected" : ""}`} onClick={() => select(layer.mlp)}>
                          <span className="module-icon">M</span><span><strong>MLP</strong><small>{layer.mlp.id}</small></span><span className="module-count">3072 hidden</span>
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
              <div className="flow-line" />
              <NodeCard node={architecture.unembedding} selected={selected.id === architecture.unembedding.id} onClick={() => select(architecture.unembedding)} />
            </div>
          </div>
          <div className="legend"><span><i className="legend-fill" /> Architecture</span><span><i className="legend-outline" /> Selected component</span><span><i className="legend-overlay" /> Overlay ready</span></div>
        </section>

        <aside className="inspector">
          <div className="inspector-title"><p className="eyebrow">Inspector</p><button>•••</button></div>
          <div className="selected-heading"><span className={`selected-icon ${selected.kind}`}>{iconFor(selected.kind)}</span><div><h2>{selected.kind === "head" ? `Attention head ${selected.layer}.${selected.head}` : selected.label}</h2><code>{selected.id}</code></div></div>
          <div className="scope-badge">Architecture component</div>
          <dl className="property-list">
            <div><dt>Type</dt><dd>{selected.kind}</dd></div>
            {selected.layer !== undefined && <div><dt>Layer</dt><dd>{selected.layer}</dd></div>}
            {selected.head !== undefined && <div><dt>Head index</dt><dd>{selected.head}</dd></div>}
            <div><dt>Width</dt><dd>{selected.kind === "head" ? 64 : architecture.dModel}</dd></div>
          </dl>
          <section className="inspector-section"><h3>Activation points</h3>{selected.activationPoints.map((point) => <code className="activation-point" key={point}>{point}<button aria-label="Copy activation point">□</button></code>)}</section>
          <section className="inspector-section"><h3>Available actions</h3><button className="action-row"><span>⌁</span><div><strong>Inspect activations</strong><small>Open token-level values</small></div><b>›</b></button><button className="action-row"><span>◉</span><div><strong>View attention pattern</strong><small>Compare clean and corrupted</small></div><b>›</b></button><button className="action-row"><span>⚗</span><div><strong>Intervene</strong><small>Ablate, patch, or scale</small></div><b>›</b></button></section>
          <section className="inspector-section evidence"><div><h3>Evidence</h3><button>+ Add</button></div><p>No claims are attached to this component yet.</p></section>
          <div className="provenance"><span>Scope</span><strong>Model architecture</strong><small>No prompt-specific result applied</small></div>
        </aside>
      </section>

      <footer className="bottom-dock"><button className="dock-tab active">Prompt editor</button><button className="dock-tab">Python console</button><button className="dock-tab">Jobs <span>0</span></button><button className="dock-tab">Generated code</button><div className="dock-spacer" /><span>Ready</span><button className="dock-expand">⌃</button></footer>
    </main>
  );
}

function NodeCard({ node, selected, onClick }: { node: ComponentNode; selected: boolean; onClick: () => void }) {
  return <button className={`endpoint-node ${selected ? "selected" : ""}`} onClick={onClick}><span className={`module-icon ${node.kind}`}>{iconFor(node.kind)}</span><span><strong>{node.label}</strong><small>{node.id}</small></span><span className="node-shape">[tokens, {node.kind === "embedding" ? "d_model" : "vocab"}]</span></button>;
}
