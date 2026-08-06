import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App, calculateFitTransform } from "../src/App";
import type { ArchitectureGraph, ComponentNode, LayerNode, RunRecord } from "../src/types";

function node(id: string, label: string, kind: ComponentNode["kind"], layer?: number, head?: number): ComponentNode {
  return { id, label, kind, layer, head, activationPoints: [`${id}.hook`], metadata: {}, children: [] };
}

function architecture(): ArchitectureGraph {
  const layers: LayerNode[] = Array.from({ length: 6 }, (_, index) => ({
    id: `blocks.${index}`,
    index,
    label: `Layer ${index}`,
    residualPre: node(`blocks.${index}.resid_pre`, "Residual pre", "residual", index),
    norm1: node(`blocks.${index}.ln1`, "LayerNorm", "normalization", index),
    attention: node(`blocks.${index}.attn`, "Multi-Head Attention", "attention", index),
    heads: Array.from({ length: 4 }, (_, head) => node(`blocks.${index}.attn.head.${head}`, `H${head}`, "head", index, head)),
    residualMid: node(`blocks.${index}.resid_mid`, "Residual mid", "residual", index),
    norm2: node(`blocks.${index}.ln2`, "LayerNorm", "normalization", index),
    mlp: { ...node(`blocks.${index}.mlp`, "MLP", "mlp", index), metadata: { activation: "GELU" } },
    residualPost: node(`blocks.${index}.resid_post`, "Residual post", "residual", index),
  }));
  return {
    modelId: "gpt2-small", displayName: "GPT-2 Small", family: "decoder-only",
    nLayers: 6, nHeads: 4, nKeyValueHeads: 4, dModel: 768, dHead: 64, dMlp: 3072,
    vocabularySize: 50257, normType: "LN", normalizationPosition: "pre", blockTopology: "serial",
    positionalMechanism: "standard", embedding: node("embed", "Embedding", "embedding"), positionalEmbedding: node("pos_embed", "Positional embedding", "embedding"), layers,
    finalNorm: node("ln_final", "Final norm", "normalization"), unembedding: node("unembed", "Unembedding", "unembedding"),
  };
}

function mockConnectedBackend() {
  const graph = architecture();
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    let body: unknown = { status: "ok" };
    if (url.endsWith("/api/v1/models")) body = [{ id: "gpt2-small", displayName: "GPT-2 Small", repository: "gpt2-small", architectureFamily: "decoder-only", parameterCount: "124M" }];
    if (url.endsWith("/api/v1/status")) body = { backend: "ready", torchAvailable: true, transformerLensAvailable: true, cudaAvailable: false, device: "cpu", loadedModelId: "gpt2-small", loadedModelName: "GPT-2 Small", loadState: "loaded", loadError: null, runCount: 0, cacheBytes: 0 };
    if (url.endsWith("/architecture")) body = graph;
    if (url.endsWith("/api/v1/runs")) body = [];
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
}

function runRecord(kind: RunRecord["kind"] = "clean"): RunRecord {
  const intervened = kind === "intervened" || kind === "patched";
  return {
    id: kind === "clean" ? "run_clean" : kind === "corrupted" ? "run_corrupted" : kind === "patched" ? "run_patch" : "run_ablation", kind,
    label: kind === "clean" ? "Clean run" : kind === "corrupted" ? "Corrupted destination" : kind === "patched" ? "Patch L5H3" : "Ablate L5H3", status: "complete",
    modelId: "gpt2-small", modelRevision: null, prompt: kind === "corrupted" || kind === "patched" ? "The capital of Germany is" : "The capital of France is",
    tokens: [{ position: 0, tokenId: 464, text: "The", display: "The", nextToken: "·capital", nextTokenProbability: .42 }],
    topPredictions: [{ tokenId: 6342, text: " Paris", display: "·Paris", logit: 8.2, probability: .38 }],
    requestedActivations: ["blocks.5.attn.hook_pattern"],
    interventions: intervened ? [{ kind: kind === "patched" ? "activation_patch" : "zero_ablation", componentIds: ["blocks.5.attn.head.3"], tokenScope: "all", positions: [], sourceRunId: kind === "patched" ? "run_clean" : null, destinationRunId: kind === "patched" ? "run_corrupted" : "run_clean", patchMappings: [], baseline: kind === "patched" ? "source_activation" : "zero" }] : [],
    parentRunId: intervened ? (kind === "patched" ? "run_corrupted" : "run_clean") : null, device: "cpu", dtype: "torch.float32",
    seed: 0, durationMs: 14.2, cacheBytes: 4096, createdAt: new Date().toISOString(),
    provenance: { backend: "transformer_lens", exact: true },
  };
}

function mockResearchBackend() {
  const graph = architecture();
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    let body: unknown = { status: "ok" };
    if (url.endsWith("/api/v1/models")) body = [{ id: "gpt2-small", displayName: "GPT-2 Small", repository: "gpt2-small", architectureFamily: "decoder-only", parameterCount: "124M" }];
    if (url.endsWith("/api/v1/status")) body = { backend: "ready", torchAvailable: true, transformerLensAvailable: true, cudaAvailable: false, device: "cpu", loadedModelId: "gpt2-small", loadedModelName: "GPT-2 Small", loadState: "loaded", loadError: null, runCount: 0, cacheBytes: 0 };
    if (url.endsWith("/architecture")) body = graph;
    if (url.endsWith("/api/v1/runs") && init?.method !== "POST") body = [];
    if (url.endsWith("/api/v1/runs") && init?.method === "POST") body = runRecord();
    if (url.endsWith("/api/v1/contrasts")) body = {
      id: "contrast_test",
      cleanRun: runRecord("clean"),
      corruptedRun: runRecord("corrupted"),
      alignment: {
        sourceRunId: "run_clean", destinationRunId: "run_corrupted", strategy: "minimum_edit_distance",
        exactMatches: 1, sourceLength: 1, destinationLength: 1,
        pairs: [{ sourcePosition: 0, destinationPosition: 0, sourceToken: "The", destinationToken: "The", status: "exact" }],
      },
    };
    if (url.endsWith("/zero-ablate")) body = runRecord("intervened");
    if (url.endsWith("/ablate")) body = { run: runRecord("intervened"), effect: null };
    if (url.includes("/compare/")) body = { baselineRunId: "run_clean", intervenedRunId: "run_ablation", baselineTopToken: " Paris", baselineTopTokenDelta: -1.1, klDivergence: .02, tokens: [{ tokenId: 6342, text: " Paris", display: "·Paris", baselineLogit: 8.2, intervenedLogit: 7.1, deltaLogit: -1.1, baselineProbability: .38, intervenedProbability: .24, deltaProbability: -.14 }] };
    if (url.includes("/residual-stream")) body = {
      runId: "run_clean", position: 0, targetTokenId: 6342, targetToken: " Paris",
      points: Array.from({ length: 6 }, (_, layer) => ["pre", "mid", "post"].map((stage, stageIndex) => ({
        layer, stage, norm: 10 + layer, targetLogit: layer + stageIndex / 3, entropy: 5 - layer / 10,
        topPredictions: [{ tokenId: 6342, text: " Paris", display: "·Paris", logit: layer, probability: .38 }],
      }))).flat(),
    };
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
}

test("expands the model-derived preferred layer and selects its preferred head", async () => {
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  const head = await screen.findByRole("button", { name: "Select L5H3" });
  expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  await user.click(head);
  expect(screen.getByRole("heading", { name: "L5H3" })).toBeInTheDocument();
  expect(screen.getByText("blocks.5.attn.head.3")).toBeInTheDocument();

  fireEvent.contextMenu(head, { clientX: 200, clientY: 200 });
  expect(screen.getByRole("menu")).toHaveTextContent("Zero ablate");
});

test("does not substitute demo architecture when the backend is unavailable", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("connection refused"));
  render(<App />);
  await waitFor(() => expect(screen.getByText("Desktop service is not connected")).toBeInTheDocument());
  expect(screen.queryByRole("button", { name: "Select L5H3" })).not.toBeInTheDocument();
  expect(screen.getByText("connection refused")).toBeInTheDocument();
});

test("reports a model-load failure without mislabeling the healthy desktop service", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/api/v1/models/gpt2-small/load")) {
      return new Response(JSON.stringify({ detail: "Model loading failed: insufficient memory" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
    let body: unknown = { status: "ok" };
    if (url.endsWith("/api/v1/models")) body = [{ id: "gpt2-small", displayName: "GPT-2 Small", repository: "gpt2-small", architectureFamily: "decoder-only", parameterCount: "124M" }];
    if (url.endsWith("/api/v1/status")) body = { backend: "ready", torchAvailable: true, transformerLensAvailable: true, cudaAvailable: false, device: "cpu", dtype: null, loadedModelId: null, loadedModelName: null, loadState: "idle", loadError: null, runCount: 0, cacheBytes: 0 };
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });

  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "Load model" }));

  expect(await screen.findByText("Model could not be loaded")).toBeInTheDocument();
  expect(screen.queryByText("Desktop service is not connected")).not.toBeInTheDocument();
  expect(screen.getByText("Model loading failed: insufficient memory")).toBeInTheDocument();
});

test("runs a real-workflow manifest and compares an exact head ablation", async () => {
  mockResearchBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Run" }));
  expect(await screen.findByText("Prompt tokens")).toBeInTheDocument();
  expect(screen.getAllByText("·Paris").length).toBeGreaterThan(0);

  await user.click(screen.getByRole("button", { name: "Zero Ablate" }));
  expect(await screen.findByText("Baseline versus intervention")).toBeInTheDocument();
  expect(screen.getByText(/Zero ablation can be out of distribution/)).toBeInTheDocument();
});

test("fits the complete expanded diagram instead of resetting to a preset zoom", () => {
  const fitted = calculateFitTransform(1000, 600, 1600, 4000, 40);
  expect(fitted.zoom).toBeCloseTo(.13);
  expect(fitted.x).toBeGreaterThan(0);
  expect(fitted.y).toBeCloseTo(40);

  const small = calculateFitTransform(1000, 600, 300, 200, 40);
  expect(small.zoom).toBe(1);
});

test("supports Ctrl multi-selection and opens the complete selection manager", async () => {
  mockConnectedBackend();
  render(<App />);

  const initiallySelected = await screen.findByRole("button", { name: "Select L5H3" });
  await waitFor(() => expect(initiallySelected).toHaveClass("component-selected"));
  fireEvent.click(screen.getByRole("button", { name: "Select L5H0" }), { ctrlKey: true });
  fireEvent.click(screen.getByRole("button", { name: "Select L5H1" }), { ctrlKey: true });
  fireEvent.click(screen.getByRole("button", { name: "Select L5H2" }), { ctrlKey: true });
  fireEvent.click(screen.getByRole("button", { name: "Expand layer 4" }));
  fireEvent.click(screen.getByRole("button", { name: "Select L4H0" }), { ctrlKey: true });

  fireEvent.click(screen.getByRole("button", { name: "View all 5" }));
  expect(screen.getByRole("heading", { name: "Current selection" })).toBeInTheDocument();
  expect(screen.getByText("5 components. Shift-, Ctrl-, or Command-click components to toggle them.")).toBeInTheDocument();
  expect(screen.getAllByText("blocks.4.attn.head.0").length).toBeGreaterThan(0);
});

test("exposes whole-MLP and neuron controls without an empty head index", async () => {
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByTitle("blocks.5.mlp"));
  expect(screen.getByRole("heading", { name: "MLP L5" })).toBeInTheDocument();
  expect(screen.queryByText("Head Index")).not.toBeInTheDocument();
  expect(screen.getByRole("spinbutton", { name: "Neuron index" })).toHaveValue(0);
  expect(screen.getByRole("button", { name: "Sweep all MLPs" })).toBeDisabled();
});

test("runs a clean and corrupted contrast and exposes its token alignment", async () => {
  mockResearchBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Contrast" }));
  expect(screen.getByRole("heading", { name: "Contrast experiment" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Run contrast" }));

  await user.click(await screen.findByRole("button", { name: "Alignment" }));
  expect(await screen.findByText("Clean ↔ corrupted token alignment")).toBeInTheDocument();
  expect(screen.getByText("Minimum-edit alignment used for activation patch mappings")).toBeInTheDocument();
});

test("saves a durable browser workspace snapshot", async () => {
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: /Save current workspace/i }));
  await user.type(screen.getByRole("textbox", { name: "Name" }), "Geography facts");
  await user.click(screen.getByRole("button", { name: "Save workspace" }));

  const workspace = await screen.findByRole("button", { name: "Geography facts" });
  expect(workspace).toBeInTheDocument();
  expect(localStorage.getItem("kannaadi.workspaces")).toContain("Geography facts");
  await user.click(workspace);
  expect(await screen.findByText(/Workspace “Geography facts” restored/)).toBeInTheDocument();
});

test("zooms the architecture canvas on a trackpad pinch wheel gesture", async () => {
  mockConnectedBackend();
  const { container } = render(<App />);
  await screen.findByRole("button", { name: "Select L5H3" });
  const viewport = container.querySelector(".canvas-viewport")!;
  fireEvent.wheel(viewport, { ctrlKey: true, deltaY: -20, deltaMode: 0, clientX: 400, clientY: 240 });
  await waitFor(() => expect(Number((screen.getByRole("slider", { name: "Zoom" }) as HTMLInputElement).value)).toBeGreaterThan(100));
});

test("manages prompt libraries and persistent dark mode", async () => {
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Manage prompts and collections" }));
  expect(screen.getByRole("heading", { name: "Prompt library" })).toBeInTheDocument();
  await user.type(screen.getByRole("textbox", { name: "Name" }), "Induction example");
  await user.type(screen.getByRole("textbox", { name: "Prompt text" }), "A B A B A");
  await user.click(screen.getByRole("button", { name: "Add prompt" }));
  expect(screen.getAllByText("Induction example").length).toBeGreaterThan(0);
  await user.click(screen.getByRole("button", { name: "Done" }));

  await user.click(screen.getByRole("button", { name: "Switch to dark mode" }));
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(localStorage.getItem("kannaadi.theme")).toBe("dark");
});

test("compares experiment runs and exposes highlighted reproducible code", async () => {
  mockResearchBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Contrast" }));
  await user.click(screen.getByRole("button", { name: "Run contrast" }));
  await user.click(await screen.findByRole("button", { name: "Experiments" }));
  expect(screen.getByText("Experiment runs")).toBeInTheDocument();
  expect(screen.getByText("Compare saved output summaries")).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Code" }));
  expect(document.querySelector(".py-keyword")).toHaveTextContent("from");
});

test("opens a position-aware residual vocabulary readout from the canvas", async () => {
  mockResearchBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Run" }));
  await user.click(screen.getByTitle("blocks.5.resid_pre"));
  expect(await screen.findByText("Residual vocabulary readout")).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "Residual readout position" })).toBeInTheDocument();
  expect(screen.getByText("Top decoded tokens")).toBeInTheDocument();
});

test("restores auditable dataset effects for a frozen prompt-batch series", async () => {
  const baseline = runRecord("clean");
  const intervened = runRecord("intervened");
  localStorage.setItem("kannaadi.experimentSeries", JSON.stringify([{ id: "series_geo", collectionId: "collection_geo", collectionName: "Geography facts", runIds: [baseline.id], createdAt: new Date().toISOString() }]));
  localStorage.setItem("kannaadi.interventionRecipes", JSON.stringify([{ id: "recipe_head", name: "Ablate recall head", kind: "zero_ablation", componentIds: ["blocks.5.attn.head.3"], tokenScope: "all", positions: [], targetToken: " Paris", distractorToken: " Berlin", createdAt: new Date().toISOString() }]));
  localStorage.setItem("kannaadi.datasetAblations", JSON.stringify([{
    id: "dataset_geo", seriesId: "series_geo", seriesName: "Geography facts", recipeId: "recipe_head", recipeName: "Ablate recall head", kind: "zero_ablation", componentIds: ["blocks.5.attn.head.3"], tokenScope: "all", positions: [], metric: { targetToken: " Paris", distractorToken: " Berlin", position: -1 },
    rows: [{ baselineRunId: baseline.id, intervenedRunId: intervened.id, intervenedRun: intervened, label: "France", prompt: baseline.prompt, status: "complete", baselineValue: 2.5, intervenedValue: 1.75, delta: -.75, error: null }],
    summary: { requestedCount: 1, completedCount: 1, failedCount: 0, meanDelta: -.75, medianDelta: -.75, standardDeviation: 0, minimumDelta: -.75, maximumDelta: -.75, meanAbsoluteDelta: .75, directionConsistency: 1 }, durationMs: 20, caveat: "Each row is an exact intervention on one cached prompt run. The aggregate describes this selected prompt set and is not a population-level causal estimate.",
  }]));
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Experiments" }));
  expect(screen.getByText("Dataset effects")).toBeInTheDocument();
  expect(screen.getAllByText("Ablate recall head").length).toBeGreaterThan(0);
  expect(screen.getByText("Direction consistency")).toBeInTheDocument();
  expect(screen.getByText("100.0%")).toBeInTheDocument();
});

test("keeps generated code separate from a persistent non-executing scratchpad", async () => {
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Code" }));
  expect(document.querySelector(".py-keyword")).toHaveTextContent("from");
  await user.click(screen.getByRole("button", { name: /Scratchpad/ }));
  const editor = screen.getByRole("textbox", { name: "Python research scratchpad" });
  await user.type(editor, "# custom research\nprint('ready')");

  expect(editor).toHaveValue("# custom research\nprint('ready')");
  expect(localStorage.getItem("kannaadi.codeScratchpads")).toContain("custom research");
  expect(screen.getByText(/Code execution is intentionally disabled/)).toBeInTheDocument();
});
