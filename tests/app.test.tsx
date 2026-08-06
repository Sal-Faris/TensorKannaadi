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

function runRecord(kind: "clean" | "intervened" = "clean"): RunRecord {
  return {
    id: kind === "clean" ? "run_clean" : "run_ablation", kind,
    label: kind === "clean" ? "Clean run" : "Ablate L5H3", status: "complete",
    modelId: "gpt2-small", modelRevision: null, prompt: "The capital of France is",
    tokens: [{ position: 0, tokenId: 464, text: "The", display: "The", nextToken: "·capital", nextTokenProbability: .42 }],
    topPredictions: [{ tokenId: 6342, text: " Paris", display: "·Paris", logit: 8.2, probability: .38 }],
    requestedActivations: ["blocks.5.attn.hook_pattern"],
    interventions: kind === "clean" ? [] : [{ kind: "zero_ablation", componentIds: ["blocks.5.attn.head.3"], tokenScope: "all" }],
    parentRunId: kind === "clean" ? null : "run_clean", device: "cpu", dtype: "torch.float32",
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
    if (url.endsWith("/zero-ablate")) body = runRecord("intervened");
    if (url.includes("/compare/")) body = { baselineRunId: "run_clean", intervenedRunId: "run_ablation", baselineTopToken: " Paris", baselineTopTokenDelta: -1.1, klDivergence: .02, tokens: [{ tokenId: 6342, text: " Paris", display: "·Paris", baselineLogit: 8.2, intervenedLogit: 7.1, deltaLogit: -1.1, baselineProbability: .38, intervenedProbability: .24, deltaProbability: -.14 }] };
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

test("runs a real-workflow manifest and compares an exact head ablation", async () => {
  mockResearchBackend();
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "Run" }));
  expect(await screen.findByText("Prompt tokens")).toBeInTheDocument();
  expect(screen.getAllByText("·Paris").length).toBeGreaterThan(0);

  await user.click(screen.getByRole("button", { name: "Zero Ablate" }));
  expect(await screen.findByText("Clean versus intervention")).toBeInTheDocument();
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
