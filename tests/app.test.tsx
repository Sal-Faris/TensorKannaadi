import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../src/App";
import type { ArchitectureGraph, ComponentNode, LayerNode } from "../src/types";

function node(id: string, label: string, kind: ComponentNode["kind"], layer?: number, head?: number): ComponentNode {
  return { id, label, kind, layer, head, activationPoints: [`${id}.hook`], metadata: {} };
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
    positionalMechanism: "standard", embedding: node("embed", "Embedding", "embedding"), layers,
    finalNorm: node("ln_final", "Final norm", "normalization"), unembedding: node("unembed", "Unembedding", "unembedding"),
  };
}

function mockConnectedBackend() {
  const graph = architecture();
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    let body: unknown = { status: "ok" };
    if (url.endsWith("/api/v1/models")) body = [{ id: "gpt2-small", displayName: "GPT-2 Small", repository: "gpt2-small", architectureFamily: "decoder-only", parameterCount: "124M" }];
    if (url.endsWith("/api/v1/status")) body = { backend: "ready", torchAvailable: true, transformerLensAvailable: true, cudaAvailable: false, device: "cpu", loadedModelId: "gpt2-small", loadedModelName: "GPT-2 Small", loadState: "loaded", loadError: null };
    if (url.endsWith("/architecture")) body = graph;
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
}

test("expands the model-derived preferred layer and selects its preferred head", async () => {
  mockConnectedBackend();
  const user = userEvent.setup();
  render(<App />);

  const head = await screen.findByRole("button", { name: "Select L5H3" });
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
