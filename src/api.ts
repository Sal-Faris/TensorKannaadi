import type {
  ActivationSeries,
  ArchitectureGraph,
  AttentionResult,
  AttributionResult,
  ContrastResult,
  HeadSweepResult,
  InterventionResult,
  MetricResult,
  MetricSpec,
  MlpSweepResult,
  ModelCatalogEntry,
  ResidualStreamResult,
  RunComparison,
  RunRecord,
  RuntimeStatus,
  SidecarInfo,
} from "./types";

const BROWSER_API = import.meta.env.VITE_KANNAADI_API_URL || "http://127.0.0.1:8000";
const BROWSER_TOKEN = import.meta.env.VITE_KANNAADI_API_TOKEN || "";

async function desktopSidecar(): Promise<SidecarInfo | null> {
  if (!("__TAURI_INTERNALS__" in window)) return null;
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<SidecarInfo>("sidecar_info");
}

export class KannaadiApi {
  private baseUrl = BROWSER_API;
  private token = BROWSER_TOKEN;
  runtime: "desktop" | "browser" = "browser";

  async connect(): Promise<void> {
    const sidecar = await desktopSidecar();
    if (sidecar) {
      this.baseUrl = sidecar.baseUrl;
      this.token = sidecar.token;
      this.runtime = "desktop";
    }
    let lastError: unknown = null;
    for (let attempt = 0; attempt < (sidecar ? 30 : 1); attempt += 1) {
      try {
        await this.request("/health", { signal: AbortSignal.timeout(1_500) });
        return;
      } catch (error) {
        lastError = error;
        if (!sidecar) break;
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
    }
    throw lastError instanceof Error ? lastError : new Error("Sidecar did not become ready");
  }

  async restartDesktop(): Promise<void> {
    if (!("__TAURI_INTERNALS__" in window)) {
      throw new Error("Automatic service restart is only available in the desktop app");
    }
    const { invoke } = await import("@tauri-apps/api/core");
    const sidecar = await invoke<SidecarInfo>("restart_sidecar");
    this.baseUrl = sidecar.baseUrl;
    this.token = sidecar.token;
    this.runtime = "desktop";
    await this.request("/health", { signal: AbortSignal.timeout(3_000) });
  }

  async status(): Promise<RuntimeStatus> {
    const status = await this.request<Omit<RuntimeStatus, "runtime">>("/api/v1/status");
    return { ...status, runtime: this.runtime };
  }

  models(): Promise<ModelCatalogEntry[]> {
    return this.request("/api/v1/models");
  }

  registerModel(source: string, displayName?: string): Promise<ModelCatalogEntry> {
    return this.request("/api/v1/models/register", {
      method: "POST",
      body: JSON.stringify({ source, displayName }),
    });
  }

  architecture(modelId: string): Promise<ArchitectureGraph> {
    return this.request(`/api/v1/models/${encodeURIComponent(modelId)}/architecture`);
  }

  loadModel(modelId: string, device: string): Promise<ArchitectureGraph> {
    return this.request(`/api/v1/models/${encodeURIComponent(modelId)}/load`, {
      method: "POST",
      body: JSON.stringify({ device }),
    });
  }

  runs(): Promise<RunRecord[]> {
    return this.request("/api/v1/runs");
  }

  run(prompt: string, topK = 10, kind: "clean" | "corrupted" = "clean"): Promise<RunRecord> {
    return this.request("/api/v1/runs", {
      method: "POST",
      body: JSON.stringify({ prompt, kind, topK, seed: 0 }),
    });
  }

  contrast(cleanPrompt: string, corruptedPrompt: string, topK = 10): Promise<ContrastResult> {
    return this.request("/api/v1/contrasts", {
      method: "POST",
      body: JSON.stringify({ cleanPrompt, corruptedPrompt, topK, seed: 0 }),
    });
  }

  zeroAblate(runId: string, componentIds: string[]): Promise<RunRecord> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/zero-ablate`, {
      method: "POST",
      body: JSON.stringify({ kind: "zero_ablation", componentIds, tokenScope: "all" }),
    });
  }

  ablate(
    runId: string,
    componentIds: string[],
    kind: "zero_ablation" | "mean_ablation",
    positions: number[] | null,
    metric?: MetricSpec,
  ): Promise<InterventionResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/ablate`, {
      method: "POST",
      body: JSON.stringify({
        kind,
        componentIds,
        tokenScope: positions ? "positions" : "all",
        positions: positions ?? [],
        metric,
      }),
    });
  }

  patch(
    destinationRunId: string,
    sourceRunId: string,
    componentIds: string[],
    mappings: { sourcePosition: number; destinationPosition: number }[],
    metric?: MetricSpec,
  ): Promise<InterventionResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(destinationRunId)}/patch`, {
      method: "POST",
      body: JSON.stringify({ sourceRunId, componentIds, mappings, metric }),
    });
  }

  metric(runId: string, metric: MetricSpec): Promise<MetricResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/metric`, {
      method: "POST",
      body: JSON.stringify(metric),
    });
  }

  headSweep(
    runId: string,
    kind: "zero_ablation" | "mean_ablation",
    positions: number[] | null,
    metric: MetricSpec,
  ): Promise<HeadSweepResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/head-sweep`, {
      method: "POST",
      body: JSON.stringify({
        kind,
        tokenScope: positions ? "positions" : "all",
        positions: positions ?? [],
        metric,
      }),
    });
  }

  mlpSweep(
    runId: string,
    kind: "zero_ablation" | "mean_ablation",
    positions: number[] | null,
    metric: MetricSpec,
  ): Promise<MlpSweepResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/mlp-sweep`, {
      method: "POST",
      body: JSON.stringify({
        kind,
        tokenScope: positions ? "positions" : "all",
        positions: positions ?? [],
        metric,
      }),
    });
  }

  directAttribution(runId: string, metric: MetricSpec): Promise<AttributionResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/direct-attribution`, {
      method: "POST",
      body: JSON.stringify(metric),
    });
  }

  attention(runId: string, layer: number, head: number): Promise<AttentionResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/attention/${layer}/${head}`);
  }

  activation(runId: string, componentId: string): Promise<ActivationSeries> {
    const query = new URLSearchParams({ component_id: componentId });
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/activation?${query}`);
  }

  residualStream(runId: string): Promise<ResidualStreamResult> {
    return this.request(`/api/v1/runs/${encodeURIComponent(runId)}/residual-stream`);
  }

  compare(baselineRunId: string, intervenedRunId: string): Promise<RunComparison> {
    return this.request(`/api/v1/runs/${encodeURIComponent(baselineRunId)}/compare/${encodeURIComponent(intervenedRunId)}`);
  }

  private async request<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    if (init.body) headers.set("Content-Type", "application/json");
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    const response = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
  }
}
