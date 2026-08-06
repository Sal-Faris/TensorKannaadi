import type { ArchitectureGraph, ModelCatalogEntry, RuntimeStatus, SidecarInfo } from "./types";

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

  async status(): Promise<RuntimeStatus> {
    const status = await this.request<Omit<RuntimeStatus, "runtime">>("/api/v1/status");
    return { ...status, runtime: this.runtime };
  }

  models(): Promise<ModelCatalogEntry[]> {
    return this.request("/api/v1/models");
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
