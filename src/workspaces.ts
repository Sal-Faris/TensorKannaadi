import type { WorkspaceSnapshot } from "./types";

const BROWSER_KEY = "kannaadi.workspaces";

type WorkspaceEntry = { name: string };

function isTauri(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

function validateSnapshot(value: unknown): WorkspaceSnapshot {
  if (!value || typeof value !== "object") throw new Error("Workspace data is not an object");
  const snapshot = value as Partial<WorkspaceSnapshot>;
  if (snapshot.format !== "kannaadi-workspace" || ![1, 2, 3].includes(snapshot.version ?? 0) || typeof snapshot.name !== "string") {
    throw new Error("This is not a supported Kannaadi workspace snapshot");
  }
  return snapshot as WorkspaceSnapshot;
}

function browserWorkspaces(): Record<string, WorkspaceSnapshot> {
  try {
    return JSON.parse(localStorage.getItem(BROWSER_KEY) ?? "{}") as Record<string, WorkspaceSnapshot>;
  } catch {
    return {};
  }
}

export async function saveWorkspace(snapshot: WorkspaceSnapshot): Promise<string> {
  if (isTauri()) {
    const { invoke } = await import("@tauri-apps/api/core");
    const entry = await invoke<WorkspaceEntry>("save_workspace", {
      name: snapshot.name,
      contents: JSON.stringify(snapshot, null, 2),
    });
    return entry.name;
  }
  const workspaces = browserWorkspaces();
  workspaces[snapshot.name] = snapshot;
  localStorage.setItem(BROWSER_KEY, JSON.stringify(workspaces));
  return snapshot.name;
}

export async function listWorkspaces(): Promise<string[]> {
  if (isTauri()) {
    const { invoke } = await import("@tauri-apps/api/core");
    return (await invoke<WorkspaceEntry[]>("list_workspaces")).map((entry) => entry.name);
  }
  return Object.keys(browserWorkspaces()).sort((left, right) => left.localeCompare(right));
}

export async function loadWorkspace(name: string): Promise<WorkspaceSnapshot> {
  if (isTauri()) {
    const { invoke } = await import("@tauri-apps/api/core");
    return validateSnapshot(JSON.parse(await invoke<string>("load_workspace", { name })));
  }
  const snapshot = browserWorkspaces()[name];
  if (!snapshot) throw new Error(`Unknown workspace: ${name}`);
  return validateSnapshot(snapshot);
}
