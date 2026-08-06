# Kannaadi

Kannaadi is a desktop-first visual workbench for mechanistic interpretability. It combines a native Tauri shell, a React/TypeScript research interface, and a managed local Python sidecar powered by TransformerLens.

## Current milestone

Milestone A establishes the native application and an honest, model-derived architecture view.

- Native Windows window with desktop-owned sidecar startup and shutdown
- Loopback-only FastAPI service on a reserved per-launch port
- Per-launch bearer token passed directly from the native process to the webview
- Real TransformerLens model loading; production UI never falls back to invented architecture data
- Canonical component identities such as `blocks.5.attn.head.3`
- Residual stream, normalization, attention heads, MLPs, and final normalization represented from the loaded model configuration
- Selectable, zoomable, pannable architecture canvas with inspector and context actions
- Clear unavailable and not-yet-implemented states for research results

Prompt execution, activation heatmaps, run comparison, and interventions are Milestone B. Their UI surfaces are intentionally present but do not display fabricated results.

## Run the desktop app

Prerequisites on Windows:

- Node.js 22.13 or newer
- Python 3.11 or newer
- Rust stable with the MSVC toolchain
- Visual Studio C++ Build Tools and Windows 10/11 SDK
- WebView2 runtime

Install the application dependencies:

```powershell
npm install
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -e ".\backend[transformers,test]"
```

Start the native development build:

```powershell
npm run desktop:dev
```

Tauri starts and stops the private Python sidecar automatically. Set `KANNAADI_PYTHON` only when you need to point the shell at a different Python executable.

Build and launch a standalone local executable:

```powershell
npm run desktop:build
.\src-tauri\target\release\kannaadi.exe
```

Use the executable under `target\release`. A binary produced by plain `cargo build` under `target\debug` is a development shell: it expects the Vite server from `npm run dev` to be listening on `127.0.0.1:1420` and will show a connection-refused page when that server is absent.

## Browser-only frontend development

The browser is a frontend development convenience, not the shipping product. Start the API explicitly, then Vite:

```powershell
$env:VITE_KANNAADI_API_TOKEN = "local-development-token-that-is-at-least-32-characters"
$env:KANNAADI_API_TOKEN = $env:VITE_KANNAADI_API_TOKEN
backend\.venv\Scripts\python -m kannaadi.sidecar --host 127.0.0.1 --port 8000 --token $env:KANNAADI_API_TOKEN
npm run dev
```

Open `http://127.0.0.1:1420`.

## Verification

```powershell
npm run build
npm test
backend\.venv\Scripts\python -m pytest backend\tests
cargo check --manifest-path src-tauri\Cargo.toml
```

## Architecture

```text
Tauri desktop process
  ├─ native application window (React + TypeScript)
  └─ managed Python child process
       └─ FastAPI + TransformerLens + PyTorch
```

The sidecar listens only on `127.0.0.1`. The native process reserves a free port, creates a fresh 48-character token, starts Python without a console window, streams logs, detects crashes, and terminates the child when the app exits. The webview receives the connection information through a Tauri command instead of a file or a fixed secret.

The frontend consumes normalized domain objects and does not infer architecture from the selected model name. Head IDs remain canonical even though TransformerLens stores head outputs in a shared hook tensor; each head records its slice metadata.

## Repository layout

- `src/` — desktop workbench UI and typed sidecar client
- `src-tauri/` — native shell and sidecar lifecycle manager
- `backend/kannaadi/` — API, normalized domain model, and TransformerLens adapter
- `tests/` — frontend interaction and honest-state tests
- `backend/tests/` — API, authentication, and architecture-contract tests

## Known scope boundary

Model loading and architecture inspection are real. The Run button and intervention actions are staged for Milestone B; until execution is implemented, Kannaadi shows explanatory empty states rather than demo tensors, logits, or attention values.
