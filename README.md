# Kannaadi

Kannaadi is a desktop-first visual workbench for mechanistic interpretability. It combines a native Tauri shell, a React/TypeScript research interface, and a managed local Python sidecar powered by TransformerLens.

## Current milestone

Version 0.4 adds a practical causal-research workflow on top of the real-model foundation: define a clean/corrupted contrast, review tokenizer alignment, choose an explicit logit metric, patch or ablate selected heads at chosen token positions, and sweep every head in the model.

- Native Windows window with desktop-owned sidecar startup and shutdown
- Loopback-only FastAPI service on a reserved per-launch port
- Per-launch bearer token passed directly from the native process to the webview
- Real TransformerLens model loading; production UI never falls back to invented architecture data
- Canonical component identities such as `blocks.5.attn.head.3`
- Token and positional embeddings, residual stream, normalization, attention heads, MLPs, final normalization, and unembedding represented from the loaded model configuration
- Progressive component disclosure, including Q/K/V, attention scores, softmax patterns, weighted values, projected results, and MLP internals
- Real tokenization, immutable run manifests, top next-token predictions, and server-side activation caches
- Real attention-pattern and QK-score heatmaps for any selected cached head
- Per-token activation magnitudes and a residual-stream logit lens
- Clean-source and corrupted-destination prompt pairs with deterministic minimum-edit token alignment
- Explicit target-logit or target-minus-distractor logit metrics with strict single-token validation
- Exact zero ablation and within-prompt mean ablation through `hook_result`, at all or selected token positions
- Exact clean-to-corrupted activation patching with persisted source/destination token mappings
- Vectorized whole-model head sweeps: one batched forward pass per layer rather than one pass per head
- Diverging causal-effect overlays on expanded heads and a complete layer-by-head sweep heatmap
- Named component groups for repeatable multi-head interventions
- Native workspace snapshots for prompts, metric setup, selections, groups, expanded components, and panel layout
- Generated TransformerLens Python for zero ablation, mean ablation, and activation patching
- Geometry-based Fit all, 5–200% zoom, trackpad pinch zoom, two-finger panning, and resizable persisted panels
- Registration of another TransformerLens model name or compatible local Hugging Face-format directory

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

## Known scope boundaries

- Interactive tensors are currently limited to 512 prompt tokens and live in memory for the current desktop session. Workspace metadata persists, while run tensors are deliberately recomputed after restart.
- Current causal interventions target attention-head results. MLP/neuron patching, path patching, gradients, attribution patching, and dataset-scale statistics remain future workflows.
- The normalized architecture renderer currently targets TransformerLens-compatible decoder-only models with attention/MLP blocks. Unsupported or unusual architectures may load in TransformerLens but still need a dedicated Kannaadi architecture adapter.
- Model registration accepts TransformerLens identifiers and compatible local Hugging Face directories; a standalone weight file cannot describe enough topology to load safely.
- TransformerLens 3.x is introducing `TransformerBridge`. Kannaadi keeps backend-specific behavior behind one adapter so the loading path can migrate without changing experiment manifests or the frontend.
- Arbitrary custom-script execution is not enabled yet; generated TransformerLens code remains read-only so the desktop app does not silently execute untrusted Python. A future trusted/restricted code workflow should make that boundary explicit.
- Dataset runners, resumable background jobs, experiment export bundles, plugins, installer signing, and unusual non-decoder architecture renderers remain future milestones.
