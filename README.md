# Kannaadi

Kannaadi is a desktop-first visual workbench for mechanistic interpretability. It combines a native Tauri shell, a React/TypeScript research interface, and a managed local Python sidecar powered by TransformerLens.

## Current milestone

Version 0.6.0 turns Kannaadi into a persistent experiment workbench: reusable prompt libraries and batches, intervention recipes, cross-run comparisons, position-aware residual vocabulary readouts, a low-glare observatory theme, and verified Pythia support now sit alongside the component-level causal tools introduced in 0.5.

- Native Windows window with desktop-owned sidecar startup and shutdown
- Loopback-only FastAPI service on a reserved per-launch port
- Per-launch bearer token passed directly from the native process to the webview
- Real TransformerLens model loading; production UI never falls back to invented architecture data
- Memory-efficient CPU loading with bfloat16, TransformerLens' centered-unembedding logit gauge, and explicit runtime precision
- Truthful stage-based model-loading progress, non-blocking service status, and in-app desktop-service recovery
- Canonical component identities such as `blocks.5.attn.head.3`
- Token and positional embeddings, residual stream, normalization, attention heads, MLPs, final normalization, and unembedding represented from the loaded model configuration
- Progressive component disclosure, including Q/K/V, attention scores, softmax patterns, weighted values, projected results, and MLP internals
- Real tokenization, immutable run manifests, top next-token predictions, and server-side activation caches
- Real attention-pattern and QK-score heatmaps for any selected cached head
- Per-token activation magnitudes and a clickable residual-stream vocabulary readout at pre-attention, post-attention, and post-MLP points, with position/target controls, entropy, norms, and top decoded tokens
- Clean-source and corrupted-destination prompt pairs with deterministic minimum-edit token alignment
- Explicit target-logit or target-minus-distractor logit metrics with strict single-token validation
- Exact zero ablation and within-prompt mean ablation through `hook_result`, at all or selected token positions
- Exact clean-to-corrupted activation patching with persisted source/destination token mappings
- Exact zero/within-prompt-mean ablation and clean-to-corrupted patching for whole MLPs and individual post-activation neurons
- Vectorized whole-model head sweeps: one batched forward pass per layer rather than one pass per head
- Bounded-batch whole-model MLP sweeps, with diverging causal overlays and ranked effects
- Direct logit attribution across embeddings, every attention head, every MLP, and an explicit reconciled remainder
- Named component groups and reusable intervention recipes for repeatable multi-component experiments
- Prompt libraries, prompt collections, sequential batch execution, custom run labels, and a cross-run experiment table
- Frozen prompt-batch series and exact collection-scale ablation experiments with per-prompt evidence, partial-failure reporting, mean/median/spread, effect extrema, mean absolute effect, and directional consistency
- Native versioned workspace snapshots for registered models, prompt libraries, metric setup, recipes, batch-series membership, dataset results, run/result summaries, code scratchpads, selections, expanded components, viewport, theme, and panel layout
- Downloadable JSON research bundles containing architecture, run provenance, interventions, sweeps, attribution, selection, and groups
- Syntax-highlighted generated TransformerLens Python plus a separate persistent, editable, copyable, and exportable research scratchpad that is never executed silently
- Geometry-based Fit all, 5–200% zoom, Chromium and WebView gesture-based trackpad pinch zoom, two-finger panning, and resizable persisted panels
- Persistent light mode and a restrained sci-fi dark observatory theme
- Ctrl/Command/Shift multi-selection, box selection, and a searchable complete-selection manager
- Explicit, model-faithful input and output boundaries: tokenizer outside the model, embeddings summed once into the initial residual, then final normalization and unembedding
- Registration of another TransformerLens model name or compatible local Hugging Face-format directory
- Transformers 5 compatibility shims and real-model validation for `EleutherAI/pythia-70m`

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
backend\.venv\Scripts\python backend\scripts\validate_research.py --model gpt2-small
```

## Architecture

```text
Tauri desktop process
  ├─ native application window (React + TypeScript)
  └─ managed Python child process
       └─ FastAPI + TransformerLens + PyTorch
```

The sidecar listens only on `127.0.0.1`. The native process reserves a free port, creates a fresh 48-character token, starts Python without a console window, streams logs, detects crashes, and terminates the child when the app exits. The webview receives the connection information through a Tauri command instead of a file or a fixed secret.

The frontend consumes normalized domain objects and does not infer architecture from the selected model name. Head IDs remain canonical even though TransformerLens stores head outputs in a shared hook tensor; each head records its slice metadata. On Windows, native workspace files are stored under `%APPDATA%\io.github.sal-faris.tensor-kannaadi\workspaces`; the browser-only development fallback uses local storage.

## Repository layout

- `src/` — desktop workbench UI and typed sidecar client
- `src-tauri/` — native shell and sidecar lifecycle manager
- `backend/kannaadi/` — API, normalized domain model, and TransformerLens adapter
- `backend/scripts/validate_research.py` — opt-in real-model research smoke test
- `tests/` — frontend interaction and honest-state tests
- `backend/tests/` — API, authentication, and architecture-contract tests

## Known scope boundaries

- Interactive tensors are currently limited to 512 prompt tokens and live in memory for the current desktop session. Versioned workspace files persist model registrations, experiment definitions, and numeric result summaries; live activation tensors are deliberately recomputed after restart.
- Current causal interventions target attention-head results, whole MLP outputs, and post-activation MLP neurons. Collection-scale ablation statistics are available; path patching, gradients, attribution patching, paired-collection activation patching, and whole-model dataset sweeps remain future workflows.
- The normalized architecture renderer currently targets TransformerLens-compatible decoder-only models with attention/MLP blocks. Unsupported or unusual architectures may load in TransformerLens but still need a dedicated Kannaadi architecture adapter.
- Model registration accepts TransformerLens identifiers and compatible local Hugging Face directories; a standalone weight file cannot describe enough topology to load safely.
- TransformerLens 3.x is introducing `TransformerBridge`. Kannaadi keeps backend-specific behavior behind one adapter so the loading path can migrate without changing experiment manifests or the frontend.
- Arbitrary custom-script execution is not enabled. The editable scratchpad is deliberately non-executing, locally persistent, and exportable so the desktop app cannot silently run untrusted Python; execution happens in the researcher's chosen Python environment.
- Resumable background jobs, plugin APIs, installer signing, whole-model dataset sweeps, and unusual non-decoder architecture renderers remain future milestones.
