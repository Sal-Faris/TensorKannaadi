# Kannaadi

Kannaadi is a programmable visual workbench for mechanistic interpretability research. This repository contains the first vertical milestone: a typed Python model adapter and architecture API plus a selectable React architecture canvas.

## What works

- Stable canonical component IDs such as `blocks.9.attn.head.6`
- A backend-neutral `ModelAdapter` contract
- A TransformerLens adapter for model loading, tokenization, execution, activation discovery, and architecture normalization
- A metadata-only architecture endpoint that does not silently download a model
- A responsive transformer canvas with semantic layer expansion and component selection
- A right-side inspector showing exact component and activation-point identifiers
- A built-in demo graph when the local backend is not running

## Local setup

### Frontend

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

### Backend

Requires Python 3.11 or newer.

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -e ".[test]"
uvicorn kannaadi.api.app:app --reload --port 8000
```

The frontend automatically reads `GET /api/v1/models/gpt2-small/architecture`. Add the optional model runtime only when you want actual TransformerLens loading:

```bash
pip install -e ".[transformers,test]"
```

Passing `?load=true` to the architecture endpoint performs the real model load. The default metadata route stays deterministic, fast, and network-free.

## Tests

```bash
npm test
cd backend
pytest
```

## Architecture decisions

- The browser consumes normalized domain objects and never depends on TransformerLens hook names beyond their appearance as metadata.
- Head IDs are canonical even though TransformerLens stores all head results in one hook tensor; `metadata.slice` records the head-axis selection.
- Pydantic models reject extra fields and validate graph dimensions to prevent a visually plausible but incorrect graph.
- Real model loading is explicit because downloading weights is a material operation. Browsing known architecture metadata is not.

## Current limitations

- The catalog contains GPT-2 Small only.
- The canvas is architecture-only; result overlays and intervention execution are not connected yet.
- Model instances are not retained between requests yet.
- The current API uses in-process execution and is not a job queue.

## Next milestone

Add a model session service and prompt dataset flow: tokenize clean/corrupted pairs, run cached forward passes, and publish a labelled logit-difference result onto the existing canvas.
