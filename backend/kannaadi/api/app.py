from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.domain import ArchitectureGraph, ModelSpec

app = FastAPI(title="Kannaadi API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODEL_CATALOG = {
    "gpt2-small": ModelSpec(id="gpt2-small", display_name="GPT-2 Small", backend="transformer_lens", repository="gpt2-small")
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/models/{model_id}/architecture", response_model=ArchitectureGraph, response_model_by_alias=True)
def architecture(model_id: str, load: bool = False) -> ArchitectureGraph:
    spec = MODEL_CATALOG.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    adapter = TransformerLensAdapter()
    if load:
        adapter.load(spec)
        return adapter.architecture()
    # Metadata-only mode keeps architecture browsing fast and avoids an implicit download.
    return adapter.from_dimensions(spec, n_layers=12, n_heads=12, d_model=768, vocabulary_size=50257)
