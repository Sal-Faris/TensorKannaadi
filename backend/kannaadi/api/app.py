from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.domain import (
    ActivationSeries,
    ArchitectureGraph,
    AttentionResult,
    InterventionSpec,
    ModelSpec,
    ResidualStreamResult,
    RunComparison,
    RunRecord,
    RunRequest,
)
from kannaadi.experiments import ExperimentEngine


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    displayName: str
    repository: str
    architectureFamily: str
    parameterCount: str


class LoadModelRequest(BaseModel):
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] | None = None


class RegisterModelRequest(BaseModel):
    source: str
    displayName: str | None = None


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.adapter: TransformerLensAdapter | None = None
        self.experiments: ExperimentEngine | None = None
        self.loaded_model_id: str | None = None
        self.loaded_model_name: str | None = None
        self.device = "cpu"
        self.load_state: Literal["idle", "loading", "loaded", "error"] = "idle"
        self.load_error: str | None = None

    def load(self, spec: ModelSpec) -> ArchitectureGraph:
        with self.lock:
            self.load_state = "loading"
            self.load_error = None
            try:
                adapter = TransformerLensAdapter()
                adapter.load(spec)
                graph = adapter.architecture()
                self.adapter = adapter
                self.experiments = ExperimentEngine(adapter, graph, spec)
                self.loaded_model_id = spec.id
                self.loaded_model_name = spec.display_name
                self.device = spec.device
                self.load_state = "loaded"
                return graph
            except Exception as exc:
                self.adapter = None
                self.experiments = None
                self.loaded_model_id = None
                self.loaded_model_name = None
                self.load_state = "error"
                self.load_error = str(exc)
                raise


class ApiTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = getattr(request.app.state, "api_token", None)
        if token and request.url.path.startswith("/api/"):
            supplied = request.headers.get("Authorization", "")
            if supplied != f"Bearer {token}":
                return JSONResponse({"detail": "Invalid sidecar authorization token"}, status_code=401)
        return await call_next(request)


app = FastAPI(title="Kannaadi API", version="0.3.0")
app.state.api_token = None
app.state.runtime = RuntimeState()
app.add_middleware(ApiTokenMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://tauri.localhost",
        "tauri://localhost",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

MODEL_CATALOG = {
    "gpt2-small": {
        "spec": ModelSpec(id="gpt2-small", display_name="GPT-2 Small", backend="transformer_lens", repository="gpt2-small"),
        "entry": ModelCatalogEntry(id="gpt2-small", displayName="GPT-2 Small", repository="gpt2-small", architectureFamily="decoder-only", parameterCount="124M"),
    },
}


def dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def cuda_available() -> bool:
    if not dependency_available("torch"):
        return False
    import torch
    return bool(torch.cuda.is_available())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/status")
def status(request: Request) -> dict[str, object]:
    runtime: RuntimeState = request.app.state.runtime
    return {
        "backend": "ready",
        "torchAvailable": dependency_available("torch"),
        "transformerLensAvailable": dependency_available("transformer_lens"),
        "cudaAvailable": cuda_available(),
        "device": runtime.device,
        "loadedModelId": runtime.loaded_model_id,
        "loadedModelName": runtime.loaded_model_name,
        "loadState": runtime.load_state,
        "loadError": runtime.load_error,
        "runCount": len(runtime.experiments.list_runs()) if runtime.experiments else 0,
        "cacheBytes": sum(run.cache_bytes for run in runtime.experiments.list_runs()) if runtime.experiments else 0,
    }


@app.get("/api/v1/models", response_model=list[ModelCatalogEntry])
def list_models() -> list[ModelCatalogEntry]:
    return [item["entry"] for item in MODEL_CATALOG.values()]


@app.post("/api/v1/models/register", response_model=ModelCatalogEntry)
def register_model(payload: RegisterModelRequest) -> ModelCatalogEntry:
    source = payload.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Enter a TransformerLens model name or local Hugging Face directory")
    candidate_path = Path(source).expanduser()
    is_local = candidate_path.exists()
    if is_local and not candidate_path.is_dir():
        raise HTTPException(status_code=400, detail="Local models must be Hugging Face-format directories, not individual weight files")
    canonical_source = str(candidate_path.resolve()) if is_local else source
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", candidate_path.name if is_local else source).strip("-").lower()
    if not slug:
        raise HTTPException(status_code=400, detail="The model source could not be converted into an identifier")
    model_id = slug
    suffix = 2
    while model_id in MODEL_CATALOG and (
        MODEL_CATALOG[model_id]["spec"].repository != canonical_source
        and MODEL_CATALOG[model_id]["spec"].local_path != canonical_source
    ):
        model_id = f"{slug}-{suffix}"
        suffix += 1
    display_name = payload.displayName.strip() if payload.displayName else candidate_path.name if is_local else source.split("/")[-1]
    spec = ModelSpec(
        id=model_id,
        display_name=display_name,
        backend="transformer_lens",
        repository=None if is_local else canonical_source,
        local_path=canonical_source if is_local else None,
    )
    entry = ModelCatalogEntry(
        id=model_id,
        displayName=display_name,
        repository=canonical_source,
        architectureFamily="TransformerLens-compatible",
        parameterCount="unknown",
    )
    MODEL_CATALOG[model_id] = {"spec": spec, "entry": entry}
    return entry


@app.post("/api/v1/models/{model_id}/load", response_model=ArchitectureGraph, response_model_by_alias=True)
async def load_model(model_id: str, payload: LoadModelRequest, request: Request) -> ArchitectureGraph:
    item = MODEL_CATALOG.get(model_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    if payload.device == "cuda" and not cuda_available():
        raise HTTPException(status_code=400, detail="CUDA was requested but is not available")
    if not dependency_available("torch") or not dependency_available("transformer_lens"):
        raise HTTPException(status_code=503, detail="PyTorch and TransformerLens are not installed in the sidecar environment")
    source: ModelSpec = item["spec"]
    spec = source.model_copy(update={
        "device": payload.device,
        "dtype": payload.dtype or ("float16" if payload.device == "cuda" else "float32"),
    })
    runtime: RuntimeState = request.app.state.runtime
    try:
        return await run_in_threadpool(runtime.load, spec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc


@app.get("/api/v1/models/{model_id}/architecture", response_model=ArchitectureGraph, response_model_by_alias=True)
def architecture(model_id: str, request: Request) -> ArchitectureGraph:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    runtime: RuntimeState = request.app.state.runtime
    if runtime.adapter is None or runtime.loaded_model_id != model_id:
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is not loaded")
    return runtime.adapter.architecture()


def experiment_engine(request: Request) -> ExperimentEngine:
    runtime: RuntimeState = request.app.state.runtime
    if runtime.experiments is None or runtime.adapter is None:
        raise HTTPException(status_code=409, detail="Load a model before running an experiment")
    return runtime.experiments


def run_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Experiment failed: {exc}")


@app.get("/api/v1/runs", response_model=list[RunRecord], response_model_by_alias=True)
def list_runs(request: Request) -> list[RunRecord]:
    return experiment_engine(request).list_runs()


@app.post("/api/v1/runs", response_model=RunRecord, response_model_by_alias=True)
async def create_run(payload: RunRequest, request: Request) -> RunRecord:
    engine = experiment_engine(request)
    try:
        return await run_in_threadpool(engine.run_clean, payload.prompt, top_k=payload.top_k, seed=payload.seed)
    except Exception as exc:
        raise run_error(exc) from exc


@app.get("/api/v1/runs/{run_id}", response_model=RunRecord, response_model_by_alias=True)
def get_run(run_id: str, request: Request) -> RunRecord:
    try:
        return experiment_engine(request).get_run(run_id)
    except Exception as exc:
        raise run_error(exc) from exc


@app.post("/api/v1/runs/{run_id}/zero-ablate", response_model=RunRecord, response_model_by_alias=True)
async def zero_ablate(run_id: str, payload: InterventionSpec, request: Request) -> RunRecord:
    engine = experiment_engine(request)
    try:
        return await run_in_threadpool(
            engine.zero_ablate,
            run_id,
            payload.component_ids,
            token_scope=payload.token_scope,
        )
    except Exception as exc:
        raise run_error(exc) from exc


@app.get(
    "/api/v1/runs/{run_id}/attention/{layer}/{head}",
    response_model=AttentionResult,
    response_model_by_alias=True,
)
def attention(run_id: str, layer: int, head: int, request: Request) -> AttentionResult:
    try:
        return experiment_engine(request).attention(run_id, layer, head)
    except Exception as exc:
        raise run_error(exc) from exc


@app.get("/api/v1/runs/{run_id}/activation", response_model=ActivationSeries, response_model_by_alias=True)
def activation(run_id: str, component_id: str, request: Request) -> ActivationSeries:
    try:
        return experiment_engine(request).activation(run_id, component_id)
    except Exception as exc:
        raise run_error(exc) from exc


@app.get(
    "/api/v1/runs/{run_id}/residual-stream",
    response_model=ResidualStreamResult,
    response_model_by_alias=True,
)
async def residual_stream(
    run_id: str,
    request: Request,
    position: int = -1,
    target_token_id: int | None = None,
) -> ResidualStreamResult:
    engine = experiment_engine(request)
    try:
        return await run_in_threadpool(
            engine.residual_stream,
            run_id,
            position=position,
            target_token_id=target_token_id,
        )
    except Exception as exc:
        raise run_error(exc) from exc


@app.get(
    "/api/v1/runs/{baseline_run_id}/compare/{intervened_run_id}",
    response_model=RunComparison,
    response_model_by_alias=True,
)
def compare_runs(baseline_run_id: str, intervened_run_id: str, request: Request) -> RunComparison:
    try:
        return experiment_engine(request).compare(baseline_run_id, intervened_run_id)
    except Exception as exc:
        raise run_error(exc) from exc
