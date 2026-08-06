import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.api.app import RuntimeState, app
from kannaadi.domain import Prediction, RunRecord, ModelSpec, TokenRecord


client = TestClient(app)


def graph():
    return TransformerLensAdapter.from_dimensions(
        ModelSpec(id="gpt2-small", display_name="GPT-2 Small", backend="transformer_lens"),
        n_layers=2, n_heads=3, d_model=12, d_head=4, d_mlp=48, vocabulary_size=101,
    )


class LoadedAdapter:
    def architecture(self):
        return graph()


def run_record(kind="clean", parent=None):
    return RunRecord(
        id="run_test" if kind == "clean" else "run_ablation",
        kind=kind,
        label="Clean run" if kind == "clean" else "Ablate L1H2",
        modelId="gpt2-small",
        prompt="The capital of France is",
        tokens=[TokenRecord(position=0, tokenId=1, text="The", display="The")],
        topPredictions=[Prediction(tokenId=2, text=" Paris", display="·Paris", logit=4.2, probability=.42)],
        requestedActivations=["blocks.1.attn.hook_pattern"],
        interventions=[] if kind == "clean" else [{"componentIds": ["blocks.1.attn.head.2"], "tokenScope": "all"}],
        parentRunId=parent,
        device="cpu",
        dtype="torch.float32",
        seed=0,
        durationMs=12.5,
        cacheBytes=1024,
        createdAt=datetime.now(timezone.utc),
        provenance={"backend": "transformer_lens", "exact": True},
    )


class ExperimentStub:
    def __init__(self):
        self.clean = run_record()

    def list_runs(self):
        return [self.clean]

    def run_clean(self, prompt, *, top_k=10, seed=0):
        assert prompt == "The capital of France is"
        assert top_k == 7
        assert seed == 3
        return self.clean

    def zero_ablate(self, run_id, component_ids, *, token_scope="all"):
        assert run_id == "run_test"
        assert component_ids == ["blocks.1.attn.head.2"]
        assert token_scope == "all"
        return run_record("intervened", "run_test")


@pytest.fixture(autouse=True)
def reset_runtime():
    app.state.api_token = None
    app.state.runtime = RuntimeState()
    yield
    app.state.api_token = None
    app.state.runtime = RuntimeState()


def test_architecture_requires_a_real_loaded_model() -> None:
    response = client.get("/api/v1/models/gpt2-small/architecture")
    assert response.status_code == 409
    assert "not loaded" in response.json()["detail"]


def test_architecture_endpoint_returns_loaded_adapter_graph() -> None:
    app.state.runtime.adapter = LoadedAdapter()
    app.state.runtime.loaded_model_id = "gpt2-small"
    response = client.get("/api/v1/models/gpt2-small/architecture")
    assert response.status_code == 200
    payload = response.json()
    assert payload["modelId"] == "gpt2-small"
    assert payload["nLayers"] == 2
    assert payload["layers"][0]["heads"][0]["id"] == "blocks.0.attn.head.0"


def test_unknown_model_is_explicit() -> None:
    response = client.get("/api/v1/models/not-a-model/architecture")
    assert response.status_code == 404


def test_registers_a_transformer_lens_model_source_without_hardcoding_its_shape() -> None:
    response = client.post(
        "/api/v1/models/register",
        json={"source": "EleutherAI/pythia-70m", "displayName": "Pythia 70M"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "eleutherai-pythia-70m"
    assert response.json()["architectureFamily"] == "TransformerLens-compatible"
    assert any(model["id"] == "eleutherai-pythia-70m" for model in client.get("/api/v1/models").json())


def test_rejects_a_standalone_local_weight_file(tmp_path) -> None:
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"not-a-model-directory")
    response = client.post("/api/v1/models/register", json={"source": str(weights)})
    assert response.status_code == 400
    assert "directories" in response.json()["detail"]


def test_desktop_token_protects_api_routes() -> None:
    app.state.api_token = "a" * 48
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 401
    response = client.get("/api/v1/status", headers={"Authorization": f"Bearer {'a' * 48}"})
    assert response.status_code == 200
    assert response.json()["backend"] == "ready"


def test_prompt_run_requires_a_loaded_experiment_engine() -> None:
    response = client.post("/api/v1/runs", json={"prompt": "hello"})
    assert response.status_code == 409
    assert "Load a model" in response.json()["detail"]


def test_prompt_and_zero_ablation_endpoints_return_immutable_run_manifests() -> None:
    app.state.runtime.adapter = LoadedAdapter()
    app.state.runtime.experiments = ExperimentStub()
    clean = client.post(
        "/api/v1/runs",
        json={"prompt": "The capital of France is", "topK": 7, "seed": 3},
    )
    assert clean.status_code == 200
    assert clean.json()["topPredictions"][0]["display"] == "·Paris"

    ablated = client.post(
        "/api/v1/runs/run_test/zero-ablate",
        json={"kind": "zero_ablation", "componentIds": ["blocks.1.attn.head.2"], "tokenScope": "all"},
    )
    assert ablated.status_code == 200
    assert ablated.json()["kind"] == "intervened"
    assert ablated.json()["parentRunId"] == "run_test"
