import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.api.app import RuntimeState, app
from kannaadi.domain import (
    AlignmentPair,
    ContrastResult,
    InterventionResult,
    ModelSpec,
    Prediction,
    RunRecord,
    TokenAlignment,
    TokenRecord,
)


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
    run_id = {
        "clean": "run_test",
        "corrupted": "run_corrupted",
        "intervened": "run_ablation",
        "patched": "run_patch",
    }[kind]
    return RunRecord(
        id=run_id,
        kind=kind,
        label={"clean": "Clean run", "corrupted": "Corrupted run", "intervened": "Ablate L1H2", "patched": "Patch L1H2"}[kind],
        modelId="gpt2-small",
        prompt="The capital of Germany is" if kind in {"corrupted", "patched"} else "The capital of France is",
        tokens=[TokenRecord(position=0, tokenId=1, text="The", display="The")],
        topPredictions=[Prediction(tokenId=2, text=" Paris", display="·Paris", logit=4.2, probability=.42)],
        requestedActivations=["blocks.1.attn.hook_pattern"],
        interventions=[] if kind in {"clean", "corrupted"} else [{"kind": "zero_ablation", "componentIds": ["blocks.1.attn.head.2"], "tokenScope": "all"}],
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

    def run_prompt(self, prompt, *, kind="clean", label=None, top_k=10, seed=0):
        assert prompt == "The capital of France is"
        assert kind == "clean"
        assert label is None
        assert top_k == 7
        assert seed == 3
        return self.clean

    def zero_ablate(self, run_id, component_ids, *, token_scope="all", positions=None):
        assert run_id == "run_test"
        assert component_ids == ["blocks.1.attn.head.2"]
        assert token_scope == "all"
        assert positions == []
        return run_record("intervened", "run_test")

    def run_contrast(self, clean_prompt, corrupted_prompt, *, top_k=10, seed=0):
        assert clean_prompt == "The capital of France is"
        assert corrupted_prompt == "The capital of Germany is"
        alignment = TokenAlignment(
            sourceRunId="run_test",
            destinationRunId="run_corrupted",
            pairs=[AlignmentPair(sourcePosition=0, destinationPosition=0, sourceToken="The", destinationToken="The", status="exact")],
            exactMatches=1,
            sourceLength=1,
            destinationLength=1,
        )
        return ContrastResult(id="contrast_test", cleanRun=self.clean, corruptedRun=run_record("corrupted"), alignment=alignment)

    def ablate(self, run_id, component_ids, **kwargs):
        assert kwargs["kind"] == "mean_ablation"
        assert kwargs["positions"] == [0]
        return InterventionResult(run=run_record("intervened", run_id))


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


def test_cpu_model_load_defaults_to_memory_efficient_bfloat16(monkeypatch) -> None:
    captured: dict[str, ModelSpec] = {}

    def fake_load(runtime, spec: ModelSpec):
        captured["spec"] = spec
        runtime.loaded_model_id = spec.id
        runtime.loaded_model_name = spec.display_name
        runtime.device = spec.device
        runtime.dtype = spec.dtype
        runtime.load_state = "loaded"
        return graph()

    monkeypatch.setattr(RuntimeState, "load", fake_load)
    response = client.post("/api/v1/models/gpt2-small/load", json={"device": "cpu"})

    assert response.status_code == 200
    assert captured["spec"].dtype == "bfloat16"
    status = client.get("/api/v1/status").json()
    assert status["dtype"] == "bfloat16"


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


def test_contrast_and_position_scoped_mean_ablation_are_first_class_workflows() -> None:
    app.state.runtime.adapter = LoadedAdapter()
    app.state.runtime.experiments = ExperimentStub()
    contrast = client.post(
        "/api/v1/contrasts",
        json={"cleanPrompt": "The capital of France is", "corruptedPrompt": "The capital of Germany is"},
    )
    assert contrast.status_code == 200
    assert contrast.json()["cleanRun"]["kind"] == "clean"
    assert contrast.json()["corruptedRun"]["kind"] == "corrupted"
    assert contrast.json()["alignment"]["strategy"] == "minimum_edit_distance"

    ablated = client.post(
        "/api/v1/runs/run_test/ablate",
        json={
            "kind": "mean_ablation",
            "componentIds": ["blocks.1.attn.head.2"],
            "tokenScope": "positions",
            "positions": [0],
        },
    )
    assert ablated.status_code == 200
    assert ablated.json()["run"]["kind"] == "intervened"
