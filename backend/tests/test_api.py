import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from fastapi.testclient import TestClient

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.api.app import RuntimeState, app
from kannaadi.code_execution import CodeSession
from kannaadi.domain import (
    AlignmentPair,
    AttributionEffect,
    AttributionResult,
    ContrastResult,
    DatasetAblationResult,
    DatasetAblationRow,
    DatasetAblationSummary,
    InterventionResult,
    MetricResult,
    ModelSpec,
    MlpEffect,
    MlpSweepResult,
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

    @staticmethod
    def metric(run_id="run_test"):
        return MetricResult(
            runId=run_id,
            metric="logit_difference",
            position=0,
            targetTokenId=2,
            targetToken=" Paris",
            distractorTokenId=3,
            distractorToken=" Berlin",
            value=2.5,
        )

    def mlp_sweep(self, run_id, **kwargs):
        assert run_id == "run_test"
        assert kwargs["kind"] == "zero_ablation"
        return MlpSweepResult(
            runId=run_id,
            kind="zero_ablation",
            metric=self.metric(),
            effects=[MlpEffect(componentId="blocks.1.mlp", layer=1, metricValue=1.75, delta=-.75)],
            minimum=-.75,
            maximum=-.75,
            durationMs=18,
        )

    def direct_attribution(self, run_id, metric):
        assert run_id == "run_test"
        assert metric.target_token == " Paris"
        return AttributionResult(
            runId=run_id,
            metric=self.metric(),
            effects=[AttributionEffect(componentId="blocks.1.mlp", label="MLP L1", kind="mlp", layer=1, value=2.0, fraction=.8)],
            componentSum=2.0,
            remainder=.5,
            minimum=.5,
            maximum=2.0,
            durationMs=12,
        )

    def dataset_ablation(self, run_ids, component_ids, **kwargs):
        assert run_ids == ["run_test"]
        assert component_ids == ["blocks.1.attn.head.2"]
        assert kwargs["kind"] == "zero_ablation"
        assert kwargs["metric"].target_token == " Paris"
        return DatasetAblationResult(
            id="dataset_test",
            kind="zero_ablation",
            componentIds=component_ids,
            tokenScope="all",
            positions=[],
            metric=kwargs["metric"],
            rows=[DatasetAblationRow(
                baselineRunId="run_test",
                intervenedRunId="run_ablation",
                intervenedRun=run_record("intervened", "run_test"),
                label="Clean run",
                prompt="The capital of France is",
                status="complete",
                baselineValue=2.5,
                intervenedValue=1.75,
                delta=-.75,
            )],
            summary=DatasetAblationSummary(
                requestedCount=1,
                completedCount=1,
                failedCount=0,
                meanDelta=-.75,
                medianDelta=-.75,
                standardDeviation=0,
                minimumDelta=-.75,
                maximumDelta=-.75,
                meanAbsoluteDelta=.75,
                directionConsistency=1,
            ),
            durationMs=20,
        )


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
    assert {"loadStage", "loadMessage", "loadElapsedSeconds"} <= status.keys()


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


def test_browser_development_origins_can_reach_the_local_api() -> None:
    response = client.options(
        "/api/v1/status",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


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


def test_mlp_sweep_and_direct_attribution_are_first_class_api_workflows() -> None:
    app.state.runtime.adapter = LoadedAdapter()
    app.state.runtime.experiments = ExperimentStub()

    sweep = client.post(
        "/api/v1/runs/run_test/mlp-sweep",
        json={
            "kind": "zero_ablation",
            "tokenScope": "all",
            "positions": [],
            "metric": {"targetToken": " Paris", "distractorToken": " Berlin", "position": -1},
        },
    )
    assert sweep.status_code == 200
    assert sweep.json()["effects"][0]["componentId"] == "blocks.1.mlp"

    attribution = client.post(
        "/api/v1/runs/run_test/direct-attribution",
        json={"targetToken": " Paris", "distractorToken": " Berlin", "position": -1},
    )
    assert attribution.status_code == 200
    assert attribution.json()["method"] == "direct_logit_attribution_fixed_final_norm"
    assert attribution.json()["componentSum"] + attribution.json()["remainder"] == pytest.approx(2.5)


def test_dataset_ablation_returns_per_prompt_evidence_and_aggregate_statistics() -> None:
    app.state.runtime.adapter = LoadedAdapter()
    app.state.runtime.experiments = ExperimentStub()

    response = client.post(
        "/api/v1/experiments/dataset-ablation",
        json={
            "runIds": ["run_test"],
            "kind": "zero_ablation",
            "componentIds": ["blocks.1.attn.head.2"],
            "tokenScope": "all",
            "positions": [],
            "metric": {"targetToken": " Paris", "distractorToken": " Berlin", "position": -1},
        },
    )
    assert response.status_code == 200
    assert response.json()["rows"][0]["intervenedRunId"] == "run_ablation"
    assert response.json()["rows"][0]["intervenedRun"]["kind"] == "intervened"
    assert response.json()["rows"][0]["delta"] == -.75
    assert response.json()["summary"]["directionConsistency"] == 1
    assert "selected prompt set" in response.json()["caveat"]


def test_code_lab_endpoint_requires_trust_and_returns_structured_output() -> None:
    engine = ExperimentStub()
    adapter = SimpleNamespace(model=SimpleNamespace(name="fake-model"))
    app.state.runtime.adapter = adapter
    app.state.runtime.experiments = engine
    app.state.runtime.code_session = CodeSession(engine, adapter, graph(), "gpt2-small")

    denied = client.post("/api/v1/code/execute", json={"code": "1 + 1", "trusted": False})
    assert denied.status_code == 403

    response = client.post("/api/v1/code/execute", json={
        "code": "print('ready')\nkannaadi.table([{'component': selection[0], 'effect': 1.5}], title='Effects')",
        "cellId": "cell-1",
        "trusted": True,
        "selection": ["blocks.1.mlp"],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["stdout"] == "ready\n"
    assert payload["artifact"]["kind"] == "table"
    assert payload["artifact"]["data"][0]["component"] == "blocks.1.mlp"
    assert client.get("/api/v1/code/session").json()["state"] == "idle"
