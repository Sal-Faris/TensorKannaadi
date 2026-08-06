import pytest
from fastapi.testclient import TestClient

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.api.app import RuntimeState, app
from kannaadi.domain import ModelSpec


client = TestClient(app)


def graph():
    return TransformerLensAdapter.from_dimensions(
        ModelSpec(id="gpt2-small", display_name="GPT-2 Small", backend="transformer_lens"),
        n_layers=2, n_heads=3, d_model=12, d_head=4, d_mlp=48, vocabulary_size=101,
    )


class LoadedAdapter:
    def architecture(self):
        return graph()


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


def test_desktop_token_protects_api_routes() -> None:
    app.state.api_token = "a" * 48
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 401
    response = client.get("/api/v1/status", headers={"Authorization": f"Bearer {'a' * 48}"})
    assert response.status_code == 200
    assert response.json()["backend"] == "ready"
