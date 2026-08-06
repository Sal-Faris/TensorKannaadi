from fastapi.testclient import TestClient

from kannaadi.api.app import app


client = TestClient(app)


def test_architecture_endpoint() -> None:
    response = client.get("/api/v1/models/gpt2-small/architecture")
    assert response.status_code == 200
    payload = response.json()
    assert payload["modelId"] == "gpt2-small"
    assert payload["nLayers"] == 12
    assert payload["layers"][0]["heads"][0]["id"] == "blocks.0.attn.head.0"


def test_unknown_model_is_explicit() -> None:
    response = client.get("/api/v1/models/not-a-model/architecture")
    assert response.status_code == 404
