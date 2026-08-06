import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.domain import ArchitectureGraph, ComponentNode, ModelSpec


def graph() -> ArchitectureGraph:
    return TransformerLensAdapter.from_dimensions(
        ModelSpec(id="test", display_name="Test model", backend="transformer_lens"),
        n_layers=2, n_heads=3, d_model=12, vocabulary_size=101,
    )


def test_graph_has_stable_canonical_ids() -> None:
    architecture = graph()
    assert architecture.layers[1].heads[2].id == "blocks.1.attn.head.2"
    assert architecture.layers[1].mlp.id == "blocks.1.mlp"
    assert architecture.layers[1].residual_mid.id == "blocks.1.resid_mid"
    assert architecture.layers[1].heads[2].children[0].id == "blocks.1.attn.head.2.q"
    assert architecture.layers[1].heads[2].children[-1].activation_points == ["blocks.1.attn.hook_result"]
    assert architecture.positional_embedding is not None
    assert architecture.positional_embedding.activation_points == ["hook_pos_embed"]
    assert architecture.unembedding.activation_points == ["unembed.hook_in", "unembed.hook_out"]
    assert architecture.d_head == 4
    assert architecture.normalization_position == "pre"
    assert architecture.model_dump(by_alias=True)["nLayers"] == 2


def test_graph_rejects_head_count_mismatch() -> None:
    architecture = graph().model_dump(by_alias=True)
    architecture["layers"][0]["heads"].pop()
    with pytest.raises(ValidationError, match="nHeads"):
        ArchitectureGraph.model_validate(architecture)


def test_head_requires_coordinates() -> None:
    with pytest.raises(ValidationError, match="indices"):
        ComponentNode(id="bad", label="Bad head", kind="head")


@pytest.mark.parametrize(
    ("dtype", "expected_loader"),
    [("float32", "processed"), ("bfloat16", "no_processing")],
)
def test_adapter_uses_memory_efficient_loader_for_reduced_precision(
    monkeypatch: pytest.MonkeyPatch, dtype: str, expected_loader: str
) -> None:
    calls: list[str] = []
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.float64 = "float64"

    class FakeHookedTransformer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("processed")
            return SimpleNamespace(set_use_attn_result=lambda _enabled: None)

        @classmethod
        def from_pretrained_no_processing(cls, *_args, **_kwargs):
            calls.append("no_processing")
            return SimpleNamespace(set_use_attn_result=lambda _enabled: None)

    fake_transformer_lens = ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = FakeHookedTransformer
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformer_lens", fake_transformer_lens)

    adapter = TransformerLensAdapter()
    adapter.load(
        ModelSpec(
            id="test",
            display_name="Test model",
            backend="transformer_lens",
            repository="test",
            dtype=dtype,
        )
    )

    assert calls == [expected_loader]
