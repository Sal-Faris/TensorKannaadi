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
