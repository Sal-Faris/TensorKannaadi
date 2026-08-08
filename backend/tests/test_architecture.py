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


def test_pythia_installs_transformers_five_output_head_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNeoX:
        def get_output_embeddings(self):
            return "language-model-head"

    fake_transformers = ModuleType("transformers")
    fake_transformers.GPTNeoXForCausalLM = FakeNeoX
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    TransformerLensAdapter._install_transformers_compatibility("EleutherAI/pythia-70m")

    assert FakeNeoX().embed_out == "language-model-head"


def test_parallel_blocks_do_not_claim_a_nonexistent_mid_residual_hook() -> None:
    graph = TransformerLensAdapter.from_dimensions(
        ModelSpec(id="parallel", display_name="Parallel", backend="transformer_lens"),
        n_layers=2,
        n_heads=2,
        d_model=16,
        vocabulary_size=100,
        block_topology="parallel",
    )

    assert graph.block_topology == "parallel"
    assert graph.layers[0].residual_mid.id == "blocks.0.parallel_merge"
    assert graph.layers[0].residual_mid.activation_points == []


def test_canonical_flow_has_explicit_directed_serial_residual_adds() -> None:
    architecture = graph()
    modules = {module.id: module for module in architecture.flow.modules}
    edges = {edge.id: edge for edge in architecture.flow.edges}

    assert architecture.flow.schema_version == 1
    assert modules["input.embedding_sum"].metadata["operator"] == "add"
    assert edges["block.0.resid.attn_add"].source_module_id == "block.0.resid_pre"
    assert edges["block.0.attn.add"].target_module_id == "block.0.add_attention"
    assert edges["block.0.mlp.add"].target_module_id == "block.0.add_mlp"
    assert edges["output.norm.unembed"].target_module_id == "output.unembedding"
    assert all(edge.source_module_id != edge.target_module_id for edge in architecture.flow.edges)


def test_parallel_flow_merges_attention_and_mlp_from_the_same_residual() -> None:
    architecture = TransformerLensAdapter.from_dimensions(
        ModelSpec(id="parallel", display_name="Parallel", backend="transformer_lens"),
        n_layers=1,
        n_heads=2,
        d_model=16,
        vocabulary_size=100,
        block_topology="parallel",
        positional_mechanism="rotary",
    )
    edges = {edge.id: edge for edge in architecture.flow.edges}

    assert edges["block.0.resid.ln1"].source_module_id == "block.0.resid_pre"
    assert edges["block.0.resid.ln2"].source_module_id == "block.0.resid_pre"
    assert edges["block.0.resid.merge"].target_module_id == "block.0.merge"
    assert edges["block.0.attn.merge"].target_module_id == "block.0.merge"
    assert edges["block.0.mlp.merge"].target_module_id == "block.0.merge"
    assert edges["block.0.position.attention"].tensor_role == "position_indices"


def test_shortformer_position_embedding_enters_attention_not_initial_residual() -> None:
    architecture = TransformerLensAdapter.from_dimensions(
        ModelSpec(id="shortformer", display_name="Shortformer", backend="transformer_lens"),
        n_layers=1,
        n_heads=2,
        d_model=16,
        vocabulary_size=100,
        positional_mechanism="shortformer",
    )
    edges = {edge.id: edge for edge in architecture.flow.edges}
    modules = {module.id: module for module in architecture.flow.modules}

    assert architecture.positional_embedding is not None
    assert "input.embedding_sum" not in modules
    assert edges["input.initial_residual"].source_module_id == "input.token_embedding"
    assert edges["block.0.position.attention"].source_module_id == "input.position_embedding"


def test_gated_mlp_exposes_both_branches_and_the_real_transformer_lens_hooks() -> None:
    architecture = TransformerLensAdapter.from_dimensions(
        ModelSpec(id="gated", display_name="Gated", backend="transformer_lens"),
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_mlp=48,
        vocabulary_size=100,
        activation="silu",
        gated_mlp=True,
    )
    mlp = architecture.layers[0].mlp

    assert mlp.metadata["gated"] is True
    assert [child.label for child in mlp.children] == [
        "Gate projection", "Value projection", "silu × value", "Linear out",
    ]
    assert mlp.children[0].activation_points == ["blocks.0.mlp.hook_pre"]
    assert mlp.children[1].activation_points == ["blocks.0.mlp.hook_pre_linear"]
    assert mlp.children[2].activation_points == ["blocks.0.mlp.hook_post"]
