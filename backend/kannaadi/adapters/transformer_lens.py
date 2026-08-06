from __future__ import annotations

from typing import Any, Iterable

from kannaadi.adapters.base import ModelAdapter
from kannaadi.domain import ArchitectureGraph, ComponentNode, LayerNode, ModelSpec


class TransformerLensAdapter(ModelAdapter):
    """Thin adapter that contains every TransformerLens-specific name in one place."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._spec: ModelSpec | None = None

    def load(self, spec: ModelSpec) -> None:
        try:
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise RuntimeError("Install Kannaadi with the 'transformers' extra to load models") from exc
        model_name = spec.repository or spec.id
        self._model = HookedTransformer.from_pretrained(model_name, device=spec.device, dtype=spec.dtype)
        self._spec = spec

    def architecture(self) -> ArchitectureGraph:
        if self._model is None or self._spec is None:
            raise RuntimeError("load() must be called before architecture()")
        cfg = self._model.cfg
        return self.from_dimensions(
            self._spec,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            vocabulary_size=cfg.d_vocab,
        )

    @staticmethod
    def from_dimensions(spec: ModelSpec, *, n_layers: int, n_heads: int, d_model: int, vocabulary_size: int) -> ArchitectureGraph:
        layers: list[LayerNode] = []
        for layer in range(n_layers):
            attention = ComponentNode(
                id=f"blocks.{layer}.attn", label="Attention", kind="attention", layer=layer,
                activationPoints=[f"blocks.{layer}.attn.hook_pattern"],
            )
            heads = [
                ComponentNode(
                    id=f"blocks.{layer}.attn.head.{head}", label=f"H{head}", kind="head", layer=layer, head=head,
                    activationPoints=[f"blocks.{layer}.attn.hook_result"],
                    metadata={"slice": {"axis": "head", "index": head}},
                )
                for head in range(n_heads)
            ]
            mlp = ComponentNode(
                id=f"blocks.{layer}.mlp", label="MLP", kind="mlp", layer=layer,
                activationPoints=[f"blocks.{layer}.hook_mlp_out"],
            )
            layers.append(LayerNode(id=f"blocks.{layer}", index=layer, label=f"Layer {layer}", attention=attention, heads=heads, mlp=mlp))
        return ArchitectureGraph(
            modelId=spec.id,
            displayName=spec.display_name,
            family="decoder-only",
            nLayers=n_layers,
            nHeads=n_heads,
            dModel=d_model,
            vocabularySize=vocabulary_size,
            embedding=ComponentNode(id="embed", label="Embedding", kind="embedding", activationPoints=["hook_embed"]),
            layers=layers,
            unembedding=ComponentNode(id="unembed", label="Unembedding", kind="unembedding", activationPoints=["ln_final.hook_normalized"]),
        )

    def tokenize(self, prompts: Iterable[str]) -> Any:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        return self._model.to_tokens(list(prompts))

    def run(self, tokens: Any, requested_activations: Iterable[str] | None = None) -> Any:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        if requested_activations:
            names = set(requested_activations)
            return self._model.run_with_cache(tokens, names_filter=lambda name: name in names)
        return self._model(tokens)

    def supported_activation_points(self) -> list[str]:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        return sorted(self._model.hook_dict.keys())

    def estimate_memory(self, request: dict[str, Any]) -> int:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        batch = int(request.get("batch_size", 1))
        positions = int(request.get("positions", 1))
        activations = int(request.get("activation_count", 1))
        bytes_per_value = 2 if str(self._model.cfg.dtype) in {"torch.float16", "torch.bfloat16"} else 4
        return batch * positions * activations * self._model.cfg.d_model * bytes_per_value
