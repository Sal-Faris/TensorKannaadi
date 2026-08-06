from __future__ import annotations

from typing import Any, Iterable

from kannaadi.adapters.base import ModelAdapter
from kannaadi.domain import ArchitectureGraph, ComponentNode, LayerNode, ModelSpec


class TransformerLensAdapter(ModelAdapter):
    """Contains TransformerLens-specific names and tensor conventions in one boundary."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._spec: ModelSpec | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        return self._model

    def load(self, spec: ModelSpec) -> None:
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise RuntimeError("Install Kannaadi with the 'transformers' extra to load models") from exc
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(spec.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {spec.dtype}")
        model_name = spec.local_path or spec.repository or spec.id
        load_kwargs: dict[str, Any] = {"device": spec.device, "dtype": dtype}
        if spec.revision:
            load_kwargs["revision"] = spec.revision
        self._model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
        self._model.set_use_attn_result(True)
        self._spec = spec

    def architecture(self) -> ArchitectureGraph:
        if self._model is None or self._spec is None:
            raise RuntimeError("load() must be called before architecture()")
        cfg = self._model.cfg
        return self.from_dimensions(
            self._spec,
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            n_key_value_heads=int(getattr(cfg, "n_key_value_heads", cfg.n_heads) or cfg.n_heads),
            d_model=int(cfg.d_model),
            d_head=int(cfg.d_head),
            d_mlp=int(cfg.d_mlp) if getattr(cfg, "d_mlp", None) else None,
            vocabulary_size=int(cfg.d_vocab),
            norm_type=str(getattr(cfg, "normalization_type", "LN")),
            normalization_position="pre" if bool(getattr(cfg, "use_attn_in", False) or not getattr(cfg, "post_embedding_ln", False)) else "post",
            block_topology="parallel" if bool(getattr(cfg, "parallel_attn_mlp", False)) else "serial",
            positional_mechanism=str(getattr(cfg, "positional_embedding_type", "standard")),
            activation=str(getattr(cfg, "act_fn", "activation")),
        )

    @staticmethod
    def from_dimensions(
        spec: ModelSpec,
        *,
        n_layers: int,
        n_heads: int,
        d_model: int,
        vocabulary_size: int,
        d_head: int | None = None,
        d_mlp: int | None = None,
        n_key_value_heads: int | None = None,
        norm_type: str = "LN",
        normalization_position: str = "pre",
        block_topology: str = "serial",
        positional_mechanism: str = "standard",
        activation: str = "gelu_new",
    ) -> ArchitectureGraph:
        resolved_d_head = d_head or d_model // n_heads
        layers: list[LayerNode] = []
        for layer in range(n_layers):
            residual_pre = ComponentNode(id=f"blocks.{layer}.resid_pre", label="Residual pre", kind="residual", layer=layer, activationPoints=[f"blocks.{layer}.hook_resid_pre"])
            norm1 = ComponentNode(
                id=f"blocks.{layer}.ln1", label=norm_type, kind="normalization", layer=layer,
                activationPoints=[f"blocks.{layer}.ln1.hook_normalized"],
                children=[
                    ComponentNode(id=f"blocks.{layer}.ln1.scale", label="Scale", kind="operation", layer=layer, activationPoints=[f"blocks.{layer}.ln1.hook_scale"]),
                    ComponentNode(id=f"blocks.{layer}.ln1.normalized", label="Normalized", kind="operation", layer=layer, activationPoints=[f"blocks.{layer}.ln1.hook_normalized"]),
                ],
            )
            attention = ComponentNode(id=f"blocks.{layer}.attn", label="Multi-Head Attention", kind="attention", layer=layer, activationPoints=[f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.hook_attn_out"])
            heads = [
                ComponentNode(
                    id=f"blocks.{layer}.attn.head.{head}",
                    label=f"H{head}",
                    kind="head",
                    layer=layer,
                    head=head,
                    activationPoints=[f"blocks.{layer}.attn.hook_result"],
                    metadata={"slice": {"axis": "head", "index": head}, "d_head": resolved_d_head},
                    children=[
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.q", label="Query", kind="projection", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_q"], metadata={"slice": {"axis": "head", "index": head}}),
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.k", label="Key", kind="projection", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_k"], metadata={"slice": {"axis": "head", "index": head}}),
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.v", label="Value", kind="projection", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_v"], metadata={"slice": {"axis": "head", "index": head}}),
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.scores", label="Attention scores", kind="operation", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_attn_scores"], metadata={"slice": {"axis": "head", "index": head}}),
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.pattern", label="Softmax pattern", kind="operation", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_pattern"], metadata={"slice": {"axis": "head", "index": head}}),
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.z", label="Weighted values", kind="operation", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_z"], metadata={"slice": {"axis": "head", "index": head}}),
                        ComponentNode(id=f"blocks.{layer}.attn.head.{head}.result", label="Result", kind="projection", layer=layer, head=head, activationPoints=[f"blocks.{layer}.attn.hook_result"], metadata={"slice": {"axis": "head", "index": head}}),
                    ],
                )
                for head in range(n_heads)
            ]
            residual_mid = ComponentNode(id=f"blocks.{layer}.resid_mid", label="Residual mid", kind="residual", layer=layer, activationPoints=[f"blocks.{layer}.hook_resid_mid"])
            norm2 = ComponentNode(
                id=f"blocks.{layer}.ln2", label=norm_type, kind="normalization", layer=layer,
                activationPoints=[f"blocks.{layer}.ln2.hook_normalized"],
                children=[
                    ComponentNode(id=f"blocks.{layer}.ln2.scale", label="Scale", kind="operation", layer=layer, activationPoints=[f"blocks.{layer}.ln2.hook_scale"]),
                    ComponentNode(id=f"blocks.{layer}.ln2.normalized", label="Normalized", kind="operation", layer=layer, activationPoints=[f"blocks.{layer}.ln2.hook_normalized"]),
                ],
            )
            mlp = ComponentNode(
                id=f"blocks.{layer}.mlp", label="MLP", kind="mlp", layer=layer,
                activationPoints=[f"blocks.{layer}.hook_mlp_out"],
                metadata={"activation": activation, "d_mlp": d_mlp},
                children=[
                    ComponentNode(id=f"blocks.{layer}.mlp.in", label="Linear in", kind="projection", layer=layer, activationPoints=[f"blocks.{layer}.mlp.hook_pre"]),
                    ComponentNode(id=f"blocks.{layer}.mlp.activation", label=activation, kind="activation", layer=layer, activationPoints=[f"blocks.{layer}.mlp.hook_post"]),
                    ComponentNode(id=f"blocks.{layer}.mlp.out", label="Linear out", kind="projection", layer=layer, activationPoints=[f"blocks.{layer}.hook_mlp_out"]),
                ],
            )
            residual_post = ComponentNode(id=f"blocks.{layer}.resid_post", label="Residual post", kind="residual", layer=layer, activationPoints=[f"blocks.{layer}.hook_resid_post"])
            layers.append(LayerNode(
                id=f"blocks.{layer}", index=layer, label=f"Layer {layer}", residualPre=residual_pre,
                norm1=norm1, attention=attention, heads=heads, residualMid=residual_mid,
                norm2=norm2, mlp=mlp, residualPost=residual_post,
            ))
        return ArchitectureGraph(
            modelId=spec.id,
            displayName=spec.display_name,
            family="decoder-only",
            nLayers=n_layers,
            nHeads=n_heads,
            nKeyValueHeads=n_key_value_heads or n_heads,
            dModel=d_model,
            dHead=resolved_d_head,
            dMlp=d_mlp,
            vocabularySize=vocabulary_size,
            normType=norm_type,
            normalizationPosition=normalization_position,
            blockTopology=block_topology,
            positionalMechanism=positional_mechanism,
            embedding=ComponentNode(id="embed", label="Token embedding", kind="embedding", activationPoints=["hook_embed"]),
            positionalEmbedding=(
                ComponentNode(id="pos_embed", label="Positional embedding", kind="embedding", activationPoints=["hook_pos_embed"], metadata={"mechanism": positional_mechanism})
                if positional_mechanism in {"standard", "shortformer"} else None
            ),
            layers=layers,
            finalNorm=ComponentNode(
                id="ln_final", label="Final normalization", kind="normalization",
                activationPoints=["ln_final.hook_normalized"],
                children=[
                    ComponentNode(id="ln_final.scale", label="Scale", kind="operation", activationPoints=["ln_final.hook_scale"]),
                    ComponentNode(id="ln_final.normalized", label="Normalized", kind="operation", activationPoints=["ln_final.hook_normalized"]),
                ],
            ),
            unembedding=ComponentNode(
                id="unembed", label="Unembedding", kind="unembedding",
                activationPoints=["unembed.hook_in", "unembed.hook_out"],
            ),
        )

    def tokenize(self, prompts: Iterable[str]) -> Any:
        return self.model.to_tokens(list(prompts))

    def run(self, tokens: Any, requested_activations: Iterable[str] | None = None) -> Any:
        if requested_activations:
            names = set(requested_activations)
            return self.model.run_with_cache(tokens, names_filter=lambda name: name in names)
        return self.model(tokens)

    def install_intervention(self, spec: Any) -> Any:
        raise NotImplementedError("Persistent adapter interventions are not installed directly; use an experiment run specification")

    def supported_activation_points(self) -> list[str]:
        return sorted(self.model.hook_dict.keys())

    def estimate_memory(self, request: dict[str, Any]) -> int:
        batch = int(request.get("batch_size", 1))
        positions = int(request.get("positions", 1))
        activations = int(request.get("activation_count", 1))
        bytes_per_value = 2 if str(self.model.cfg.dtype) in {"torch.float16", "torch.bfloat16"} else 4
        return batch * positions * activations * self.model.cfg.d_model * bytes_per_value
