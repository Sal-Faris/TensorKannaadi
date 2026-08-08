from __future__ import annotations

from collections.abc import Iterable

from .architecture import (
    CanonicalFlowGraph,
    ComponentKind,
    ComponentNode,
    FlowEdge,
    FlowModule,
    FlowPort,
    LayerNode,
)


def _shape_for(tensor_role: str) -> list[str | int]:
    if tensor_role == "text":
        return ["batch"]
    if tensor_role in {"token_ids", "position_indices"}:
        return ["batch", "position"]
    if tensor_role == "logits":
        return ["batch", "position", "vocabulary"]
    if tensor_role == "predictions":
        return ["batch", "position", "top_k"]
    return ["batch", "position", "d_model"]


def _port(port_id: str, label: str, direction: str, tensor_role: str) -> FlowPort:
    return FlowPort(
        id=port_id,
        label=label,
        direction=direction,  # type: ignore[arg-type]
        tensorRole=tensor_role,
        shape=_shape_for(tensor_role),
    )


def _module(
    module_id: str,
    label: str,
    role: str,
    kind: ComponentKind | str,
    *,
    parent: str | None = None,
    component: ComponentNode | None = None,
    layer: int | None = None,
    inputs: Iterable[tuple[str, str, str]] = (),
    outputs: Iterable[tuple[str, str, str]] = (),
    children: Iterable[str] = (),
    metadata: dict[str, object] | None = None,
) -> FlowModule:
    ports = [
        *(_port(port_id, label, "input", role) for port_id, label, role in inputs),
        *(_port(port_id, label, "output", role) for port_id, label, role in outputs),
    ]
    return FlowModule(
        id=module_id,
        label=label,
        role=role,
        kind=kind,  # type: ignore[arg-type]
        parentId=parent,
        componentId=component.id if component else None,
        layer=layer,
        ports=ports,
        childIds=list(children),
        metadata=metadata or {},
    )


def _edge(
    edge_id: str,
    source: str,
    source_port: str,
    target: str,
    target_port: str,
    tensor_role: str,
    *,
    kind: str = "data",
    label: str | None = None,
) -> FlowEdge:
    return FlowEdge(
        id=edge_id,
        sourceModuleId=source,
        sourcePortId=source_port,
        targetModuleId=target,
        targetPortId=target_port,
        tensorRole=tensor_role,
        kind=kind,  # type: ignore[arg-type]
        label=label,
    )


def build_canonical_flow(
    *,
    embedding: ComponentNode,
    positional_embedding: ComponentNode | None,
    layers: list[LayerNode],
    final_norm: ComponentNode,
    unembedding: ComponentNode,
    block_topology: str,
    positional_mechanism: str,
    norm_type: str,
    n_key_value_heads: int,
) -> CanonicalFlowGraph:
    """Build the renderer-independent, directed architecture graph.

    The IR deliberately contains explicit addition modules and tensor ports. A
    renderer may rearrange these modules, but it cannot accidentally reverse an
    edge or turn parallel GPT-NeoX branches into a serial GPT-2 block.
    """

    modules: list[FlowModule] = []
    edges: list[FlowEdge] = []
    stage_ids = ["stage.input", *(f"stage.block.{layer.index}" for layer in layers), "stage.output"]
    modules.append(_module("model", "Transformer", "model", "model", children=stage_ids))

    input_children = ["input.prompt", "input.tokenizer", "input.token_embedding"]
    if positional_embedding:
        input_children.append("input.position_embedding")
        if positional_mechanism == "standard":
            input_children.append("input.embedding_sum")
    input_children.append("input.residual")
    modules.append(_module("stage.input", "Input / initialization", "input_stage", "stage", parent="model", children=input_children))
    modules.extend([
        _module("input.prompt", "Prompt", "prompt", "stage", parent="stage.input", outputs=[("text", "Text", "text")]),
        _module(
            "input.tokenizer", "Tokenizer", "tokenizer", "tokenizer", parent="stage.input",
            inputs=[("text", "Text", "text")],
            outputs=[("tokens", "Token IDs", "token_ids"), ("positions", "Positions", "position_indices")],
        ),
        _module(
            "input.token_embedding", embedding.label, "token_embedding", embedding.kind,
            parent="stage.input", component=embedding,
            inputs=[("token_ids", "Token IDs", "token_ids")], outputs=[("embedding", "Token vectors", "token_embedding")],
        ),
    ])
    edges.extend([
        _edge("input.prompt.tokenizer", "input.prompt", "text", "input.tokenizer", "text", "text"),
        _edge("input.tokenizer.embedding", "input.tokenizer", "tokens", "input.token_embedding", "token_ids", "token_ids"),
    ])
    if positional_embedding:
        modules.append(_module(
            "input.position_embedding", positional_embedding.label, "position_embedding", positional_embedding.kind,
            parent="stage.input", component=positional_embedding,
            inputs=[("positions", "Positions", "position_indices")], outputs=[("embedding", "Position vectors", "position_embedding")],
        ))
        edges.append(_edge("input.tokenizer.positions", "input.tokenizer", "positions", "input.position_embedding", "positions", "position_indices"))
    if positional_embedding and positional_mechanism == "standard":
        modules.append(
            _module(
                "input.embedding_sum", "Embedding sum", "residual_add", ComponentKind.OPERATION,
                parent="stage.input",
                inputs=[("token", "Token embedding", "token_embedding"), ("position", "Position embedding", "position_embedding")],
                outputs=[("residual", "Initial residual", "residual")], metadata={"operator": "add"},
            )
        )
        edges.extend([
            _edge("input.token.add", "input.token_embedding", "embedding", "input.embedding_sum", "token", "token_embedding", kind="add"),
            _edge("input.position.add", "input.position_embedding", "embedding", "input.embedding_sum", "position", "position_embedding", kind="add"),
        ])
        residual_source = ("input.embedding_sum", "residual")
    else:
        residual_source = ("input.token_embedding", "embedding")
    modules.append(_module(
        "input.residual", "Initial residual x₀", "residual", ComponentKind.RESIDUAL,
        parent="stage.input", inputs=[("in", "Initialized stream", "residual")], outputs=[("out", "Residual x₀", "residual")],
        metadata={"positionalMechanism": positional_mechanism},
    ))
    edges.append(_edge("input.initial_residual", *residual_source, "input.residual", "in", "residual", kind="residual"))

    previous_module = "input.residual"
    previous_port = "out"
    for layer in layers:
        prefix = f"block.{layer.index}"
        stage = f"stage.block.{layer.index}"
        modules.append(_module(
            stage, layer.label, "transformer_block", "stage", parent="model", layer=layer.index,
            children=(
                [f"{prefix}.resid_pre", f"{prefix}.ln1", f"{prefix}.attention", f"{prefix}.ln2", f"{prefix}.mlp", f"{prefix}.merge", f"{prefix}.resid_post"]
                if block_topology == "parallel" else
                [f"{prefix}.resid_pre", f"{prefix}.ln1", f"{prefix}.attention", f"{prefix}.add_attention", f"{prefix}.resid_mid", f"{prefix}.ln2", f"{prefix}.mlp", f"{prefix}.add_mlp", f"{prefix}.resid_post"]
            ),
            metadata={"topology": block_topology, "normalization": norm_type},
        ))
        modules.append(_module(
            f"{prefix}.resid_pre", layer.residual_pre.label, "residual", layer.residual_pre.kind,
            parent=stage, component=layer.residual_pre, layer=layer.index,
            inputs=[("in", "Residual input", "residual")], outputs=[("out", "Residual input", "residual")],
        ))
        edges.append(_edge(f"{prefix}.enter", previous_module, previous_port, f"{prefix}.resid_pre", "in", "residual", kind="residual"))
        for suffix, node, role in (("ln1", layer.norm1, "attention_norm"), ("attention", layer.attention, "attention"), ("ln2", layer.norm2, "mlp_norm"), ("mlp", layer.mlp, "mlp")):
            detail_nodes = layer.heads if suffix == "attention" else node.children
            component_inputs = [("in", "Input", "residual")]
            if suffix == "attention" and positional_mechanism != "standard":
                component_inputs.append(("positions", "Position signal", "position_embedding" if positional_embedding else "position_indices"))
            modules.append(_module(
                f"{prefix}.{suffix}", node.label, role, node.kind, parent=stage, component=node, layer=layer.index,
                inputs=component_inputs, outputs=[("out", "Output", f"{role}_output")],
                children=[child.id for child in detail_nodes],
                metadata={**node.metadata, **({"positionalMechanism": positional_mechanism} if suffix == "attention" else {})},
            ))
            # Component children are addressable research nodes even when the
            # compact renderer keeps them inside their parent module.
            for child in node.children:
                modules.append(_module(
                    child.id, child.label, f"{role}_detail", child.kind, parent=f"{prefix}.{suffix}",
                    component=child, layer=layer.index, metadata=child.metadata,
                ))
            if suffix == "attention":
                for head in layer.heads:
                    modules.append(_module(
                        head.id, head.label, "attention_head", head.kind, parent=f"{prefix}.attention",
                        component=head, layer=layer.index, children=[child.id for child in head.children], metadata=head.metadata,
                    ))
                    for child in head.children:
                        modules.append(_module(
                            child.id, child.label, "attention_head_detail", child.kind, parent=head.id,
                            component=child, layer=layer.index, metadata=child.metadata,
                        ))
                if positional_mechanism != "standard":
                    position_source = ("input.position_embedding", "embedding") if positional_embedding else ("input.tokenizer", "positions")
                    edges.append(_edge(
                        f"{prefix}.position.attention", *position_source, f"{prefix}.attention", "positions",
                        "position_embedding" if positional_embedding else "position_indices",
                    ))
        if block_topology == "parallel":
            modules.append(_module(
                f"{prefix}.merge", "Joint residual merge", "residual_add", ComponentKind.OPERATION,
                parent=stage, component=layer.residual_mid, layer=layer.index,
                inputs=[("residual", "Residual", "residual"), ("attention", "Attention output", "attention_output"), ("mlp", "MLP output", "mlp_output")],
                outputs=[("out", "Merged residual", "residual")], metadata={"operator": "add", "arity": 3},
            ))
            edges.extend([
                _edge(f"{prefix}.resid.ln1", f"{prefix}.resid_pre", "out", f"{prefix}.ln1", "in", "residual"),
                _edge(f"{prefix}.ln1.attn", f"{prefix}.ln1", "out", f"{prefix}.attention", "in", "normalized_residual"),
                _edge(f"{prefix}.resid.ln2", f"{prefix}.resid_pre", "out", f"{prefix}.ln2", "in", "residual"),
                _edge(f"{prefix}.ln2.mlp", f"{prefix}.ln2", "out", f"{prefix}.mlp", "in", "normalized_residual"),
                _edge(f"{prefix}.resid.merge", f"{prefix}.resid_pre", "out", f"{prefix}.merge", "residual", "residual", kind="add"),
                _edge(f"{prefix}.attn.merge", f"{prefix}.attention", "out", f"{prefix}.merge", "attention", "attention_output", kind="add"),
                _edge(f"{prefix}.mlp.merge", f"{prefix}.mlp", "out", f"{prefix}.merge", "mlp", "mlp_output", kind="add"),
            ])
            merge_source = (f"{prefix}.merge", "out")
        else:
            modules.extend([
                _module(
                    f"{prefix}.add_attention", "Attention residual add", "residual_add", ComponentKind.OPERATION,
                    parent=stage, layer=layer.index,
                    inputs=[("residual", "Residual", "residual"), ("contribution", "Attention output", "attention_output")],
                    outputs=[("out", "Residual mid", "residual")], metadata={"operator": "add"},
                ),
                _module(
                    f"{prefix}.resid_mid", layer.residual_mid.label, "residual", layer.residual_mid.kind,
                    parent=stage, component=layer.residual_mid, layer=layer.index,
                    inputs=[("in", "Residual mid", "residual")], outputs=[("out", "Residual mid", "residual")],
                ),
                _module(
                    f"{prefix}.add_mlp", "MLP residual add", "residual_add", ComponentKind.OPERATION,
                    parent=stage, layer=layer.index,
                    inputs=[("residual", "Residual", "residual"), ("contribution", "MLP output", "mlp_output")],
                    outputs=[("out", "Residual output", "residual")], metadata={"operator": "add"},
                ),
            ])
            edges.extend([
                _edge(f"{prefix}.resid.ln1", f"{prefix}.resid_pre", "out", f"{prefix}.ln1", "in", "residual"),
                _edge(f"{prefix}.ln1.attn", f"{prefix}.ln1", "out", f"{prefix}.attention", "in", "normalized_residual"),
                _edge(f"{prefix}.resid.attn_add", f"{prefix}.resid_pre", "out", f"{prefix}.add_attention", "residual", "residual", kind="add"),
                _edge(f"{prefix}.attn.add", f"{prefix}.attention", "out", f"{prefix}.add_attention", "contribution", "attention_output", kind="add"),
                _edge(f"{prefix}.add.mid", f"{prefix}.add_attention", "out", f"{prefix}.resid_mid", "in", "residual", kind="residual"),
                _edge(f"{prefix}.mid.ln2", f"{prefix}.resid_mid", "out", f"{prefix}.ln2", "in", "residual"),
                _edge(f"{prefix}.ln2.mlp", f"{prefix}.ln2", "out", f"{prefix}.mlp", "in", "normalized_residual"),
                _edge(f"{prefix}.mid.mlp_add", f"{prefix}.resid_mid", "out", f"{prefix}.add_mlp", "residual", "residual", kind="add"),
                _edge(f"{prefix}.mlp.add", f"{prefix}.mlp", "out", f"{prefix}.add_mlp", "contribution", "mlp_output", kind="add"),
            ])
            merge_source = (f"{prefix}.add_mlp", "out")
        modules.append(_module(
            f"{prefix}.resid_post", layer.residual_post.label, "residual", layer.residual_post.kind,
            parent=stage, component=layer.residual_post, layer=layer.index,
            inputs=[("in", "Residual output", "residual")], outputs=[("out", "Residual output", "residual")],
        ))
        edges.append(_edge(f"{prefix}.exit", *merge_source, f"{prefix}.resid_post", "in", "residual", kind="residual"))
        previous_module, previous_port = f"{prefix}.resid_post", "out"

    output_children = ["output.final_residual", "output.final_norm", "output.unembedding", "output.logits"]
    modules.append(_module("stage.output", "Output / readout", "output_stage", "stage", parent="model", children=output_children))
    modules.extend([
        _module("output.final_residual", "Final residual xL", "residual", ComponentKind.RESIDUAL, parent="stage.output", inputs=[("in", "Final residual", "residual")], outputs=[("out", "Final residual", "residual")]),
        _module("output.final_norm", final_norm.label, "final_normalization", final_norm.kind, parent="stage.output", component=final_norm, inputs=[("in", "Final residual", "residual")], outputs=[("out", "Normalized residual", "normalized_residual")], children=[child.id for child in final_norm.children]),
        *(_module(child.id, child.label, "final_normalization_detail", child.kind, parent="output.final_norm", component=child) for child in final_norm.children),
        _module("output.unembedding", unembedding.label, "unembedding", unembedding.kind, parent="stage.output", component=unembedding, inputs=[("in", "Normalized residual", "normalized_residual")], outputs=[("out", "Vocabulary logits", "logits")]),
        _module("output.logits", "Logits / top predictions", "logits", "logits", parent="stage.output", inputs=[("in", "Vocabulary logits", "logits")], outputs=[("out", "Predictions", "predictions")]),
    ])
    edges.extend([
        _edge("output.final_residual", previous_module, previous_port, "output.final_residual", "in", "residual", kind="residual"),
        _edge("output.residual.norm", "output.final_residual", "out", "output.final_norm", "in", "residual", kind="readout"),
        _edge("output.norm.unembed", "output.final_norm", "out", "output.unembedding", "in", "normalized_residual", kind="readout"),
        _edge("output.unembed.logits", "output.unembedding", "out", "output.logits", "in", "logits", kind="readout"),
    ])
    capabilities = [
        "explicit_tensor_ports",
        "nested_components",
        "transformer_lens_hooks",
        f"block_topology:{block_topology}",
        f"position:{positional_mechanism}",
        f"normalization:{norm_type}",
        f"kv_heads:{n_key_value_heads}",
    ]
    return CanonicalFlowGraph(
        schemaVersion=1,
        rootModuleId="model",
        modules=modules,
        edges=edges,
        capabilities=capabilities,
    )
