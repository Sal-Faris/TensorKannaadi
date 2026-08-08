from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComponentKind(StrEnum):
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    RESIDUAL = "residual"
    ATTENTION = "attention"
    HEAD = "head"
    MLP = "mlp"
    UNEMBEDDING = "unembedding"
    PROJECTION = "projection"
    ACTIVATION = "activation"
    OPERATION = "operation"


class FlowPort(BaseModel):
    """A typed tensor port in Kannaadi's model-independent architecture IR."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    label: str
    direction: Literal["input", "output"]
    tensor_role: str = Field(alias="tensorRole")
    shape: list[str | int] = Field(default_factory=list)


class FlowModule(BaseModel):
    """A nestable visual/computational module, independent of any renderer."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    label: str
    role: str
    kind: ComponentKind | Literal["model", "stage", "tokenizer", "logits"]
    parent_id: str | None = Field(alias="parentId", default=None)
    component_id: str | None = Field(alias="componentId", default=None)
    layer: int | None = None
    ports: list[FlowPort] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list, alias="childIds")
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowEdge(BaseModel):
    """A directed tensor-flow edge between module ports."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    source_module_id: str = Field(alias="sourceModuleId")
    source_port_id: str = Field(alias="sourcePortId")
    target_module_id: str = Field(alias="targetModuleId")
    target_port_id: str = Field(alias="targetPortId")
    kind: Literal["data", "residual", "add", "readout"] = "data"
    tensor_role: str = Field(alias="tensorRole")
    label: str | None = None


class CanonicalFlowGraph(BaseModel):
    """Canonical architecture IR used by renderers, recipes, and code generation.

    Existing TransformerLens component IDs remain the stable research-facing IDs;
    this graph adds explicit ports and topology so Kannaadi never has to infer
    computation direction from screen position.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = Field(alias="schemaVersion", default=1)
    root_module_id: str = Field(alias="rootModuleId")
    modules: list[FlowModule]
    edges: list[FlowEdge]
    capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "CanonicalFlowGraph":
        module_index = {module.id: module for module in self.modules}
        if len(module_index) != len(self.modules):
            raise ValueError("canonical flow module IDs must be unique")
        if self.root_module_id not in module_index:
            raise ValueError("rootModuleId must reference a canonical flow module")
        for module in self.modules:
            if module.parent_id is not None and module.parent_id not in module_index:
                raise ValueError(f"unknown parent module: {module.parent_id}")
            if any(child not in module_index for child in module.child_ids):
                raise ValueError(f"module {module.id} contains an unknown child")
        port_index = {
            module.id: {port.id: port for port in module.ports}
            for module in self.modules
        }
        for edge in self.edges:
            if edge.source_module_id not in module_index or edge.target_module_id not in module_index:
                raise ValueError(f"edge {edge.id} references an unknown module")
            source = port_index[edge.source_module_id].get(edge.source_port_id)
            target = port_index[edge.target_module_id].get(edge.target_port_id)
            if source is None or source.direction != "output":
                raise ValueError(f"edge {edge.id} has an invalid source port")
            if target is None or target.direction != "input":
                raise ValueError(f"edge {edge.id} has an invalid target port")
        return self


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    backend: str
    repository: str | None = None
    revision: str | None = None
    local_path: str | None = None
    tokenizer_id: str | None = None
    architecture_metadata: dict[str, Any] = Field(default_factory=dict)
    dtype: str = "float32"
    device: str = "cpu"


class ComponentNode(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    label: str
    kind: ComponentKind
    layer: int | None = None
    head: int | None = None
    activation_points: list[str] = Field(default_factory=list, alias="activationPoints")
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[ComponentNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_head_coordinates(self) -> "ComponentNode":
        if self.kind == ComponentKind.HEAD and (self.layer is None or self.head is None):
            raise ValueError("head components require layer and head indices")
        return self


class LayerNode(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    index: int
    label: str
    residual_pre: ComponentNode = Field(alias="residualPre")
    norm1: ComponentNode
    attention: ComponentNode
    heads: list[ComponentNode]
    residual_mid: ComponentNode = Field(alias="residualMid")
    norm2: ComponentNode
    mlp: ComponentNode
    residual_post: ComponentNode = Field(alias="residualPost")


class ArchitectureGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_id: str = Field(alias="modelId")
    display_name: str = Field(alias="displayName")
    family: str
    n_layers: int = Field(alias="nLayers", gt=0)
    n_heads: int = Field(alias="nHeads", gt=0)
    n_key_value_heads: int = Field(alias="nKeyValueHeads", gt=0)
    d_model: int = Field(alias="dModel", gt=0)
    d_head: int = Field(alias="dHead", gt=0)
    d_mlp: int | None = Field(alias="dMlp", default=None)
    vocabulary_size: int = Field(alias="vocabularySize", gt=0)
    norm_type: str = Field(alias="normType")
    normalization_position: Literal["pre", "post"] = Field(alias="normalizationPosition")
    block_topology: Literal["serial", "parallel"] = Field(alias="blockTopology")
    positional_mechanism: str = Field(alias="positionalMechanism")
    embedding: ComponentNode
    positional_embedding: ComponentNode | None = Field(alias="positionalEmbedding", default=None)
    layers: list[LayerNode]
    final_norm: ComponentNode = Field(alias="finalNorm")
    unembedding: ComponentNode
    flow: CanonicalFlowGraph

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ArchitectureGraph":
        if len(self.layers) != self.n_layers:
            raise ValueError("nLayers must match the number of layer nodes")
        if any(len(layer.heads) != self.n_heads for layer in self.layers):
            raise ValueError("every layer must expose nHeads head nodes")
        if self.n_key_value_heads > self.n_heads:
            raise ValueError("nKeyValueHeads cannot exceed nHeads")
        return self
