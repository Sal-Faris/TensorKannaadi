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
    layers: list[LayerNode]
    final_norm: ComponentNode = Field(alias="finalNorm")
    unembedding: ComponentNode

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ArchitectureGraph":
        if len(self.layers) != self.n_layers:
            raise ValueError("nLayers must match the number of layer nodes")
        if any(len(layer.heads) != self.n_heads for layer in self.layers):
            raise ValueError("every layer must expose nHeads head nodes")
        if self.n_key_value_heads > self.n_heads:
            raise ValueError("nKeyValueHeads cannot exceed nHeads")
        return self
