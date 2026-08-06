from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TokenRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    position: int
    token_id: int = Field(alias="tokenId")
    text: str
    display: str
    next_token: str | None = Field(alias="nextToken", default=None)
    next_token_probability: float | None = Field(alias="nextTokenProbability", default=None)


class Prediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    token_id: int = Field(alias="tokenId")
    text: str
    display: str
    logit: float
    probability: float


class InterventionSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["zero_ablation"] = "zero_ablation"
    component_ids: list[str] = Field(alias="componentIds", min_length=1)
    token_scope: Literal["all"] = Field(alias="tokenScope", default="all")


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(alias="topK", default=10, ge=1, le=50)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @field_validator("prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value


class RunRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    kind: Literal["clean", "intervened"]
    label: str
    status: Literal["complete"] = "complete"
    model_id: str = Field(alias="modelId")
    model_revision: str | None = Field(alias="modelRevision", default=None)
    prompt: str
    tokens: list[TokenRecord]
    top_predictions: list[Prediction] = Field(alias="topPredictions")
    requested_activations: list[str] = Field(alias="requestedActivations")
    interventions: list[InterventionSpec] = Field(default_factory=list)
    parent_run_id: str | None = Field(alias="parentRunId", default=None)
    device: str
    dtype: str
    seed: int
    duration_ms: float = Field(alias="durationMs")
    cache_bytes: int = Field(alias="cacheBytes")
    created_at: datetime = Field(alias="createdAt")
    provenance: dict[str, Any]


class AttentionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    component_id: str = Field(alias="componentId")
    layer: int
    head: int
    pattern_hook: str = Field(alias="patternHook")
    score_hook: str = Field(alias="scoreHook")
    query_tokens: list[str] = Field(alias="queryTokens")
    key_tokens: list[str] = Field(alias="keyTokens")
    pattern: list[list[float]]
    scores: list[list[float]]
    result_norms: list[float] = Field(alias="resultNorms")
    q_norms: list[float] = Field(alias="qNorms")
    k_norms: list[float] = Field(alias="kNorms")
    axes: list[str] = Field(default_factory=lambda: ["query_token", "key_token"])


class ActivationSeries(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    component_id: str = Field(alias="componentId")
    hook_name: str = Field(alias="hookName")
    measure: str
    token_labels: list[str] = Field(alias="tokenLabels")
    values: list[float]
    axes: list[str] = Field(default_factory=lambda: ["position"])


class ResidualPoint(BaseModel):
    layer: int
    stage: Literal["pre", "mid", "post"]
    norm: float
    target_logit: float = Field(alias="targetLogit")


class ResidualStreamResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    position: int
    target_token_id: int = Field(alias="targetTokenId")
    target_token: str = Field(alias="targetToken")
    points: list[ResidualPoint]
    method: Literal["logit_lens"] = "logit_lens"


class TokenComparison(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    token_id: int = Field(alias="tokenId")
    text: str
    display: str
    baseline_logit: float = Field(alias="baselineLogit")
    intervened_logit: float = Field(alias="intervenedLogit")
    delta_logit: float = Field(alias="deltaLogit")
    baseline_probability: float = Field(alias="baselineProbability")
    intervened_probability: float = Field(alias="intervenedProbability")
    delta_probability: float = Field(alias="deltaProbability")


class RunComparison(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    baseline_run_id: str = Field(alias="baselineRunId")
    intervened_run_id: str = Field(alias="intervenedRunId")
    baseline_top_token: str = Field(alias="baselineTopToken")
    baseline_top_token_delta: float = Field(alias="baselineTopTokenDelta")
    kl_divergence: float = Field(alias="klDivergence")
    tokens: list[TokenComparison]
