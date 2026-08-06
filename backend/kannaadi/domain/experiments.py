from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class MetricSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target_token: str = Field(alias="targetToken", min_length=1)
    distractor_token: str | None = Field(alias="distractorToken", default=None)
    position: int = -1


class InterventionSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["zero_ablation", "mean_ablation", "activation_patch"]
    component_ids: list[str] = Field(alias="componentIds", min_length=1)
    token_scope: Literal["all", "positions"] = Field(alias="tokenScope", default="all")
    positions: list[int] = Field(default_factory=list)
    source_run_id: str | None = Field(alias="sourceRunId", default=None)
    destination_run_id: str | None = Field(alias="destinationRunId", default=None)
    patch_mappings: list[dict[str, int]] = Field(alias="patchMappings", default_factory=list)
    baseline: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "InterventionSpec":
        if self.token_scope == "positions" and not self.positions:
            raise ValueError("position-scoped interventions require at least one token position")
        if self.kind == "activation_patch" and not self.source_run_id:
            raise ValueError("activation patching requires a source run")
        return self


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    kind: Literal["clean", "corrupted"] = "clean"
    label: str | None = Field(default=None, max_length=120)
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
    kind: Literal["clean", "corrupted", "intervened", "patched"]
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


class AlignmentPair(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_position: int | None = Field(alias="sourcePosition")
    destination_position: int | None = Field(alias="destinationPosition")
    source_token: str | None = Field(alias="sourceToken")
    destination_token: str | None = Field(alias="destinationToken")
    status: Literal["exact", "substitution", "source_gap", "destination_gap"]


class TokenAlignment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_run_id: str = Field(alias="sourceRunId")
    destination_run_id: str = Field(alias="destinationRunId")
    strategy: Literal["minimum_edit_distance"] = "minimum_edit_distance"
    pairs: list[AlignmentPair]
    exact_matches: int = Field(alias="exactMatches")
    source_length: int = Field(alias="sourceLength")
    destination_length: int = Field(alias="destinationLength")


class ContrastRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    clean_prompt: str = Field(alias="cleanPrompt", min_length=1, max_length=20_000)
    corrupted_prompt: str = Field(alias="corruptedPrompt", min_length=1, max_length=20_000)
    top_k: int = Field(alias="topK", default=10, ge=1, le=50)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @field_validator("clean_prompt", "corrupted_prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("contrast prompts must contain non-whitespace text")
        return value


class ContrastResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    clean_run: RunRecord = Field(alias="cleanRun")
    corrupted_run: RunRecord = Field(alias="corruptedRun")
    alignment: TokenAlignment


class MetricResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    metric: Literal["target_logit", "logit_difference"]
    position: int
    target_token_id: int = Field(alias="targetTokenId")
    target_token: str = Field(alias="targetToken")
    distractor_token_id: int | None = Field(alias="distractorTokenId", default=None)
    distractor_token: str | None = Field(alias="distractorToken", default=None)
    value: float


class CausalEffect(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    baseline: MetricResult
    intervened: MetricResult
    delta: float


class AblationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["zero_ablation", "mean_ablation"] = "zero_ablation"
    component_ids: list[str] = Field(alias="componentIds", min_length=1)
    token_scope: Literal["all", "positions"] = Field(alias="tokenScope", default="all")
    positions: list[int] = Field(default_factory=list)
    metric: MetricSpec | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "AblationRequest":
        if self.token_scope == "positions" and not self.positions:
            raise ValueError("Choose one or more token positions for a position-scoped ablation")
        return self


class PatchMapping(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_position: int = Field(alias="sourcePosition", ge=0)
    destination_position: int = Field(alias="destinationPosition", ge=0)


class PatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_run_id: str = Field(alias="sourceRunId")
    component_ids: list[str] = Field(alias="componentIds", min_length=1)
    mappings: list[PatchMapping] = Field(default_factory=list)
    metric: MetricSpec | None = None


class InterventionResult(BaseModel):
    run: RunRecord
    effect: CausalEffect | None = None


class HeadEffect(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    component_id: str = Field(alias="componentId")
    layer: int
    head: int
    metric_value: float = Field(alias="metricValue")
    delta: float


class HeadSweepRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["zero_ablation", "mean_ablation"] = "zero_ablation"
    token_scope: Literal["all", "positions"] = Field(alias="tokenScope", default="all")
    positions: list[int] = Field(default_factory=list)
    metric: MetricSpec

    @model_validator(mode="after")
    def validate_scope(self) -> "HeadSweepRequest":
        if self.token_scope == "positions" and not self.positions:
            raise ValueError("Choose one or more token positions for a position-scoped sweep")
        return self


class HeadSweepResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    kind: Literal["zero_ablation", "mean_ablation"]
    metric: MetricResult
    effects: list[HeadEffect]
    minimum: float
    maximum: float
    duration_ms: float = Field(alias="durationMs")


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
