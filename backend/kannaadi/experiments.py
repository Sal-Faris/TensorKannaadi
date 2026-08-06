from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import statistics
import threading
import time
from typing import Any, Iterable, Literal
from uuid import uuid4

from kannaadi.domain import (
    ActivationSeries,
    AlignmentPair,
    ArchitectureGraph,
    AttentionResult,
    AttributionEffect,
    AttributionResult,
    CausalEffect,
    ComponentNode,
    ContrastResult,
    DatasetAblationResult,
    DatasetAblationRow,
    DatasetAblationSummary,
    HeadEffect,
    HeadSweepResult,
    InterventionResult,
    InterventionSpec,
    MetricResult,
    MetricSpec,
    MlpEffect,
    MlpSweepResult,
    PatchMapping,
    Prediction,
    ResidualPoint,
    ResidualStreamResult,
    RunComparison,
    RunRecord,
    TokenAlignment,
    TokenComparison,
    TokenRecord,
)


HEAD_ID = re.compile(r"blocks\.(?P<layer>\d+)\.attn\.head\.(?P<head>\d+)$")
MLP_ID = re.compile(r"blocks\.(?P<layer>\d+)\.mlp$")
NEURON_ID = re.compile(r"blocks\.(?P<layer>\d+)\.mlp\.neuron\.(?P<neuron>\d+)$")


@dataclass(slots=True)
class RunArtifacts:
    record: RunRecord
    tokens: Any
    logits: Any
    cache: dict[str, Any]


class ExperimentEngine:
    """Owns immutable run manifests and the tensors backing interactive analysis."""

    max_prompt_tokens = 512
    _cache_suffixes = (
        "hook_embed",
        "hook_pos_embed",
        "hook_q",
        "hook_k",
        "hook_v",
        "hook_z",
        "hook_attn_scores",
        "hook_pattern",
        "hook_result",
        "hook_attn_out",
        "hook_mlp_out",
        "hook_resid_pre",
        "hook_resid_mid",
        "hook_resid_post",
        "hook_normalized",
        "hook_scale",
        "mlp.hook_pre",
        "mlp.hook_post",
    )

    def __init__(self, adapter: Any, architecture: ArchitectureGraph, model_spec: Any) -> None:
        self.adapter = adapter
        self.architecture = architecture
        self.model_spec = model_spec
        self._runs: dict[str, RunArtifacts] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._component_index = self._index_components(architecture)

    @property
    def model(self) -> Any:
        return self.adapter.model

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return [self._runs[run_id].record for run_id in reversed(self._order)]

    def get_run(self, run_id: str) -> RunRecord:
        return self._artifacts(run_id).record

    def run_prompt(
        self,
        prompt: str,
        *,
        kind: Literal["clean", "corrupted"] = "clean",
        label: str | None = None,
        top_k: int = 10,
        seed: int = 0,
    ) -> RunRecord:
        with self._lock:
            tokens = self.adapter.tokenize([prompt])
            if int(tokens.shape[-1]) > self.max_prompt_tokens:
                raise ValueError(
                    f"Prompt tokenized to {int(tokens.shape[-1])} tokens; the interactive cache limit is {self.max_prompt_tokens}"
                )
            return self._execute(
                tokens=tokens,
                prompt=prompt,
                kind=kind,
                label=label or ("Clean run" if kind == "clean" else "Corrupted run"),
                interventions=[],
                parent_run_id=None,
                top_k=top_k,
                seed=seed,
            ).record

    def run_clean(self, prompt: str, *, top_k: int = 10, seed: int = 0) -> RunRecord:
        return self.run_prompt(prompt, kind="clean", top_k=top_k, seed=seed)

    def run_contrast(self, clean_prompt: str, corrupted_prompt: str, *, top_k: int = 10, seed: int = 0) -> ContrastResult:
        with self._lock:
            clean = self.run_prompt(clean_prompt, kind="clean", label="Clean source", top_k=top_k, seed=seed)
            corrupted = self.run_prompt(
                corrupted_prompt,
                kind="corrupted",
                label="Corrupted destination",
                top_k=top_k,
                seed=seed,
            )
            return ContrastResult(
                id=f"contrast_{uuid4().hex[:12]}",
                cleanRun=clean,
                corruptedRun=corrupted,
                alignment=self.align_runs(clean.id, corrupted.id),
            )

    def align_runs(self, source_run_id: str, destination_run_id: str) -> TokenAlignment:
        source = self._artifacts(source_run_id).record
        destination = self._artifacts(destination_run_id).record
        pairs = self.minimum_edit_alignment(source.tokens, destination.tokens)
        return TokenAlignment(
            sourceRunId=source_run_id,
            destinationRunId=destination_run_id,
            pairs=pairs,
            exactMatches=sum(pair.status == "exact" for pair in pairs),
            sourceLength=len(source.tokens),
            destinationLength=len(destination.tokens),
        )

    @staticmethod
    def minimum_edit_alignment(source: list[TokenRecord], destination: list[TokenRecord]) -> list[AlignmentPair]:
        """Align tokenizer output with deterministic minimum edit distance."""
        rows, columns = len(source), len(destination)
        cost = [[0] * (columns + 1) for _ in range(rows + 1)]
        for row in range(rows + 1):
            cost[row][0] = row
        for column in range(columns + 1):
            cost[0][column] = column
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                substitution = 0 if source[row - 1].token_id == destination[column - 1].token_id else 1
                cost[row][column] = min(
                    cost[row - 1][column] + 1,
                    cost[row][column - 1] + 1,
                    cost[row - 1][column - 1] + substitution,
                )

        aligned: list[AlignmentPair] = []
        row, column = rows, columns
        while row or column:
            if row and column:
                substitution = 0 if source[row - 1].token_id == destination[column - 1].token_id else 1
                if cost[row][column] == cost[row - 1][column - 1] + substitution:
                    source_token = source[row - 1]
                    destination_token = destination[column - 1]
                    aligned.append(
                        AlignmentPair(
                            sourcePosition=source_token.position,
                            destinationPosition=destination_token.position,
                            sourceToken=source_token.display,
                            destinationToken=destination_token.display,
                            status="exact" if substitution == 0 else "substitution",
                        )
                    )
                    row -= 1
                    column -= 1
                    continue
            if row and cost[row][column] == cost[row - 1][column] + 1:
                source_token = source[row - 1]
                aligned.append(
                    AlignmentPair(
                        sourcePosition=source_token.position,
                        destinationPosition=None,
                        sourceToken=source_token.display,
                        destinationToken=None,
                        status="destination_gap",
                    )
                )
                row -= 1
            else:
                destination_token = destination[column - 1]
                aligned.append(
                    AlignmentPair(
                        sourcePosition=None,
                        destinationPosition=destination_token.position,
                        sourceToken=None,
                        destinationToken=destination_token.display,
                        status="source_gap",
                    )
                )
                column -= 1
        aligned.reverse()
        return aligned

    def ablate(
        self,
        baseline_run_id: str,
        component_ids: list[str],
        *,
        kind: Literal["zero_ablation", "mean_ablation"] = "zero_ablation",
        token_scope: Literal["all", "positions"] = "all",
        positions: list[int] | None = None,
        metric: MetricSpec | None = None,
        top_k: int = 10,
    ) -> InterventionResult:
        with self._lock:
            baseline = self._artifacts(baseline_run_id)
            resolved_positions = self._resolve_positions(baseline, token_scope, positions or [])
            grouped, mlp_layers, grouped_neurons = self._partition_intervenable_components(component_ids)
            hooks: list[tuple[str, Any]] = []
            for layer, heads in grouped.items():
                selected_heads = tuple(sorted(heads))
                reference = baseline.cache[f"blocks.{layer}.attn.hook_result"].float().mean(dim=1)[0]

                def ablate_selected(
                    result: Any,
                    hook: Any,
                    indices: tuple[int, ...] = selected_heads,
                    token_positions: tuple[int, ...] = tuple(resolved_positions),
                    reference_values: Any = reference,
                    ablation_kind: str = kind,
                ) -> Any:
                    del hook
                    updated = result.clone()
                    for head in indices:
                        if ablation_kind == "zero_ablation":
                            updated[:, list(token_positions), head, :] = 0
                        else:
                            mean_value = reference_values[head].to(device=updated.device, dtype=updated.dtype)
                            updated[:, list(token_positions), head, :] = mean_value
                    return updated

                hooks.append((f"blocks.{layer}.attn.hook_result", ablate_selected))

            for layer in sorted(mlp_layers):
                reference = baseline.cache[f"blocks.{layer}.hook_mlp_out"].float().mean(dim=1)[0]

                def ablate_mlp(
                    result: Any,
                    hook: Any,
                    token_positions: tuple[int, ...] = tuple(resolved_positions),
                    reference_value: Any = reference,
                    ablation_kind: str = kind,
                ) -> Any:
                    del hook
                    updated = result.clone()
                    if ablation_kind == "zero_ablation":
                        updated[:, list(token_positions), :] = 0
                    else:
                        value = reference_value.to(device=updated.device, dtype=updated.dtype)
                        updated[:, list(token_positions), :] = value
                    return updated

                hooks.append((f"blocks.{layer}.hook_mlp_out", ablate_mlp))

            for layer, neurons in grouped_neurons.items():
                selected_neurons = tuple(sorted(neurons))
                reference = baseline.cache[f"blocks.{layer}.mlp.hook_post"].float().mean(dim=1)[0]

                def ablate_neurons(
                    result: Any,
                    hook: Any,
                    indices: tuple[int, ...] = selected_neurons,
                    token_positions: tuple[int, ...] = tuple(resolved_positions),
                    reference_values: Any = reference,
                    ablation_kind: str = kind,
                ) -> Any:
                    del hook
                    updated = result.clone()
                    for neuron in indices:
                        if ablation_kind == "zero_ablation":
                            updated[:, list(token_positions), neuron] = 0
                        else:
                            value = reference_values[neuron].to(device=updated.device, dtype=updated.dtype)
                            updated[:, list(token_positions), neuron] = value
                    return updated

                hooks.append((f"blocks.{layer}.mlp.hook_post", ablate_neurons))

            intervention = InterventionSpec(
                kind=kind,
                componentIds=component_ids,
                tokenScope=token_scope,
                positions=[] if token_scope == "all" else resolved_positions,
                destinationRunId=baseline_run_id,
                baseline="zero" if kind == "zero_ablation" else "within_prompt_position_mean",
            )
            operation = "Zero ablate" if kind == "zero_ablation" else "Mean ablate"
            label = operation + " " + ", ".join(self._display_component(value) for value in component_ids)
            device = str(getattr(self.model.cfg, "device", self.model_spec.device))
            created = self._execute(
                tokens=baseline.tokens.to(device),
                prompt=baseline.record.prompt,
                kind="intervened",
                label=label,
                interventions=[intervention],
                parent_run_id=baseline_run_id,
                top_k=top_k,
                seed=baseline.record.seed,
                hooks=hooks,
            ).record
            return InterventionResult(run=created, effect=self._effect(baseline_run_id, created.id, metric) if metric else None)

    def zero_ablate(
        self,
        baseline_run_id: str,
        component_ids: list[str],
        *,
        token_scope: str = "all",
        positions: list[int] | None = None,
        top_k: int = 10,
    ) -> RunRecord:
        return self.ablate(
            baseline_run_id,
            component_ids,
            kind="zero_ablation",
            token_scope=token_scope,  # type: ignore[arg-type]
            positions=positions,
            top_k=top_k,
        ).run

    def dataset_ablation(
        self,
        run_ids: list[str],
        component_ids: list[str],
        *,
        kind: Literal["zero_ablation", "mean_ablation"] = "zero_ablation",
        token_scope: Literal["all", "positions"] = "all",
        positions: list[int] | None = None,
        metric: MetricSpec,
    ) -> DatasetAblationResult:
        """Apply one intervention recipe across cached baselines and aggregate exact effects.

        Failures are retained per prompt so a long collection does not discard successful
        work. Shared component IDs are validated before execution; every successful row
        still points to the immutable baseline and intervention manifests used to measure it.
        """

        with self._lock:
            if not run_ids:
                raise ValueError("Choose at least one baseline run for a dataset experiment")
            if len(set(run_ids)) != len(run_ids):
                raise ValueError("Dataset experiments require unique baseline run IDs")
            self._partition_intervenable_components(component_ids)
            baselines = [self._artifacts(run_id) for run_id in run_ids]
            for baseline in baselines:
                if baseline.record.kind not in {"clean", "corrupted"}:
                    raise ValueError(
                        f"Dataset baseline {baseline.record.id} is {baseline.record.kind}; "
                        "choose original clean or corrupted runs instead of derived interventions"
                    )

            started = time.perf_counter()
            rows: list[DatasetAblationRow] = []
            deltas: list[float] = []
            for baseline in baselines:
                try:
                    result = self.ablate(
                        baseline.record.id,
                        component_ids,
                        kind=kind,
                        token_scope=token_scope,
                        positions=positions or [],
                        metric=metric,
                    )
                    if result.effect is None:  # pragma: no cover - guarded by metric above
                        raise RuntimeError("The intervention completed without its requested metric")
                    delta = float(result.effect.delta)
                    deltas.append(delta)
                    rows.append(
                        DatasetAblationRow(
                            baselineRunId=baseline.record.id,
                            intervenedRunId=result.run.id,
                            intervenedRun=result.run,
                            label=baseline.record.label,
                            prompt=baseline.record.prompt,
                            status="complete",
                            baselineValue=result.effect.baseline.value,
                            intervenedValue=result.effect.intervened.value,
                            delta=delta,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        DatasetAblationRow(
                            baselineRunId=baseline.record.id,
                            label=baseline.record.label,
                            prompt=baseline.record.prompt,
                            status="error",
                            error=str(exc),
                        )
                    )

            summary = self._dataset_summary(len(run_ids), deltas)
            return DatasetAblationResult(
                id=f"dataset_{uuid4().hex[:12]}",
                kind=kind,
                componentIds=component_ids,
                tokenScope=token_scope,
                positions=[] if token_scope == "all" else list(positions or []),
                metric=metric,
                rows=rows,
                summary=summary,
                durationMs=round((time.perf_counter() - started) * 1000, 3),
            )

    @staticmethod
    def _dataset_summary(requested_count: int, deltas: list[float]) -> DatasetAblationSummary:
        completed = len(deltas)
        if not deltas:
            return DatasetAblationSummary(
                requestedCount=requested_count,
                completedCount=0,
                failedCount=requested_count,
                meanDelta=None,
                medianDelta=None,
                standardDeviation=None,
                minimumDelta=None,
                maximumDelta=None,
                meanAbsoluteDelta=None,
                directionConsistency=None,
            )
        mean_delta = statistics.fmean(deltas)
        if mean_delta > 0:
            direction_consistency = sum(delta > 0 for delta in deltas) / completed
        elif mean_delta < 0:
            direction_consistency = sum(delta < 0 for delta in deltas) / completed
        else:
            direction_consistency = sum(delta == 0 for delta in deltas) / completed
        return DatasetAblationSummary(
            requestedCount=requested_count,
            completedCount=completed,
            failedCount=requested_count - completed,
            meanDelta=mean_delta,
            medianDelta=statistics.median(deltas),
            standardDeviation=statistics.pstdev(deltas),
            minimumDelta=min(deltas),
            maximumDelta=max(deltas),
            meanAbsoluteDelta=statistics.fmean(abs(delta) for delta in deltas),
            directionConsistency=direction_consistency,
        )

    def patch(
        self,
        destination_run_id: str,
        source_run_id: str,
        component_ids: list[str],
        *,
        mappings: list[PatchMapping] | None = None,
        metric: MetricSpec | None = None,
        top_k: int = 10,
    ) -> InterventionResult:
        with self._lock:
            source = self._artifacts(source_run_id)
            destination = self._artifacts(destination_run_id)
            if source.record.model_id != destination.record.model_id:
                raise ValueError("Activation patching requires runs from the same loaded model")
            if not mappings:
                alignment = self.align_runs(source_run_id, destination_run_id)
                mappings = [
                    PatchMapping(sourcePosition=pair.source_position, destinationPosition=pair.destination_position)
                    for pair in alignment.pairs
                    if pair.source_position is not None and pair.destination_position is not None
                ]
            if not mappings:
                raise ValueError("The source and destination runs have no patchable token alignment")
            for mapping in mappings:
                if mapping.source_position >= len(source.record.tokens) or mapping.destination_position >= len(destination.record.tokens):
                    raise ValueError("A patch mapping is outside the source or destination token range")

            grouped, mlp_layers, grouped_neurons = self._partition_intervenable_components(component_ids)
            hooks: list[tuple[str, Any]] = []
            for layer, heads in grouped.items():
                selected_heads = tuple(sorted(heads))
                source_result = source.cache[f"blocks.{layer}.attn.hook_result"][0]

                def patch_selected(
                    result: Any,
                    hook: Any,
                    indices: tuple[int, ...] = selected_heads,
                    source_values: Any = source_result,
                    token_mappings: tuple[PatchMapping, ...] = tuple(mappings),
                ) -> Any:
                    del hook
                    updated = result.clone()
                    for mapping in token_mappings:
                        for head in indices:
                            value = source_values[mapping.source_position, head].to(
                                device=updated.device,
                                dtype=updated.dtype,
                            )
                            updated[:, mapping.destination_position, head, :] = value
                    return updated

                hooks.append((f"blocks.{layer}.attn.hook_result", patch_selected))

            for layer in sorted(mlp_layers):
                source_mlp = source.cache[f"blocks.{layer}.hook_mlp_out"][0]

                def patch_mlp(
                    result: Any,
                    hook: Any,
                    source_values: Any = source_mlp,
                    token_mappings: tuple[PatchMapping, ...] = tuple(mappings),
                ) -> Any:
                    del hook
                    updated = result.clone()
                    for mapping in token_mappings:
                        value = source_values[mapping.source_position].to(
                            device=updated.device,
                            dtype=updated.dtype,
                        )
                        updated[:, mapping.destination_position, :] = value
                    return updated

                hooks.append((f"blocks.{layer}.hook_mlp_out", patch_mlp))

            for layer, neurons in grouped_neurons.items():
                selected_neurons = tuple(sorted(neurons))
                source_post = source.cache[f"blocks.{layer}.mlp.hook_post"][0]

                def patch_neurons(
                    result: Any,
                    hook: Any,
                    indices: tuple[int, ...] = selected_neurons,
                    source_values: Any = source_post,
                    token_mappings: tuple[PatchMapping, ...] = tuple(mappings),
                ) -> Any:
                    del hook
                    updated = result.clone()
                    for mapping in token_mappings:
                        for neuron in indices:
                            value = source_values[mapping.source_position, neuron].to(
                                device=updated.device,
                                dtype=updated.dtype,
                            )
                            updated[:, mapping.destination_position, neuron] = value
                    return updated

                hooks.append((f"blocks.{layer}.mlp.hook_post", patch_neurons))

            destination_positions = sorted({mapping.destination_position for mapping in mappings})
            intervention = InterventionSpec(
                kind="activation_patch",
                componentIds=component_ids,
                tokenScope="positions",
                positions=destination_positions,
                sourceRunId=source_run_id,
                destinationRunId=destination_run_id,
                patchMappings=[
                    {"sourcePosition": mapping.source_position, "destinationPosition": mapping.destination_position}
                    for mapping in mappings
                ],
                baseline="source_activation",
            )
            label = "Patch " + ", ".join(self._display_component(value) for value in component_ids)
            device = str(getattr(self.model.cfg, "device", self.model_spec.device))
            created = self._execute(
                tokens=destination.tokens.to(device),
                prompt=destination.record.prompt,
                kind="patched",
                label=label,
                interventions=[intervention],
                parent_run_id=destination_run_id,
                top_k=top_k,
                seed=destination.record.seed,
                hooks=hooks,
            ).record
            return InterventionResult(
                run=created,
                effect=self._effect(destination_run_id, created.id, metric) if metric else None,
            )

    def metric(self, run_id: str, spec: MetricSpec) -> MetricResult:
        artifacts = self._artifacts(run_id)
        target_id = self._single_token_id(spec.target_token)
        distractor_id = self._single_token_id(spec.distractor_token) if spec.distractor_token else None
        position = self._resolve_output_position(artifacts, spec.position)
        logits = artifacts.logits[0, position].float()
        value = float(logits[target_id].item())
        if distractor_id is not None:
            value -= float(logits[distractor_id].item())
        return MetricResult(
            runId=run_id,
            metric="logit_difference" if distractor_id is not None else "target_logit",
            position=position,
            targetTokenId=target_id,
            targetToken=self._decode_token(target_id),
            distractorTokenId=distractor_id,
            distractorToken=self._decode_token(distractor_id) if distractor_id is not None else None,
            value=value,
        )

    def head_sweep(
        self,
        run_id: str,
        *,
        kind: Literal["zero_ablation", "mean_ablation"],
        token_scope: Literal["all", "positions"],
        positions: list[int],
        metric: MetricSpec,
    ) -> HeadSweepResult:
        import torch

        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            baseline = self._artifacts(run_id)
            baseline_metric = self.metric(run_id, metric)
            target_id = baseline_metric.target_token_id
            distractor_id = baseline_metric.distractor_token_id
            output_position = baseline_metric.position
            token_positions = self._resolve_positions(baseline, token_scope, positions)
            device = str(getattr(self.model.cfg, "device", self.model_spec.device))
            effects: list[HeadEffect] = []

            for layer in range(self.architecture.n_layers):
                tokens = baseline.tokens.to(device).repeat(self.architecture.n_heads, 1)
                reference = baseline.cache[f"blocks.{layer}.attn.hook_result"].float().mean(dim=1)[0]

                def ablate_each_head(result: Any, hook: Any) -> Any:
                    del hook
                    updated = result.clone()
                    for head in range(self.architecture.n_heads):
                        if kind == "zero_ablation":
                            updated[head, token_positions, head, :] = 0
                        else:
                            mean_value = reference[head].to(device=updated.device, dtype=updated.dtype)
                            updated[head, token_positions, head, :] = mean_value
                    return updated

                logits = self.model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_result", ablate_each_head)],
                ).detach().to("cpu").float()
                values = logits[:, output_position, target_id]
                if distractor_id is not None:
                    values = values - logits[:, output_position, distractor_id]
                for head, value in enumerate(values.tolist()):
                    effects.append(
                        HeadEffect(
                            componentId=f"blocks.{layer}.attn.head.{head}",
                            layer=layer,
                            head=head,
                            metricValue=float(value),
                            delta=float(value - baseline_metric.value),
                        )
                    )

            deltas = [effect.delta for effect in effects]
            return HeadSweepResult(
                runId=run_id,
                kind=kind,
                metric=baseline_metric,
                effects=effects,
                minimum=min(deltas, default=0.0),
                maximum=max(deltas, default=0.0),
                durationMs=round((time.perf_counter() - started) * 1000, 3),
            )

    def mlp_sweep(
        self,
        run_id: str,
        *,
        kind: Literal["zero_ablation", "mean_ablation"],
        token_scope: Literal["all", "positions"],
        positions: list[int],
        metric: MetricSpec,
    ) -> MlpSweepResult:
        """Measure every MLP with exact interventions in bounded batched forwards."""
        import torch

        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            baseline = self._artifacts(run_id)
            baseline_metric = self.metric(run_id, metric)
            target_id = baseline_metric.target_token_id
            distractor_id = baseline_metric.distractor_token_id
            output_position = baseline_metric.position
            token_positions = self._resolve_positions(baseline, token_scope, positions)
            device = str(getattr(self.model.cfg, "device", self.model_spec.device))
            effects: list[MlpEffect] = []
            layers = list(range(self.architecture.n_layers))

            # Bounding the sweep batch keeps deeper models usable on modest GPUs/CPUs.
            for start in range(0, len(layers), 16):
                batch_layers = layers[start : start + 16]
                tokens = baseline.tokens.to(device).repeat(len(batch_layers), 1)
                hooks: list[tuple[str, Any]] = []
                for row, layer in enumerate(batch_layers):
                    reference = baseline.cache[f"blocks.{layer}.hook_mlp_out"].float().mean(dim=1)[0]

                    def ablate_layer(
                        result: Any,
                        hook: Any,
                        batch_row: int = row,
                        token_indices: tuple[int, ...] = tuple(token_positions),
                        reference_value: Any = reference,
                        ablation_kind: str = kind,
                    ) -> Any:
                        del hook
                        updated = result.clone()
                        if ablation_kind == "zero_ablation":
                            updated[batch_row, list(token_indices), :] = 0
                        else:
                            value = reference_value.to(device=updated.device, dtype=updated.dtype)
                            updated[batch_row, list(token_indices), :] = value
                        return updated

                    hooks.append((f"blocks.{layer}.hook_mlp_out", ablate_layer))

                logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks).detach().to("cpu").float()
                values = logits[:, output_position, target_id]
                if distractor_id is not None:
                    values = values - logits[:, output_position, distractor_id]
                for layer, value in zip(batch_layers, values.tolist(), strict=True):
                    effects.append(
                        MlpEffect(
                            componentId=f"blocks.{layer}.mlp",
                            layer=layer,
                            metricValue=float(value),
                            delta=float(value - baseline_metric.value),
                        )
                    )

            deltas = [effect.delta for effect in effects]
            return MlpSweepResult(
                runId=run_id,
                kind=kind,
                metric=baseline_metric,
                effects=effects,
                minimum=min(deltas, default=0.0),
                maximum=max(deltas, default=0.0),
                durationMs=round((time.perf_counter() - started) * 1000, 3),
            )

    def direct_attribution(self, run_id: str, metric: MetricSpec) -> AttributionResult:
        """Decompose a logit under the observed final-normalization scale.

        This is the standard direct-logit-attribution view: it is linear and useful
        for ranking writes to the residual stream, but it is deliberately not
        described as a causal or path-specific effect.
        """
        import torch

        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            artifacts = self._artifacts(run_id)
            metric_result = self.metric(run_id, metric)
            position = metric_result.position
            direction = self.model.W_U[:, metric_result.target_token_id].detach().to("cpu").float()
            if metric_result.distractor_token_id is not None:
                direction -= self.model.W_U[:, metric_result.distractor_token_id].detach().to("cpu").float()

            final_residual = artifacts.cache[
                f"blocks.{self.architecture.n_layers - 1}.hook_resid_post"
            ][0, position].float()
            normalization = str(getattr(self.model.cfg, "normalization_type", "LN"))
            centers = normalization.startswith("LN")
            scale_hook = artifacts.cache.get("ln_final.hook_scale")
            if scale_hook is not None:
                scale = float(scale_hook[0, position].float().reshape(-1)[0].item())
            else:
                normalized = final_residual - final_residual.mean() if centers else final_residual
                epsilon = float(getattr(self.model.cfg, "eps", 1e-5))
                scale = float((normalized.square().mean() + epsilon).sqrt().item())
            if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
                raise ValueError("Final-normalization scale is not finite")
            final_weight = getattr(getattr(self.model, "ln_final", None), "w", None)
            weight = final_weight.detach().to("cpu").float() if final_weight is not None else None

            def contribution(vector: Any) -> float:
                value = vector.detach().to("cpu").float()
                if centers:
                    value = value - value.mean()
                value = value / scale
                if weight is not None:
                    value = value * weight
                return float((value * direction).sum().item())

            effects: list[AttributionEffect] = []
            embed = artifacts.cache.get("hook_embed")
            if embed is not None:
                effects.append(
                    AttributionEffect(
                        componentId="embed",
                        label="Token embedding",
                        kind="embedding",
                        value=contribution(embed[0, position]),
                    )
                )
            positional = artifacts.cache.get("hook_pos_embed")
            if positional is not None:
                effects.append(
                    AttributionEffect(
                        componentId="pos_embed",
                        label="Positional embedding",
                        kind="embedding",
                        value=contribution(positional[0, position]),
                    )
                )
            for layer in range(self.architecture.n_layers):
                results = artifacts.cache[f"blocks.{layer}.attn.hook_result"][0, position]
                for head in range(self.architecture.n_heads):
                    effects.append(
                        AttributionEffect(
                            componentId=f"blocks.{layer}.attn.head.{head}",
                            label=f"L{layer}H{head}",
                            kind="head",
                            layer=layer,
                            head=head,
                            value=contribution(results[head]),
                        )
                    )
                effects.append(
                    AttributionEffect(
                        componentId=f"blocks.{layer}.mlp",
                        label=f"MLP L{layer}",
                        kind="mlp",
                        layer=layer,
                        value=contribution(artifacts.cache[f"blocks.{layer}.hook_mlp_out"][0, position]),
                    )
                )

            component_sum = sum(effect.value for effect in effects)
            remainder = metric_result.value - component_sum
            denominator = metric_result.value
            for effect in effects:
                effect.fraction = effect.value / denominator if abs(denominator) > 1e-8 else None
            effects.append(
                AttributionEffect(
                    componentId="unattributed_remainder",
                    label="Biases and unattributed remainder",
                    kind="remainder",
                    value=remainder,
                    fraction=remainder / denominator if abs(denominator) > 1e-8 else None,
                )
            )
            values = [effect.value for effect in effects]
            return AttributionResult(
                runId=run_id,
                metric=metric_result,
                effects=effects,
                componentSum=component_sum,
                remainder=remainder,
                minimum=min(values, default=0.0),
                maximum=max(values, default=0.0),
                durationMs=round((time.perf_counter() - started) * 1000, 3),
            )

    def attention(self, run_id: str, layer: int, head: int) -> AttentionResult:
        artifacts = self._artifacts(run_id)
        if layer < 0 or layer >= self.architecture.n_layers or head < 0 or head >= self.architecture.n_heads:
            raise ValueError("Attention coordinates are outside the loaded architecture")
        pattern_hook = f"blocks.{layer}.attn.hook_pattern"
        score_hook = f"blocks.{layer}.attn.hook_attn_scores"
        result_hook = f"blocks.{layer}.attn.hook_result"
        q_hook = f"blocks.{layer}.attn.hook_q"
        k_hook = f"blocks.{layer}.attn.hook_k"
        pattern = artifacts.cache[pattern_hook][0, head].float()
        scores = artifacts.cache[score_hook][0, head].float()
        scores = scores.nan_to_num(nan=0.0, posinf=10_000.0, neginf=-10_000.0)
        result_norms = artifacts.cache[result_hook][0, :, head].float().norm(dim=-1)
        q_norms = artifacts.cache[q_hook][0, :, head].float().norm(dim=-1)
        k_norms = artifacts.cache[k_hook][0, :, head].float().norm(dim=-1)
        labels = [token.display for token in artifacts.record.tokens]
        return AttentionResult(
            runId=run_id,
            componentId=f"blocks.{layer}.attn.head.{head}",
            layer=layer,
            head=head,
            patternHook=pattern_hook,
            scoreHook=score_hook,
            queryTokens=labels,
            keyTokens=labels,
            pattern=pattern.tolist(),
            scores=scores.tolist(),
            resultNorms=result_norms.tolist(),
            qNorms=q_norms.tolist(),
            kNorms=k_norms.tolist(),
        )

    def activation(self, run_id: str, component_id: str) -> ActivationSeries:
        artifacts = self._artifacts(run_id)
        node = self._component_index.get(component_id)
        if node is None:
            raise ValueError(f"Unknown component: {component_id}")
        hook_name = self._activation_hook(node, artifacts.cache)
        tensor = artifacts.cache[hook_name].float()
        head_index = node.head
        if head_index is not None and tensor.ndim >= 4:
            tensor = tensor[:, head_index] if "hook_pattern" in hook_name or "hook_attn_scores" in hook_name else tensor[:, :, head_index]
        tensor = tensor[0]
        values = tensor.abs() if tensor.ndim == 1 else tensor.reshape(tensor.shape[0], -1).norm(dim=-1)
        return ActivationSeries(
            runId=run_id,
            componentId=component_id,
            hookName=hook_name,
            measure="l2_norm",
            tokenLabels=[token.display for token in artifacts.record.tokens],
            values=values.tolist(),
        )

    def residual_stream(
        self,
        run_id: str,
        *,
        position: int = -1,
        target_token_id: int | None = None,
    ) -> ResidualStreamResult:
        artifacts = self._artifacts(run_id)
        resolved_position = self._resolve_output_position(artifacts, position)
        if target_token_id is None:
            target_token_id = artifacts.record.top_predictions[0].token_id

        point_specs: list[tuple[int, str, Any]] = []
        device = str(getattr(self.model.cfg, "device", self.model_spec.device))
        import torch

        with torch.inference_mode():
            for layer in range(self.architecture.n_layers):
                for stage in ("pre", "mid", "post"):
                    hook_name = f"blocks.{layer}.hook_resid_{stage}"
                    if hook_name not in artifacts.cache:
                        # Parallel attention/MLP blocks have no real mid-residual
                        # activation.  Omitting it preserves the model's actual
                        # computational graph instead of inventing a serial state.
                        continue
                    resid = artifacts.cache[hook_name][0, resolved_position].float()
                    point_specs.append((layer, stage, resid))
            stacked = torch.stack([spec[2] for spec in point_specs]).to(device)[:, None, :]
            lens_logits = self.model.unembed(self.model.ln_final(stacked))[:, 0].to("cpu").float()
            lens_probabilities = lens_logits.softmax(dim=-1)
            target_logits = lens_logits[:, target_token_id]
            entropies = -(lens_probabilities * lens_probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            top_probabilities, top_ids = lens_probabilities.topk(min(5, lens_probabilities.shape[-1]), dim=-1)
        points = []
        for index, (layer, stage, resid) in enumerate(point_specs):
            predictions = []
            for probability, token_id in zip(
                top_probabilities[index].tolist(), top_ids[index].tolist(), strict=True
            ):
                text = self._decode_token(int(token_id))
                predictions.append(
                    Prediction(
                        tokenId=int(token_id),
                        text=text,
                        display=self._display_token(text),
                        logit=float(lens_logits[index, token_id].item()),
                        probability=float(probability),
                    )
                )
            points.append(
                ResidualPoint(
                    layer=layer,
                    stage=stage,
                    norm=float(resid.norm().item()),
                    targetLogit=float(target_logits[index].item()),
                    entropy=float(entropies[index].item()),
                    topPredictions=predictions,
                )
            )
        return ResidualStreamResult(
            runId=run_id,
            position=resolved_position,
            targetTokenId=target_token_id,
            targetToken=self._decode_token(target_token_id),
            points=points,
        )

    def compare(self, baseline_run_id: str, intervened_run_id: str, *, top_k: int = 10) -> RunComparison:
        import torch

        baseline = self._artifacts(baseline_run_id)
        intervened = self._artifacts(intervened_run_id)
        if baseline.record.prompt != intervened.record.prompt:
            raise ValueError("Run comparison requires the same prompt")
        baseline_logits = baseline.logits[0, -1].float()
        intervened_logits = intervened.logits[0, -1].float()
        baseline_probs = baseline_logits.softmax(dim=-1)
        intervened_probs = intervened_logits.softmax(dim=-1)
        ids = torch.cat((baseline_logits.topk(top_k).indices, intervened_logits.topk(top_k).indices)).unique()
        rows = []
        for token_id_value in ids.tolist():
            baseline_logit = float(baseline_logits[token_id_value].item())
            intervened_logit = float(intervened_logits[token_id_value].item())
            baseline_probability = float(baseline_probs[token_id_value].item())
            intervened_probability = float(intervened_probs[token_id_value].item())
            text = self._decode_token(token_id_value)
            rows.append(
                TokenComparison(
                    tokenId=token_id_value,
                    text=text,
                    display=self._display_token(text),
                    baselineLogit=baseline_logit,
                    intervenedLogit=intervened_logit,
                    deltaLogit=intervened_logit - baseline_logit,
                    baselineProbability=baseline_probability,
                    intervenedProbability=intervened_probability,
                    deltaProbability=intervened_probability - baseline_probability,
                )
            )
        rows.sort(key=lambda row: max(row.baseline_probability, row.intervened_probability), reverse=True)
        baseline_top_id = int(baseline_logits.argmax().item())
        kl = torch.sum(baseline_probs * (baseline_probs.clamp_min(1e-12).log() - intervened_probs.clamp_min(1e-12).log()))
        return RunComparison(
            baselineRunId=baseline_run_id,
            intervenedRunId=intervened_run_id,
            baselineTopToken=self._decode_token(baseline_top_id),
            baselineTopTokenDelta=float((intervened_logits[baseline_top_id] - baseline_logits[baseline_top_id]).item()),
            klDivergence=float(kl.item()),
            tokens=rows,
        )

    def _effect(self, baseline_run_id: str, intervened_run_id: str, metric: MetricSpec) -> CausalEffect:
        baseline = self.metric(baseline_run_id, metric)
        intervened = self.metric(intervened_run_id, metric)
        return CausalEffect(baseline=baseline, intervened=intervened, delta=intervened.value - baseline.value)

    def _execute(
        self,
        *,
        tokens: Any,
        prompt: str,
        kind: str,
        label: str,
        interventions: list[InterventionSpec],
        parent_run_id: str | None,
        top_k: int,
        seed: int,
        hooks: Iterable[tuple[str, Any]] | None = None,
    ) -> RunArtifacts:
        import torch

        torch.manual_seed(seed)
        started = time.perf_counter()
        hook_context = self.model.hooks(fwd_hooks=list(hooks)) if hooks else nullcontext()
        with torch.inference_mode(), hook_context:
            logits, cache = self.model.run_with_cache(tokens, names_filter=self._cache_filter)
        duration_ms = (time.perf_counter() - started) * 1000
        cache_dict = {
            name: value.detach().to("cpu")
            for name, value in cache.cache_dict.items()
            if self._cache_filter(name)
        }
        cpu_tokens = tokens.detach().to("cpu")
        cpu_logits = logits.detach().to("cpu")
        cache_bytes = sum(int(value.nelement() * value.element_size()) for value in cache_dict.values())
        run_id = f"run_{uuid4().hex[:12]}"
        # Decode the exact token tensor used for this run.  TransformerLens'
        # ``to_str_tokens`` may apply its model-specific default BOS policy a
        # second time (notably for Pythia), which can produce one more display
        # token than there are activation positions.
        token_strings = [self._decode_token(int(token_id)) for token_id in cpu_tokens[0].tolist()]
        record = RunRecord(
            id=run_id,
            kind=kind,
            label=label,
            modelId=self.model_spec.id,
            modelRevision=self.model_spec.revision,
            prompt=prompt,
            tokens=self._token_rows(cpu_tokens, cpu_logits, token_strings),
            topPredictions=self._predictions(cpu_logits[0, -1], top_k),
            requestedActivations=sorted(cache_dict),
            interventions=interventions,
            parentRunId=parent_run_id,
            device=str(getattr(self.model.cfg, "device", self.model_spec.device)),
            dtype=str(getattr(self.model.cfg, "dtype", self.model_spec.dtype)),
            seed=seed,
            durationMs=round(duration_ms, 3),
            cacheBytes=cache_bytes,
            createdAt=datetime.now(timezone.utc),
            provenance={
                "backend": "transformer_lens",
                "exact": True,
                "cachePolicy": "interactive-core",
                "interventionCount": len(interventions),
            },
        )
        artifacts = RunArtifacts(record=record, tokens=cpu_tokens, logits=cpu_logits, cache=cache_dict)
        self._runs[run_id] = artifacts
        self._order.append(run_id)
        return artifacts

    def _token_rows(self, tokens: Any, logits: Any, token_strings: list[str]) -> list[TokenRecord]:
        probabilities = logits[0].float().softmax(dim=-1)
        top_probabilities, top_ids = probabilities.max(dim=-1)
        rows = []
        for position, (token_id, text) in enumerate(zip(tokens[0].tolist(), token_strings, strict=True)):
            next_id = int(top_ids[position].item())
            rows.append(
                TokenRecord(
                    position=position,
                    tokenId=int(token_id),
                    text=text,
                    display=self._display_token(text),
                    nextToken=self._display_token(self._decode_token(next_id)),
                    nextTokenProbability=float(top_probabilities[position].item()),
                )
            )
        return rows

    def _predictions(self, logits: Any, top_k: int) -> list[Prediction]:
        probabilities = logits.float().softmax(dim=-1)
        values, indices = logits.float().topk(top_k)
        return [
            Prediction(
                tokenId=int(token_id),
                text=(text := self._decode_token(int(token_id))),
                display=self._display_token(text),
                logit=float(logit),
                probability=float(probabilities[int(token_id)].item()),
            )
            for logit, token_id in zip(values.tolist(), indices.tolist(), strict=True)
        ]

    def _single_token_id(self, text: str | None) -> int:
        if text is None:
            raise ValueError("A token value is required")
        token_ids = self.model.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            displays = [self._display_token(self._decode_token(int(token_id))) for token_id in token_ids]
            raise ValueError(
                f"Metric token {text!r} encodes to {len(token_ids)} tokens ({', '.join(displays) or 'none'}); enter exactly one model token"
            )
        return int(token_ids[0])

    def _resolve_output_position(self, artifacts: RunArtifacts, position: int) -> int:
        token_count = len(artifacts.record.tokens)
        resolved = position if position >= 0 else token_count + position
        if resolved < 0 or resolved >= token_count:
            raise ValueError("Metric position is outside the prompt")
        return resolved

    def _resolve_positions(self, artifacts: RunArtifacts, token_scope: str, positions: list[int]) -> list[int]:
        token_count = len(artifacts.record.tokens)
        if token_scope == "all":
            return list(range(token_count))
        if token_scope != "positions" or not positions:
            raise ValueError("Choose one or more token positions")
        resolved = sorted(set(positions))
        if resolved[0] < 0 or resolved[-1] >= token_count:
            raise ValueError("An intervention position is outside the prompt")
        return resolved

    def _partition_intervenable_components(
        self, component_ids: list[str]
    ) -> tuple[dict[int, set[int]], set[int], dict[int, set[int]]]:
        heads: dict[int, set[int]] = {}
        mlps: set[int] = set()
        neurons: dict[int, set[int]] = {}
        for component_id in component_ids:
            match = HEAD_ID.fullmatch(component_id)
            if match:
                layer = int(match.group("layer"))
                head = int(match.group("head"))
                if layer >= self.architecture.n_layers or head >= self.architecture.n_heads:
                    raise ValueError(f"Component is outside the loaded architecture: {component_id}")
                heads.setdefault(layer, set()).add(head)
                continue
            match = MLP_ID.fullmatch(component_id)
            if match:
                layer = int(match.group("layer"))
                if layer >= self.architecture.n_layers:
                    raise ValueError(f"Component is outside the loaded architecture: {component_id}")
                mlps.add(layer)
                continue
            match = NEURON_ID.fullmatch(component_id)
            if match:
                layer = int(match.group("layer"))
                neuron = int(match.group("neuron"))
                d_mlp = self.architecture.d_mlp
                if layer >= self.architecture.n_layers or d_mlp is None or neuron >= d_mlp:
                    raise ValueError(f"Component is outside the loaded architecture: {component_id}")
                neurons.setdefault(layer, set()).add(neuron)
                continue
            raise ValueError(
                "Causal interventions support attention-head IDs, whole-MLP IDs, "
                f"and MLP-neuron IDs; received: {component_id}"
            )
        return heads, mlps, neurons

    def _decode_token(self, token_id: int) -> str:
        return str(self.model.tokenizer.decode([token_id]))

    @staticmethod
    def _display_token(text: str) -> str:
        return text.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥") or "∅"

    @staticmethod
    def _display_component(component_id: str) -> str:
        match = HEAD_ID.fullmatch(component_id)
        if match:
            return f"L{match.group('layer')}H{match.group('head')}"
        match = MLP_ID.fullmatch(component_id)
        if match:
            return f"MLP L{match.group('layer')}"
        match = NEURON_ID.fullmatch(component_id)
        if match:
            return f"L{match.group('layer')}N{match.group('neuron')}"
        return component_id

    def _artifacts(self, run_id: str) -> RunArtifacts:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run: {run_id}") from exc

    @classmethod
    def _cache_filter(cls, name: str) -> bool:
        return any(name.endswith(suffix) for suffix in cls._cache_suffixes)

    @staticmethod
    def _index_components(graph: ArchitectureGraph) -> dict[str, ComponentNode]:
        index: dict[str, ComponentNode] = {}

        def add(node: ComponentNode | None) -> None:
            if node is None:
                return
            index[node.id] = node
            for child in node.children:
                add(child)

        add(graph.embedding)
        add(graph.positional_embedding)
        add(graph.final_norm)
        add(graph.unembedding)
        for layer in graph.layers:
            for node in (
                layer.residual_pre,
                layer.norm1,
                layer.attention,
                *layer.heads,
                layer.residual_mid,
                layer.norm2,
                layer.mlp,
                layer.residual_post,
            ):
                add(node)
        return index

    @staticmethod
    def _activation_hook(node: ComponentNode, cache: dict[str, Any]) -> str:
        preferred = list(node.activation_points)
        if node.kind == "attention":
            preferred.reverse()
        for hook in preferred:
            if hook in cache:
                return hook
        raise ValueError(f"No interactive activation was cached for {node.id}")
