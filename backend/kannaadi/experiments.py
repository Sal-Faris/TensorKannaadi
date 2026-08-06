from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import threading
import time
from typing import Any, Iterable
from uuid import uuid4

from kannaadi.domain import (
    ActivationSeries,
    ArchitectureGraph,
    AttentionResult,
    ComponentNode,
    InterventionSpec,
    Prediction,
    ResidualPoint,
    ResidualStreamResult,
    RunComparison,
    RunRecord,
    TokenComparison,
    TokenRecord,
)


HEAD_ID = re.compile(r"blocks\.(?P<layer>\d+)\.attn\.head\.(?P<head>\d+)$")


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

    def run_clean(self, prompt: str, *, top_k: int = 10, seed: int = 0) -> RunRecord:
        with self._lock:
            tokens = self.adapter.tokenize([prompt])
            if int(tokens.shape[-1]) > self.max_prompt_tokens:
                raise ValueError(
                    f"Prompt tokenized to {int(tokens.shape[-1])} tokens; the interactive cache limit is {self.max_prompt_tokens}"
                )
            return self._execute(
                tokens=tokens,
                prompt=prompt,
                kind="clean",
                label="Clean run",
                interventions=[],
                parent_run_id=None,
                top_k=top_k,
                seed=seed,
            ).record

    def zero_ablate(
        self,
        baseline_run_id: str,
        component_ids: list[str],
        *,
        token_scope: str = "all",
        top_k: int = 10,
    ) -> RunRecord:
        with self._lock:
            if token_scope != "all":
                raise ValueError("Only all-token zero ablation is currently supported")
            baseline = self._artifacts(baseline_run_id)
            grouped: dict[int, set[int]] = {}
            for component_id in component_ids:
                match = HEAD_ID.fullmatch(component_id)
                if not match:
                    raise ValueError(f"Zero ablation currently requires an attention-head ID, received: {component_id}")
                layer = int(match.group("layer"))
                head = int(match.group("head"))
                if layer >= self.architecture.n_layers or head >= self.architecture.n_heads:
                    raise ValueError(f"Component is outside the loaded architecture: {component_id}")
                grouped.setdefault(layer, set()).add(head)

            hooks = []
            for layer, heads in grouped.items():
                selected_heads = tuple(sorted(heads))

                def zero_selected(result: Any, hook: Any, indices: tuple[int, ...] = selected_heads) -> Any:
                    del hook
                    updated = result.clone()
                    updated[:, :, list(indices), :] = 0
                    return updated

                hooks.append((f"blocks.{layer}.attn.hook_result", zero_selected))

            intervention = InterventionSpec(componentIds=component_ids, tokenScope="all")
            label = "Ablate " + ", ".join(self._display_component(component_id) for component_id in component_ids)
            device = str(getattr(self.model.cfg, "device", self.model_spec.device))
            tokens = baseline.tokens.to(device)
            return self._execute(
                tokens=tokens,
                prompt=baseline.record.prompt,
                kind="intervened",
                label=label,
                interventions=[intervention],
                parent_run_id=baseline_run_id,
                top_k=top_k,
                seed=baseline.record.seed,
                hooks=hooks,
            ).record

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
            if "hook_pattern" in hook_name or "hook_attn_scores" in hook_name:
                tensor = tensor[:, head_index]
            else:
                tensor = tensor[:, :, head_index]
        tensor = tensor[0]
        if tensor.ndim == 1:
            values = tensor.abs()
        else:
            values = tensor.reshape(tensor.shape[0], -1).norm(dim=-1)
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
        token_count = len(artifacts.record.tokens)
        resolved_position = position if position >= 0 else token_count + position
        if resolved_position < 0 or resolved_position >= token_count:
            raise ValueError("Residual-stream position is outside the prompt")
        if target_token_id is None:
            target_token_id = artifacts.record.top_predictions[0].token_id

        point_specs: list[tuple[int, str, Any]] = []
        device = str(getattr(self.model.cfg, "device", self.model_spec.device))
        import torch

        with torch.inference_mode():
            for layer in range(self.architecture.n_layers):
                for stage in ("pre", "mid", "post"):
                    resid = artifacts.cache[f"blocks.{layer}.hook_resid_{stage}"][0, resolved_position].float()
                    point_specs.append((layer, stage, resid))
            stacked = torch.stack([spec[2] for spec in point_specs]).to(device)[:, None, :]
            target_logits = self.model.unembed(self.model.ln_final(stacked))[:, 0, target_token_id].to("cpu")
        points = [
            ResidualPoint(
                layer=layer,
                stage=stage,
                norm=float(resid.norm().item()),
                targetLogit=float(target_logits[index].item()),
            )
            for index, (layer, stage, resid) in enumerate(point_specs)
        ]
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
        token_strings = [str(token) for token in self.model.to_str_tokens(cpu_tokens[0])]
        token_rows = self._token_rows(cpu_tokens, cpu_logits, token_strings)
        predictions = self._predictions(cpu_logits[0, -1], top_k)
        requested = sorted(cache_dict)
        record = RunRecord(
            id=run_id,
            kind=kind,
            label=label,
            modelId=self.model_spec.id,
            modelRevision=self.model_spec.revision,
            prompt=prompt,
            tokens=token_rows,
            topPredictions=predictions,
            requestedActivations=requested,
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
                "tokenScope": "all",
                "cachePolicy": "interactive-core",
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

    def _decode_token(self, token_id: int) -> str:
        return str(self.model.tokenizer.decode([token_id]))

    @staticmethod
    def _display_token(text: str) -> str:
        return text.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥") or "∅"

    @staticmethod
    def _display_component(component_id: str) -> str:
        match = HEAD_ID.fullmatch(component_id)
        return f"L{match.group('layer')}H{match.group('head')}" if match else component_id

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
