"""Run Kannaadi's research primitives against a real TransformerLens model.

This is intentionally separate from the fast unit suite. It downloads/loads weights,
runs exact interventions, and exits non-zero if the result contracts do not reconcile.
"""

from __future__ import annotations

import argparse
import json

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.code_execution import CodeExecutionRequest, CodeSession
from kannaadi.domain import MetricSpec, ModelSpec, PatchMapping
from kannaadi.experiments import ExperimentEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    spec = ModelSpec(
        id=args.model.replace("/", "-").lower(),
        display_name=args.model,
        backend="transformer_lens",
        repository=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    adapter = TransformerLensAdapter()
    adapter.load(spec, lambda stage, message: print(f"[{stage}] {message}", flush=True))
    architecture = adapter.architecture()
    engine = ExperimentEngine(adapter, architecture, spec)
    metric = MetricSpec(targetToken=" Paris", distractorToken=" Berlin", position=-1)

    clean = engine.run_prompt("The capital of France is")
    mlp = engine.ablate(clean.id, ["blocks.5.mlp"], kind="zero_ablation", metric=metric)
    neuron = engine.ablate(
        clean.id,
        ["blocks.5.mlp.neuron.0"],
        kind="mean_ablation",
        metric=metric,
    )
    contrast = engine.run_contrast("The capital of France is", "The capital of Germany is")
    mappings = [
        PatchMapping(sourcePosition=pair.source_position, destinationPosition=pair.destination_position)
        for pair in contrast.alignment.pairs
        if pair.source_position is not None and pair.destination_position is not None
    ]
    patch = engine.patch(
        contrast.corrupted_run.id,
        contrast.clean_run.id,
        ["blocks.5.mlp"],
        mappings=mappings,
        metric=metric,
    )
    sweep = engine.mlp_sweep(
        clean.id,
        kind="zero_ablation",
        token_scope="all",
        positions=[],
        metric=metric,
    )
    attribution = engine.direct_attribution(clean.id, metric)
    residual = engine.residual_stream(clean.id)
    dataset = engine.dataset_ablation(
        [clean.id, contrast.clean_run.id],
        ["blocks.5.mlp"],
        kind="zero_ablation",
        metric=metric,
    )
    code_session = CodeSession(engine, adapter, architecture, spec.id)
    code_result = code_session.execute(CodeExecutionRequest(
        code="kannaadi.table([{'run': active_run.label, 'cache_points': len(cache)}], title='Live cache')",
        cellId="real-model-smoke",
        trusted=True,
        activeRunId=clean.id,
        selection=["blocks.5.mlp"],
    ))

    assert mlp.effect is not None and neuron.effect is not None and patch.effect is not None
    assert len(sweep.effects) == architecture.n_layers
    assert {effect.component_id for effect in sweep.effects} == {
        f"blocks.{layer}.mlp" for layer in range(architecture.n_layers)
    }
    assert any(effect.kind == "head" for effect in attribution.effects)
    assert any(effect.kind == "mlp" for effect in attribution.effects)
    assert abs(
        attribution.component_sum + attribution.remainder - attribution.metric.value
    ) < 1e-6
    expected_residual_points = architecture.n_layers * (2 if architecture.block_topology == "parallel" else 3)
    assert len(residual.points) == expected_residual_points
    assert all(point.top_predictions for point in residual.points)
    assert all(point.entropy >= 0 for point in residual.points)
    assert dataset.summary.completed_count == 2
    assert dataset.summary.failed_count == 0
    assert all(row.intervened_run is not None for row in dataset.rows)
    assert architecture.flow.edges
    assert code_result.status == "complete"
    assert code_result.artifact.kind == "table"
    assert code_result.artifact.data[0]["cache_points"] > 0

    print(json.dumps({
        "model": architecture.model_id,
        "layers": architecture.n_layers,
        "heads": architecture.n_heads,
        "blockTopology": architecture.block_topology,
        "flowModules": len(architecture.flow.modules),
        "flowEdges": len(architecture.flow.edges),
        "cleanRun": clean.id,
        "mlpAblationDelta": mlp.effect.delta,
        "neuronAblationDelta": neuron.effect.delta,
        "mlpPatchDelta": patch.effect.delta,
        "mlpSweepEffects": len(sweep.effects),
        "attributionEffects": len(attribution.effects),
        "attributionMetric": attribution.metric.value,
        "attributionReconciled": attribution.component_sum + attribution.remainder,
        "residualReadoutPoints": len(residual.points),
        "datasetMeanDelta": dataset.summary.mean_delta,
        "codeArtifact": code_result.artifact.title,
    }, indent=2))


if __name__ == "__main__":
    main()
