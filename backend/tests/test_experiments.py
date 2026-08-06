import pytest

from kannaadi.adapters import TransformerLensAdapter
from kannaadi.domain import ModelSpec, TokenRecord
from kannaadi.experiments import ExperimentEngine


def token(position: int, token_id: int, display: str) -> TokenRecord:
    return TokenRecord(position=position, tokenId=token_id, text=display, display=display)


def test_minimum_edit_alignment_preserves_positions_and_marks_substitutions() -> None:
    source = [token(0, 1, "The"), token(1, 2, "·capital"), token(2, 3, "·France")]
    destination = [token(0, 1, "The"), token(1, 2, "·capital"), token(2, 4, "·Germany")]

    alignment = ExperimentEngine.minimum_edit_alignment(source, destination)

    assert [pair.status for pair in alignment] == ["exact", "exact", "substitution"]
    assert alignment[-1].source_position == 2
    assert alignment[-1].destination_position == 2


def test_minimum_edit_alignment_represents_inserted_tokens_without_guessing() -> None:
    source = [token(0, 1, "A"), token(1, 3, "C")]
    destination = [token(0, 1, "A"), token(1, 2, "B"), token(2, 3, "C")]

    alignment = ExperimentEngine.minimum_edit_alignment(source, destination)

    assert [pair.status for pair in alignment] == ["exact", "source_gap", "exact"]
    assert alignment[1].source_position is None
    assert alignment[1].destination_position == 1


def test_intervention_ids_cover_heads_whole_mlps_and_individual_neurons() -> None:
    architecture = TransformerLensAdapter.from_dimensions(
        ModelSpec(id="test", display_name="Test", backend="transformer_lens"),
        n_layers=2,
        n_heads=3,
        d_model=12,
        d_head=4,
        d_mlp=24,
        vocabulary_size=101,
    )
    engine = ExperimentEngine(object(), architecture, object())

    heads, mlps, neurons = engine._partition_intervenable_components([
        "blocks.1.attn.head.2",
        "blocks.0.mlp",
        "blocks.1.mlp.neuron.23",
    ])

    assert heads == {1: {2}}
    assert mlps == {0}
    assert neurons == {1: {23}}
    with pytest.raises(ValueError, match="outside the loaded architecture"):
        engine._partition_intervenable_components(["blocks.1.mlp.neuron.24"])
    with pytest.raises(ValueError, match="support attention-head IDs"):
        engine._partition_intervenable_components(["blocks.0.attn"])


def test_dataset_summary_keeps_effect_size_spread_and_direction_auditable() -> None:
    summary = ExperimentEngine._dataset_summary(4, [-2.0, -1.0, 0.5])

    assert summary.requested_count == 4
    assert summary.completed_count == 3
    assert summary.failed_count == 1
    assert summary.mean_delta == pytest.approx(-5 / 6)
    assert summary.median_delta == -1.0
    assert summary.minimum_delta == -2.0
    assert summary.maximum_delta == 0.5
    assert summary.mean_absolute_delta == pytest.approx(7 / 6)
    assert summary.direction_consistency == pytest.approx(2 / 3)
    assert summary.standard_deviation is not None


def test_empty_dataset_summary_reports_missing_statistics_instead_of_inventing_zeroes() -> None:
    summary = ExperimentEngine._dataset_summary(2, [])

    assert summary.completed_count == 0
    assert summary.failed_count == 2
    assert summary.mean_delta is None
    assert summary.direction_consistency is None
