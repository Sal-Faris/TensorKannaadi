from kannaadi.domain import TokenRecord
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
