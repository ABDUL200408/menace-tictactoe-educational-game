"""
test_menace.py

Unit tests for src.menace.

These tests verify that MENACE:
    - creates matchboxes correctly
    - chooses legal moves
    - supports safe symmetry reduction
    - maps canonical moves back to original board moves
    - records decisions
    - updates bead counts after win/loss/draw
    - explains decisions for educational use
    - serialises and deserialises correctly.

Run:
    pytest tests/test_menace.py
"""

from __future__ import annotations

import random

import pytest

from src.board import Board
from src.menace import Matchbox, MENACEDecision, MENACEPlayer


def test_matchbox_from_board_creates_beads_for_legal_moves_without_symmetry() -> None:
    board = Board.from_string("X O      ")
    matchbox = Matchbox.from_board(board, initial_beads=3, use_symmetry=False)

    assert matchbox.state == board.to_string()
    assert set(matchbox.beads.keys()) == set(board.available_moves())
    assert all(count == 3 for count in matchbox.beads.values())


def test_matchbox_from_board_creates_canonical_state_with_symmetry() -> None:
    board = Board.from_string("X O      ")
    matchbox = Matchbox.from_board(board, initial_beads=3, use_symmetry=True)

    assert matchbox.state == board.canonical_state()
    assert len(matchbox.beads) == len(board.available_moves())
    assert all(count == 3 for count in matchbox.beads.values())


def test_matchbox_rejects_terminal_board() -> None:
    board = Board.from_string("XXXOO    ")

    with pytest.raises(ValueError):
        Matchbox.from_board(board)


def test_matchbox_probabilities_sum_to_one() -> None:
    matchbox = Matchbox(state="test", beads={0: 1, 1: 3, 2: 6})
    probabilities = matchbox.probabilities()

    assert pytest.approx(sum(probabilities.values())) == 1.0
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_matchbox_choose_move_returns_legal_move() -> None:
    matchbox = Matchbox(state="test", beads={0: 1, 4: 10})
    rng = random.Random(42)

    move = matchbox.choose_move(rng, legal_moves=[0, 4])

    assert move in {0, 4}


def test_matchbox_choose_move_filters_illegal_stale_moves() -> None:
    matchbox = Matchbox(state="test", beads={0: 100, 4: 1})
    rng = random.Random(42)

    move = matchbox.choose_move(rng, legal_moves=[4])

    assert move == 4
    assert set(matchbox.beads.keys()) == {4}


def test_matchbox_sync_adds_missing_legal_moves() -> None:
    matchbox = Matchbox(state="test", beads={0: 3})
    matchbox.sync_with_legal_moves(legal_moves=[0, 1], initial_beads=3)

    assert set(matchbox.beads.keys()) == {0, 1}
    assert matchbox.beads[1] == 3


def test_matchbox_update_move_adds_beads() -> None:
    matchbox = Matchbox(state="test", beads={0: 3})

    matchbox.update_move(0, 2)

    assert matchbox.beads[0] == 5


def test_matchbox_update_move_does_not_go_below_minimum() -> None:
    matchbox = Matchbox(state="test", beads={0: 3}, minimum_beads=1)

    matchbox.update_move(0, -100)

    assert matchbox.beads[0] == 1


def test_matchbox_update_unknown_move_is_safe_noop() -> None:
    matchbox = Matchbox(state="test", beads={0: 3})

    matchbox.update_move(1, 2)

    assert matchbox.beads == {0: 3}


def test_matchbox_explain_contains_beads_and_probabilities() -> None:
    matchbox = Matchbox(state="test", beads={0: 1, 1: 3})
    explanation = matchbox.explain()

    assert "Matchbox state" in explanation
    assert "Total beads" in explanation
    assert "Move 0" in explanation
    assert "Move 1" in explanation
    assert "chance" in explanation


def test_matchbox_to_dict_and_from_dict_round_trip() -> None:
    original = Matchbox(state="test", beads={0: 2, 4: 5}, minimum_beads=1)

    restored = Matchbox.from_dict(original.to_dict())

    assert restored.state == original.state
    assert restored.beads == original.beads
    assert restored.minimum_beads == original.minimum_beads


def test_menace_initialises_with_valid_defaults() -> None:
    menace = MENACEPlayer(mark="O", seed=42)

    assert menace.mark == "O"
    assert menace.name == "MENACE"
    assert menace.games_played == 0
    assert menace.matchboxes == {}
    assert menace.use_symmetry is True


def test_menace_rejects_invalid_reward_configuration() -> None:
    with pytest.raises(ValueError):
        MENACEPlayer(mark="O", reward_win=-1)

    with pytest.raises(ValueError):
        MENACEPlayer(mark="O", reward_draw=-1)

    with pytest.raises(ValueError):
        MENACEPlayer(mark="O", penalty_loss=1)


def test_menace_choose_move_creates_matchbox_and_records_decision() -> None:
    menace = MENACEPlayer(mark="O", seed=42, use_symmetry=False)
    board = Board.from_string("X        ")

    move = menace.choose_move(board)

    assert move in board.available_moves()
    assert len(menace.matchboxes) == 1
    assert len(menace.decisions_this_game) == 1
    assert isinstance(menace.decisions_this_game[0], MENACEDecision)
    assert menace.decisions_this_game[0].move == move


def test_menace_choose_move_with_explanation_returns_text() -> None:
    menace = MENACEPlayer(mark="O", seed=42, use_symmetry=False)
    board = Board.from_string("X        ")

    move, explanation = menace.choose_move_with_explanation(board)

    assert move in board.available_moves()
    assert "MENACE used matchbox state" in explanation
    assert "MENACE selected original move" in explanation
    assert "bead counts" in explanation


def test_menace_cannot_move_on_terminal_board() -> None:
    menace = MENACEPlayer(mark="O")
    board = Board.from_string("XXXOO    ")

    with pytest.raises(ValueError):
        menace.choose_move(board)


def test_menace_learn_from_win_increases_selected_canonical_move_beads() -> None:
    menace = MENACEPlayer(
        mark="O",
        seed=42,
        use_symmetry=True,
        initial_beads=3,
        reward_win=3,
    )
    board = Board.from_string("X        ")

    menace.choose_move(board)
    decision = menace.decisions_this_game[0]
    before = menace.matchboxes[decision.state].beads[decision.canonical_move]

    menace.learn_from_result("win")

    after = menace.matchboxes[decision.state].beads[decision.canonical_move]
    assert after == before + 3
    assert menace.games_played == 1
    assert menace.wins == 1
    assert menace.decisions_this_game == []


def test_menace_learn_from_draw_increases_selected_move_slightly() -> None:
    menace = MENACEPlayer(
        mark="O",
        seed=42,
        use_symmetry=True,
        initial_beads=3,
        reward_draw=1,
    )
    board = Board.from_string("X        ")

    menace.choose_move(board)
    decision = menace.decisions_this_game[0]
    before = menace.matchboxes[decision.state].beads[decision.canonical_move]

    menace.learn_from_result("draw")

    after = menace.matchboxes[decision.state].beads[decision.canonical_move]
    assert after == before + 1
    assert menace.draws == 1


def test_menace_learn_from_loss_decreases_selected_move_beads() -> None:
    menace = MENACEPlayer(
        mark="O",
        seed=42,
        use_symmetry=True,
        initial_beads=5,
        penalty_loss=-2,
        minimum_beads=1,
    )
    board = Board.from_string("X        ")

    menace.choose_move(board)
    decision = menace.decisions_this_game[0]
    before = menace.matchboxes[decision.state].beads[decision.canonical_move]

    menace.learn_from_result("loss")

    after = menace.matchboxes[decision.state].beads[decision.canonical_move]
    assert after == max(1, before - 2)
    assert menace.losses == 1


def test_menace_rejects_unknown_learning_result() -> None:
    menace = MENACEPlayer(mark="O")

    with pytest.raises(ValueError):
        menace.learn_from_result("unknown")


def test_reset_game_memory_clears_decisions_only() -> None:
    menace = MENACEPlayer(mark="O", seed=42)
    board = Board.from_string("X        ")

    menace.choose_move(board)
    assert menace.decisions_this_game

    menace.reset_game_memory()

    assert menace.decisions_this_game == []
    assert menace.matchboxes


def test_reset_learning_clears_all_training_state() -> None:
    menace = MENACEPlayer(mark="O", seed=42)
    board = Board.from_string("X        ")

    menace.choose_move(board)
    menace.learn_from_result("win")

    menace.reset_learning()

    assert menace.matchboxes == {}
    assert menace.decisions_this_game == []
    assert menace.games_played == 0
    assert menace.wins == 0
    assert menace.losses == 0
    assert menace.draws == 0


def test_explain_state_creates_matchbox_explanation() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=True)
    board = Board.from_string("X        ")

    explanation = menace.explain_state(board)

    assert "Symmetry reduction is enabled" in explanation
    assert "Matchbox state" in explanation
    assert "Move probabilities" in explanation


def test_explain_state_for_terminal_board_is_clear() -> None:
    menace = MENACEPlayer(mark="O")
    board = Board.from_string("XXXOO    ")

    explanation = menace.explain_state(board)

    assert "terminal" in explanation.lower()


def test_training_summary_reports_rates_counts_and_symmetry() -> None:
    menace = MENACEPlayer(mark="O", seed=42, use_symmetry=True)
    board = Board.from_string("X        ")

    menace.choose_move(board)
    menace.learn_from_result("win")

    summary = menace.training_summary()

    assert summary["games_played"] == 1
    assert summary["wins"] == 1
    assert summary["win_rate"] == 1.0
    assert summary["matchboxes"] >= 1
    assert summary["total_beads"] >= 1
    assert summary["use_symmetry"] is True


def test_last_decision_explanation_before_any_move() -> None:
    menace = MENACEPlayer(mark="O")

    explanation = menace.last_decision_explanation()

    assert "has not made a decision" in explanation


def test_last_decision_explanation_after_move() -> None:
    menace = MENACEPlayer(mark="O", seed=42)
    board = Board.from_string("X        ")

    menace.choose_move(board)
    explanation = menace.last_decision_explanation()

    assert "MENACE selected original move" in explanation
    assert "beads" in explanation


def test_menace_to_dict_and_from_dict_round_trip_with_symmetry() -> None:
    menace = MENACEPlayer(mark="O", seed=42, use_symmetry=True)
    board = Board.from_string("X        ")

    menace.choose_move(board)
    menace.learn_from_result("win")

    restored = MENACEPlayer.from_dict(menace.to_dict())

    assert restored.mark == menace.mark
    assert restored.name == menace.name
    assert restored.use_symmetry == menace.use_symmetry
    assert restored.games_played == menace.games_played
    assert restored.wins == menace.wins
    assert restored.losses == menace.losses
    assert restored.draws == menace.draws
    assert restored.matchboxes.keys() == menace.matchboxes.keys()


def test_menace_uses_canonical_state_when_symmetry_enabled() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=True)
    board = Board.from_string("X O      ")

    menace.choose_move(board)
    expected_state = board.canonical_state()

    assert expected_state in menace.matchboxes


def test_menace_uses_raw_state_when_symmetry_disabled() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=False)
    board = Board.from_string("X O      ")

    menace.choose_move(board)
    expected_state = board.to_string()

    assert expected_state in menace.matchboxes


def test_symmetry_mapping_never_returns_illegal_original_move() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=True, seed=7)

    boards = [
        Board.from_string("X        "),
        Board.from_string("    X    "),
        Board.from_string("O X      "),
        Board.from_string("XO  X    "),
        Board.from_string(" X O     "),
    ]

    for board in boards:
        if board.is_terminal():
            continue

        move = menace.choose_move(board)

        assert move in board.available_moves()
        assert board.is_legal_move(move)


def test_decision_records_original_and_canonical_moves() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=True, seed=42)
    board = Board.from_string("X O      ")

    move = menace.choose_move(board)
    decision = menace.decisions_this_game[0]

    assert decision.move == move
    assert decision.move in board.available_moves()
    assert decision.canonical_move in decision.canonical_legal_moves
    assert decision.used_symmetry is True
    assert decision.state == board.canonical_state()


def test_matchbox_sync_prevents_stale_illegal_move_selection() -> None:
    matchbox = Matchbox(state="test", beads={7: 100, 2: 1}, minimum_beads=1)
    rng = random.Random(1)

    move = matchbox.choose_move(rng, legal_moves=[2])

    assert move == 2
    assert matchbox.beads == {2: 1}


def test_menace_explanation_mentions_symmetry_when_enabled() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=True, seed=42)
    board = Board.from_string("X        ")

    _, explanation = menace.choose_move_with_explanation(board)

    assert "Symmetry enabled: True" in explanation
    assert "Legal original-board moves" in explanation
    assert "Legal matchbox moves" in explanation