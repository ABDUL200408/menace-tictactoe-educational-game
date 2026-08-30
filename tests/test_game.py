"""
test_game.py

Unit tests for src.game.Game.

These tests verify that the game controller correctly manages:
    - player turns
    - legal move application
    - move history
    - final results
    - learning hooks
    - reset behaviour
    - safe play error wrapping
    - AI-vs-AI operation
    - educational explanation output.

Run:
    pytest tests/test_game.py
"""

from __future__ import annotations

import pytest

from src.board import Board
from src.game import Game, GameResult, MoveRecord
from src.players import HeuristicPlayer, RandomPlayer, ScriptedPlayer


class LearningSpyPlayer(ScriptedPlayer):
    """Scripted player that records learning results."""

    def __init__(self, mark: str, moves: list[int], name: str = "Learning Spy") -> None:
        super().__init__(mark=mark, moves=moves, name=name)
        self.learned_results: list[str] = []

    def learn_from_result(self, result: str) -> None:
        self.learned_results.append(result)


def test_game_initialises_with_valid_players() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])

    game = Game(player_x=player_x, player_o=player_o)

    assert game.player_x.mark == "X"
    assert game.player_o.mark == "O"
    assert game.board.to_string() == " " * 9
    assert game.move_history == []


def test_game_rejects_wrong_player_marks() -> None:
    player_x = ScriptedPlayer(mark="O", moves=[0])
    player_o = ScriptedPlayer(mark="X", moves=[1])

    with pytest.raises(ValueError):
        Game(player_x=player_x, player_o=player_o)


def test_current_player_x_starts() -> None:
    game = Game(
        player_x=ScriptedPlayer(mark="X", moves=[0]),
        player_o=ScriptedPlayer(mark="O", moves=[1]),
    )

    assert game.current_player() is game.player_x
    assert game.current_turn_mark() == "X"


def test_current_player_alternates_after_move() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])
    game = Game(player_x=player_x, player_o=player_o)

    game.play_turn(player_x)

    assert game.current_player() is player_o
    assert game.current_turn_mark() == "O"


def test_play_turn_rejects_wrong_player_turn() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])
    game = Game(player_x=player_x, player_o=player_o)

    with pytest.raises(ValueError):
        game.play_turn(player_o)


def test_play_turn_records_move_history_and_decision_reason() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])
    game = Game(player_x=player_x, player_o=player_o)

    record = game.play_turn(player_x)

    assert isinstance(record, MoveRecord)
    assert record.turn_number == 1
    assert record.player_name == player_x.name
    assert record.mark == "X"
    assert record.move == 0
    assert record.board_before == " " * 9
    assert record.board_after == "X" + " " * 8
    assert record.legal_moves_before == list(range(9))
    assert "ScriptedPlayer" in record.decision_reason
    assert len(game.move_history) == 1


def test_move_record_as_dict_is_serialisable() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])
    game = Game(player_x=player_x, player_o=player_o)

    record = game.play_turn(player_x)
    data = record.as_dict()

    assert data["turn_number"] == 1
    assert data["player_name"] == player_x.name
    assert data["mark"] == "X"
    assert data["move"] == 0
    assert data["board_before"] == " " * 9
    assert data["board_after"] == "X" + " " * 8
    assert data["legal_moves_before"] == list(range(9))
    assert "decision_reason" in data


def test_play_turn_rejects_illegal_move() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[0])
    game = Game(player_x=player_x, player_o=player_o)

    game.play_turn(player_x)

    with pytest.raises(ValueError):
        game.play_turn(player_o)


def test_play_turn_rejects_move_after_game_over() -> None:
    terminal_board = Board.from_string("XXXOO    ")
    game = Game(
        player_x=ScriptedPlayer(mark="X", moves=[0]),
        player_o=ScriptedPlayer(mark="O", moves=[1]),
        board=terminal_board,
    )

    with pytest.raises(ValueError):
        game.play_turn(game.player_x)


def test_play_x_wins_game() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0, 1, 2], name="X scripted")
    player_o = ScriptedPlayer(mark="O", moves=[3, 4], name="O scripted")

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play(learn=False)

    assert isinstance(result, GameResult)
    assert result.winner == "X"
    assert result.winner_name == "X scripted"
    assert result.outcome == "X_win"
    assert result.outcome_label() == "X scripted wins"
    assert result.result_for("X") == "win"
    assert result.result_for("O") == "loss"
    assert result.final_board == "XXXOO    "
    assert result.total_moves == 5
    assert result.is_terminal is True
    assert result.player_x_name == "X scripted"
    assert result.player_o_name == "O scripted"
    assert len(result.moves) == 5


def test_game_result_as_dict_contains_metadata() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0, 1, 2], name="X scripted")
    player_o = ScriptedPlayer(mark="O", moves=[3, 4], name="O scripted")

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play(learn=False)
    data = result.as_dict()

    assert data["winner"] == "X"
    assert data["winner_name"] == "X scripted"
    assert data["outcome"] == "X_win"
    assert data["outcome_label"] == "X scripted wins"
    assert data["final_board"] == "XXXOO    "
    assert data["total_moves"] == 5
    assert data["is_terminal"] is True
    assert data["player_x_name"] == "X scripted"
    assert data["player_o_name"] == "O scripted"


def test_play_o_wins_game() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0, 1, 6])
    player_o = ScriptedPlayer(mark="O", moves=[3, 4, 5], name="O winner")

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play(learn=False)

    assert result.winner == "O"
    assert result.winner_name == "O winner"
    assert result.outcome == "O_win"
    assert result.outcome_label() == "O winner wins"
    assert result.result_for("O") == "win"
    assert result.result_for("X") == "loss"
    assert result.total_moves == 6
    assert result.is_terminal is True


def test_play_draw_game() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0, 2, 5, 6, 7])
    player_o = ScriptedPlayer(mark="O", moves=[1, 3, 4, 8])

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play(learn=False)

    assert result.winner is None
    assert result.winner_name is None
    assert result.outcome == "draw"
    assert result.outcome_label() == "Draw"
    assert result.result_for("X") == "draw"
    assert result.result_for("O") == "draw"
    assert result.total_moves == 9
    assert result.is_terminal is True
    assert len(result.moves) == 9


def test_result_can_report_ongoing_game() -> None:
    game = Game(
        player_x=ScriptedPlayer(mark="X", moves=[0]),
        player_o=ScriptedPlayer(mark="O", moves=[1]),
    )

    result = game.result()

    assert result.outcome == "ongoing"
    assert result.outcome_label() == "Ongoing"
    assert result.result_for("X") == "ongoing"
    assert result.result_for("O") == "ongoing"
    assert result.total_moves == 0
    assert result.is_terminal is False
    assert result.winner_name is None


def test_current_turn_mark_is_none_after_terminal_game() -> None:
    game = Game(
        player_x=ScriptedPlayer(mark="X", moves=[0, 1, 2]),
        player_o=ScriptedPlayer(mark="O", moves=[3, 4]),
    )

    game.play(learn=False)

    assert game.current_turn_mark() is None


def test_learning_hooks_called_after_game() -> None:
    player_x = LearningSpyPlayer(mark="X", moves=[0, 1, 2], name="X learner")
    player_o = LearningSpyPlayer(mark="O", moves=[3, 4], name="O learner")

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play(learn=True)

    assert result.outcome == "X_win"
    assert player_x.learned_results == ["win"]
    assert player_o.learned_results == ["loss"]


def test_learning_hooks_not_called_when_disabled() -> None:
    player_x = LearningSpyPlayer(mark="X", moves=[0, 1, 2], name="X learner")
    player_o = LearningSpyPlayer(mark="O", moves=[3, 4], name="O learner")

    game = Game(player_x=player_x, player_o=player_o)
    game.play(learn=False)

    assert player_x.learned_results == []
    assert player_o.learned_results == []


def test_reset_clears_board_and_history() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])
    game = Game(player_x=player_x, player_o=player_o)

    game.play_turn(player_x)
    assert game.board.move_count() == 1
    assert len(game.move_history) == 1

    game.reset()

    assert game.board.to_string() == " " * 9
    assert game.move_history == []


def test_move_history_as_dataframe_rows_returns_serialisable_rows() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[1])
    game = Game(player_x=player_x, player_o=player_o)

    game.play_turn(player_x)
    rows = game.move_history_as_dataframe_rows()

    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["move"] == 0
    assert rows[0]["mark"] == "X"
    assert "decision_reason" in rows[0]


def test_explain_game_state_contains_teaching_information() -> None:
    game = Game(
        player_x=ScriptedPlayer(mark="X", moves=[0]),
        player_o=ScriptedPlayer(mark="O", moves=[1]),
    )

    explanation = game.explain_game_state()

    assert "Current board" in explanation
    assert "Board key" in explanation
    assert "Canonical key" in explanation
    assert "Available moves" in explanation
    assert "Current player" in explanation
    assert "Outcome" in explanation
    assert "MENACE treats board keys as matchbox states" in explanation
    assert "fair AI-vs-AI comparison" in explanation


def test_game_without_history_still_returns_result() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0, 1, 2])
    player_o = ScriptedPlayer(mark="O", moves=[3, 4])

    game = Game(player_x=player_x, player_o=player_o, record_history=False)
    result = game.play(learn=False)

    assert result.outcome == "X_win"
    assert result.moves == []
    assert game.move_history == []
    assert result.total_moves == 5
    assert result.is_terminal is True


def test_play_safely_wraps_player_errors_with_game_context() -> None:
    player_x = ScriptedPlayer(mark="X", moves=[0])
    player_o = ScriptedPlayer(mark="O", moves=[0])
    game = Game(player_x=player_x, player_o=player_o)

    with pytest.raises(RuntimeError) as exc_info:
        game.play_safely(learn=False)

    message = str(exc_info.value)

    assert "Game failed during play" in message
    assert "Current board" in message
    assert "Move history" in message


def test_random_vs_heuristic_ai_game_completes() -> None:
    player_x = RandomPlayer(mark="X", seed=42)
    player_o = HeuristicPlayer(mark="O")

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play_safely(learn=False)

    assert result.is_terminal is True
    assert result.outcome in {"X_win", "O_win", "draw"}
    assert result.total_moves >= 5
    assert len(result.moves) == result.total_moves


def test_ai_move_records_include_decision_reasons() -> None:
    player_x = RandomPlayer(mark="X", seed=42)
    player_o = HeuristicPlayer(mark="O")

    game = Game(player_x=player_x, player_o=player_o)
    result = game.play_safely(learn=False)

    assert result.moves
    assert all(move.decision_reason for move in result.moves)