"""
test_board.py

Unit tests for src.board.Board.

These tests verify the correctness of the Noughts and Crosses environment.
This is essential because MENACE learns from board states and game outcomes.
If the Board class is wrong, the training results are not academically valid.

Run:
    pytest tests/test_board.py
"""

from __future__ import annotations

import pytest

from src.board import Board


def test_empty_board_has_nine_empty_cells() -> None:
    board = Board.empty()

    assert board.cells == [" "] * 9
    assert board.available_moves() == list(range(9))
    assert board.move_count() == 0
    assert not board.is_terminal()


def test_board_from_string_valid_state() -> None:
    board = Board.from_string("X O   X  ")

    assert board.to_string() == "X O   X  "
    assert board.cells[0] == "X"
    assert board.cells[2] == "O"
    assert board.cells[6] == "X"


def test_board_from_string_rejects_invalid_length() -> None:
    with pytest.raises(ValueError):
        Board.from_string("XO")


def test_board_from_string_rejects_invalid_symbols() -> None:
    with pytest.raises(ValueError):
        Board.from_string("XOA      ")


@pytest.mark.parametrize("invalid_mark", ["", "A", "x", "0", None])
def test_validate_mark_rejects_invalid_marks(invalid_mark) -> None:
    with pytest.raises((ValueError, TypeError)):
        Board.validate_mark(invalid_mark)


@pytest.mark.parametrize("move", [-1, 9, 10])
def test_validate_move_index_rejects_out_of_range(move: int) -> None:
    with pytest.raises(ValueError):
        Board.validate_move_index(move)


def test_validate_move_index_rejects_non_integer() -> None:
    with pytest.raises(TypeError):
        Board.validate_move_index("1")  # type: ignore[arg-type]


def test_make_move_updates_board() -> None:
    board = Board.empty()

    board.make_move(4, "X")

    assert board.cells[4] == "X"
    assert board.available_moves() == [0, 1, 2, 3, 5, 6, 7, 8]


def test_place_mark_alias_updates_board() -> None:
    board = Board.empty()

    board.place_mark(4, "X")

    assert board.cells[4] == "X"


def test_make_move_rejects_occupied_cell() -> None:
    board = Board.empty()
    board.make_move(0, "X")

    with pytest.raises(ValueError):
        board.make_move(0, "O")


def test_make_move_rejects_after_terminal_state() -> None:
    board = Board.from_string("XXXOO    ")

    assert board.is_terminal()

    with pytest.raises(ValueError):
        board.make_move(5, "O")


@pytest.mark.parametrize(
    "state,winner",
    [
        ("XXX      ", "X"),
        ("   OOO   ", "O"),
        ("      XXX", "X"),
        ("X  X  X  ", "X"),
        (" O  O  O ", "O"),
        ("  X  X  X", "X"),
        ("X   X   X", "X"),
        ("  O O O  ", "O"),
    ],
)
def test_winner_detects_all_winning_lines(state: str, winner: str) -> None:
    board = Board.from_string(state)

    assert board.winner() == winner
    assert board.check_winner() == winner
    assert board.is_terminal()
    assert board.game_over()


def test_draw_detection() -> None:
    board = Board.from_string("XOXOOXXXO")

    assert board.winner() is None
    assert board.is_draw()
    assert board.is_terminal()


def test_ongoing_game_is_not_terminal() -> None:
    board = Board.from_string("XOX O    ")

    assert board.winner() is None
    assert not board.is_draw()
    assert not board.is_terminal()


def test_result_for_win_loss_draw_and_ongoing() -> None:
    x_win = Board.from_string("XXXOO    ")
    draw = Board.from_string("XOXOOXXXO")
    ongoing = Board.from_string("XOX O    ")

    assert x_win.result_for("X") == "win"
    assert x_win.result_for("O") == "loss"
    assert draw.result_for("X") == "draw"
    assert ongoing.result_for("X") == "ongoing"


def test_copy_is_independent() -> None:
    board = Board.empty()
    copied = board.copy()

    copied.make_move(0, "X")

    assert board.cells[0] == " "
    assert copied.cells[0] == "X"


def test_undo_move_clears_cell() -> None:
    board = Board.empty()
    board.make_move(3, "O")

    board.undo_move(3)

    assert board.cells[3] == " "
    assert 3 in board.available_moves()


def test_to_list_returns_copy_not_reference() -> None:
    board = Board.empty()
    cells = board.to_list()
    cells[0] = "X"

    assert board.cells[0] == " "


def test_rows_returns_three_rows() -> None:
    board = Board.from_string("XOX O    ")

    assert board.rows() == [
        ["X", "O", "X"],
        [" ", "O", " "],
        [" ", " ", " "],
    ]


def test_display_cells_uses_indexes_for_empty_cells() -> None:
    board = Board.from_string("X O      ")

    assert board.display_cells()[0] == "X"
    assert board.display_cells()[1] == "1"
    assert board.display_cells()[2] == "O"


def test_string_representation_contains_board_symbols() -> None:
    board = Board.from_string("X O      ")
    text = str(board)

    assert "X" in text
    assert "O" in text
    assert "---+---+---" in text


def test_symmetry_variants_returns_eight_named_variants() -> None:
    board = Board.from_string("X O      ")
    variants = board.symmetry_variants()

    assert len(variants) == 8

    for transform_name, state in variants:
        assert isinstance(transform_name, str)
        assert isinstance(state, str)
        assert len(state) == 9


def test_canonical_state_is_stable_across_calls() -> None:
    board = Board.from_string("X O      ")

    first = board.canonical_state()
    second = board.canonical_state()

    assert first == second
    assert isinstance(first, str)
    assert len(first) == 9


def test_explain_state_contains_teaching_information() -> None:
    board = Board.from_string("X O      ")
    explanation = board.explain_state()

    assert "Original state" in explanation
    assert "Canonical state" in explanation
    assert "Available moves" in explanation
    assert "Moves played" in explanation
    assert "Canonical transform" in explanation