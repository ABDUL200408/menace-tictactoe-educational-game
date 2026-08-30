"""
game.py

This module manages a complete game between two players. It does not
implement MENACE, Minimax, heuristics, or Streamlit. Instead, it coordinates:

    - board state updates
    - player turns
    - legal move validation
    - result detection
    - move history logging
    - optional learning hooks
    - educational explanations.

The Game class is intentionally independent from AI strategy code. This allows
the same engine to support:

    - Human vs MENACE (Play page)
    - MENACE vs Random (Train page)
    - MENACE vs Heuristic (Compare page)
    - MENACE vs Minimax (Compare page)
    - AI vs AI comparative evaluation (Compare page)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from src.board import Board


@runtime_checkable
class PlayerProtocol(Protocol):
    """Defines the common interface that every player must implement."""

    mark: str
    name: str

    def choose_move(self, board: Board) -> int:
        """Choose a legal move for the current board."""


@dataclass(frozen=True)
class MoveRecord:
    """
    Stores the details of each move, including board states before and after the move.

    Attributes:
        turn_number:
            Move number starting from 1.
        player_name:
            Name of player who moved.
        mark:
            Player mark.
        move:
            Move index.
        board_before:
            Board state before move.
        board_after:
            Board state after move.
        legal_moves_before:
            Legal moves before the player moved.
        decision_reason:
            Optional explanation from the player strategy.
    """

    turn_number: int
    player_name: str
    mark: str
    move: int
    board_before: str
    board_after: str
    legal_moves_before: List[int]
    decision_reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Return a serialisable dictionary representation of the move."""
        return {
            "turn_number": self.turn_number,
            "player_name": self.player_name,
            "mark": self.mark,
            "move": self.move,
            "board_before": self.board_before,
            "board_after": self.board_after,
            "legal_moves_before": self.legal_moves_before,
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class GameResult:
    """
    Represents the final outcome of a game and its recorded move history.

    outcome:
        "X_win", "O_win", "draw", or "ongoing".
    winner:
        Winning mark, or None.
    winner_name:
        Name of the winning player, or None.
    moves:
        Move history.
    final_board:
        Final board state.
    total_moves:
        Number of moves played.
    is_terminal:
        Whether the game is complete.
    player_x_name:
        Name of X player.
    player_o_name:
        Name of O player.
    """

    winner: Optional[str]
    outcome: str
    moves: List[MoveRecord]
    final_board: str
    total_moves: int
    is_terminal: bool
    player_x_name: str = "X"
    player_o_name: str = "O"
    winner_name: Optional[str] = None

    def result_for(self, mark: str) -> str:
        """
        Return result from a player perspective.

        Returns:
            "win", "loss", "draw", or "ongoing".
        """
        Board.validate_mark(mark)

        if not self.is_terminal:
            return "ongoing"

        if self.winner is None:
            return "draw"

        return "win" if self.winner == mark else "loss"

    def outcome_label(self) -> str:
        """Return human-readable outcome label."""
        if self.outcome == "draw":
            return "Draw"

        if self.outcome == "ongoing":
            return "Ongoing"

        if self.winner_name:
            return f"{self.winner_name} wins"

        return f"{self.winner} wins"

    def as_dict(self) -> Dict[str, object]:
        """Return serialisable summary dictionary."""
        return {
            "winner": self.winner,
            "winner_name": self.winner_name,
            "outcome": self.outcome,
            "outcome_label": self.outcome_label(),
            "final_board": self.final_board,
            "total_moves": self.total_moves,
            "is_terminal": self.is_terminal,
            "player_x_name": self.player_x_name,
            "player_o_name": self.player_o_name,
        }


@dataclass
class Game:
    """Coordinates players, turns, move validation, history, and game results."""

    player_x: PlayerProtocol
    player_o: PlayerProtocol
    board: Board = field(default_factory=Board.empty)
    record_history: bool = True
    move_history: List[MoveRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate players and board."""
        self._validate_player(self.player_x, expected_mark="X")
        self._validate_player(self.player_o, expected_mark="O")

        if self.player_x.mark == self.player_o.mark:
            raise ValueError("Players must use different marks.")

    def play(self, learn: bool = True) -> GameResult:
        """
        Play until terminal state.

        Args:
            learn:
                If True, apply learning hooks at the end of the game.

        Returns:
            GameResult.
        """
        while not self.board.is_terminal():
            self.play_turn(self.current_player())

        result = self.result()

        if learn:
            self._apply_learning_hooks(result)

        return result

    def play_safely(self, learn: bool = True) -> GameResult:
        """
        Play a game and re-raise errors with useful game context."""
        try:
            return self.play(learn=learn)
        except Exception as exc:
            current_player_name = (
                self.current_player().name
                if not self.board.is_terminal()
                else "None"
            )
            raise RuntimeError(
                "Game failed during play.\n"
                f"Current board: {self.board.to_string()!r}\n"
                f"Move count: {self.board.move_count()}\n"
                f"Current player: {current_player_name}\n"
                f"Move history: {[move.as_dict() for move in self.move_history]}"
            ) from exc

    def play_turn(self, player: PlayerProtocol) -> MoveRecord:
        """
        Play one turn for the expected current player.

        Raises:
            ValueError if the game is over, the wrong player is used,
            or the selected move is illegal.
        """
        if self.board.is_terminal():
            raise ValueError("Cannot play a turn because the game is already over.")

        expected_player = self.current_player()

        if player.mark != expected_player.mark:
            raise ValueError(
                f"It is {expected_player.name}'s turn ({expected_player.mark}), "
                f"but {player.name} ({player.mark}) was asked to play."
            )

        board_before = self.board.to_string()
        legal_moves_before = self.board.available_moves()

        move = player.choose_move(self.board.copy())

        # bool is a subclass of int in Python, but True/False are not valid
        # board moves and must not silently become indexes 1/0.
        if isinstance(move, bool) or not isinstance(move, int):
            raise TypeError(f"{player.name} returned non-integer move {move!r}.")

        if not self.board.is_legal_move(move):
            raise ValueError(
                f"{player.name} selected illegal move {move} for board state "
                f"{board_before!r}. Legal moves were {legal_moves_before}."
            )

        decision_reason = getattr(player, "last_decision_reason", "")

        self.board.make_move(move, player.mark)
        board_after = self.board.to_string()

        record = MoveRecord(
            turn_number=self.board.move_count(),
            player_name=player.name,
            mark=player.mark,
            move=move,
            board_before=board_before,
            board_after=board_after,
            legal_moves_before=legal_moves_before,
            decision_reason=decision_reason,
        )

        if self.record_history:
            self.move_history.append(record)

        return record

    def current_player(self) -> PlayerProtocol:
        """Return the player whose turn it is."""
        return self.player_x if self.board.move_count() % 2 == 0 else self.player_o

    def current_turn_mark(self) -> Optional[str]:
        """Return current mark to play, or None if game is over."""
        if self.board.is_terminal():
            return None

        return self.current_player().mark

    def result(self) -> GameResult:
        """Return current game result."""
        winner = self.board.winner()

        if winner == "X":
            outcome = "X_win"
            winner_name = self.player_x.name
        elif winner == "O":
            outcome = "O_win"
            winner_name = self.player_o.name
        elif self.board.is_draw():
            outcome = "draw"
            winner_name = None
        else:
            outcome = "ongoing"
            winner_name = None

        return GameResult(
            winner=winner,
            outcome=outcome,
            moves=self.move_history.copy(),
            final_board=self.board.to_string(),
            total_moves=self.board.move_count(),
            is_terminal=self.board.is_terminal(),
            player_x_name=self.player_x.name,
            player_o_name=self.player_o.name,
            winner_name=winner_name,
        )

    def reset(self) -> None:
        """Reset board and move history."""
        self.board.reset()
        self.move_history.clear()

    def move_history_as_dataframe_rows(self) -> List[Dict[str, object]]:
        """
        Return move history as serialisable rows for tables and visualisation."""
        return [move.as_dict() for move in self.move_history]

    def explain_game_state(self) -> str:
        """
        Return educational explanation of the current game state.
        """
        current_player = None if self.board.is_terminal() else self.current_player()

        current_text = (
            "None - the game has ended."
            if current_player is None
            else f"{current_player.name} ({current_player.mark})"
        )

        result = self.result()

        lines = [
            f"Current board:\n{self.board}",
            "",
            f"Board key: {self.board.to_string()!r}",
            f"Canonical key: {self.board.canonical_state()!r}",
            f"Moves played: {self.board.move_count()}",
            f"Available moves: {self.board.available_moves()}",
            f"Current player: {current_text}",
            f"Outcome: {result.outcome_label()}",
            f"Winner: {result.winner_name if result.winner_name else 'None'}",
            "",
            "Educational interpretation:",
            "- MENACE treats board keys as matchbox states.",
            "- Legal moves correspond to possible bead choices.",
            "- Random, Heuristic, Minimax, and MENACE players can all use the same game engine.",
            "- This separation allows fair AI-vs-AI comparison.",
        ]

        return "\n".join(lines)

    def _apply_learning_hooks(self, result: GameResult) -> None:
        """
        Notify players that the game ended and apply any available learning hook."""
        for player in (self.player_x, self.player_o):
            player_result = result.result_for(player.mark)

            if hasattr(player, "learn_from_result"):
                player.learn_from_result(player_result)
            elif hasattr(player, "learn_from_game"):
                player.learn_from_game(player_result)

    @staticmethod
    def _validate_player(player: PlayerProtocol, expected_mark: str) -> None:
        """Validate player interface and mark."""
        if not hasattr(player, "choose_move"):
            raise TypeError("Player must implement choose_move(board).")

        if not hasattr(player, "name"):
            raise TypeError("Player must define a name attribute.")

        if not hasattr(player, "mark"):
            raise TypeError("Player must define a mark attribute.")

        Board.validate_mark(player.mark)

        if player.mark != expected_mark:
            raise ValueError(
                f"Expected player mark {expected_mark!r}, got {player.mark!r}."
            )
