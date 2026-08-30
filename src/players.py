"""
players.py

This file defines the common player interface and the AI opponents used in the
project. It is independent from the MENACE learning algorithm and provides
human, random, rule-based, and search-based players for gameplay, training,
testing, and comparative evaluation.

MENACE is implemented separately because it is the only learning agent in the
project. Keeping the opponent strategies separate makes the design modular,
simplifies testing, and allows the same Game engine to compare learning,
rule-based, and optimal search strategies.

These player implementations support the Streamlit interface, AI comparison
experiments, training, unit testing, and dissertation evaluation.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from src.board import Board


PlayerType = str


def _normalise_player_type(player_type: str) -> str:
    """Normalise a user-supplied player type string."""
    if not isinstance(player_type, str):
        raise TypeError("player_type must be a string.")

    normalised = player_type.strip().lower()

    if not normalised:
        raise ValueError("player_type cannot be empty.")

    return normalised


def _require_playable_board(board: Board) -> list[int]:
    """
    Validate that a player can choose a move from the supplied board.

    Returns:
        The list of legal moves.

    Raises:
        TypeError:
            If board is not a Board instance.
        ValueError:
            If the board is terminal or no legal moves exist.
    """
    if not isinstance(board, Board):
        raise TypeError("choose_move expects a Board instance.")

    if board.is_terminal():
        raise ValueError("Cannot choose a move from a terminal board.")

    legal_moves = board.available_moves()

    if not legal_moves:
        raise ValueError("No legal moves available.")

    return legal_moves


# ============================================================
# Base Player
# ============================================================

@dataclass
class Player(ABC):
    """
    Defines the common interface implemented by all player types.

    Every player provides a mark, display name, move-selection method, and
    optional strategy or decision explanation. This allows the game engine and
    training components to work with different player strategies consistently.
    """

    mark: str
    name: str = "Player"

    last_decision_reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Validate the player's mark immediately after construction."""
        Board.validate_mark(self.mark)

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Player name must be a non-empty string.")

        self.name = self.name.strip()

    @property
    def opponent_mark(self) -> str:
        """Return the opposite Tic-Tac-Toe mark."""
        return "O" if self.mark == "X" else "X"

    @abstractmethod
    def choose_move(self, board: Board) -> int:
        """
        Select a legal move for the current board.

        Args:
            board:
                Current Tic-Tac-Toe board.

        Returns:
            A legal board index from 0 to 8.
        """

    def learn_from_result(self, result: str) -> None:
        """
        Optional learning hook.
        MENACE overrides this method, while the other players do not learn from
        game outcomes.
        """
        if result not in {"win", "loss", "draw", "ongoing"}:
            raise ValueError(
                "result must be one of: 'win', 'loss', 'draw', or 'ongoing'."
            )

    def explain_strategy(self) -> str:
        """Return a human-readable explanation of the player's strategy."""
        return (
            f"{self.name} uses the generic player interface. "
            "Specific move-selection behaviour is implemented by subclasses."
        )

    def explain_last_decision(self) -> str:
        """Return the explanation for the player's most recent move."""
        if not self.last_decision_reason:
            return f"{self.name} has not made a decision yet."

        return self.last_decision_reason


# ============================================================
# Human Player
# ============================================================

@dataclass
class HumanPlayer(Player):
    """Represents a human user whose moves are supplied by the interface."""
    name: str = "Human"
    pending_move: Optional[int] = None

    def set_move(self, move: int) -> None:
        """Store the next move selected by the user interface."""
        Board.validate_move_index(move)
        self.pending_move = move

    def choose_move(self, board: Board) -> int:
        """Return the pending move selected by the human user."""
        _require_playable_board(board)

        if self.pending_move is None:
            raise ValueError("HumanPlayer requires a pending move before choosing.")

        move = self.pending_move
        self.pending_move = None

        if not board.is_legal_move(move):
            raise ValueError(f"Human selected illegal move {move}.")

        self.last_decision_reason = (
            f"The human user manually selected square {move}."
        )
        return move

    def explain_strategy(self) -> str:
        """Explain the human player's role."""
        return (
            "HumanPlayer represents the user interacting with the Streamlit "
            "interface. It supports human-vs-MENACE educational gameplay."
        )


# ============================================================
# Random Baseline
# ============================================================

@dataclass
class RandomPlayer(Player):
    """
    Random baseline player.

    This is the weakest AI baseline. It is important academically because
    MENACE should be able to improve against random play during training.
    """

    name: str = "Random Baseline"
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate fields and initialise the local random generator."""
        super().__post_init__()
        self._rng = random.Random(self.seed)

    def choose_move(self, board: Board) -> int:
        """Choose uniformly from the legal moves."""
        legal_moves = _require_playable_board(board)
        move = self._rng.choice(legal_moves)

        self.last_decision_reason = (
            f"RandomPlayer selected move {move} uniformly from legal moves "
            f"{legal_moves}."
        )
        return move

    def explain_strategy(self) -> str:
        """Explain the random baseline."""
        return (
            "RandomPlayer chooses uniformly from available legal moves. "
            "It provides a weak stochastic baseline for measuring MENACE "
            "learning progress."
        )

# ============================================================
# Scripted Player
# ============================================================

@dataclass
class ScriptedPlayer(Player):
    """Follows a predefined sequence of moves for deterministic scenarios."""

    moves: Sequence[int] = field(default_factory=list)
    name: str = "Scripted Player"

    _position: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate scripted moves."""
        super().__post_init__()

        for move in self.moves:
            Board.validate_move_index(move)

    def choose_move(self, board: Board) -> int:
        """Return the next move from the scripted sequence."""
        _require_playable_board(board)

        if self._position >= len(self.moves):
            raise ValueError("ScriptedPlayer has no remaining moves.")

        move = int(self.moves[self._position])
        self._position += 1

        if not board.is_legal_move(move):
            raise ValueError(f"Illegal scripted move: {move}")

        self.last_decision_reason = (
            f"ScriptedPlayer followed predefined move sequence and selected {move}."
        )
        return move

    def reset_script(self) -> None:
        """Restart the scripted move sequence from the beginning."""
        self._position = 0
        self.last_decision_reason = ""

    def explain_strategy(self) -> str:
        """Explain the scripted player."""
        return (
            "ScriptedPlayer follows a predefined sequence of moves. "
            "It is mainly used for deterministic tests and debugging."
        )


# ============================================================
# First Available Baseline
# ============================================================


@dataclass
class FirstAvailablePlayer(Player):
    """
    Simple deterministic baseline.

    It always takes the first legal move. This is not a strong opponent, but it
    provides a predictable baseline for debugging and simple comparisons.
    """

    name: str = "First Available Baseline"

    def choose_move(self, board: Board) -> int:
        """Choose the first legal move."""
        legal_moves = _require_playable_board(board)
        move = legal_moves[0]

        self.last_decision_reason = (
            f"FirstAvailablePlayer selected the first legal move: {move}."
        )
        return move

    def explain_strategy(self) -> str:
        """Explain the first-available baseline."""
        return (
            "FirstAvailablePlayer always selects the first available legal move. "
            "It is a simple deterministic baseline rather than an intelligent AI."
        )
# ============================================================
# Heuristic AI Player
# ============================================================

@dataclass
class HeuristicPlayer(Player):
    """
    Uses predefined strategic rules, such as winning, blocking, 
    and controlling key positions, instead of learning.

    Strategy order:
        1. Win immediately if possible.
        2. Block the opponent's immediate win.
        3. Take the centre.
        4. Take a corner.
        5. Take any remaining legal move.

    This is an explainable, non-learning baseline. It helps compare MENACE
    against a human-designed rule strategy.
    """

    name: str = "Heuristic AI"

    def choose_move(self, board: Board) -> int:
        """Choose a move using a simple rule-based strategy."""
        _require_playable_board(board)

        winning_move = self._find_winning_move(board, self.mark)
        if winning_move is not None:
            self.last_decision_reason = (
                f"Heuristic AI selected move {winning_move} to win immediately."
            )
            return winning_move

        blocking_move = self._find_winning_move(board, self.opponent_mark)
        if blocking_move is not None:
            self.last_decision_reason = (
                f"Heuristic AI selected move {blocking_move} to block the "
                "opponent's immediate winning move."
            )
            return blocking_move

        if board.is_legal_move(4):
            self.last_decision_reason = (
                "Heuristic AI selected the centre square because it gives the "
                "most flexible future opportunities."
            )
            return 4

        corner_move = self._first_legal_from(board, [0, 2, 6, 8])
        if corner_move is not None:
            self.last_decision_reason = (
                f"Heuristic AI selected corner square {corner_move} because "
                "corners are strategically strong in Tic-Tac-Toe."
            )
            return corner_move

        move = board.available_moves()[0]
        self.last_decision_reason = (
            f"Heuristic AI selected remaining legal move {move}."
        )
        return move

    def _find_winning_move(self, board: Board, mark: str) -> Optional[int]:
        """
        Return a move that would immediately win for the supplied mark.

        Args:
            board:
                Current board.
            mark:
                Player mark to test.

        Returns:
            A winning move index, or None if no immediate win exists.
        """
        Board.validate_mark(mark)

        for move in board.available_moves():
            temp = board.copy()
            temp.place_mark(move, mark)

            if temp.check_winner() == mark:
                return move

        return None

    @staticmethod
    def _first_legal_from(board: Board, moves: Sequence[int]) -> Optional[int]:
        """Return the first legal move from a preferred move list."""
        for move in moves:
            if board.is_legal_move(move):
                return move

        return None

    def explain_strategy(self) -> str:
        """Explain the heuristic strategy."""
        return (
            "HeuristicPlayer uses fixed human-designed rules rather than "
            "learning. It prioritises immediate wins, blocking losses, centre "
            "control, and corner control."
        )

# ============================================================
# Minimax AI Player
# ============================================================

@dataclass
class MinimaxPlayer(Player): 
    """
    Uses the Minimax algorithm to search the game tree and always selects 
    the mathematically optimal move.  

    Tic-Tac-Toe is small enough for full game-tree search. Minimax is used as
    the strongest baseline because it represents optimal play rather than
    learning from experience.
    """

    name: str = "Minimax AI"
    prefer_fast_wins: bool = True

    def choose_move(self, board: Board) -> int:
        """Select the highest-scoring legal move using Minimax search."""
        legal_moves = _require_playable_board(board)

        cache: Dict[tuple[str, bool], int] = {}
        best_score = -math.inf
        best_move = legal_moves[0] 

        for move in self._ordered_moves(legal_moves):
            temp = board.copy()
            temp.place_mark(move, self.mark)

            score = self._minimax(
                board=temp,
                maximizing=False,
                depth=1,
                cache=cache,
            )

            if score > best_score:
                best_score = score
                best_move = move

        self.last_decision_reason = (
            f"Minimax AI selected move {best_move} with evaluated utility "
            f"score {best_score} after searching the game tree."
        )
        return best_move

    def _minimax(
        self,
        board: Board,
        maximizing: bool,
        depth: int = 0,
        cache: Optional[Dict[tuple[str, bool], int]] = None,
    ) -> int:
        """
        Recursive Minimax search.

        Returns a positive score for a win, a negative score for a loss,
        and zero for a draw.

        Depth is used to prefer faster wins and delay losses.
        """
        if cache is None:
            cache = {}

        cache_key = (board.to_string(), maximizing)
        if cache_key in cache:
            return cache[cache_key]

        winner = board.check_winner()

        if winner == self.mark:
            score = 10 - depth if self.prefer_fast_wins else 1
            cache[cache_key] = score
            return score

        if winner == self.opponent_mark:
            score = depth - 10 if self.prefer_fast_wins else -1
            cache[cache_key] = score
            return score

        if board.is_full():
            cache[cache_key] = 0
            return 0

        if maximizing:
            best_score = -math.inf

            for move in self._ordered_moves(board.available_moves()):
                temp = board.copy()
                temp.place_mark(move, self.mark)
                score = self._minimax(temp, False, depth + 1, cache)
                best_score = max(best_score, score)

            cache[cache_key] = int(best_score)
            return int(best_score)

        best_score = math.inf

        for move in self._ordered_moves(board.available_moves()):
            temp = board.copy()
            temp.place_mark(move, self.opponent_mark)
            score = self._minimax(temp, True, depth + 1, cache)
            best_score = min(best_score, score)

        cache[cache_key] = int(best_score)
        return int(best_score)

    @staticmethod
    def _ordered_moves(legal_moves: Sequence[int]) -> list[int]:
        """
        Return legal moves in a strategic order for stable tie-breaking.

        Centre and corners are considered before edges. This does not replace
        Minimax search; it only makes equal-score choices deterministic and
        strategically sensible.
        """
        preference = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        return [move for move in preference if move in legal_moves]

    def explain_strategy(self) -> str:
        """Explain the Minimax strategy."""
        return (
            "MinimaxPlayer performs recursive game-tree search and selects an "
            "optimal move. It is a search-based AI baseline, not a learning "
            "agent, and is effectively unbeatable in Tic-Tac-Toe."
        )


def create_player(
    player_type: PlayerType,
    mark: str,
    seed: Optional[int] = None,
) -> Player: 
    """
    Create a player object from a string label.

    Supported types:
        - human
        - random
        - first / first_available
        - heuristic
        - minimax

    Args:
        player_type:
            Player type label.
        mark:
            Board mark for the player.
        seed:
            Optional seed for stochastic players.

    Returns:
        A concrete Player instance.
    """
    Board.validate_mark(mark)
    normalised_type = _normalise_player_type(player_type)

    aliases = {
        "human": "human",
        "user": "human",
        "random": "random",
        "random_baseline": "random",
        "first": "first",
        "first_available": "first",
        "first_available_baseline": "first",
        "heuristic": "heuristic",
        "heuristic_ai": "heuristic",
        "minimax": "minimax",
        "minimax_ai": "minimax",
    }

    canonical = aliases.get(normalised_type)

    if canonical is None:
        raise ValueError(
            f"Unknown player type '{player_type}'. "
            "Expected one of: human, random, first, first_available, "
            "heuristic, minimax."
        )

    if canonical == "human":
        return HumanPlayer(mark=mark)

    if canonical == "random":
        return RandomPlayer(mark=mark, seed=seed)

    if canonical == "first":
        return FirstAvailablePlayer(mark=mark)

    if canonical == "heuristic":
        return HeuristicPlayer(mark=mark)

    if canonical == "minimax":
        return MinimaxPlayer(mark=mark)

    raise ValueError(f"Unhandled player type: {canonical}")

# Public module interface used by explicit wildcard imports.
__all__ = [
    "Player",
    "HumanPlayer",
    "RandomPlayer",
    "ScriptedPlayer",
    "FirstAvailablePlayer",
    "HeuristicPlayer",
    "MinimaxPlayer",
    "create_player",
]
