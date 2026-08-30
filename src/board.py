"""
board.py

Board engine for the MSc MENACE Tic-Tac-Toe project.

This module implements the Tic-Tac-Toe game rules independently from the user interface, 
MENACE learning, training, and persistence. It manages board creation, move validation, 
move execution, win and draw detection, and symmetry-based canonical board representations 
used to reduce the number of unique game states. The board is represented as a simple nine-cell list, 
making move indexing, legal move generation, and symmetry transformations efficient and easy to understand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple


BoardCell = str
Move = int
Winner = Optional[str]


@dataclass(frozen=True)
class SymmetryTransform:
    """
    Represents one board symmetry transformation.

    transform:
        canonical[index] = original[transform[index]]

    inverse_transform:
        original[index] = canonical[inverse_transform[index]]
    """

    name: str
    transform: Tuple[int, ...]
    inverse_transform: Tuple[int, ...]

    def apply_to_cells(self, cells: Sequence[str]) -> List[str]:
        """Apply transform to board cells."""
        return [cells[index] for index in self.transform]

    def apply_to_move(self, canonical_move: Move) -> Move:
        """Map canonical move -> original move."""
        Board.validate_move_index(canonical_move)
        return self.transform[canonical_move]

    def inverse_move(self, original_move: Move) -> Move:
        """Map original move -> canonical move."""
        Board.validate_move_index(original_move)
        return self.inverse_transform[original_move]


@dataclass(frozen=True)
class CanonicalBoard:
    """Result of canonical board analysis."""

    canonical_state: str
    transform_name: str
    transform: SymmetryTransform

    def to_original_move(self, canonical_move: Move) -> Move:
        """Convert canonical move to original move."""
        return self.transform.apply_to_move(canonical_move)

    def to_canonical_move(self, original_move: Move) -> Move:
        """Convert original move to canonical move."""
        return self.transform.inverse_move(original_move)


@dataclass
class Board:
    """
    Represents a Noughts and Crosses / Tic-Tac-Toe board.

    Primary API:
        - make_move()
        - winner()
        - available_moves()
        - is_terminal()

    Alternative API names:
        - place_mark()
        - check_winner()
        - get_available_moves()
        - game_over()

    These aliases provide consistent access to the same board operations for
    game players, tests, and supporting project components.
    """

    cells: List[BoardCell] = field(default_factory=lambda: [" "] * 9)

    EMPTY: ClassVar[str] = " "
    VALID_MARKS: ClassVar[set[str]] = {"X", "O"}
    BOARD_SIZE: ClassVar[int] = 9

    WINNING_LINES: ClassVar[Tuple[Tuple[int, int, int], ...]] = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    SYMMETRY_TRANSFORMS: ClassVar[Tuple[SymmetryTransform, ...]] = (
        SymmetryTransform(
            "identity",
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
        ),
        SymmetryTransform(
            "rotate_90",
            (6, 3, 0, 7, 4, 1, 8, 5, 2),
            (2, 5, 8, 1, 4, 7, 0, 3, 6),
        ),
        SymmetryTransform(
            "rotate_180",
            (8, 7, 6, 5, 4, 3, 2, 1, 0),
            (8, 7, 6, 5, 4, 3, 2, 1, 0),
        ),
        SymmetryTransform(
            "rotate_270",
            (2, 5, 8, 1, 4, 7, 0, 3, 6),
            (6, 3, 0, 7, 4, 1, 8, 5, 2),
        ),
        SymmetryTransform(
            "reflect_vertical",
            (2, 1, 0, 5, 4, 3, 8, 7, 6),
            (2, 1, 0, 5, 4, 3, 8, 7, 6),
        ),
        SymmetryTransform(
            "reflect_horizontal",
            (6, 7, 8, 3, 4, 5, 0, 1, 2),
            (6, 7, 8, 3, 4, 5, 0, 1, 2),
        ),
        SymmetryTransform(
            "reflect_main_diagonal",
            (0, 3, 6, 1, 4, 7, 2, 5, 8),
            (0, 3, 6, 1, 4, 7, 2, 5, 8),
        ),
        SymmetryTransform(
            "reflect_anti_diagonal",
            (8, 5, 2, 7, 4, 1, 6, 3, 0),
            (8, 5, 2, 7, 4, 1, 6, 3, 0),
        ),
    )

    def __post_init__(self) -> None:
        self._validate_cells(self.cells)

    # ==========================================================
    # Construction
    # ==========================================================

    @classmethod
    def empty(cls) -> "Board":
        """Create an empty board."""
        return cls()

    @classmethod
    def from_string(cls, state: str) -> "Board":
        """Create a board from a compact 9-character state string."""
        if not isinstance(state, str):
            raise TypeError("Board state must be a string.")

        if len(state) != cls.BOARD_SIZE:
            raise ValueError(
                f"Board state must contain exactly {cls.BOARD_SIZE} characters."
            )

        return cls(list(state))

    @classmethod
    def from_sequence(cls, cells: Sequence[str]) -> "Board":
        """Create a board from a sequence of cells."""
        return cls(list(cells))

    # ==========================================================
    # Validation
    # ==========================================================

    @classmethod
    def _validate_cells(cls, cells: Sequence[str]) -> None:
        if len(cells) != cls.BOARD_SIZE:
            raise ValueError(f"Board must contain exactly {cls.BOARD_SIZE} cells.")

        allowed = cls.VALID_MARKS.union({cls.EMPTY})
        invalid = [cell for cell in cells if cell not in allowed]

        if invalid:
            raise ValueError(f"Invalid board values detected: {invalid}.")

    @classmethod
    def validate_mark(cls, mark: str) -> None:
        """Validate player mark."""
        if mark not in cls.VALID_MARKS:
            raise ValueError(f"Invalid mark '{mark}'. Expected {cls.VALID_MARKS}.")

    @classmethod
    def validate_move_index(cls, move: int) -> None:
        """Validate board move index."""
        # bool is a subclass of int in Python, but True/False are not valid
        # board positions and should not silently become moves 1/0.
        if isinstance(move, bool) or not isinstance(move, int):
            raise TypeError("Move must be an integer.")

        if move < 0 or move >= cls.BOARD_SIZE:
            raise ValueError("Move must be between 0 and 8.")

    # ==========================================================
    # Board Queries
    # ==========================================================

    def copy(self) -> "Board":
        """Return a deep copy of the board."""
        return Board(self.cells.copy())

    def reset(self) -> None:
        """Reset board to empty."""
        self.cells = [self.EMPTY] * self.BOARD_SIZE

    def to_string(self) -> str:
        """Return compact board state string."""
        return "".join(self.cells)

    def to_list(self) -> List[str]:
        """Return board cells as a copy."""
        return self.cells.copy()

    def available_moves(self) -> List[Move]:
        """Return all currently empty legal cell indexes."""
        if self.is_terminal():
            return []

        return [
            index
            for index, cell in enumerate(self.cells)
            if cell == self.EMPTY
        ]

    def get_available_moves(self) -> List[Move]:
        """
        Alias for available_moves() used by supporting project components.
        """
        return self.available_moves()

    def occupied_moves(self) -> List[Move]:
        """Return indexes of occupied cells."""
        return [
            index
            for index, cell in enumerate(self.cells)
            if cell != self.EMPTY
        ]

    def move_count(self) -> int:
        """Return number of occupied cells."""
        return len(self.occupied_moves())

    def is_legal_move(self, move: int) -> bool:
        """Return True if move is inside board, empty, and game not terminal."""
        try:
            self.validate_move_index(move)
        except (TypeError, ValueError):
            return False

        return self.cells[move] == self.EMPTY and not self.is_terminal()

    def is_full(self) -> bool:
        """Return True if no empty cells remain."""
        return self.EMPTY not in self.cells

    # ==========================================================
    # Game Logic
    # ==========================================================

    def make_move(self, move: int, mark: str) -> None:
        """
        Place mark on the board.

        This is the primary method used by the current architecture.
        """
        self.validate_mark(mark)
        self.validate_move_index(move)

        if self.is_terminal():
            raise ValueError("Cannot move on a terminal board.")

        if self.cells[move] != self.EMPTY:
            raise ValueError(f"Cell {move} is already occupied.")

        self.cells[move] = mark

    def place_mark(self, move: int, mark: str) -> None:
        """
        Alias for make_move() used by supporting project components.
        """
        self.make_move(move, mark)

    def undo_move(self, move: int) -> None:
        """Undo a move by emptying the selected cell."""
        self.validate_move_index(move)
        self.cells[move] = self.EMPTY

    def winner(self) -> Winner:
        """Return winning mark if a player has won, otherwise None."""
        for a, b, c in self.WINNING_LINES:
            if (
                self.cells[a] != self.EMPTY
                and self.cells[a] == self.cells[b] == self.cells[c]
            ):
                return self.cells[a]

        return None

    def check_winner(self) -> Winner:
        """
        Alias for winner() used by supporting project components.
        """
        return self.winner()

    def is_draw(self) -> bool:
        """Return True if board is full and there is no winner."""
        return self.is_full() and self.winner() is None

    def is_terminal(self) -> bool:
        """Return True if the game is won or drawn."""
        return self.winner() is not None or self.is_draw()

    def game_over(self) -> bool:
        """Alias for is_terminal() used by supporting project components."""
        return self.is_terminal()

    def result_for(self, mark: str) -> str:
        """
        Return result from the perspective of mark.

        Returns:
            win, loss, draw, or ongoing.
        """
        self.validate_mark(mark)

        current_winner = self.winner()

        if current_winner == mark:
            return "win"

        if current_winner is not None and current_winner != mark:
            return "loss"

        if self.is_draw():
            return "draw"

        return "ongoing"

    # ==========================================================
    # Symmetry Engine
    # ==========================================================

    def symmetry_variants(self) -> List[Tuple[str, str]]:
        """
        Return all board symmetry variants.

        Returns:
            List of (transform_name, transformed_state).
        """
        variants: List[Tuple[str, str]] = []

        for transform in self.SYMMETRY_TRANSFORMS:
            transformed = transform.apply_to_cells(self.cells)
            variants.append((transform.name, "".join(transformed)))

        return variants

    def canonical_analysis(self) -> CanonicalBoard:
        """
        Compute canonical symmetry representation.

        Returns:
            CanonicalBoard containing canonical state and move mappings.
        """
        candidates: List[Tuple[str, SymmetryTransform]] = []

        for transform in self.SYMMETRY_TRANSFORMS:
            transformed = transform.apply_to_cells(self.cells)
            state = "".join(transformed)
            candidates.append((state, transform))

        canonical_state, canonical_transform = min(
            candidates,
            key=lambda item: item[0],
        )

        return CanonicalBoard(
            canonical_state=canonical_state,
            transform_name=canonical_transform.name,
            transform=canonical_transform,
        )

    def canonical_state(self) -> str:
        """Return canonical board state string."""
        return self.canonical_analysis().canonical_state

    def original_index_grid(self) -> List[List[int]]:
        """
        Return the standard board index layout.

        This is mainly used by the Streamlit interface to explain how the
        player's visible board positions relate to MENACE's canonical matchbox
        positions.
        """
        return [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]

    def canonical_index_grid(self) -> List[List[int]]:
        """
        Return the index grid after the current canonical transform.

        Each value shows which original board index appears at that canonical
        position. For example, under an anti-diagonal reflection the grid is:
            8 5 2
            7 4 1
            6 3 0

        This is useful for explaining why an original move number and a
        matchbox move number can be different when symmetry reduction is used.
        """
        canonical = self.canonical_analysis()
        mapping = list(canonical.transform.transform)
        return [
            mapping[0:3],
            mapping[3:6],
            mapping[6:9],
        ]

    def canonical_to_original_move_map(self) -> Dict[int, int]:
        """
        Return mapping from canonical/matchbox move index to original board index.

        The keys are the move indexes used inside MENACE's canonical matchbox.
        The values are the visible board indexes used in the user interface.
        """
        canonical = self.canonical_analysis()
        return {
            canonical_move: canonical.to_original_move(canonical_move)
            for canonical_move in range(self.BOARD_SIZE)
        }

    def original_to_canonical_move_map(self) -> Dict[int, int]:
        """
        Return mapping from original board index to canonical/matchbox index.

        The keys are the visible board indexes used in the interface. The values
        are the move indexes used inside MENACE's canonical matchbox.
        """
        canonical = self.canonical_analysis()
        return {
            original_move: canonical.to_canonical_move(original_move)
            for original_move in range(self.BOARD_SIZE)
        }

    def legal_move_mapping_rows(self) -> List[Dict[str, int]]:
        """
        Return legal move mappings for the current board.

        Each row links the visible move number to the canonical matchbox move
        number used by MENACE. This supports explainable AI output in the
        Streamlit Play page.
        """
        original_to_canonical = self.original_to_canonical_move_map()
        return [
            {
                "original_board_move": move,
                "matchbox_move": original_to_canonical[move],
            }
            for move in self.available_moves()
        ]

    def explain_symmetry_mapping(self, selected_original_move: Optional[int] = None) -> str:
        """
        Explain how the current board's symmetry transform maps indexes.

        Args:
            selected_original_move:
                Optional visible board move selected by MENACE. If supplied, the
                explanation includes its corresponding canonical matchbox move.

        Returns:
            Human-readable text suitable for Streamlit, screenshots, and viva
            explanation.
        """
        canonical = self.canonical_analysis()
        original_to_canonical = self.original_to_canonical_move_map()

        lines = [
            f"Canonical transform: {canonical.transform_name}",
            "",
            "Original board indexes:",
            self._format_index_grid(self.original_index_grid()),
            "",
            "Canonical / matchbox index grid:",
            self._format_index_grid(self.canonical_index_grid()),
            "",
            "Legal move mapping:",
        ]

        for row in self.legal_move_mapping_rows():
            marker = ""
            if selected_original_move is not None and row["original_board_move"] == selected_original_move:
                marker = "  <-- selected by MENACE"
            lines.append(
                f"- Original move {row['original_board_move']} -> "
                f"matchbox move {row['matchbox_move']}{marker}"
            )

        if selected_original_move is not None:
            self.validate_move_index(selected_original_move)
            canonical_move = original_to_canonical[selected_original_move]
            lines.extend(
                [
                    "",
                    f"Selected original move: {selected_original_move}",
                    f"Corresponding matchbox move: {canonical_move}",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_index_grid(grid: Sequence[Sequence[int]]) -> str:
        """Format a 3x3 integer grid as readable text."""
        return "\n".join(" ".join(str(value) for value in row) for row in grid)

    # ==========================================================
    # Educational / Research Support
    # ==========================================================

    def board_features(self) -> Dict[str, int]:
        """Extract simple AI-friendly board features."""
        return {
            "x_count": self.cells.count("X"),
            "o_count": self.cells.count("O"),
            "empty_count": self.cells.count(self.EMPTY),
            "move_count": self.move_count(),
            "available_moves": len(self.available_moves()),
        }

    def explain_state(self) -> str:
        """Educational explanation of current board state."""
        canonical = self.canonical_analysis()
        winner = self.winner()

        if winner is not None:
            status = f"Player {winner} has won."
        elif self.is_draw():
            status = "The game ended in a draw."
        else:
            status = "The game is still in progress."

        return (
            f"Original state: '{self.to_string()}'.\n"
            f"Canonical state: '{canonical.canonical_state}'.\n"
            f"Canonical transform: '{canonical.transform_name}'.\n"
            f"Available moves: {self.available_moves()}.\n"
            f"Moves played: {self.move_count()}.\n"
            f"{status}"
        )

    # ==========================================================
    # Display Helpers
    # ==========================================================

    def rows(self) -> List[List[str]]:
        """Return board as three rows."""
        return [
            self.cells[0:3],
            self.cells[3:6],
            self.cells[6:9],
        ]

    def display_cells(self) -> List[str]:
        """Return cells with empty spaces replaced by indexes."""
        return [
            cell if cell != self.EMPTY else str(index)
            for index, cell in enumerate(self.cells)
        ]

    def __str__(self) -> str:
        display = self.display_cells()

        return (
            f" {display[0]} | {display[1]} | {display[2]} \n"
            "---+---+---\n"
            f" {display[3]} | {display[4]} | {display[5]} \n"
            "---+---+---\n"
            f" {display[6]} | {display[7]} | {display[8]} "
        )

    def __repr__(self) -> str:
        return f"Board.from_string({self.to_string()!r})"
