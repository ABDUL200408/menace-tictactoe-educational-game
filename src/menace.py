"""
menace.py

This module implements the matchbox-and-bead learning mechanism used by MENACE.
It relies on the Board module as the single source of truth for symmetry reduction,
so the decision explanation, matchbox mapping, and board explanation all use the
same canonical state and move transformation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.board import Board, CanonicalBoard
from src.players import Player


Move = int
BoardState = str
MoveProbabilities = Dict[Move, float]

# Transform tuples are exposed from Board so symmetry definitions remain
# consistent across MENACE, explanations, and board-state mapping.
TRANSFORMS: Tuple[Tuple[int, ...], ...] = tuple(
    transform.transform for transform in Board.SYMMETRY_TRANSFORMS
)


@dataclass(frozen=True)
class SymmetryMapping:
    """
    Mapping between the original board and the canonical matchbox representation.

    MENACE stores this mapping at decision time so the UI can explain the exact
    transform used before the board changes. This avoids displaying a mapping
    computed from a later board state.
    """

    canonical_state: BoardState
    transform_name: str
    transform: Tuple[int, ...]
    inverse_transform: Tuple[int, ...]
    original_to_canonical: Dict[Move, Move]
    original_state: BoardState

    def to_canonical_move(self, original_move: Move) -> Move:
        """Map a visible original-board move to a canonical matchbox move."""
        Board.validate_move_index(original_move)
        return self.original_to_canonical[original_move]

    def to_original_move(self, canonical_move: Move) -> Move:
        """Map a canonical matchbox move back to the visible original board."""
        Board.validate_move_index(canonical_move)
        return self.transform[canonical_move]

    def canonical_to_original(self) -> Dict[Move, Move]:
        """Return inverse mapping: matchbox move -> original board square."""
        return {canonical: original for original, canonical in self.original_to_canonical.items()}

    def canonical_index_grid(self) -> List[List[int]]:
        """Return the canonical/matchbox grid used for this mapping."""
        return [
            list(self.transform[0:3]),
            list(self.transform[3:6]),
            list(self.transform[6:9]),
        ]

    def legal_mapping_rows(self, legal_moves: List[Move]) -> List[Dict[str, int]]:
        """Return original-to-matchbox mapping rows for legal moves."""
        return [
            {
                "original_board_move": move,
                "matchbox_move": self.to_canonical_move(move),
            }
            for move in legal_moves
        ]


def _canonical_mapping(board: Board) -> SymmetryMapping:
    """
    Return MENACE's canonical mapping for a board.

    Board.canonical_analysis() provides the shared symmetry definition used
    throughout the project, keeping canonical state and move mapping consistent.
    """
    canonical: CanonicalBoard = board.canonical_analysis()
    original_to_canonical = {
        original_index: canonical.to_canonical_move(original_index)
        for original_index in range(Board.BOARD_SIZE)
    }

    return SymmetryMapping(
        canonical_state=canonical.canonical_state,
        transform_name=canonical.transform_name,
        transform=canonical.transform.transform,
        inverse_transform=canonical.transform.inverse_transform,
        original_to_canonical=original_to_canonical,
        original_state=board.to_string(),
    )


@dataclass
class Matchbox:
    """
    One MENACE matchbox.

    A matchbox stores bead counts for legal moves in one board state. More beads
    mean a move is more likely to be selected. If symmetry is enabled, moves are
    stored in canonical matchbox coordinates.
    """

    state: BoardState
    beads: Dict[Move, int]
    minimum_beads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("Matchbox state must be a non-empty string.")
        if self.minimum_beads < 1:
            raise ValueError("minimum_beads must be at least 1.")
        if not self.beads:
            raise ValueError("Matchbox must contain at least one move.")
        for move, count in self.beads.items():
            Board.validate_move_index(move)
            if count < self.minimum_beads:
                raise ValueError(
                    f"Move {move} has {count} beads, below minimum {self.minimum_beads}."
                )

    @classmethod
    def from_legal_moves(
        cls,
        state: BoardState,
        legal_moves: List[Move],
        initial_beads: int = 3,
        minimum_beads: int = 1,
    ) -> "Matchbox":
        """Create a matchbox from a state and legal move list."""
        if not legal_moves:
            raise ValueError("Cannot create matchbox without legal moves.")
        if initial_beads < minimum_beads:
            raise ValueError("initial_beads must be >= minimum_beads.")
        return cls(
            state=state,
            beads={move: initial_beads for move in sorted(set(legal_moves))},
            minimum_beads=minimum_beads,
        )

    @classmethod
    def from_board(
        cls,
        board: Board,
        initial_beads: int = 3,
        use_symmetry: bool = True,
        minimum_beads: int = 1,
    ) -> "Matchbox":
        """Create a matchbox for the current board."""
        if board.is_terminal():
            raise ValueError("Cannot create a matchbox for a terminal board.")

        if use_symmetry:
            mapping = _canonical_mapping(board)
            legal_moves = [
                mapping.to_canonical_move(move)
                for move in board.available_moves()
            ]
            state = mapping.canonical_state
        else:
            legal_moves = board.available_moves()
            state = board.to_string()

        return cls.from_legal_moves(
            state=state,
            legal_moves=legal_moves,
            initial_beads=initial_beads,
            minimum_beads=minimum_beads,
        )

    def legal_beads(self, legal_moves: List[Move]) -> Dict[Move, int]:
        """Return bead counts restricted to currently legal moves."""
        legal_set = set(legal_moves)
        return {
            move: max(self.minimum_beads, beads)
            for move, beads in self.beads.items()
            if move in legal_set
        }

    def sync_with_legal_moves(self, legal_moves: List[Move], initial_beads: int) -> None:
        """Ensure the matchbox contains exactly the currently legal moves."""
        if not legal_moves:
            raise ValueError("Cannot synchronise matchbox without legal moves.")

        legal_set = set(legal_moves)
        self.beads = {
            move: max(self.minimum_beads, beads)
            for move, beads in self.beads.items()
            if move in legal_set
        }
        for move in legal_moves:
            self.beads.setdefault(move, max(self.minimum_beads, initial_beads))
        if not self.beads:
            raise ValueError("No legal moves available after matchbox synchronisation.")

    def choose_move(self, rng: random.Random, legal_moves: List[Move]) -> Move:
        """Choose a legal move using weighted random selection."""
        self.sync_with_legal_moves(legal_moves, initial_beads=self.minimum_beads)
        filtered = self.legal_beads(legal_moves)
        moves = list(filtered.keys())
        weights = list(filtered.values())
        if not moves:
            raise ValueError("No legal weighted moves available for MENACE.")
        selected = rng.choices(moves, weights=weights, k=1)[0]
        if selected not in legal_moves:
            raise ValueError(
                f"MENACE attempted illegal matchbox move {selected}; legal moves are {legal_moves}."
            )
        return selected

    def update_move(self, move: Move, delta: int) -> None:
        """Add or remove beads for a move, never below minimum_beads."""
        if move not in self.beads:
            return
        self.beads[move] = max(self.minimum_beads, self.beads[move] + delta)

    def probabilities(self) -> MoveProbabilities:
        """Return move probabilities implied by bead counts."""
        total = sum(self.beads.values())
        if total <= 0:
            raise ValueError("Total bead count must be positive.")
        return {move: count / total for move, count in self.beads.items()}

    def total_beads(self) -> int:
        """Return total beads in this matchbox."""
        return sum(self.beads.values())

    def explain(self) -> str:
        """Return educational text explaining this matchbox."""
        probabilities = self.probabilities()
        lines = [
            f"Matchbox state: '{self.state}'",
            f"Total beads: {self.total_beads()}",
            "Move probabilities:",
        ]
        for move in sorted(self.beads):
            lines.append(
                f"- Move {move}: {self.beads[move]} beads "
                f"({probabilities[move]:.1%} chance)"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        """Convert to JSON-serialisable dictionary."""
        return {
            "state": self.state,
            "beads": {str(move): count for move, count in self.beads.items()},
            "minimum_beads": self.minimum_beads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Matchbox":
        """Restore Matchbox from JSON dictionary."""
        state = str(data["state"])
        beads_raw = data["beads"]
        minimum_beads = int(data.get("minimum_beads", 1))
        if not isinstance(beads_raw, dict):
            raise TypeError("Serialised beads must be a dictionary.")
        beads = {int(move): int(count) for move, count in beads_raw.items()}
        return cls(state=state, beads=beads, minimum_beads=minimum_beads)


@dataclass(frozen=True)
class MENACEDecision:
    """
    Records the details of a single MENACE decision, including the selected move,
    probabilities, and bead distribution for explanation.

    `move` is the visible original-board move actually played.
    `canonical_move` is the move index used inside the matchbox.
    """

    state: BoardState
    move: Move
    canonical_move: Move
    probabilities: MoveProbabilities
    bead_snapshot: Dict[Move, int]
    legal_moves: List[Move]
    canonical_legal_moves: List[Move]
    used_symmetry: bool
    board_before: BoardState
    canonical_state: BoardState
    transform_name: str
    transform: Tuple[int, ...]
    original_to_canonical: Dict[Move, Move]

    def canonical_to_original(self) -> Dict[Move, Move]:
        """Return inverse mapping: matchbox move -> original board square."""
        return {
            canonical_move: original_move
            for original_move, canonical_move in self.original_to_canonical.items()
        }

    def matchbox_move_to_original_square(self, matchbox_move: Move) -> Move:
        """Return the original board square represented by a matchbox move."""
        Board.validate_move_index(matchbox_move)
        mapping = self.canonical_to_original()
        if matchbox_move not in mapping:
            raise KeyError(f"Matchbox move {matchbox_move} is not valid for this decision.")
        return mapping[matchbox_move]

    def mapping_rows(self) -> List[Dict[str, int]]:
        """Return legal original-to-matchbox move mappings for this decision."""
        return [
            {
                "original_board_move": move,
                "matchbox_move": self.original_to_canonical[move],
            }
            for move in self.legal_moves
        ]

    def matchbox_probability_rows(self) -> List[Dict[str, object]]:
        """Return report-ready rows for matchbox moves and visible board squares."""
        canonical_to_original = self.canonical_to_original()
        rows: List[Dict[str, object]] = []
        for matchbox_move in sorted(self.bead_snapshot):
            rows.append(
                {
                    "matchbox_move": matchbox_move,
                    "original_board_square": canonical_to_original.get(matchbox_move),
                    "beads": self.bead_snapshot[matchbox_move],
                    "probability": self.probabilities[matchbox_move],
                    "selected": matchbox_move == self.canonical_move,
                }
            )
        return rows

    def explain_matchbox_probabilities(self) -> str:
        """Explain bead probabilities with direct matchbox-to-board mapping."""
        lines: List[str] = []
        for row in self.matchbox_probability_rows():
            selected_text = " <-- selected" if row["selected"] else ""
            original_text = (
                "not currently mapped"
                if row["original_board_square"] is None
                else f"original board square {row['original_board_square']}"
            )
            lines.append(
                f"- Matchbox move {row['matchbox_move']} -> {original_text}: "
                f"{row['beads']} beads, {float(row['probability']):.1%} chance"
                f"{selected_text}"
            )
        return "\n".join(lines)

    def original_index_grid(self) -> List[List[int]]:
        """Return standard visible board index grid."""
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def canonical_index_grid(self) -> List[List[int]]:
        """Return canonical/matchbox index grid used for this decision."""
        return [
            list(self.transform[0:3]),
            list(self.transform[3:6]),
            list(self.transform[6:9]),
        ]

    def explain_symmetry_mapping(self) -> str:
        """Explain the exact mapping used when this decision was made."""
        lines = [
            f"Canonical transform: {self.transform_name}",
            f"Board before MENACE: '{self.board_before}'",
            f"Canonical matchbox state: '{self.canonical_state}'",
            "",
            "Original board indexes:",
            self._format_index_grid(self.original_index_grid()),
            "",
            "Canonical / matchbox index grid:",
            self._format_index_grid(self.canonical_index_grid()),
            "",
            "Matchbox move -> original board square:",
        ]
        lines.extend(
            [
                self.explain_matchbox_probabilities(),
                "",
                f"Selected original board square: {self.move}",
                f"Corresponding matchbox move: {self.canonical_move}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_index_grid(grid: List[List[int]]) -> str:
        """Format a 3x3 integer grid as readable text."""
        return "\n".join(" ".join(str(value) for value in row) for row in grid)


@dataclass
class MENACEPlayer(Player):
    """Implements MENACE move selection, reinforcement, symmetry, and matchbox management."""

    mark: str = "O"
    name: str = "MENACE"
    initial_beads: int = 3
    minimum_beads: int = 1
    reward_win: int = 3
    reward_draw: int = 1
    penalty_loss: int = -1
    use_symmetry: bool = True
    seed: Optional[int] = None
    matchboxes: Dict[BoardState, Matchbox] = field(default_factory=dict)
    decisions_this_game: List[MENACEDecision] = field(default_factory=list)
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    evaluation_frozen: bool = False
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.minimum_beads < 1:
            raise ValueError("minimum_beads must be at least 1.")
        if self.initial_beads < self.minimum_beads:
            raise ValueError("initial_beads must be >= minimum_beads.")
        if self.reward_win < 0:
            raise ValueError("reward_win must be non-negative.")
        if self.reward_draw < 0:
            raise ValueError("reward_draw must be non-negative.")
        if self.penalty_loss > 0:
            raise ValueError("penalty_loss should be zero or negative.")
        self._rng = random.Random(self.seed)

    def choose_move(self, board: Board) -> Move:
        """Choose a legal original-board move."""
        move, _ = self.choose_move_with_explanation(board)
        return move

    def choose_move_with_explanation(self, board: Board) -> Tuple[Move, str]:
        """Choose a legal move and return an educational explanation."""
        if board.is_terminal():
            raise ValueError("MENACE cannot move on a terminal board.")
        legal_moves = board.available_moves()
        if not legal_moves:
            raise ValueError("MENACE cannot move because there are no legal moves.")

        state, canonical_legal_moves, mapping = self._state_and_legal_moves(board)
        if self.evaluation_frozen:
            # Evaluation must observe the trained policy without changing it.
            # An unseen state therefore gets a temporary, uniform matchbox.
            matchbox = self.matchboxes.get(state)
            if matchbox is None:
                matchbox = Matchbox.from_legal_moves(
                    state=state,
                    legal_moves=canonical_legal_moves,
                    initial_beads=self.initial_beads,
                    minimum_beads=self.minimum_beads,
                )
            else:
                # choose_move() synchronises its Matchbox before sampling.
                # Sample from a detached copy so even that housekeeping cannot
                # alter the persisted policy during evaluation.
                matchbox = Matchbox(
                    state=matchbox.state,
                    beads=matchbox.beads.copy(),
                    minimum_beads=matchbox.minimum_beads,
                )
        else:
            matchbox = self._get_or_create_matchbox_from_state(
                state, canonical_legal_moves
            )
            matchbox.sync_with_legal_moves(
                canonical_legal_moves, self.initial_beads
            )

        canonical_move = matchbox.choose_move(self._rng, canonical_legal_moves)
        original_move = mapping.to_original_move(canonical_move) if mapping is not None else canonical_move

        if original_move not in legal_moves or not board.is_legal_move(original_move):
            raise ValueError(
                f"MENACE selected illegal move {original_move} for board state "
                f"{board.to_string()!r}. Legal moves: {legal_moves}. "
                f"Canonical move: {canonical_move}, canonical legal moves: {canonical_legal_moves}."
            )

        probabilities_before = matchbox.probabilities()
        bead_snapshot = matchbox.beads.copy()

        if mapping is not None:
            board_before = mapping.original_state
            canonical_state = mapping.canonical_state
            transform_name = mapping.transform_name
            transform = mapping.transform
            original_to_canonical = mapping.original_to_canonical.copy()
        else:
            board_before = board.to_string()
            canonical_state = board_before
            transform_name = "identity"
            transform = Board.SYMMETRY_TRANSFORMS[0].transform
            original_to_canonical = {index: index for index in range(Board.BOARD_SIZE)}

        decision = MENACEDecision(
            state=matchbox.state,
            move=original_move,
            canonical_move=canonical_move,
            probabilities=probabilities_before,
            bead_snapshot=bead_snapshot,
            legal_moves=legal_moves.copy(),
            canonical_legal_moves=canonical_legal_moves.copy(),
            used_symmetry=self.use_symmetry,
            board_before=board_before,
            canonical_state=canonical_state,
            transform_name=transform_name,
            transform=transform,
            original_to_canonical=original_to_canonical,
        )
        self.decisions_this_game.append(decision)
        return original_move, self._build_decision_explanation(decision)

    def _state_and_legal_moves(self, board: Board) -> Tuple[BoardState, List[Move], Optional[SymmetryMapping]]:
        """Return state key, legal moves in matchbox coordinates, and mapping."""
        if self.use_symmetry:
            mapping = _canonical_mapping(board)
            canonical_legal_moves = [mapping.to_canonical_move(move) for move in board.available_moves()]
            return mapping.canonical_state, canonical_legal_moves, mapping
        return board.to_string(), board.available_moves(), None

    def _get_or_create_matchbox_from_state(self, state: BoardState, legal_moves: List[Move]) -> Matchbox:
        """Retrieve or create a matchbox for a state."""
        if state not in self.matchboxes:
            self.matchboxes[state] = Matchbox.from_legal_moves(
                state=state,
                legal_moves=legal_moves,
                initial_beads=self.initial_beads,
                minimum_beads=self.minimum_beads,
            )
        return self.matchboxes[state]

    def _get_or_create_matchbox(self, board: Board) -> Matchbox:
        """Return the matchbox associated with the supplied board state."""
        state, legal_moves, _ = self._state_and_legal_moves(board)
        return self._get_or_create_matchbox_from_state(state, legal_moves)

    def _state_key(self, board: Board) -> BoardState:
        """Return the state key used by MENACE."""
        state, _, _ = self._state_and_legal_moves(board)
        return state

    def learn_from_result(self, result: str) -> None:
        """Update beads after a game result from MENACE's perspective."""
        if result not in {"win", "loss", "draw"}:
            raise ValueError("result must be one of: win, loss, draw.")
        if result == "win":
            delta = self.reward_win
            self.wins += 1
        elif result == "draw":
            delta = self.reward_draw
            self.draws += 1
        else:
            delta = self.penalty_loss
            self.losses += 1
        for decision in self.decisions_this_game:
            matchbox = self.matchboxes.get(decision.state)
            if matchbox is not None:
                matchbox.update_move(decision.canonical_move, delta)
        self.games_played += 1
        self.decisions_this_game.clear()

    def learn_from_game(self, result: str) -> None:
        """Apply learning from a completed game result."""
        self.learn_from_result(result)

    def reset_game_memory(self) -> None:
        """Clear decisions from the current incomplete game."""
        self.decisions_this_game.clear()

    def reset_learning(self) -> None:
        """Clear all matchboxes and training counters."""
        self.matchboxes.clear()
        self.decisions_this_game.clear()
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    # ============================================================
    # How MENACE Learns
    # ============================================================
    def explain_state(self, board: Board) -> str:
        """Explain the current board's MENACE matchbox."""
        if board.is_terminal():
            return "This board is terminal (win, loss, or draw), so MENACE does not create a new matchbox for this state."
        state, legal_moves, _ = self._state_and_legal_moves(board)
        matchbox = self._get_or_create_matchbox_from_state(state, legal_moves)
        matchbox.sync_with_legal_moves(legal_moves, self.initial_beads)
        symmetry_text = (
            "Symmetry reduction is enabled: equivalent rotations/reflections share the same matchbox."
            if self.use_symmetry
            else "Symmetry reduction is disabled: raw board states are used."
        )
        return f"{symmetry_text}\n\n{matchbox.explain()}"

    def explain_strategy(self) -> str:
        """Return high-level educational explanation."""
        return (
            "MENACE uses matchboxes and beads. Each board state has a matchbox. "
            "Each legal move has beads. MENACE samples moves according to bead "
            "counts, then changes those bead counts after wins, losses, or draws. "
            "When symmetry is enabled, rotations and reflections of equivalent "
            "boards share one canonical matchbox, which improves data efficiency."
        )

    def training_summary(self) -> Dict[str, float | int | bool]:
        """Return training summary statistics."""
        win_rate = self.wins / self.games_played if self.games_played else 0.0
        loss_rate = self.losses / self.games_played if self.games_played else 0.0
        draw_rate = self.draws / self.games_played if self.games_played else 0.0
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
            "matchboxes": len(self.matchboxes),
            "total_beads": self.total_beads(),
            "use_symmetry": self.use_symmetry,
        }

    def total_beads(self) -> int:
        """Return total beads across all matchboxes."""
        return sum(matchbox.total_beads() for matchbox in self.matchboxes.values())

    def latest_decision(self) -> Optional[MENACEDecision]:
        """Return the latest MENACE decision in the current game, if any."""
        if not self.decisions_this_game:
            return None
        return self.decisions_this_game[-1]

    def last_decision_explanation(self) -> str:
        """Return the latest MENACE decision explanation, or a clear empty-state message."""
        decision = self.latest_decision()
        if decision is None:
            return "MENACE has not made a decision in the current game."
        return self._build_decision_explanation(decision)

    def decision_mapping_explanations(self) -> List[str]:
        """Return exact symmetry-mapping explanations for current-game decisions."""
        return [decision.explain_symmetry_mapping() for decision in self.decisions_this_game]

    def _build_decision_explanation(self, decision: MENACEDecision) -> str:
        """Build educational explanation for a MENACE decision."""
        lines = [
            f"MENACE used matchbox state: '{decision.state}'.",
            f"Canonical transform used for this decision: {decision.transform_name}.",
            f"Symmetry enabled: {decision.used_symmetry}.",
            f"Legal original-board moves were: {decision.legal_moves}.",
            f"Legal matchbox moves were: {decision.canonical_legal_moves}.",
            f"MENACE selected original move {decision.move} (original board square {decision.move}).",
            f"The corresponding matchbox move was {decision.canonical_move}.",
            "At the time of selection, the bead counts were:",
            decision.explain_matchbox_probabilities(),
        ]
        lines.append(
            "This shows that MENACE is not following fixed rules. "
            "It samples actions from probabilities shaped by reinforcement."
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        """Convert MENACEPlayer to JSON-serialisable dictionary."""
        return {
            "mark": self.mark,
            "name": self.name,
            "initial_beads": self.initial_beads,
            "minimum_beads": self.minimum_beads,
            "reward_win": self.reward_win,
            "reward_draw": self.reward_draw,
            "penalty_loss": self.penalty_loss,
            "use_symmetry": self.use_symmetry,
            "seed": self.seed,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "matchboxes": {state: matchbox.to_dict() for state, matchbox in self.matchboxes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MENACEPlayer":
        """Restore MENACEPlayer from JSON dictionary."""
        player = cls(
            mark=str(data.get("mark", "O")),
            name=str(data.get("name", "MENACE")),
            initial_beads=int(data.get("initial_beads", 3)),
            minimum_beads=int(data.get("minimum_beads", 1)),
            reward_win=int(data.get("reward_win", 3)),
            reward_draw=int(data.get("reward_draw", 1)),
            penalty_loss=int(data.get("penalty_loss", -1)),
            use_symmetry=bool(data.get("use_symmetry", True)),
            seed=data.get("seed"),  # type: ignore[arg-type]
        )
        player.games_played = int(data.get("games_played", 0))
        player.wins = int(data.get("wins", 0))
        player.losses = int(data.get("losses", 0))
        player.draws = int(data.get("draws", 0))
        matchboxes_raw = data.get("matchboxes", {})
        if isinstance(matchboxes_raw, dict):
            player.matchboxes = {
                str(state): Matchbox.from_dict(matchbox_data)  # type: ignore[arg-type]
                for state, matchbox_data in matchboxes_raw.items()
            }
        return player
