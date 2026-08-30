"""
trainer.py

Coordinates MENACE training, evaluation, and AI comparison experiments.

This module runs complete experiments by combining the Game engine,
MENACE player, and opponent players. It records results, builds training
histories, and produces quantitative evidence for interface display
and CSV export.

It supports:
    - MENACE training
    - evaluation without learning
    - repeated experiments
    - comparison against Random, Heuristic, and Minimax opponents
    - CSV summaries

This module coordinates experiments only. MENACE learning is implemented
inside menace.py, statistical analysis is handled by statistics.py,
and visualisations are produced by visualisation.py.
"""

from __future__ import annotations

import copy
import random
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from src.board import Board
from src.game import Game, GameResult
from src.menace import MENACEPlayer


@runtime_checkable
class TrainableOpponent(Protocol):
    """Defines the minimum interface required for training and evaluation opponents."""

    mark: str
    name: str

    def choose_move(self, board: Board) -> int:
        """Choose a legal move from the current board."""


PlayerFactory = Callable[[], TrainableOpponent]
MENACEFactory = Callable[[], MENACEPlayer]


@dataclass(frozen=True)
class TrainingConfig:
    """
    Stores the configuration for one MENACE training or evaluation experiment.

    It defines the number of games, reward settings, logging frequency,
    experiment name, output location, and other options required to run
    a reproducible experiment.
    """

    games: int = 1000
    reward_win: int = 3
    reward_draw: int = 1
    penalty_loss: int = -1
    seed: Optional[int] = None
    log_interval: int = 1
    output_path: Optional[Path] = None
    experiment_name: str = "menace_vs_baseline"
    keep_raw_results: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values early."""
        if self.games <= 0:
            raise ValueError("games must be greater than zero.")
        if self.reward_win < 0:
            raise ValueError("reward_win must be non-negative.")
        if self.reward_draw < 0:
            raise ValueError("reward_draw must be non-negative.")
        if self.penalty_loss > 0:
            raise ValueError("penalty_loss should be zero or negative.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be greater than zero.")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")


@dataclass(frozen=True)
class TrainingRow:
    """Represents one cumulative row of training or evaluation results."""

    experiment_name: str
    mode: str
    game: int
    outcome: str
    menace_wins: int
    opponent_wins: int
    random_wins: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    matchboxes: int
    total_beads: int
    menace_games_played: int
    opponent_name: str
    seed: Optional[int]
    total_moves: int
    winner: Optional[str]
    final_board: str


@dataclass(frozen=True)
class TrainingReport:
    """
    Stores the complete output of a training or evaluation session,
    including the history, summary, and game results.
    """

    history: pd.DataFrame
    final_summary: Dict[str, float | int | str | None]
    results: List[GameResult]


@dataclass(frozen=True)
class OpponentSpec:
    """Describes an evaluation opponent and its player factory."""

    name: str
    opponent_type: str
    factory: PlayerFactory

    def __post_init__(self) -> None:
        """Validate opponent labels and factory."""
        if not self.name.strip():
            raise ValueError("OpponentSpec.name cannot be empty.")
        if not self.opponent_type.strip():
            raise ValueError("OpponentSpec.opponent_type cannot be empty.")
        if not callable(self.factory):
            raise TypeError("OpponentSpec.factory must be callable.")


@dataclass(frozen=True)
class ComparisonConfig:
    """
    Configuration for comparative MENACE evaluation.

    The comparison flow is:
        1. Train a fresh MENACE player.
        2. Evaluate the trained model against each opponent without learning.
        3. Repeat the process when repetitions > 1.
    """

    trained_games: int = 5000
    evaluation_games: int = 1000
    repetitions: int = 1
    reward_win: int = 3
    reward_draw: int = 1
    penalty_loss: int = -1
    seed: Optional[int] = 42
    experiment_name: str = "menace_comparison"

    def __post_init__(self) -> None:
        """Validate comparison configuration values."""
        if self.trained_games <= 0:
            raise ValueError("trained_games must be greater than zero.")
        if self.evaluation_games <= 0:
            raise ValueError("evaluation_games must be greater than zero.")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be greater than zero.")
        if self.reward_win < 0:
            raise ValueError("reward_win must be non-negative.")
        if self.reward_draw < 0:
            raise ValueError("reward_draw must be non-negative.")
        if self.penalty_loss > 0:
            raise ValueError("penalty_loss should be zero or negative.")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")


class Trainer:
    """
    Coordinates MENACE training and evaluation against one opponent.

    Trainer deliberately calls Game to play matches and MENACEPlayer to learn.
    It does not implement board rules or the MENACE algorithm itself.
    """

    REQUIRED_COLUMNS = {
        "experiment_name",
        "mode",
        "game",
        "outcome",
        "menace_wins",
        "opponent_wins",
        "random_wins",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "matchboxes",
        "total_beads",
        "menace_games_played",
        "opponent_name",
        "seed",
    }

    def __init__(
        self,
        menace_player: MENACEPlayer,
        opponent: TrainableOpponent,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.menace_player = menace_player
        self.opponent = opponent
        self.config = config or TrainingConfig()

        self._validate_players()
        self._apply_reward_config_to_menace()

    def run(self, train: bool = True) -> pd.DataFrame:
        """
        Run the configured experiment and return the history DataFrame.

        If config.output_path is set, the history is saved automatically.
        """
        report = self.run_with_report(train=train)

        if self.config.output_path is not None:
            self.save_history(report.history, self.config.output_path)

        return report.history

    def run_with_report(self, train: bool = True) -> TrainingReport:
        """
        Run training/evaluation and return a full TrainingReport.

        Args:
            train:
                True means MENACE updates bead counts after each game.
                False means evaluation only; decisions are not reinforced.
        """
        mode = "train" if train else "evaluate"
        previous_frozen_state = self.menace_player.evaluation_frozen
        self.menace_player.evaluation_frozen = not train

        menace_wins = 0
        opponent_wins = 0
        draws = 0

        rows: List[TrainingRow] = []
        results: List[GameResult] = []

        for game_number in range(1, self.config.games + 1):
            result = self._play_single_game(train=train)

            if self.config.keep_raw_results:
                results.append(result)

            menace_result = result.result_for(self.menace_player.mark)

            if menace_result == "win":
                menace_wins += 1
                outcome = "menace_win"
            elif menace_result == "loss":
                opponent_wins += 1
                outcome = "opponent_win"
            elif menace_result == "draw":
                draws += 1
                outcome = "draw"
            else:
                raise ValueError(f"Unexpected MENACE game result: {menace_result!r}")

            if self._should_log(game_number):
                rows.append(
                    self._build_training_row(
                        game_number=game_number,
                        outcome=outcome,
                        menace_wins=menace_wins,
                        opponent_wins=opponent_wins,
                        draws=draws,
                        mode=mode,
                        result=result,
                    )
                )

        history = pd.DataFrame([asdict(row) for row in rows])
        self._validate_history(history)

        final_summary = self._build_final_summary(
            history=history,
            mode=mode,
            total_games=self.config.games,
        )

        report = TrainingReport(
            history=history,
            final_summary=final_summary,
            results=results,
        )
        self.menace_player.evaluation_frozen = previous_frozen_state
        return report

    def save_history(self, history: pd.DataFrame, path: Path | str) -> None:
        """Validate and save a training/evaluation history to CSV."""
        self._validate_history(history)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_frame(history).to_csv(output_path, index=False)

    @staticmethod
    def evidence_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Return clear, learner-facing evidence with unambiguous columns."""
        evidence = frame.copy()
        evidence = evidence.drop(columns=["random_wins"], errors="ignore")
        if "final_board" in evidence.columns:
            def letter_board(value: object) -> str:
                cells = str(value).ljust(Board.BOARD_SIZE)[: Board.BOARD_SIZE]
                return "; ".join(
                    f"{letter}={cell if cell in {'X', 'O'} else '-'}"
                    for letter, cell in zip("ABCDEFGHI", cells)
                )
            evidence["final_board_letters"] = evidence["final_board"].map(
                letter_board
            )
        return evidence

    def save_summary(
        self,
        summary: Dict[str, float | int | str | None],
        path: Path | str,
        append: bool = True,
    ) -> None:
        """Save or append a one-row experiment summary CSV."""
        if not summary:
            raise ValueError("summary cannot be empty.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        new_row = pd.DataFrame([summary])

        if append and output_path.exists():
            existing = pd.read_csv(output_path)
            new_row = pd.concat([existing, new_row], ignore_index=True)

        new_row.to_csv(output_path, index=False)

    def _play_single_game(self, train: bool) -> GameResult:
        """Play one complete game and optionally reinforce MENACE."""
        self.menace_player.reset_game_memory()

        player_x, player_o = self._players_by_mark()

        game = Game(
            player_x=player_x,
            player_o=player_o,
            board=Board.empty(),
            record_history=True,
        )

        result = game.play(learn=False)
        menace_result = result.result_for(self.menace_player.mark)

        if train:
            self.menace_player.learn_from_result(menace_result)
        else:
            self.menace_player.reset_game_memory()

        return result

    def _players_by_mark(self) -> tuple[TrainableOpponent | MENACEPlayer, TrainableOpponent | MENACEPlayer]:
        """Return players in X/O order for the Game engine."""
        players: Dict[str, TrainableOpponent | MENACEPlayer] = {
            self.menace_player.mark: self.menace_player,
            self.opponent.mark: self.opponent,
        }

        if "X" not in players or "O" not in players:
            raise ValueError("One player must use X and the other must use O.")

        return players["X"], players["O"]

    def _should_log(self, game_number: int) -> bool:
        """Return True when the current cumulative state should be logged."""
        return (
            game_number % self.config.log_interval == 0
            or game_number == self.config.games
        )

    def _build_training_row(
        self,
        game_number: int,
        outcome: str,
        menace_wins: int,
        opponent_wins: int,
        draws: int,
        mode: str,
        result: GameResult,
    ) -> TrainingRow:
        """Build one cumulative TrainingRow."""
        total = menace_wins + opponent_wins + draws

        if total <= 0:
            raise ValueError("Cannot build a training row before any game result exists.")

        return TrainingRow(
            experiment_name=self.config.experiment_name,
            mode=mode,
            game=game_number,
            outcome=outcome,
            menace_wins=menace_wins,
            opponent_wins=opponent_wins,
            random_wins=opponent_wins,
            draws=draws,
            win_rate=menace_wins / total,
            loss_rate=opponent_wins / total,
            draw_rate=draws / total,
            matchboxes=len(self.menace_player.matchboxes),
            total_beads=self.menace_player.total_beads(),
            menace_games_played=self.menace_player.games_played,
            opponent_name=self.opponent.name,
            seed=self.config.seed,
            total_moves=result.total_moves,
            winner=result.winner,
            final_board=result.final_board,
        )

    def _build_final_summary(
        self,
        history: pd.DataFrame,
        mode: str,
        total_games: int,
    ) -> Dict[str, float | int | str | None]:
        """Build report-ready final summary from the last history row."""
        final = history.iloc[-1]

        return {
            "experiment_name": self.config.experiment_name,
            "mode": mode,
            "games": total_games,
            "opponent_name": self.opponent.name,
            "seed": self.config.seed,
            "reward_win": self.config.reward_win,
            "reward_draw": self.config.reward_draw,
            "penalty_loss": self.config.penalty_loss,
            "menace_wins": int(final["menace_wins"]),
            "opponent_wins": int(final["opponent_wins"]),
            "random_wins": int(final["random_wins"]),
            "draws": int(final["draws"]),
            "win_rate": float(final["win_rate"]),
            "loss_rate": float(final["loss_rate"]),
            "draw_rate": float(final["draw_rate"]),
            "matchboxes": int(final["matchboxes"]),
            "total_beads": int(final["total_beads"]),
            "menace_games_played": int(final["menace_games_played"]),
            "final_board": str(final.get("final_board", "")),
        }

    def _validate_players(self) -> None:
        """Validate that MENACE and opponent can legally play a game."""
        Board.validate_mark(self.menace_player.mark)
        Board.validate_mark(self.opponent.mark)

        if self.menace_player.mark == self.opponent.mark:
            raise ValueError("MENACE and opponent must use different marks.")

        if not hasattr(self.opponent, "choose_move"):
            raise TypeError("Opponent must implement choose_move(board).")

        if not callable(getattr(self.opponent, "choose_move")):
            raise TypeError("Opponent.choose_move must be callable.")

    def _validate_history(self, history: pd.DataFrame) -> None:
        """Validate a Trainer-produced history DataFrame."""
        if history.empty:
            raise ValueError("Training history cannot be empty.")

        missing = self.REQUIRED_COLUMNS.difference(history.columns)
        if missing:
            raise ValueError(
                "Training history missing required columns: "
                + ", ".join(sorted(missing))
            )

        if (history["game"] <= 0).any():
            raise ValueError("All game values must be positive.")

        for column in ["win_rate", "loss_rate", "draw_rate"]:
            invalid = (history[column] < 0) | (history[column] > 1)
            if invalid.any():
                raise ValueError(f"{column} must be between 0 and 1.")

        totals = history["menace_wins"] + history["opponent_wins"] + history["draws"]
        if (totals != history["game"]).any():
            raise ValueError("Outcome counts must sum to the game number.")

    def _apply_reward_config_to_menace(self) -> None:
        """Apply experiment reward settings to the MENACE player."""
        self.menace_player.reward_win = self.config.reward_win
        self.menace_player.reward_draw = self.config.reward_draw
        self.menace_player.penalty_loss = self.config.penalty_loss


class ExperimentSuite:
    """Runs repeated independent training experiments to assess performance consistency."""

    def __init__(
        self,
        menace_factory: MENACEFactory,
        opponent_factory: PlayerFactory,
        config: TrainingConfig,
        repetitions: int = 5,
    ) -> None:
        if repetitions <= 0:
            raise ValueError("repetitions must be greater than zero.")
        if not callable(menace_factory):
            raise TypeError("menace_factory must be callable.")
        if not callable(opponent_factory):
            raise TypeError("opponent_factory must be callable.")

        self.menace_factory = menace_factory
        self.opponent_factory = opponent_factory
        self.config = config
        self.repetitions = repetitions

    def _seed_for(self, repetition: int, label: str) -> Optional[int]:
        """Derive one stable non-negative seed for a repeated experiment."""
        if self.config.seed is None:
            return None
        label_offset = sum((index + 1) * ord(char) for index, char in enumerate(label))
        return int((int(self.config.seed) + repetition * 10_007 + label_offset) % (2**32))

    def run(self) -> pd.DataFrame:
        """Run repeated training experiments and return summary rows."""
        summaries: List[Dict[str, float | int | str | None]] = []

        for repetition in range(1, self.repetitions + 1):
            menace = self.menace_factory()
            opponent = self.opponent_factory()

            config = TrainingConfig(
                games=self.config.games,
                reward_win=self.config.reward_win,
                reward_draw=self.config.reward_draw,
                penalty_loss=self.config.penalty_loss,
                seed=self._seed_for(repetition, "training-opponent"),
                log_interval=self.config.log_interval,
                output_path=None,
                experiment_name=f"{self.config.experiment_name}_rep_{repetition}",
                keep_raw_results=False,
            )

            trainer = Trainer(
                menace_player=menace,
                opponent=opponent,
                config=config,
            )

            report = trainer.run_with_report(train=True)
            summary = dict(report.final_summary)
            summary["repetition"] = repetition
            summaries.append(summary)

        results = pd.DataFrame(summaries)
        if results.empty:
            raise ValueError("ExperimentSuite produced no results.")

        return results


class ComparisonExperiment:
    """Trains MENACE and evaluates it against multiple opponent strategies."""

    REQUIRED_COLUMNS = {
        "phase",
        "repetition",
        "trained_games",
        "evaluation_games",
        "opponent_type",
        "opponent_name",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "menace_wins",
        "opponent_wins",
        "draws",
    }

    def __init__(
        self,
        menace_factory: Optional[MENACEFactory] = None,
        training_opponent_factory: Optional[PlayerFactory] = None,
        evaluation_opponents: Optional[List[OpponentSpec]] = None,
        config: Optional[ComparisonConfig] = None,
        *,
        menace_player: Optional[MENACEPlayer] = None,
        games_per_opponent: Optional[int] = None,
        output_path: Optional[Path | str] = None,
    ) -> None:
        if menace_factory is None and menace_player is not None:
            menace_factory = lambda: menace_player

        if games_per_opponent is not None:
            config = config or ComparisonConfig(
                trained_games=max(100, games_per_opponent),
                evaluation_games=games_per_opponent,
                repetitions=1,
                experiment_name="streamlit_menace_ai_comparison",
            )

        if training_opponent_factory is None:
            from src.players import RandomPlayer

            training_opponent_factory = lambda: RandomPlayer(mark="X", seed=42)

        if evaluation_opponents is None:
            evaluation_opponents = self.default_evaluation_opponents()

        if menace_factory is None:
            raise ValueError("menace_factory or menace_player must be provided.")

        if not callable(menace_factory):
            raise TypeError("menace_factory must be callable.")

        if not callable(training_opponent_factory):
            raise TypeError("training_opponent_factory must be callable.")

        if not evaluation_opponents:
            raise ValueError("At least one evaluation opponent is required.")

        self.menace_factory = menace_factory
        self.training_opponent_factory = training_opponent_factory
        self.evaluation_opponents = evaluation_opponents
        self.config = config or ComparisonConfig()
        self.output_path = Path(output_path) if output_path is not None else None

    @staticmethod
    def default_evaluation_opponents() -> List[OpponentSpec]:
        """
        Return the standard evaluation opponents used by the comparison workflow.

        The set includes a Random baseline, rule-based Heuristic AI,
        and optimal Minimax search.
        """
        from src.players import HeuristicPlayer, MinimaxPlayer, RandomPlayer

        return [
            OpponentSpec(
                name="Random Baseline",
                opponent_type="Random",
                factory=lambda: RandomPlayer(mark="X", seed=42),
            ),
            OpponentSpec(
                name="Heuristic AI",
                opponent_type="Heuristic",
                factory=lambda: HeuristicPlayer(mark="X"),
            ),
            OpponentSpec(
                name="Minimax AI",
                opponent_type="Minimax",
                factory=lambda: MinimaxPlayer(mark="X"),
            ),
        ]

    def run(self) -> pd.DataFrame:
        """Run the full comparison experiment and return result rows."""
        rows: List[Dict[str, float | int | str | None]] = []

        for repetition in range(1, self.config.repetitions + 1):
            menace = self.menace_factory()
            self._reseed_player(
                menace, self._seed_for(repetition, "menace-training")
            )

            training_report = self._train_menace_for_repetition(
                menace=menace,
                repetition=repetition,
            )

            training_summary = dict(training_report.final_summary)
            training_summary.update(
                {
                    "phase": "training",
                    "repetition": repetition,
                    "trained_games": self.config.trained_games,
                    "evaluation_games": 0,
                    "opponent_type": "training_baseline",
                }
            )
            rows.append(training_summary)

            for opponent_spec in self.evaluation_opponents:
                evaluation_menace = copy.deepcopy(menace)
                self._reseed_player(
                    evaluation_menace,
                    self._seed_for(repetition, opponent_spec.opponent_type),
                )
                evaluation_menace.evaluation_frozen = True
                evaluation_report = self._evaluate_against_opponent(
                    menace=evaluation_menace,
                    opponent_spec=opponent_spec,
                    repetition=repetition,
                )

                evaluation_summary = dict(evaluation_report.final_summary)
                evaluation_summary.update(
                    {
                        "phase": "evaluation",
                        "repetition": repetition,
                        "trained_games": self.config.trained_games,
                        "evaluation_games": self.config.evaluation_games,
                        "opponent_type": opponent_spec.opponent_type,
                    }
                )
                rows.append(evaluation_summary)

        results = pd.DataFrame(rows)
        self._validate_results(results)

        if self.output_path is not None:
            self.save_comparison(results, self.output_path)

        return results

    def _train_menace_for_repetition(
        self,
        menace: MENACEPlayer,
        repetition: int,
    ) -> TrainingReport:
        """Train one MENACE instance for a comparison repetition."""
        opponent = self.training_opponent_factory()
        self._reseed_player(
            opponent, self._seed_for(repetition, "training-opponent")
        )

        trainer = Trainer(
            menace_player=menace,
            opponent=opponent,
            config=TrainingConfig(
                games=self.config.trained_games,
                reward_win=self.config.reward_win,
                reward_draw=self.config.reward_draw,
                penalty_loss=self.config.penalty_loss,
                seed=self._seed_for(repetition, "training-opponent"),
                log_interval=max(1, self.config.trained_games),
                output_path=None,
                experiment_name=f"{self.config.experiment_name}_training_rep_{repetition}",
                keep_raw_results=False,
            ),
        )

        return trainer.run_with_report(train=True)

    def _evaluate_against_opponent(
        self,
        menace: MENACEPlayer,
        opponent_spec: OpponentSpec,
        repetition: int,
    ) -> TrainingReport:
        """Evaluate trained MENACE against one opponent without learning."""
        opponent = opponent_spec.factory()
        self._reseed_player(
            opponent,
            self._seed_for(repetition, f"opponent-{opponent_spec.opponent_type}"),
        )

        trainer = Trainer(
            menace_player=menace,
            opponent=opponent,
            config=TrainingConfig(
                games=self.config.evaluation_games,
                reward_win=self.config.reward_win,
                reward_draw=self.config.reward_draw,
                penalty_loss=self.config.penalty_loss,
                seed=self._seed_for(
                    repetition, f"opponent-{opponent_spec.opponent_type}"
                ),
                log_interval=max(1, self.config.evaluation_games),
                output_path=None,
                experiment_name=(
                    f"{self.config.experiment_name}_vs_"
                    f"{opponent_spec.opponent_type}_rep_{repetition}"
                ),
                keep_raw_results=False,
            ),
        )

        return trainer.run_with_report(train=False)

    def _seed_for(self, repetition: int, label: str) -> Optional[int]:
        """Derive a stable independent seed for each repetition and role."""
        if self.config.seed is None:
            return None
        label_code = zlib.crc32(label.encode("utf-8"))
        return (int(self.config.seed) + repetition * 100_003 + label_code) % (
            2**31 - 1
        )

    @staticmethod
    def _reseed_player(player: Player, seed: Optional[int]) -> None:
        """Reset a player's private random stream when it exposes one."""
        if seed is None:
            return
        if hasattr(player, "seed"):
            player.seed = seed
        if hasattr(player, "_rng"):
            player._rng = random.Random(seed)

    @classmethod
    def _validate_results(cls, results: pd.DataFrame) -> None:
        """Validate comparison output before saving or dashboard display."""
        if results.empty:
            raise ValueError("Comparison experiment produced no rows.")

        missing = cls.REQUIRED_COLUMNS.difference(results.columns)
        if missing:
            raise ValueError(
                "Comparison results missing required columns: "
                + ", ".join(sorted(missing))
            )

        for column in ["win_rate", "loss_rate", "draw_rate"]:
            invalid = (results[column] < 0) | (results[column] > 1)
            if invalid.any():
                raise ValueError(f"{column} values must be between 0 and 1.")

    @classmethod
    def save_comparison(cls, results: pd.DataFrame, path: Path | str) -> None:
        """Validate and save comparison results to CSV."""
        cls._validate_results(results)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Trainer.evidence_frame(results).to_csv(output_path, index=False)
