"""
main.py

Command-line experiment runner for reproducible MENACE experiments.

This module provides a reproducible command-line interface for running
MENACE experiments without using the Streamlit application.

It supports:

    - training MENACE
    - evaluating trained models
    - comparing MENACE against Random, Heuristic, and Minimax opponents
    - saving models
    - exporting report-ready CSV evidence

This module coordinates experiments only. Game logic, reinforcement
learning, statistics, persistence, and visualisation are implemented
in their respective modules.

Typical usage:

    python main.py train
    python main.py evaluate
    python main.py compare
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from src.menace import MENACEPlayer
    from src.players import HeuristicPlayer, MinimaxPlayer, RandomPlayer
    from src.trainer import (
        ComparisonConfig,
        ComparisonExperiment,
        OpponentSpec,
        Trainer,
        TrainingConfig,
    )
    from src.persistence import ModelPersistence
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: Required src modules could not be imported.\n"
        "Make sure this file is run from the project root folder and that "
        "the src/ package exists.\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


@dataclass(frozen=True)
class ExperimentPaths:
    """Centralised paths used by the command-line experiment runner."""

    model_path: Path = Path("saved_models/menace_model.json")
    results_dir: Path = Path("results")
    training_log_path: Path = Path("results/training_log.csv")
    evaluation_log_path: Path = Path("results/evaluation_log.csv")
    comparison_results_path: Path = Path("results/comparison_results.csv")
    experiment_summary_path: Path = Path("results/experiment_summary.csv")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one MENACE training or evaluation run."""

    games: int
    reward_win: int = 3
    reward_draw: int = 1
    penalty_loss: int = -1
    seed: Optional[int] = 42
    save_model: Optional[Path] = None
    load_model: Optional[Path] = None
    experiment_name: str = "menace_cli_experiment"


@dataclass(frozen=True)
class ComparisonRunConfig:
    """Configuration for comparative AI evaluation."""

    trained_games: int
    evaluation_games: int
    repetitions: int = 1
    reward_win: int = 3
    reward_draw: int = 1
    penalty_loss: int = -1
    seed: Optional[int] = 42
    experiment_name: str = "menace_cli_comparison"


class MENACEExperimentRunner:
    """
    Command-line controller for reproducible MENACE experiments.

    This class does not implement game rules or MENACE learning. It coordinates
    the project modules and saves reproducible experiment evidence.
    """

    def __init__(self, paths: ExperimentPaths) -> None:
        self.paths = paths
        self._prepare_directories()

    def train(self, config: ExperimentConfig) -> pd.DataFrame:
        """Train MENACE against a Random baseline and save CSV/model evidence."""
        logging.info("Starting MENACE training for %s games.", config.games)

        menace = self._load_or_create_menace(
            model_path=config.load_model,
            seed=config.seed,
        )
        opponent = RandomPlayer(mark="X", seed=config.seed)

        training_config = TrainingConfig(
            games=config.games,
            reward_win=config.reward_win,
            reward_draw=config.reward_draw,
            penalty_loss=config.penalty_loss,
            seed=config.seed,
            output_path=self.paths.training_log_path,
            experiment_name=config.experiment_name,
        )

        trainer = Trainer(
            menace_player=menace,
            opponent=opponent,
            config=training_config,
        )

        history = trainer.run(train=True)
        self._validate_training_history(history)

        self._append_experiment_metadata(
            mode="train",
            history=history,
            games=config.games,
            reward_win=config.reward_win,
            reward_draw=config.reward_draw,
            penalty_loss=config.penalty_loss,
            seed=config.seed,
            output_path=self.paths.training_log_path,
        )

        model_path = config.save_model or self.paths.model_path
        self._save_model(menace, model_path)

        logging.info("Training log saved to %s", self.paths.training_log_path)
        logging.info("MENACE model saved to %s", model_path)

        return history

    def evaluate(self, config: ExperimentConfig) -> pd.DataFrame:
        """
        Evaluate MENACE against Random without learning.

        This is useful for measuring a saved model after training.
        """
        logging.info("Starting MENACE evaluation for %s games.", config.games)

        menace = self._load_or_create_menace(
            model_path=config.load_model,
            seed=config.seed,
        )
        opponent = RandomPlayer(mark="X", seed=config.seed)

        evaluation_config = TrainingConfig(
            games=config.games,
            reward_win=0,
            reward_draw=0,
            penalty_loss=0,
            seed=config.seed,
            output_path=self.paths.evaluation_log_path,
            experiment_name=config.experiment_name,
        )

        trainer = Trainer(
            menace_player=menace,
            opponent=opponent,
            config=evaluation_config,
        )

        results = trainer.run(train=False)
        self._validate_training_history(results)

        self._append_experiment_metadata(
            mode="evaluate",
            history=results,
            games=config.games,
            reward_win=0,
            reward_draw=0,
            penalty_loss=0,
            seed=config.seed,
            output_path=self.paths.evaluation_log_path,
        )

        logging.info("Evaluation log saved to %s", self.paths.evaluation_log_path)

        return results

    def compare(self, config: ComparisonRunConfig) -> pd.DataFrame:
        """
        Train MENACE and evaluate it against Random, Heuristic, and Minimax.

        This command produces comparative evaluation evidence across multiple
        opponents rather than only running an interactive game.
        """
        logging.info(
            "Starting comparison: trained_games=%s, evaluation_games=%s, repetitions=%s.",
            config.trained_games,
            config.evaluation_games,
            config.repetitions,
        )

        experiment = ComparisonExperiment(
            menace_factory=lambda: MENACEPlayer(
                mark="O",
                seed=config.seed,
                use_symmetry=True,
                reward_win=config.reward_win,
                reward_draw=config.reward_draw,
                penalty_loss=config.penalty_loss,
            ),
            training_opponent_factory=lambda: RandomPlayer(
                mark="X",
                seed=config.seed,
            ),
            evaluation_opponents=[
                OpponentSpec(
                    name="Random Baseline",
                    opponent_type="Random",
                    factory=lambda: RandomPlayer(mark="X", seed=config.seed),
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
            ],
            config=ComparisonConfig(
                trained_games=config.trained_games,
                evaluation_games=config.evaluation_games,
                repetitions=config.repetitions,
                reward_win=config.reward_win,
                reward_draw=config.reward_draw,
                penalty_loss=config.penalty_loss,
                seed=config.seed,
                experiment_name=config.experiment_name,
            ),
            output_path=self.paths.comparison_results_path,
        )

        results = experiment.run()
        self._validate_comparison_results(results)

        self._append_comparison_metadata(config=config, results=results)

        logging.info(
            "Comparison results saved to %s",
            self.paths.comparison_results_path,
        )

        return results

    def reset_results(self) -> None:
        """Delete generated result files without deleting source code."""
        for path in [
            self.paths.training_log_path,
            self.paths.evaluation_log_path,
            self.paths.comparison_results_path,
            self.paths.experiment_summary_path,
        ]:
            if path.exists():
                path.unlink()
                logging.info("Deleted %s", path)

    def print_project_info(self) -> None:
        """Print project runner information and recommended workflow."""
        info = f"""
MENACE MSc AI Project - Command-Line Experiment Runner

Purpose:
  - Train MENACE using matchboxes, beads, rewards, and punishments.
  - Evaluate MENACE without further learning.
  - Compare MENACE against Random, Heuristic, and Minimax opponents.
  - Export CSV evidence for the dissertation Results/Evaluation chapter.
  - Save and load trained MENACE models.

Default paths:
  Model:              {self.paths.model_path}
  Training log:       {self.paths.training_log_path}
  Evaluation log:     {self.paths.evaluation_log_path}
  Comparison results: {self.paths.comparison_results_path}
  Experiment summary: {self.paths.experiment_summary_path}

Recommended reproducible workflow:
  1. python main.py train --games 5000 --seed 42
  2. python main.py evaluate --games 1000 --seed 42 --load-model saved_models/menace_model.json
  3. python main.py compare --trained-games 5000 --evaluation-games 1000 --repetitions 3 --seed 42
  4. Use results/*.csv in the dissertation Results and Evaluation chapter.
"""
        print(info)

    def _prepare_directories(self) -> None:
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.comparison_results_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_or_create_menace(
        self,
        model_path: Optional[Path],
        seed: Optional[int],
    ) -> MENACEPlayer:
        """Load an existing MENACE model or create a seeded new model."""
        if model_path is not None:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logging.info("Loading MENACE model from %s", model_path)
            return ModelPersistence(model_path).load()

        logging.info("Creating a new MENACE model with seed=%s.", seed)
        return MENACEPlayer(mark="O", seed=seed, use_symmetry=True)

    def _save_model(self, menace: MENACEPlayer, model_path: Path) -> None:
        """Save MENACE model with metadata."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        ModelPersistence(model_path).save(
            menace,
            extra_metadata={
                "source": "main.py",
                "saved_for": "dissertation_evidence",
            },
        )

    def _validate_training_history(self, history: pd.DataFrame) -> None:
        """Validate training/evaluation output before saving or reporting."""
        if history.empty:
            raise ValueError("Experiment returned no rows.")

        required_columns = {
            "game",
            "menace_wins",
            "opponent_wins",
            "random_wins",
            "draws",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "matchboxes",
            "total_beads",
            "seed",
        }

        missing = required_columns.difference(history.columns)
        if missing:
            raise ValueError(
                "Experiment result is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for column in ["win_rate", "loss_rate", "draw_rate"]:
            invalid = (history[column] < 0) | (history[column] > 1)
            if invalid.any():
                raise ValueError(f"{column} must be between 0 and 1.")

    def _validate_comparison_results(self, results: pd.DataFrame) -> None:
        """Validate comparison output before it is used as experiment evidence."""
        if results.empty:
            raise ValueError("Comparison experiment returned no rows.")

        required_columns = {
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

        missing = required_columns.difference(results.columns)
        if missing:
            raise ValueError(
                "Comparison result is missing required columns: "
                + ", ".join(sorted(missing))
            )

    def _append_experiment_metadata(
        self,
        mode: str,
        history: pd.DataFrame,
        games: int,
        reward_win: int,
        reward_draw: int,
        penalty_loss: int,
        seed: Optional[int],
        output_path: Path,
    ) -> None:
        """Append one metadata row for traceability and reproducibility."""
        final_row = history.iloc[-1]

        summary_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "games": games,
            "reward_win": reward_win,
            "reward_draw": reward_draw,
            "penalty_loss": penalty_loss,
            "seed": seed,
            "output_file": str(output_path),
            "final_menace_wins": int(final_row["menace_wins"]),
            "final_opponent_wins": int(final_row["opponent_wins"]),
            "final_draws": int(final_row["draws"]),
            "final_win_rate": float(final_row["win_rate"]),
            "final_loss_rate": float(final_row["loss_rate"]),
            "final_draw_rate": float(final_row["draw_rate"]),
            "final_matchboxes": int(final_row["matchboxes"]),
            "final_total_beads": int(final_row["total_beads"]),
        }

        self._append_summary_row(summary_row)

    def _append_comparison_metadata(
        self,
        config: ComparisonRunConfig,
        results: pd.DataFrame,
    ) -> None:
        """Append summary metadata for comparative AI evaluation."""
        evaluation_rows = results[results["phase"] == "evaluation"].copy()

        if evaluation_rows.empty:
            raise ValueError("Comparison results contain no evaluation rows.")

        opponent_rates = (
            evaluation_rows.groupby("opponent_type", as_index=False)["win_rate"]
            .mean()
        )
        best = opponent_rates.sort_values(
            ["win_rate", "opponent_type"],
            ascending=[False, True],
        ).iloc[0]
        worst = opponent_rates.sort_values(
            ["win_rate", "opponent_type"],
            ascending=[True, True],
        ).iloc[0]

        summary_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": "compare",
            "games": config.evaluation_games,
            "trained_games": config.trained_games,
            "repetitions": config.repetitions,
            "reward_win": config.reward_win,
            "reward_draw": config.reward_draw,
            "penalty_loss": config.penalty_loss,
            "seed": config.seed,
            "output_file": str(self.paths.comparison_results_path),
            "best_opponent_type": str(best["opponent_type"]),
            "best_win_rate": float(best["win_rate"]),
            "worst_opponent_type": str(worst["opponent_type"]),
            "worst_win_rate": float(worst["win_rate"]),
            "evaluation_rows": int(len(evaluation_rows)),
        }

        self._append_summary_row(summary_row)

    def _append_summary_row(self, row: dict[str, object]) -> None:
        """Append a row to experiment_summary.csv."""
        new_row = pd.DataFrame([row])

        if self.paths.experiment_summary_path.exists():
            previous = pd.read_csv(self.paths.experiment_summary_path)
            summary = pd.concat([previous, new_row], ignore_index=True)
        else:
            summary = new_row

        summary.to_csv(self.paths.experiment_summary_path, index=False)
        logging.info("Experiment summary updated at %s.", self.paths.experiment_summary_path)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run MENACE training, evaluation, and comparison experiments "
            "for the MSc AI project."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train MENACE against a Random baseline.",
    )
    _add_shared_learning_arguments(train_parser)
    train_parser.add_argument(
        "--save-model",
        type=Path,
        default=ExperimentPaths().model_path,
        help="Path where the trained MENACE model will be saved.",
    )
    train_parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Optional existing MENACE model to continue training.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate MENACE against a Random baseline without learning.",
    )
    evaluate_parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of evaluation games.",
    )
    evaluate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    evaluate_parser.add_argument(
        "--load-model",
        type=Path,
        default=ExperimentPaths().model_path,
        help="Path to an existing MENACE model.",
    )
    evaluate_parser.add_argument(
        "--experiment-name",
        default="menace_cli_evaluation",
        help="Label stored in CSV evidence.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Train MENACE and compare it against Random, Heuristic, and Minimax.",
    )
    compare_parser.add_argument(
        "--trained-games",
        type=int,
        default=5000,
        help="Number of games used to train MENACE before comparison.",
    )
    compare_parser.add_argument(
        "--evaluation-games",
        type=int,
        default=1000,
        help="Number of evaluation games per opponent.",
    )
    compare_parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repeated comparison runs.",
    )
    _add_reward_arguments(compare_parser)
    compare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    compare_parser.add_argument(
        "--experiment-name",
        default="menace_cli_comparison",
        help="Label stored in comparison CSV evidence.",
    )

    subparsers.add_parser(
        "reset-results",
        help="Delete generated CSV result files.",
    )

    subparsers.add_parser(
        "info",
        help="Show project runner information.",
    )

    return parser


def _add_shared_learning_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared training arguments."""
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to run.",
    )
    _add_reward_arguments(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--experiment-name",
        default="menace_cli_training",
        help="Label stored in CSV evidence.",
    )


def _add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach reward/punishment arguments."""
    parser.add_argument(
        "--reward-win",
        type=int,
        default=3,
        help="Bead reward after a MENACE win.",
    )
    parser.add_argument(
        "--reward-draw",
        type=int,
        default=1,
        help="Bead reward after a draw.",
    )
    parser.add_argument(
        "--penalty-loss",
        type=int,
        default=-1,
        help="Bead penalty after a MENACE loss.",
    )


def configure_logging(level: str) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments before execution."""
    if hasattr(args, "games") and args.games <= 0:
        raise ValueError("--games must be greater than zero.")

    if hasattr(args, "trained_games") and args.trained_games <= 0:
        raise ValueError("--trained-games must be greater than zero.")

    if hasattr(args, "evaluation_games") and args.evaluation_games <= 0:
        raise ValueError("--evaluation-games must be greater than zero.")

    if hasattr(args, "repetitions") and args.repetitions <= 0:
        raise ValueError("--repetitions must be greater than zero.")

    if hasattr(args, "reward_win") and args.reward_win < 0:
        raise ValueError("--reward-win must be non-negative.")

    if hasattr(args, "reward_draw") and args.reward_draw < 0:
        raise ValueError("--reward-draw must be non-negative.")

    if hasattr(args, "penalty_loss") and args.penalty_loss > 0:
        raise ValueError("--penalty-loss should be zero or negative.")


def main(
    argv: Optional[list[str]] = None,
    paths: Optional[ExperimentPaths] = None,
) -> int:
    """
    Program entry point.

    Args:
        argv:
            Optional argument list, useful for unit testing.
        paths:
            Optional isolated output paths. The command-line application uses
            the project defaults, while tests can provide temporary paths so
            generated evidence is never deleted or overwritten.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    try:
        validate_args(args)

        runner = MENACEExperimentRunner(paths or ExperimentPaths())

        if args.command == "train":
            runner.train(
                ExperimentConfig(
                    games=args.games,
                    reward_win=args.reward_win,
                    reward_draw=args.reward_draw,
                    penalty_loss=args.penalty_loss,
                    seed=args.seed,
                    save_model=args.save_model,
                    load_model=args.load_model,
                    experiment_name=args.experiment_name,
                )
            )

        elif args.command == "evaluate":
            runner.evaluate(
                ExperimentConfig(
                    games=args.games,
                    seed=args.seed,
                    load_model=args.load_model,
                    experiment_name=args.experiment_name,
                )
            )

        elif args.command == "compare":
            runner.compare(
                ComparisonRunConfig(
                    trained_games=args.trained_games,
                    evaluation_games=args.evaluation_games,
                    repetitions=args.repetitions,
                    reward_win=args.reward_win,
                    reward_draw=args.reward_draw,
                    penalty_loss=args.penalty_loss,
                    seed=args.seed,
                    experiment_name=args.experiment_name,
                )
            )

        elif args.command == "reset-results":
            runner.reset_results()

        elif args.command == "info":
            runner.print_project_info()

        else:
            parser.error(f"Unknown command: {args.command}")

    except Exception as exc:
        logging.error("Execution failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
