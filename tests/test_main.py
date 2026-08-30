"""
test_main.py

Unit and integration-style tests for main.py.

These tests verify that the command-line experiment runner supports:

- reproducible MENACE training
- evaluation without learning
- AI comparison experiments
- model saving/loading
- CSV evidence generation
- safe argument validation
- clean exit codes

Run:
    pytest tests/test_main.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import main
from main import (
    ComparisonRunConfig,
    ExperimentConfig,
    ExperimentPaths,
    MENACEExperimentRunner,
    build_parser,
    validate_args,
)


def make_temp_paths(tmp_path: Path) -> ExperimentPaths:
    """Create isolated experiment paths for tests."""
    return ExperimentPaths(
        model_path=tmp_path / "saved_models" / "menace_model.json",
        results_dir=tmp_path / "results",
        training_log_path=tmp_path / "results" / "training_log.csv",
        evaluation_log_path=tmp_path / "results" / "evaluation_log.csv",
        comparison_results_path=tmp_path / "results" / "comparison_results.csv",
        experiment_summary_path=tmp_path / "results" / "experiment_summary.csv",
    )


def test_build_parser_accepts_train_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "--games",
            "10",
            "--seed",
            "42",
            "--reward-win",
            "3",
            "--reward-draw",
            "1",
            "--penalty-loss",
            "-1",
        ]
    )

    assert args.command == "train"
    assert args.games == 10
    assert args.seed == 42
    assert args.reward_win == 3
    assert args.reward_draw == 1
    assert args.penalty_loss == -1


def test_build_parser_accepts_evaluate_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate",
            "--games",
            "10",
            "--seed",
            "42",
            "--load-model",
            "saved_models/menace_model.json",
        ]
    )

    assert args.command == "evaluate"
    assert args.games == 10
    assert args.seed == 42
    assert args.load_model == Path("saved_models/menace_model.json")


def test_build_parser_accepts_compare_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "compare",
            "--trained-games",
            "20",
            "--evaluation-games",
            "5",
            "--repetitions",
            "2",
            "--seed",
            "42",
        ]
    )

    assert args.command == "compare"
    assert args.trained_games == 20
    assert args.evaluation_games == 5
    assert args.repetitions == 2
    assert args.seed == 42


def test_validate_args_rejects_invalid_games() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--games", "0"])

    with pytest.raises(ValueError):
        validate_args(args)


def test_validate_args_rejects_invalid_trained_games() -> None:
    parser = build_parser()
    args = parser.parse_args(["compare", "--trained-games", "0"])

    with pytest.raises(ValueError):
        validate_args(args)


def test_validate_args_rejects_invalid_evaluation_games() -> None:
    parser = build_parser()
    args = parser.parse_args(["compare", "--evaluation-games", "0"])

    with pytest.raises(ValueError):
        validate_args(args)


def test_validate_args_rejects_invalid_repetitions() -> None:
    parser = build_parser()
    args = parser.parse_args(["compare", "--repetitions", "0"])

    with pytest.raises(ValueError):
        validate_args(args)


def test_validate_args_rejects_positive_penalty_loss() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--penalty-loss", "1"])

    with pytest.raises(ValueError):
        validate_args(args)


def test_runner_creates_required_directories(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)

    MENACEExperimentRunner(paths)

    assert paths.results_dir.exists()
    assert paths.model_path.parent.exists()


def test_train_creates_training_log_and_model(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    history = runner.train(
        ExperimentConfig(
            games=5,
            seed=42,
            save_model=paths.model_path,
            experiment_name="unit_train",
        )
    )

    assert not history.empty
    assert paths.training_log_path.exists()
    assert paths.model_path.exists()
    assert paths.experiment_summary_path.exists()

    saved_history = pd.read_csv(paths.training_log_path)

    assert len(saved_history) == 5
    assert {"game", "menace_wins", "opponent_wins", "draws", "win_rate"}.issubset(
        saved_history.columns
    )


def test_evaluate_uses_saved_model_and_creates_evaluation_log(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    runner.train(
        ExperimentConfig(
            games=5,
            seed=42,
            save_model=paths.model_path,
            experiment_name="unit_train",
        )
    )

    results = runner.evaluate(
        ExperimentConfig(
            games=5,
            seed=42,
            load_model=paths.model_path,
            experiment_name="unit_evaluate",
        )
    )

    assert not results.empty
    assert paths.evaluation_log_path.exists()

    saved_results = pd.read_csv(paths.evaluation_log_path)

    assert len(saved_results) == 5
    assert {"game", "win_rate", "loss_rate", "draw_rate"}.issubset(
        saved_results.columns
    )


def test_evaluate_missing_model_raises_error(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    with pytest.raises(FileNotFoundError):
        runner.evaluate(
            ExperimentConfig(
                games=5,
                seed=42,
                load_model=tmp_path / "missing_model.json",
            )
        )


def test_compare_creates_comparison_results(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    results = runner.compare(
        ComparisonRunConfig(
            trained_games=5,
            evaluation_games=2,
            repetitions=1,
            seed=42,
            experiment_name="unit_compare",
        )
    )

    assert not results.empty
    assert paths.comparison_results_path.exists()
    assert paths.experiment_summary_path.exists()

    saved_results = pd.read_csv(paths.comparison_results_path)

    assert not saved_results.empty
    assert {"phase", "opponent_type", "win_rate", "loss_rate", "draw_rate"}.issubset(
        saved_results.columns
    )

    evaluation_rows = saved_results[saved_results["phase"] == "evaluation"]

    assert set(evaluation_rows["opponent_type"]) == {
        "Random",
        "Heuristic",
        "Minimax",
    }


def test_reset_results_deletes_generated_csv_files(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    paths.training_log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.training_log_path.write_text("dummy", encoding="utf-8")
    paths.evaluation_log_path.write_text("dummy", encoding="utf-8")
    paths.comparison_results_path.write_text("dummy", encoding="utf-8")
    paths.experiment_summary_path.write_text("dummy", encoding="utf-8")

    runner.reset_results()

    assert not paths.training_log_path.exists()
    assert not paths.evaluation_log_path.exists()
    assert not paths.comparison_results_path.exists()
    assert not paths.experiment_summary_path.exists()


def test_print_project_info_outputs_expected_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    runner.print_project_info()

    captured = capsys.readouterr()

    assert "MENACE MSc AI Project" in captured.out
    assert "Command-Line Experiment Runner" in captured.out
    assert "compare" in captured.out


def test_main_info_returns_success_exit_code() -> None:
    exit_code = main.main(["info"])

    assert exit_code == 0


def test_main_invalid_train_arguments_returns_failure_exit_code() -> None:
    exit_code = main.main(["train", "--games", "0"])

    assert exit_code == 1


def test_main_reset_results_returns_success_exit_code(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    paths.training_log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.training_log_path.write_text("evidence", encoding="utf-8")

    exit_code = main.main(["reset-results"], paths=paths)

    assert exit_code == 0
    assert not paths.training_log_path.exists()


def test_train_reproducibility_with_same_seed(tmp_path: Path) -> None:
    paths_a = make_temp_paths(tmp_path / "run_a")
    paths_b = make_temp_paths(tmp_path / "run_b")

    runner_a = MENACEExperimentRunner(paths_a)
    runner_b = MENACEExperimentRunner(paths_b)

    history_a = runner_a.train(
        ExperimentConfig(
            games=10,
            seed=42,
            save_model=paths_a.model_path,
            experiment_name="seed_test_a",
        )
    )

    history_b = runner_b.train(
        ExperimentConfig(
            games=10,
            seed=42,
            save_model=paths_b.model_path,
            experiment_name="seed_test_b",
        )
    )

    assert list(history_a["outcome"]) == list(history_b["outcome"])
    assert list(history_a["win_rate"]) == list(history_b["win_rate"])
    assert list(history_a["loss_rate"]) == list(history_b["loss_rate"])
    assert list(history_a["draw_rate"]) == list(history_b["draw_rate"])


def test_training_history_contains_report_evidence_columns(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    history = runner.train(
        ExperimentConfig(
            games=5,
            seed=42,
            save_model=paths.model_path,
        )
    )

    required_columns = {
        "experiment_name",
        "mode",
        "game",
        "outcome",
        "menace_wins",
        "opponent_wins",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "matchboxes",
        "total_beads",
        "seed",
    }

    assert required_columns.issubset(history.columns)


def test_experiment_summary_contains_metadata_after_training(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    runner.train(
        ExperimentConfig(
            games=5,
            seed=42,
            save_model=paths.model_path,
        )
    )

    summary = pd.read_csv(paths.experiment_summary_path)

    assert not summary.empty
    assert summary.iloc[-1]["mode"] == "train"
    assert summary.iloc[-1]["games"] == 5
    assert summary.iloc[-1]["seed"] == 42
    assert "final_win_rate" in summary.columns
    assert "final_matchboxes" in summary.columns
    assert "final_total_beads" in summary.columns


def test_experiment_summary_contains_metadata_after_comparison(tmp_path: Path) -> None:
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)

    runner.compare(
        ComparisonRunConfig(
            trained_games=5,
            evaluation_games=2,
            repetitions=1,
            seed=42,
        )
    )

    summary = pd.read_csv(paths.experiment_summary_path)

    assert not summary.empty
    assert summary.iloc[-1]["mode"] == "compare"
    assert summary.iloc[-1]["trained_games"] == 5
    assert summary.iloc[-1]["repetitions"] == 1
    assert "best_opponent_type" in summary.columns
    assert "worst_opponent_type" in summary.columns


def test_comparison_metadata_ranks_mean_rates_across_repetitions(
    tmp_path: Path,
) -> None:
    """One lucky run must not determine the best opponent in the summary."""
    paths = make_temp_paths(tmp_path)
    runner = MENACEExperimentRunner(paths)
    results = pd.DataFrame(
        [
            {"phase": "evaluation", "opponent_type": "Random", "win_rate": 1.0},
            {"phase": "evaluation", "opponent_type": "Random", "win_rate": 0.0},
            {"phase": "evaluation", "opponent_type": "Heuristic", "win_rate": 0.6},
            {"phase": "evaluation", "opponent_type": "Heuristic", "win_rate": 0.6},
        ]
    )

    runner._append_comparison_metadata(
        ComparisonRunConfig(
            trained_games=10,
            evaluation_games=5,
            repetitions=2,
        ),
        results,
    )

    summary = pd.read_csv(paths.experiment_summary_path).iloc[-1]
    assert summary["best_opponent_type"] == "Heuristic"
    assert summary["best_win_rate"] == pytest.approx(0.6)
    assert summary["worst_opponent_type"] == "Random"
    assert summary["worst_win_rate"] == pytest.approx(0.5)
