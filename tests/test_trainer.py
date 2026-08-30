"""
test_trainer.py

Unit tests for src.trainer.

These tests verify that the MENACE training and evaluation engine:
    - validates configuration
    - runs training games
    - produces valid result DataFrames
    - updates MENACE during training
    - avoids learning during evaluation
    - saves history and summaries
    - supports repeated experiment summaries
    - supports comparative AI evaluation against Random, Heuristic, and Minimax.

Run:
    pytest tests/test_trainer.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.menace import MENACEPlayer
from src.players import HeuristicPlayer, MinimaxPlayer, RandomPlayer
from src.trainer import (
    ComparisonConfig,
    ComparisonExperiment,
    ExperimentSuite,
    OpponentSpec,
    Trainer,
    TrainingConfig,
    TrainingReport,
)


def make_menace(seed: int = 42) -> MENACEPlayer:
    """Create a deterministic MENACE player for tests."""
    return MENACEPlayer(mark="O", seed=seed, use_symmetry=True)


def make_random(seed: int = 123) -> RandomPlayer:
    """Create a deterministic random baseline."""
    return RandomPlayer(mark="X", seed=seed)


def make_heuristic() -> HeuristicPlayer:
    """Create a heuristic opponent."""
    return HeuristicPlayer(mark="X")


def make_minimax() -> MinimaxPlayer:
    """Create a minimax opponent."""
    return MinimaxPlayer(mark="X")


def required_training_columns() -> set[str]:
    return {
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


def test_training_config_accepts_valid_values() -> None:
    config = TrainingConfig(
        games=100,
        reward_win=3,
        reward_draw=1,
        penalty_loss=-1,
        seed=42,
        log_interval=10,
        experiment_name="test_experiment",
        keep_raw_results=False,
    )

    assert config.games == 100
    assert config.reward_win == 3
    assert config.reward_draw == 1
    assert config.penalty_loss == -1
    assert config.seed == 42
    assert config.log_interval == 10
    assert config.experiment_name == "test_experiment"
    assert config.keep_raw_results is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"games": 0},
        {"games": -1},
        {"games": 10, "reward_win": -1},
        {"games": 10, "reward_draw": -1},
        {"games": 10, "penalty_loss": 1},
        {"games": 10, "log_interval": 0},
        {"games": 10, "experiment_name": ""},
        {"games": 10, "experiment_name": "   "},
    ],
)
def test_training_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_comparison_config_accepts_valid_values() -> None:
    config = ComparisonConfig(
        trained_games=50,
        evaluation_games=20,
        repetitions=2,
        reward_win=3,
        reward_draw=1,
        penalty_loss=-1,
        seed=42,
        experiment_name="comparison_test",
    )

    assert config.trained_games == 50
    assert config.evaluation_games == 20
    assert config.repetitions == 2
    assert config.experiment_name == "comparison_test"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trained_games": 0},
        {"evaluation_games": 0},
        {"repetitions": 0},
        {"reward_win": -1},
        {"reward_draw": -1},
        {"penalty_loss": 1},
        {"experiment_name": ""},
        {"experiment_name": "   "},
    ],
)
def test_comparison_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ComparisonConfig(**kwargs)


def test_opponent_spec_accepts_valid_values() -> None:
    spec = OpponentSpec(
        name="Random",
        opponent_type="Random",
        factory=make_random,
    )

    opponent = spec.factory()

    assert spec.name == "Random"
    assert spec.opponent_type == "Random"
    assert opponent.mark == "X"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "opponent_type": "Random", "factory": make_random},
        {"name": "Random", "opponent_type": "", "factory": make_random},
        {"name": "   ", "opponent_type": "Random", "factory": make_random},
        {"name": "Random", "opponent_type": "   ", "factory": make_random},
    ],
)
def test_opponent_spec_rejects_invalid_labels(kwargs) -> None:
    with pytest.raises(ValueError):
        OpponentSpec(**kwargs)


def test_trainer_rejects_players_with_same_mark() -> None:
    menace = MENACEPlayer(mark="O", use_symmetry=True)
    opponent = RandomPlayer(mark="O")

    with pytest.raises(ValueError):
        Trainer(menace_player=menace, opponent=opponent)


def test_trainer_applies_reward_config_to_menace() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(
        games=10,
        reward_win=5,
        reward_draw=2,
        penalty_loss=-3,
    )

    Trainer(menace_player=menace, opponent=opponent, config=config)

    assert menace.reward_win == 5
    assert menace.reward_draw == 2
    assert menace.penalty_loss == -3


def test_training_run_returns_dataframe_with_required_columns() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=20, seed=42, experiment_name="unit_test")

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    history = trainer.run(train=True)

    assert isinstance(history, pd.DataFrame)
    assert required_training_columns().issubset(history.columns)
    assert not history.empty
    assert int(history.iloc[-1]["game"]) == 20
    assert history.iloc[-1]["experiment_name"] == "unit_test"
    assert history.iloc[-1]["mode"] == "train"


def test_training_updates_menace_statistics() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=30, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    trainer.run(train=True)

    assert menace.games_played == 30
    assert menace.wins + menace.losses + menace.draws == 30
    assert len(menace.matchboxes) > 0


def test_training_history_counts_sum_to_games() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=50, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    history = trainer.run(train=True)

    final = history.iloc[-1]
    total = int(final["menace_wins"] + final["opponent_wins"] + final["draws"])

    assert total == 50
    assert int(final["random_wins"]) == int(final["opponent_wins"])


def test_training_rates_are_between_zero_and_one() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=50, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    history = trainer.run(train=True)

    for column in ["win_rate", "loss_rate", "draw_rate"]:
        assert ((history[column] >= 0) & (history[column] <= 1)).all()


def test_log_interval_reduces_number_of_rows() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=100, log_interval=10, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    history = trainer.run(train=True)

    assert len(history) == 10
    assert int(history.iloc[-1]["game"]) == 100


def test_run_with_report_returns_training_report() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=10, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    report = trainer.run_with_report(train=True)

    assert isinstance(report, TrainingReport)
    assert isinstance(report.history, pd.DataFrame)
    assert isinstance(report.final_summary, dict)
    assert len(report.results) == 10
    assert report.final_summary["games"] == 10
    assert report.final_summary["mode"] == "train"
    assert "win_rate" in report.final_summary


def test_run_with_report_can_skip_raw_results() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=10, keep_raw_results=False)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    report = trainer.run_with_report(train=True)

    assert report.results == []
    assert int(report.history.iloc[-1]["game"]) == 10


def test_evaluation_mode_does_not_update_menace_learning_statistics() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=20, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    history = trainer.run(train=False)

    assert menace.games_played == 0
    assert menace.wins == 0
    assert menace.losses == 0
    assert menace.draws == 0
    assert history.iloc[-1]["mode"] == "evaluate"


def test_evaluation_mode_may_create_matchboxes_but_does_not_learn_results() -> None:
    menace = make_menace()
    opponent = make_random()
    config = TrainingConfig(games=20, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    trainer.run(train=False)

    assert menace.games_played == 0
    assert menace.wins + menace.losses + menace.draws == 0


def test_save_history_writes_csv(tmp_path) -> None:
    menace = make_menace()
    opponent = make_random()
    output = tmp_path / "training_log.csv"
    config = TrainingConfig(games=10, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    history = trainer.run(train=True)
    trainer.save_history(history, output)

    assert output.exists()

    loaded = pd.read_csv(output)
    assert not loaded.empty
    assert "win_rate" in loaded.columns
    assert int(loaded.iloc[-1]["game"]) == 10


def test_config_output_path_saves_history_automatically(tmp_path) -> None:
    menace = make_menace()
    opponent = make_random()
    output = tmp_path / "auto_training_log.csv"
    config = TrainingConfig(games=10, seed=42, output_path=output)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    trainer.run(train=True)

    assert output.exists()

    loaded = pd.read_csv(output)
    assert int(loaded.iloc[-1]["game"]) == 10


def test_save_summary_writes_summary_csv(tmp_path) -> None:
    menace = make_menace()
    opponent = make_random()
    output = tmp_path / "summary.csv"
    config = TrainingConfig(games=10, seed=42)

    trainer = Trainer(menace_player=menace, opponent=opponent, config=config)
    report = trainer.run_with_report(train=True)
    trainer.save_summary(report.final_summary, output)

    assert output.exists()

    loaded = pd.read_csv(output)
    assert not loaded.empty
    assert int(loaded.iloc[-1]["games"]) == 10
    assert "win_rate" in loaded.columns


def test_repeated_experiment_suite_returns_summary_dataframe() -> None:
    config = TrainingConfig(games=10, seed=42, experiment_name="suite_test")

    suite = ExperimentSuite(
        menace_factory=lambda: make_menace(seed=42),
        opponent_factory=lambda: make_random(seed=123),
        config=config,
        repetitions=3,
    )

    summaries = suite.run()

    assert isinstance(summaries, pd.DataFrame)
    assert len(summaries) == 3
    assert "repetition" in summaries.columns
    assert "win_rate" in summaries.columns
    assert "draw_rate" in summaries.columns
    assert "loss_rate" in summaries.columns
    assert summaries["experiment_name"].str.contains("suite_test").all()


def test_experiment_suite_rejects_invalid_repetitions() -> None:
    config = TrainingConfig(games=10)

    with pytest.raises(ValueError):
        ExperimentSuite(
            menace_factory=make_menace,
            opponent_factory=make_random,
            config=config,
            repetitions=0,
        )


def test_comparison_experiment_rejects_empty_opponent_list() -> None:
    with pytest.raises(ValueError):
        ComparisonExperiment(
            menace_factory=make_menace,
            training_opponent_factory=make_random,
            evaluation_opponents=[],
            config=ComparisonConfig(trained_games=10, evaluation_games=5),
        )


def test_comparison_experiment_runs_against_multiple_opponents() -> None:
    config = ComparisonConfig(
        trained_games=20,
        evaluation_games=10,
        repetitions=1,
        seed=42,
        experiment_name="comparison_unit_test",
    )

    experiment = ComparisonExperiment(
        menace_factory=make_menace,
        training_opponent_factory=make_random,
        evaluation_opponents=[
            OpponentSpec(
                name="Random",
                opponent_type="Random",
                factory=make_random,
            ),
            OpponentSpec(
                name="Heuristic",
                opponent_type="Heuristic",
                factory=make_heuristic,
            ),
            OpponentSpec(
                name="Minimax",
                opponent_type="Minimax",
                factory=make_minimax,
            ),
        ],
        config=config,
    )

    results = experiment.run()

    assert isinstance(results, pd.DataFrame)
    assert not results.empty

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

    assert required_columns.issubset(results.columns)

    training_rows = results[results["phase"] == "training"]
    evaluation_rows = results[results["phase"] == "evaluation"]

    assert len(training_rows) == 1
    assert len(evaluation_rows) == 3
    assert set(evaluation_rows["opponent_type"]) == {
        "Random",
        "Heuristic",
        "Minimax",
    }
    assert set(evaluation_rows["evaluation_games"]) == {10}
    assert set(evaluation_rows["trained_games"]) == {20}


def test_comparison_experiment_repetitions_create_expected_rows() -> None:
    config = ComparisonConfig(
        trained_games=10,
        evaluation_games=5,
        repetitions=2,
        seed=42,
        experiment_name="comparison_repetition_test",
    )

    experiment = ComparisonExperiment(
        menace_factory=make_menace,
        training_opponent_factory=make_random,
        evaluation_opponents=[
            OpponentSpec(
                name="Random",
                opponent_type="Random",
                factory=make_random,
            ),
            OpponentSpec(
                name="Heuristic",
                opponent_type="Heuristic",
                factory=make_heuristic,
            ),
        ],
        config=config,
    )

    results = experiment.run()

    # For each repetition:
    # 1 training row + 2 evaluation rows = 3 rows.
    assert len(results) == 6
    assert set(results["repetition"]) == {1, 2}
    assert len(results[results["phase"] == "training"]) == 2
    assert len(results[results["phase"] == "evaluation"]) == 4


def test_comparison_experiment_evaluation_does_not_change_trained_game_count() -> None:
    config = ComparisonConfig(
        trained_games=15,
        evaluation_games=5,
        repetitions=1,
        seed=42,
    )

    experiment = ComparisonExperiment(
        menace_factory=make_menace,
        training_opponent_factory=make_random,
        evaluation_opponents=[
            OpponentSpec(
                name="Random",
                opponent_type="Random",
                factory=make_random,
            )
        ],
        config=config,
    )

    results = experiment.run()
    training_row = results[results["phase"] == "training"].iloc[0]
    evaluation_row = results[results["phase"] == "evaluation"].iloc[0]

    assert int(training_row["menace_games_played"]) == 15
    assert int(evaluation_row["menace_games_played"]) == 15


def test_comparison_experiment_save_comparison_writes_csv(tmp_path) -> None:
    config = ComparisonConfig(
        trained_games=10,
        evaluation_games=5,
        repetitions=1,
        seed=42,
    )

    experiment = ComparisonExperiment(
        menace_factory=make_menace,
        training_opponent_factory=make_random,
        evaluation_opponents=[
            OpponentSpec(
                name="Random",
                opponent_type="Random",
                factory=make_random,
            )
        ],
        config=config,
    )

    results = experiment.run()
    output = tmp_path / "comparison_results.csv"

    ComparisonExperiment.save_comparison(results, output)

    assert output.exists()

    loaded = pd.read_csv(output)
    assert not loaded.empty
    assert "opponent_type" in loaded.columns
    assert "phase" in loaded.columns


def test_comparison_experiment_does_not_save_empty_results(tmp_path) -> None:
    output = tmp_path / "empty_comparison.csv"

    with pytest.raises(ValueError):
        ComparisonExperiment.save_comparison(pd.DataFrame(), output)