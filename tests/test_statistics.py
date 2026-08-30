"""
test_statistics.py

Unit tests for src.statistics.

These tests verify that MENACE statistical analysis works correctly for:

- training summaries
- win/loss/draw rates
- early-vs-late learning comparison
- learning gain
- repeated experiment aggregation
- opponent comparison
- report-ready textual interpretation
- validation of invalid experiment data

Run:
    pytest tests/test_statistics.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.statistics import (
    ComparisonStatistics,
    ExperimentComparator,
    OpponentComparator,
    StatisticsTracker,
    SummaryStatistics,
    compare_summaries,
)


def make_training_history() -> pd.DataFrame:
    """Create valid cumulative MENACE training history."""
    return pd.DataFrame(
        [
            {
                "experiment_name": "unit_training",
                "mode": "train",
                "game": 1,
                "menace_wins": 0,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 0.0,
                "loss_rate": 1.0,
                "draw_rate": 0.0,
                "matchboxes": 1,
                "total_beads": 20,
                "opponent_name": "Random Baseline",
            },
            {
                "experiment_name": "unit_training",
                "mode": "train",
                "game": 2,
                "menace_wins": 1,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 0.5,
                "loss_rate": 0.5,
                "draw_rate": 0.0,
                "matchboxes": 3,
                "total_beads": 35,
                "opponent_name": "Random Baseline",
            },
            {
                "experiment_name": "unit_training",
                "mode": "train",
                "game": 3,
                "menace_wins": 2,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 2 / 3,
                "loss_rate": 1 / 3,
                "draw_rate": 0.0,
                "matchboxes": 5,
                "total_beads": 50,
                "opponent_name": "Random Baseline",
            },
            {
                "experiment_name": "unit_training",
                "mode": "train",
                "game": 4,
                "menace_wins": 3,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 0.75,
                "loss_rate": 0.25,
                "draw_rate": 0.0,
                "matchboxes": 7,
                "total_beads": 70,
                "opponent_name": "Random Baseline",
            },
        ]
    )


def make_history_with_random_wins_only() -> pd.DataFrame:
    """Create history using the supported random_wins column alias."""
    history = make_training_history()
    return history.drop(columns=["opponent_wins"])


def make_repeated_summaries() -> pd.DataFrame:
    """Create repeated experiment summary data."""
    return pd.DataFrame(
        [
            {"win_rate": 0.40, "loss_rate": 0.40, "draw_rate": 0.20},
            {"win_rate": 0.55, "loss_rate": 0.25, "draw_rate": 0.20},
            {"win_rate": 0.60, "loss_rate": 0.20, "draw_rate": 0.20},
        ]
    )


def make_comparison_results() -> pd.DataFrame:
    """Create MENACE comparison results against multiple opponents."""
    return pd.DataFrame(
        [
            {
                "phase": "training",
                "opponent_type": "training_baseline",
                "opponent_name": "Random Baseline",
                "win_rate": 0.55,
                "loss_rate": 0.30,
                "draw_rate": 0.15,
            },
            {
                "phase": "evaluation",
                "opponent_type": "Random",
                "opponent_name": "Random Baseline",
                "win_rate": 0.60,
                "loss_rate": 0.20,
                "draw_rate": 0.20,
            },
            {
                "phase": "evaluation",
                "opponent_type": "Heuristic",
                "opponent_name": "Heuristic AI",
                "win_rate": 0.30,
                "loss_rate": 0.50,
                "draw_rate": 0.20,
            },
            {
                "phase": "evaluation",
                "opponent_type": "Minimax",
                "opponent_name": "Minimax AI",
                "win_rate": 0.00,
                "loss_rate": 0.40,
                "draw_rate": 0.60,
            },
        ]
    )


def test_statistics_tracker_summary_returns_expected_values() -> None:
    tracker = StatisticsTracker(make_training_history())
    summary = tracker.summary()

    assert isinstance(summary, SummaryStatistics)
    assert summary.total_games == 4
    assert summary.menace_wins == 3
    assert summary.opponent_wins == 1
    assert summary.draws == 0
    assert summary.win_rate == 0.75
    assert summary.loss_rate == 0.25
    assert summary.draw_rate == 0.0
    assert summary.final_matchboxes == 7
    assert summary.final_total_beads == 70
    assert summary.opponent_name == "Random Baseline"
    assert summary.experiment_name == "unit_training"
    assert summary.mode == "train"


def test_summary_statistics_to_dict_contains_report_ready_fields() -> None:
    summary = StatisticsTracker(make_training_history()).summary()
    data = summary.to_dict()

    assert data["total_games"] == 4
    assert data["menace_wins"] == 3
    assert data["opponent_wins"] == 1
    assert data["win_rate"] == 0.75
    assert "experiment_name" in data


def test_statistics_tracker_normalises_random_wins_to_opponent_wins() -> None:
    tracker = StatisticsTracker(make_history_with_random_wins_only())
    summary = tracker.summary()

    assert summary.opponent_wins == 1
    assert summary.loss_rate == 0.25


def test_add_moving_averages_adds_expected_columns() -> None:
    tracker = StatisticsTracker(make_training_history())
    output = tracker.add_moving_averages(window=2)

    assert "win_rate_ma" in output.columns
    assert "loss_rate_ma" in output.columns
    assert "draw_rate_ma" in output.columns
    assert len(output) == 4


def test_add_moving_averages_rejects_invalid_window() -> None:
    tracker = StatisticsTracker(make_training_history())

    with pytest.raises(ValueError):
        tracker.add_moving_averages(window=0)


def test_split_early_late_returns_two_phases() -> None:
    tracker = StatisticsTracker(make_training_history())
    phases = tracker.split_early_late(fraction=0.25)

    assert list(phases["phase"]) == ["early", "late"]
    assert "mean_win_rate" in phases.columns
    assert phases.loc[0, "mean_win_rate"] == 0.0
    # The final game was a MENACE win, so the late phase's own
    # win rate is 1/1 rather than the cumulative 3/4 rate.
    assert phases.loc[1, "mean_win_rate"] == 1.0


def test_split_early_late_rejects_invalid_fraction() -> None:
    tracker = StatisticsTracker(make_training_history())

    with pytest.raises(ValueError):
        tracker.split_early_late(fraction=0)

    with pytest.raises(ValueError):
        tracker.split_early_late(fraction=0.75)


def test_learning_gain_returns_expected_improvement() -> None:
    tracker = StatisticsTracker(make_training_history())
    gain = tracker.learning_gain(fraction=0.25)

    assert isinstance(gain, ComparisonStatistics)
    assert gain.baseline_label == "early_training"
    assert gain.comparison_label == "late_training"
    assert gain.baseline_win_rate == 0.0
    assert gain.comparison_win_rate == 1.0
    assert gain.absolute_improvement == 1.0
    assert gain.relative_improvement_percent is None


def test_rate_stability_returns_standard_deviations() -> None:
    tracker = StatisticsTracker(make_training_history())
    stability = tracker.rate_stability()

    assert "std_win_rate" in stability
    assert "std_loss_rate" in stability
    assert "std_draw_rate" in stability
    assert stability["std_win_rate"] >= 0


def test_educational_interpretation_contains_report_language() -> None:
    tracker = StatisticsTracker(make_training_history())
    interpretation = tracker.educational_interpretation()

    assert "MENACE played" in interpretation
    assert "final win rate" in interpretation
    assert "reinforcement" in interpretation


def test_export_summary_frame_returns_one_row_dataframe() -> None:
    tracker = StatisticsTracker(make_training_history())
    frame = tracker.export_summary_frame()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.iloc[0]["total_games"] == 4


def test_export_analysis_frame_contains_gain_and_stability() -> None:
    tracker = StatisticsTracker(make_training_history())
    frame = tracker.export_analysis_frame()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert "absolute_improvement" in frame.columns
    assert "std_win_rate" in frame.columns


def test_statistics_tracker_rejects_empty_history() -> None:
    with pytest.raises(ValueError):
        StatisticsTracker(pd.DataFrame())


def test_statistics_tracker_rejects_missing_required_columns() -> None:
    invalid = pd.DataFrame(
        [
            {
                "game": 1,
                "menace_wins": 1,
            }
        ]
    )

    with pytest.raises(ValueError):
        StatisticsTracker(invalid)


def test_statistics_tracker_rejects_missing_opponent_win_columns() -> None:
    invalid = make_training_history().drop(columns=["opponent_wins", "random_wins"])

    with pytest.raises(ValueError):
        StatisticsTracker(invalid)


def test_statistics_tracker_rejects_non_positive_game_numbers() -> None:
    invalid = make_training_history()
    invalid.loc[0, "game"] = 0

    with pytest.raises(ValueError):
        StatisticsTracker(invalid)


def test_statistics_tracker_rejects_invalid_rates() -> None:
    invalid = make_training_history()
    invalid.loc[0, "win_rate"] = 1.5

    with pytest.raises(ValueError):
        StatisticsTracker(invalid)


def test_experiment_comparator_aggregate_returns_summary() -> None:
    comparator = ExperimentComparator(make_repeated_summaries())
    aggregate = comparator.aggregate()

    assert isinstance(aggregate, pd.DataFrame)
    assert len(aggregate) == 1
    assert aggregate.iloc[0]["runs"] == 3
    assert aggregate.iloc[0]["mean_win_rate"] == pytest.approx((0.40 + 0.55 + 0.60) / 3)


def test_experiment_comparator_rank_by_win_rate_orders_descending() -> None:
    comparator = ExperimentComparator(make_repeated_summaries())
    ranked = comparator.rank_by_win_rate()

    assert list(ranked["win_rate"]) == [0.60, 0.55, 0.40]


def test_experiment_comparator_rejects_empty_summaries() -> None:
    with pytest.raises(ValueError):
        ExperimentComparator(pd.DataFrame())


def test_experiment_comparator_rejects_missing_rate_columns() -> None:
    invalid = pd.DataFrame([{"win_rate": 0.5}])

    with pytest.raises(ValueError):
        ExperimentComparator(invalid)


def test_opponent_comparator_evaluation_rows_filters_training() -> None:
    comparator = OpponentComparator(make_comparison_results())
    rows = comparator.evaluation_rows()

    assert not rows.empty
    assert set(rows["phase"]) == {"evaluation"}
    assert set(rows["opponent_type"]) == {"Random", "Heuristic", "Minimax"}


def test_opponent_comparator_aggregate_by_opponent_returns_expected_rows() -> None:
    comparator = OpponentComparator(make_comparison_results())
    aggregate = comparator.aggregate_by_opponent()

    assert set(aggregate["opponent_type"]) == {"Random", "Heuristic", "Minimax"}
    assert "mean_win_rate" in aggregate.columns
    assert "std_win_rate" in aggregate.columns


def test_opponent_comparator_ranking_orders_by_mean_win_rate() -> None:
    comparator = OpponentComparator(make_comparison_results())
    ranking = comparator.ranking()

    assert list(ranking["opponent_type"]) == ["Random", "Heuristic", "Minimax"]
    assert list(ranking["rank_for_menace"]) == [1, 2, 3]


def test_opponent_comparator_best_and_worst_returns_expected_opponents() -> None:
    comparator = OpponentComparator(make_comparison_results())
    result = comparator.best_and_worst()

    assert result["best_opponent_type"] == "Random"
    assert result["worst_opponent_type"] == "Minimax"
    assert result["best_mean_win_rate"] == 0.60
    assert result["worst_mean_win_rate"] == 0.00


def test_opponent_comparator_educational_interpretation_mentions_ai_types() -> None:
    comparator = OpponentComparator(make_comparison_results())
    interpretation = comparator.educational_interpretation()

    assert "Random" in interpretation
    assert "Heuristic" in interpretation
    assert "Minimax" in interpretation
    assert "reinforcement learning" in interpretation
    assert "search-based AI" in interpretation


def test_opponent_comparator_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        OpponentComparator(pd.DataFrame())


def test_opponent_comparator_rejects_missing_columns() -> None:
    invalid = pd.DataFrame(
        [
            {
                "opponent_type": "Random",
                "win_rate": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError):
        OpponentComparator(invalid)


def test_opponent_comparator_rejects_results_without_evaluation_rows() -> None:
    invalid = pd.DataFrame(
        [
            {
                "phase": "training",
                "opponent_type": "training_baseline",
                "opponent_name": "Random Baseline",
                "win_rate": 0.5,
                "loss_rate": 0.3,
                "draw_rate": 0.2,
            }
        ]
    )

    comparator = OpponentComparator(invalid)

    with pytest.raises(ValueError):
        comparator.evaluation_rows()


def test_compare_summaries_returns_absolute_and_relative_improvement() -> None:
    baseline = SummaryStatistics(
        total_games=100,
        menace_wins=40,
        opponent_wins=40,
        draws=20,
        win_rate=0.40,
        loss_rate=0.40,
        draw_rate=0.20,
    )
    comparison = SummaryStatistics(
        total_games=100,
        menace_wins=60,
        opponent_wins=20,
        draws=20,
        win_rate=0.60,
        loss_rate=0.20,
        draw_rate=0.20,
    )

    result = compare_summaries(
        baseline=baseline,
        comparison=comparison,
        baseline_label="before_training",
        comparison_label="after_training",
    )

    assert isinstance(result, ComparisonStatistics)
    assert result.baseline_label == "before_training"
    assert result.comparison_label == "after_training"
    assert result.absolute_improvement == pytest.approx(0.20)
    assert result.relative_improvement_percent == pytest.approx(50.0)


def test_compare_summaries_handles_zero_baseline() -> None:
    baseline = SummaryStatistics(
        total_games=100,
        menace_wins=0,
        opponent_wins=80,
        draws=20,
        win_rate=0.0,
        loss_rate=0.8,
        draw_rate=0.2,
    )
    comparison = SummaryStatistics(
        total_games=100,
        menace_wins=20,
        opponent_wins=60,
        draws=20,
        win_rate=0.2,
        loss_rate=0.6,
        draw_rate=0.2,
    )

    result = compare_summaries(baseline, comparison)

    assert result.absolute_improvement == pytest.approx(0.2)
    assert result.relative_improvement_percent is None
