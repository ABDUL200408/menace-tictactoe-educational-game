"""
test_visualisation.py

Unit tests for src.visualisation.

These tests verify that the MENACE visualisation layer can produce reliable
figures, learner-facing explanations, and exportable visual evidence for the
application and project reporting.

Run:
    pytest tests/test_visualisation.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.menace import Matchbox
from src.visualisation import (
    VisualisationConfig,
    VisualisationError,
    Visualiser,
    build_comparison_textual_summary,
    build_textual_visual_summary,
)


def make_training_history() -> pd.DataFrame:
    """Create valid MENACE training history for visualisation tests."""
    return pd.DataFrame(
        [
            {
                "game": 1,
                "menace_wins": 0,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 0.0,
                "loss_rate": 1.0,
                "draw_rate": 0.0,
                "matchboxes": 1,
                "total_beads": 24,
                "opponent_name": "Random Baseline",
            },
            {
                "game": 2,
                "menace_wins": 1,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 0.5,
                "loss_rate": 0.5,
                "draw_rate": 0.0,
                "matchboxes": 3,
                "total_beads": 40,
                "opponent_name": "Random Baseline",
            },
            {
                "game": 3,
                "menace_wins": 2,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 0,
                "win_rate": 2 / 3,
                "loss_rate": 1 / 3,
                "draw_rate": 0.0,
                "matchboxes": 5,
                "total_beads": 55,
                "opponent_name": "Random Baseline",
            },
            {
                "game": 4,
                "menace_wins": 2,
                "opponent_wins": 1,
                "random_wins": 1,
                "draws": 1,
                "win_rate": 0.5,
                "loss_rate": 0.25,
                "draw_rate": 0.25,
                "matchboxes": 6,
                "total_beads": 62,
                "opponent_name": "Random Baseline",
            },
        ]
    )


def make_comparison_results() -> pd.DataFrame:
    """Create valid MENACE comparison results for visualisation tests."""
    return pd.DataFrame(
        [
            {
                "phase": "training",
                "repetition": 1,
                "opponent_type": "training_baseline",
                "opponent_name": "Random Baseline",
                "games": 100,
                "menace_wins": 55,
                "opponent_wins": 30,
                "draws": 15,
                "win_rate": 0.55,
                "loss_rate": 0.30,
                "draw_rate": 0.15,
                "matchboxes": 50,
                "total_beads": 400,
            },
            {
                "phase": "evaluation",
                "repetition": 1,
                "opponent_type": "Random",
                "opponent_name": "Random Baseline",
                "games": 50,
                "menace_wins": 30,
                "opponent_wins": 10,
                "draws": 10,
                "win_rate": 0.60,
                "loss_rate": 0.20,
                "draw_rate": 0.20,
                "matchboxes": 50,
                "total_beads": 400,
            },
            {
                "phase": "evaluation",
                "repetition": 1,
                "opponent_type": "Heuristic",
                "opponent_name": "Heuristic AI",
                "games": 50,
                "menace_wins": 15,
                "opponent_wins": 25,
                "draws": 10,
                "win_rate": 0.30,
                "loss_rate": 0.50,
                "draw_rate": 0.20,
                "matchboxes": 50,
                "total_beads": 400,
            },
            {
                "phase": "evaluation",
                "repetition": 1,
                "opponent_type": "Minimax",
                "opponent_name": "Minimax AI",
                "games": 50,
                "menace_wins": 0,
                "opponent_wins": 20,
                "draws": 30,
                "win_rate": 0.00,
                "loss_rate": 0.40,
                "draw_rate": 0.60,
                "matchboxes": 50,
                "total_beads": 400,
            },
        ]
    )


def make_matchbox() -> Matchbox:
    """Create a valid matchbox for bead/probability visualisation tests."""
    return Matchbox(
        state="X O      ",
        beads={1: 3, 3: 6, 4: 9},
        minimum_beads=1,
    )


def test_visualiser_initialises_with_default_config() -> None:
    visualiser = Visualiser()

    assert visualiser.config.title_prefix == "MENACE Training"
    assert visualiser.config.moving_average_window == 100


def test_visualiser_initialises_with_custom_config(tmp_path: Path) -> None:
    config = VisualisationConfig(
        title_prefix="Custom MENACE",
        moving_average_window=2,
        figure_output_dir=tmp_path,
    )

    visualiser = Visualiser(config)

    assert visualiser.config.title_prefix == "Custom MENACE"
    assert visualiser.config.moving_average_window == 2
    assert visualiser.config.figure_output_dir == tmp_path


def test_learning_curve_returns_plotly_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.learning_curve(make_training_history())

    assert fig is not None
    assert "Learning Curve" in fig.layout.title.text
    assert fig.layout.xaxis.range[1] == make_training_history()["game"].max()
    assert fig.layout.showlegend is not False
    assert {trace.name for trace in fig.data} == {
        "MENACE win rate", "Opponent win rate", "Draw rate"
    }


def test_rolling_learning_curve_returns_plotly_figure() -> None:
    visualiser = Visualiser(VisualisationConfig(moving_average_window=2))
    fig = visualiser.rolling_learning_curve(make_training_history())

    assert fig is not None
    assert "Rolling Average" in fig.layout.title.text
    assert fig.layout.xaxis.range[1] == make_training_history()["game"].max()
    assert {trace.name for trace in fig.data} == {
        "MENACE win rate", "Opponent win rate", "Draw rate"
    }


def test_outcome_counts_returns_plotly_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.outcome_counts(make_training_history())

    assert fig is not None
    assert "Final Outcome Counts" in fig.layout.title.text


def test_outcome_rates_returns_plotly_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.outcome_rates(make_training_history())

    assert fig is not None
    assert "Final Outcome Rates" in fig.layout.title.text


def test_early_late_training_comparison_uses_phase_outcomes() -> None:
    """Early/late bars must represent each phase's own game outcomes."""
    visualiser = Visualiser()
    fig = visualiser.early_late_training_comparison(
        make_training_history(),
        fraction=0.25,
    )

    assert fig is not None
    assert "Early vs Late" in fig.layout.title.text

    traces = {trace.name: list(trace.y) for trace in fig.data}
    assert traces["Mean Win Rate"] == pytest.approx([0.0, 0.0])
    assert traces["Mean Loss Rate"] == pytest.approx([1.0, 0.0])
    assert traces["Mean Draw Rate"] == pytest.approx([0.0, 1.0])


def test_early_late_training_comparison_rejects_invalid_fraction() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError):
        visualiser.early_late_training_comparison(
            make_training_history(),
            fraction=0,
        )

    with pytest.raises(ValueError):
        visualiser.early_late_training_comparison(
            make_training_history(),
            fraction=0.75,
        )


def test_training_dashboard_figures_contains_expected_keys() -> None:
    visualiser = Visualiser()
    figures = visualiser.training_dashboard_figures(make_training_history())

    assert set(figures.keys()) == {
        "learning_curve",
        "rolling_learning_curve",
        "outcome_counts",
        "outcome_rates",
        "early_late_comparison",
        "matchbox_growth",
    }


def test_comparison_rates_returns_plotly_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.comparison_rates(make_comparison_results())

    assert fig is not None
    assert "MENACE Performance Against Different Opponents" in fig.layout.title.text


def test_comparison_win_rate_returns_plotly_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.comparison_win_rate(make_comparison_results())

    assert fig is not None
    assert "MENACE Win Rate by Opponent" in fig.layout.title.text


def test_comparison_loss_rate_returns_plotly_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.comparison_loss_rate(make_comparison_results())

    assert fig is not None
    assert "MENACE Loss Rate by Opponent" in fig.layout.title.text


def test_comparison_summary_table_returns_evaluation_rows_only() -> None:
    visualiser = Visualiser()
    table = visualiser.comparison_summary_table(make_comparison_results())

    assert not table.empty
    assert set(table["opponent_type"]) == {"Random", "Heuristic", "Minimax"}


def test_comparison_dashboard_figures_contains_expected_keys() -> None:
    visualiser = Visualiser()
    figures = visualiser.comparison_dashboard_figures(make_comparison_results())

    assert set(figures.keys()) == {
        "comparison_rates",
        "comparison_win_rate",
        "comparison_loss_rate",
    }


def test_repeated_experiment_boxplot_returns_figure() -> None:
    visualiser = Visualiser()
    summaries = pd.DataFrame(
        [
            {"win_rate": 0.40, "loss_rate": 0.40, "draw_rate": 0.20},
            {"win_rate": 0.55, "loss_rate": 0.25, "draw_rate": 0.20},
            {"win_rate": 0.60, "loss_rate": 0.20, "draw_rate": 0.20},
        ]
    )

    fig = visualiser.repeated_experiment_boxplot(summaries)

    assert fig is not None
    assert "Variation Across Repeated MENACE Experiments" in fig.layout.title.text


def test_repeated_comparison_boxplot_returns_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.repeated_comparison_boxplot(make_comparison_results())

    assert fig is not None
    assert "Repeated Comparison" in fig.layout.title.text


def test_matchbox_beads_returns_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.matchbox_beads(make_matchbox())

    assert fig is not None
    assert "Matchbox Bead Counts" in fig.layout.title.text


def test_matchbox_probabilities_returns_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.matchbox_probabilities(make_matchbox())

    assert fig is not None
    assert "Move Probabilities" in fig.layout.title.text


def test_matchbox_board_heatmap_returns_figure() -> None:
    visualiser = Visualiser()
    fig = visualiser.matchbox_board_heatmap(make_matchbox())

    assert fig is not None
    assert "Matchbox Bead Heatmap" in fig.layout.title.text


def test_move_frequency_heatmap_returns_figure() -> None:
    visualiser = Visualiser()
    move_rows = pd.DataFrame(
        [
            {"move": 0},
            {"move": 0},
            {"move": 4},
            {"move": 8},
        ]
    )

    fig = visualiser.move_frequency_heatmap(move_rows)

    assert fig is not None
    assert "Move Frequency Heatmap" in fig.layout.title.text


def test_move_frequency_heatmap_rejects_missing_move_column() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError):
        visualiser.move_frequency_heatmap(pd.DataFrame([{"cell": 0}]))


def test_move_frequency_heatmap_rejects_empty_dataframe() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError):
        visualiser.move_frequency_heatmap(pd.DataFrame())


def test_training_text_summary_contains_interpretation() -> None:
    visualiser = Visualiser()
    summary = visualiser.training_text_summary(make_training_history())

    assert "MENACE completed" in summary
    assert "final win rate" in summary
    assert "interpreted critically" in summary


def test_comparison_text_summary_contains_opponent_names() -> None:
    visualiser = Visualiser()
    summary = visualiser.comparison_text_summary(make_comparison_results())

    assert "Random" in summary
    assert "Minimax" in summary
    assert "search-based AI" in summary


def test_training_summary_helper_returns_expected_text() -> None:
    summary = build_textual_visual_summary(make_training_history())

    assert "MENACE completed" in summary
    assert "final win rate" in summary


def test_comparison_summary_helper_returns_expected_text() -> None:
    summary = build_comparison_textual_summary(make_comparison_results())

    assert "comparison experiment" in summary.lower()
    assert "opponents" in summary.lower()


def test_export_figure_html_creates_file(tmp_path: Path) -> None:
    visualiser = Visualiser(
        VisualisationConfig(
            figure_output_dir=tmp_path,
        )
    )

    fig = visualiser.learning_curve(make_training_history())
    output_path = visualiser.export_figure_html(fig, "learning_curve_test")

    assert output_path.exists()
    assert output_path.suffix == ".html"
    assert "learning_curve_test" in output_path.name


def test_export_figures_html_creates_multiple_files(tmp_path: Path) -> None:
    visualiser = Visualiser(
        VisualisationConfig(
            figure_output_dir=tmp_path,
        )
    )

    figures = visualiser.training_dashboard_figures(make_training_history())
    exported = visualiser.export_figures_html(figures)

    assert set(exported.keys()) == set(figures.keys())
    assert all(path.exists() for path in exported.values())
    assert all(path.suffix == ".html" for path in exported.values())


def test_export_training_dashboard_html_creates_report_evidence(tmp_path: Path) -> None:
    visualiser = Visualiser(
        VisualisationConfig(
            figure_output_dir=tmp_path,
        )
    )

    exported = visualiser.export_training_dashboard_html(make_training_history())

    assert "learning_curve" in exported
    assert "early_late_comparison" in exported
    assert all(path.exists() for path in exported.values())


def test_export_comparison_dashboard_html_creates_report_evidence(tmp_path: Path) -> None:
    visualiser = Visualiser(
        VisualisationConfig(
            figure_output_dir=tmp_path,
        )
    )

    exported = visualiser.export_comparison_dashboard_html(make_comparison_results())

    assert "comparison_rates" in exported
    assert "comparison_win_rate" in exported
    assert all(path.exists() for path in exported.values())


def test_export_figure_rejects_empty_filename(tmp_path: Path) -> None:
    visualiser = Visualiser(
        VisualisationConfig(
            figure_output_dir=tmp_path,
        )
    )

    fig = visualiser.learning_curve(make_training_history())

    with pytest.raises(ValueError):
        visualiser.export_figure_html(fig, "   ")


def test_visualiser_rejects_empty_training_history() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError):
        visualiser.learning_curve(pd.DataFrame())


def test_visualiser_rejects_missing_training_columns() -> None:
    visualiser = Visualiser()
    invalid = pd.DataFrame(
        [
            {
                "game": 1,
                "menace_wins": 1,
            }
        ]
    )

    with pytest.raises(ValueError):
        visualiser.learning_curve(invalid)


def test_visualiser_rejects_invalid_rate_values() -> None:
    visualiser = Visualiser()
    invalid = make_training_history()
    invalid.loc[0, "win_rate"] = 1.5

    with pytest.raises(ValueError):
        visualiser.learning_curve(invalid)


def test_visualiser_rejects_invalid_outcome_totals() -> None:
    visualiser = Visualiser()
    invalid = make_training_history()
    invalid.loc[0, "menace_wins"] = 99

    with pytest.raises(ValueError):
        visualiser.learning_curve(invalid)


def test_visualiser_rejects_empty_comparison_results() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError):
        visualiser.comparison_rates(pd.DataFrame())


def test_visualiser_rejects_missing_comparison_columns() -> None:
    visualiser = Visualiser()
    invalid = pd.DataFrame(
        [
            {
                "phase": "evaluation",
                "opponent_type": "Random",
                "win_rate": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError):
        visualiser.comparison_rates(invalid)


def test_visualiser_rejects_invalid_matchbox_type() -> None:
    visualiser = Visualiser()

    with pytest.raises(TypeError):
        visualiser.matchbox_beads("not a matchbox")  # type: ignore[arg-type]


def test_visualiser_rejects_matchbox_without_beads() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError):
        visualiser.matchbox_beads(
            Matchbox(
                state="test",
                beads={},
            )
        )

# ============================================================
# Educational matchbox visualisation tests
# ============================================================


def make_equal_matchbox() -> Matchbox:
    """Create a matchbox where all legal moves have equal bead counts."""
    return Matchbox(
        state="X        ",
        beads={1: 3, 2: 3, 4: 3},
        minimum_beads=1,
    )


def make_large_bead_matchbox() -> Matchbox:
    """Create a matchbox containing a large bead count for overflow testing."""
    return Matchbox(
        state="XO       ",
        beads={2: 3, 4: 25, 8: 100},
        minimum_beads=1,
    )


def test_educational_matchbox_rows_map_moves_to_visible_squares() -> None:
    """Matchbox moves should be translated to learner-facing board labels."""
    visualiser = Visualiser()
    matchbox = make_matchbox()
    mapping = {1: 8, 3: 4, 4: 0}

    rows = visualiser.educational_matchbox_rows(
        matchbox,
        move_to_board_square=mapping,
        selected_move=3,
    )

    assert [row.matchbox_move for row in rows] == [1, 3, 4]
    assert [row.board_square for row in rows] == [8, 4, 0]
    assert [row.square_label for row in rows] == ["I", "E", "A"]
    assert [row.beads for row in rows] == [3, 6, 9]
    assert sum(row.probability for row in rows) == pytest.approx(1.0)
    assert [row.selected for row in rows] == [False, True, False]


def test_educational_matchbox_rows_use_identity_mapping_by_default() -> None:
    """Without a mapping, matchbox move indexes should map to the same squares."""
    visualiser = Visualiser()
    rows = visualiser.educational_matchbox_rows(
        make_matchbox(),
        selected_move=4,
    )

    assert [row.board_square for row in rows] == [1, 3, 4]
    assert [row.square_label for row in rows] == ["B", "D", "E"]
    assert rows[-1].selected is True


def test_educational_matchbox_rows_support_custom_square_labels() -> None:
    """The learner-facing labels should be configurable."""
    visualiser = Visualiser()
    custom_labels = ("1", "2", "3", "4", "5", "6", "7", "8", "9")

    rows = visualiser.educational_matchbox_rows(
        make_matchbox(),
        square_labels=custom_labels,
    )

    assert [row.square_label for row in rows] == ["2", "4", "5"]


def test_educational_matchbox_rows_reject_invalid_label_count() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError, match="nine"):
        visualiser.educational_matchbox_rows(
            make_matchbox(),
            square_labels=("A", "B"),
        )


def test_educational_matchbox_rows_reject_out_of_range_mapping() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError, match="between 0 and 8"):
        visualiser.educational_matchbox_rows(
            make_matchbox(),
            move_to_board_square={1: 9, 3: 4, 4: 0},
        )


def test_educational_matchbox_table_has_learner_friendly_columns() -> None:
    """The simple table should avoid internal matchbox terminology."""
    visualiser = Visualiser()
    table = visualiser.educational_matchbox_table(
        make_matchbox(),
        move_to_board_square={1: 8, 3: 4, 4: 0},
        selected_move=3,
    )

    assert list(table.columns) == ["Square", "Beads", "Chance", "Chosen"]
    assert table["Square"].tolist() == ["I", "E", "A"]
    assert table["Beads"].tolist() == [3, 6, 9]
    assert table["Chance"].tolist() == ["16.7%", "33.3%", "50.0%"]
    assert table["Chosen"].tolist() == ["", "Yes", ""]


def test_educational_matchbox_html_contains_matchbox_and_bead_circles() -> None:
    """The HTML should look like a matchbox and include one circle per bead."""
    visualiser = Visualiser(
        VisualisationConfig(max_visible_beads_per_move=20)
    )

    html = visualiser.educational_matchbox_html(
        make_equal_matchbox(),
        selected_move=2,
    )

    assert "menace-matchbox" in html
    assert "Look inside MENACE&#x27;s matchbox" in html
    assert html.count('class="menace-bead"') == 9
    assert "MENACE chose this square" in html
    assert "Square C" in html


def test_educational_matchbox_html_caps_large_bead_displays() -> None:
    """Large bead counts should not create hundreds of circles in the page."""
    visualiser = Visualiser(
        VisualisationConfig(max_visible_beads_per_move=10)
    )

    html = visualiser.educational_matchbox_html(
        make_large_bead_matchbox(),
        selected_move=8,
    )

    # Three moves are shown, but each is capped at ten visible circles.
    assert html.count('class="menace-bead"') == 23
    assert "+15" in html
    assert "+90" in html
    assert "100 bead(s)" in html
    assert "MENACE chose this square" in html


def test_educational_matchbox_html_escapes_custom_title() -> None:
    """Learner-facing HTML must escape user-supplied text."""
    visualiser = Visualiser()

    html = visualiser.educational_matchbox_html(
        make_equal_matchbox(),
        title="<script>alert('x')</script>",
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_educational_decision_summary_explains_equal_probabilities() -> None:
    """Equal bead counts should be explained as equal chances."""
    visualiser = Visualiser()

    summary = visualiser.educational_decision_summary(
        make_equal_matchbox(),
        selected_move=2,
    )

    assert "MENACE chose square C" in summary
    assert "3 bead(s)" in summary
    assert "33.3% chance" in summary
    assert "same number of beads" in summary
    assert "same chance" in summary


def test_educational_decision_summary_explains_learned_probabilities() -> None:
    """Unequal bead counts should be linked to earlier learning."""
    visualiser = Visualiser()

    summary = visualiser.educational_decision_summary(
        make_matchbox(),
        selected_move=4,
        move_to_board_square={1: 8, 3: 4, 4: 0},
    )

    assert "MENACE chose square A" in summary
    assert "9 bead(s)" in summary
    assert "50.0% chance" in summary
    assert "Past games changed the bead numbers" in summary
    assert "more likely than others" in summary


def test_educational_decision_summary_rejects_unknown_selected_move() -> None:
    visualiser = Visualiser()

    with pytest.raises(ValueError, match="selected_move"):
        visualiser.educational_decision_summary(
            make_matchbox(),
            selected_move=8,
        )


def test_matchbox_rejects_empty_bead_collection_before_visualisation() -> None:
    """Invalid empty matchboxes are rejected before reaching the visual layer."""
    with pytest.raises(ValueError, match="at least one move"):
        Matchbox(state="test", beads={})


def test_visualisation_config_rejects_invalid_visible_bead_limit() -> None:
    with pytest.raises(ValueError, match="max_visible_beads_per_move"):
        VisualisationConfig(max_visible_beads_per_move=0)


def test_visualisation_config_rejects_invalid_square_labels() -> None:
    with pytest.raises(ValueError, match="square_labels"):
        VisualisationConfig(square_labels=("A", "B"))

    with pytest.raises(ValueError, match="square_labels"):
        VisualisationConfig(
            square_labels=("A", "B", "C", "D", "E", "F", "G", "H", "")
        )


def test_export_training_evidence_returns_ordered_html_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Streamlit export adapter must return the list expected by app.py."""
    visualiser = Visualiser(
        VisualisationConfig(figure_output_dir=tmp_path)
    )
    exported = {
        "learning_curve": tmp_path / "learning_curve.html",
        "outcome_rates": tmp_path / "outcome_rates.html",
    }
    monkeypatch.setattr(
        visualiser,
        "export_training_dashboard_html",
        lambda history: exported,
    )

    paths = visualiser.export_training_evidence(make_training_history())

    assert paths == list(exported.values())
