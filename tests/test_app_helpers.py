"""
test_app_helpers.py

Focused tests for learner-facing helper behaviour in app.py.

These tests verify the final learner-facing interface behaviour:
    - A–I square labels while preserving internal indexes 0–8;
    - the single learner-friendly display mode;
    - plain-language reward and move explanations;
    - safe session-state defaults;
    - helper behaviour used by the Streamlit interface.

Run:
    pytest tests/test_app_helpers.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import streamlit as st

from app import AppConfig, MENACEStreamlitApp, SessionKey
from src.board import Board


@pytest.fixture()
def app() -> MENACEStreamlitApp:
    """Create the Streamlit controller without running the interface."""
    return MENACEStreamlitApp(AppConfig())


@pytest.fixture(autouse=True)
def clean_session_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Replace Streamlit session state with a normal dictionary.

    This keeps helper tests independent from an active Streamlit server.
    """
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


# ---------------------------------------------------------------------------
# Configuration and square-label mapping
# ---------------------------------------------------------------------------


def test_app_config_uses_simple_mode_by_default() -> None:
    config = AppConfig()

    assert config.default_display_mode == "Simple"
    assert config.square_labels == (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
    )


def test_app_config_accepts_the_central_project_config() -> None:
    """The entry point must be able to pass DEFAULT_CONFIG explicitly."""
    from src.config import DEFAULT_CONFIG

    config = AppConfig(DEFAULT_CONFIG)

    assert config.project_config is DEFAULT_CONFIG


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, "A"),
        (1, "B"),
        (2, "C"),
        (3, "D"),
        (4, "E"),
        (5, "F"),
        (6, "G"),
        (7, "H"),
        (8, "I"),
    ],
)
def test_square_label_maps_internal_indexes_to_a_to_i(
    app: MENACEStreamlitApp,
    index: int,
    expected: str,
) -> None:
    assert app._square_label(index) == expected


@pytest.mark.parametrize("invalid_index", [-1, 9, 99])
def test_square_label_rejects_invalid_indexes(
    app: MENACEStreamlitApp,
    invalid_index: int,
) -> None:
    with pytest.raises((ValueError, IndexError)):
        app._square_label(invalid_index)


@pytest.mark.parametrize("invalid_index", [True, 1.5, "1", None])
def test_square_label_rejects_non_integer_indexes(
    app: MENACEStreamlitApp,
    invalid_index: Any,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        app._square_label(invalid_index)  # type: ignore[arg-type]


def test_move_label_returns_question_mark_for_missing_move(
    app: MENACEStreamlitApp,
) -> None:
    assert app._move_label(None) == "?"


def test_move_label_uses_learner_facing_letter(
    app: MENACEStreamlitApp,
) -> None:
    assert app._move_label(7) == "H"


def test_internal_board_indexes_are_not_changed_by_display_labels(
    app: MENACEStreamlitApp,
) -> None:
    """
    A–I are presentation labels only; Board still receives integer indexes.
    """
    board = Board.empty()
    board.make_move(7, "X")

    assert app._square_label(7) == "H"
    assert board.cells[7] == "X"
    assert board.cells[0] == Board.EMPTY


# ---------------------------------------------------------------------------
# Single learner-friendly display mode
# ---------------------------------------------------------------------------


def test_is_simple_mode_uses_config_default_when_state_is_missing(
    app: MENACEStreamlitApp,
) -> None:
    assert app._is_simple_mode() is True


def test_is_simple_mode_returns_true_for_simple_mode(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Simple"

    assert app._is_simple_mode() is True


def test_technical_session_value_cannot_enable_a_second_interface(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Technical"

    assert app._is_simple_mode() is True


def test_training_results_button_targets_results_page(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    """The training call-to-action must navigate without changing learning."""
    menace_before = clean_session_state.get(SessionKey.MENACE)

    app._open_results_from_training()

    assert (
        clean_session_state[SessionKey.ACTIVE_PAGE]
        == app.config.simple_page_labels[3]
    )
    assert clean_session_state[SessionKey.REQUESTED_PAGE] is None
    assert clean_session_state.get(SessionKey.MENACE) is menace_before


def test_training_page_names_random_as_the_only_training_opponent() -> None:
    """Learners must not mistake comparison opponents for training opponents."""
    from pathlib import Path

    source = (
        Path(__file__).parents[1] / "ui" / "pages" / "training.py"
    ).read_text(encoding="utf-8")

    assert "**The Guesser (Random Player)**" in source
    assert "Heuristic and Minimax" in source
    assert "evaluation opponents" in source
    assert "training_go_to_results" in source


def test_session_keys_for_educational_interface_are_unique() -> None:
    values = [
        SessionKey.DISPLAY_MODE,
        SessionKey.SHOW_WELCOME,
        SessionKey.MESSAGE,
        SessionKey.EXPLANATION,
        SessionKey.REWARD_FEEDBACK,
    ]

    assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Plain-language reward feedback
# ---------------------------------------------------------------------------


def make_learning_rows() -> list[dict[str, Any]]:
    """Return representative before/after bead changes."""
    return [
        {
            "state": "X        ",
            "canonical_move": 5,
            "original_move": 3,
            "before": 3,
            "after": 6,
            "change": 3,
        },
        {
            "state": " O XX    ",
            "canonical_move": 0,
            "original_move": 8,
            "before": 3,
            "after": 6,
            "change": 3,
        },
    ]


@pytest.mark.parametrize(
    ("result", "expected_phrase"),
    [
        ("win", "MENACE won"),
        ("loss", "MENACE lost"),
        ("draw", "The game was a draw"),
    ],
)
def test_simple_reward_feedback_uses_child_friendly_language(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
    result: str,
    expected_phrase: str,
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Simple"

    feedback = app._build_reward_feedback(result, make_learning_rows())

    assert expected_phrase in feedback
    assert "Square D" in feedback
    assert "Square I" in feedback
    assert "3 bead(s) → 6 bead(s)" in feedback
    assert "canonical" not in feedback.lower()
    assert "original move" not in feedback.lower()
    assert "matchbox move" not in feedback.lower()


def test_simple_reward_feedback_handles_no_recorded_decisions(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Simple"

    feedback = app._build_reward_feedback("loss", [])

    assert "no beads to change" in feedback.lower()
    assert "Game result for MENACE" not in feedback


def test_single_learner_view_keeps_clear_reward_evidence(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Technical"

    feedback = app._build_reward_feedback("win", make_learning_rows())

    assert "MENACE won" in feedback
    assert "Square D: 3 bead(s) → 6 bead(s)" in feedback
    assert "Square I: 3 bead(s) → 6 bead(s)" in feedback
    assert "more likely to be chosen" in feedback


# ---------------------------------------------------------------------------
# Decision-row handling and mapping
# ---------------------------------------------------------------------------


class DecisionWithRows:
    """Small test double exposing the preferred decision-row method."""

    def matchbox_probability_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "matchbox_move": 2,
                "original_board_square": 6,
                "beads": 3,
                "probability": 1.0,
                "selected": True,
            }
        ]


def test_decision_probability_rows_uses_public_decision_method(
    app: MENACEStreamlitApp,
) -> None:
    rows = app._decision_probability_rows(DecisionWithRows())

    assert rows == [
        {
            "matchbox_move": 2,
            "original_board_square": 6,
            "beads": 3,
            "probability": 1.0,
            "selected": True,
        }
    ]


def test_decision_probability_rows_builds_fallback_mapping(
    app: MENACEStreamlitApp,
) -> None:
    decision = SimpleNamespace(
        original_to_canonical={1: 7, 3: 5},
        bead_snapshot={5: 6, 7: 3},
        probabilities={5: 2 / 3, 7: 1 / 3},
        canonical_move=5,
    )

    rows = app._decision_probability_rows(decision)

    assert rows == [
        {
            "matchbox_move": 5,
            "original_board_square": 3,
            "beads": 6,
            "probability": pytest.approx(2 / 3),
            "selected": True,
        },
        {
            "matchbox_move": 7,
            "original_board_square": 1,
            "beads": 3,
            "probability": pytest.approx(1 / 3),
            "selected": False,
        },
    ]


def test_decision_probability_rows_returns_independent_dicts(
    app: MENACEStreamlitApp,
) -> None:
    source = DecisionWithRows()
    rows = app._decision_probability_rows(source)

    rows[0]["beads"] = 999
    fresh_rows = source.matchbox_probability_rows()

    assert fresh_rows[0]["beads"] == 3


# ---------------------------------------------------------------------------
# Training-history normalisation helpers
# ---------------------------------------------------------------------------


def test_normalise_history_adds_opponent_wins_from_random_wins(
    app: MENACEStreamlitApp,
) -> None:
    history = pd.DataFrame(
        [{"game": 1, "random_wins": 1, "menace_wins": 0, "draws": 0}]
    )

    result = app._normalise_history(history)

    assert result.loc[0, "opponent_wins"] == 1
    assert "opponent_wins" not in history.columns


def test_normalise_history_adds_random_wins_from_opponent_wins(
    app: MENACEStreamlitApp,
) -> None:
    history = pd.DataFrame(
        [{"game": 1, "opponent_wins": 1, "menace_wins": 0, "draws": 0}]
    )

    result = app._normalise_history(history)

    assert result.loc[0, "random_wins"] == 1
    assert "random_wins" not in history.columns


def test_normalise_history_preserves_existing_columns(
    app: MENACEStreamlitApp,
) -> None:
    history = pd.DataFrame(
        [
            {
                "game": 1,
                "opponent_wins": 1,
                "random_wins": 1,
                "menace_wins": 0,
                "draws": 0,
            }
        ]
    )

    result = app._normalise_history(history)

    pd.testing.assert_frame_equal(result, history)


# ---------------------------------------------------------------------------
# Stored board and mapping helpers
# ---------------------------------------------------------------------------


def test_latest_symmetry_mapping_record_returns_none_when_empty(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.SYMMETRY_MAPPINGS] = []

    assert app._latest_symmetry_mapping_record() is None


def test_latest_symmetry_mapping_record_returns_last_record(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    first = {"move_number": 1}
    last = {"move_number": 2}
    clean_session_state[SessionKey.SYMMETRY_MAPPINGS] = [first, last]

    assert app._latest_symmetry_mapping_record() == last


def test_explain_board_state_from_string_returns_board_explanation() -> None:
    explanation = MENACEStreamlitApp._explain_board_state_from_string(
        "X   O    "
    )

    assert "Original state" in explanation
    assert "Available moves" in explanation


# ---------------------------------------------------------------------------
# Notification behaviour
# ---------------------------------------------------------------------------


def test_notify_uses_streamlit_toast_when_available(
    app: MENACEStreamlitApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_toast(message: str, icon: str) -> None:
        calls.append((message, icon))

    monkeypatch.setattr(st, "toast", fake_toast)

    app._notify("MENACE chose square H.", icon="🤖")

    assert calls == [("MENACE chose square H.", "🤖")]


def test_notify_is_safe_when_toast_is_unavailable(
    app: MENACEStreamlitApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st, "toast", None)

    # The helper should quietly do nothing when toast is unavailable.
    app._notify("Test message")


# ---------------------------------------------------------------------------
# Final-game synchronisation and tied comparison helpers
# ---------------------------------------------------------------------------


def test_friendly_join_handles_one_two_and_many_labels(
    app: MENACEStreamlitApp,
) -> None:
    assert app._friendly_join(["A"]) == "A"
    assert app._friendly_join(["A", "B"]) == "A and B"
    assert app._friendly_join(["A", "B", "C"]) == "A, B, and C"


def test_simple_result_update_sentence_links_win_to_lettered_squares(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Simple"
    clean_session_state[SessionKey.LAST_GAME_RESULT] = "win"
    clean_session_state[SessionKey.LAST_BEAD_CHANGES] = make_learning_rows()

    sentence = app._simple_result_update_sentence()

    assert "Because MENACE won" in sentence
    assert "squares D and I" in sentence
    assert "original move" not in sentence


def test_simple_result_update_sentence_uses_smaller_draw_wording(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    clean_session_state[SessionKey.DISPLAY_MODE] = "Simple"
    clean_session_state[SessionKey.LAST_GAME_RESULT] = "draw"
    clean_session_state[SessionKey.LAST_BEAD_CHANGES] = make_learning_rows()

    assert "smaller bead changes" in app._simple_result_update_sentence()


def test_hardest_opponent_types_returns_all_tied_opponents(
    app: MENACEStreamlitApp,
) -> None:
    grouped = pd.DataFrame(
        [
            {"opponent_type": "Random", "win_rate": 0.30},
            {"opponent_type": "Heuristic", "win_rate": 0.00},
            {"opponent_type": "Minimax", "win_rate": 0.00},
        ]
    )

    assert app._hardest_opponent_types(grouped) == ["Heuristic", "Minimax"]


def test_refresh_decision_records_uses_post_learning_bead_counts(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    from src.menace import Matchbox, MENACEPlayer

    menace = MENACEPlayer(mark="O", seed=42)
    menace.matchboxes["state"] = Matchbox(state="state", beads={2: 6, 5: 3})
    clean_session_state[SessionKey.MENACE] = menace
    clean_session_state[SessionKey.SYMMETRY_MAPPINGS] = [
        {
            "matchbox_state": "state",
            "probability_rows": [
                {"matchbox_move": 2, "beads": 3, "probability": 0.5, "selected": True},
                {"matchbox_move": 5, "beads": 3, "probability": 0.5, "selected": False},
            ],
        }
    ]

    app._refresh_decision_records_after_learning()
    rows = clean_session_state[SessionKey.SYMMETRY_MAPPINGS][0]["probability_rows"]

    assert rows[0]["beads"] == 6
    assert rows[0]["probability"] == pytest.approx(2 / 3)
    assert rows[1]["beads"] == 3

# ---------------------------------------------------------------------------
# Visual demo and polished learner interface
# ---------------------------------------------------------------------------


def test_demo_frames_form_complete_learning_story(app: MENACEStreamlitApp) -> None:
    frames = app._demo_frames()
    assert len(frames) == 5
    assert frames[0]["board"] == [" "] * 9
    assert frames[-1]["last"] == 6
    assert "learns" in frames[-1]["title"].lower()


def test_demo_frames_never_expose_internal_board_indexes(app: MENACEStreamlitApp) -> None:
    frames = app._demo_frames()
    for frame in frames:
        assert len(frame["board"]) == 9
        assert set(frame["board"]).issubset({" ", "X", "O"})


def test_demo_session_keys_are_unique() -> None:
    assert SessionKey.SHOW_DEMO != SessionKey.DEMO_STEP
    assert SessionKey.SHOW_DEMO != SessionKey.SHOW_WELCOME


def test_exported_training_figure_state_is_initialised_as_a_list(
    app: MENACEStreamlitApp,
    clean_session_state: dict[str, Any],
) -> None:
    """The state type must match the exported HTML path list used by app.py."""
    app._initialise_session_state()

    assert clean_session_state[SessionKey.EXPORTED_TRAINING_FIGURES] == []
    assert isinstance(
        clean_session_state[SessionKey.EXPORTED_TRAINING_FIGURES],
        list,
    )


def test_render_exportable_figure_saves_html_and_png(monkeypatch, tmp_path) -> None:
    """Each displayed figure button must save both formats below figures."""
    from types import SimpleNamespace
    from ui.common import CommonMixin

    html_path = tmp_path / "sample.html"
    png_path = tmp_path / "sample.png"
    calls = []
    fake_streamlit = SimpleNamespace(
        plotly_chart=lambda *args, **kwargs: calls.append(("chart", kwargs)),
        button=lambda *args, **kwargs: True,
        success=lambda message: calls.append(("success", message)),
        warning=lambda message: calls.append(("warning", message)),
    )
    monkeypatch.setattr("ui.common.st", fake_streamlit)
    ui = object.__new__(CommonMixin)
    ui.visualiser = SimpleNamespace(
        export_figure_html=lambda figure, name: html_path,
        export_figure_png=lambda figure, name: png_path,
    )

    ui._render_exportable_figure(object(), name="Sample figure", chart_key="sample")

    assert calls[0][0] == "chart"
    assert calls[-1][0] == "success"
    assert str(html_path) in calls[-1][1]
    assert str(png_path) in calls[-1][1]


def test_symmetry_demo_has_original_and_all_seven_equivalents(
    app: MENACEStreamlitApp,
) -> None:
    steps = app._symmetry_demo_steps()

    assert len(steps) == 8
    assert steps[0]["x_square"] == "A"
    assert steps[1]["x_square"] == "C"
    assert "0°" in steps[0]["title"]
    assert "90°" in steps[2]["title"]
    assert "180°" in steps[3]["title"]
    assert "270°" in steps[4]["title"]
    assert sum("Reflection" in step["title"] for step in steps) == 4


def test_symmetry_demo_board_uses_letters_and_requested_x(
    app: MENACEStreamlitApp,
) -> None:
    board_text = app._symmetry_demo_board_text("C")

    assert board_text == "A | B | X\n---------\nD | E | F\n---------\nG | H | I"


def test_symmetry_comparison_keeps_original_separate_from_transformations(
    app: MENACEStreamlitApp,
) -> None:
    steps = app._symmetry_demo_steps()

    original = steps[0]
    transformations = steps[1:]

    assert original["title"] == "Original board — 0° rotation"
    assert original["x_square"] == "A"
    assert transformations[0]["title"] == "Reflection 1 — vertical mirror"
    assert transformations[0]["x_square"] == "C"
    assert len(transformations) == 7
