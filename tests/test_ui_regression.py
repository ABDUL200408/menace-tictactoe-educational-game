"""Static regression checks for the MENACE interface."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_sidebar_uses_page_buttons_with_hover_help() -> None:
    navigation = read("ui/navigation.py")
    assert 'key=f"sidebar_page_{page_index}"' in navigation
    assert "help=self._button_help(page_label)" in navigation
    assert 'button_label = f"✓ {page_label}"' in navigation


def test_every_page_has_location_banner_and_two_line_description() -> None:
    navigation = read("ui/navigation.py")
    controller = read("ui/controller.py")
    assert "current-page-banner" in navigation
    assert "You are here" in navigation
    assert "<br>{line_two}" in navigation
    assert "self._render_current_page_banner(selected_page)" in controller


def test_board_marks_and_beads_use_semantic_colours() -> None:
    css = read("ui/assets/styles.css")
    assert ".demo-cell.human {color:#1769aa" in css
    assert ".demo-cell.menace {color:#24834b" in css
    assert ".live-board-cell.human {color:#1769aa" in css
    assert ".live-board-cell.menace {color:#24834b" in css
    assert "background:#2f9e55 !important" in css


def test_training_and_results_have_semantic_colour_classes() -> None:
    training = read("ui/pages/training.py")
    results = read("ui/pages/results.py")
    css = read("ui/assets/styles.css")
    assert 'training-result-card {colour_class}' in training
    assert 'results-summary-card {colour_class}' in results
    assert ".training-result-card.loss" in css
    assert ".training-result-card.draw" in css
    assert ".results-summary-card.loss" in css
    assert ".results-summary-card.draw" in css


def test_matchbox_inline_beads_are_green() -> None:
    visualisation = read("src/visualisation.py")
    assert "background:#2f9e55;border:1px solid #176b34" in visualisation
    assert ".matchbox-lid {{background:#dff2e4;color:#185a2c" in visualisation


def test_navigation_is_blue_while_actions_remain_green() -> None:
    css = read("ui/assets/styles.css")
    assert ".st-key-sidebar_page_0 button" in css
    assert "background: var(--menace-blue) !important" in css
    assert "Green identifies actions" in css
    assert "background: var(--menace-green) !important" in css


def test_demo_secondary_controls_are_visually_distinct() -> None:
    css = read("ui/assets/styles.css")
    demos = read("ui/demos.py")
    assert ".st-key-demo_previous button" in css
    assert ".st-key-demo_close button" in css
    assert 'key="demo_next"' in demos
    assert 'type="primary"' in demos


def test_learning_journey_controls_have_distinct_semantic_colours() -> None:
    navigation = read("ui/navigation.py")
    controller = read("ui/controller.py")
    persistence = read("ui/persistence_controls.py")
    css = read("ui/assets/styles.css")

    assert 'key="continue_previous_learning"' in navigation
    assert 'key="reset_learning_journey"' in navigation
    assert ".st-key-continue_previous_learning button" in css
    assert "background: #e6f5ea !important" in css
    assert ".st-key-reset_learning_journey button:not(:disabled)" in css
    assert "background: #fdeaea !important" in css

    # Browser identities must remain isolated and stable across server restarts.
    assert "_browser_identity_key" in controller
    assert "_participant_registry_path" in controller
    assert ".participant_registry.json" in controller
    assert "_identity_owned_by_other_browser" in controller
    assert "_recover_single_saved_participant" not in controller
    assert "same_server_run" not in controller
    assert "return self.config.model_path.exists()" in persistence


def test_demo_previous_is_light_blue_and_close_is_blue() -> None:
    css = read("ui/assets/styles.css")
    assert ".st-key-demo_previous button" in css
    assert "background: #e8f3f9 !important" in css
    assert ".st-key-demo_close button" in css
    assert "background: var(--menace-blue) !important" in css


def test_simple_play_page_avoids_duplicate_bead_recap() -> None:
    play = read("ui/pages/play.py")
    assert 'st.markdown("### How the beads teach it")' not in play


def test_use_symmetry_technical_expanders_are_not_shown() -> None:
    demos = read("ui/demos.py")
    play = read("ui/pages/play.py")
    assert "What does use_symmetry mean in this demo?" not in demos
    assert "Show computer-science details about use_symmetry" not in play


def test_sidebar_blue_rules_override_generic_sidebar_action_colour() -> None:
    css = read("ui/assets/styles.css")
    assert '[data-testid="stSidebar"] .st-key-sidebar_page_0 button' in css
    assert 'button[data-testid="stBaseButton-secondary"]' in css
    assert "background: var(--menace-blue) !important" in css


def test_symmetry_explainer_buttons_are_light_blue() -> None:
    css = read("ui/assets/styles.css")
    for key in (
        "symmetry_demo_previous",
        "symmetry_demo_original",
        "symmetry_demo_next",
    ):
        assert f".st-key-{key} button" in css
    assert "background: #e8f3f9 !important" in css
    assert "border-color: #9cc4d8 !important" in css
