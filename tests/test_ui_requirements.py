"""Static regression checks for core MENACE interface requirements."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_primary_action_palette_is_green_and_focus_is_visible() -> None:
    css = (ROOT / "ui" / "assets" / "styles.css").read_text(encoding="utf-8")

    assert "--menace-accent: #267a3f" in css
    assert 'data-testid="stBaseButton-primary"' in css
    assert "button:focus-visible" in css
    assert "--menace-focus: #0b5cab" in css


def test_header_explains_matchboxes_beads_and_rewards() -> None:
    source = (ROOT / "ui" / "navigation.py").read_text(encoding="utf-8")

    assert "Each board" in source
    assert "situation has a matchbox" in source
    assert "Helpful choices gain beads" in source


def test_beginner_guide_is_self_explanatory() -> None:
    source = (ROOT / "ui" / "common.py").read_text(encoding="utf-8")

    assert "What are the matchboxes and beads?" in source
    assert "Each different board" in source
    assert "reward-and-penalty process" in source
    assert 'st.markdown("#### Start here")' in source


def test_sidebar_has_a_clear_start_here_entry() -> None:
    source = (ROOT / "ui" / "navigation.py").read_text(encoding="utf-8")

    assert 'instructions_label = "Start here: beginner guide"' in source
