"""Static regression checks for MENACE light/dark theming."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
CSS = (ROOT / "ui" / "assets" / "styles.css").read_text(encoding="utf-8")
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_streamlit_config_defines_light_and_dark_themes():
    assert "[theme.light]" in CONFIG
    assert "[theme.dark]" in CONFIG
    assert "[theme.light.sidebar]" in CONFIG
    assert "[theme.dark.sidebar]" in CONFIG


def test_theme_uses_streamlit_css_variables_for_surfaces_and_text():
    assert "var(--st-background-color" in CSS
    assert "var(--st-secondary-background-color" in CSS
    assert "var(--st-text-color" in CSS
    assert "var(--st-border-color" in CSS


def test_semantic_menace_colours_are_preserved():
    assert "color: #1769aa !important" in CSS  # X remains blue
    assert "color: #24834b !important" in CSS  # O remains green
    assert "background: #2f9e55 !important" in CSS  # beads remain green
    assert "st-key-sidebar_page_0" in CSS and "var(--menace-blue)" in CSS


def test_streamlit_version_supports_dual_theme_tables():
    assert "streamlit>=1.51.0,<2.0" in REQUIREMENTS


def test_no_local_storage_theme_hack():
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [ROOT / "ui" / "common.py", ROOT / "ui" / "assets" / "styles.css"]
    )
    assert "stActiveTheme" not in combined
    assert "menace-dark-mode" not in combined
