"""Regression tests for learner-facing Flesch-Kincaid readability."""
from pathlib import Path
import importlib.util

from src.readability import readability_metrics


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "readability_audit.py"
_spec = importlib.util.spec_from_file_location("readability_audit", AUDIT_PATH)
assert _spec and _spec.loader
_audit = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = _audit
_spec.loader.exec_module(_audit)


def test_flesch_metrics_return_plausible_values_for_plain_text() -> None:
    result = readability_metrics(
        "MENACE learns by playing games. Good moves gain beads. "
        "This makes those moves more likely next time."
    )
    assert result.sentences == 3
    assert result.words > 10
    assert 0.0 <= result.flesch_reading_ease <= 120.0
    assert -5.0 <= result.flesch_kincaid_grade <= 12.0


def test_core_learner_text_meets_project_readability_target() -> None:
    rows = _audit.build_audit()
    assert rows
    failed = [row for row in rows if row.status == "Review wording"]
    assert not failed, "Learner-facing text exceeds the agreed FK target: " + ", ".join(
        f"{row.source}={row.flesch_kincaid_grade:.2f}" for row in failed
    )


def test_readability_audit_uses_actual_ui_source_text() -> None:
    rows = {row.source: row for row in _audit.build_audit()}
    assert rows["ui/pages/play.py"].words > 100
    assert rows["ui/pages/training.py"].words > 100
    assert rows["ui/pages/comparison.py"].words > 100
    assert rows["ui/pages/results.py"].words > 100
