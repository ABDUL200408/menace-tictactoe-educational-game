"""Audit the real learner-facing Streamlit text and export readability evidence.

Run from the project root:
    python tools/readability_audit.py

The script parses the UI source with Python's AST, collects literal text passed
to learner-facing Streamlit output functions, calculates Flesch metrics and
writes results/readability_results.csv.  This avoids keeping a second copy of
interface text only for testing.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.readability import readability_metrics  # noqa: E402


OUTPUT_FUNCTIONS = {
    "markdown",
    "write",
    "caption",
    "info",
    "success",
    "warning",
    "error",
    "header",
    "subheader",
    "title",
}

# These files contain the main explanations presented to novice learners.
CORE_LEARNER_FILES = (
    "ui/navigation.py",
    "ui/common.py",
    "ui/demos.py",
    "ui/pages/play.py",
    "ui/pages/training.py",
    "ui/pages/comparison.py",
    "ui/pages/results.py",
    "ui/pages/learning.py",
)

TARGET_MAX_GRADE = 10.0
MIN_WORDS_FOR_THRESHOLD = 50


@dataclass(frozen=True)
class AuditRow:
    source: str
    passages: int
    sentences: int
    words: int
    syllables: int
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    target_max_grade: float
    status: str


def _static_text(node: ast.AST) -> str:
    """Return static wording from constants and f-strings without executing UI."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                # Preserve sentence shape without depending on runtime values.
                parts.append(" value ")
        return "".join(parts)
    return ""


def extract_streamlit_text(path: Path) -> list[str]:
    """Collect static learner-facing strings passed to Streamlit display calls."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    passages: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function_name = None
        if isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            function_name = node.func.id
        if function_name not in OUTPUT_FUNCTIONS:
            continue
        text = _static_text(node.args[0]).strip()
        if text:
            passages.append(text)
    return passages


def audit_file(relative_path: str) -> AuditRow:
    """Audit one UI source file and classify it against the project target."""

    passages = extract_streamlit_text(ROOT / relative_path)
    result = readability_metrics(" ".join(passages))
    if result.words < MIN_WORDS_FOR_THRESHOLD:
        status = "Informational: sample too short for project threshold"
    elif result.flesch_kincaid_grade <= TARGET_MAX_GRADE:
        status = "Pass"
    else:
        status = "Review wording"
    return AuditRow(
        source=relative_path,
        passages=len(passages),
        sentences=result.sentences,
        words=result.words,
        syllables=result.syllables,
        flesch_reading_ease=result.flesch_reading_ease,
        flesch_kincaid_grade=result.flesch_kincaid_grade,
        target_max_grade=TARGET_MAX_GRADE,
        status=status,
    )


def build_audit() -> list[AuditRow]:
    """Calculate readability for all core learner-facing interface modules."""

    return [audit_file(path) for path in CORE_LEARNER_FILES]


def write_csv(rows: list[AuditRow], output: Path) -> None:
    """Write reproducible readability evidence for the dissertation."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AuditRow.__dataclass_fields__)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    rows = build_audit()
    output = ROOT / "results" / "readability_results.csv"
    write_csv(rows, output)

    print("Flesch-Kincaid readability audit")
    print(f"Target for core learner-facing text: grade <= {TARGET_MAX_GRADE:.1f}")
    for row in rows:
        print(
            f"{row.source}: words={row.words}, "
            f"FRE={row.flesch_reading_ease:.1f}, "
            f"FK grade={row.flesch_kincaid_grade:.1f}, {row.status}"
        )
    print(f"Evidence written to: {output.relative_to(ROOT)}")
    return 1 if any(row.status == "Review wording" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
