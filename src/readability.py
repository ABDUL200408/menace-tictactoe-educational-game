"""Readability utilities for auditing learner-facing interface text.

The module implements the standard Flesch Reading Ease and Flesch-Kincaid
Grade Level formulae. Syllables are estimated deterministically with the
British-English Pyphen dictionary so that the same audit can be reproduced locally.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import pyphen


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_HTML_RE = re.compile(r"<[^>]+>")
_MARKDOWN_RE = re.compile(r"[#*_`>|]")
_HYPHENATOR = pyphen.Pyphen(lang="en_GB")


@dataclass(frozen=True)
class ReadabilityResult:
    """Stores readability statistics calculated for one body of text."""

    sentences: int
    words: int
    syllables: int
    flesch_reading_ease: float
    flesch_kincaid_grade: float


def normalise_visible_text(text: str) -> str:
    """Remove simple HTML and Markdown decoration before readability scoring."""

    text = _HTML_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_syllables(word: str) -> int:
    """Estimate syllable count using Pyphen's British-English dictionary."""

    cleaned = re.sub(r"[^A-Za-z']", "", word).lower()
    if not cleaned:
        return 0
    return max(1, _HYPHENATOR.inserted(cleaned).count("-") + 1)


def readability_metrics(text: str) -> ReadabilityResult:
    """Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.

    Flesch Reading Ease = 206.835 - 1.015(ASL) - 84.6(ASW)
    Flesch-Kincaid Grade = 0.39(ASL) + 11.8(ASW) - 15.59

    where ASL is average sentence length and ASW is average syllables per word.
    """

    visible = normalise_visible_text(text)
    words = _WORD_RE.findall(visible)
    if not words:
        return ReadabilityResult(0, 0, 0, 0.0, 0.0)

    sentences = [
        sentence
        for sentence in _SENTENCE_RE.split(visible)
        if _WORD_RE.search(sentence)
    ]
    sentence_count = max(1, len(sentences))
    word_count = len(words)
    syllable_count = sum(count_syllables(word) for word in words)

    average_sentence_length = word_count / sentence_count
    average_syllables_per_word = syllable_count / word_count

    reading_ease = (
        206.835
        - 1.015 * average_sentence_length
        - 84.6 * average_syllables_per_word
    )
    grade = (
        0.39 * average_sentence_length
        + 11.8 * average_syllables_per_word
        - 15.59
    )

    return ReadabilityResult(
        sentences=sentence_count,
        words=word_count,
        syllables=syllable_count,
        flesch_reading_ease=round(reading_ease, 2),
        flesch_kincaid_grade=round(grade, 2),
    )
