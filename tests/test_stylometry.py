"""Tests for Detection Signal 2 (stylometry). Deterministic, no Groq calls."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stylometry import StylometryError, run_stylometric_signal  # noqa: E402


def _scores_in_range(result):
    """Assert ai_score, reliability, and every component score are in 0.0-1.0."""
    assert 0.0 <= result["ai_score"] <= 1.0
    assert 0.0 <= result["reliability"] <= 1.0
    for value in result["component_scores"].values():
        assert 0.0 <= value <= 1.0


# A long, highly uniform text: every sentence identical length, identical
# opener, repeated vocabulary -> should look very "AI-like regular".
UNIFORM_TEXT = (
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way. "
    "The system works in a steady and reliable way."
)

# An irregular human-style text: varied sentence lengths, varied openers,
# rich vocabulary, mixed punctuation.
IRREGULAR_TEXT = (
    "Rain. "
    "I never expected the morning to unravel quite like that, but somehow it did, "
    "spilling coffee across my grandmother's old letters and a half-finished sketch. "
    "Why? "
    "Nobody could say, and frankly I stopped asking after the third strange phone call. "
    "Later, much later, we laughed about the chaos — the dog, the umbrella, that absurd taxi driver "
    "who insisted he knew a shortcut through the flooded underpass!"
)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        run_stylometric_signal("")


def test_whitespace_only_raises():
    with pytest.raises(StylometryError):
        run_stylometric_signal("    \n\t  ")


def test_short_text_low_reliability():
    result = run_stylometric_signal("A short little sentence about cats.")
    _scores_in_range(result)
    # Fewer than 50 words -> reliability 0.20.
    assert result["reliability"] == 0.20
    assert result["signal"] == "stylometry"


def test_long_uniform_text_scores_high_and_in_range():
    result = run_stylometric_signal(UNIFORM_TEXT)
    _scores_in_range(result)
    # Uniform, repetitive text should land clearly on the AI-like side.
    assert result["ai_score"] > 0.5
    assert result["features"]["repeated_opener_rate"] > 0.5


def test_irregular_text_scores_lower_than_uniform():
    uniform = run_stylometric_signal(UNIFORM_TEXT)
    irregular = run_stylometric_signal(IRREGULAR_TEXT)
    _scores_in_range(irregular)
    # The irregular human-style text must look less AI-like than uniform text.
    assert irregular["ai_score"] < uniform["ai_score"]


def test_poem_repetition_damped():
    # A repetitive poem: identical structure would normally read as AI-like.
    poem = (
        "I rise. I rise. I rise again.\n\n"
        "I rise. I rise. I rise again.\n\n"
        "I rise. I rise. I rise again."
    )
    as_poem = run_stylometric_signal(poem, content_type="poem")
    as_other = run_stylometric_signal(poem, content_type="other")
    _scores_in_range(as_poem)
    _scores_in_range(as_other)
    # Repetition is damped for poems, so its repetition component (and thus the
    # overall AI score) must be no higher than the non-poem reading.
    assert (
        as_poem["component_scores"]["repetition_score"]
        < as_other["component_scores"]["repetition_score"]
    )
    assert as_poem["ai_score"] <= as_other["ai_score"]


@pytest.mark.parametrize(
    "n_words,expected_reliability",
    [(10, 0.20), (70, 0.50), (150, 0.75), (300, 1.00)],
)
def test_reliability_changes_with_word_count(n_words, expected_reliability):
    # Build a text with exactly n_words words across a few sentences.
    text = " ".join(["word"] * n_words) + "."
    result = run_stylometric_signal(text)
    assert result["features"]["word_count"] == n_words
    assert result["reliability"] == expected_reliability


def test_no_divide_by_zero_single_sentence():
    # Single sentence, single paragraph: must not raise and stay in range.
    result = run_stylometric_signal("Just one solitary sentence here")
    _scores_in_range(result)
    assert result["features"]["sentence_count"] == 1


def test_sentence_parsing_handles_all_terminators():
    result = run_stylometric_signal("First one. Second one! Third one? Fourth one.")
    assert result["features"]["sentence_count"] == 4
