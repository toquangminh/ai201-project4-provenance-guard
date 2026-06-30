"""Tests for the exact transparency labels. Pure function, no I/O."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labels import (  # noqa: E402
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_UNCERTAIN,
    get_transparency_label,
)

# Verbatim expected strings (independent copies) so a paraphrase in labels.py
# is caught here. Note the typographic apostrophe in "creator’s".
EXPECTED_AI = (
    "This content shows strong signals of AI generation. This automated "
    "assessment is not proof of authorship and may be incorrect. The creator "
    "can appeal this result and provide additional context or evidence."
)
EXPECTED_HUMAN = (
    "This content shows strong signals of human authorship. This automated "
    "assessment is probabilistic and does not verify the creator’s "
    "identity, ownership, or full writing process."
)
EXPECTED_UNCERTAIN = (
    "The system could not confidently determine whether this content was "
    "human-written or AI-generated. No definitive attribution is being made, "
    "and the content should not be penalized based on this result."
)


def test_label_constants_are_exact():
    assert LABEL_AI == EXPECTED_AI
    assert LABEL_HUMAN == EXPECTED_HUMAN
    assert LABEL_UNCERTAIN == EXPECTED_UNCERTAIN


# --- High-confidence AI ---------------------------------------------------
def test_likely_ai_high_confidence_returns_ai_label():
    assert get_transparency_label("likely_ai", 0.85) == EXPECTED_AI
    assert get_transparency_label("likely_ai", 0.95) == EXPECTED_AI


def test_likely_ai_below_threshold_is_uncertain():
    # Inconsistent combo: AI attribution but confidence < 0.85.
    assert get_transparency_label("likely_ai", 0.84) == EXPECTED_UNCERTAIN


# --- High-confidence human ------------------------------------------------
def test_likely_human_high_confidence_returns_human_label():
    assert get_transparency_label("likely_human", 0.80) == EXPECTED_HUMAN
    assert get_transparency_label("likely_human", 0.99) == EXPECTED_HUMAN


def test_likely_human_below_threshold_is_uncertain():
    assert get_transparency_label("likely_human", 0.79) == EXPECTED_UNCERTAIN


# --- Uncertain & edge cases ----------------------------------------------
def test_uncertain_attribution_returns_uncertain_label():
    assert get_transparency_label("uncertain", 0.95) == EXPECTED_UNCERTAIN
    assert get_transparency_label("uncertain", 0.10) == EXPECTED_UNCERTAIN


def test_unknown_attribution_returns_uncertain_label():
    assert get_transparency_label("banana", 0.99) == EXPECTED_UNCERTAIN
    assert get_transparency_label("", 0.99) == EXPECTED_UNCERTAIN


def test_bad_confidence_value_is_uncertain():
    assert get_transparency_label("likely_ai", None) == EXPECTED_UNCERTAIN
    assert get_transparency_label("likely_human", "oops") == EXPECTED_UNCERTAIN


def test_always_returns_one_of_three():
    valid = {EXPECTED_AI, EXPECTED_HUMAN, EXPECTED_UNCERTAIN}
    for attribution in ("likely_ai", "likely_human", "uncertain", "weird"):
        for confidence in (0.0, 0.5, 0.8, 0.85, 1.0):
            assert get_transparency_label(attribution, confidence) in valid
