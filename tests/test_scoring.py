"""Tests for multi-signal scoring. Deterministic, hardcoded signal dicts."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import ScoringError, combine_signals  # noqa: E402


def groq(ai_score, reliability=1.0):
    """Build a Groq-shaped signal dict."""
    return {
        "ai_score": ai_score,
        "reliability": reliability,
        "flags": [],
        "signal": "groq",
    }


def style(ai_score, reliability=1.0, word_count=200):
    """Build a stylometry-shaped signal dict with a word_count feature."""
    return {
        "ai_score": ai_score,
        "reliability": reliability,
        "features": {"word_count": word_count},
        "component_scores": {},
        "signal": "stylometry",
    }


def _outputs_in_range(result):
    assert 0.0 <= result["ai_likelihood"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


# 1. Strong agreement, high scores, sufficient length -> likely_ai.
def test_case1_likely_ai():
    result = combine_signals(groq(0.94), style(0.89))
    _outputs_in_range(result)
    assert result["attribution"] == "likely_ai"
    assert result["forced_uncertain"] is False
    assert result["signal_gap"] == pytest.approx(0.05)


# 2. Strong agreement, low scores, sufficient length -> likely_human.
def test_case2_likely_human():
    result = combine_signals(groq(0.12), style(0.18))
    _outputs_in_range(result)
    assert result["attribution"] == "likely_human"
    assert result["forced_uncertain"] is False


# 3. Middling scores -> uncertain.
def test_case3_uncertain_middle():
    result = combine_signals(groq(0.67), style(0.54))
    _outputs_in_range(result)
    assert result["attribution"] == "uncertain"


# 4. Large gap -> uncertain via the signal-disagreement rule.
def test_case4_uncertain_disagreement():
    result = combine_signals(groq(0.91), style(0.31))
    _outputs_in_range(result)
    assert result["attribution"] == "uncertain"
    assert result["signal_gap"] >= 0.35
    assert result["forced_uncertain"] is True
    assert "signal_disagreement" in result["uncertainty_reasons"]


# 5. High scores but fewer than 50 words -> uncertain, confidence capped 0.69.
def test_case5_short_text_capped():
    result = combine_signals(groq(0.95), style(0.92, word_count=30))
    _outputs_in_range(result)
    assert result["attribution"] == "uncertain"
    assert result["confidence"] <= 0.69
    assert result["forced_uncertain"] is True
    assert "insufficient_length" in result["uncertainty_reasons"]


# 6. One signal missing -> uncertain, confidence capped 0.60.
def test_case6_one_missing():
    result = combine_signals(groq(0.95), None)
    _outputs_in_range(result)
    assert result["attribution"] == "uncertain"
    assert result["confidence"] <= 0.60
    assert result["signal_gap"] is None
    assert result["forced_uncertain"] is True
    assert "missing_signal" in result["uncertainty_reasons"]

    # Symmetric: the other signal missing behaves the same.
    result2 = combine_signals(None, style(0.10))
    assert result2["attribution"] == "uncertain"
    assert result2["confidence"] <= 0.60


# 7. Both missing -> controlled exception.
def test_case7_both_missing_raises():
    with pytest.raises(ScoringError):
        combine_signals(None, None)


def test_confidence_is_max_of_likelihood_complement():
    result = combine_signals(groq(0.94), style(0.89))
    expected = max(result["ai_likelihood"], 1 - result["ai_likelihood"])
    assert result["confidence"] == pytest.approx(expected)


def test_reliability_weighting_shifts_likelihood():
    # When stylometry is unreliable, the combined score leans toward Groq.
    leans_groq = combine_signals(
        groq(0.90, reliability=1.0), style(0.10, reliability=0.20)
    )
    # Groq dominates, so likelihood should sit well above the plain average 0.5.
    assert leans_groq["ai_likelihood"] > 0.5
