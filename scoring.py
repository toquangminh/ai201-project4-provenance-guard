"""Multi-signal confidence scoring for Provenance Guard.

Combines the Groq signal (Signal 1) and the stylometric signal (Signal 2)
into a single AI likelihood, confidence, and attribution, applying the
conservative rules from planning.md:

  * reliability-adjusted weighting (Groq 0.65, stylometry 0.35 base)
  * signal-disagreement rule (gap >= 0.35 -> uncertain)
  * short-text rule (< 50 words -> uncertain, confidence capped at 0.69)
  * missing-signal rule (one missing -> uncertain, confidence capped at 0.60;
    both missing -> controlled error)

A false positive (calling human work AI) is treated as more harmful than a
false negative, so anything ambiguous resolves to "uncertain". This module
never fabricates a score for a missing signal.
"""

# Base weights. The LLM signal gets more weight because it can interpret
# meaning, genre, and voice; stylometry is an independent structural check.
GROQ_BASE_WEIGHT = 0.65
STYLE_BASE_WEIGHT = 0.35

# Decision thresholds (see planning.md "Attribution thresholds").
LIKELY_AI_THRESHOLD = 0.85
LIKELY_HUMAN_THRESHOLD = 0.20
# Signals are considered to strongly disagree at/above this absolute gap.
SIGNAL_GAP_THRESHOLD = 0.35
# Minimum word count for a confident (non-uncertain) attribution.
MIN_WORD_COUNT = 50

# Confidence caps applied by the conservative rules.
SHORT_TEXT_CONFIDENCE_CAP = 0.69
MISSING_SIGNAL_CONFIDENCE_CAP = 0.60


class ScoringError(Exception):
    """Raised when scoring cannot proceed at all (e.g. both signals missing)."""


def _clamp(value: float) -> float:
    """Clamp a numeric value into the inclusive range 0.0-1.0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def combine_signals(
    groq_result: dict | None,
    stylometric_result: dict | None,
) -> dict:
    """Combine the two detection signals into a single attribution decision.

    Args:
        groq_result: the dict returned by detector.run_groq_signal, or None if
            that signal is missing/failed.
        stylometric_result: the dict returned by
            stylometry.run_stylometric_signal, or None if missing/failed.

    Returns:
        {
            "ai_likelihood": float,           # 0.0-1.0
            "confidence": float,              # 0.0-1.0
            "attribution": str,               # likely_ai | likely_human | uncertain
            "signal_gap": float | None,       # None when a signal is missing
            "forced_uncertain": bool,
            "uncertainty_reasons": list[str],
        }

    Raises:
        ScoringError: if BOTH signals are missing — there is nothing to score.
    """
    groq_ok = isinstance(groq_result, dict)
    style_ok = isinstance(stylometric_result, dict)

    # --- Both signals missing: controlled error ------------------------
    if not groq_ok and not style_ok:
        raise ScoringError(
            "Both detection signals are missing; cannot produce an attribution."
        )

    # --- Exactly one signal missing: forced uncertain, capped at 0.60 --
    # We do NOT invent a score for the missing signal. We still surface the
    # surviving signal's likelihood so the response is useful.
    if not groq_ok or not style_ok:
        present = groq_result if groq_ok else stylometric_result
        ai_likelihood = _clamp(present.get("ai_score"))
        confidence = min(
            max(ai_likelihood, 1 - ai_likelihood), MISSING_SIGNAL_CONFIDENCE_CAP
        )
        return {
            "ai_likelihood": ai_likelihood,
            "confidence": _clamp(confidence),
            "attribution": "uncertain",
            "signal_gap": None,
            "forced_uncertain": True,
            "uncertainty_reasons": ["missing_signal"],
        }

    # --- Both signals present ------------------------------------------
    groq_score = _clamp(groq_result.get("ai_score"))
    style_score = _clamp(stylometric_result.get("ai_score"))
    groq_reliability = _clamp(groq_result.get("reliability"))
    style_reliability = _clamp(stylometric_result.get("reliability"))

    # Word count comes from the stylometric features; fall back to 0 (which
    # triggers the short-text rule) if it is absent.
    word_count = 0
    features = stylometric_result.get("features")
    if isinstance(features, dict):
        try:
            word_count = int(features.get("word_count", 0))
        except (TypeError, ValueError):
            word_count = 0

    # Reliability-adjusted effective weights.
    effective_groq_weight = GROQ_BASE_WEIGHT * groq_reliability
    effective_style_weight = STYLE_BASE_WEIGHT * style_reliability
    denominator = effective_groq_weight + effective_style_weight

    uncertainty_reasons: list[str] = []

    if denominator == 0:
        # Both reliabilities are 0 — we can't trust the weighting. Fall back to
        # a plain average for the likelihood and force uncertain.
        ai_likelihood = _clamp((groq_score + style_score) / 2)
        uncertainty_reasons.append("zero_reliability_weight")
    else:
        ai_likelihood = _clamp(
            (groq_score * effective_groq_weight + style_score * effective_style_weight)
            / denominator
        )

    signal_gap = abs(groq_score - style_score)

    # Base confidence: how strongly the likelihood leans away from 0.5.
    confidence = max(ai_likelihood, 1 - ai_likelihood)

    # --- Conservative forcing rules ------------------------------------
    forced_uncertain = bool(uncertainty_reasons)

    if signal_gap >= SIGNAL_GAP_THRESHOLD:
        # Strong disagreement: the two signals point in conflicting directions.
        forced_uncertain = True
        uncertainty_reasons.append("signal_disagreement")

    if word_count < MIN_WORD_COUNT:
        # Too little text for a stable structural measurement.
        forced_uncertain = True
        confidence = min(confidence, SHORT_TEXT_CONFIDENCE_CAP)
        uncertainty_reasons.append("insufficient_length")

    # --- Attribution ----------------------------------------------------
    if (
        not forced_uncertain
        and ai_likelihood >= LIKELY_AI_THRESHOLD
        and signal_gap < SIGNAL_GAP_THRESHOLD
        and word_count >= MIN_WORD_COUNT
    ):
        attribution = "likely_ai"
    elif (
        not forced_uncertain
        and ai_likelihood <= LIKELY_HUMAN_THRESHOLD
        and signal_gap < SIGNAL_GAP_THRESHOLD
        and word_count >= MIN_WORD_COUNT
    ):
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    return {
        "ai_likelihood": _clamp(ai_likelihood),
        "confidence": _clamp(confidence),
        "attribution": attribution,
        "signal_gap": signal_gap,
        "forced_uncertain": forced_uncertain,
        "uncertainty_reasons": uncertainty_reasons,
    }
