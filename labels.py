"""Transparency-label text for Provenance Guard.

These are the FINAL, reader-facing labels (Milestone 5). The exact wording is
centralized here so the API and README stay in sync. Do not paraphrase or
shorten these strings — callers compare against them verbatim.

Design goals (see planning.md): avoid claiming certainty, avoid accusing the
creator, make uncertainty visible, and mention that appeals are available.
"""

# High-confidence AI label.
LABEL_AI = (
    "This content shows strong signals of AI generation. This automated "
    "assessment is not proof of authorship and may be incorrect. The creator "
    "can appeal this result and provide additional context or evidence."
)

# High-confidence human label. Note the typographic apostrophe in
# "creator’s" (U+2019), matching planning.md exactly.
LABEL_HUMAN = (
    "This content shows strong signals of human authorship. This automated "
    "assessment is probabilistic and does not verify the creator’s "
    "identity, ownership, or full writing process."
)

# Uncertain label.
LABEL_UNCERTAIN = (
    "The system could not confidently determine whether this content was "
    "human-written or AI-generated. No definitive attribution is being made, "
    "and the content should not be penalized based on this result."
)

# Confidence thresholds required for a high-confidence label. These mirror the
# attribution thresholds in planning.md and must not be loosened here.
AI_LABEL_MIN_CONFIDENCE = 0.85
HUMAN_LABEL_MIN_CONFIDENCE = 0.80


def get_transparency_label(attribution: str, confidence: float) -> str:
    """Return the exact transparency label for an attribution + confidence.

    Returns one of LABEL_AI, LABEL_HUMAN, or LABEL_UNCERTAIN.

    A high-confidence label is only used when BOTH the attribution and the
    confidence support it. Everything else — uncertain attributions,
    inconsistent attribution/confidence combinations, unknown attribution
    values, missing-signal cases, short-text forced-uncertain cases, and
    signal-disagreement cases — resolves to the conservative uncertain label.
    """
    # Coerce confidence defensively; a missing/garbage value means we cannot
    # justify a high-confidence label, so we fall through to uncertain.
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if attribution == "likely_ai" and confidence >= AI_LABEL_MIN_CONFIDENCE:
        return LABEL_AI
    if attribution == "likely_human" and confidence >= HUMAN_LABEL_MIN_CONFIDENCE:
        return LABEL_HUMAN
    return LABEL_UNCERTAIN
