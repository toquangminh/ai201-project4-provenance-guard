"""Detection Signal 2: stylometric heuristics for Provenance Guard.

This signal measures *structural* properties of the text (sentence/paragraph
regularity, repetition, punctuation, vocabulary) without interpreting meaning.
It is intentionally independent of the Groq LLM signal so the two can act as a
cross-check in scoring.py.

IMPORTANT: these heuristics do NOT prove authorship. They measure surface
regularity that is *correlated* with machine-generated text, but a polished
human writer or a structured poem can trigger the same patterns. The output is
probabilistic evidence only. The score direction is: higher == more "AI-like
regularity".

Standard library only — no external models or APIs.
"""

import re

# Tunable heuristic thresholds. These are deliberately conservative and are
# explained where they are used. They are NOT calibrated against a labeled
# corpus; they are reasonable defaults for a course project.
#
# A coefficient of variation (std / mean) at or above this value is treated as
# "fully varied" (human-like) -> uniformity score 0.0. A CV of 0 (perfectly
# uniform) -> uniformity 1.0. Human prose typically shows sentence-length CV in
# roughly the 0.4-0.8 range, so 0.75 marks the high-variation end.
_SENTENCE_CV_FULL_VARIATION = 0.75
_PARAGRAPH_CV_FULL_VARIATION = 0.75
# Punctuation-per-sentence variation: low variation == very regular == AI-like.
_PUNCT_CV_FULL_VARIATION = 1.0
# For poems, repetition is often an intentional human technique, so we damp its
# contribution rather than letting it inflate the AI score.
_POEM_REPETITION_DAMPING = 0.5


class StylometryError(ValueError):
    """Raised when the input text cannot be analyzed (e.g. empty/whitespace).

    Subclasses ValueError so callers that catch ValueError still work.
    """


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


def _mean(values: list[float]) -> float:
    """Arithmetic mean; 0.0 for an empty list (never divides by zero)."""
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    """Population variance; 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _coefficient_of_variation(values: list[float]) -> float:
    """Std / mean. Returns 0.0 when the mean is 0 (no divide-by-zero)."""
    m = _mean(values)
    if m == 0:
        return 0.0
    return (_variance(values) ** 0.5) / m


def _tokenize_words(text: str) -> list[str]:
    """Split into word tokens (letters/digits/apostrophes), lowercased."""
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', and '?'.

    Consecutive terminators (e.g. '?!' or '...') are treated as one boundary.
    Empty fragments are discarded.
    """
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on one or more blank lines."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _count_punctuation(text: str) -> int:
    """Count common punctuation marks."""
    return len(re.findall(r"[.,;:!?\"'()\-]", text))


def _reliability_for_word_count(word_count: int) -> float:
    """Reliability rises with length; short texts are structurally unstable.

    <50 words -> 0.20, 50-99 -> 0.50, 100-249 -> 0.75, 250+ -> 1.00.
    """
    if word_count < 50:
        return 0.20
    if word_count < 100:
        return 0.50
    if word_count < 250:
        return 0.75
    return 1.00


def run_stylometric_signal(text: str, content_type: str = "other") -> dict:
    """Run the stylometric (structural) detection signal on a piece of text.

    Args:
        text: raw creative text.
        content_type: one of "poem", "short_story", "blog_post", "other".
            Only "poem" changes behavior (repetition is damped).

    Returns:
        A dict with ai_score, reliability, features, component_scores, and
        signal == "stylometry". All scores are clamped to 0.0-1.0.

    Raises:
        StylometryError (a ValueError subclass): if text is empty or only
        whitespace, since there is nothing to measure.
    """
    if text is None or not text.strip():
        raise StylometryError("Cannot analyze empty or whitespace-only text.")

    words = _tokenize_words(text)
    word_count = len(words)

    sentences = _split_sentences(text)
    sentence_count = len(sentences)
    # Sentence length measured in words.
    sentence_lengths = [len(_tokenize_words(s)) for s in sentences]

    paragraphs = _split_paragraphs(text)
    paragraph_lengths = [len(_tokenize_words(p)) for p in paragraphs]

    # --- Raw features --------------------------------------------------
    average_sentence_length = _mean(sentence_lengths)
    sentence_length_variance = _variance(sentence_lengths)
    sentence_length_cv = _coefficient_of_variation(sentence_lengths)

    # Type-token ratio: unique words / total words (guarded).
    type_token_ratio = (len(set(words)) / word_count) if word_count else 0.0

    # Punctuation density: punctuation marks per word (guarded).
    punctuation_density = (_count_punctuation(text) / word_count) if word_count else 0.0

    paragraph_length_variance = _variance(paragraph_lengths)

    # Repeated-opener rate: fraction of sentences whose first word duplicates a
    # first word already seen. AI text often reuses sentence openers.
    if sentence_count > 1:
        openers = [
            _tokenize_words(s)[0] for s in sentences if _tokenize_words(s)
        ]
        unique_openers = len(set(openers))
        repeated_opener_rate = (
            1 - (unique_openers / len(openers)) if openers else 0.0
        )
    else:
        repeated_opener_rate = 0.0

    # Repeated-bigram rate: fraction of adjacent word pairs that are repeats.
    bigrams = list(zip(words, words[1:]))
    if bigrams:
        repeated_bigram_rate = 1 - (len(set(bigrams)) / len(bigrams))
    else:
        repeated_bigram_rate = 0.0

    # --- Component scores (0.0-1.0, higher == more AI-like regularity) --
    # Sentence uniformity: low length variation == uniform == AI-like.
    sentence_uniformity = _clamp(
        1 - (sentence_length_cv / _SENTENCE_CV_FULL_VARIATION)
    )

    # Paragraph uniformity: same idea across paragraphs. With fewer than 2
    # paragraphs there is nothing to compare, so we return a neutral 0.5.
    if len(paragraph_lengths) >= 2:
        paragraph_cv = _coefficient_of_variation(paragraph_lengths)
        paragraph_uniformity = _clamp(
            1 - (paragraph_cv / _PARAGRAPH_CV_FULL_VARIATION)
        )
    else:
        paragraph_uniformity = 0.5

    # Repetition score: blend repeated openers and repeated bigrams. For poems
    # we damp this because deliberate repetition is a common human technique.
    repetition_score = _clamp(
        0.5 * repeated_opener_rate + 0.5 * repeated_bigram_rate
    )
    if content_type == "poem":
        repetition_score = _clamp(repetition_score * _POEM_REPETITION_DAMPING)

    # Punctuation regularity: low variation of punctuation-per-sentence ==
    # very regular == AI-like. Needs at least 2 sentences to be meaningful.
    if sentence_count >= 2:
        punct_per_sentence = [_count_punctuation(s) for s in sentences]
        punct_cv = _coefficient_of_variation(punct_per_sentence)
        punctuation_regularity = _clamp(1 - (punct_cv / _PUNCT_CV_FULL_VARIATION))
    else:
        punctuation_regularity = 0.5

    # Lexical uniformity: a low type-token ratio means vocabulary is reused,
    # i.e. more uniform. (Caveat: TTR naturally falls as texts get longer, so
    # this feature is length-sensitive — reliability weighting compensates.)
    lexical_uniformity = _clamp(1 - type_token_ratio)

    # --- Weighted stylometric AI score (weights per planning.md) -------
    ai_score = _clamp(
        0.30 * sentence_uniformity
        + 0.20 * paragraph_uniformity
        + 0.20 * repetition_score
        + 0.15 * punctuation_regularity
        + 0.15 * lexical_uniformity
    )

    reliability = _reliability_for_word_count(word_count)

    return {
        "ai_score": ai_score,
        "reliability": reliability,
        "features": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_sentence_length": average_sentence_length,
            "sentence_length_variance": sentence_length_variance,
            "sentence_length_coefficient_variation": sentence_length_cv,
            "type_token_ratio": type_token_ratio,
            "punctuation_density": punctuation_density,
            "paragraph_length_variance": paragraph_length_variance,
            "repeated_opener_rate": repeated_opener_rate,
        },
        "component_scores": {
            "sentence_uniformity": sentence_uniformity,
            "paragraph_uniformity": paragraph_uniformity,
            "repetition_score": repetition_score,
            "punctuation_regularity": punctuation_regularity,
            "lexical_uniformity": lexical_uniformity,
        },
        "signal": "stylometry",
    }
