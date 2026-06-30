"""Detection signals for Provenance Guard.

Milestone 3 implements Detection Signal 1 only: a Groq LLM classification
of whether submitted creative text leans toward AI generation or human
authorship. The stylometric signal (Signal 2) and the multi-signal scoring
described in planning.md belong to later milestones.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables (e.g. GROQ_API_KEY) from a local .env file.
# The .env file is gitignored and must never be committed or logged.
load_dotenv()

# Groq model used for the holistic LLM classification signal.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Content types the prompt understands. Anything else falls back to "other"
# so a poem is not judged with the same expectations as a blog post.
KNOWN_CONTENT_TYPES = {"poem", "short_story", "blog_post", "other"}


class GroqSignalError(Exception):
    """Raised when the Groq detection signal cannot produce a real result.

    The application layer is expected to catch this, log a detection_error
    audit entry, and return a controlled HTTP 503 response. We never silently
    substitute a fabricated score when the signal is unavailable.
    """


def _clamp(value: float) -> float:
    """Clamp a numeric score into the inclusive range 0.0-1.0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        # A missing/garbage numeric field is treated as "no signal" (0.0)
        # rather than crashing; this only affects the individual field, not
        # whether we report a result at all.
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _extract_json(raw: str) -> dict:
    """Parse a JSON object from a model response.

    Handles common LLM quirks: Markdown code fences (```json ... ```),
    surrounding prose, and leading/trailing whitespace. Raises ValueError
    if no JSON object can be recovered.
    """
    if raw is None:
        raise ValueError("empty model response")

    text = raw.strip()

    # Strip Markdown code fences if present, e.g. ```json\n{...}\n```
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing fence.
        lines = text.splitlines()
        # Remove first line (the opening fence).
        lines = lines[1:]
        # Remove a trailing fence line if present.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # First, try parsing the whole (de-fenced) string directly.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the first {...} block from surrounding prose.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("could not locate a JSON object in the model response")


def _build_prompt(text: str, content_type: str) -> str:
    """Build the user prompt, embedding the content type and the text."""
    return (
        f"Content type: {content_type}\n\n"
        "Analyze the following creative text and assess whether it shows "
        "stronger signals of AI generation or of human authorship. Consider "
        "coherence, voice consistency, generic or repetitive transitions, "
        "predictable organization, overly balanced phrasing, personal "
        "specificity, and whether the writing feels templated.\n\n"
        "Calibrate your expectations to the stated content type: a poem is "
        "not judged with the same structural expectations as a blog post or "
        "a short story.\n\n"
        "Respond with STRICT JSON only, no prose and no Markdown fences, "
        "matching exactly this shape:\n"
        '{"ai_score": 0.0, "reliability": 0.0, "flags": []}\n\n'
        "Where:\n"
        "- ai_score is a float 0.0-1.0 (0.0 = strongly human, 1.0 = strongly "
        "AI-generated).\n"
        "- reliability is a float 0.0-1.0 expressing how confident you are "
        "given the length and nature of the text.\n"
        "- flags is a list of short string observations supporting your "
        "assessment.\n\n"
        "TEXT TO ANALYZE:\n"
        f"{text}"
    )


def run_groq_signal(text: str, content_type: str = "other") -> dict:
    """Run the Groq LLM detection signal on a piece of creative text.

    Args:
        text: The raw creative text to analyze.
        content_type: One of "poem", "short_story", "blog_post", "other".
            Unknown values are normalized to "other".

    Returns:
        {
            "ai_score": float,      # 0.0-1.0, clamped
            "reliability": float,   # 0.0-1.0, clamped
            "flags": list[str],
            "signal": "groq",
        }

    Raises:
        GroqSignalError: if the API key is missing, the Groq call fails, or
            the model response cannot be parsed into the required structure.
            We never return a fabricated score on failure.
    """
    # Normalize the content type so the prompt always describes a known genre.
    if content_type not in KNOWN_CONTENT_TYPES:
        content_type = "other"

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Controlled, descriptive error. Note: we report that the key is
        # missing but never include the key value anywhere.
        raise GroqSignalError(
            "GROQ_API_KEY is not set; cannot run the Groq detection signal."
        )

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful, conservative AI-content detection "
                        "assistant. AI-content detection is probabilistic and "
                        "you must avoid overconfident claims. You always reply "
                        "with strict JSON only."
                    ),
                },
                {"role": "user", "content": _build_prompt(text, content_type)},
            ],
            temperature=0.0,
            # Ask the API for a JSON object response where supported.
            response_format={"type": "json_object"},
        )
    except GroqSignalError:
        raise
    except Exception as exc:  # noqa: BLE001 - we surface a controlled error
        # Wrap any client/transport/API error in our own exception type.
        # We deliberately do not echo the raw exception verbatim if it could
        # contain request internals; a short message is enough for the caller.
        raise GroqSignalError(f"Groq API call failed: {exc}") from exc

    try:
        raw_content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise GroqSignalError(
            "Groq API returned an unexpected response structure."
        ) from exc

    try:
        parsed = _extract_json(raw_content)
    except (ValueError, json.JSONDecodeError) as exc:
        raise GroqSignalError(
            f"Could not parse JSON from the Groq response: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise GroqSignalError("Groq response JSON was not an object.")

    # Normalize flags to a list of strings.
    flags = parsed.get("flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(flag) for flag in flags]

    return {
        "ai_score": _clamp(parsed.get("ai_score")),
        "reliability": _clamp(parsed.get("reliability")),
        "flags": flags,
        "signal": "groq",
    }
