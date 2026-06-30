"""Provenance Guard Flask application.

Milestone 4 wires BOTH detection signals into POST /submit:
  * Signal 1 (Groq LLM) via detector.run_groq_signal
  * Signal 2 (stylometry) via stylometry.run_stylometric_signal
and combines them with scoring.combine_signals to produce a single
attribution, AI likelihood, and confidence.

The transparency labels here are TEMPORARY Milestone 4 text. The final
reader-facing labels and the appeals workflow arrive in Milestone 5.

Milestone 3 behavior preserved: validation (400/413), the health endpoint,
rate limiting, no-raw-text-in-/log, and graceful detection failure handling.
"""

import datetime
import hashlib
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import database
from detector import GroqSignalError, run_groq_signal
from scoring import ScoringError, combine_signals
from stylometry import run_stylometric_signal

# Maximum accepted text length (characters). Texts longer than this are
# rejected with HTTP 413 before any signal runs.
MAX_TEXT_LENGTH = 20_000

# Temporary Milestone 4 labels (replaced by the final transparency labels in
# Milestone 5).
LABELS = {
    "likely_ai": "Multi-signal result: this text shows strong AI-generation signals.",
    "likely_human": "Multi-signal result: this text shows strong human-authorship signals.",
    "uncertain": "Multi-signal result: the system cannot confidently determine attribution.",
}

app = Flask(__name__)

# Rate limiter. storage_uri="memory://" keeps counters in process memory,
# which suits this single-process course project. Production would use a
# shared store (e.g. Redis) so limits hold across workers.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
)

# Ensure the audit log table exists and is migrated to the latest schema.
database.init_db()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the submitted text.

    We log this instead of the raw text so /log never exposes creative work.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signal_block(result: dict | None, *, include_details: bool) -> dict:
    """Shape a single signal's output for the response.

    Returns null-valued fields when the signal failed/was missing so the
    response shape stays stable. Never fabricates a score.
    """
    if include_details:
        if result is None:
            return {
                "ai_score": None,
                "reliability": None,
                "features": {},
                "component_scores": {},
                "failed": True,
            }
        return {
            "ai_score": result.get("ai_score"),
            "reliability": result.get("reliability"),
            "features": result.get("features", {}),
            "component_scores": result.get("component_scores", {}),
        }
    # Groq block (no features/component_scores).
    if result is None:
        return {"ai_score": None, "reliability": None, "flags": [], "failed": True}
    return {
        "ai_score": result.get("ai_score"),
        "reliability": result.get("reliability"),
        "flags": result.get("flags", []),
    }


@app.get("/health")
def health():
    """Liveness check."""
    return jsonify({"status": "ok"})


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    """Validate a submission, run both signals, combine, log, and respond."""
    # --- Parse and validate the JSON body (unchanged from Milestone 3) --
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    text = data.get("text")
    creator_id = data.get("creator_id")
    content_type = data.get("content_type") or "other"

    if text is None:
        return jsonify({"error": "Field 'text' is required."}), 400
    if not isinstance(text, str):
        return jsonify({"error": "Field 'text' must be a string."}), 400
    if not text.strip():
        return jsonify({"error": "Field 'text' must not be empty or whitespace."}), 400

    if creator_id is None or not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return (
            jsonify(
                {
                    "error": (
                        f"Field 'text' exceeds the maximum length of "
                        f"{MAX_TEXT_LENGTH} characters."
                    )
                }
            ),
            413,
        )

    # --- Identifiers and common metadata --------------------------------
    content_id = str(uuid.uuid4())
    timestamp = _utc_now_iso()
    content_hash = _content_hash(text)

    # --- Run both detection signals independently -----------------------
    # Each signal can fail without crashing the request. We never substitute a
    # fabricated score for a failed signal; combine_signals handles missingness.
    groq_result: dict | None = None
    stylometric_result: dict | None = None

    try:
        groq_result = run_groq_signal(text, content_type)
    except GroqSignalError:
        groq_result = None  # recorded below via null columns + uncertainty

    try:
        stylometric_result = run_stylometric_signal(text, content_type)
    except Exception:  # noqa: BLE001 - any stylometry failure is non-fatal
        # Stylometry is pure-Python and should not normally fail on validated
        # (non-empty) text, but we degrade gracefully just in case.
        stylometric_result = None

    # --- Both signals failed: controlled 503 ----------------------------
    if groq_result is None and stylometric_result is None:
        database.log_submission(
            {
                "content_id": content_id,
                "creator_id": creator_id,
                "timestamp": timestamp,
                "content_hash": content_hash,
                "content_type": content_type,
                "status": "detection_error",
            }
        )
        return (
            jsonify(
                {
                    "content_id": content_id,
                    "status": "detection_error",
                    "error": (
                        "Both detection signals are unavailable. No attribution "
                        "was made for this submission."
                    ),
                }
            ),
            503,
        )

    # --- Combine the available signal(s) --------------------------------
    try:
        decision = combine_signals(groq_result, stylometric_result)
    except ScoringError:
        # Defensive: combine_signals only raises when BOTH are missing, which
        # we already handled above. Treat any other case as a service error.
        database.log_submission(
            {
                "content_id": content_id,
                "creator_id": creator_id,
                "timestamp": timestamp,
                "content_hash": content_hash,
                "content_type": content_type,
                "status": "detection_error",
            }
        )
        return (
            jsonify(
                {
                    "content_id": content_id,
                    "status": "detection_error",
                    "error": "Unable to score this submission.",
                }
            ),
            503,
        )

    attribution = decision["attribution"]
    label = LABELS[attribution]
    status = "classified"

    # --- Write the audit entry BEFORE returning -------------------------
    database.log_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "content_hash": content_hash,
            "content_type": content_type,
            # Signal 1 (None if Groq failed — logs the failure).
            "groq_ai_score": groq_result["ai_score"] if groq_result else None,
            "groq_reliability": groq_result["reliability"] if groq_result else None,
            "groq_flags": groq_result["flags"] if groq_result else None,
            # Signal 2 (None if stylometry failed).
            "stylometric_ai_score": (
                stylometric_result["ai_score"] if stylometric_result else None
            ),
            "stylometric_reliability": (
                stylometric_result["reliability"] if stylometric_result else None
            ),
            "stylometric_features": (
                stylometric_result["features"] if stylometric_result else None
            ),
            "stylometric_component_scores": (
                stylometric_result["component_scores"] if stylometric_result else None
            ),
            # Combined decision.
            "signal_gap": decision["signal_gap"],
            "combined_ai_score": decision["ai_likelihood"],
            "confidence": decision["confidence"],
            "attribution": attribution,
            "status": status,
            "uncertainty_reasons": decision["uncertainty_reasons"],
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "ai_likelihood": decision["ai_likelihood"],
            "confidence": decision["confidence"],
            "status": status,
            "label": label,
            "signals": {
                "groq": _signal_block(groq_result, include_details=False),
                "stylometry": _signal_block(stylometric_result, include_details=True),
                "signal_gap": decision["signal_gap"],
            },
            "uncertainty": {
                "forced": decision["forced_uncertain"],
                "reasons": decision["uncertainty_reasons"],
            },
        }
    )


@app.get("/log")
def log():
    """Return recent audit-log entries, newest first.

    NOTE: In a real production system this endpoint would require
    administrator authentication. It is intentionally open here for the
    course demo. Raw submitted text is never exposed (only a content hash).
    Both individual signal scores and the final combined result are included.
    """
    return jsonify({"entries": database.get_log()})


if __name__ == "__main__":
    app.run(debug=True)
