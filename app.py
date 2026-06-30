"""Provenance Guard Flask application.

Milestone 3 scope: the /health, /submit, and /log endpoints wired to the
Groq detection signal (Signal 1) and the SQLite audit log.

The /submit response in this milestone is PROVISIONAL. It uses only Signal 1
to produce a single-signal attribution, a placeholder confidence, and
temporary label text. The stylometric signal, reliability-weighted multi-
signal scoring, the exact transparency labels, and the appeals workflow all
arrive in later milestones (see planning.md).
"""

import datetime
import hashlib
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import database
from detector import GroqSignalError, run_groq_signal

# Maximum accepted text length (characters). Texts longer than this are
# rejected with HTTP 413 before any Groq call is made.
MAX_TEXT_LENGTH = 20_000

app = Flask(__name__)

# Rate limiter. storage_uri="memory://" keeps counters in process memory,
# which is appropriate for this single-process course project. A production
# deployment would use a shared store (e.g. Redis) so limits hold across
# multiple workers.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
)

# Ensure the audit log table exists when the app starts.
database.init_db()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the submitted text.

    We log this instead of the raw text so /log never exposes creative work.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.get("/health")
def health():
    """Liveness check."""
    return jsonify({"status": "ok"})


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    """Validate a submission, run Signal 1, log it, and return a result.

    Milestone 3 produces a single-signal (Groq-only) provisional attribution.
    """
    # --- Parse and validate the JSON body -------------------------------
    # silent=True so a missing/invalid body yields None instead of raising.
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

    # --- Generate identifiers and common metadata ----------------------
    content_id = str(uuid.uuid4())
    timestamp = _utc_now_iso()
    content_hash = _content_hash(text)

    # --- Run Detection Signal 1 (Groq) ----------------------------------
    try:
        signal_1 = run_groq_signal(text, content_type)
    except GroqSignalError as exc:
        # Graceful degradation: do not crash and do not invent a score.
        # Log a detection_error entry and return HTTP 503.
        database.log_submission(
            {
                "content_id": content_id,
                "creator_id": creator_id,
                "timestamp": timestamp,
                "content_hash": content_hash,
                "content_type": content_type,
                "attribution": None,
                "confidence": None,
                "llm_score": None,
                "llm_reliability": None,
                "signal_flags": [],
                "status": "detection_error",
            }
        )
        return (
            jsonify(
                {
                    "content_id": content_id,
                    "status": "detection_error",
                    "error": (
                        "The detection service is temporarily unavailable. "
                        "No attribution was made for this submission."
                    ),
                }
            ),
            503,
        )

    ai_score = signal_1["ai_score"]

    # --- PROVISIONAL Milestone 3 attribution (Signal 1 only) ------------
    # NOTE: This single-signal logic is a placeholder. Milestone 4 replaces it
    # with reliability-weighted multi-signal scoring, the signal-disagreement
    # rule, and the short-text rule from planning.md.
    if ai_score >= 0.70:
        attribution = "likely_ai"
        label = (
            "Preliminary result: the first detection signal leans toward "
            "AI-generated."
        )
    elif ai_score <= 0.30:
        attribution = "likely_human"
        label = (
            "Preliminary result: the first detection signal leans toward "
            "human-written."
        )
    else:
        attribution = "uncertain"
        label = "Preliminary result: the first detection signal is uncertain."

    # PROVISIONAL confidence: distance of the score from the midpoint.
    # Milestone 4 will compute confidence from the combined multi-signal score.
    confidence = max(ai_score, 1 - ai_score)

    status = "classified"

    # --- Write the audit entry BEFORE returning the response ------------
    database.log_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "content_hash": content_hash,
            "content_type": content_type,
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": ai_score,
            "llm_reliability": signal_1["reliability"],
            "signal_flags": signal_1["flags"],
            "status": status,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "status": status,
            "signal_1": {
                "name": signal_1["signal"],
                "ai_score": ai_score,
                "reliability": signal_1["reliability"],
                "flags": signal_1["flags"],
            },
        }
    )


@app.get("/log")
def log():
    """Return recent audit-log entries, newest first.

    NOTE: In a real production system this endpoint would require
    administrator authentication. It is intentionally open here for the
    course demo. Raw submitted text is never exposed (only a content hash).
    """
    return jsonify({"entries": database.get_log()})


if __name__ == "__main__":
    app.run(debug=True)
