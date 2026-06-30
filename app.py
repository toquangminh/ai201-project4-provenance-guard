"""Provenance Guard Flask application.

Milestone 5 adds the production layer on top of the Milestone 3/4 detection
pipeline:
  * final transparency labels (labels.get_transparency_label)
  * persistent content records
  * the appeal workflow (POST /appeal -> status "under_review")
  * GET /content/<content_id>
  * appeal rate limits + a JSON 429 handler

Preserved from earlier milestones: request validation (400/413), the two-signal
detection pipeline (Groq + stylometry), the uncertainty/confidence rules,
graceful detection-failure handling, /submit rate limiting, the data-preserving
SQLite migration, and no-raw-text-in-/log.
"""

import datetime
import hashlib
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import database
from detector import GroqSignalError, run_groq_signal
from labels import get_transparency_label
from scoring import ScoringError, combine_signals
from stylometry import run_stylometric_signal

# Maximum accepted submission text length (characters).
MAX_TEXT_LENGTH = 20_000
# Maximum accepted appeal reasoning length (characters).
MAX_REASONING_LENGTH = 5_000

app = Flask(__name__)

# Rate limiter. storage_uri="memory://" keeps counters in process memory, which
# suits this single-process course project; production would use a shared store
# (e.g. Redis). No default limit is set, so GET /health, /log, and
# /content/<id> are NOT rate limited — only the explicitly decorated routes.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
)

# Ensure all tables exist and are migrated to the latest schema.
database.init_db()


@app.errorhandler(429)
def handle_rate_limit(_exc):
    """Return JSON (not HTML/500) when a rate limit is exceeded."""
    return (
        jsonify(
            {
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
            }
        ),
        429,
    )


@app.errorhandler(500)
def handle_internal_error(_exc):
    """Never leak a Python traceback to the client."""
    return (
        jsonify({"error": "internal_error", "message": "An unexpected error occurred."}),
        500,
    )


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the submitted text (logged instead of text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signal_block(result: dict | None, *, include_details: bool) -> dict:
    """Shape one signal's output for the response (null fields if it failed)."""
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
    """Validate, run both signals, combine, label, persist, and respond."""
    # --- Parse and validate the JSON body -------------------------------
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

    # --- Identifiers and metadata ---------------------------------------
    content_id = str(uuid.uuid4())
    timestamp = _utc_now_iso()
    content_hash = _content_hash(text)

    # --- Run both detection signals independently -----------------------
    groq_result: dict | None = None
    stylometric_result: dict | None = None

    try:
        groq_result = run_groq_signal(text, content_type)
    except GroqSignalError:
        groq_result = None

    try:
        stylometric_result = run_stylometric_signal(text, content_type)
    except Exception:  # noqa: BLE001 - any stylometry failure is non-fatal
        stylometric_result = None

    # --- Both signals failed: controlled 503 ----------------------------
    if groq_result is None and stylometric_result is None:
        database.log_submission(
            {
                "event_type": "submission",
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
        database.log_submission(
            {
                "event_type": "submission",
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
    confidence = decision["confidence"]
    ai_likelihood = decision["ai_likelihood"]
    # Final, exact transparency label (Milestone 5).
    transparency_label = get_transparency_label(attribution, confidence)
    status = "classified"

    # --- Audit log: full submission event -------------------------------
    database.log_submission(
        {
            "event_type": "submission",
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "content_hash": content_hash,
            "content_type": content_type,
            "groq_ai_score": groq_result["ai_score"] if groq_result else None,
            "groq_reliability": groq_result["reliability"] if groq_result else None,
            "groq_flags": groq_result["flags"] if groq_result else None,
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
            "signal_gap": decision["signal_gap"],
            "combined_ai_score": ai_likelihood,
            "confidence": confidence,
            "attribution": attribution,
            "transparency_label": transparency_label,
            "status": status,
            "uncertainty_reasons": decision["uncertainty_reasons"],
        }
    )

    # --- Persist the current-state content record (for /content & /appeal)
    database.save_content_record(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "content_hash": content_hash,
            "content_type": content_type,
            "attribution": attribution,
            "ai_likelihood": ai_likelihood,
            "confidence": confidence,
            "transparency_label": transparency_label,
            "status": status,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "ai_likelihood": ai_likelihood,
            "confidence": confidence,
            "transparency_label": transparency_label,
            # `label` is kept for backward compatibility and carries the SAME
            # exact transparency text.
            "label": transparency_label,
            "status": status,
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


@app.post("/appeal")
@limiter.limit("3 per hour;10 per day")
def appeal():
    """Record a creator appeal and move the content to 'under_review'.

    The original attribution and confidence are preserved; we never
    automatically reverse a classification (a human reviewer would).
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")
    provided_creator_id = data.get("creator_id")
    evidence_description = data.get("evidence_description")

    if not content_id or not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "Field 'content_id' is required."}), 400

    if (
        not creator_reasoning
        or not isinstance(creator_reasoning, str)
        or not creator_reasoning.strip()
    ):
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    if len(creator_reasoning) > MAX_REASONING_LENGTH:
        return (
            jsonify(
                {
                    "error": (
                        f"Field 'creator_reasoning' exceeds the maximum length "
                        f"of {MAX_REASONING_LENGTH} characters."
                    )
                }
            ),
            400,
        )

    # 5. Content must exist.
    record = database.get_content_record(content_id)
    if record is None:
        return jsonify({"error": "No content found for that content_id."}), 404

    # 6-7. If a creator_id is provided, it must match the original submission.
    if provided_creator_id is not None and provided_creator_id != record.get("creator_id"):
        return (
            jsonify({"error": "creator_id does not match the original submission."}),
            403,
        )

    # 8. Reject a duplicate appeal or content already under review.
    if (
        record.get("status") == "under_review"
        or database.get_appeal_for_content(content_id) is not None
    ):
        return (
            jsonify({"error": "An appeal already exists for this content."}),
            409,
        )

    # 9-11. Create the appeal, preserving the original decision.
    appeal_id = str(uuid.uuid4())
    timestamp = _utc_now_iso()
    original_attribution = record.get("attribution")
    original_confidence = record.get("confidence")

    database.create_appeal(
        {
            "appeal_id": appeal_id,
            "content_id": content_id,
            "creator_id": record.get("creator_id"),
            "creator_reasoning": creator_reasoning,
            "evidence_description": evidence_description,
            "original_attribution": original_attribution,
            "original_confidence": original_confidence,
            "status": "under_review",
            "created_at": timestamp,
        }
    )

    # 12. Move the content into review (original decision is NOT overwritten).
    database.update_content_status(content_id, "under_review")

    # 13. Write the appeal event into the audit log.
    database.log_event(
        {
            "event_type": "appeal_submitted",
            "appeal_id": appeal_id,
            "content_id": content_id,
            "creator_id": record.get("creator_id"),
            "creator_reasoning": creator_reasoning,
            "evidence_description": evidence_description,
            "original_attribution": original_attribution,
            "original_confidence": original_confidence,
            "status": "under_review",
            "timestamp": timestamp,
        }
    )

    # 14.
    return (
        jsonify(
            {
                "appeal_id": appeal_id,
                "content_id": content_id,
                "status": "under_review",
                "message": (
                    "Your appeal has been recorded and the content is now "
                    "under review."
                ),
            }
        ),
        201,
    )


@app.get("/content/<content_id>")
def content(content_id):
    """Return the current state of a content record (404 if unknown)."""
    record = database.get_content_record(content_id)
    if record is None:
        return jsonify({"error": "No content found for that content_id."}), 404

    has_appeal = database.get_appeal_for_content(content_id) is not None
    return jsonify(
        {
            "content_id": record["content_id"],
            "attribution": record["attribution"],
            "ai_likelihood": record["ai_likelihood"],
            "confidence": record["confidence"],
            "transparency_label": record["transparency_label"],
            "status": record["status"],
            "has_appeal": has_appeal,
        }
    )


@app.get("/log")
def log():
    """Return recent audit-log entries, newest first (submissions + appeals).

    NOTE: In a real production system this endpoint would require
    administrator authentication. It is intentionally open here for the
    course demo. Raw submitted text is never exposed (only a content hash),
    and the Groq API key is never present in any record.
    """
    return jsonify({"entries": database.get_log()})


if __name__ == "__main__":
    app.run(debug=True)
