"""SQLite audit logging for Provenance Guard.

Milestone 3 implements a structured attribution audit log using Python's
built-in sqlite3 module. The appeal table and appeal fields described in
planning.md belong to a later milestone and are intentionally not created
here.

Privacy note: the raw submitted creative text is NEVER stored or returned
through /log. We store a SHA-256 hash of the content for traceability.
"""

import json
import os
import sqlite3

# Path to the SQLite database file. Overridable via env so tests can point at
# a temporary database instead of the development one.
DB_PATH = os.getenv("PROVENANCE_DB_PATH", "provenance_guard.db")


def _get_connection() -> sqlite3.Connection:
    """Open a connection to the configured SQLite database.

    Reads DB_PATH at call time (not import time) so tests that set
    PROVENANCE_DB_PATH before calling are respected.
    """
    path = os.getenv("PROVENANCE_DB_PATH", DB_PATH)
    conn = sqlite3.connect(path)
    # Return rows that behave like dicts for convenient column access.
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the audit_log table if it does not already exist."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id      TEXT NOT NULL,
                creator_id      TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                content_hash    TEXT NOT NULL,
                content_type    TEXT NOT NULL,
                attribution     TEXT,
                confidence      REAL,
                llm_score       REAL,
                llm_reliability REAL,
                signal_flags    TEXT,
                status          TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_submission(entry: dict) -> None:
    """Insert a single structured audit entry.

    Expected keys in `entry`:
        content_id, creator_id, timestamp, content_hash, content_type,
        attribution, confidence, llm_score, llm_reliability, signal_flags,
        status

    signal_flags may be a list (it is stored as JSON text). Missing optional
    fields default to None so that detection_error entries (which have no
    score/attribution) can still be logged.
    """
    # signal_flags is stored as JSON text so the list round-trips cleanly.
    flags = entry.get("signal_flags", [])
    if not isinstance(flags, str):
        flags = json.dumps(flags if flags is not None else [])

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO audit_log (
                content_id, creator_id, timestamp, content_hash,
                content_type, attribution, confidence, llm_score,
                llm_reliability, signal_flags, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("content_id"),
                entry.get("creator_id"),
                entry.get("timestamp"),
                entry.get("content_hash"),
                entry.get("content_type"),
                entry.get("attribution"),
                entry.get("confidence"),
                entry.get("llm_score"),
                entry.get("llm_reliability"),
                flags,
                entry.get("status"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_log(limit: int = 20) -> list[dict]:
    """Return the most recent audit entries, newest first.

    signal_flags is decoded from JSON text back into a Python list. The raw
    submitted text is never present here (only content_hash is stored).
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, content_id, creator_id, timestamp, content_hash,
                   content_type, attribution, confidence, llm_score,
                   llm_reliability, signal_flags, status
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    entries = []
    for row in rows:
        entry = dict(row)
        # Decode signal_flags JSON text back into a list for the response.
        raw_flags = entry.get("signal_flags")
        try:
            entry["signal_flags"] = json.loads(raw_flags) if raw_flags else []
        except (json.JSONDecodeError, TypeError):
            entry["signal_flags"] = []
        entries.append(entry)
    return entries
