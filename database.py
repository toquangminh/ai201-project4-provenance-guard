"""SQLite audit logging for Provenance Guard.

Milestone 4 expands the attribution audit log to record BOTH detection signals
(Groq + stylometry) and the combined decision. A safe, additive schema
migration upgrades any pre-existing Milestone 3 table without dropping data.

Privacy note: the raw submitted creative text is NEVER stored or returned
through /log. We store a SHA-256 hash of the content for traceability.
"""

import json
import os
import sqlite3

# Path to the SQLite database file. Overridable via env so tests can point at
# a temporary database instead of the development one.
DB_PATH = os.getenv("PROVENANCE_DB_PATH", "provenance_guard.db")

# Desired audit_log columns (besides the autoincrement id), as
# (name, sql_type). init_db creates this table; _migrate adds any of these
# columns that are missing from a pre-existing table.
AUDIT_COLUMNS: list[tuple[str, str]] = [
    ("content_id", "TEXT"),
    ("creator_id", "TEXT"),
    ("timestamp", "TEXT"),
    ("content_hash", "TEXT"),
    ("content_type", "TEXT"),
    # Signal 1: Groq
    ("groq_ai_score", "REAL"),
    ("groq_reliability", "REAL"),
    ("groq_flags", "TEXT"),  # JSON list
    # Signal 2: stylometry
    ("stylometric_ai_score", "REAL"),
    ("stylometric_reliability", "REAL"),
    ("stylometric_features", "TEXT"),  # JSON dict
    ("stylometric_component_scores", "TEXT"),  # JSON dict
    # Combined decision
    ("signal_gap", "REAL"),
    ("combined_ai_score", "REAL"),
    ("confidence", "REAL"),
    ("attribution", "TEXT"),
    ("status", "TEXT"),
    ("uncertainty_reasons", "TEXT"),  # JSON list
]

# Columns whose values are stored as JSON text and must be decoded on read.
# signal_flags is a legacy Milestone 3 column kept for backward compatibility
# with rows written before this migration.
_JSON_COLUMNS = {
    "groq_flags",
    "stylometric_features",
    "stylometric_component_scores",
    "uncertainty_reasons",
    "signal_flags",
}


def _get_connection() -> sqlite3.Connection:
    """Open a connection to the configured SQLite database.

    Reads the path at call time (not import time) so tests that set
    PROVENANCE_DB_PATH before calling are respected.
    """
    path = os.getenv("PROVENANCE_DB_PATH", DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    """Return the set of column names currently on the audit_log table."""
    rows = conn.execute("PRAGMA table_info(audit_log)").fetchall()
    return {row["name"] for row in rows}


def _migrate(conn: sqlite3.Connection) -> None:
    """Additively add any missing audit_log columns.

    Uses PRAGMA table_info to inspect the existing schema and ALTER TABLE to
    add only the columns that are absent. Never drops columns or rows, so an
    existing Milestone 3 database is upgraded in place with its data intact.
    """
    existing = _existing_columns(conn)
    for name, sql_type in AUDIT_COLUMNS:
        if name not in existing:
            # SQLite ALTER TABLE ADD COLUMN only appends nullable columns,
            # which is exactly what we want for backfilling old rows.
            conn.execute(f"ALTER TABLE audit_log ADD COLUMN {name} {sql_type}")


def init_db() -> None:
    """Create the audit_log table if needed, then migrate to the latest schema."""
    conn = _get_connection()
    try:
        # Build the CREATE statement from AUDIT_COLUMNS so it stays in sync.
        column_defs = ",\n                ".join(
            f"{name} {sql_type}" for name, sql_type in AUDIT_COLUMNS
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {column_defs}
            )
            """
        )
        # Upgrade any pre-existing table (e.g. a Milestone 3 schema) in place.
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _encode(value) -> str | None:
    """JSON-encode a list/dict for storage; pass None through untouched."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def log_submission(entry: dict) -> None:
    """Insert one structured audit entry.

    Dict/list fields (groq_flags, stylometric_features,
    stylometric_component_scores, uncertainty_reasons) are stored as JSON text.
    Missing fields default to None so a detection_error entry (no signals) can
    still be logged.
    """
    values = {
        "content_id": entry.get("content_id"),
        "creator_id": entry.get("creator_id"),
        "timestamp": entry.get("timestamp"),
        "content_hash": entry.get("content_hash"),
        "content_type": entry.get("content_type"),
        "groq_ai_score": entry.get("groq_ai_score"),
        "groq_reliability": entry.get("groq_reliability"),
        "groq_flags": _encode(entry.get("groq_flags")),
        "stylometric_ai_score": entry.get("stylometric_ai_score"),
        "stylometric_reliability": entry.get("stylometric_reliability"),
        "stylometric_features": _encode(entry.get("stylometric_features")),
        "stylometric_component_scores": _encode(
            entry.get("stylometric_component_scores")
        ),
        "signal_gap": entry.get("signal_gap"),
        "combined_ai_score": entry.get("combined_ai_score"),
        "confidence": entry.get("confidence"),
        "attribution": entry.get("attribution"),
        "status": entry.get("status"),
        "uncertainty_reasons": _encode(entry.get("uncertainty_reasons")),
    }

    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)

    conn = _get_connection()
    try:
        conn.execute(
            f"INSERT INTO audit_log ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        conn.commit()
    finally:
        conn.close()


def get_log(limit: int = 20) -> list[dict]:
    """Return the most recent audit entries, newest first.

    JSON-text columns are decoded back into Python lists/dicts. Both the
    individual signal scores and the final combined result are exposed. The raw
    submitted text is never present (only content_hash is stored).
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    entries = []
    for row in rows:
        entry = dict(row)
        for key in _JSON_COLUMNS:
            if key in entry and entry[key] is not None:
                try:
                    entry[key] = json.loads(entry[key])
                except (json.JSONDecodeError, TypeError):
                    # Leave undecodable legacy values as-is rather than crash.
                    pass
        entries.append(entry)
    return entries
