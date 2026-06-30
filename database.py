"""SQLite persistence for Provenance Guard.

Tables:
  * audit_log       — append-only event log (submissions AND appeals)
  * content_records — current state of each submission (for /content & /appeal)
  * appeals         — appeal records (for lookup; events also go to audit_log)

Schema changes are applied additively via PRAGMA table_info + ALTER TABLE so a
pre-existing Milestone 3/4 database is upgraded in place — no table is dropped
and no row is deleted.

Privacy note: the raw submitted creative text is NEVER stored or returned. We
store a SHA-256 hash for traceability.
"""

import json
import os
import sqlite3

# Path to the SQLite database file. Overridable via env so tests can point at
# a temporary database instead of the development one.
DB_PATH = os.getenv("PROVENANCE_DB_PATH", "provenance_guard.db")

# --- audit_log schema -----------------------------------------------------
# Holds both submission ("submission") and appeal ("appeal_submitted") events.
AUDIT_COLUMNS: list[tuple[str, str]] = [
    ("event_type", "TEXT"),
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
    ("transparency_label", "TEXT"),
    ("status", "TEXT"),
    ("uncertainty_reasons", "TEXT"),  # JSON list
    # Appeal-event fields (null on submission events)
    ("appeal_id", "TEXT"),
    ("creator_reasoning", "TEXT"),
    ("evidence_description", "TEXT"),
    ("original_attribution", "TEXT"),
    ("original_confidence", "REAL"),
]

# --- content_records schema ----------------------------------------------
CONTENT_COLUMNS: list[tuple[str, str]] = [
    ("content_id", "TEXT"),
    ("creator_id", "TEXT"),
    ("content_hash", "TEXT"),
    ("content_type", "TEXT"),
    ("attribution", "TEXT"),
    ("ai_likelihood", "REAL"),
    ("confidence", "REAL"),
    ("transparency_label", "TEXT"),
    ("status", "TEXT"),
    ("created_at", "TEXT"),
    ("updated_at", "TEXT"),
]

# --- appeals schema -------------------------------------------------------
APPEAL_COLUMNS: list[tuple[str, str]] = [
    ("appeal_id", "TEXT"),
    ("content_id", "TEXT"),
    ("creator_id", "TEXT"),
    ("creator_reasoning", "TEXT"),
    ("evidence_description", "TEXT"),
    ("original_attribution", "TEXT"),
    ("original_confidence", "REAL"),
    ("status", "TEXT"),
    ("created_at", "TEXT"),
]

_TABLES: dict[str, list[tuple[str, str]]] = {
    "audit_log": AUDIT_COLUMNS,
    "content_records": CONTENT_COLUMNS,
    "appeals": APPEAL_COLUMNS,
}

# audit_log columns whose values are stored as JSON text.
_AUDIT_JSON_COLUMNS = {
    "groq_flags",
    "stylometric_features",
    "stylometric_component_scores",
    "uncertainty_reasons",
}
# Columns decoded from JSON when reading the log (includes the legacy
# Milestone 3 "signal_flags" column for backward compatibility).
_JSON_DECODE_COLUMNS = _AUDIT_JSON_COLUMNS | {"signal_flags"}


def _get_connection() -> sqlite3.Connection:
    """Open a connection to the configured SQLite database.

    Reads the path at call time (not import time) so tests that set
    PROVENANCE_DB_PATH before calling are respected.
    """
    path = os.getenv("PROVENANCE_DB_PATH", DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names currently on a table."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def _create_and_migrate(conn: sqlite3.Connection, table: str, columns) -> None:
    """Create a table if absent, then additively add any missing columns."""
    column_defs = ",\n                ".join(
        f"{name} {sql_type}" for name, sql_type in columns
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {column_defs}
        )
        """
    )
    existing = _existing_columns(conn, table)
    for name, sql_type in columns:
        if name not in existing:
            # ALTER TABLE ADD COLUMN only appends nullable columns — exactly
            # what we want when backfilling an older schema.
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def init_db() -> None:
    """Create/migrate every table to the latest schema (data-preserving)."""
    conn = _get_connection()
    try:
        for table, columns in _TABLES.items():
            _create_and_migrate(conn, table, columns)
        conn.commit()
    finally:
        conn.close()


def _encode(value) -> str | None:
    """JSON-encode a list/dict for storage; pass None/str through."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


# ==========================================================================
# audit_log
# ==========================================================================
def log_submission(entry: dict) -> None:
    """Insert one audit event (submission OR appeal).

    Built generically from AUDIT_COLUMNS so adding a column does not require
    editing this function. JSON-typed fields are encoded to text. Missing
    fields default to None, so a detection_error or appeal event can be logged
    even though it does not populate every column.
    """
    columns = [name for name, _ in AUDIT_COLUMNS]
    values = []
    for name in columns:
        value = entry.get(name)
        if name in _AUDIT_JSON_COLUMNS:
            value = _encode(value)
        values.append(value)

    placeholders = ", ".join("?" for _ in columns)
    conn = _get_connection()
    try:
        conn.execute(
            f"INSERT INTO audit_log ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(values),
        )
        conn.commit()
    finally:
        conn.close()


# Appeal events are the same kind of audit row; this alias documents intent.
log_event = log_submission


def get_log(limit: int = 20) -> list[dict]:
    """Return the most recent audit entries, newest first.

    JSON-text columns are decoded back into Python lists/dicts. Both individual
    signal scores and the final combined result are exposed; appeal events are
    interleaved by recency. The raw submitted text is never present.
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
        for key in _JSON_DECODE_COLUMNS:
            if key in entry and entry[key] is not None:
                try:
                    entry[key] = json.loads(entry[key])
                except (json.JSONDecodeError, TypeError):
                    pass  # leave undecodable legacy values as-is
        entries.append(entry)
    return entries


# ==========================================================================
# content_records
# ==========================================================================
def save_content_record(record: dict) -> None:
    """Insert or update the current-state record for a submission.

    Keyed on content_id. On update, created_at is preserved and only the
    mutable fields (status, scores, label, updated_at) change.
    """
    content_id = record.get("content_id")
    existing = get_content_record(content_id) if content_id else None

    conn = _get_connection()
    try:
        if existing is None:
            cols = [name for name, _ in CONTENT_COLUMNS]
            placeholders = ", ".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO content_records ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                tuple(record.get(name) for name in cols),
            )
        else:
            # Update everything except the immutable content_id/created_at.
            updatable = [
                name
                for name, _ in CONTENT_COLUMNS
                if name not in ("content_id", "created_at")
            ]
            set_clause = ", ".join(f"{name} = ?" for name in updatable)
            params = [record.get(name) for name in updatable]
            params.append(content_id)
            conn.execute(
                f"UPDATE content_records SET {set_clause} WHERE content_id = ?",
                tuple(params),
            )
        conn.commit()
    finally:
        conn.close()


def get_content_record(content_id: str) -> dict | None:
    """Return the content record for content_id, or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM content_records WHERE content_id = ?",
            (content_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def update_content_status(content_id: str, status: str) -> None:
    """Update a content record's status and bump updated_at."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE content_records "
            "SET status = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE content_id = ?",
            (status, content_id),
        )
        conn.commit()
    finally:
        conn.close()


# ==========================================================================
# appeals
# ==========================================================================
def create_appeal(appeal: dict) -> None:
    """Insert an appeal record. (Automated reclassification is NOT performed.)"""
    cols = [name for name, _ in APPEAL_COLUMNS]
    placeholders = ", ".join("?" for _ in cols)
    conn = _get_connection()
    try:
        conn.execute(
            f"INSERT INTO appeals ({', '.join(cols)}) VALUES ({placeholders})",
            tuple(appeal.get(name) for name in cols),
        )
        conn.commit()
    finally:
        conn.close()


def get_appeal_for_content(content_id: str) -> dict | None:
    """Return the most recent appeal for content_id, or None if none exists."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM appeals WHERE content_id = ? ORDER BY id DESC LIMIT 1",
            (content_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None
