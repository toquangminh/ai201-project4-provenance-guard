"""Milestone 3 tests for Provenance Guard.

These tests exercise the Flask API (/health, /submit, /log) and the SQLite
audit log. The Groq signal is always MOCKED so the suite never consumes a
real Groq API call. A temporary SQLite database is used so the development
database is never touched.

NOTE: Milestone 4 integrated the second (stylometric) signal and the
multi-signal scoring into POST /submit. That intentionally changed the
response contract (signals/ai_likelihood/uncertainty instead of signal_1)
and the attribution semantics (the short-text rule now forces "uncertain").
The validation, security, health, and rate-limit tests below are unchanged
from Milestone 3. The response-shape/attribution tests were updated to the
Milestone 4 contract while preserving their original intent; both signals are
mocked so outcomes stay deterministic.
"""

import os
import sys
import tempfile

import pytest

# Point the database at a throwaway temp path BEFORE importing the app, so the
# import-time init_db() call never touches the real development database.
# Each test then overrides this with its own per-test temp file (see the
# db_path autouse fixture below).
os.environ["PROVENANCE_DB_PATH"] = os.path.join(
    tempfile.gettempdir(), "provenance_import_placeholder.db"
)

# Make the project root importable when pytest is run from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import database  # noqa: E402
from detector import GroqSignalError  # noqa: E402
from labels import LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN  # noqa: E402


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    """Give every test a fresh, isolated SQLite database."""
    path = tmp_path / "test_audit.db"
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(path))
    database.init_db()
    yield str(path)


@pytest.fixture
def client():
    """Flask test client with rate limiting disabled.

    The limiter is disabled so the many /submit calls across the suite do not
    trip the 10/minute limit. Rate-limit behavior itself is verified in a
    dedicated test that re-enables it.
    """
    app_module.limiter.enabled = False
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _mock_signal(
    monkeypatch,
    ai_score=0.78,
    reliability=0.80,
    flags=None,
    style_score=None,
    style_reliability=1.0,
    word_count=200,
):
    """Patch BOTH signal functions (as imported into app) to fixed dicts.

    Mocking stylometry too keeps the integrated /submit outcome deterministic.
    style_score defaults to ai_score so the two signals agree by default.
    """
    if flags is None:
        flags = []
    if style_score is None:
        style_score = ai_score

    def fake_run_groq_signal(text, content_type="other"):
        return {
            "ai_score": ai_score,
            "reliability": reliability,
            "flags": flags,
            "signal": "groq",
        }

    def fake_run_stylometric_signal(text, content_type="other"):
        return {
            "ai_score": style_score,
            "reliability": style_reliability,
            "features": {"word_count": word_count, "sentence_count": 10},
            "component_scores": {"sentence_uniformity": style_score},
            "signal": "stylometry",
        }

    monkeypatch.setattr(app_module, "run_groq_signal", fake_run_groq_signal)
    monkeypatch.setattr(
        app_module, "run_stylometric_signal", fake_run_stylometric_signal
    )


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


# --------------------------------------------------------------------------
# /submit — happy path
# --------------------------------------------------------------------------
def test_submit_valid(client, monkeypatch):
    # Both signals agree high -> likely_ai under the Milestone 4 contract.
    _mock_signal(
        monkeypatch, ai_score=0.94, reliability=1.0, flags=["generic transitions"],
        style_score=0.89,
    )
    resp = client.post(
        "/submit",
        json={
            "text": "A reasonably long piece of creative writing to analyze.",
            "creator_id": "test-user-1",
            "content_type": "blog_post",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()

    # Attribution / likelihood / confidence / status / label present & correct.
    assert body["attribution"] == "likely_ai"
    assert 0.0 <= body["ai_likelihood"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["status"] == "classified"
    # Milestone 5: the final transparency label is returned.
    assert body["label"] == LABEL_AI
    assert body["transparency_label"] == LABEL_AI

    # Both signal blocks reflect the mocked outputs.
    assert body["signals"]["groq"]["ai_score"] == pytest.approx(0.94)
    assert body["signals"]["groq"]["flags"] == ["generic transitions"]
    assert body["signals"]["stylometry"]["ai_score"] == pytest.approx(0.89)
    assert body["signals"]["signal_gap"] == pytest.approx(0.05)
    assert body["uncertainty"]["forced"] is False


def test_submit_attribution_human(client, monkeypatch):
    _mock_signal(monkeypatch, ai_score=0.15, style_score=0.18)
    resp = client.post(
        "/submit",
        json={"text": "A personal, specific story about my grandmother.", "creator_id": "u"},
    )
    body = resp.get_json()
    assert body["attribution"] == "likely_human"
    assert body["confidence"] > 0.5
    assert body["label"] == LABEL_HUMAN


def test_submit_attribution_uncertain(client, monkeypatch):
    _mock_signal(monkeypatch, ai_score=0.50, style_score=0.50)
    resp = client.post(
        "/submit",
        json={"text": "Some ambiguous middling text here.", "creator_id": "u"},
    )
    body = resp.get_json()
    assert body["attribution"] == "uncertain"
    assert body["label"] == LABEL_UNCERTAIN


def test_submit_defaults_content_type(client, monkeypatch):
    captured = {}

    def fake(text, content_type="other"):
        captured["content_type"] = content_type
        return {"ai_score": 0.5, "reliability": 0.5, "flags": [], "signal": "groq"}

    monkeypatch.setattr(app_module, "run_groq_signal", fake)
    client.post("/submit", json={"text": "hello world text", "creator_id": "u"})
    assert captured["content_type"] == "other"


# --------------------------------------------------------------------------
# /submit — validation errors
# --------------------------------------------------------------------------
def test_submit_missing_body(client):
    # No JSON body / wrong content type -> 400.
    resp = client.post("/submit", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_submit_missing_text(client):
    resp = client.post("/submit", json={"creator_id": "u"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_submit_empty_text(client):
    resp = client.post("/submit", json={"text": "   ", "creator_id": "u"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_submit_missing_creator_id(client):
    resp = client.post("/submit", json={"text": "some text"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_submit_text_too_long(client):
    long_text = "a" * 20_001
    resp = client.post("/submit", json={"text": long_text, "creator_id": "u"})
    assert resp.status_code == 413
    assert "error" in resp.get_json()


# --------------------------------------------------------------------------
# /submit — total detection failure
# --------------------------------------------------------------------------
def test_submit_both_signals_fail_returns_503(client, monkeypatch):
    # Under Milestone 4, 503 requires BOTH signals to fail. (A single-signal
    # failure now degrades gracefully to an "uncertain" 200 response.)
    def boom_groq(text, content_type="other"):
        raise GroqSignalError("simulated groq outage")

    def boom_style(text, content_type="other"):
        raise ValueError("simulated stylometry failure")

    monkeypatch.setattr(app_module, "run_groq_signal", boom_groq)
    monkeypatch.setattr(app_module, "run_stylometric_signal", boom_style)
    resp = client.post(
        "/submit",
        json={"text": "some creative text", "creator_id": "u"},
    )
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "detection_error"
    assert "error" in body
    # No fabricated attribution/score is returned on failure.
    assert "attribution" not in body

    # A detection_error audit entry was written, with no signal scores.
    entries = database.get_log()
    assert any(e["status"] == "detection_error" for e in entries)
    err_entry = next(e for e in entries if e["status"] == "detection_error")
    assert err_entry["attribution"] is None
    assert err_entry["groq_ai_score"] is None
    assert err_entry["stylometric_ai_score"] is None


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------
def test_submit_creates_audit_entry(client, monkeypatch):
    _mock_signal(
        monkeypatch, ai_score=0.9, reliability=0.7, flags=["templated"],
        style_score=0.88,
    )
    resp = client.post(
        "/submit",
        json={"text": "uniform polished text", "creator_id": "creator-x", "content_type": "blog_post"},
    )
    content_id = resp.get_json()["content_id"]

    entries = database.get_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["content_id"] == content_id
    assert entry["creator_id"] == "creator-x"
    assert entry["content_type"] == "blog_post"
    assert entry["status"] == "classified"
    assert entry["attribution"] == "likely_ai"
    # Both signal scores and the combined score are recorded.
    assert entry["groq_ai_score"] == pytest.approx(0.9)
    assert entry["groq_reliability"] == pytest.approx(0.7)
    assert entry["stylometric_ai_score"] == pytest.approx(0.88)
    assert entry["combined_ai_score"] is not None
    # groq_flags is decoded from JSON back into a list.
    assert entry["groq_flags"] == ["templated"]
    # Raw text is never stored; a content hash is.
    assert entry["content_hash"] and len(entry["content_hash"]) == 64
    assert "uniform polished text" not in str(entry)


def test_log_returns_structured_entries_newest_first(client, monkeypatch):
    _mock_signal(monkeypatch, ai_score=0.8, style_score=0.8)
    client.post("/submit", json={"text": "first submission text", "creator_id": "a"})
    client.post("/submit", json={"text": "second submission text", "creator_id": "b"})

    resp = client.get("/log")
    assert resp.status_code == 200
    entries = resp.get_json()["entries"]
    assert len(entries) == 2
    # Newest first: the second submission (creator b) comes first.
    assert entries[0]["creator_id"] == "b"
    assert entries[1]["creator_id"] == "a"
    # Each entry is structured with the expected keys.
    for entry in entries:
        for key in ("content_id", "timestamp", "content_hash", "status", "groq_flags"):
            assert key in entry
        # JSON columns are decoded back into Python containers.
        assert isinstance(entry["groq_flags"], list)
        assert isinstance(entry["uncertainty_reasons"], list)


def test_content_id_is_unique(client, monkeypatch):
    _mock_signal(monkeypatch, ai_score=0.8)
    r1 = client.post("/submit", json={"text": "text one here", "creator_id": "a"})
    r2 = client.post("/submit", json={"text": "text two here", "creator_id": "a"})
    id1 = r1.get_json()["content_id"]
    id2 = r2.get_json()["content_id"]
    assert id1 != id2
    assert len(id1) > 0


# --------------------------------------------------------------------------
# Rate limiting (dedicated test re-enables the limiter)
# --------------------------------------------------------------------------
def test_submit_rate_limit_returns_429(monkeypatch):
    _mock_signal(monkeypatch, ai_score=0.5)
    app_module.limiter.enabled = True
    # Reset any counters accumulated in this process for a clean window.
    try:
        app_module.limiter.reset()
    except Exception:
        pass
    app_module.app.config["TESTING"] = True
    try:
        with app_module.app.test_client() as c:
            statuses = []
            for _ in range(12):
                resp = c.post("/submit", json={"text": "rate limit text", "creator_id": "a"})
                statuses.append(resp.status_code)
        # The 10/minute limit means at least one of the 12 calls is 429.
        assert 429 in statuses
        assert statuses[:10].count(200) == 10
    finally:
        app_module.limiter.enabled = False
