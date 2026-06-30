"""Tests for the appeal workflow, content records, and the audit log.

Both detection signals are MOCKED so the suite is deterministic and never
consumes a real Groq API call. A temporary SQLite database is used.
"""

import os
import sys
import tempfile

import pytest

os.environ["PROVENANCE_DB_PATH"] = os.path.join(
    tempfile.gettempdir(), "provenance_appeals_import_placeholder.db"
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import database  # noqa: E402


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test_appeals.db"
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(path))
    database.init_db()
    yield str(path)


@pytest.fixture
def client():
    app_module.limiter.enabled = False
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _mock_both(monkeypatch, ai=0.94, style=0.89, word_count=200):
    def fake_groq(text, content_type="other"):
        return {"ai_score": ai, "reliability": 1.0, "flags": ["x"], "signal": "groq"}

    def fake_style(text, content_type="other"):
        return {
            "ai_score": style,
            "reliability": 1.0,
            "features": {"word_count": word_count},
            "component_scores": {"sentence_uniformity": style},
            "signal": "stylometry",
        }

    monkeypatch.setattr(app_module, "run_groq_signal", fake_groq)
    monkeypatch.setattr(app_module, "run_stylometric_signal", fake_style)


def _submit(client, creator="creator-1", text="A sufficiently long creative passage."):
    return client.post(
        "/submit",
        json={"text": text, "creator_id": creator, "content_type": "blog_post"},
    )


# --------------------------------------------------------------------------
# Content records / GET /content
# --------------------------------------------------------------------------
def test_content_record_created_on_submit(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client).get_json()["content_id"]

    resp = client.get(f"/content/{content_id}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["content_id"] == content_id
    assert body["status"] == "classified"
    assert body["has_appeal"] is False
    assert body["attribution"] == "likely_ai"
    assert "transparency_label" in body and body["transparency_label"]


def test_content_unknown_returns_404(client):
    assert client.get("/content/does-not-exist").status_code == 404


# --------------------------------------------------------------------------
# POST /appeal — happy path
# --------------------------------------------------------------------------
def test_valid_appeal(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client).get_json()["content_id"]

    resp = client.post(
        "/appeal",
        json={
            "content_id": content_id,
            "creator_reasoning": "I wrote this myself from personal experience.",
            "evidence_description": "I can provide dated drafts.",
        },
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "under_review"
    assert body["content_id"] == content_id
    assert body["appeal_id"]
    assert body["message"] == (
        "Your appeal has been recorded and the content is now under review."
    )

    # Content status flips to under_review and has_appeal becomes true.
    content = client.get(f"/content/{content_id}").get_json()
    assert content["status"] == "under_review"
    assert content["has_appeal"] is True


def test_appeal_with_matching_creator_id(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client, creator="alice").get_json()["content_id"]
    resp = client.post(
        "/appeal",
        json={
            "content_id": content_id,
            "creator_id": "alice",
            "creator_reasoning": "Mine.",
        },
    )
    assert resp.status_code == 201


# --------------------------------------------------------------------------
# POST /appeal — validation & authorization
# --------------------------------------------------------------------------
def test_appeal_non_json_body(client):
    assert client.post("/appeal", data="x", content_type="text/plain").status_code == 400


def test_appeal_missing_content_id(client):
    resp = client.post("/appeal", json={"creator_reasoning": "Mine."})
    assert resp.status_code == 400


def test_appeal_missing_reasoning(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client).get_json()["content_id"]
    assert client.post("/appeal", json={"content_id": content_id}).status_code == 400
    assert client.post(
        "/appeal", json={"content_id": content_id, "creator_reasoning": "   "}
    ).status_code == 400


def test_appeal_reasoning_too_long(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client).get_json()["content_id"]
    resp = client.post(
        "/appeal",
        json={"content_id": content_id, "creator_reasoning": "a" * 5001},
    )
    assert resp.status_code == 400


def test_appeal_unknown_content_returns_404(client):
    resp = client.post(
        "/appeal",
        json={"content_id": "nope", "creator_reasoning": "Mine."},
    )
    assert resp.status_code == 404


def test_appeal_wrong_creator_returns_403(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client, creator="alice").get_json()["content_id"]
    resp = client.post(
        "/appeal",
        json={
            "content_id": content_id,
            "creator_id": "mallory",
            "creator_reasoning": "Let me in.",
        },
    )
    assert resp.status_code == 403


def test_duplicate_appeal_returns_409(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client).get_json()["content_id"]
    first = client.post(
        "/appeal", json={"content_id": content_id, "creator_reasoning": "Mine."}
    )
    assert first.status_code == 201
    second = client.post(
        "/appeal", json={"content_id": content_id, "creator_reasoning": "Again."}
    )
    assert second.status_code == 409


# --------------------------------------------------------------------------
# Audit log — appeal event + preserved submission decision
# --------------------------------------------------------------------------
def test_appeal_event_logged_and_submission_preserved(client, monkeypatch):
    _mock_both(monkeypatch)
    content_id = _submit(client, creator="bob").get_json()["content_id"]
    client.post(
        "/appeal",
        json={
            "content_id": content_id,
            "creator_reasoning": "I wrote it.",
            "evidence_description": "Dated drafts available.",
        },
    )

    entries = client.get("/log").get_json()["entries"]
    appeal_events = [e for e in entries if e["event_type"] == "appeal_submitted"]
    submission_events = [
        e for e in entries if e["event_type"] == "submission" and e["content_id"] == content_id
    ]

    # The appeal event carries the required fields.
    assert len(appeal_events) == 1
    ev = appeal_events[0]
    assert ev["content_id"] == content_id
    assert ev["creator_id"] == "bob"
    assert ev["creator_reasoning"] == "I wrote it."
    assert ev["evidence_description"] == "Dated drafts available."
    assert ev["original_attribution"] == "likely_ai"
    assert ev["original_confidence"] is not None
    assert ev["status"] == "under_review"
    assert ev["appeal_id"]
    assert ev["timestamp"]

    # The original submission decision is still present (not overwritten).
    assert len(submission_events) == 1
    assert submission_events[0]["attribution"] == "likely_ai"
    assert submission_events[0]["transparency_label"]


def test_no_raw_text_in_appeal_flow_log(client, monkeypatch):
    _mock_both(monkeypatch)
    secret = "ZEBRA-SECRET-98765"
    content_id = _submit(client, text=f"A passage with {secret} inside.").get_json()["content_id"]
    client.post(
        "/appeal", json={"content_id": content_id, "creator_reasoning": "Mine."}
    )
    serialized = str(client.get("/log").get_json())
    assert secret not in serialized
    assert "GROQ_API_KEY" not in serialized
