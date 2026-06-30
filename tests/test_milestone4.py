"""Milestone 4 integration tests for POST /submit.

Both detection signals are MOCKED so the suite is deterministic and never
consumes a real Groq API call. A temporary SQLite database is used so the
development database is never touched.
"""

import os
import sys
import tempfile

import pytest

# Throwaway DB path for the import-time init_db() call; each test overrides it.
os.environ["PROVENANCE_DB_PATH"] = os.path.join(
    tempfile.gettempdir(), "provenance_m4_import_placeholder.db"
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import database  # noqa: E402
from detector import GroqSignalError  # noqa: E402
from labels import LABEL_AI, LABEL_HUMAN, LABEL_UNCERTAIN  # noqa: E402


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test_m4.db"
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(path))
    database.init_db()
    yield str(path)


@pytest.fixture
def client():
    app_module.limiter.enabled = False
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _groq(ai_score, reliability=1.0, flags=None):
    flags = flags or []
    return {
        "ai_score": ai_score,
        "reliability": reliability,
        "flags": flags,
        "signal": "groq",
    }


def _style(ai_score, reliability=1.0, word_count=200):
    return {
        "ai_score": ai_score,
        "reliability": reliability,
        "features": {"word_count": word_count, "sentence_count": 10},
        "component_scores": {"sentence_uniformity": ai_score},
        "signal": "stylometry",
    }


def _mock_both(monkeypatch, groq_dict, style_dict):
    """Patch both signal functions (as imported into app) to fixed dicts."""
    if groq_dict is _RAISE:
        def fake_groq(text, content_type="other"):
            raise GroqSignalError("simulated groq outage")
    else:
        def fake_groq(text, content_type="other"):
            return groq_dict

    if style_dict is _RAISE:
        def fake_style(text, content_type="other"):
            raise ValueError("simulated stylometry failure")
    else:
        def fake_style(text, content_type="other"):
            return style_dict

    monkeypatch.setattr(app_module, "run_groq_signal", fake_groq)
    monkeypatch.setattr(app_module, "run_stylometric_signal", fake_style)


_RAISE = object()  # sentinel meaning "make this signal raise"


def _submit(client, text="A sufficiently long creative passage for testing.", creator="u", ctype="blog_post"):
    return client.post(
        "/submit",
        json={"text": text, "creator_id": creator, "content_type": ctype},
    )


# --------------------------------------------------------------------------
# Attribution outcomes
# --------------------------------------------------------------------------
def test_strong_agreement_high_is_likely_ai(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.94), _style(0.89))
    body = _submit(client).get_json()
    assert body["attribution"] == "likely_ai"
    # Milestone 5: label/transparency_label now carry the final exact text.
    assert body["transparency_label"] == LABEL_AI
    assert body["label"] == LABEL_AI
    assert body["uncertainty"]["forced"] is False


def test_strong_agreement_low_is_likely_human(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.12), _style(0.18))
    body = _submit(client).get_json()
    assert body["attribution"] == "likely_human"
    assert body["transparency_label"] == LABEL_HUMAN
    assert body["label"] == LABEL_HUMAN


def test_middle_scores_are_uncertain(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.60), _style(0.55))
    body = _submit(client).get_json()
    assert body["attribution"] == "uncertain"
    assert body["transparency_label"] == LABEL_UNCERTAIN
    assert body["label"] == LABEL_UNCERTAIN


def test_signal_disagreement_is_uncertain(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.92), _style(0.30))
    body = _submit(client).get_json()
    assert body["attribution"] == "uncertain"
    assert body["signals"]["signal_gap"] >= 0.35
    assert "signal_disagreement" in body["uncertainty"]["reasons"]


def test_short_text_is_uncertain_and_capped(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.95), _style(0.95, word_count=30))
    body = _submit(client).get_json()
    assert body["attribution"] == "uncertain"
    assert body["confidence"] <= 0.69
    assert "insufficient_length" in body["uncertainty"]["reasons"]


# --------------------------------------------------------------------------
# Partial / total failure behavior
# --------------------------------------------------------------------------
def test_groq_failure_stylometry_success_is_uncertain(client, monkeypatch):
    _mock_both(monkeypatch, _RAISE, _style(0.10))
    resp = _submit(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["attribution"] == "uncertain"
    assert body["confidence"] <= 0.60
    assert "missing_signal" in body["uncertainty"]["reasons"]
    # The failed Groq signal is recorded, not fabricated.
    assert body["signals"]["groq"]["ai_score"] is None


def test_stylometry_failure_groq_success_is_uncertain(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.95), _RAISE)
    resp = _submit(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["attribution"] == "uncertain"
    assert body["confidence"] <= 0.60
    assert body["signals"]["stylometry"]["ai_score"] is None


def test_both_signals_fail_returns_503(client, monkeypatch):
    _mock_both(monkeypatch, _RAISE, _RAISE)
    resp = _submit(client)
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "detection_error"
    assert "attribution" not in body


# --------------------------------------------------------------------------
# Response and audit-log structure
# --------------------------------------------------------------------------
def test_response_contains_both_signal_outputs(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.80, flags=["templated"]), _style(0.75))
    body = _submit(client).get_json()
    groq = body["signals"]["groq"]
    sty = body["signals"]["stylometry"]
    assert groq["ai_score"] == pytest.approx(0.80)
    assert groq["flags"] == ["templated"]
    assert sty["ai_score"] == pytest.approx(0.75)
    assert "features" in sty and "component_scores" in sty
    assert "ai_likelihood" in body and "confidence" in body
    assert body["signals"]["signal_gap"] == pytest.approx(0.05)


def test_audit_log_contains_both_signals_and_combined(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.94, flags=["generic"]), _style(0.89))
    content_id = _submit(client, creator="creator-9").get_json()["content_id"]

    entries = database.get_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["content_id"] == content_id
    assert entry["creator_id"] == "creator-9"
    assert entry["groq_ai_score"] == pytest.approx(0.94)
    assert entry["groq_reliability"] == pytest.approx(1.0)
    assert entry["groq_flags"] == ["generic"]  # decoded from JSON
    assert entry["stylometric_ai_score"] == pytest.approx(0.89)
    assert isinstance(entry["stylometric_features"], dict)
    assert isinstance(entry["stylometric_component_scores"], dict)
    assert entry["combined_ai_score"] is not None
    assert entry["signal_gap"] == pytest.approx(0.05)
    assert entry["attribution"] == "likely_ai"
    assert entry["status"] == "classified"


def test_log_via_endpoint_newest_first(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.80), _style(0.80))
    _submit(client, creator="first")
    _submit(client, creator="second")
    entries = client.get("/log").get_json()["entries"]
    assert len(entries) == 2
    assert entries[0]["creator_id"] == "second"
    assert entries[1]["creator_id"] == "first"


# --------------------------------------------------------------------------
# Preserved Milestone 3 validation
# --------------------------------------------------------------------------
def test_validation_still_works(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.5), _style(0.5))
    # Missing JSON body.
    assert client.post("/submit", data="x", content_type="text/plain").status_code == 400
    # Missing text.
    assert client.post("/submit", json={"creator_id": "u"}).status_code == 400
    # Empty text.
    assert client.post("/submit", json={"text": "  ", "creator_id": "u"}).status_code == 400
    # Missing creator_id.
    assert client.post("/submit", json={"text": "hello there"}).status_code == 400
    # Too long.
    assert client.post(
        "/submit", json={"text": "a" * 20_001, "creator_id": "u"}
    ).status_code == 413


def test_health_unchanged(client):
    assert client.get("/health").get_json() == {"status": "ok"}


# --------------------------------------------------------------------------
# Security: no API key, no raw text in the log
# --------------------------------------------------------------------------
def test_no_raw_text_or_api_key_in_log(client, monkeypatch):
    _mock_both(monkeypatch, _groq(0.80), _style(0.80))
    secret_phrase = "PURPLE-ELEPHANT-SENTINEL-12345"
    _submit(client, text=f"A passage containing {secret_phrase} inside it.")

    entries = database.get_log()
    serialized = str(entries)
    assert secret_phrase not in serialized

    # A content hash is stored instead of the raw text.
    assert entries[0]["content_hash"] and len(entries[0]["content_hash"]) == 64

    # The Groq API key value (if configured) never appears in the log.
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        assert api_key not in serialized
    assert "GROQ_API_KEY" not in serialized
