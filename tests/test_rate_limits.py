"""Rate-limit tests. Re-enables the limiter and resets counters per test.

Both detection signals are MOCKED so /submit succeeds without a real API call.
"""

import os
import sys
import tempfile

import pytest

os.environ["PROVENANCE_DB_PATH"] = os.path.join(
    tempfile.gettempdir(), "provenance_ratelimit_import_placeholder.db"
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import database  # noqa: E402


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test_ratelimit.db"
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(path))
    database.init_db()
    yield str(path)


@pytest.fixture
def limited_client(monkeypatch):
    """Test client WITH the limiter enabled and counters reset."""
    def fake_groq(text, content_type="other"):
        return {"ai_score": 0.5, "reliability": 1.0, "flags": [], "signal": "groq"}

    def fake_style(text, content_type="other"):
        return {
            "ai_score": 0.5,
            "reliability": 1.0,
            "features": {"word_count": 200},
            "component_scores": {},
            "signal": "stylometry",
        }

    monkeypatch.setattr(app_module, "run_groq_signal", fake_groq)
    monkeypatch.setattr(app_module, "run_stylometric_signal", fake_style)

    app_module.limiter.enabled = True
    try:
        app_module.limiter.reset()
    except Exception:
        pass
    app_module.app.config["TESTING"] = True
    try:
        with app_module.app.test_client() as c:
            yield c
    finally:
        app_module.limiter.enabled = False


def test_submit_rate_limit_returns_429_json(limited_client):
    statuses = []
    last_body = None
    for _ in range(12):
        resp = limited_client.post(
            "/submit", json={"text": "rate limit text", "creator_id": "a"}
        )
        statuses.append(resp.status_code)
        last_body = resp.get_json()
    # 10/minute limit: first 10 succeed, then 429 (never 500).
    assert statuses[:10].count(200) == 10
    assert 429 in statuses
    assert 500 not in statuses
    # The final 429 response is JSON with the documented shape.
    assert last_body["error"] == "rate_limit_exceeded"
    assert "message" in last_body


def test_appeal_rate_limit_returns_429_json(limited_client):
    statuses = []
    last_body = None
    # Appeals to nonexistent content still count against the 3/hour limit.
    for _ in range(4):
        resp = limited_client.post(
            "/appeal", json={"content_id": "missing", "creator_reasoning": "Mine."}
        )
        statuses.append(resp.status_code)
        last_body = resp.get_json()
    # 3/hour limit: the 4th request is throttled.
    assert statuses[3] == 429
    assert 500 not in statuses
    assert last_body["error"] == "rate_limit_exceeded"


def test_get_endpoints_not_rate_limited(limited_client):
    # /health, /log, /content/<id> have no default limit and never 429.
    for _ in range(15):
        assert limited_client.get("/health").status_code == 200
    for _ in range(15):
        assert limited_client.get("/log").status_code == 200
    for _ in range(15):
        # Unknown content -> 404, but never 429 (not rate limited).
        assert limited_client.get("/content/unknown").status_code == 404
