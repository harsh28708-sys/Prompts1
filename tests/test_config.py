"""Tests for the shared has_any_api_key() check used by both cli.py and web.py."""

from promteval.config import API_KEY_VARS, has_any_api_key


def test_has_any_api_key_false_when_nothing_set(monkeypatch):
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)

    assert has_any_api_key() is False


def test_has_any_api_key_true_when_any_one_is_set(monkeypatch):
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    assert has_any_api_key() is True


def test_has_any_api_key_ignores_blank_values(monkeypatch):
    # In plain terms: a key that's set but empty (e.g. an unfilled .env template
    # line like `GROQ_API_KEY=`) shouldn't count as "configured".
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "")

    assert has_any_api_key() is False
