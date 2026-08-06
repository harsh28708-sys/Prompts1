"""Tests for `prompteval web`: the local FastAPI backend in promteval/web.py.
Uses FastAPI's TestClient (sync, in-process -- no real server/port needed) and
mocks the same AI-calling functions test_cli.py mocks, since web.py just wires
those same functions to JSON endpoints instead of terminal input()."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import promteval.web as web_module
from promteval.generator import GenerationError
from promteval.judge import call_id_for
from promteval.schemas import JudgeResult, LLMCallResult, PromptFeedback

client = TestClient(web_module.app)


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Same reasoning as test_cli.py's fixture of the same name: most tests here
    exercise behavior *after* the "is any key configured?" guard, so fake one by
    default. The one test that checks the guard itself overrides this."""
    monkeypatch.setattr(web_module, "has_any_api_key", lambda: True)


def fake_call_result(text: str, variant_id: str = "v1") -> LLMCallResult:
    return LLMCallResult(
        variant_id=variant_id,
        test_case_id="tc1",
        model="groq/llama-3.3-70b-versatile",
        rendered_prompt="Summarize: hello",
        output=text,
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
        error=None,
    )


def test_index_serves_the_html_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_api_status_reports_true_when_a_key_is_configured(monkeypatch):
    monkeypatch.setattr(web_module, "has_any_api_key", lambda: True)

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {"has_api_key": True}


def test_api_status_reports_false_when_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(web_module, "has_any_api_key", lambda: False)

    response = client.get("/api/status")

    assert response.json() == {"has_api_key": False}


def test_api_improve_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(web_module, "has_any_api_key", lambda: False)

    response = client.post("/api/improve", json={"prompt": "Summarize: {input}"})

    assert response.status_code == 400
    assert "No API key found" in response.json()["detail"]


def test_api_improve_rejects_an_empty_prompt():
    response = client.post("/api/improve", json={"prompt": "   "})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_api_improve_happy_path(monkeypatch):
    monkeypatch.setattr(
        web_module, "generate_test_cases", AsyncMock(return_value=["msg one", "msg two", "msg three"])
    )
    monkeypatch.setattr(
        web_module, "execute_matrix", AsyncMock(return_value=[fake_call_result("some output")])
    )
    monkeypatch.setattr(
        web_module,
        "generate_feedback",
        AsyncMock(
            return_value=PromptFeedback(
                score=4, reasoning="Solid but a bit vague.", improved_prompt="Summarize concisely: {input}"
            )
        ),
    )

    response = client.post("/api/improve", json={"prompt": "Summarize: {input}"})

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 4
    assert body["reasoning"] == "Solid but a bit vague."
    assert body["improved_prompt"] == "Summarize concisely: {input}"
    assert body["scenarios"] == ["msg one", "msg two", "msg three"]


def test_api_improve_reports_generation_failure_as_a_clean_error(monkeypatch):
    monkeypatch.setattr(
        web_module, "generate_test_cases", AsyncMock(side_effect=GenerationError("model unavailable"))
    )

    response = client.post("/api/improve", json={"prompt": "Summarize: {input}"})

    assert response.status_code == 502
    assert "model unavailable" in response.json()["detail"]


def test_api_quickstart_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(web_module, "has_any_api_key", lambda: False)

    response = client.post("/api/quickstart", json={"prompts": ["a", "b"]})

    assert response.status_code == 400
    assert "No API key found" in response.json()["detail"]


def test_api_quickstart_requires_at_least_two_prompts():
    response = client.post("/api/quickstart", json={"prompts": ["only one"]})

    assert response.status_code == 400
    assert "at least 2" in response.json()["detail"]


def test_api_quickstart_happy_path(monkeypatch):
    monkeypatch.setattr(web_module, "generate_random_task", AsyncMock(return_value="Random task"))
    monkeypatch.setattr(
        web_module, "generate_test_cases", AsyncMock(return_value=["a", "b", "c", "d", "e"])
    )
    monkeypatch.setattr(
        web_module,
        "execute_matrix",
        AsyncMock(
            return_value=[
                fake_call_result("output 1", variant_id="v1"),
                fake_call_result("output 2", variant_id="v2"),
            ]
        ),
    )
    async def fake_judge(result, _criteria, _model, _semaphore):
        return JudgeResult(score=4, reasoning="fine", call_id=call_id_for(result))

    monkeypatch.setattr(web_module, "judge_call_result", fake_judge)

    response = client.post(
        "/api/quickstart",
        json={"task": "", "prompts": ["Prompt A: {input}", "Prompt B: {input}"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_name"] == "Random task"
    assert len(body["variants"]) == 2
    assert body["recommended"] in {"v1", "v2"}
    assert any(v["is_winner"] for v in body["variants"])


def test_api_quickstart_reports_generation_failure_as_a_clean_error(monkeypatch):
    monkeypatch.setattr(
        web_module, "generate_random_task", AsyncMock(side_effect=GenerationError("model unavailable"))
    )

    response = client.post("/api/quickstart", json={"task": "", "prompts": ["a: {input}", "b: {input}"]})

    assert response.status_code == 502
    assert "model unavailable" in response.json()["detail"]
