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


@pytest.fixture(autouse=True)
def _isolated_history_db(monkeypatch, tmp_path):
    """Every endpoint that runs successfully writes to history -- point that at
    a throwaway file so tests never touch (or get polluted by) the real
    ~/.promteval/history.db."""
    monkeypatch.setattr(web_module, "HISTORY_DB_PATH", tmp_path / "history.db")


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


# --- History ----------------------------------------------------------------


def test_api_improve_success_is_saved_to_history(monkeypatch):
    monkeypatch.setattr(web_module, "generate_test_cases", AsyncMock(return_value=["a", "b", "c"]))
    monkeypatch.setattr(web_module, "execute_matrix", AsyncMock(return_value=[fake_call_result("out")]))
    monkeypatch.setattr(
        web_module, "generate_feedback",
        AsyncMock(return_value=PromptFeedback(score=4, reasoning="Solid.", improved_prompt="Better: {input}")),
    )

    client.post("/api/improve", json={"prompt": "Summarize: {input}"})

    history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["kind"] == "improve"
    assert history[0]["summary"] == "Summarize: {input}"


def test_api_improve_failure_is_not_saved_to_history(monkeypatch):
    monkeypatch.setattr(
        web_module, "generate_test_cases", AsyncMock(side_effect=GenerationError("model unavailable"))
    )

    client.post("/api/improve", json={"prompt": "Summarize: {input}"})

    assert client.get("/api/history").json() == []


def test_api_history_detail_returns_the_full_request_and_response(monkeypatch):
    monkeypatch.setattr(web_module, "generate_test_cases", AsyncMock(return_value=["a", "b", "c"]))
    monkeypatch.setattr(web_module, "execute_matrix", AsyncMock(return_value=[fake_call_result("out")]))
    monkeypatch.setattr(
        web_module, "generate_feedback",
        AsyncMock(return_value=PromptFeedback(score=4, reasoning="Solid.", improved_prompt="Better: {input}")),
    )
    client.post("/api/improve", json={"prompt": "Summarize: {input}"})
    entry_id = client.get("/api/history").json()[0]["id"]

    detail = client.get(f"/api/history/{entry_id}")

    assert detail.status_code == 200
    body = detail.json()
    assert body["response"]["score"] == 4
    assert body["request"]["prompt"] == "Summarize: {input}"


def test_api_history_detail_404s_for_a_missing_id():
    response = client.get("/api/history/999999")

    assert response.status_code == 404


def test_api_history_delete_removes_one_entry(monkeypatch):
    monkeypatch.setattr(web_module, "generate_test_cases", AsyncMock(return_value=["a", "b", "c"]))
    monkeypatch.setattr(web_module, "execute_matrix", AsyncMock(return_value=[fake_call_result("out")]))
    monkeypatch.setattr(
        web_module, "generate_feedback",
        AsyncMock(return_value=PromptFeedback(score=4, reasoning="Solid.", improved_prompt="Better: {input}")),
    )
    client.post("/api/improve", json={"prompt": "Summarize: {input}"})
    entry_id = client.get("/api/history").json()[0]["id"]

    delete_response = client.delete(f"/api/history/{entry_id}")

    assert delete_response.status_code == 200
    assert client.get("/api/history").json() == []


def test_api_history_delete_404s_for_a_missing_id():
    response = client.delete("/api/history/999999")

    assert response.status_code == 404


def test_api_history_clear_removes_everything(monkeypatch):
    monkeypatch.setattr(web_module, "generate_test_cases", AsyncMock(return_value=["a", "b", "c"]))
    monkeypatch.setattr(web_module, "execute_matrix", AsyncMock(return_value=[fake_call_result("out")]))
    monkeypatch.setattr(
        web_module, "generate_feedback",
        AsyncMock(return_value=PromptFeedback(score=4, reasoning="Solid.", improved_prompt="Better: {input}")),
    )
    client.post("/api/improve", json={"prompt": "First"})
    client.post("/api/improve", json={"prompt": "Second"})

    response = client.delete("/api/history")

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 2
    assert client.get("/api/history").json() == []
