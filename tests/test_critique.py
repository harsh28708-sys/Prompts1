"""Tests for critique.py (`prompteval improve`'s judge -- a written critique,
not a 1-5 score), with litellm mocked so no real API calls happen."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import promteval.critique as critique_module
from promteval.critique import generate_critique, write_critique_report
from promteval.generator import GenerationError
from promteval.schemas import LLMCallResult

MODEL = "groq/llama-3.3-70b-versatile"


def fake_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def make_result(output="A summary.", error=None) -> LLMCallResult:
    return LLMCallResult(
        variant_id="v1", test_case_id="tc1", model=MODEL,
        rendered_prompt="Summarize: hi", output=output, error=error,
    )


async def test_generate_critique_returns_the_models_text(monkeypatch):
    monkeypatch.setattr(
        critique_module.litellm, "acompletion",
        AsyncMock(return_value=fake_response("Here's how to improve your prompt: ...")),
    )

    text = await generate_critique("some context", "Summarize: {input}", [make_result()], MODEL)

    assert text == "Here's how to improve your prompt: ..."


async def test_generate_critique_fails_cleanly_when_every_run_failed(monkeypatch):
    # In plain terms: if the prompt never produced a single real output, there's
    # nothing genuine to critique -- that should be a clear error, not a made-up
    # critique of outputs that don't exist.
    mock = AsyncMock()
    monkeypatch.setattr(critique_module.litellm, "acompletion", mock)
    failed = make_result(output=None, error="timeout")

    try:
        await generate_critique("ctx", "Summarize: {input}", [failed], MODEL)
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "nothing real to critique" in str(exc)
    mock.assert_not_called()  # never even asks the model to critique nothing


async def test_generate_critique_call_failure_becomes_generation_error(monkeypatch):
    monkeypatch.setattr(critique_module.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("boom")))

    try:
        await generate_critique("ctx", "Summarize: {input}", [make_result()], MODEL)
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "boom" in str(exc)


async def test_generate_critique_still_works_with_some_failures_mixed_in(monkeypatch):
    # In plain terms: if 2 of 3 scenarios succeeded, critique the 2 real ones
    # rather than refusing outright.
    mock = AsyncMock(return_value=fake_response("Feedback based on the 2 successful runs."))
    monkeypatch.setattr(critique_module.litellm, "acompletion", mock)
    results = [make_result(), make_result(), make_result(output=None, error="rate limited")]

    text = await generate_critique("ctx", "Summarize: {input}", results, MODEL)

    assert "Feedback" in text
    sent_prompt = mock.call_args.kwargs["messages"][0]["content"]
    assert "1 of 3 test runs failed" in sent_prompt


def test_write_critique_report_creates_a_real_file(tmp_path):
    results = [make_result(output="Output A"), make_result(output=None, error="timeout")]

    path = write_critique_report("my context", "Summarize: {input}", results, "Great feedback here.", tmp_path)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "my context" in content
    assert "Output A" in content
    assert "Failed: timeout" in content
    assert "Great feedback here." in content
