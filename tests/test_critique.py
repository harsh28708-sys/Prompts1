"""Tests for critique.py (`prompteval improve`'s evaluator -- a score + rewritten
prompt, not a 1-5 score against other variants, and not prose critique), with
litellm mocked so no real API calls happen."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import promteval.critique as critique_module
from promteval.critique import generate_feedback, write_feedback_report
from promteval.generator import GenerationError
from promteval.schemas import LLMCallResult, PromptFeedback

MODEL = "groq/llama-3.3-70b-versatile"

VALID_JSON = '{"score": 4, "reasoning": "Solid but a bit vague.", "improved_prompt": "Summarize concisely: {input}"}'


def fake_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def make_result(output="A summary.", error=None) -> LLMCallResult:
    return LLMCallResult(
        variant_id="v1", test_case_id="tc1", model=MODEL,
        rendered_prompt="Summarize: hi", output=output, error=error,
    )


async def test_generate_feedback_returns_score_reasoning_and_improved_prompt(monkeypatch):
    monkeypatch.setattr(critique_module.litellm, "acompletion", AsyncMock(return_value=fake_response(VALID_JSON)))

    feedback = await generate_feedback("Summarize: {input}", [make_result()], MODEL)

    assert feedback.score == 4
    assert feedback.reasoning == "Solid but a bit vague."
    assert feedback.improved_prompt == "Summarize concisely: {input}"


async def test_generate_feedback_tolerates_markdown_fences(monkeypatch):
    wrapped = f"```json\n{VALID_JSON}\n```"
    monkeypatch.setattr(critique_module.litellm, "acompletion", AsyncMock(return_value=fake_response(wrapped)))

    feedback = await generate_feedback("Summarize: {input}", [make_result()], MODEL)

    assert feedback.score == 4


async def test_generate_feedback_retries_once_on_malformed_json_then_succeeds(monkeypatch):
    mock = AsyncMock(side_effect=[fake_response("not json at all"), fake_response(VALID_JSON)])
    monkeypatch.setattr(critique_module.litellm, "acompletion", mock)

    feedback = await generate_feedback("Summarize: {input}", [make_result()], MODEL)

    assert feedback.score == 4
    assert mock.await_count == 2


async def test_generate_feedback_gives_up_after_two_bad_attempts(monkeypatch):
    mock = AsyncMock(return_value=fake_response("still not json"))
    monkeypatch.setattr(critique_module.litellm, "acompletion", mock)

    try:
        await generate_feedback("Summarize: {input}", [make_result()], MODEL)
        raise AssertionError("expected GenerationError")
    except GenerationError:
        pass
    assert mock.await_count == 2


async def test_generate_feedback_out_of_range_score_is_treated_as_unparseable(monkeypatch):
    # A "7" would violate PromptFeedback's own 1-5 constraint -- must be caught
    # as a parse failure (and retried) rather than crashing when building the model.
    bad = '{"score": 7, "reasoning": "way too generous", "improved_prompt": "x"}'
    mock = AsyncMock(side_effect=[fake_response(bad), fake_response(VALID_JSON)])
    monkeypatch.setattr(critique_module.litellm, "acompletion", mock)

    feedback = await generate_feedback("Summarize: {input}", [make_result()], MODEL)

    assert feedback.score == 4


async def test_generate_feedback_fails_cleanly_when_every_run_failed(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(critique_module.litellm, "acompletion", mock)
    failed = make_result(output=None, error="timeout")

    try:
        await generate_feedback("Summarize: {input}", [failed], MODEL)
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "nothing real to evaluate" in str(exc)
    mock.assert_not_called()


async def test_generate_feedback_call_failure_becomes_generation_error(monkeypatch):
    monkeypatch.setattr(critique_module.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("boom")))

    try:
        await generate_feedback("Summarize: {input}", [make_result()], MODEL)
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "boom" in str(exc)


def test_write_feedback_report_creates_a_real_file(tmp_path):
    feedback = PromptFeedback(score=4, reasoning="Solid but a bit vague.", improved_prompt="Summarize concisely: {input}")
    results = [make_result(output="Output A"), make_result(output=None, error="timeout")]

    path = write_feedback_report("Summarize: {input}", results, feedback, tmp_path)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Score: 4/5" in content
    assert "Solid but a bit vague." in content
    assert "Summarize concisely: {input}" in content
    assert "Output A" in content
    assert "Failed: timeout" in content
