"""T4.4: judge engine tests with mocked judge responses (no real API calls)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import promteval.judge as judge_module
from promteval.judge import call_id_for, judge_call_result
from promteval.schemas import LLMCallResult

JUDGE_MODEL = "gemini/gemini-2.0-flash"


def make_call_result(output="A concise summary.", error=None) -> LLMCallResult:
    return LLMCallResult(
        variant_id="v1",
        test_case_id="tc1",
        model="groq/llama-3.3-70b-versatile",
        rendered_prompt="Summarize: ...",
        output=output,
        error=error,
    )


def fake_response(text: str):
    """Mimics litellm's ModelResponse shape: response.choices[0].message.content."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


async def test_execution_failure_skips_judge_call(monkeypatch):
    mock_acompletion = AsyncMock()
    monkeypatch.setattr(judge_module.litellm, "acompletion", mock_acompletion)

    result = make_call_result(output=None, error="timeout after 3 retries")
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 0
    assert "timeout after 3 retries" in jr.reasoning
    mock_acompletion.assert_not_called()  # a failed call never even reaches the judge


async def test_valid_json_response(monkeypatch):
    mock_acompletion = AsyncMock(
        return_value=fake_response('{"score": 4, "reasoning": "Accurate and concise."}')
    )
    monkeypatch.setattr(judge_module.litellm, "acompletion", mock_acompletion)

    result = make_call_result()
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 4
    assert jr.reasoning == "Accurate and concise."
    assert jr.call_id == call_id_for(result)
    mock_acompletion.assert_awaited_once()


async def test_markdown_wrapped_json_response(monkeypatch):
    wrapped = '```json\n{"score": 3, "reasoning": "Missed the sentiment."}\n```'
    monkeypatch.setattr(judge_module.litellm, "acompletion", AsyncMock(return_value=fake_response(wrapped)))

    result = make_call_result()
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 3
    assert jr.reasoning == "Missed the sentiment."


async def test_malformed_json_retries_once_then_succeeds(monkeypatch):
    mock_acompletion = AsyncMock(
        side_effect=[
            fake_response("not json at all"),
            fake_response('{"score": 2, "reasoning": "Too verbose."}'),
        ]
    )
    monkeypatch.setattr(judge_module.litellm, "acompletion", mock_acompletion)

    result = make_call_result()
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 2
    assert mock_acompletion.await_count == 2  # 1 initial attempt + 1 retry


async def test_malformed_json_both_attempts_falls_back_to_zero(monkeypatch):
    mock_acompletion = AsyncMock(return_value=fake_response("still not json"))
    monkeypatch.setattr(judge_module.litellm, "acompletion", mock_acompletion)

    result = make_call_result()
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 0
    assert jr.reasoning == "judge parse failure"
    assert mock_acompletion.await_count == 2  # never retries a third time


async def test_out_of_range_score_treated_as_unparseable(monkeypatch):
    # A "7" would violate JudgeResult's own 1-5 constraint, so it must be caught
    # as a parse failure rather than crashing when building the JudgeResult.
    mock_acompletion = AsyncMock(return_value=fake_response('{"score": 7, "reasoning": "way too generous"}'))
    monkeypatch.setattr(judge_module.litellm, "acompletion", mock_acompletion)

    result = make_call_result()
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 0
    assert jr.reasoning == "judge parse failure"


async def test_judge_call_exception_falls_back_to_zero_instead_of_crashing(monkeypatch):
    # In plain terms: this is the bug found by a real run hitting Groq's rate limit --
    # a network/API error from the judge call itself (not a malformed response) must
    # never escape and crash the whole CLI. It should retry once, then give up cleanly.
    mock_acompletion = AsyncMock(side_effect=RuntimeError("429 Too Many Requests"))
    monkeypatch.setattr(judge_module.litellm, "acompletion", mock_acompletion)

    result = make_call_result()
    jr = await judge_call_result(result, "Score 1-5.", JUDGE_MODEL, asyncio.Semaphore(1))

    assert jr.score == 0
    assert "judge call failed" in jr.reasoning
    assert "429 Too Many Requests" in jr.reasoning
    assert mock_acompletion.await_count == 2  # 1 initial attempt + 1 retry, then gives up
