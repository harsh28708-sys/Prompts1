"""Tests for generator.py (AI-generated task/test cases for `prompteval quickstart`),
with litellm mocked so no real API calls happen."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import promteval.generator as generator_module
from promteval.generator import GenerationError, generate_random_task, generate_test_cases

MODEL = "groq/llama-3.3-70b-versatile"


def fake_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


async def test_generate_random_task_returns_the_models_text(monkeypatch):
    # In plain terms: whatever one-sentence task the model comes up with should
    # come back cleanly, with surrounding quotes stripped.
    monkeypatch.setattr(
        generator_module.litellm, "acompletion",
        AsyncMock(return_value=fake_response('"Write a polite decline email"')),
    )

    task = await generate_random_task(MODEL)

    assert task == "Write a polite decline email"


async def test_generate_random_task_call_failure_becomes_generation_error(monkeypatch):
    # In plain terms: a network/API error while generating the task should never
    # crash the wizard -- it should turn into a clear, catchable error instead.
    monkeypatch.setattr(generator_module.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("boom")))

    try:
        await generate_random_task(MODEL)
        raise AssertionError("expected GenerationError")
    except GenerationError as exc:
        assert "boom" in str(exc)


async def test_generate_test_cases_parses_a_valid_json_array(monkeypatch):
    monkeypatch.setattr(
        generator_module.litellm, "acompletion",
        AsyncMock(return_value=fake_response('["a", "b", "c", "d", "e"]')),
    )

    result = await generate_test_cases("some task", MODEL, n=5)

    assert result == ["a", "b", "c", "d", "e"]


async def test_generate_test_cases_retries_once_then_succeeds(monkeypatch):
    mock = AsyncMock(
        side_effect=[
            fake_response("not a json array"),
            fake_response('["a", "b", "c", "d", "e"]'),
        ]
    )
    monkeypatch.setattr(generator_module.litellm, "acompletion", mock)

    result = await generate_test_cases("some task", MODEL, n=5)

    assert result == ["a", "b", "c", "d", "e"]
    assert mock.await_count == 2


async def test_generate_test_cases_wrong_length_is_treated_as_invalid(monkeypatch):
    # In plain terms: if the model returns only 3 items when 5 were asked for,
    # that's just as unusable as broken JSON -- it should retry, not silently accept it.
    mock = AsyncMock(
        side_effect=[
            fake_response('["only", "three", "items"]'),
            fake_response('["a", "b", "c", "d", "e"]'),
        ]
    )
    monkeypatch.setattr(generator_module.litellm, "acompletion", mock)

    result = await generate_test_cases("some task", MODEL, n=5)

    assert result == ["a", "b", "c", "d", "e"]


async def test_generate_test_cases_gives_up_after_two_bad_attempts(monkeypatch):
    monkeypatch.setattr(
        generator_module.litellm, "acompletion",
        AsyncMock(return_value=fake_response("still not json")),
    )

    try:
        await generate_test_cases("some task", MODEL, n=5)
        raise AssertionError("expected GenerationError")
    except GenerationError:
        pass
