"""Tests for the interactive `prompteval init`/`quickstart` wizards. input() is
mocked with a scripted list of answers, in the exact order the wizard asks its
questions. Quickstart's own AI-generation calls (generate_random_task /
generate_test_cases) are mocked directly rather than mocking litellm -- those
functions get their own dedicated tests in test_generator.py."""

from unittest.mock import AsyncMock

import promteval.wizard as wizard_module
from promteval.wizard import _prompt, _prompt_yes_no, run_init_wizard, run_quickstart_wizard


def scripted_input(monkeypatch, answers: list[str]):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt_text="": next(it))


def test_prompt_returns_the_given_answer(monkeypatch):
    scripted_input(monkeypatch, ["hello"])
    assert _prompt("Question") == "hello"


def test_prompt_retries_until_a_non_empty_answer_is_given(monkeypatch):
    # In plain terms: pressing Enter with nothing typed shouldn't silently accept
    # a blank task name -- it should ask again.
    scripted_input(monkeypatch, ["", "", "finally an answer"])
    assert _prompt("Question") == "finally an answer"


def test_prompt_falls_back_to_default_on_empty_answer(monkeypatch):
    scripted_input(monkeypatch, [""])
    assert _prompt("Question", default="fallback") == "fallback"


def test_prompt_yes_no_uses_default_on_empty_answer(monkeypatch):
    scripted_input(monkeypatch, [""])
    assert _prompt_yes_no("Continue?", default=True) is True


def test_prompt_yes_no_recognizes_a_no_answer(monkeypatch):
    scripted_input(monkeypatch, ["n"])
    assert _prompt_yes_no("Continue?", default=True) is False


def test_run_init_wizard_builds_a_valid_eval_run(monkeypatch):
    # In plain terms: walking through the whole Q&A -- task name, two prompt
    # variants sharing one {msg} placeholder, one test case, then accepting the
    # default rubric and model -- should produce a fully valid, ready-to-run EvalRun.
    answers = [
        "Test task",                              # task name
        "Direct", "Summarize: {msg}",              # variant 1
        "y",                                       # add another variant?
        "Verbose", "Please summarize in detail: {msg}",  # variant 2
        "n",                                       # add another variant? -> stop at 2
        "Hello world",                             # test case 1's value for {msg}
        "n",                                       # add another test case? -> stop at 1
        "",                                        # judge rubric -> default
        "",                                        # model -> default
    ]
    scripted_input(monkeypatch, answers)

    run = run_init_wizard()

    assert run.task_name == "Test task"
    assert [v.name for v in run.prompt_variants] == ["Direct", "Verbose"]
    assert run.test_cases[0].variables == {"msg": "Hello world"}
    assert run.models == ["groq/llama-3.3-70b-versatile"]
    assert "accuracy" in run.judge_criteria


GENERATED_TEST_CASES = ["case one", "case two", "case three", "case four", "case five"]


async def test_quickstart_uses_typed_task_and_skips_random_generation(monkeypatch):
    generate_task_mock = AsyncMock()
    monkeypatch.setattr(wizard_module, "generate_random_task", generate_task_mock)
    monkeypatch.setattr(wizard_module, "generate_test_cases", AsyncMock(return_value=GENERATED_TEST_CASES))

    answers = [
        "My own task",                     # task typed directly -> skips generation
        "Summarize: {input}", "Explain: {input}", "TL;DR: {input}",  # 3 prompts
        "",                                 # judge rubric -> default
    ]
    scripted_input(monkeypatch, answers)

    run = await run_quickstart_wizard("groq/llama-3.3-70b-versatile")

    assert run.task_name == "My own task"
    generate_task_mock.assert_not_called()


async def test_quickstart_generates_a_random_task_when_left_blank(monkeypatch):
    monkeypatch.setattr(wizard_module, "generate_random_task", AsyncMock(return_value="AI-picked task"))
    monkeypatch.setattr(wizard_module, "generate_test_cases", AsyncMock(return_value=GENERATED_TEST_CASES))

    answers = [
        "",                                 # blank task -> triggers random generation
        "Summarize: {input}", "Explain: {input}", "TL;DR: {input}",
        "",
    ]
    scripted_input(monkeypatch, answers)

    run = await run_quickstart_wizard("groq/llama-3.3-70b-versatile")

    assert run.task_name == "AI-picked task"


async def test_quickstart_builds_3_variants_and_5_test_cases_using_input_placeholder(monkeypatch):
    monkeypatch.setattr(wizard_module, "generate_random_task", AsyncMock(return_value="Some task"))
    monkeypatch.setattr(wizard_module, "generate_test_cases", AsyncMock(return_value=GENERATED_TEST_CASES))

    answers = ["", "Summarize: {input}", "Explain: {input}", "TL;DR: {input}", ""]
    scripted_input(monkeypatch, answers)

    run = await run_quickstart_wizard("groq/llama-3.3-70b-versatile")

    assert len(run.prompt_variants) == 3
    assert len(run.test_cases) == 5
    assert [tc.variables["input"] for tc in run.test_cases] == GENERATED_TEST_CASES
    # every variant's template must actually use the placeholder the test cases fill
    assert all("{input}" in v.template for v in run.prompt_variants)


async def test_quickstart_warns_but_does_not_fail_when_placeholder_missing(monkeypatch, capsys):
    # In plain terms: forgetting to include {input} in a prompt shouldn't crash the
    # wizard -- it's still a valid (if pointless) prompt -- but the user should be warned.
    monkeypatch.setattr(wizard_module, "generate_random_task", AsyncMock(return_value="Some task"))
    monkeypatch.setattr(wizard_module, "generate_test_cases", AsyncMock(return_value=GENERATED_TEST_CASES))

    answers = ["", "This prompt forgot the placeholder", "Explain: {input}", "TL;DR: {input}", ""]
    scripted_input(monkeypatch, answers)

    run = await run_quickstart_wizard("groq/llama-3.3-70b-versatile")

    assert len(run.prompt_variants) == 3
    assert "Warning" in capsys.readouterr().out
