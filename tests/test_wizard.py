"""Tests for the interactive `prompteval init` wizard. input() is mocked with a
scripted list of answers, in the exact order the wizard asks its questions."""

from promteval.wizard import _prompt, _prompt_yes_no, run_init_wizard


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
