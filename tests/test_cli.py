"""Tests for T9.1-T9.2: the CLI entry point."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import litellm
import pytest

import promteval.cli as cli_module
from promteval.cli import DEFAULT_JUDGE_MODEL, build_parser, main
from promteval.executor import DEFAULT_CONCURRENCY, DEFAULT_TIMEOUT_S
from promteval.generator import GenerationError
from promteval.schemas import EvalRun


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Most tests below exercise behavior that happens *after* the CLI's own
    "is any API key configured?" guard -- this fakes one key so that guard never
    blocks them, regardless of whether the machine running the tests has a real
    .env file. The one test that actually checks the guard clears it explicitly."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

SAMPLE_INPUT = {
    "task_name": "cli_test",
    "models": ["groq/llama-3.3-70b-versatile"],
    "prompt_variants": [{"id": "v1", "name": "Direct", "template": "Summarize: {msg}"}],
    "test_cases": [{"id": "tc1", "variables": {"msg": "hello"}}],
    "judge_criteria": "n/a",
}


def fake_llm_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def fake_ask(value):
    """Mimics questionary's `.select(...).ask()` / `.text(...).ask()` chain
    without touching a real console (which questionary needs and this
    sandboxed test environment doesn't have)."""
    return SimpleNamespace(ask=lambda: value)


def test_build_parser_uses_plan_md_defaults():
    # In plain terms: if you don't pass any flags, the CLI should fall back to
    # the exact defaults written down in Plan.md, not something else.
    args = build_parser().parse_args(["run", "input.json"])

    assert args.judge_model == DEFAULT_JUDGE_MODEL
    assert args.concurrency == DEFAULT_CONCURRENCY
    assert args.timeout == DEFAULT_TIMEOUT_S
    assert args.format == "json"


def test_build_parser_flags_override_defaults():
    args = build_parser().parse_args(
        ["run", "input.json", "--judge-model", "groq/x", "--concurrency", "3", "--timeout", "10", "--format", "markdown"]
    )

    assert args.judge_model == "groq/x"
    assert args.concurrency == 3
    assert args.timeout == 10
    assert args.format == "markdown"


@pytest.mark.parametrize("token", ["/help", "help", "-h", "--help"])
def test_main_shows_friendly_help_for_any_help_token(token, capsys):
    # In plain terms: someone who has no idea what to do should be able to type
    # any of the obvious things ("/help", "help", "-h", "--help") and get the
    # same warm, example-filled guide -- not argparse's terse default.
    exit_code = main([token])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "NEW HERE?" in out
    assert "prompteval init" in out
    assert "prompteval run" in out


def test_main_subcommand_specific_help_still_works(capsys):
    # In plain terms: `prompteval run -h` should still show argparse's own
    # flag-by-flag help for `run` specifically, not the general friendly guide.
    with pytest.raises(SystemExit):
        main(["run", "-h"])

    out = capsys.readouterr().out
    assert "NEW HERE?" not in out
    assert "input_file" in out or "--judge-model" in out


def test_main_reports_missing_api_key_without_crashing(monkeypatch, capsys):
    # In plain terms: running the tool with zero API keys configured anywhere
    # should give a clear, actionable message -- not a cryptic network error
    # buried three layers deep in a real LLM call.
    #
    # python-dotenv's load_dotenv() finds .env by walking up from cli.py's own
    # location, not from the process's current directory -- so chdir alone can't
    # stop main() from finding this repo's real .env (which has real keys in
    # it). Neutralizing load_dotenv() itself is the only reliable way to test
    # "no key configured" without depending on whatever machine runs this test.
    monkeypatch.setattr(cli_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code = main(["run", "some_file.json"])

    assert exit_code == 1
    assert "No API key found" in capsys.readouterr().err


def test_main_reports_missing_file_without_crashing(capsys):
    # In plain terms: pointing the tool at a file that doesn't exist should
    # print a clear error and exit cleanly, not crash with a scary traceback.
    exit_code = main(["run", "does_not_exist.json"])

    assert exit_code == 1
    assert "not found" in capsys.readouterr().err


def test_main_reports_invalid_json_without_crashing(tmp_path, capsys):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    exit_code = main(["run", str(bad_file)])

    assert exit_code == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_main_reports_schema_mismatch_without_crashing(tmp_path, capsys):
    # In plain terms: a JSON file that's valid JSON but missing required fields
    # (e.g. no prompt_variants) should also fail with a readable message.
    bad_file = tmp_path / "incomplete.json"
    bad_file.write_text(json.dumps({"task_name": "oops"}), encoding="utf-8")

    exit_code = main(["run", str(bad_file)])

    assert exit_code == 1
    assert "expected EvalRun format" in capsys.readouterr().err


def test_main_runs_full_pipeline_and_writes_report(tmp_path, monkeypatch, capsys):
    # In plain terms: this is the big one -- with real API calls swapped out for
    # a fake response, running the actual CLI command end-to-end should still
    # print a results table and save a real report file, exactly like a real run.
    fake_call = AsyncMock(return_value=fake_llm_response('{"score": 4, "reasoning": "fine"}'))
    monkeypatch.setattr(litellm, "acompletion", fake_call)
    monkeypatch.chdir(tmp_path)  # so the written report lands in tmp_path, not the real repo

    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(SAMPLE_INPUT), encoding="utf-8")

    exit_code = main(["run", str(input_file), "--judge-model", "groq/llama-3.3-70b-versatile"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Recommended:" in out
    assert "Report written to:" in out

    written_files = list(tmp_path.glob("cli_test_*.json"))
    assert len(written_files) == 1
    saved = json.loads(written_files[0].read_text(encoding="utf-8"))
    assert saved["recommended_variant_id"] == "v1"


def test_main_init_writes_a_file_that_run_can_actually_load(tmp_path, monkeypatch):
    # In plain terms: the interactive wizard's output isn't just "some JSON" --
    # it must be a real, valid input file that `prompteval run` accepts.
    answers = iter([
        "Wizard test", "Direct", "Summarize: {msg}", "n",
        "Hello", "n", "", "",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt_text="": next(answers))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])

    assert exit_code == 0
    output_path = tmp_path / "Wizard_test.json"
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["task_name"] == "Wizard test"
    assert saved["prompt_variants"][0]["template"] == "Summarize: {msg}"
    EvalRun(**saved)  # doesn't raise -- proves `prompteval run` could load this file


def test_main_quickstart_runs_immediately_without_a_separate_run_command(tmp_path, monkeypatch, capsys):
    # In plain terms: quickstart shouldn't require running `prompteval run`
    # afterward -- one command should generate, ask, and evaluate immediately.
    canned_run = EvalRun(
        task_name="Quickstart demo",
        models=["groq/llama-3.3-70b-versatile"],
        prompt_variants=[{"id": "v1", "name": "Prompt 1", "template": "Summarize: {input}"}],
        test_cases=[{"id": "tc1", "variables": {"input": "hello"}}],
        judge_criteria="n/a",
    )
    monkeypatch.setattr(cli_module, "run_quickstart_wizard", AsyncMock(return_value=canned_run))
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_llm_response('{"score": 4, "reasoning": "fine"}')))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["quickstart"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Saved input file to:" in out
    assert "Recommended:" in out
    assert "Report written to:" in out

    # Both the reusable input file AND the report should exist.
    assert (tmp_path / "Quickstart_demo.json").exists()
    assert list(tmp_path.glob("Quickstart_demo_*.json"))  # the timestamped report


def test_main_improve_runs_immediately_and_saves_written_feedback(tmp_path, monkeypatch, capsys):
    # In plain terms: `improve` should test the one prompt for real and print a
    # score plus a rewritten prompt -- not a ranked table against other variants.
    canned_answers = ("Summarize: {input}", ["msg one", "msg two", "msg three"])
    monkeypatch.setattr(cli_module, "run_improve_wizard", AsyncMock(return_value=canned_answers))
    feedback_json = '{"score": 4, "reasoning": "Solid but a bit vague.", "improved_prompt": "Summarize concisely: {input}"}'
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_llm_response(feedback_json)))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["improve", "--judge-model", "groq/llama-3.3-70b-versatile"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Score: 4/5" in out
    assert "Solid but a bit vague." in out
    assert "Summarize concisely: {input}" in out
    assert "Feedback saved to:" in out
    assert list(tmp_path.glob("Summarize_*_feedback_*.md"))


def test_main_improve_reports_generation_failure_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "run_improve_wizard", AsyncMock(side_effect=GenerationError("model unavailable")))

    exit_code = main(["improve"])

    assert exit_code == 1
    assert "model unavailable" in capsys.readouterr().err


def test_main_with_no_args_shows_menu_then_runs_the_picked_flow(tmp_path, monkeypatch):
    # In plain terms: bare `prompteval` should show the arrow-key menu, and
    # picking "improve" there should behave exactly like `prompteval improve`.
    canned_answers = ("Summarize: {input}", ["msg one", "msg two", "msg three"])
    generate_wizard = AsyncMock(return_value=canned_answers)
    monkeypatch.setattr(cli_module, "run_improve_wizard", generate_wizard)
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("improve"))
    feedback_json = '{"score": 4, "reasoning": "Solid.", "improved_prompt": "Summarize concisely: {input}"}'
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_llm_response(feedback_json)))
    monkeypatch.chdir(tmp_path)

    exit_code = main([])  # bare `prompteval`, no subcommand at all

    assert exit_code == 0
    generate_wizard.assert_awaited_once()  # proves it actually ran the improve flow
    assert list(tmp_path.glob("Summarize_*_feedback_*.md"))


def test_main_with_only_flags_shows_menu_and_applies_them(tmp_path, monkeypatch):
    # In plain terms: `prompteval --judge-model X` (no subcommand) should still
    # apply --judge-model to whatever gets picked from the menu, not lose it.
    canned_answers = ("Summarize: {input}", ["msg one", "msg two", "msg three"])
    monkeypatch.setattr(cli_module, "run_improve_wizard", AsyncMock(return_value=canned_answers))
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("improve"))
    feedback_json = '{"score": 4, "reasoning": "Solid.", "improved_prompt": "Summarize concisely: {input}"}'
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_llm_response(feedback_json)))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--judge-model", "groq/llama-3.3-70b-versatile"])

    assert exit_code == 0
    assert list(tmp_path.glob("Summarize_*_feedback_*.md"))


def test_main_web_starts_the_local_server(monkeypatch):
    # In plain terms: `prompteval web` should hand off to the local browser
    # server with whatever port/--no-browser flags were given, not run a pipeline.
    fake_run_server = Mock()
    monkeypatch.setattr(cli_module, "run_server", fake_run_server)

    exit_code = main(["web", "--port", "9999", "--no-browser"])

    assert exit_code == 0
    fake_run_server.assert_called_once_with(port=9999, open_browser=False)


def test_main_web_defaults_to_opening_a_browser_on_the_default_port(monkeypatch):
    fake_run_server = Mock()
    monkeypatch.setattr(cli_module, "run_server", fake_run_server)

    exit_code = main(["web"])

    assert exit_code == 0
    fake_run_server.assert_called_once_with(port=8420, open_browser=True)


def test_main_quickstart_reports_generation_failure_cleanly(monkeypatch, capsys):
    # In plain terms: if the AI generation step fails (bad network, bad response),
    # quickstart should print a clear error and exit, not crash with a traceback.
    monkeypatch.setattr(cli_module, "run_quickstart_wizard", AsyncMock(side_effect=GenerationError("model unavailable")))

    exit_code = main(["quickstart"])

    assert exit_code == 1
    assert "model unavailable" in capsys.readouterr().err


# --- _run_interactive_menu -------------------------------------------------


def test_menu_improve_choice_returns_improve_argv(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("improve"))

    result = cli_module._run_interactive_menu(["--judge-model", "groq/x"])

    assert result == ["improve", "--judge-model", "groq/x"]  # existing flags preserved


def test_menu_quickstart_choice_returns_quickstart_argv(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("quickstart"))

    result = cli_module._run_interactive_menu([])

    assert result == ["quickstart"]


def test_menu_web_choice_returns_web_argv(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("web"))

    result = cli_module._run_interactive_menu([])

    assert result == ["web"]


def test_menu_run_choice_asks_for_a_file_path(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("run"))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: fake_ask("my_input.json"))

    result = cli_module._run_interactive_menu([])

    assert result == ["run", "my_input.json"]


def test_menu_run_choice_with_blank_path_loops_back_to_the_menu(monkeypatch):
    # In plain terms: cancelling the file-path question shouldn't crash or run
    # with no file -- it should just show the menu again.
    selects = iter(["run", "improve"])
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask(next(selects)))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: fake_ask(""))

    result = cli_module._run_interactive_menu([])

    assert result == ["improve"]


def test_menu_init_choice_with_blank_filename_uses_no_extra_arg(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("init"))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: fake_ask(""))

    result = cli_module._run_interactive_menu([])

    assert result == ["init"]


def test_menu_init_choice_with_a_filename_includes_it(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("init"))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: fake_ask("saved.json"))

    result = cli_module._run_interactive_menu([])

    assert result == ["init", "saved.json"]


def test_menu_help_choice_prints_menu_specific_help_then_loops_back(monkeypatch, capsys):
    # In plain terms: the menu's own Help option should explain what each menu
    # choice actually does (not just repeat the general /help install guide).
    selects = iter(["help", "exit"])
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask(next(selects)))

    result = cli_module._run_interactive_menu([])

    assert result is None
    out = capsys.readouterr().out
    assert "SAVE A PROMPT COMPARISON TO A FILE FOR LATER" in out
    assert "RUN A PROMPT COMPARISON FILE YOU ALREADY HAVE" in out


def test_menu_exit_choice_returns_none(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("exit"))

    assert cli_module._run_interactive_menu([]) is None


def test_menu_ctrl_c_or_esc_is_treated_like_exit(monkeypatch):
    # In plain terms: questionary reports Ctrl+C/Esc as None -- must not crash.
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask(None))

    assert cli_module._run_interactive_menu([]) is None


def test_main_no_args_menu_exit_returns_cleanly(monkeypatch):
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: fake_ask("exit"))

    assert main([]) == 0


# --- fallback when the terminal can't render questionary's arrow-key menu --


def test_ask_select_falls_back_to_a_numbered_menu_when_questionary_cannot_render(monkeypatch):
    # In plain terms: some real terminals (Git Bash/mintty on Windows, confirmed
    # via a live test) can't render the fancy menu at all and raise instead of
    # just looking worse -- this proves the tool still works there.
    def raise_no_console(*_a, **_k):
        raise RuntimeError("No Windows console found.")

    monkeypatch.setattr(cli_module.questionary, "select", raise_no_console)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")  # picks choice #1

    result = cli_module._ask_select("Pick one", cli_module._MENU_CHOICES)

    assert result == cli_module._MENU_CHOICES[0].value  # "improve"


def test_ask_select_fallback_retries_on_an_invalid_number(monkeypatch):
    def raise_no_console(*_a, **_k):
        raise RuntimeError("no console")

    monkeypatch.setattr(cli_module.questionary, "select", raise_no_console)
    answers = iter(["not a number", "99", "2"])  # garbage, out of range, then valid
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    result = cli_module._ask_select("Pick one", cli_module._MENU_CHOICES)

    assert result == cli_module._MENU_CHOICES[1].value  # "quickstart"


def test_ask_text_falls_back_to_plain_input_when_questionary_cannot_render(monkeypatch):
    def raise_no_console(*_a, **_k):
        raise RuntimeError("no console")

    monkeypatch.setattr(cli_module.questionary, "text", raise_no_console)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "my_answer.json")

    result = cli_module._ask_text("File path:", default="fallback.json")

    assert result == "my_answer.json"


def test_ask_text_fallback_uses_default_on_blank_answer(monkeypatch):
    def raise_no_console(*_a, **_k):
        raise RuntimeError("no console")

    monkeypatch.setattr(cli_module.questionary, "text", raise_no_console)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    result = cli_module._ask_text("File path:", default="fallback.json")

    assert result == "fallback.json"
