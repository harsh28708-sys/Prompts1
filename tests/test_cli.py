"""Tests for T9.1-T9.2: the CLI entry point."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def test_main_with_no_args_defaults_to_quickstart(tmp_path, monkeypatch):
    canned_run = EvalRun(
        task_name="No args demo",
        models=["groq/llama-3.3-70b-versatile"],
        prompt_variants=[{"id": "v1", "name": "Prompt 1", "template": "Summarize: {input}"}],
        test_cases=[{"id": "tc1", "variables": {"input": "hello"}}],
        judge_criteria="n/a",
    )
    generate_wizard = AsyncMock(return_value=canned_run)
    monkeypatch.setattr(cli_module, "run_quickstart_wizard", generate_wizard)
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_llm_response('{"score": 4, "reasoning": "fine"}')))
    monkeypatch.chdir(tmp_path)

    exit_code = main([])  # bare `prompteval`, no subcommand at all

    assert exit_code == 0
    generate_wizard.assert_awaited_once()  # proves it actually ran the quickstart flow
    assert (tmp_path / "No_args_demo.json").exists()


def test_main_with_only_flags_defaults_to_quickstart_and_applies_them(tmp_path, monkeypatch):
    # In plain terms: `prompteval --judge-model X` (no subcommand) should still
    # apply --judge-model, not silently ignore it.
    canned_run = EvalRun(
        task_name="Flags only demo",
        models=["groq/llama-3.3-70b-versatile"],
        prompt_variants=[{"id": "v1", "name": "Prompt 1", "template": "Summarize: {input}"}],
        test_cases=[{"id": "tc1", "variables": {"input": "hello"}}],
        judge_criteria="n/a",
    )
    monkeypatch.setattr(cli_module, "run_quickstart_wizard", AsyncMock(return_value=canned_run))
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_llm_response('{"score": 4, "reasoning": "fine"}')))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--judge-model", "groq/llama-3.3-70b-versatile"])

    assert exit_code == 0
    assert (tmp_path / "Flags_only_demo.json").exists()


def test_main_quickstart_reports_generation_failure_cleanly(monkeypatch, capsys):
    # In plain terms: if the AI generation step fails (bad network, bad response),
    # quickstart should print a clear error and exit, not crash with a traceback.
    monkeypatch.setattr(cli_module, "run_quickstart_wizard", AsyncMock(side_effect=GenerationError("model unavailable")))

    exit_code = main(["quickstart"])

    assert exit_code == 1
    assert "model unavailable" in capsys.readouterr().err
