"""Tests for T9.1-T9.2: the CLI entry point."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm

from promteval.cli import DEFAULT_JUDGE_MODEL, build_parser, main
from promteval.executor import DEFAULT_CONCURRENCY, DEFAULT_TIMEOUT_S
from promteval.schemas import EvalRun

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
