"""T9.1-T9.2: CLI entry point. Wires load -> render -> execute -> judge -> score -> report
into the single command described in Plan.md: `prompteval run <input.json>`.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import questionary
from dotenv import load_dotenv
from pydantic import ValidationError

from promteval.critique import generate_feedback, write_feedback_report
from promteval.executor import DEFAULT_CONCURRENCY, DEFAULT_TIMEOUT_S, execute_matrix
from promteval.generator import GenerationError
from promteval.judge import judge_call_result
from promteval.renderer import TemplateRenderError, render_matrix
from promteval.reporter import (
    format_table,
    safe_filename_stub,
    write_json_report,
    write_markdown_report,
)
from promteval.schemas import EvalRun, PromptVariant, RunReport, TestCase
from promteval.scoring import build_run_report
from promteval.wizard import (
    DEFAULT_MODEL,
    QUICKSTART_VARIABLE,
    run_improve_wizard,
    run_init_wizard,
    run_quickstart_wizard,
)

DEFAULT_JUDGE_MODEL = "gemini/gemini-2.0-flash"

_API_KEY_VARS = ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")

HELP_TEXT = """
PromtEval -- test a prompt against a real AI and get a score plus a better
version of it, instead of guessing whether your wording is any good.

NEW HERE? JUST RUN:
    prompteval

It shows a menu (arrow keys + Enter) so you can pick what to do without
needing to know a command name.

THE WAYS TO USE IT
    prompteval                     Shows an arrow-key menu of everything
                                    below -- pick one instead of typing a
                                    command name.

    prompteval improve             Got ONE prompt and want it better? AI tests
                                    it against a few realistic scenarios and
                                    gives you a score (1-5) plus a rewritten,
                                    improved version of your prompt.

    prompteval quickstart          Want to compare 3 different prompts instead
                                    of improving one? AI makes up a task and
                                    test data for you; you write 3 prompts.

    prompteval init <file>         Answer the same kind of questions as
                                    quickstart, but SAVE them to a file you
                                    can reuse or edit later.

    prompteval run <file>          Run an input file you already have (one you
                                    built with `init`, or wrote by hand).

YOU NEED ONE FREE API KEY
    Pick any one of these (all have a free tier):
        Groq:       https://console.groq.com/keys
        OpenRouter: https://openrouter.ai/keys
        Gemini:     https://aistudio.google.com/apikey
    Copy .env.example to .env and paste your key in, e.g.:
        GROQ_API_KEY=your-key-here

    Also add --judge-model groq/llama-3.3-70b-versatile to whichever command
    you run -- the tool's built-in default judge needs Google billing set up,
    which most free accounts don't have.

EXAMPLE
    prompteval --judge-model groq/llama-3.3-70b-versatile

MORE HELP
    prompteval /help            Show this message again
    prompteval run -h           Show flags for a specific command
    See README.md for the full guide.
""".strip("\n")

NO_API_KEY_MESSAGE = """
No API key found, so there's nothing to actually run prompts against yet.

Pick any one of these (all have a free tier):
    Groq:       https://console.groq.com/keys
    OpenRouter: https://openrouter.ai/keys
    Gemini:     https://aistudio.google.com/apikey

Then:
    1. Copy .env.example to .env (or create a .env file yourself)
    2. Paste your key in, e.g.:  GROQ_API_KEY=your-key-here
    3. Run this command again

Run `prompteval /help` for the full guide.
""".strip("\n")


def _has_any_api_key() -> bool:
    return any(os.environ.get(key) for key in _API_KEY_VARS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prompteval", description="Test prompt variants against LLMs and rank them.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an evaluation from an input JSON file.")
    run_parser.add_argument("input_file", help="Path to an EvalRun JSON file.")
    run_parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"Model used to judge outputs (default: {DEFAULT_JUDGE_MODEL}).",
    )
    run_parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (default: {DEFAULT_CONCURRENCY}).",
    )
    run_parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Per-call timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    run_parser.add_argument(
        "--format", choices=["json", "markdown"], default="json",
        help="Report file format to write (default: json).",
    )

    init_parser = subparsers.add_parser("init", help="Interactively build a new input file by answering questions.")
    init_parser.add_argument(
        "output_file", nargs="?", default=None,
        help="Where to save the input file (default: <task_name>.json).",
    )

    quickstart_parser = subparsers.add_parser(
        "quickstart",
        help="AI generates a task + 5 test cases (or use your own task); you write 3 prompts; runs immediately.",
    )
    quickstart_parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model used to generate the task/test cases, run the prompts, and judge them (default: {DEFAULT_MODEL}).",
    )
    quickstart_parser.add_argument(
        "--judge-model", default=None,
        help="Model used to judge outputs (default: same as --model).",
    )
    quickstart_parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (default: {DEFAULT_CONCURRENCY}).",
    )
    quickstart_parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Per-call timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )

    improve_parser = subparsers.add_parser(
        "improve",
        help="Submit ONE prompt; AI tests it against generated scenarios and returns a score + a rewritten, improved prompt.",
    )
    improve_parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model used to generate scenarios and run the prompt (default: {DEFAULT_MODEL}).",
    )
    improve_parser.add_argument(
        "--judge-model", default=None,
        help="Model used to write the critique (default: same as --model).",
    )
    improve_parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (default: {DEFAULT_CONCURRENCY}).",
    )
    improve_parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Per-call timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )

    return parser


async def run_pipeline(run: EvalRun, judge_model: str, concurrency: int, timeout: float) -> RunReport:
    rendered = render_matrix(run)
    # One semaphore shared between execution and judging (per Plan.md), so the two
    # phases together never exceed `concurrency` calls in flight -- not `concurrency`
    # execution calls followed by an unthrottled burst of judge calls on top.
    semaphore = asyncio.Semaphore(concurrency)
    raw_results = await execute_matrix(rendered, run.models, concurrency=concurrency, timeout=timeout, semaphore=semaphore)
    judge_results = await asyncio.gather(
        *[judge_call_result(r, run.judge_criteria, judge_model, semaphore) for r in raw_results]
    )
    return build_run_report(run, raw_results, list(judge_results))


_SUBCOMMANDS = {"run", "init", "quickstart", "improve"}
_HELP_TOKENS = {"/help", "help", "-h", "--help"}

_MENU_CHOICES = [
    questionary.Choice("Improve one prompt (score + rewrite)", value="improve"),
    questionary.Choice("Compare 3 prompts against AI-generated test data", value="quickstart"),
    questionary.Choice("Build a reusable input file", value="init"),
    questionary.Choice("Run an existing input file", value="run"),
    questionary.Choice("Help", value="help"),
    questionary.Choice("Exit", value="exit"),
]


def _fallback_select(question: str, choices: list) -> str | None:
    """Plain numbered-list menu, read via input(). Used when questionary's
    arrow-key menu can't render at all -- some terminals (e.g. Git Bash/mintty
    on Windows) don't expose a real console screen buffer, which crashes
    questionary outright rather than just looking worse."""
    print(question)
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice.title}")
    answer = input("Enter a number: ").strip()
    try:
        index = int(answer) - 1
        if 0 <= index < len(choices):
            return choices[index].value
    except ValueError:
        pass
    print("Not a valid choice -- try again.\n")
    return _fallback_select(question, choices)


def _ask_select(question: str, choices: list) -> str | None:
    try:
        return questionary.select(question, choices=choices).ask()
    except Exception:  # noqa: BLE001 -- if the fancy menu can't render, fall back instead of crashing
        return _fallback_select(question, choices)


def _ask_text(question: str, default: str = "") -> str:
    try:
        return questionary.text(question, default=default).ask() or ""
    except Exception:  # noqa: BLE001 -- same fallback reasoning as _ask_select
        prompt = f"{question} [{default}]: " if default else f"{question}: "
        return input(prompt).strip() or default


def _run_interactive_menu(existing_argv: list[str]) -> list[str] | None:
    """Shown when no subcommand is given (arrow keys, not typing a command name).
    Returns the full argv to parse next (any flags already typed are preserved,
    e.g. `prompteval --judge-model x` still applies whatever gets picked), or
    None if the user chose to exit (or hit Ctrl+C/Esc, which questionary also
    reports as None)."""
    while True:
        choice = _ask_select("What do you want to do?", _MENU_CHOICES)

        if choice is None or choice == "exit":
            return None

        if choice == "help":
            print(HELP_TEXT)
            print()
            continue

        if choice == "run":
            file_path = _ask_text("Path to your input file:", default="examples/sample_run.json")
            if not file_path:  # blank/cancelled -- back to the menu rather than erroring
                continue
            return [choice, file_path, *existing_argv]

        if choice == "init":
            output_file = _ask_text("Save as (leave blank for an automatic name):")
            extra = [output_file] if output_file else []
            return [choice, *extra, *existing_argv]

        return [choice, *existing_argv]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    if argv is None:
        argv = sys.argv[1:]

    # A bare help request as the very first word always shows the friendly guide,
    # not argparse's terser default. `prompteval run -h` (help for a *specific*
    # command) still falls through to argparse below, since "run" is argv[0] there.
    if argv and argv[0] in _HELP_TOKENS:
        print(HELP_TEXT)
        return 0

    # No subcommand (or flags with no subcommand, e.g. `prompteval --judge-model x`)
    # shows an interactive menu instead of guessing -- `run`/`init`/`quickstart`/
    # `improve` still work exactly as before when named explicitly.
    if not argv or argv[0] not in _SUBCOMMANDS:
        resolved = _run_interactive_menu(argv)
        if resolved is None:
            return 0
        argv = resolved
    args = build_parser().parse_args(argv)

    if args.command in ("run", "quickstart", "improve") and not _has_any_api_key():
        print(NO_API_KEY_MESSAGE, file=sys.stderr)
        return 1

    if args.command == "init":
        run = run_init_wizard()
        output_path = Path(args.output_file) if args.output_file else Path(f"{safe_filename_stub(run.task_name)}.json")
        output_path.write_text(json.dumps(run.model_dump(), indent=2), encoding="utf-8")
        print(f"\nSaved to: {output_path}")
        print(f"Run it with: prompteval run {output_path}")
        return 0

    if args.command == "improve":
        try:
            prompt_template, scenario_values = asyncio.run(run_improve_wizard(args.model))
        except GenerationError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

        run = EvalRun(
            task_name=f"Prompt feedback: {prompt_template[:60]}",
            models=[args.model],
            prompt_variants=[PromptVariant(id="v1", name="Your prompt", template=prompt_template)],
            test_cases=[
                TestCase(id=f"tc{i}", variables={QUICKSTART_VARIABLE: v})
                for i, v in enumerate(scenario_values, start=1)
            ],
            judge_criteria="n/a",  # unused -- improve mode scores/rewrites instead of ranking
        )

        try:
            rendered = render_matrix(run)
        except TemplateRenderError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        raw_results = asyncio.run(
            execute_matrix(rendered, run.models, concurrency=args.concurrency, timeout=args.timeout)
        )

        judge_model = args.judge_model or args.model
        try:
            feedback = asyncio.run(generate_feedback(prompt_template, raw_results, judge_model))
        except GenerationError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

        print(f"\nScore: {feedback.score}/5")
        print(feedback.reasoning)
        print("\nImproved prompt:")
        print(feedback.improved_prompt)

        feedback_path = write_feedback_report(prompt_template, raw_results, feedback)
        print(f"\nFeedback saved to: {feedback_path}")
        return 0

    if args.command == "quickstart":
        try:
            run = asyncio.run(run_quickstart_wizard(args.model))
        except GenerationError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

        input_path = Path(f"{safe_filename_stub(run.task_name)}.json")
        input_path.write_text(json.dumps(run.model_dump(), indent=2), encoding="utf-8")
        print(f"\nSaved input file to: {input_path} (re-run it later with `prompteval run {input_path}`)\n")

        judge_model = args.judge_model or args.model
        try:
            report = asyncio.run(run_pipeline(run, judge_model, args.concurrency, args.timeout))
        except TemplateRenderError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(format_table(report))
        report_path = write_json_report(report)
        print(f"\nReport written to: {report_path}")
        return 0

    if args.command != "run":
        return 1

    try:
        with open(args.input_file, encoding="utf-8") as f:
            run = EvalRun(**json.load(f))
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Error: {args.input_file} is not valid JSON: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"Error: {args.input_file} doesn't match the expected EvalRun format:\n{exc}", file=sys.stderr)
        return 1

    try:
        report = asyncio.run(run_pipeline(run, args.judge_model, args.concurrency, args.timeout))
    except TemplateRenderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_table(report))
    write_fn = write_markdown_report if args.format == "markdown" else write_json_report
    path = write_fn(report)
    print(f"\nReport written to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
