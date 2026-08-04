"""T9.1-T9.2: CLI entry point. Wires load -> render -> execute -> judge -> score -> report
into the single command described in Plan.md: `prompteval run <input.json>`.
"""

import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from promteval.executor import DEFAULT_CONCURRENCY, DEFAULT_TIMEOUT_S, execute_matrix
from promteval.judge import judge_call_result
from promteval.renderer import TemplateRenderError, render_matrix
from promteval.reporter import format_table, write_json_report, write_markdown_report
from promteval.schemas import EvalRun, RunReport
from promteval.scoring import build_run_report

DEFAULT_JUDGE_MODEL = "gemini/gemini-2.0-flash"


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


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

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
