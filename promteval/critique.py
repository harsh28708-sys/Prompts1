"""`prompteval improve`: takes ONE prompt (not multiple variants), runs it for real
against a few AI-generated scenarios, and asks a model to write plain-English
feedback on how to improve the prompt -- not a 1-5 score, a written critique.
"""

from datetime import UTC, datetime
from pathlib import Path

import litellm

from promteval.generator import GenerationError
from promteval.reporter import safe_filename_stub
from promteval.schemas import LLMCallResult

CRITIQUE_PROMPT_TEMPLATE = """You are an expert prompt engineer helping someone improve a prompt they wrote.

## Context (what they're trying to do)
{context}

## The Prompt They Wrote
{prompt_template}

## Real Outputs From Running This Prompt
{examples}

## Instructions
Write a short, plain-English critique for someone new to prompt engineering. Cover:
1. What's working well.
2. What's not working -- point to specific examples above.
3. 2-4 concrete, actionable suggestions to improve the prompt, with example rewordings where helpful.

No jargon. Be encouraging but honest. Keep the whole thing under 300 words."""


def _format_examples(results: list[LLMCallResult]) -> str:
    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]

    examples = "\n\n".join(
        f"Example {i}:\nInput: {r.rendered_prompt}\nOutput: {r.output}" for i, r in enumerate(successes, start=1)
    )
    if failures:
        examples += f"\n\n(Note: {len(failures)} of {len(results)} test runs failed and are excluded above.)"
    return examples


async def generate_critique(context: str, prompt_template: str, results: list[LLMCallResult], model: str) -> str:
    successes = [r for r in results if r.error is None]
    if not successes:
        raise GenerationError(
            "every test run of your prompt failed, so there's nothing real to critique -- "
            "check your model name and API key"
        )

    prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        context=context, prompt_template=prompt_template, examples=_format_examples(results)
    )

    try:
        response = await litellm.acompletion(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.3
        )
    except Exception as exc:
        raise GenerationError(f"couldn't generate a critique: {exc}") from exc

    text = response.choices[0].message.content.strip()
    if not text:
        raise GenerationError("the model returned an empty critique")
    return text


def write_critique_report(
    context: str,
    prompt_template: str,
    results: list[LLMCallResult],
    critique: str,
    output_dir: str | Path = ".",
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stub = safe_filename_stub(context) or "prompt_feedback"
    path = Path(output_dir) / f"{stub}_feedback_{timestamp}.md"

    lines = [
        "# Prompt Improvement Feedback",
        "",
        "## Context",
        context,
        "",
        "## Your Prompt",
        "```",
        prompt_template,
        "```",
        "",
        "## Test Scenarios & Outputs",
    ]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. Input: {r.rendered_prompt}")
        lines.append(f"   {'Failed: ' + r.error if r.error else 'Output: ' + str(r.output)}")
        lines.append("")

    lines += ["## Critique", critique]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
