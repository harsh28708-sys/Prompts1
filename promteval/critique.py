"""`prompteval improve`: takes ONE prompt (not multiple variants), runs it for real
against a few AI-generated scenarios, and asks a model to score it AND rewrite a
better version -- not a comparison against other variants, and not just prose.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import litellm
from pydantic import ValidationError

from promteval.generator import GenerationError
from promteval.reporter import safe_filename_stub
from promteval.schemas import LLMCallResult, PromptFeedback

IMPROVE_PROMPT_TEMPLATE = """You are an expert prompt engineer scoring and improving a prompt.

## The Prompt Being Tested
{prompt_template}

## Real Outputs From Running This Prompt
{examples}

## Instructions
Return ONLY a JSON object with exactly these keys:
- "score": integer 1-5 for how well this prompt works (5 = excellent, 1 = poor)
- "reasoning": a short, plain-English explanation (2-3 sentences) for the score,
  pointing to specific examples above
- "improved_prompt": a rewritten, better version of the prompt that fixes the
  issues you found

Do not include markdown fences or any text outside the JSON object."""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_examples(results: list[LLMCallResult]) -> str:
    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]

    examples = "\n\n".join(
        f"Example {i}:\nInput: {r.rendered_prompt}\nOutput: {r.output}" for i, r in enumerate(successes, start=1)
    )
    if failures:
        examples += f"\n\n(Note: {len(failures)} of {len(results)} test runs failed and are excluded above.)"
    return examples


def _parse_feedback(text: str) -> PromptFeedback | None:
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        return PromptFeedback(
            score=int(data["score"]),
            reasoning=str(data["reasoning"]),
            improved_prompt=str(data["improved_prompt"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
        return None


async def generate_feedback(prompt_template: str, results: list[LLMCallResult], model: str) -> PromptFeedback:
    successes = [r for r in results if r.error is None]
    if not successes:
        raise GenerationError(
            "every test run of your prompt failed, so there's nothing real to evaluate -- "
            "check your model name and API key"
        )

    prompt = IMPROVE_PROMPT_TEMPLATE.format(prompt_template=prompt_template, examples=_format_examples(results))
    last_error: str | None = None

    for _ in range(2):  # 1 initial attempt + 1 retry, matching the judge's parse-retry pattern
        try:
            response = await litellm.acompletion(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.3
            )
        except Exception as exc:  # noqa: BLE001 -- a failed attempt just triggers the retry below
            last_error = str(exc)
            prompt += "\n\n(The previous attempt failed to respond -- please try again.)"
            continue

        parsed = _parse_feedback(response.choices[0].message.content)
        if parsed is not None:
            return parsed
        prompt += '\n\nReturn ONLY a valid JSON object with "score", "reasoning", and "improved_prompt".'

    if last_error:
        raise GenerationError(f"couldn't generate feedback: {last_error}")
    raise GenerationError("the model didn't return usable feedback (invalid JSON) after 2 attempts")


def write_feedback_report(
    prompt_template: str,
    results: list[LLMCallResult],
    feedback: PromptFeedback,
    output_dir: str | Path = ".",
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stub = safe_filename_stub(prompt_template) or "prompt_feedback"
    path = Path(output_dir) / f"{stub}_feedback_{timestamp}.md"

    lines = [
        "# Prompt Feedback",
        "",
        "## Your Prompt",
        "```",
        prompt_template,
        "```",
        "",
        f"## Score: {feedback.score}/5",
        feedback.reasoning,
        "",
        "## Improved Prompt",
        "```",
        feedback.improved_prompt,
        "```",
        "",
        "## Test Scenarios & Outputs",
    ]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. Input: {r.rendered_prompt}")
        lines.append(f"   {'Failed: ' + r.error if r.error else 'Output: ' + str(r.output)}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
