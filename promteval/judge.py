"""T4.1-T4.3: judge engine. Scores each LLMCallResult against the run's rubric.

A failed LLM call is never sent to the judge -- it's scored 0 immediately. A judge
response that isn't valid JSON, OR a judge call that itself fails (rate limit,
timeout, etc.), is retried once with a stricter reminder; if it still doesn't
produce a usable score, it falls back to score=0 rather than raising.
"""

import asyncio
import json
import re

import litellm

from promteval.schemas import JudgeResult, LLMCallResult

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator. Score the candidate output against the rubric below.

## Rubric
{judge_criteria}

## Candidate Output
{output}

## Instructions
- Return ONLY a JSON object with exactly these keys: "score" (integer 1-5) and "reasoning" (string).
- Score 1 = fails the rubric entirely; 5 = fully meets the rubric.
- Be strict -- do not inflate scores.
- Do not include markdown fences or any text outside the JSON object."""

RETRY_REMINDER = "\n\nReturn ONLY valid JSON."

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_id_for(result: LLMCallResult) -> str:
    """The composite key JudgeResult.call_id links back to, since LLMCallResult
    has no id field of its own: variant + test case + model together are unique."""
    return f"{result.variant_id}:{result.test_case_id}:{result.model}"


def _parse_judge_response(text: str) -> tuple[int, str] | None:
    """Extract {"score": int, "reasoning": str} from the judge's raw text, tolerating
    markdown fences or stray text around the JSON. Returns None if unparseable."""
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        score = int(data["score"])
        reasoning = str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not 1 <= score <= 5:
        return None
    return score, reasoning


async def judge_call_result(
    call_result: LLMCallResult,
    judge_criteria: str,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> JudgeResult:
    """Score one LLMCallResult. Never raises: execution failures, judge-call failures
    (rate limits, timeouts, etc.), and unparseable judge responses all resolve to a
    JudgeResult with score=0 instead of propagating an exception.

    `semaphore` should be the SAME semaphore used for execution calls (see cli.py) --
    per Plan.md, judge calls share the concurrency limit rather than firing unthrottled
    on top of it.
    """
    call_id = call_id_for(call_result)

    if call_result.error is not None:
        return JudgeResult(
            score=0,
            reasoning=f"execution failed: {call_result.error}",
            call_id=call_id,
        )

    prompt = JUDGE_PROMPT_TEMPLATE.format(judge_criteria=judge_criteria, output=call_result.output)
    last_error: str | None = None

    for _ in range(2):  # 1 initial attempt + 1 retry, per Plan.md
        try:
            async with semaphore:
                response = await litellm.acompletion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
        except Exception as exc:  # noqa: BLE001 -- any judge-call failure must become score=0, never raise
            last_error = str(exc)
            prompt += RETRY_REMINDER
            continue

        parsed = _parse_judge_response(response.choices[0].message.content)
        if parsed is not None:
            score, reasoning = parsed
            return JudgeResult(score=score, reasoning=reasoning, call_id=call_id)
        prompt += RETRY_REMINDER

    reasoning = f"judge call failed: {last_error}" if last_error else "judge parse failure"
    return JudgeResult(score=0, reasoning=reasoning, call_id=call_id)
