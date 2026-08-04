"""AI-generated task ideas and test cases, used by `prompteval quickstart`.

Both functions never raise a raw exception -- a failed generation call or an
unparseable response becomes a GenerationError with a clear message, so the CLI
can report it cleanly instead of crashing.
"""

import json
import re

import litellm

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class GenerationError(Exception):
    """Raised when the generation model's response can't be used, even after a retry."""


async def generate_random_task(model: str) -> str:
    prompt = (
        "Suggest one specific, interesting task that an AI language model could do via "
        "a text prompt (e.g. 'Write a polite decline email for a job applicant', "
        "'Summarize a customer support ticket', 'Explain a legal clause in plain English'). "
        "Respond with ONLY the task description as one short sentence -- no quotes, "
        "no extra commentary, no markdown."
    )
    try:
        response = await litellm.acompletion(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=1.0
        )
    except Exception as exc:
        raise GenerationError(f"couldn't generate a random task: {exc}") from exc

    task = response.choices[0].message.content.strip().strip('"')
    if not task:
        raise GenerationError("the model returned an empty task description")
    return task


async def generate_test_cases(task: str, model: str, n: int = 5) -> list[str]:
    prompt = (
        f'For this task: "{task}"\n\n'
        f"Generate {n} diverse, realistic example inputs a real user would provide for this task. "
        f"Return ONLY a JSON array of exactly {n} strings, no other text, no markdown fences."
    )

    for _ in range(2):  # 1 initial attempt + 1 retry, matching the judge's parse-retry pattern
        try:
            response = await litellm.acompletion(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.9
            )
        except Exception:  # noqa: BLE001 -- a failed attempt just triggers the retry below
            prompt += "\n\n(The previous attempt failed to respond -- please try again.)"
            continue

        match = _JSON_ARRAY_RE.search(response.choices[0].message.content)
        if match:
            try:
                items = json.loads(match.group(0))
                if isinstance(items, list) and len(items) == n and all(isinstance(i, str) for i in items):
                    return items
            except json.JSONDecodeError:
                pass
        prompt += "\n\nReturn ONLY a valid JSON array of strings, nothing else."

    raise GenerationError(f"couldn't get {n} valid test cases from the model after 2 attempts")
