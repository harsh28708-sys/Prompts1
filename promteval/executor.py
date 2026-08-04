"""T3.1-T3.4: async multi-LLM execution engine.

Runs every (variant, test_case, model) combination through litellm, bounded by a
concurrency limit, with retry/backoff on transient errors. A single call failing
never aborts the batch -- it's recorded on LLMCallResult.error instead.
"""

import asyncio
import time

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from promteval.renderer import RenderedPrompt, render_matrix
from promteval.schemas import EvalRun, LLMCallResult

DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT_S = 60.0

# Transient/server-side errors worth retrying. Auth, bad-request, and content-policy
# errors are deliberately excluded -- retrying those just wastes 3 more calls on a
# guaranteed failure. Timeout is excluded too: per Plan.md, a hung call is recorded
# as an error directly rather than retried.
RETRYABLE_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.BadGatewayError,
)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    stop=stop_after_attempt(4),  # 1 initial try + 3 retries: 2s -> 4s -> 8s
    reraise=True,
)
async def _call_model(model: str, prompt: str, timeout: float) -> litellm.ModelResponse:
    return await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )


async def execute_call(
    variant_id: str,
    test_case_id: str,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> LLMCallResult:
    """Run one prompt against one model. Never raises: any failure (after retries
    are exhausted, or a timeout) is captured in LLMCallResult.error."""
    async with semaphore:
        start = time.monotonic()
        try:
            response = await _call_model(model, prompt, timeout)
        except Exception as exc:  # noqa: BLE001 -- any failure must become LLMCallResult.error, never raise
            return LLMCallResult(
                variant_id=variant_id,
                test_case_id=test_case_id,
                model=model,
                rendered_prompt=prompt,
                error=str(exc),
            )

        latency_ms = (time.monotonic() - start) * 1000
        usage = response.usage
        return LLMCallResult(
            variant_id=variant_id,
            test_case_id=test_case_id,
            model=model,
            rendered_prompt=prompt,
            output=response.choices[0].message.content,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
        )


async def execute_matrix(
    rendered_prompts: list[RenderedPrompt],
    models: list[str],
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[LLMCallResult]:
    """Fire every (rendered prompt) x model combination concurrently, at most
    `concurrency` calls in flight at once."""
    semaphore = asyncio.Semaphore(concurrency)
    calls = [
        execute_call(rp.variant_id, rp.test_case_id, model, rp.prompt, semaphore, timeout)
        for rp in rendered_prompts
        for model in models
    ]
    return await asyncio.gather(*calls)


async def execute_run(
    run: EvalRun,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[LLMCallResult]:
    """Convenience wrapper: render + execute a full EvalRun in one call."""
    rendered = render_matrix(run)
    return await execute_matrix(rendered, run.models, concurrency, timeout)
