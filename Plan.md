# Plan — Prompt Evaluator

## Architecture

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM integration | [LiteLLM](https://github.com/BerriAI/litellm) (unified API for OpenRouter, Gemini, Groq) |
| Schemas | Pydantic v2 |
| Concurrency | `asyncio` + `asyncio.Semaphore` + `asyncio.gather` |
| Retries | `tenacity` (exponential backoff on rate limits / transient errors) |
| Template rendering | Python `str.format` (simple `{variable}` placeholders; no Jinja2 in v1) |
| CLI | `typer` or `argparse` — single entry point: `prompteval run input.json` |
| Output | JSON file (canonical) + human-readable CLI table; optional Markdown export |

### Pipeline Flow

```
input.json (EvalRun)
  → render templates (variant × test case)
  → execute LLM calls in parallel (variant × test case × model)
  → judge each output in parallel
  → aggregate VariantScore per variant
  → generate RunReport + write output
```

---

## Decisions on Open Questions

| Question | Decision |
|---|---|
| **Provider scope** | OpenRouter, Google Gemini, and Groq via LiteLLM. Default models for E2E: `openrouter/meta-llama/llama-3.3-70b-instruct`, `gemini/gemini-2.0-flash`, `groq/llama-3.3-70b-versatile`. Users may override via `EvalRun.models`. |
| **Judge model** | Same as execution unless overridden. Default judge: `gemini/gemini-2.0-flash` (fast, cheap, good at structured output). Configurable via CLI flag `--judge-model`. |
| **Judge determinism** | Judge calls use `temperature=0`, `response_format={"type": "json_object"}` where supported. |
| **Cost handling** | v1: compute `cost_usd` from a static pricing table when token counts are available; `None` for unknown models. Free-tier models → `0.0`. Cost is included in ranking but weighted lower than quality (see formula below). |
| **Persistence format** | Each run writes `{task_name}_{timestamp}.json` (full `RunReport`) and prints a CLI summary table. Optional `--format markdown` for a `.md` report. No database. |
| **Concurrency limit** | Default `10` concurrent LLM calls (configurable via `--concurrency`). Judge calls share the same semaphore. |
| **Per-call timeout** | Default `60s` per LLM call (configurable via `--timeout`). Timeout → `LLMCallResult.error` set, run continues. |
| **Failed execution calls** | Skipped by judge; assigned `JudgeResult` with `score=0`, `reasoning="execution failed: {error}"`. |
| **Template engine** | `str.format` only. Missing variables raise `TemplateRenderError` with the variant and test case IDs. |

---

## Data & API Contracts

### Input: `EvalRun`

```python
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str
    variables: dict[str, str]  # e.g. {"customer_message": "..."}


class PromptVariant(BaseModel):
    id: str                    # e.g. "v1"
    name: str
    template: str              # e.g. "Summarize: {customer_message}"


class EvalRun(BaseModel):
    task_name: str
    models: list[str]          # e.g. ["gemini/gemini-2.0-flash"]
    prompt_variants: list[PromptVariant]
    test_cases: list[TestCase]
    judge_criteria: str        # freeform rubric text injected into judge prompt
```

### Example `input.json`

```json
{
  "task_name": "customer_summarization",
  "models": ["gemini/gemini-2.0-flash"],
  "prompt_variants": [
    {
      "id": "v1",
      "name": "Direct",
      "template": "Summarize this customer message in one sentence:\n\n{customer_message}"
    },
    {
      "id": "v2",
      "name": "Structured",
      "template": "You are a support agent. Summarize the issue and sentiment:\n\n{customer_message}"
    },
    {
      "id": "v3",
      "name": "Bullet points",
      "template": "List the key points from this message as bullets:\n\n{customer_message}"
    }
  ],
  "test_cases": [
    {"id": "tc1", "variables": {"customer_message": "My order arrived broken and I want a refund."}},
    {"id": "tc2", "variables": {"customer_message": "How do I change my subscription plan?"}},
    {"id": "tc3", "variables": {"customer_message": "The app keeps crashing on login."}},
    {"id": "tc4", "variables": {"customer_message": "Can I get an invoice for last month?"}},
    {"id": "tc5", "variables": {"customer_message": "Your support team was very helpful, thank you!"}}
  ],
  "judge_criteria": "Score 1-5 on: accuracy of summary, conciseness, and capture of customer sentiment. 5 = perfect summary."
}
```

### Per-Call Result: `LLMCallResult`

```python
class LLMCallResult(BaseModel):
    call_id: str                 # unique, e.g. "{variant_id}_{test_case_id}_{model}"
    variant_id: str
    test_case_id: str
    model: str
    rendered_prompt: str
    output: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None   # None if pricing unknown
    error: str | None = None        # populated on failure; output/tokens null
```

### Judge Result: `JudgeResult`

```python
class JudgeResult(BaseModel):
    call_id: str                 # links to LLMCallResult.call_id
    score: int = Field(ge=0, le=5)  # 0 = execution or parse failure
    reasoning: str
```

### Aggregated Score: `VariantScore`

```python
class VariantScore(BaseModel):
    variant_id: str
    variant_name: str
    avg_quality: float           # mean JudgeResult.score
    avg_cost_usd: float | None   # mean cost; None if all calls unknown
    avg_latency_ms: float
    weighted_score: float
    rank: int                    # 1 = best
```

### Final Report: `RunReport`

```python
class RunReport(BaseModel):
    task_name: str
    run_timestamp: str           # ISO 8601
    variant_scores: list[VariantScore]   # sorted by rank
    recommended_variant_id: str
    rationale: str
    raw_results: list[LLMCallResult]
    judge_results: list[JudgeResult]
```

---

## Judge Prompt Design

```
You are a strict evaluator. Score the candidate output against the rubric below.

## Rubric
{judge_criteria}

## Candidate Output
{output}

## Instructions
- Return ONLY a JSON object with exactly these keys: "score" (integer 1-5) and "reasoning" (string).
- Score 1 = fails the rubric entirely; 5 = fully meets the rubric.
- Be strict — do not inflate scores.
- Do not include markdown fences or any text outside the JSON object.
```

On parse failure: retry once with an appended reminder (`Return ONLY valid JSON.`). On second failure: `score=0`, `reasoning="judge parse failure"`.

---

## Scoring & Ranking

Per variant, across all test cases (and models if multiple):

- `avg_quality` = mean of judge scores (failed executions count as 0)
- `avg_cost_usd` = mean of `cost_usd` (ignore `None` values; if all `None`, set to `None`)
- `avg_latency_ms` = mean latency

**Weighted score** (higher is better):

```
weighted_score = (0.60 × avg_quality)
               + (0.25 × cost_score)
               + (0.15 × latency_score)
```

Where:
- `cost_score` = normalized inverse cost: `1 - (cost / max_cost)` across variants (if cost unknown for all, weight redistributed: 0.75 quality + 0.25 latency)
- `latency_score` = normalized inverse latency: `1 - (latency / max_latency)`

Rank variants by `weighted_score` descending. Tie-break: higher `avg_quality`, then lower `avg_latency_ms`.

The recommendation rationale must name, for each non-winning variant, which dimension (quality, cost, or latency) caused it to lose.

---

## Edge Cases & Reliability

1. **Rate limits** — exponential backoff (tenacity): 3 retries, wait 2s → 4s → 8s.
2. **Bad judge JSON** — retry once, then fallback score 0.
3. **Partial batch failure** — completed results are always included in `RunReport`; errored calls have `error` set and score 0.
4. **Hung calls** — per-call timeout cancels the request and records timeout in `error`.
5. **Unknown model pricing** — `cost_usd = None`; ranking falls back to quality + latency only.

---

## Project Structure (proposed)

```
prompteval/
├── pyproject.toml
├── README.md
├── .env.example
├── examples/
│   └── customer_summarization.json
├── src/prompteval/
│   ├── __init__.py
│   ├── cli.py              # entry point
│   ├── schemas.py          # Pydantic models
│   ├── renderer.py         # template rendering
│   ├── executor.py         # async LLM calls via litellm
│   ├── judge.py            # judge prompt + parsing
│   ├── scoring.py          # aggregation + ranking
│   ├── reporter.py         # CLI table + JSON/MD output
│   └── pricing.py          # static pricing table
└── tests/
    ├── test_renderer.py
    ├── test_judge.py
    └── test_scoring.py
```
