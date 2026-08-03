"""Pydantic data contracts for PromtEval, per Plan.md."""

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TestCase(Strict):
    __test__ = False  # tell pytest this isn't a test class, just named like one

    id: str
    variables: dict[str, str]


class PromptVariant(Strict):
    id: str
    name: str
    template: str


class EvalRun(Strict):
    task_name: str
    models: list[str]
    prompt_variants: list[PromptVariant]
    test_cases: list[TestCase]
    judge_criteria: str


class LLMCallResult(Strict):
    variant_id: str
    test_case_id: str
    model: str
    rendered_prompt: str
    # Optional: a failed call (see `error`) has no output/tokens/latency to report.
    output: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None  # None if pricing table lacks that model
    error: str | None = None  # populated on failure, output/tokens left null


class JudgeResult(Strict):
    score: int = Field(ge=1, le=5)
    reasoning: str
    call_id: str  # links back to the LLMCallResult it graded


class VariantScore(Strict):
    variant_id: str
    avg_quality: float
    # Optional: a variant averaging in an unpriced model has no known cost.
    avg_cost_usd: float | None = None
    avg_latency_ms: float
    weighted_score: float


class RunReport(Strict):
    task_name: str
    variant_scores: list[VariantScore]
    recommended_variant_id: str
    rationale: str
    raw_results: list[LLMCallResult]
    judge_results: list[JudgeResult]
