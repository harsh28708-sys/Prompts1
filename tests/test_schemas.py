"""T1.3 DoD: each schema parses one hand-written sample instance."""

import pytest
from pydantic import ValidationError

from promteval.schemas import (
    EvalRun,
    JudgeResult,
    LLMCallResult,
    PromptVariant,
    RunReport,
    TestCase,
    VariantScore,
)


def test_test_case_sample():
    tc = TestCase(id="tc1", variables={"customer_message": "My order is late"})
    assert tc.id == "tc1"


def test_prompt_variant_sample():
    pv = PromptVariant(id="v1", name="Concise", template="Summarize: {customer_message}")
    assert pv.template.startswith("Summarize")


def test_eval_run_sample():
    run = EvalRun(
        task_name="Summarize support tickets",
        models=["openrouter/meta-llama/llama-3.3-70b", "gemini/gemini-2.5-flash"],
        prompt_variants=[PromptVariant(id="v1", name="Concise", template="Summarize: {customer_message}")],
        test_cases=[TestCase(id="tc1", variables={"customer_message": "My order is late"})],
        judge_criteria="Score 1-5 on accuracy and brevity.",
    )
    assert run.models


def test_llm_call_result_success_sample():
    r = LLMCallResult(
        variant_id="v1",
        test_case_id="tc1",
        model="gemini/gemini-2.5-flash",
        rendered_prompt="Summarize: My order is late",
        output="The customer's order is delayed.",
        input_tokens=12,
        output_tokens=8,
        latency_ms=350.5,
        cost_usd=0.0002,
        error=None,
    )
    assert r.error is None


def test_llm_call_result_error_sample():
    # A failed call still has to be a valid LLMCallResult (see PBI-3 DoD).
    r = LLMCallResult(
        variant_id="v1",
        test_case_id="tc1",
        model="groq/llama-3.3-70b-versatile",
        rendered_prompt="Summarize: My order is late",
        error="timeout after 3 retries",
    )
    assert r.output is None
    assert r.error is not None


def test_judge_result_sample():
    jr = JudgeResult(score=4, reasoning="Accurate but slightly verbose.", call_id="v1:tc1:gemini")
    assert 1 <= jr.score <= 5


def test_judge_result_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        JudgeResult(score=6, reasoning="bad", call_id="v1:tc1:gemini")


def test_variant_score_sample():
    vs = VariantScore(
        variant_id="v1",
        avg_quality=4.2,
        avg_cost_usd=0.0002,
        avg_latency_ms=350.5,
        weighted_score=4.1,
    )
    assert vs.avg_cost_usd is not None


def test_run_report_sample():
    report = RunReport(
        task_name="Summarize support tickets",
        variant_scores=[
            VariantScore(
                variant_id="v1",
                avg_quality=4.2,
                avg_cost_usd=0.0002,
                avg_latency_ms=350.5,
                weighted_score=4.1,
            )
        ],
        recommended_variant_id="v1",
        rationale="v1 scored highest on quality with the lowest cost.",
        raw_results=[],
        judge_results=[],
    )
    assert report.recommended_variant_id == "v1"


def test_strict_schema_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TestCase(id="tc1", variables={}, typo_field="oops")
