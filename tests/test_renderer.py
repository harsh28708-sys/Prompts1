"""T2.3: renderer tests over the full variant x test-case matrix,
plus a missing-variable case that must raise a clear error."""

import json
from pathlib import Path

import pytest

from promteval.renderer import TemplateRenderError, render_matrix, render_prompt
from promteval.schemas import EvalRun, PromptVariant, TestCase

SAMPLE_RUN_PATH = Path(__file__).parent.parent / "examples" / "sample_run.json"


def load_sample_run() -> EvalRun:
    with open(SAMPLE_RUN_PATH, encoding="utf-8") as f:
        return EvalRun(**json.load(f))


def test_render_prompt_fills_variable():
    variant = PromptVariant(id="v1", name="Direct", template="Summarize: {customer_message}")
    test_case = TestCase(id="tc1", variables={"customer_message": "My order is late"})
    assert render_prompt(variant, test_case) == "Summarize: My order is late"


def test_render_matrix_covers_every_pair():
    run = load_sample_run()
    results = render_matrix(run)

    # 3 variants x 5 test cases from examples/sample_run.json = 15, matching the spec's
    # acceptance criteria (3 variants x 5 calls = 15 total LLM calls downstream).
    assert len(results) == len(run.prompt_variants) * len(run.test_cases) == 15

    pairs = {(r.variant_id, r.test_case_id) for r in results}
    expected_pairs = {
        (v.id, tc.id) for v in run.prompt_variants for tc in run.test_cases
    }
    assert pairs == expected_pairs

    # every rendered prompt actually substituted the variable, not left the placeholder
    for r in results:
        assert "{customer_message}" not in r.prompt


def test_missing_variable_raises_clear_error_not_a_raw_exception():
    variant = PromptVariant(id="v_bad", name="Broken", template="Summarize: {customer_message}")
    test_case = TestCase(id="tc_empty", variables={})  # missing customer_message

    with pytest.raises(TemplateRenderError) as exc_info:
        render_prompt(variant, test_case)

    message = str(exc_info.value)
    assert "v_bad" in message
    assert "tc_empty" in message
    assert "customer_message" in message


def test_missing_variable_in_matrix_stops_with_clear_error():
    run = EvalRun(
        task_name="Broken run",
        models=["gemini/gemini-2.5-flash"],
        prompt_variants=[PromptVariant(id="v1", name="Direct", template="Summarize: {customer_message}")],
        test_cases=[TestCase(id="tc1", variables={"wrong_key": "oops"})],
        judge_criteria="n/a",
    )
    with pytest.raises(TemplateRenderError, match="customer_message"):
        render_matrix(run)
