"""Tests for T6.1 (aggregation/ranking) and T6.2 (recommendation rationale).
Expected numbers below are hand-computed against Plan.md's weighted_score formula,
not just re-derived from the code under test.
"""

import pytest

from promteval.judge import call_id_for
from promteval.schemas import (
    EvalRun,
    JudgeResult,
    LLMCallResult,
    PromptVariant,
    TestCase,
    VariantScore,
)
from promteval.scoring import (
    _rank,
    aggregate_variant_scores,
    build_recommendation,
    build_run_report,
)


def make_run(variant_ids_names: list[tuple[str, str]], n_test_cases: int = 2) -> EvalRun:
    return EvalRun(
        task_name="test run",
        models=["groq/llama-3.3-70b-versatile"],
        prompt_variants=[
            PromptVariant(id=vid, name=name, template="Summarize: {x}") for vid, name in variant_ids_names
        ],
        test_cases=[TestCase(id=f"tc{i}", variables={"x": "hi"}) for i in range(1, n_test_cases + 1)],
        judge_criteria="n/a",
    )


def make_call(variant_id: str, test_case_id: str, latency_ms: float, cost_usd: float | None = None) -> LLMCallResult:
    return LLMCallResult(
        variant_id=variant_id,
        test_case_id=test_case_id,
        model="groq/llama-3.3-70b-versatile",
        rendered_prompt="Summarize: hi",
        output="hi summarized",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def make_judge(call: LLMCallResult, score: int) -> JudgeResult:
    return JudgeResult(score=score, reasoning="n/a", call_id=call_id_for(call))


def test_aggregate_no_cost_redistributes_weights():
    # In plain terms: without knowing prices, a variant with much better answers
    # should still win overall, even if a worse, faster variant is quicker.
    # v1: quality [4, 5] -> avg 4.5, latency [100, 200] -> avg 150
    # v2: quality [2, 3] -> avg 2.5, latency [50, 50]  -> avg 50
    # max_latency = 150 -> latency_score: v1=0.0, v2=1-50/150=0.6667
    # no cost anywhere -> weighted = 0.75*quality + 0.25*latency_score
    # v1 = 0.75*4.5 + 0.25*0       = 3.375
    # v2 = 0.75*2.5 + 0.25*0.66667 = 2.04167
    run = make_run([("v1", "Direct"), ("v2", "Verbose")])
    calls = [
        make_call("v1", "tc1", latency_ms=100), make_call("v1", "tc2", latency_ms=200),
        make_call("v2", "tc1", latency_ms=50), make_call("v2", "tc2", latency_ms=50),
    ]
    judges = [
        make_judge(calls[0], 4), make_judge(calls[1], 5),
        make_judge(calls[2], 2), make_judge(calls[3], 3),
    ]

    scores = aggregate_variant_scores(run, calls, judges)

    assert [s.variant_id for s in scores] == ["v1", "v2"]  # v1 ranked first
    v1, v2 = scores
    assert v1.avg_quality == 4.5
    assert v1.avg_latency_ms == 150
    assert v1.avg_cost_usd is None
    assert v1.weighted_score == pytest.approx(3.375)
    assert v2.weighted_score == pytest.approx(2.04167, abs=1e-4)


def test_aggregate_with_known_cost_uses_full_formula():
    # In plain terms: when we DO know the price, a cheaper option should get a
    # real, measurable boost in the ranking, not just quality and speed mattering.
    # Both variants have known cost -> full 0.60/0.25/0.15 formula applies.
    run = make_run([("v1", "Cheap"), ("v2", "Pricey")], n_test_cases=1)
    c1 = make_call("v1", "tc1", latency_ms=100, cost_usd=0.001)
    c2 = make_call("v2", "tc1", latency_ms=100, cost_usd=0.002)
    judges = [make_judge(c1, 4), make_judge(c2, 4)]

    scores = aggregate_variant_scores(run, [c1, c2], judges)
    v1 = next(s for s in scores if s.variant_id == "v1")
    v2 = next(s for s in scores if s.variant_id == "v2")

    # max_cost=0.002 -> cost_score: v1=1-0.001/0.002=0.5, v2=1-0.002/0.002=0.0
    # max_latency=100 for both -> latency_score=0.0 for both
    # v1 = 0.60*4 + 0.25*0.5 + 0.15*0 = 2.4 + 0.125 = 2.525
    # v2 = 0.60*4 + 0.25*0.0 + 0.15*0 = 2.4
    assert v1.weighted_score == pytest.approx(2.525)
    assert v2.weighted_score == pytest.approx(2.4)
    assert v1.weighted_score > v2.weighted_score  # cheaper variant wins on cost alone


def test_missing_judge_result_counts_as_zero_quality():
    # In plain terms: an answer nobody ever graded should count as a failure (0),
    # not get quietly left out as if it never happened.
    # Defensive path: a raw result with no matching JudgeResult at all (not just a
    # failed call) must still be treated as quality=0, not silently skipped.
    run = make_run([("v1", "Solo")], n_test_cases=1)
    call = make_call("v1", "tc1", latency_ms=100)

    scores = aggregate_variant_scores(run, [call], judge_results=[])

    assert scores[0].avg_quality == 0


def test_rank_tie_break_prefers_higher_quality_on_equal_weighted_score():
    # In plain terms: if two options end up in a dead-even tie, the one with the
    # better answers should be declared the winner of the tiebreaker.
    a = VariantScore(variant_id="a", avg_quality=4.0, avg_latency_ms=100, weighted_score=3.0)
    b = VariantScore(variant_id="b", avg_quality=3.6667, avg_latency_ms=0, weighted_score=3.0)

    ranked = _rank([b, a])  # deliberately passed in the "wrong" order

    assert [s.variant_id for s in ranked] == ["a", "b"]  # higher avg_quality wins the tie


def test_rank_tie_break_prefers_lower_latency_when_quality_also_ties():
    # In plain terms: if it's STILL a tie after that, the faster option should win.
    a = VariantScore(variant_id="a", avg_quality=4.0, avg_latency_ms=200, weighted_score=3.0)
    b = VariantScore(variant_id="b", avg_quality=4.0, avg_latency_ms=100, weighted_score=3.0)

    ranked = _rank([a, b])

    assert [s.variant_id for s in ranked] == ["b", "a"]  # lower latency wins the tie


def test_recommendation_names_quality_as_the_losing_dimension():
    # In plain terms: the "why this one lost" explanation should actually make
    # sense and show real numbers, not just say "it was worse" with no detail.
    run = make_run([("v1", "Good"), ("v2", "Bad")], n_test_cases=1)
    winner = VariantScore(variant_id="v1", avg_quality=4.5, avg_latency_ms=100, weighted_score=3.375)
    loser = VariantScore(variant_id="v2", avg_quality=2.0, avg_latency_ms=100, weighted_score=1.5)

    recommended_id, rationale = build_recommendation([winner, loser], run)

    assert recommended_id == "v1"
    assert "'Good'" in rationale
    assert "'Bad'" in rationale
    assert "lower quality" in rationale
    assert "2.00" in rationale and "4.50" in rationale


def test_recommendation_with_single_variant_has_no_loser_lines():
    # In plain terms: with only one option to begin with, the report shouldn't
    # awkwardly talk about other options "losing" when there weren't any.
    run = make_run([("v1", "Only")], n_test_cases=1)
    winner = VariantScore(variant_id="v1", avg_quality=4.0, avg_latency_ms=100, weighted_score=3.0)

    recommended_id, rationale = build_recommendation([winner], run)

    assert recommended_id == "v1"
    assert "lost" not in rationale


def test_build_run_report_end_to_end_with_a_forced_failure():
    # In plain terms: if one AI call totally fails, you should still get a
    # complete, usable report at the end instead of the whole thing crashing.
    # One call fails outright -- proves the whole PBI-6 pipeline still produces a
    # complete RunReport instead of crashing on a partial batch (echoes PBI-3's DoD).
    run = make_run([("v1", "Direct")], n_test_cases=2)
    ok_call = make_call("v1", "tc1", latency_ms=100)
    failed_call = LLMCallResult(
        variant_id="v1", test_case_id="tc2", model="groq/llama-3.3-70b-versatile",
        rendered_prompt="Summarize: hi", error="timeout after 3 retries",
    )
    judges = [
        make_judge(ok_call, 5),
        JudgeResult(score=0, reasoning="execution failed: timeout after 3 retries", call_id=call_id_for(failed_call)),
    ]

    report = build_run_report(run, [ok_call, failed_call], judges)

    assert report.recommended_variant_id == "v1"
    assert len(report.raw_results) == 2
    assert len(report.judge_results) == 2
    assert report.variant_scores[0].avg_quality == 2.5  # mean of 5 and 0
