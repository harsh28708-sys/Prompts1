"""T6.1-T6.2: aggregate judge scores per variant, rank them, and recommend a winner.

VariantScore has no `rank` field (see schemas.py) -- rank is implicit: the returned
list is sorted best-first, so index 0 is always the winner.
"""

from promteval.judge import call_id_for
from promteval.schemas import EvalRun, JudgeResult, LLMCallResult, RunReport, VariantScore

QUALITY_WEIGHT = 0.60
COST_WEIGHT = 0.25
LATENCY_WEIGHT = 0.15
# Plan.md: "if cost unknown for all, weight redistributed: 0.75 quality + 0.25 latency".
# We take the same redistributed weights whenever ANY variant's cost is unknown, since
# comparing "$0.002" against "unknown" isn't meaningful and the plan doesn't define a
# partial-cost case. In practice this is the only case that occurs today: cost_usd is
# always None until PBI-5 (pricing table) is built.
NO_COST_QUALITY_WEIGHT = 0.75
NO_COST_LATENCY_WEIGHT = 0.25


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rank(scores: list[VariantScore]) -> list[VariantScore]:
    """Sort best-first by weighted_score descending; ties broken by higher
    avg_quality, then lower avg_latency_ms (per Plan.md's scoring rules)."""
    return sorted(scores, key=lambda vs: (-vs.weighted_score, -vs.avg_quality, vs.avg_latency_ms))


def aggregate_variant_scores(
    run: EvalRun,
    raw_results: list[LLMCallResult],
    judge_results: list[JudgeResult],
) -> list[VariantScore]:
    """Group results by variant, average quality/cost/latency, and rank by
    weighted_score descending (tie-break: higher avg_quality, then lower avg_latency_ms).
    Returns variants sorted best-first -- index 0 is the winner."""
    judge_by_call_id = {jr.call_id: jr for jr in judge_results}

    raw_stats: dict[str, dict] = {}
    for variant in run.prompt_variants:
        variant_calls = [r for r in raw_results if r.variant_id == variant.id]

        # A judged score of 0 covers both "the call failed" and "the judge gave up" --
        # either way, Plan.md says failed executions count as 0 toward avg_quality.
        qualities = [
            judge_by_call_id[call_id_for(call)].score
            if call_id_for(call) in judge_by_call_id
            else 0
            for call in variant_calls
        ]
        costs = [c.cost_usd for c in variant_calls if c.cost_usd is not None]
        latencies = [c.latency_ms for c in variant_calls if c.latency_ms is not None]

        raw_stats[variant.id] = {
            "avg_quality": _mean(qualities),
            "avg_cost_usd": _mean(costs) if costs else None,
            "avg_latency_ms": _mean(latencies),
        }

    cost_known = bool(raw_stats) and all(s["avg_cost_usd"] is not None for s in raw_stats.values())
    max_cost = max((s["avg_cost_usd"] for s in raw_stats.values()), default=0.0) if cost_known else 0.0
    max_latency = max((s["avg_latency_ms"] for s in raw_stats.values()), default=0.0)

    scored = []
    for variant_id, stats in raw_stats.items():
        latency_score = 1.0 if max_latency == 0 else 1 - (stats["avg_latency_ms"] / max_latency)

        if cost_known:
            cost_score = 1.0 if max_cost == 0 else 1 - (stats["avg_cost_usd"] / max_cost)
            weighted = (
                QUALITY_WEIGHT * stats["avg_quality"]
                + COST_WEIGHT * cost_score
                + LATENCY_WEIGHT * latency_score
            )
        else:
            weighted = (
                NO_COST_QUALITY_WEIGHT * stats["avg_quality"]
                + NO_COST_LATENCY_WEIGHT * latency_score
            )

        scored.append(
            VariantScore(
                variant_id=variant_id,
                avg_quality=stats["avg_quality"],
                avg_cost_usd=stats["avg_cost_usd"],
                avg_latency_ms=stats["avg_latency_ms"],
                weighted_score=weighted,
            )
        )

    return _rank(scored)


def _loss_reason(winner: VariantScore, loser: VariantScore) -> str:
    """Name the dimension that contributed most to `loser` scoring below `winner`,
    using each metric's relative gap so scores/dollars/milliseconds are comparable."""
    reasons = []

    quality_gap = (winner.avg_quality - loser.avg_quality) / winner.avg_quality if winner.avg_quality > 0 else 0.0
    reasons.append((quality_gap, f"lower quality ({loser.avg_quality:.2f} vs {winner.avg_quality:.2f})"))

    latency_gap = (
        (loser.avg_latency_ms - winner.avg_latency_ms) / winner.avg_latency_ms if winner.avg_latency_ms > 0 else 0.0
    )
    reasons.append((latency_gap, f"slower ({loser.avg_latency_ms:.0f}ms vs {winner.avg_latency_ms:.0f}ms)"))

    if winner.avg_cost_usd is not None and loser.avg_cost_usd is not None and winner.avg_cost_usd > 0:
        cost_gap = (loser.avg_cost_usd - winner.avg_cost_usd) / winner.avg_cost_usd
        reasons.append((cost_gap, f"more expensive (${loser.avg_cost_usd:.5f} vs ${winner.avg_cost_usd:.5f})"))

    gap, description = max(reasons, key=lambda r: r[0])
    if gap <= 0:
        return "scored essentially the same on every dimension but ranked lower on tie-break"
    return description


def build_recommendation(variant_scores: list[VariantScore], run: EvalRun) -> tuple[str, str]:
    """Pick the winner and write a rationale naming why every other variant lost."""
    winner = variant_scores[0]
    winner_name = next(v.name for v in run.prompt_variants if v.id == winner.variant_id)

    lines = [f"'{winner_name}' ({winner.variant_id}) is the recommended variant."]
    for loser in variant_scores[1:]:
        loser_name = next(v.name for v in run.prompt_variants if v.id == loser.variant_id)
        lines.append(f"'{loser_name}' ({loser.variant_id}) lost: {_loss_reason(winner, loser)}.")

    return winner.variant_id, " ".join(lines)


def build_run_report(
    run: EvalRun,
    raw_results: list[LLMCallResult],
    judge_results: list[JudgeResult],
) -> RunReport:
    """T6.1-T6.2 wired together: aggregate, rank, and recommend in one call."""
    variant_scores = aggregate_variant_scores(run, raw_results, judge_results)
    recommended_variant_id, rationale = build_recommendation(variant_scores, run)

    return RunReport(
        task_name=run.task_name,
        variant_scores=variant_scores,
        recommended_variant_id=recommended_variant_id,
        rationale=rationale,
        raw_results=raw_results,
        judge_results=judge_results,
    )
