"""Tests for T6.3: report formatting and file output."""

import json

from promteval.reporter import (
    format_markdown,
    format_table,
    write_json_report,
    write_markdown_report,
)
from promteval.schemas import RunReport, VariantScore


def make_report() -> RunReport:
    return RunReport(
        task_name="Summarize support tickets",
        variant_scores=[
            VariantScore(variant_id="v1", avg_quality=4.5, avg_latency_ms=226, weighted_score=3.456),
            VariantScore(variant_id="v2", avg_quality=4.5, avg_latency_ms=336, weighted_score=3.375),
        ],
        recommended_variant_id="v1",
        rationale="'v1' is the recommended variant. 'v2' lost: slower (336ms vs 226ms).",
        raw_results=[],
        judge_results=[],
    )


def test_format_table_includes_winner_and_rationale():
    table = format_table(make_report())

    assert "Summarize support tickets" in table
    assert "v1" in table and "v2" in table
    assert "<- winner" in table
    assert "Recommended: v1" in table
    assert "slower (336ms vs 226ms)" in table


def test_format_table_marks_only_the_winner_row():
    table = format_table(make_report())
    lines_with_marker = [line for line in table.splitlines() if "<- winner" in line]

    assert len(lines_with_marker) == 1
    assert lines_with_marker[0].startswith("1")  # rank 1 = the winner


def test_format_markdown_uses_table_syntax_and_bolds_winner():
    md = format_markdown(make_report())

    assert "| Rank | Variant |" in md
    assert "**(winner)**" in md
    assert "**Recommended:** v1" in md


def test_write_json_report_round_trips(tmp_path):
    report = make_report()
    path = write_json_report(report, output_dir=tmp_path)

    assert path.exists()
    assert path.suffix == ".json"
    assert "Summarize_support_tickets" in path.name

    reloaded = RunReport(**json.loads(path.read_text(encoding="utf-8")))
    assert reloaded.recommended_variant_id == report.recommended_variant_id
    assert reloaded.variant_scores[0].variant_id == "v1"


def test_write_markdown_report_creates_a_real_file(tmp_path):
    path = write_markdown_report(make_report(), output_dir=tmp_path)

    assert path.exists()
    assert path.suffix == ".md"
    assert "**Recommended:** v1" in path.read_text(encoding="utf-8")
