"""Tests for ui.py (Rich-based terminal presentation). Rich auto-detects
non-terminal output (like pytest's capsys) and renders plain text with no
color codes, so these check the actual wording, not ANSI escape sequences."""

from promteval.schemas import PromptFeedback, RunReport, VariantScore
from promteval.ui import (
    _score_style,
    print_error,
    print_feedback,
    print_report_table,
    print_scenarios,
)


def test_score_style_is_green_for_a_good_score():
    assert "green" in _score_style(5)
    assert "green" in _score_style(4)


def test_score_style_is_yellow_for_a_middling_score():
    assert "yellow" in _score_style(3)


def test_score_style_is_red_for_a_poor_score():
    assert "red" in _score_style(2)
    assert "red" in _score_style(1)


def test_print_feedback_shows_score_reasoning_and_improved_prompt(capsys):
    feedback = PromptFeedback(score=4, reasoning="Solid but a bit vague.", improved_prompt="Summarize concisely: {input}")

    print_feedback(feedback)

    out = capsys.readouterr().out
    assert "4/5" in out
    assert "Solid but a bit vague." in out
    assert "Improved Prompt" in out
    assert "Summarize concisely: {input}" in out


def test_print_scenarios_lists_every_value(capsys):
    print_scenarios(["first one", "second one", "third one"])

    out = capsys.readouterr().out
    assert "first one" in out
    assert "second one" in out
    assert "third one" in out


def test_print_error_writes_to_stderr_not_stdout(capsys):
    print_error("something went wrong")

    captured = capsys.readouterr()
    assert "something went wrong" in captured.err
    assert "something went wrong" not in captured.out


def make_report() -> RunReport:
    return RunReport(
        task_name="Test task",
        variant_scores=[
            VariantScore(variant_id="v1", avg_quality=4.5, avg_latency_ms=200, weighted_score=3.5),
            VariantScore(variant_id="v2", avg_quality=3.0, avg_latency_ms=100, weighted_score=2.5),
        ],
        recommended_variant_id="v1",
        rationale="'v1' won because it scored higher.",
        raw_results=[],
        judge_results=[],
    )


def test_print_report_table_marks_the_winner(capsys):
    print_report_table(make_report())

    out = capsys.readouterr().out
    assert "v1" in out and "v2" in out
    assert "<- winner" in out
    assert "Recommended: v1" in out
    assert "won because it scored higher" in out


def test_print_report_table_does_not_mark_the_loser(capsys):
    print_report_table(make_report())

    out = capsys.readouterr().out
    # exactly one winner marker, not one per row
    assert out.count("<- winner") == 1
