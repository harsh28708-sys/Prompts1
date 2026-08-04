"""T6.3: report formatter -- CLI table, JSON file, optional Markdown export.

RunReport has no run_timestamp field (see schemas.py), so the timestamp used in the
output filename is generated here, at write time, rather than read off the report.
"""

from datetime import UTC, datetime
from pathlib import Path

from promteval.schemas import RunReport

_COLUMNS = ("Rank", "Variant", "Quality", "Cost ($)", "Latency (ms)", "Weighted")
_WIDTHS = (5, 10, 9, 12, 14, 10)


def _row(values: tuple, widths: tuple = _WIDTHS) -> str:
    return "".join(str(v).ljust(w) for v, w in zip(values, widths, strict=True))


def format_table(report: RunReport) -> str:
    """A human-readable ranked table for the terminal. Rank is the variant's
    position in report.variant_scores -- the list is already sorted best-first."""
    lines = [f"Task: {report.task_name}", "", _row(_COLUMNS), "-" * sum(_WIDTHS)]

    for rank, vs in enumerate(report.variant_scores, start=1):
        cost = f"{vs.avg_cost_usd:.5f}" if vs.avg_cost_usd is not None else "unknown"
        marker = "  <- winner" if vs.variant_id == report.recommended_variant_id else ""
        row = _row((rank, vs.variant_id, f"{vs.avg_quality:.2f}", cost, f"{vs.avg_latency_ms:.0f}", f"{vs.weighted_score:.3f}"))
        lines.append(row + marker)

    lines += ["", f"Recommended: {report.recommended_variant_id}", report.rationale]
    return "\n".join(lines)


def format_markdown(report: RunReport) -> str:
    lines = [f"# {report.task_name}", "", "| Rank | Variant | Quality | Cost ($) | Latency (ms) | Weighted |",
              "|---|---|---|---|---|---|"]
    for rank, vs in enumerate(report.variant_scores, start=1):
        cost = f"{vs.avg_cost_usd:.5f}" if vs.avg_cost_usd is not None else "unknown"
        winner = " **(winner)**" if vs.variant_id == report.recommended_variant_id else ""
        lines.append(
            f"| {rank} | {vs.variant_id}{winner} | {vs.avg_quality:.2f} | {cost} "
            f"| {vs.avg_latency_ms:.0f} | {vs.weighted_score:.3f} |"
        )
    lines += ["", f"**Recommended:** {report.recommended_variant_id}", "", report.rationale]
    return "\n".join(lines)


def write_json_report(report: RunReport, output_dir: str | Path = ".") -> Path:
    """Writes {task_name}_{timestamp}.json, per Plan.md's persistence format."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_task_name = report.task_name.replace(" ", "_")
    path = Path(output_dir) / f"{safe_task_name}_{timestamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_markdown_report(report: RunReport, output_dir: str | Path = ".") -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_task_name = report.task_name.replace(" ", "_")
    path = Path(output_dir) / f"{safe_task_name}_{timestamp}.md"
    path.write_text(format_markdown(report), encoding="utf-8")
    return path
