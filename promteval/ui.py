"""Shared terminal presentation (Rich): colors, panels, tables, spinners.
Keeps cli.py/wizard.py focused on orchestration -- this module owns how
things actually look. Rich auto-detects non-terminal output (e.g. piped
into a test's capsys) and falls back to plain text with no color codes,
so the underlying wording still matches what tests check for.
"""

from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promteval.schemas import PromptFeedback, RunReport

console = Console()
err_console = Console(stderr=True)


def print_banner() -> None:
    console.print(
        Panel(
            "[bold cyan]PromtEval[/bold cyan]  [dim]test a prompt against a real AI[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()


def print_error(message: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")


@contextmanager
def thinking(message: str):
    """A spinner for the duration of a real (often slow) AI call. Rich's Status
    runs its own refresh loop on a background thread, so it updates correctly
    while the foreground `await` inside the `with` block is in progress."""
    with console.status(f"[cyan]{message}...[/cyan]", spinner="dots"):
        yield


def _score_style(score: int) -> str:
    if score >= 4:
        return "bold green"
    if score == 3:
        return "bold yellow"
    return "bold red"


def print_feedback(feedback: PromptFeedback) -> None:
    style = _score_style(feedback.score)
    border = style.split()[-1]  # "green"/"yellow"/"red" without the "bold"
    console.print()
    console.print(f"Score: [{style}]{feedback.score}/5[/{style}]")
    console.print(feedback.reasoning)
    console.print()
    console.print(Panel(feedback.improved_prompt, title="Improved Prompt", border_style=border, expand=False))


def print_scenarios(values: list[str]) -> None:
    console.print("[dim]Generated scenarios:[/dim]")
    for i, value in enumerate(values, start=1):
        console.print(f"  [cyan]{i}.[/cyan] {value}")


def print_report_table(report: RunReport) -> None:
    table = Table(title=report.task_name, header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Variant")
    table.add_column("Quality", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Weighted", justify="right")

    for rank, vs in enumerate(report.variant_scores, start=1):
        is_winner = vs.variant_id == report.recommended_variant_id
        label = f"{vs.variant_id}  <- winner" if is_winner else vs.variant_id
        table.add_row(
            str(rank),
            label,
            f"{vs.avg_quality:.2f}",
            f"{vs.avg_latency_ms:.0f}",
            f"{vs.weighted_score:.3f}",
            style="bold green" if is_winner else None,
        )

    console.print()
    console.print(table)
    console.print()
    console.print(f"[bold]Recommended:[/bold] {report.recommended_variant_id}")
    console.print(report.rationale)
