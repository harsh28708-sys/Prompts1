"""T2.2: fills a PromptVariant's template with a TestCase's variables."""

from typing import NamedTuple

from promteval.schemas import EvalRun, PromptVariant, TestCase


class TemplateRenderError(Exception):
    """Raised when a template references a variable a test case doesn't provide."""


class RenderedPrompt(NamedTuple):
    variant_id: str
    test_case_id: str
    prompt: str


def render_prompt(variant: PromptVariant, test_case: TestCase) -> str:
    """Fill variant.template with test_case.variables. Raises TemplateRenderError
    (not a raw KeyError) if the template needs a variable the test case lacks."""
    try:
        return variant.template.format(**test_case.variables)
    except KeyError as exc:
        missing = exc.args[0]
        raise TemplateRenderError(
            f"Prompt variant '{variant.id}' requires variable '{{{missing}}}' "
            f"but test case '{test_case.id}' does not provide it."
        ) from exc


def render_matrix(run: EvalRun) -> list[RenderedPrompt]:
    """Render every (variant, test_case) pair in the run. All-or-nothing:
    the first missing variable raises immediately rather than returning partial results."""
    return [
        RenderedPrompt(variant.id, test_case.id, render_prompt(variant, test_case))
        for variant in run.prompt_variants
        for test_case in run.test_cases
    ]
