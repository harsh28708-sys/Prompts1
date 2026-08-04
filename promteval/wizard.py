"""Interactive `prompteval init` flow: asks for the task, prompt variants, test
cases, and rubric in the terminal, then builds a real EvalRun from the answers.
"""

import re

from promteval.schemas import EvalRun, PromptVariant, TestCase

DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_JUDGE_CRITERIA = "Score 1-5 on accuracy, clarity, and completeness."

_VARIABLE_RE = re.compile(r"\{(\w+)\}")


def _prompt(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{question}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("This can't be empty -- try again.")


def _prompt_yes_no(question: str, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    answer = input(f"{question}{suffix}: ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def _collect_variants() -> list[PromptVariant]:
    print("\nNow add the prompt variants you want to compare -- different styles for")
    print("the same task. Use {variable_name} for the parts that change per test case.\n")

    variants: list[PromptVariant] = []
    i = 1
    while True:
        name = _prompt(f"Variant {i} name (e.g. 'Direct')")
        template = _prompt(f"Variant {i} template")
        variants.append(PromptVariant(id=f"v{i}", name=name, template=template))
        i += 1
        if not _prompt_yes_no(f"Add another variant? (you have {len(variants)} so far)", default=len(variants) < 3):
            return variants


def _collect_test_cases(variable_names: list[str]) -> list[TestCase]:
    print(f"\nNow add test cases. Each one provides a value for: {', '.join(variable_names) or '(none found)'}\n")

    test_cases: list[TestCase] = []
    i = 1
    while True:
        variables = {name: _prompt(f"  {name}") for name in variable_names}
        test_cases.append(TestCase(id=f"tc{i}", variables=variables))
        i += 1
        if not _prompt_yes_no(f"Add another test case? (you have {len(test_cases)} so far)", default=len(test_cases) < 3):
            return test_cases


def run_init_wizard() -> EvalRun:
    """Interactively builds an EvalRun by asking questions in the terminal."""
    print("Let's set up a prompt evaluation.\n")
    task_name = _prompt("Task name (e.g. 'Summarize support tickets')")

    variants = _collect_variants()

    variable_names = sorted({m for v in variants for m in _VARIABLE_RE.findall(v.template)})
    if not variable_names:
        print("\nNote: none of your templates use a {variable} placeholder, so every")
        print("test case will send the exact same prompt.")

    test_cases = _collect_test_cases(variable_names)

    judge_criteria = _prompt(
        "\nJudge rubric (what makes a 5/5 answer?)", default=DEFAULT_JUDGE_CRITERIA
    )
    model = _prompt("Model to run prompts against (provider/model-name)", default=DEFAULT_MODEL)

    return EvalRun(
        task_name=task_name,
        models=[model],
        prompt_variants=variants,
        test_cases=test_cases,
        judge_criteria=judge_criteria,
    )
