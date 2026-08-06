"""Interactive `prompteval init`/`quickstart` flows: ask for the task, prompt
variants, test cases, and rubric in the terminal, then build a real EvalRun.
"""

import re

from promteval.generator import generate_random_task, generate_test_cases
from promteval.schemas import EvalRun, PromptVariant, TestCase
from promteval.ui import console, print_scenarios, thinking

DEFAULT_MODEL = "groq/llama-3.1-8b-instant"  # Groq's free tier allows 14,400 req/day
# on this model vs 1,000/day on the 70B version -- far less likely to hit a rate
# limit while testing/comparing prompts, which is what this tool spends calls on.
DEFAULT_JUDGE_CRITERIA = "Score 1-5 on accuracy, clarity, and completeness."

# `quickstart` fixes the placeholder name so AI-generated test cases can be slotted
# into whatever 3 prompts the user writes, without needing to infer it after the fact.
QUICKSTART_VARIABLE = "input"

# `improve` reuses the same placeholder convention for the one prompt being critiqued.
IMPROVE_SCENARIO_COUNT = 3

_VARIABLE_RE = re.compile(r"\{(\w+)\}")


_BOM = chr(0xFEFF)


def _read_input(prompt_text: str) -> str:
    """input(), cleaned. Some terminals/pipes (observed via PowerShell piping
    into a .cmd-wrapped run) inject a leading BOM character as the very first
    byte of stdin -- str.strip() alone does NOT remove it (it's not whitespace
    to Python), so a "blank" answer meant to trigger a default would otherwise
    come through as one invisible non-empty character instead of ""."""
    return input(prompt_text).replace(_BOM, "").strip()


def _prompt(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = _read_input(f"{question}{suffix}: ")
        if answer:
            return answer
        if default is not None:
            return default
        print("This can't be empty -- try again.")


def _prompt_yes_no(question: str, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    answer = _read_input(f"{question}{suffix}: ").lower()
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


async def run_improve_wizard(model: str) -> tuple[str, list[str]]:
    """Asks for ONE prompt (not multiple variants, no context question -- the
    prompt's own wording is used to infer realistic scenarios), then AI-generates
    a few scenarios to actually test it against. Returns (prompt_template,
    scenario_values) -- the caller runs the prompt for real and gets back a
    score + a rewritten, improved version, not a ranking against other variants."""
    print("Let's get feedback on your prompt.\n")

    placeholder = "{" + QUICKSTART_VARIABLE + "}"
    prompt_template = _prompt(f"Your prompt (use {placeholder} where the real input goes)")
    if placeholder not in prompt_template:
        print(f"  Warning: this doesn't include {placeholder} -- it'll run the exact same way every time.")

    console.print()
    with thinking(f"Generating {IMPROVE_SCENARIO_COUNT} sample scenarios"):
        scenario_values = await generate_test_cases(prompt_template, model, n=IMPROVE_SCENARIO_COUNT)
    print_scenarios(scenario_values)

    return prompt_template, scenario_values


async def run_quickstart_wizard(model: str) -> EvalRun:
    """AI generates the task (or uses one you type) and 5 test cases; you write 3
    prompts against them; the caller runs the evaluation immediately afterward."""
    print("Let's set up a quick evaluation with AI-generated test data.\n")

    task_input = _read_input("What task do you want to test? (press Enter for a random AI-generated task): ")
    if task_input:
        task_name = task_input
    else:
        with thinking("Thinking of a random task"):
            task_name = await generate_random_task(model)
        console.print(f"Task: [bold]{task_name}[/bold]")

    console.print()
    with thinking("Generating 5 test cases"):
        test_case_values = await generate_test_cases(task_name, model, n=5)
    print_scenarios(test_case_values)

    placeholder = "{" + QUICKSTART_VARIABLE + "}"
    print(f"\nNow enter your 3 prompts to test. Use {placeholder} where the test data goes.\n")

    variants: list[PromptVariant] = []
    for i in range(1, 4):
        template = _prompt(f"Prompt {i} template")
        if placeholder not in template:
            print(f"  Warning: this template doesn't include {placeholder} -- it won't see the test data.")
        variants.append(PromptVariant(id=f"v{i}", name=f"Prompt {i}", template=template))

    test_cases = [
        TestCase(id=f"tc{i}", variables={QUICKSTART_VARIABLE: value})
        for i, value in enumerate(test_case_values, start=1)
    ]

    judge_criteria = _prompt("\nJudge rubric (what makes a 5/5 answer?)", default=DEFAULT_JUDGE_CRITERIA)

    return EvalRun(
        task_name=task_name,
        models=[model],
        prompt_variants=variants,
        test_cases=test_cases,
        judge_criteria=judge_criteria,
    )
