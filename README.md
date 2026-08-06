# PromtEval

A CLI tool for testing prompts against real LLMs. Give it one prompt and it scores it
and hands back a better version; give it several and it ranks them by quality, cost,
and latency to recommend a winner. Colored output, a live "thinking..." spinner while
the AI works, and a real bordered table for ranked results (via [Rich](https://github.com/Textualize/rich)).

> Not sure where to start? Once installed, run `prompteval /help` (or just `-h`) for
> a friendly walkthrough right in your terminal — no need to read this whole file.

## Install

```bash
git clone <this repo>
cd PromtEval
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
```

This installs the `prompteval` command into your virtual environment, along with
`pytest` and `ruff` for development.

**Don't want to activate the venv every time?** Two options:
- Double-click **`Run PromtEval.cmd`** (Windows) — a literal run button. Opens a
  window with the interactive menu (below), and stays open afterward so you can
  read the results.
- Or run `.\prompteval.cmd <command>` from a terminal in this folder (e.g.
  `.\prompteval.cmd run examples/sample_run.json`) — same idea, but lets you pass
  any command/flags, for when you're already in a terminal.

Both just call the venv's `prompteval` directly, so no PATH changes or activation
needed. (You can also add `.venv\Scripts` to your PATH permanently instead, if
you'd rather use the plain `prompteval` command everywhere.)

## Set up API keys

Copy the template and fill in real keys for whichever providers you want to use:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
ANTHROPIC_API_KEY=
```

You only need a key for the provider(s) named in your input file's `models` list —
you don't need all four. Get keys at:

- Groq: [console.groq.com/keys](https://console.groq.com/keys)
- OpenRouter: [openrouter.ai/keys](https://openrouter.ai/keys)
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Claude: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) — **no free
  tier**, needs a paid credit balance before any call works (model strings look like
  `anthropic/claude-sonnet-5`)

`.env` is git-ignored — your keys never get committed.

## The interactive menu

Just run `prompteval` (or double-click `Run PromtEval.cmd`) with no arguments and
you'll see an arrow-key menu instead of needing to know a command name:

```
? What do you want to do?
❯ Improve one prompt (get a score + a better version)
  Compare 3 prompts (AI makes up the test data)
  Open the browser UI instead (Improve + Compare, visually)
  Save a prompt comparison to a file for later (doesn't run yet)
  Run a prompt comparison file you already have
  Help -- what do these options actually mean?
  Exit
```

Use ↑/↓ and Enter to pick. Any flags you already typed still apply to whatever you
pick, e.g. `prompteval --judge-model groq/llama-3.3-70b-versatile` shows the menu
and applies that flag to whichever option you choose. Picking **Help** explains
each option in plain terms (not the full install guide — that's `prompteval /help`).

> Some terminals (Git Bash/mintty on Windows in particular) can't render the
> arrow-key menu at all — it automatically falls back to a plain numbered list
> in that case, so it still works, just without arrow keys.

Every command below still works directly too (`prompteval improve`, `prompteval
run <file>`, etc.) if you'd rather skip the menu and type the command by name.

## Input file format

An input file is a JSON object with five fields:

```json
{
  "task_name": "Summarize support tickets",
  "models": ["groq/llama-3.3-70b-versatile"],
  "prompt_variants": [
    { "id": "v1", "name": "Direct", "template": "Summarize in one sentence: {customer_message}" }
  ],
  "test_cases": [
    { "id": "tc1", "variables": { "customer_message": "My order arrived broken and I want a refund." } }
  ],
  "judge_criteria": "Score 1-5. A 5 means accurate, concise, and captures the core issue."
}
```

- **`models`** — one or more model strings in `provider/model-name` form (LiteLLM's
  format). The tool runs every variant against every test case against every model.
- **`prompt_variants`** — each has a `template` with `{variable}` placeholders that
  must match keys in `test_cases[].variables`.
- **`test_cases`** — the real inputs to plug into each template.
- **`judge_criteria`** — freeform rubric text given to the judge model.

See [`examples/sample_run.json`](examples/sample_run.json) for a full working example.

You don't have to hand-write this file — see the commands below that build it
(or skip it entirely) for you.

## Have ONE prompt you want feedback on? (the default)

**`prompteval improve`** — takes the ONE prompt you're working on, tests it for
real against a few AI-generated scenarios, and gives you a **score (1-5)** plus a
**rewritten, improved version** of your prompt:

```bash
prompteval improve --judge-model groq/llama-3.3-70b-versatile

# or, since improve is the default when no subcommand is given:
prompteval --judge-model groq/llama-3.3-70b-versatile
```

It only asks for your prompt (use `{input}` as the placeholder for the part that
changes) — no separate context question; the prompt's own wording is enough to
generate realistic test scenarios. Saves the score, reasoning, rewritten prompt,
and every test case's real output to a markdown file afterward.

## Want to compare several different prompts instead?

**`prompteval quickstart`** — AI generates a task (or you can type your own) plus
5 realistic test cases, you type 3 different prompts to compare, and it evaluates
immediately — no file, no second command:

```bash
prompteval quickstart --judge-model groq/llama-3.3-70b-versatile
```

Your 3 prompts should use `{input}` as the placeholder — that's the fixed variable
name the AI-generated test cases fill in. It still saves a copy of the generated
input file afterward, so you can re-run the exact same test later with `prompteval run`.

## Prefer a browser over the terminal?

**`prompteval web`** — opens the same Improve and Compare flows as a page in
your own browser instead of the terminal: type your prompt(s) into text boxes,
click a button, and watch the score / ranked results appear on the page.

```bash
prompteval web
```

It's still 100% local — the server only binds to `127.0.0.1` (your own
machine), so nothing outside your computer can reach it, and closing the
browser tab doesn't stop it (Ctrl+C in the terminal does). Options:

- `--port 9000` — use a different local port (default `8420`)
- `--no-browser` — start the server without automatically opening a tab

If no API key is configured yet, the page itself shows a warning banner
instead of the terminal's message — everything else works the same way.

Model and judge model are dropdowns grouped by provider (Groq, Gemini,
OpenRouter, Claude) — no need to remember or type the exact `provider/model-name`
string. Pick **"Custom model string…"** at the bottom of either dropdown if you
want to type one in by hand instead (e.g. a newer model not in the list yet).

**`prompteval init`** — same idea as `quickstart`, but asks you everything yourself
(task name, prompt variants, test cases, rubric) instead of generating any of it,
and saves it as a real input file you can run or edit later:

```bash
prompteval init my_task.json
prompteval run my_task.json --judge-model groq/llama-3.3-70b-versatile
```

## Run it

```bash
prompteval run examples/sample_run.json --judge-model groq/llama-3.3-70b-versatile
```

> The default judge model is `gemini/gemini-2.0-flash`. If your Google Cloud project
> doesn't have Gemini's free tier enabled (a "quota exceeded, limit: 0" error), pass
> `--judge-model` with a model from a provider you do have working, as above.

Optional flags (all override the defaults from `Plan.md`):

| Flag | Default | Meaning |
|---|---|---|
| `--judge-model` | `gemini/gemini-2.0-flash` | Model used to score outputs |
| `--concurrency` | `10` | Max LLM calls in flight at once (shared between execution and judging) |
| `--timeout` | `60` | Per-call timeout in seconds |
| `--format` | `json` | Report file format: `json` or `markdown` |

Example overriding the judge model:

```bash
prompteval run examples/sample_run.json --judge-model groq/llama-3.3-70b-versatile
```

## Reading the report

Each run prints a ranked table to the terminal, e.g.:

```
Rank Variant   Quality  Latency (ms)  Weighted
------------------------------------------------
1    v2        5.00     296           3.750       <- winner
2    v3        4.20     100           3.316
3    v1        4.20     275           3.168

Recommended: v2
'Structured' (v2) is the recommended variant. 'Concise-with-example' (v3) lost: lower quality (4.20 vs 5.00). ...
```

It also writes the full results to `{task_name}_{timestamp}.json` (or `.md` with
`--format markdown`) in the current directory, including every raw LLM output, every
judge score and reasoning, and the final ranking — not just the summary table.

A failed call (timeout, rate limit, bad model name, etc.) never crashes the run — it's
recorded with an `error` field and scored 0, and the rest of the batch still completes.

## Running the tests

```bash
pytest
ruff check .
```
