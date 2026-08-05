# PromtEval

A CLI tool that runs prompt variants against real LLMs, scores each output with a judge
LLM against your rubric, and ranks the variants by quality, cost, and latency to
recommend a winner.

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

## Set up API keys

Copy the template and fill in real keys for whichever providers you want to use:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
```

You only need a key for the provider(s) named in your input file's `models` list —
you don't need all three. Get keys at:

- Groq: [console.groq.com/keys](https://console.groq.com/keys)
- OpenRouter: [openrouter.ai/keys](https://openrouter.ai/keys)
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

`.env` is git-ignored — your keys never get committed.

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

You don't have to hand-write this file — see the two commands below that build it
(or skip it entirely) for you.

## Don't want to write a JSON file by hand?

**`prompteval init`** — asks you everything in the terminal (task name, prompt
variants, test cases, rubric) and saves it as a real input file you can run or edit later:

```bash
prompteval init my_task.json
prompteval run my_task.json --judge-model groq/llama-3.3-70b-versatile
```

**`prompteval quickstart`** (also the default — see below) — the fastest way to try
the tool. AI generates a task (or you can type your own) plus 5 realistic test cases,
you just type your 3 prompts, and it evaluates immediately — no file, no second command:

```bash
prompteval quickstart --judge-model groq/llama-3.3-70b-versatile

# or, since quickstart is the default when no subcommand is given:
prompteval --judge-model groq/llama-3.3-70b-versatile
```

Your 3 prompts should use `{input}` as the placeholder — that's the fixed variable
name the AI-generated test cases fill in. It still saves a copy of the generated
input file afterward, so you can re-run the exact same test later with `prompteval run`.

## Just have ONE prompt you want feedback on?

**`prompteval improve`** — different from the other commands: instead of comparing
multiple prompts, it takes the ONE prompt you're working on, tests it for real
against a few AI-generated scenarios, and gives you plain-English feedback on how
to improve it (not a score):

```bash
prompteval improve --judge-model groq/llama-3.3-70b-versatile
```

It asks for the context (what you're trying to do) and your prompt (again using
`{input}` as the placeholder), then saves the full feedback to a markdown file.

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
