# Tasks — Prompt Evaluator

> **Note:** PBI numbers skip 5 and 7 in sequence because those are stretch goals listed at the bottom. Must-have PBIs run PBI-0 → 1 → 2 → 3 → 4 → 6 → 8.

---

## PBI-0 — Spec-Driven Development Artifacts [Must] — Mon 3 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T0.1 | Write `Spec.md`: user story, core goals, non-goals, acceptance criteria | 1.0h | Done |
| T0.2 | Write `Plan.md`: architecture, data contracts, judge prompt, all open-question decisions | 1.0h | Done |
| T0.3 | Write this `Tasks.md` | 0.5h | Done |
| T0.4 | Stakeholder review of spec/plan/tasks; refine and get sign-off | 0.5h | In Progress |

**DoD:** `Spec.md`, `Plan.md`, `Tasks.md` exist in the PromptEval folder; `Plan.md` resolves every open question with a decision; `Tasks.md` is granular enough for Claude Code to execute without further clarification; stakeholder has signed off.

---

## PBI-1 — Project Setup & Environment [Must] — Mon 3 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T1.1 | Initialise Python project (`pyproject.toml`), git repo, ruff config, virtualenv | 1.0h | Not Started |
| T1.2 | Configure env vars (`.env` + `python-dotenv`); document keys: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY` | 0.5h | Not Started |
| T1.3 | Define core Pydantic schemas in `schemas.py` per Plan.md; validate each against a hand-written sample | 1.5h | Not Started |

**DoD:** `pip install -e .` succeeds; lint passes; all schemas parse a sample JSON fixture.

---

## PBI-2 — Prompt Variant Input & Template Rendering [Must] — Tue 4 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T2.1 | Finalise `EvalRun` input JSON format; add `examples/customer_summarization.json` matching Plan.md | 1.0h | Not Started |
| T2.2 | Build template renderer (`str.format`); produce `rendered_prompt` for each variant × test case | 1.5h | Not Started |
| T2.3 | Unit tests: full matrix render + missing-variable raises `TemplateRenderError` with clear message | 1.0h | Not Started |

**DoD:** Example input renders correctly for every variant/test-case pair; missing variables produce a readable error.

---

## PBI-3 — Multi-LLM Execution Engine [Must] — Tue–Wed 4–5 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T3.1 | Integrate LiteLLM for OpenRouter, Gemini, Groq using model strings from Plan.md | 2.0h | Not Started |
| T3.2 | Build async runner (`asyncio.gather` + `Semaphore`); fire variant × test-case × model concurrently | 2.0h | Not Started |
| T3.3 | Capture `input_tokens`, `output_tokens`, `latency_ms` per call into `LLMCallResult` | 1.0h | Not Started |
| T3.4 | Add retry/backoff (tenacity); on final failure set `LLMCallResult.error` instead of raising | 1.0h | Not Started |

**DoD:** All N×M×K calls execute concurrently; every result is a valid `LLMCallResult`; one forced failure doesn't abort the batch.

---

## PBI-4 — LLM Judge Evaluation Engine [Must] — Wed–Thu 5–6 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T4.1 | Implement judge prompt template from Plan.md | 1.0h | Not Started |
| T4.2 | Send judge prompt to judge model (`temperature=0`); validate response against `JudgeResult` | 1.5h | Not Started |
| T4.3 | Handle parse failures: retry once, then fallback `score=0`, `reasoning="judge parse failure"` | 1.0h | Not Started |
| T4.4 | Unit tests with mocked responses: valid JSON, malformed JSON, markdown-wrapped JSON | 1.0h | Not Started |

**DoD:** Every `LLMCallResult` gets a corresponding `JudgeResult`; malformed judge responses never crash the run.

---

## PBI-6 — Scoring, Ranking & Recommendation [Must] — Thu 6 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T6.1 | Aggregate per-variant `VariantScore` using weighted formula from Plan.md | 1.5h | Not Started |
| T6.2 | Recommendation generator: pick winner, write rationale naming why each other variant lost | 1.5h | Not Started |
| T6.3 | Report formatter: CLI table + JSON output file + optional Markdown | 1.0h | Not Started |

**DoD:** One end-to-end run produces a `RunReport` with ranked table, `recommended_variant_id`, and concrete per-variant rationale.

---

## PBI-8 — End-to-End Testing & Docs [Must] — Fri 7 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T8.1 | Full E2E run: 3 variants × 5 test cases × 1 model; confirm Spec.md acceptance criteria | 1.5h | Not Started |
| T8.2 | Write `README.md`: install, env vars, input file format, run command, reading the report | 1.0h | Not Started |
| T8.3 | Final bug fixes and polish from E2E run | 1.5h | Not Started |

**DoD:** Fresh clone + README-only setup produces a working run; no unhandled exceptions.

---

## PBI-9 — CLI Orchestrator [Must] — Thu 6 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T9.1 | Build `cli.py` entry point: `prompteval run <input.json> [--judge-model] [--concurrency] [--timeout] [--format]` | 1.5h | Not Started |
| T9.2 | Wire pipeline: load input → render → execute → judge → score → report | 1.0h | Not Started |

**DoD:** Single CLI command runs the full pipeline and writes output; flags override Plan.md defaults.

---

## PBI-5 — Cost & Latency Calculation [Stretch] — Fri 7 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T5.1 | Build pricing table ($/input token, $/output token) for in-scope models | 1.0h | Not Started |
| T5.2 | Compute `cost_usd` per call; spot-check against manual calculation | 1.0h | Not Started |

**DoD:** Known token counts match hand calculation; unknown models → `cost_usd = None`.

---

## PBI-7 — Error Handling & Reliability [Stretch] — Fri 7 Aug

| ID | Task | Est | Status |
|---|---|---|---|
| T7.1 | Confirm partial failures never drop completed results | 1.0h | Not Started |
| T7.2 | Explicit per-call timeout handling (distinct from retry/backoff) | 1.0h | Not Started |

**DoD:** Simulated timeout or mid-run failure still yields a complete `RunReport` for finished calls.

---

## Summary

| PBI | Priority | Est Total | Day |
|---|---|---|---|
| PBI-0 Spec artifacts | Must | 3.0h | Mon 3 Aug |
| PBI-1 Setup | Must | 3.0h | Mon 3 Aug |
| PBI-2 Rendering | Must | 3.5h | Tue 4 Aug |
| PBI-3 Execution | Must | 6.0h | Tue–Wed |
| PBI-4 Judge | Must | 4.5h | Wed–Thu |
| PBI-6 Scoring | Must | 4.0h | Thu 6 Aug |
| PBI-9 CLI | Must | 2.5h | Thu 6 Aug |
| PBI-8 E2E & Docs | Must | 4.0h | Fri 7 Aug |
| PBI-5 Cost | Stretch | 2.0h | Fri 7 Aug |
| PBI-7 Reliability | Stretch | 2.0h | Fri 7 Aug |
| **Total Must** | | **30.5h** | |
| **Total Stretch** | | **4.0h** | |
