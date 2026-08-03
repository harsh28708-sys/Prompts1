# Prompt Evaluator

## Summary

A CLI tool that accepts prompt variants and test cases, runs them against one or more LLMs, and evaluates the outputs using a structured judge prompt. It ranks variants by quality, cost, and latency and recommends a winner.

## User Story

As a user, I want to test different prompts against the same task so I can deliver work as efficiently as possible (cost, latency, and quality), without manual trial-and-error.

## Core Goals

1. Accept a task definition: test cases (inputs) and judge criteria (rubric).
2. Accept multiple prompt variants (templates with placeholders).
3. Execute prompts across configured LLMs in parallel where possible.
4. Use a judge LLM to score each output on a strict 1–5 rubric.
5. Report scores aggregated by latency, cost, and effectiveness.
6. Recommend the best prompt variant and explain why others ranked lower.

## Non-Goals

- **No automatic prompt generation** — only user-submitted variants.
- **No real-time UI** — CLI only for v1 (UI is a future extension).
- **No run history / persistence** — each run is self-contained; no database or past-run tracking.
- **No model training or fine-tuning** — this evaluates prompts against fixed models only.

## Acceptance Criteria

1. Given **3 prompt variants**, **5 test cases**, and **1 model**, the system executes **15 LLM calls** (variant × test case), plus **15 judge calls** (one per successful output).
2. The judge returns a strictly formatted JSON object with a score between **1 and 5** and reasoning.
3. A report is returned highlighting the overall winner for the given task, with per-variant breakdowns.
4. Failures and timeouts during execution or evaluation are handled gracefully — partial results are preserved and failures are recorded, not fatal.

## Example Scenario

**Task:** Summarize customer support messages.

- **Test cases:** 5 sample customer messages.
- **Prompt variants:** 3 different summarization templates.
- **Models:** e.g. `gemini/gemini-2.0-flash` (single model for acceptance test).
- **Expected:** 15 execution results, 15 judge scores, one ranked recommendation.
