"""`prompteval web`: a local FastAPI server + browser UI. Reuses the exact same
renderer/executor/judge/generator/scoring/critique modules as the CLI -- this
is just a different front door, with JSON requests instead of terminal input().
Deliberately local-only (binds 127.0.0.1, never 0.0.0.0) -- nothing here is
meant to be reachable from another machine.
"""

import asyncio
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from promteval.config import has_any_api_key
from promteval.critique import generate_feedback
from promteval.executor import DEFAULT_CONCURRENCY, execute_matrix
from promteval.generator import GenerationError, generate_random_task, generate_test_cases
from promteval.judge import judge_call_result
from promteval.renderer import TemplateRenderError, render_matrix
from promteval.schemas import EvalRun, PromptVariant, TestCase
from promteval.scoring import build_run_report
from promteval.wizard import DEFAULT_MODEL

QUICKSTART_VARIABLE = "input"
STATIC_DIR = Path(__file__).parent / "web_static"

app = FastAPI(title="PromtEval")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8420", "http://localhost:8420"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_api_key() -> None:
    if not has_any_api_key():
        raise HTTPException(
            status_code=400,
            detail="No API key found. Copy .env.example to .env and add a real key, then restart the server.",
        )


class ImproveRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    judge_model: str | None = None


class ImproveResponse(BaseModel):
    scenarios: list[str]
    score: int
    reasoning: str
    improved_prompt: str


@app.post("/api/improve", response_model=ImproveResponse)
async def api_improve(req: ImproveRequest) -> ImproveResponse:
    _require_api_key()
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt can't be empty.")

    try:
        scenario_values = await generate_test_cases(req.prompt, req.model, n=3)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    run = EvalRun(
        task_name=f"Prompt feedback: {req.prompt[:60]}",
        models=[req.model],
        prompt_variants=[PromptVariant(id="v1", name="Your prompt", template=req.prompt)],
        test_cases=[
            TestCase(id=f"tc{i}", variables={QUICKSTART_VARIABLE: v})
            for i, v in enumerate(scenario_values, start=1)
        ],
        judge_criteria="n/a",  # unused -- improve mode scores/rewrites instead of ranking
    )

    try:
        rendered = render_matrix(run)
    except TemplateRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_results = await execute_matrix(rendered, run.models)

    judge_model = req.judge_model or req.model
    try:
        feedback = await generate_feedback(req.prompt, raw_results, judge_model)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ImproveResponse(
        scenarios=scenario_values,
        score=feedback.score,
        reasoning=feedback.reasoning,
        improved_prompt=feedback.improved_prompt,
    )


class QuickstartRequest(BaseModel):
    task: str = ""  # blank -> AI picks a random one
    prompts: list[str]
    model: str = DEFAULT_MODEL
    judge_model: str | None = None


class VariantResult(BaseModel):
    variant_id: str
    label: str
    quality: float
    latency_ms: float
    weighted_score: float
    is_winner: bool


class QuickstartResponse(BaseModel):
    task_name: str
    scenarios: list[str]
    variants: list[VariantResult]
    recommended: str
    rationale: str


@app.post("/api/quickstart", response_model=QuickstartResponse)
async def api_quickstart(req: QuickstartRequest) -> QuickstartResponse:
    _require_api_key()
    prompts = [p for p in req.prompts if p.strip()]
    if len(prompts) < 2:
        raise HTTPException(status_code=400, detail="Enter at least 2 prompts to compare.")

    try:
        task_name = req.task.strip() or await generate_random_task(req.model)
        scenario_values = await generate_test_cases(task_name, req.model, n=5)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    variants = [PromptVariant(id=f"v{i}", name=f"Prompt {i}", template=p) for i, p in enumerate(prompts, start=1)]
    run = EvalRun(
        task_name=task_name,
        models=[req.model],
        prompt_variants=variants,
        test_cases=[
            TestCase(id=f"tc{i}", variables={QUICKSTART_VARIABLE: v})
            for i, v in enumerate(scenario_values, start=1)
        ],
        judge_criteria="Score 1-5 on accuracy, clarity, and completeness.",
    )

    try:
        rendered = render_matrix(run)
    except TemplateRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    raw_results = await execute_matrix(rendered, run.models, semaphore=semaphore)
    judge_model = req.judge_model or req.model
    judge_results = await asyncio.gather(
        *[judge_call_result(r, run.judge_criteria, judge_model, semaphore) for r in raw_results]
    )
    report = build_run_report(run, raw_results, list(judge_results))

    variant_names = {v.id: v.name for v in variants}
    return QuickstartResponse(
        task_name=report.task_name,
        scenarios=scenario_values,
        variants=[
            VariantResult(
                variant_id=vs.variant_id,
                label=variant_names[vs.variant_id],
                quality=vs.avg_quality,
                latency_ms=vs.avg_latency_ms,
                weighted_score=vs.weighted_score,
                is_winner=vs.variant_id == report.recommended_variant_id,
            )
            for vs in report.variant_scores
        ],
        recommended=report.recommended_variant_id,
        rationale=report.rationale,
    )


@app.get("/api/status")
async def api_status() -> dict:
    return {"has_api_key": has_any_api_key()}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def run_server(host: str = "127.0.0.1", port: int = 8420, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
