"""FastAPI service around the AutoClose graph.

POST /closes                     run a close (send a run, or {"scenario": "writeoff"} for a demo run)
GET  /queue                      closes paused for a human (E1/E2/E3 or audit sample)
POST /cases/{thread_id}/decision resume a paused close: {"resolution": "benign"|"issue", "note": "..."}
GET  /health, GET /metrics       liveness and simple usage/latency counters

NOTE: paused cases and the case memory live in process memory. Run ONE replica until the
checkpointer is moved to PostgreSQL (langgraph-checkpoint-postgres)."""
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
import random, time, itertools
from dataclasses import asdict
from typing import Optional
import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.types import Command
from .generate_data import generate, _base_run, _inject
from .tools import Baseline
from .case_store import CaseStore
from .graph import build_graph
from .review.api import router as review_router

SCENARIOS = {"clean", "duplicate", "misclassified", "writeoff", "missing_bank", "novel_vendor"}

app = FastAPI(title="AutoClose", version="0.1.0")

import os, base64, secrets
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class DemoPassword(BaseHTTPMiddleware):
    """When DEMO_PASSWORD is set, every page and API call except /health needs it (browser login box).
    Stops strangers who find the URL from spending your model tokens."""
    async def dispatch(self, request, call_next):
        pw = os.getenv("DEMO_PASSWORD")
        if not pw or request.url.path == "/health":
            return await call_next(request)
        h = request.headers.get("authorization", "")
        if h.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(h[6:]).decode().partition(":")
                if secrets.compare_digest(given, pw):
                    return await call_next(request)
            except Exception:
                pass
        return Response("Password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="AutoClose demo"'})

app.add_middleware(DemoPassword)
_store = CaseStore()
_graph = build_graph(Baseline(generate(60, anomaly_rate=0.0, seed=99)), _store)
_ids = itertools.count(1000)
_pending: dict = {}
_history: list = []          # newest first; shown on the dashboard
_stats = {"closes": 0, "auto": 0, "audit": 0, "escalated": 0, "decisions": 0, "latency_ms_total": 0.0}

class CloseRequest(BaseModel):
    run: Optional[dict] = None          # {run_id, entity, ledger, bank, history}
    scenario: Optional[str] = None      # demo shortcut

class Decision(BaseModel):
    resolution: str                     # benign | issue
    note: str = ""

def _demo_run(scenario: str, run_id: int) -> dict:
    if scenario not in SCENARIOS:
        raise HTTPException(422, f"scenario must be one of {sorted(SCENARIOS)}")
    rng = random.Random(run_id)
    r = _base_run(rng, run_id)
    if scenario != "clean":
        _inject(rng, r, scenario)
    return asdict(r)

def _summary(thread_id: str, values: dict, paused: bool) -> dict:
    return {"thread_id": thread_id, "status": "needs_review" if paused else "completed",
            "route": values.get("route"), "reasons": values.get("reasons"),
            "actions": values.get("actions"), "report": values.get("report"),
            "final": values.get("final")}

app.include_router(review_router)

@app.get("/month-end", include_in_schema=False)
def month_end_dashboard():
    return FileResponse(pathlib.Path(__file__).parent / "static" / "index.html")

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(pathlib.Path(__file__).parent / "static" / "review.html")

@app.get("/history")
def history():
    return _history[:25]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    n = max(1, _stats["closes"])
    return {**_stats, "avg_latency_ms": round(_stats["latency_ms_total"] / n, 2), "pending": len(_pending)}

@app.post("/closes")
def create_close(req: CloseRequest):
    run_id = next(_ids)
    if req.run is not None:
        missing = {"ledger", "bank", "history"} - set(req.run)
        if missing:
            raise HTTPException(422, f"run is missing: {sorted(missing)}")
        run = {"run_id": run_id, "entity": "API", **req.run}
    elif req.scenario:
        run = _demo_run(req.scenario, run_id)
    else:
        raise HTTPException(422, "send either 'run' or 'scenario'")
    thread_id = f"close-{run_id}"
    label = req.scenario or "custom data"
    cfg = {"configurable": {"thread_id": thread_id}}
    t = time.perf_counter()
    out = _graph.invoke({"run": run}, cfg)
    _stats["latency_ms_total"] += (time.perf_counter() - t) * 1000
    _stats["closes"] += 1
    values, paused = _graph.get_state(cfg).values, "__interrupt__" in out
    if paused:
        _pending[thread_id] = {"thread_id": thread_id, "run_id": run_id, "route": values["route"],
                               "reasons": values["reasons"], "actions": values["actions"], "report": values["report"]}
        _stats["escalated" if values["route"] == "escalate" else "audit"] += 1
    else:
        _stats["auto"] += 1
    summary = _summary(thread_id, values, paused)
    _history.insert(0, {"thread_id": thread_id, "scenario": label, "status": summary["status"],
                        "route": summary["route"], "reason": (summary["reasons"] or [""])[0],
                        "actions": len(summary["actions"] or []), "final": summary["final"],
                        "latency_ms": round((time.perf_counter() - t) * 1000, 1)})
    return summary

@app.get("/queue")
def queue():
    return list(_pending.values())

@app.post("/cases/{thread_id}/decision")
def decide(thread_id: str, d: Decision):
    if thread_id not in _pending:
        raise HTTPException(404, "no pending case with that id")
    if d.resolution not in ("benign", "issue"):
        raise HTTPException(422, "resolution must be 'benign' or 'issue'")
    cfg = {"configurable": {"thread_id": thread_id}}
    _graph.invoke(Command(resume={"resolution": d.resolution, "note": d.note}), cfg)
    del _pending[thread_id]
    _stats["decisions"] += 1
    out = _summary(thread_id, _graph.get_state(cfg).values, False)
    for h in _history:
        if h["thread_id"] == thread_id:
            h.update(status="completed", final=out["final"], decision=d.resolution)
    return out
