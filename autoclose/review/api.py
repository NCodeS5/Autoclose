from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .service import ReviewService

router = APIRouter(prefix="/review", tags=["review"])
_svc: Optional[ReviewService] = None

def svc() -> ReviewService:
    global _svc
    if _svc is None: _svc = ReviewService()
    return _svc

class RunReq(BaseModel):
    case_id: Optional[str] = None
    invoice: Optional[dict] = None

class DecisionReq(BaseModel):
    resolution: str            # benign | issue
    note: str = ""

@router.get("/cases")
def cases(): return svc().list_cases()

@router.post("/run")
def run(req: RunReq, wait: bool = True):
    try: return svc().run(req.case_id, req.invoice) if wait else svc().submit_run(req.case_id, req.invoice)
    except KeyError: raise HTTPException(404, "unknown case_id")
    except (TypeError, ValueError) as e: raise HTTPException(422, str(e))

@router.post("/run-all")
def run_all(wait: bool = True):
    f = svc().run if wait else svc().submit_run
    return [f(c["id"])["case_id"] for c in svc().list_cases()]

@router.get("/queue")
def queue(): return [r for r in list(svc().pending.values()) if r.get("status") == "needs_review"]

@router.post("/decision/{case_id}")
def decision(case_id: str, d: DecisionReq, wait: bool = True):
    if d.resolution not in ("benign", "issue"): raise HTTPException(422, "resolution must be benign or issue")
    try: return svc().decide(case_id, d.resolution, d.note) if wait else svc().submit_decide(case_id, d.resolution, d.note)
    except KeyError: raise HTTPException(404, "no pending case with that id")

@router.get("/history")
def history(): return svc().history

@router.get("/audit/{case_id}")
def audit(case_id: str): return svc().audit.events(case_id)

@router.get("/metrics")
def metrics(): return svc().metrics()

@router.post("/salesforce/seed-vendors")
def seed_vendors(): return svc().seed_sf_vendors()

@router.post("/reset")
def reset(): svc().reset(); return {"ok": True}
