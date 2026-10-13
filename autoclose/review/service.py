"""Holds the store, flow, pending cases and history for the API."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from langgraph.types import Command
from ..rag import get_store, ingest, CASES
from .checks import load_vendors, load_prior, DATA
from .audit import AuditLog
from .flow import build_flow
from .reasoner import get_llm, CASE_MATCH_MIN

class ReviewService:
    def __init__(self, store=None, llm="auto"):
        self.store = store or get_store()
        self.llm = get_llm() if llm == "auto" else llm
        self.provider = os.getenv("LLM_PROVIDER", "offline") if self.llm is not None else "offline"
        self.vendors, self.prior = load_vendors(), load_prior()
        self.cases = {c["id"]: c for c in json.loads((DATA / "cases.json").read_text())}
        self.audit = AuditLog()
        self.agent_mode = os.getenv("AGENT_MODE", "workflow").lower() if self.llm is not None else "workflow"
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=int(os.getenv("AUTOCLOSE_WORKERS", "4")))
        self.gen = 0
        self.kb = ingest(self.store)
        self.env = os.getenv("AUTOCLOSE_ENV", "demo").lower()     # demo: clean memory each start | production: memory persists
        if self.env != "production":
            self._forget_learned()
        self._build()
    def _forget_learned(self):
        from ..rag import load_past_cases
        self.store.purge_except(CASES, [c.id for c in load_past_cases()])
    def _build(self):
        self.flow = build_flow(self.store, self.vendors, self.prior, self.audit, self.llm,
                               audit_rate=float(os.getenv("AUDIT_RATE", "0")), agent_mode=self.agent_mode)
        self.pending, self.history, self.runs = {}, [], Counter()
    def reset(self):
        """Forget decisions and learned cases; keep the knowledge base."""
        if self.env != "production":      # in production Reset clears the screen only; learned cases are real data
            self._forget_learned()
        with self.lock:
            self.gen += 1; self.audit.clear(); self._build()
    def list_cases(self):
        return [{k: v for k, v in c.items() if k != "expected"} for c in self.cases.values()]
    def _alloc(self, case_id=None, invoice=None):
        if invoice is None:
            if case_id not in self.cases: raise KeyError(case_id)
            invoice = {k: v for k, v in self.cases[case_id].items() if k != "expected"}
        inv = dict(invoice); base = inv.get("id") or "X"
        with self.lock:
            self.runs[base] += 1
            inv["id"] = base if self.runs[base] == 1 else f"{base}-r{self.runs[base]}"
            row = {"case_id": inv["id"], "vendor": inv["vendor"], "invoice_no": inv["invoice_no"],
                   "amount": inv["amount"], "status": "running", "invoice": inv}
            self.history.insert(0, row)
        return row
    def _execute(self, row, gen):
        inv = row["invoice"]; cfg = {"configurable": {"thread_id": inv["id"]}}
        t = time.time()
        try:
            r = self.flow.invoke({"invoice": inv}, cfg); ms = round((time.time() - t) * 1000)
            paused = "__interrupt__" in r
            view = self._view(inv["id"])
            with self.lock:
                row.update(view); row.update(status="needs_review" if paused else "completed", ms=ms)
                if paused and gen == self.gen: self.pending[inv["id"]] = row
        except Exception as e:
            with self.lock: row.update(status="failed", ms=round((time.time() - t) * 1000), rationale=f"Run failed: {str(e)[:300]}")
        return row
    def run(self, case_id=None, invoice=None):
        """Blocking run (tests, Swagger)."""
        return self._execute(self._alloc(case_id, invoice), self.gen)
    def submit_run(self, case_id=None, invoice=None):
        """Non-blocking: returns the placeholder row now, finishes in the background."""
        row = self._alloc(case_id, invoice); gen = self.gen
        self.pool.submit(self._execute, row, gen)
        return row
    def _view(self, cid):
        v = self.flow.get_state({"configurable": {"thread_id": cid}}).values
        d = v.get("decision", {})
        return {"checks": v.get("checks"), "passages": v.get("passages"), "similar_cases": v.get("cases"),
                "decision": d.get("decision"), "proposed": d.get("proposed"), "edge": d.get("edge"),
                "confidence": d.get("confidence"), "rationale": d.get("rationale"), "citations": d.get("citations"),
                "guard_notes": d.get("guard_notes"), "provider": d.get("provider"),
                "trace": v.get("trace"), "final": v.get("final"), "action": v.get("proposed_action"), "route": v.get("route")}
    def _begin_decide(self, cid):
        with self.lock:
            if cid not in self.pending: raise KeyError(cid)
            row = self.pending.pop(cid); row["status"] = "finishing"
        return row
    def _finish_decide(self, row, resolution, note):
        cid = row["case_id"]; before = self.store.count(CASES)
        try:
            self.flow.invoke(Command(resume={"resolution": resolution, "note": note}), {"configurable": {"thread_id": cid}})
            view = self._view(cid)
            with self.lock:
                row.update(view); row.update(status="completed", reviewer=resolution, note=note,
                                             learned=self.store.count(CASES) > before)
            if row.get("learned"):
                try: self._sweep(cid) if resolution == "benign" else self._sweep_reject(cid)
                except Exception: pass
        except Exception as e:
            with self.lock: row.update(status="failed", rationale=f"Finalizing failed: {str(e)[:300]}")
        return row
    def _sweep(self, learned_from):
        """After a person approves a case that was learned, re-check the other waiting cases against the new memory.
        A waiting case is auto-resolved only if a fresh review approves it AND cites the newly learned case."""
        with self.lock:
            others = [r for c, r in self.pending.items() if c != learned_from]
            for r in others: r["status"] = "rechecking"
        for r in others:
            cid = r["case_id"]; ok = False; why = None
            try:
                with self.lock: self.runs[cid + "~chk"] += 1; tid = f"{cid}~chk{self.runs[cid + '~chk']}"
                res = self.flow.invoke({"invoice": r["invoice"] | {"id": tid}}, {"configurable": {"thread_id": tid}})
                v = self._view(tid)
                ok = ("__interrupt__" not in res and v.get("final") == "approved"
                      and f"CASE-{learned_from}" in (v.get("citations") or []))
                why = {"against": learned_from, "decision": v.get("decision"), "proposed": v.get("proposed"), "confidence": v.get("confidence"),
                       "cites_learned_case": f"CASE-{learned_from}" in (v.get("citations") or []), "guard_notes": v.get("guard_notes") or [],
                       "rationale": (v.get("rationale") or "")[:300]}
                self.audit.log(cid, "linked_recheck", {"triggered_by": learned_from, "auto_resolved": ok,
                                                       "decision": v.get("decision"), "proposed": v.get("proposed"),
                                                       "confidence": v.get("confidence"), "citations": v.get("citations"),
                                                       "guard_notes": v.get("guard_notes"), "rationale": (v.get("rationale") or "")[:300]})
            except Exception as e:
                why = {"against": learned_from, "error": f"{type(e).__name__}: {e}"[:200]}
                self.audit.log(cid, "linked_recheck", {"triggered_by": learned_from, "error": str(e)[:200]})
            if not ok:
                with self.lock:
                    r["status"] = "needs_review"; r["recheck"] = why
                continue
            note = f"Auto-resolved: same pattern as {learned_from}, approved by a person."
            with self.lock:
                if self.pending.get(cid) is not r: continue
                self.pending.pop(cid); r["status"] = "finishing"
            self.flow.invoke(Command(resume={"resolution": "benign", "note": note, "linked": learned_from}),
                             {"configurable": {"thread_id": cid}})
            view = self._view(cid)
            with self.lock:
                r.update(view); r.update(status="completed", reviewer=None, note=note, linked_to=learned_from,
                                         rationale=v.get("rationale"), citations=v.get("citations"),
                                         confidence=v.get("confidence"), passages=v.get("passages"),
                                         similar_cases=v.get("similar_cases"), was_escalated=True)
    def _sweep_reject(self, learned_from):
        """After a person rejects an unexplained case, put similar waiting cases (same vendor, similar description) on hold.
        Hold only: nothing is ever approved this way. A case is held when the rejected case is its close match in memory."""
        with self.lock:
            others = [r for c, r in self.pending.items() if c != learned_from and r.get("edge") == "E2"]
            for r in others: r["status"] = "rechecking"
        for r in others:
            cid, inv = r["case_id"], r["invoice"]; match = None
            try:
                hits = self.store.search(CASES, f'{inv["vendor"]} {inv["category"]} {inv["description"]}', 3, {"vendor": inv["vendor"]})
                match = next((h for h in hits if h.id == f"CASE-{learned_from}" and h.score >= CASE_MATCH_MIN), None)
                self.audit.log(cid, "linked_hold_check", {"triggered_by": learned_from, "matched": bool(match),
                                                          "score": round(match.score, 3) if match else None})
            except Exception as e:
                self.audit.log(cid, "linked_hold_check", {"triggered_by": learned_from, "error": str(e)[:200]})
            if not match:
                with self.lock: r["status"] = "needs_review"
                continue
            note = f"Held automatically: same pattern as {learned_from}, which a person rejected. Nothing was approved."
            with self.lock:
                if self.pending.get(cid) is not r: continue
                self.pending.pop(cid); r["status"] = "finishing"
            self.flow.invoke(Command(resume={"resolution": "issue", "note": note, "linked": learned_from}),
                             {"configurable": {"thread_id": cid}})
            view = self._view(cid)
            with self.lock:
                r.update(view); r.update(status="completed", reviewer=None, note=note, linked_to=learned_from)
    def decide(self, cid, resolution, note=""):
        return self._finish_decide(self._begin_decide(cid), resolution, note)
    def submit_decide(self, cid, resolution, note=""):
        row = self._begin_decide(cid)
        self.pool.submit(self._finish_decide, row, resolution, note)
        return row
    def metrics(self):
        done = [h for h in self.history if h["status"] == "completed"]
        auto = [h for h in done if not h.get("reviewer")]
        return {"runs": len(self.history), "auto": len(auto), "reviewed": len([h for h in done if h.get("reviewer")]),
                "waiting": len(self.pending), "running": len([h for h in self.history if h["status"] in ("running", "finishing", "rechecking")]), "learned": len([h for h in self.history if h.get("learned")]),
                "guard_overrides": len([h for h in self.history if h.get("guard_notes")]),
                "agent_mode": self.agent_mode, "store": self.store.name, "provider": self.provider, "kb": self.kb, "env": self.env}
