"""LangGraph review flow:
intake -> checks -> retrieve -> reason -> guard -> route -> (human_review: interrupt) -> finalize -> feedback

State is checkpointed (MemorySaver) so a case waiting for a person survives between HTTP calls."""
import os, random, time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from ..rag import CASES, Chunk
from .checks import run_checks, learned_recode
from .retrieve import retrieve
from .reasoner import reason
from .agent import investigate
from .guard import guard

class S(TypedDict, total=False):
    invoice: dict
    checks: dict
    passages: list
    cases: list
    decision: dict
    route: str            # auto | escalate | audit
    human: dict
    final: str            # approved | voided | recoded | held
    proposed_action: Optional[dict]
    steps: int
    learned_id: Optional[str]     # set when this decision was saved to memory
    trace: list

def build_flow(store, vendors, prior, audit, llm=None, audit_rate=0.0, seed=1, agent_mode="workflow", seen=None):
    rng = random.Random(seed)
    def log(state, step, detail): audit.log(state["invoice"]["id"], step, detail)

    def n_checks(s):
        ck = run_checks(s["invoice"], vendors, prior, seen); log(s, "checks", ck); return {"checks": ck, "steps": 1}
    def n_retrieve(s):
        t = time.time(); ps, cs = retrieve(store, s["invoice"], s["checks"])
        log(s, "retrieve", {"store": store.name, "passages": [p["id"] for p in ps], "cases": [c["id"] for c in cs],
                            "ms": round((time.time() - t) * 1000)})
        return {"passages": ps, "cases": cs}
    def n_reason(s):
        t = time.time(); d = reason(s["invoice"], s["checks"], s["passages"], s["cases"], llm)
        log(s, "reason", {**d, "ms": round((time.time() - t) * 1000)}); return {"decision": d}
    def n_agent(s):
        """Free-flowing mode: the model picks its own tools. If it fails, fall back to the fixed workflow."""
        t = time.time()
        try:
            d, ps, cs, trace = investigate(store, s["invoice"], s["checks"], vendors, llm)
            d["provider"] = f'{os.getenv("LLM_PROVIDER")} (agent)'
            for tr in trace: log(s, "agent_step", tr)
            log(s, "reason", {**d, "ms": round((time.time() - t) * 1000)})
            return {"passages": ps, "cases": cs, "decision": d, "trace": trace}
        except Exception as e:
            log(s, "agent_failed", {"error": f"{type(e).__name__}: {e}"[:300], "fallback": "fixed workflow"})
            ps, cs = retrieve(store, s["invoice"], s["checks"])
            d = reason(s["invoice"], s["checks"], ps, cs, llm); d.setdefault("notes", []).append(f"agent mode failed, used fixed workflow: {type(e).__name__}")
            return {"passages": ps, "cases": cs, "decision": d, "trace": []}
    def n_guard(s):
        g = guard(s["decision"], s["invoice"], s["checks"], s["passages"], s["cases"])
        log(s, "guard", {"final_decision": g["decision"], "edge": g.get("edge"), "category_check": g.get("category_check"), "notes": g["guard_notes"]})
        route = "escalate" if (g["decision"] == "escalate" or g.get("category_check")) else ("audit" if rng.random() < audit_rate else "auto")
        return {"decision": g, "route": route}
    def n_route(s): return s["route"]
    def n_human(s):
        d = s["decision"]
        label = "category check" if d.get("category_check") else (d.get("edge") or "audit")
        ans = interrupt({"reason": f'{label}: {d.get("rationale") or "; ".join(d["guard_notes"])}',
                         "proposed": d.get("proposed"), "decision": d})
        log(s, "human", ans); return {"human": ans}
    def n_final(s):
        d, h = s["decision"], s.get("human")
        if h:
            if h["resolution"] == "issue": final = "held"
            else:
                final = {"void_duplicate": "voided", "recode": "recoded"}.get(d["decision"], "approved")
                if d["decision"] == "escalate": final = "approved"
        else:
            final = {"approve": "approved", "void_duplicate": "voided", "recode": "recoded"}[d["decision"]]
        act = None
        if final == "voided": act = {"action": "void", "invoice": s["invoice"]["invoice_no"]}
        if final == "recoded": act = {"action": "recode", "invoice": s["invoice"]["invoice_no"], "to": d.get("new_category")}
        log(s, "final", {"final": final, "action": act}); return {"final": final, "proposed_action": act}
    def n_feedback(s):
        """A reviewer-decided unexplained case becomes memory the next similar case can cite (approved or rejected). Cases auto-resolved from another decision are not learned again."""
        h, inv = s.get("human"), s["invoice"]
        if h and not h.get("linked") and s["decision"].get("category_check") and h["resolution"] == "benign":
            new = s["checks"]["vendor_default_category"]
            text = f'{inv["vendor"]} {inv["category"]} {inv["description"]} coded as {inv["category"]} recoded to {new} approved by reviewer'
            chunk = Chunk(f'CASE-{inv["id"]}', text, {"vendor": inv["vendor"], "outcome": "approved", "title": inv["id"]})
            t = time.time(); store.upsert(CASES, [chunk]); visible = store.wait_visible(CASES, chunk)
            log(s, "feedback", {"saved": chunk.id, "kind": "category recode", "searchable": visible, "ms": round((time.time() - t) * 1000)})
            return {"learned_id": chunk.id}
        if h and not h.get("linked") and s["decision"].get("edge") == "E2":
            word, outcome = ("approved", "approved") if h["resolution"] == "benign" else ("rejected", "rejected")
            text = f'{inv["vendor"]} {inv["category"]} {inv["description"]} {word} by reviewer after variance of {s["checks"]["variance_pct"]} percent'
            chunk = Chunk(f'CASE-{inv["id"]}', text, {"vendor": inv["vendor"], "outcome": outcome, "title": inv["id"]})
            t = time.time(); store.upsert(CASES, [chunk]); visible = store.wait_visible(CASES, chunk)
            log(s, "feedback", {"saved": chunk.id, "searchable": visible, "ms": round((time.time() - t) * 1000)})
            return {"learned_id": chunk.id}
        return {}
    g = StateGraph(S)
    for n, f in [("checks", n_checks), ("agent", n_agent), ("retrieve", n_retrieve), ("reason", n_reason), ("guard", n_guard),
                 ("human_review", n_human), ("finalize", n_final), ("feedback", n_feedback)]:
        g.add_node(n, f)
    g.add_edge(START, "checks")
    agentic = agent_mode == "agentic" and llm is not None
    g.add_conditional_edges("checks", lambda s: "agent" if agentic else "retrieve", {"agent": "agent", "retrieve": "retrieve"})
    g.add_edge("agent", "guard"); g.add_edge("retrieve", "reason"); g.add_edge("reason", "guard")
    g.add_conditional_edges("guard", n_route, {"auto": "finalize", "audit": "human_review", "escalate": "human_review"})
    g.add_edge("human_review", "finalize"); g.add_edge("finalize", "feedback"); g.add_edge("feedback", END)
    return g.compile(checkpointer=MemorySaver())
