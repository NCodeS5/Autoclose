"""LangGraph orchestrator: ingest -> validation agent -> reconciliation agent -> analytics agent
-> report -> escalation gate -> (human review via interrupt) -> feedback -> end.

Humans see only THREE edge cases:  E1 policy (write-off/adjusting entry),
E2 unexplained (agents can't explain it with evidence), E3 novel (new vendor / unseen pattern).
Safety nets (e.g. failed citation check) route to the same queue."""
import random
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from . import agents
from .commentary import make_commentator

class State(TypedDict, total=False):
    run: dict
    working: dict
    findings: list
    actions: list
    agent_results: list
    report: str
    commentary_ok: bool
    attempts: int
    route: str          # auto | audit | escalate
    reasons: list
    decision: dict
    final: str          # published | held

def build_graph(baseline, store, audit_rate=0.05, sim_threshold=0.85, novel_threshold=0.5, seed=1, llm=None):
    rng = random.Random(seed)
    commentate = make_commentator(llm)

    def ingest(s):
        assert s["run"]["ledger"] and s["run"]["bank"], "empty inputs"
        return {"findings": [], "actions": [], "agent_results": []}

    def validation(s):
        res, findings, actions = agents.validation_agent(s["run"], baseline)
        return {"findings": findings, "actions": actions,
                "working": agents.apply_actions(s["run"], actions),
                "agent_results": s["agent_results"] + [res]}

    def reconciliation(s):
        res, f = agents.reconciliation_agent(s["working"])
        return {"findings": s["findings"] + f, "agent_results": s["agent_results"] + [res]}

    def analytics(s):
        res, f = agents.analytics_agent(s["working"], store, sim_threshold)
        return {"findings": s["findings"] + f, "agent_results": s["agent_results"] + [res]}

    def report(s):
        text, ok, attempts, _ = commentate(s["run"], s["findings"])
        head = f"Run {s['run']['run_id']} ({s['run']['entity']}): {len(s['findings'])} finding(s)."
        return {"report": head + "\n" + text, "commentary_ok": ok, "attempts": attempts}

    def esc(reason):
        return {"route": "escalate", "reasons": [reason]}

    def gate(s):
        F = s["findings"]
        open_ = [f for f in F if not f.get("resolved")]
        kinds, open_kinds = {f["kind"] for f in F}, {f["kind"] for f in open_}
        if "policy_writeoff" in kinds:
            return esc("E1 policy: write-off / adjusting entry")
        if not s.get("commentary_ok", True):
            return esc("safety net: commentary failed citation check")
        if "novel_vendor" in open_kinds:
            return esc("E3 novel: unknown vendor")
        if open_ and open_kinds <= agents.SOFT:
            sim, _ = store.best_match(agents.signature(open_))
            if sim < novel_threshold:
                return esc(f"E3 novel: no similar resolved case (sim={sim:.2f})")
        weak = [r["agent"] for r in s["agent_results"]
                if r["resolution"] == "cannot_resolve" or r["confidence"] < 0.75]
        if weak:
            return esc("E2 unexplained: " + ", ".join(weak))
        if rng.random() < audit_rate:
            return {"route": "audit", "reasons": ["random audit sample"]}
        return {"route": "auto", "reasons": ["resolved by agents" if s["actions"] else "clean"]}

    def human_review(s):
        # Pauses the run (checkpointed) until a reviewer resumes it with a decision dict.
        decision = interrupt({"run_id": s["run"]["run_id"], "report": s["report"],
                              "route": s["route"], "reasons": s["reasons"], "actions": s["actions"]})
        return {"decision": decision}

    def feedback(s):
        d = s.get("decision")
        open_ = [f for f in s["findings"] if not f.get("resolved")]
        if d and d["resolution"] == "benign" and open_ and {f["kind"] for f in open_} <= agents.SOFT:
            store.add(agents.signature(open_), "benign")   # learn: this pattern auto-resolves next time
        return {"final": "held" if d and d["resolution"] == "issue" else "published"}

    g = StateGraph(State)
    for name, fn in [("ingest", ingest), ("validation", validation), ("reconciliation", reconciliation),
                     ("analytics", analytics), ("report", report), ("gate", gate),
                     ("human_review", human_review), ("feedback", feedback)]:
        g.add_node(name, fn)
    chain = ["ingest", "validation", "reconciliation", "analytics", "report", "gate"]
    g.add_edge(START, chain[0])
    for a, b in zip(chain, chain[1:]):
        g.add_edge(a, b)
    g.add_conditional_edges("gate", lambda s: "feedback" if s["route"] == "auto" else "human_review",
                            {"feedback": "feedback", "human_review": "human_review"})
    g.add_edge("human_review", "feedback")
    g.add_edge("feedback", END)
    return g.compile(checkpointer=MemorySaver())
