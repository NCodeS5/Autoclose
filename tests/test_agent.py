import json
from langchain_core.messages import AIMessage
from autoclose.review.service import ReviewService
from autoclose.rag import LocalStore


class ScriptedLLM:
    """Stands in for a tool-calling model: it picks its own tools, then submits."""
    def __init__(self, script): self.script, self.i = script, 0
    def bind_tools(self, tools): return self
    def invoke(self, msgs):
        calls = self.script[min(self.i, len(self.script) - 1)]; self.i += 1
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"t{self.i}{j}"} for j, (n, a) in enumerate(calls)])


def svc(llm, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "agentic"); monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    return ReviewService(store=LocalStore(), llm=llm)


def test_agent_chooses_tools_and_decides(monkeypatch):
    llm = ScriptedLLM([[("get_checks", {}), ("search_policies", {"query": "invoice match tolerance purchase order"})],
                       [("compare_to_purchase_order", {})],
                       [("submit_decision", {"decision": "approve", "edge": "none", "confidence": 0.93,
                                             "citations": ["P-1"], "rationale": "Within tolerance."})]])
    s = svc(llm, monkeypatch); r = s.run("C01")
    tools = [t["tool"] for t in r["trace"]]
    assert tools[0] == "get_checks" and "search_policies" in tools and tools[-1] == "submit_decision"
    assert s.agent_mode == "agentic" and r["provider"].endswith("(agent)")
    assert any(e["step"] == "agent_step" for e in s.audit.events("C01"))


def test_guard_still_blocks_agent_on_unknown_vendor(monkeypatch):
    llm = ScriptedLLM([[("get_checks", {})],
                       [("submit_decision", {"decision": "approve", "edge": "none", "confidence": 0.99,
                                             "citations": [], "rationale": "Looks fine."})]])
    s = svc(llm, monkeypatch); r = s.run("C19")      # unknown vendor
    assert r["status"] == "needs_review" and r["edge"] == "E3"


def test_agent_that_never_submits_is_escalated(monkeypatch):
    s = svc(ScriptedLLM([[("get_checks", {})]]), monkeypatch); r = s.run("C01")
    assert r["status"] == "needs_review" and r["guard_notes"]


def test_agent_error_falls_back_to_workflow(monkeypatch):
    class Boom(ScriptedLLM):
        def invoke(self, msgs): raise RuntimeError("down")
    monkeypatch.setenv("LLM_FALLBACK", "offline")
    s = svc(Boom([[]]), monkeypatch); r = s.run("C01")
    assert r["status"] in ("completed", "needs_review") and any(e["step"] == "agent_failed" for e in s.audit.events("C01"))


def test_linked_approval_works_in_agentic_mode(monkeypatch):
    """Approving C22 re-checks waiting C23 with the agent; the rules reference must not block it."""
    esc = [[("get_checks", {})], [("submit_decision", {"decision": "escalate", "edge": "E2", "confidence": 0.5, "citations": [], "rationale": "No earlier pattern."})]]
    recheck = [[("get_checks", {}), ("search_policies", {"query": "usage-based services vary from purchase order"}),
                ("search_past_cases", {"query": "approved usage"})],
               [("submit_decision", {"decision": "approve", "edge": "none", "confidence": 0.9, "citations": ["CASE-C22", "P-9"],
                                     "rationale": "Same vendor and cause as the case a person approved."})]]
    llm = ScriptedLLM(esc + esc + recheck)
    s = svc(llm, monkeypatch)
    s.run("C22"); s.run("C23")
    s.decide("C22", "benign", "ok")
    ev = [e for e in s.audit.events("C23") if e["step"] == "linked_recheck"]
    assert ev and "guard_notes" in ev[0]["detail"]
    assert ev[0]["detail"]["auto_resolved"], ev[0]["detail"]
