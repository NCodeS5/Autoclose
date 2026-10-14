"""Free-flowing mode: the model decides which tools to call, in what order, and when it has enough evidence.
Code still computes the hard facts (tools return them) and the guard still checks the final decision.

AGENT_MODE=workflow (default, fixed steps) | agentic (this file). Needs a real model (anthropic or bedrock)."""
import json, os
from ..rag import DOCS, CASES
from .reasoner import comparison_note, offline_reason
from .retrieve import retrieve

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))

AGENT_SYSTEM = """You are an accounts payable review agent. You investigate one invoice by calling tools, then submit a decision.
Start with get_checks. Then look up the policy and contract passages you need, and past cases if the vendor has a pattern.
Use compare_to_purchase_order for any variance question: it does the arithmetic. Do not do arithmetic yourself.
Decide ONLY from what the tools return. Never approve write-offs, credit memos, unknown vendors, duplicates, or totals the policy says need higher approval.
If the evidence does not let you decide, escalate with edge E2 and low confidence. Cite only ids the tools returned.
Edge codes: E1 = policy names an approver for a routine case (credit memo, write-off, overage in the AP Manager's band, total above the materiality threshold, contract cap exceeded, unsupported late fee). E2 = unexplained or unclear (overage above the Controller limit, no purchase order, conflicting evidence, low confidence). E3 = novel (unknown vendor).
A similar past case that a person approved, for the same vendor and the same cause, may be followed even if the percentage differs.
Take every number in your rationale from tool results. When you are done, call submit_decision exactly once."""

_Q = {"type": "object", "properties": {"query": {"type": "string", "description": "what to look for, in plain words"}}, "required": ["query"]}
TOOLS = [
    {"name": "get_checks", "description": "Facts computed by code about this invoice: known vendor, duplicate, variance vs purchase order, billed this year, late fee, category mismatch, document type, total.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "lookup_vendor", "description": "Vendor master record: default category and whether billing is usage-based.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_policies", "description": "Search the company AP policies (P-1..P-9).", "input_schema": _Q},
    {"name": "search_contract", "description": "Search this vendor's contract (rates, caps, late-fee clause).", "input_schema": _Q},
    {"name": "search_past_cases", "description": "Search earlier resolved cases for this vendor, including ones a person approved.", "input_schema": _Q},
    {"name": "compare_to_purchase_order", "description": "Code-computed verdict on the invoice versus the purchase order and contract cap, using the policy passages you have retrieved so far. Search the match-tolerance and overage policies first.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "submit_decision", "description": "Final answer. Call once, when you have enough evidence.",
     "input_schema": {"type": "object", "properties": {
         "decision": {"type": "string", "enum": ["approve", "void_duplicate", "recode", "escalate"]},
         "edge": {"type": "string", "enum": ["E1", "E2", "E3", "none"]},
         "confidence": {"type": "number"}, "citations": {"type": "array", "items": {"type": "string"}},
         "rationale": {"type": "string"}, "new_category": {"type": "string"}},
         "required": ["decision", "edge", "confidence", "citations", "rationale"]}},
]

SF_LOOKUP = None          # set by the service when Salesforce is connected: callable(vendor_name) -> dict
SF_TOOL = {"name": "lookup_salesforce_vendor", "description": "Look up this vendor as an Account in the company's Salesforce org (account owner, industry). Read only; use it when you want to know who owns the vendor relationship.",
           "input_schema": {"type": "object", "properties": {}}}

def investigate(store, inv, ck, vendors, llm, max_steps=None):
    """Returns (decision_dict, passages, cases, trace). Raises on model/transport errors (the caller falls back)."""
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    passages, cases, trace, seen = [], [], [], set()
    final = {}
    def hits(ns, q, k, flt, label):
        out = []
        for h in store.search(ns, q, k, flt):
            if ns == DOCS:
                if h.id not in seen:
                    seen.add(h.id); passages.append({"id": h.id, "text": h.text, "score": round(h.score, 3), "why": label, "meta": h.meta})
            elif h.id not in {c["id"] for c in cases}:
                cases.append({"id": h.id, "text": h.text, "score": round(h.score, 3), "outcome": h.meta.get("outcome"), "vendor": h.meta.get("vendor")})
            out.append({"id": h.id, "text": h.text, **({"outcome": h.meta.get("outcome")} if ns == CASES else {})})
        return out or "no results"
    def run_tool(name, a):
        q = a.get("query", "")
        if name == "get_checks": return ck
        if name == "lookup_vendor":
            v = vendors.get(inv["vendor"]); return v or {"known_vendor": False}
        if name == "lookup_salesforce_vendor" and SF_LOOKUP: return SF_LOOKUP(inv["vendor"]) or {"found": None}
        if name == "search_policies": return hits(DOCS, q, 3, {"doc_type": "policy"}, "agent: " + q[:40])
        if name == "search_contract": return hits(DOCS, q, 4, {"doc_type": "contract", "vendor": inv["vendor"]}, "agent: " + q[:40])
        if name == "search_past_cases": return hits(CASES, q, 3, {"vendor": inv["vendor"]}, "")
        if name == "compare_to_purchase_order": return comparison_note(ck, passages, cases)
        return f"unknown tool {name}"
    bound = llm.bind_tools(TOOLS + ([SF_TOOL] if SF_LOOKUP else []))
    msgs = [SystemMessage(AGENT_SYSTEM), HumanMessage("INVOICE:\n" + json.dumps(inv))]
    nudged = False
    for step in range(max_steps or MAX_STEPS):
        ai = bound.invoke(msgs); msgs.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            if nudged: break
            nudged = True; msgs.append(HumanMessage("Call submit_decision now with your best decision.")); continue
        for c in calls:
            if c["name"] == "submit_decision":
                final = c["args"]; msgs.append(ToolMessage("recorded", tool_call_id=c["id"]))
                trace.append({"step": len(trace) + 1, "tool": "submit_decision", "args": {k: final.get(k) for k in ("decision", "edge", "confidence")}})
                continue
            try: res = run_tool(c["name"], c["args"] or {})
            except Exception as e: res = f"tool error: {str(e)[:200]}"
            trace.append({"step": len(trace) + 1, "tool": c["name"], "args": c["args"] or {},
                          "found": [x["id"] for x in res] if isinstance(res, list) else None})
            msgs.append(ToolMessage(json.dumps(res) if not isinstance(res, str) else res, tool_call_id=c["id"]))
        if final: break
    if not final:
        return ({"decision": None, "edge": None, "confidence": 0.0, "citations": [], "rationale": "",
                 "error": "agent did not submit a decision"}, passages, cases, trace)
    d = {"decision": final.get("decision"), "edge": None if final.get("edge") in (None, "none", "null", "") else final["edge"],
         "confidence": float(final.get("confidence") or 0), "citations": list(final.get("citations") or []),
         "rationale": final.get("rationale", ""), "new_category": final.get("new_category") or None}
    # The rules reference sees everything the model saw PLUS the standard planned retrieval, so it is judged on full evidence
    # and a narrow model search cannot make the rules look stricter than they are.
    try:
        sp, sc = retrieve(store, inv, ck)
    except Exception:
        sp, sc = [], []
    ids = {p["id"] for p in passages}
    best = {c["id"]: c for c in cases}
    for c in sc:                                   # keep the strongest similarity score seen for each past case
        if c["id"] not in best or c["score"] > best[c["id"]]["score"]: best[c["id"]] = c
    ref = offline_reason(inv, ck, passages + [p for p in sp if p["id"] not in ids], list(best.values()))
    d["reference"] = {"decision": ref["decision"], "edge": ref["edge"]}
    d["agent_steps"] = len(trace)
    return d, passages, cases, trace
