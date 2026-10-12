"""Specialist agents. Each returns the same JSON contract an LLM tool-calling agent would:
{agent, resolution: no_issue|auto_fixed|cannot_resolve, confidence, evidence, note}.
These are deterministic reference implementations; swap in Bedrock tool-calling agents
(see build guide, step 5) without touching the graph or the gate."""
import copy
from . import tools

SOFT = {"variance", "forecast_oor"}

def signature(findings):
    toks = set()
    for f in findings:
        toks.add(f["kind"] + (f"_{f['category']}" if f.get("category") else ""))
    return " ".join(sorted(toks))

def _result(agent, resolution, confidence, evidence, note):
    return dict(agent=agent, resolution=resolution, confidence=confidence, evidence=evidence, note=note)

def validation_agent(run, base):
    """Resolves routine issues itself by PROPOSING actions (source data is never edited)."""
    findings, actions, voided = tools.validate(run, base), [], set()
    for f in findings:
        if f["kind"] == "duplicate":
            actions.append(dict(action="void", entry_id=f["entry_id"], reason="exact duplicate"))
            voided.add(f["entry_id"]); f["resolved"] = True
    for f in findings:
        if f["kind"] == "misclassified":
            if f["entry_id"] not in voided:
                actions.append(dict(action="recategorize", entry_id=f["entry_id"],
                                    to_category=f["to_category"], reason="vendor norm"))
            f["resolved"] = True
    open_ = [f for f in findings if not f.get("resolved")]
    if open_:
        res = _result("validation", "cannot_resolve", 0.5, [f["detail"] for f in open_], "unresolved findings")
    elif actions:
        res = _result("validation", "auto_fixed", 0.88, [a["entry_id"] for a in actions], "proposed corrections")
    else:
        res = _result("validation", "no_issue", 0.95, [], "clean")
    return res, findings, actions

def apply_actions(run, actions):
    """Working copy with proposed fixes applied. Used downstream; the original stays untouched."""
    w = copy.deepcopy(run)
    void = {a["entry_id"] for a in actions if a["action"] == "void"}
    recat = {a["entry_id"]: a["to_category"] for a in actions if a["action"] == "recategorize"}
    w["ledger"] = [e for e in w["ledger"] if e["id"] not in void]
    for e in w["ledger"]:
        if e["id"] in recat:
            e["category"] = recat[e["id"]]
    return w

def reconciliation_agent(working):
    f = tools.reconcile(working)
    if f:
        return _result("reconciliation", "cannot_resolve", 0.5, [x["detail"] for x in f], "unmatched entries"), f
    return _result("reconciliation", "no_issue", 0.95, [], "all matched"), f

def analytics_agent(working, store, sim_threshold=0.85):
    f = tools.variance(working) + tools.forecast(working)
    if not f:
        return _result("analytics", "no_issue", 0.95, [], "within norms"), f
    sim, label = store.best_match(signature(f))
    if label == "benign" and sim >= sim_threshold:
        for x in f:
            x["resolved"] = True
        return _result("analytics", "no_issue", 0.9, [x["detail"] for x in f], f"known benign pattern (sim={sim:.2f})"), f
    return _result("analytics", "cannot_resolve", 0.5, [x["detail"] for x in f], "unexplained variance"), f
