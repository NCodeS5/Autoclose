import json
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from autoclose.rag import LocalStore, CASES
from autoclose.review.service import ReviewService
from autoclose.review.guard import guard
from autoclose.review.checks import run_checks, load_vendors, load_prior
from autoclose.review.retrieve import retrieve
from autoclose.review.reasoner import reason
import run_rag_eval

def svc(llm=None): return ReviewService(store=LocalStore(), llm=llm)

def prep(case_id):
    s = svc(); inv = {k: v for k, v in s.cases[case_id].items() if k != "expected"}; inv["id"] = case_id
    ck = run_checks(inv, load_vendors(), load_prior()); ps, cs = retrieve(s.store, inv, ck)
    return inv, ck, ps, cs

def test_clean_invoice_finishes_without_human():
    r = svc().run("C01")
    assert r["status"] == "completed" and r["final"] == "approved" and "P-1" in r["citations"]

def test_writeoff_pauses_and_human_resumes():
    s = svc(); r = s.run("C17")
    assert r["status"] == "needs_review" and r["edge"] == "E1" and "C17" in s.pending
    done = s.decide("C17", "benign"); assert done["final"] == "approved"
    assert any(e["step"] == "human" for e in s.audit.events("C17"))

def test_reject_holds_invoice():
    s = svc(); s.run("C19"); assert s.decide("C19", "issue")["final"] == "held"

def test_feedback_loop_changes_next_decision():
    s = svc(); assert s.run("C23")["status"] == "needs_review"          # no memory yet
    s2 = svc(); s2.run("C22"); s2.decide("C22", "benign")
    r = s2.run("C23"); assert r["status"] == "completed" and "CASE-C22" in r["citations"]

def test_reject_does_not_teach_the_agent():
    s = svc(); s.run("C22"); s.decide("C22", "issue"); assert s.run("C23")["status"] == "needs_review"

def test_guard_blocks_model_approving_a_writeoff():
    inv, ck, ps, cs = prep("C17")
    g = guard({"decision": "approve", "edge": None, "confidence": 0.99, "citations": [ps[0]["id"]], "rationale": "fine"}, inv, ck, ps, cs)
    assert g["decision"] == "escalate" and g["edge"] == "E1" and g["overridden"]

def test_guard_blocks_invented_citation_and_numbers_and_low_confidence():
    inv, ck, ps, cs = prep("C01")
    good = {"decision": "approve", "edge": None, "confidence": 0.9, "citations": ["P-1"], "rationale": "Within tolerance."}
    assert guard(good, inv, ck, ps, cs)["decision"] == "approve"
    assert guard({**good, "citations": ["P-99"]}, inv, ck, ps, cs)["decision"] == "escalate"
    assert guard({**good, "rationale": "Invoice is 0.5% off, saves $777."}, inv, ck, ps, cs)["decision"] == "escalate"
    assert guard({**good, "confidence": 0.5}, inv, ck, ps, cs)["decision"] == "escalate"
    assert guard({**good, "citations": []}, inv, ck, ps, cs)["decision"] == "escalate"

def test_unparseable_llm_output_escalates_via_safety_net(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK", "none")
    s = svc(llm=FakeListChatModel(responses=["I think it is fine!"]))
    r = s.run("C01"); assert r["status"] == "needs_review" and any("safety net" in n for n in r["guard_notes"])

def test_llm_json_is_used_and_checked():
    out = json.dumps({"decision": "approve", "edge": None, "confidence": 0.9, "citations": ["P-1"], "rationale": "Within tolerance."})
    s = svc(llm=FakeListChatModel(responses=[out])); r = s.run("C01")
    assert r["status"] == "completed" and r["final"] == "approved"

def test_api_flow_end_to_end():
    from autoclose.review import api as rapi
    from autoclose.api import app
    rapi._svc = svc(); c = TestClient(app)
    assert c.get("/review/metrics").json()["kb"]["docs"] == 25
    r = c.post("/review/run", json={"case_id": "C16"}).json(); assert r["status"] == "needs_review"
    assert [q["case_id"] for q in c.get("/review/queue").json()] == ["C16"]
    assert c.post("/review/decision/C16", json={"resolution": "benign"}).json()["final"] == "approved"
    assert c.post("/review/decision/C16", json={"resolution": "benign"}).status_code == 404
    assert c.post("/review/run", json={"case_id": "nope"}).status_code == 404
    assert len(c.get("/review/audit/C16").json()) >= 5
    assert "Invoice Review" in c.get("/").text

def test_eval_gate_passes_and_fails(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")      # never call a real model from tests, even if .env sets one
    assert run_rag_eval.main(["--min-accuracy", "0.95", "--min-recall", "1.0", "--max-auto-error", "0", "--min-retrieval", "0.95"]) == 0
    assert run_rag_eval.main(["--min-accuracy", "1.01"]) == 1


def test_comparison_note_is_computed_in_code():
    from autoclose.review.reasoner import comparison_note
    inv, ck, ps, cs = prep("C02")
    assert "WITHIN" in comparison_note(ck, ps) and "1.10%" in comparison_note(ck, ps)
    inv, ck, ps, cs = prep("C13")
    assert "ABOVE" in comparison_note(ck, ps)


def test_guard_number_check_ignores_formatting():
    inv, ck, ps, cs = prep("C05")
    d = {"decision": "approve", "edge": None, "confidence": 0.95, "citations": ["P-1", "PP-1"],
         "rationale": "Amount of $6,695 matches the rent of $6,695.00 for 2026, a 3% increase."}
    assert guard(d, inv, ck, ps, cs)["decision"] == "approve"
    d["rationale"] = "Amount of $6,999 matches."
    assert guard(d, inv, ck, ps, cs)["decision"] == "escalate"


def test_throttled_model_falls_back_to_rules_and_says_so():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    class Boom(FakeListChatModel):
        def _call(self, *a, **k): raise RuntimeError("ThrottlingException: slow down")
    s = svc(llm=Boom(responses=["x"])); r = s.run("C20")
    assert r["edge"] == "E3" and r["status"] == "needs_review"
    assert any("model unavailable" in n for n in r["guard_notes"]) and "fallback" in r["provider"]
    assert s.run("C01")["final"] == "approved"

def test_hard_rule_fixes_edge_label_when_model_output_is_garbage(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK", "none")
    s = svc(llm=FakeListChatModel(responses=["nonsense"])); r = s.run("C20")
    assert r["edge"] == "E3" and any("safety net" in n for n in r["guard_notes"])


def test_comparison_note_covers_band_and_cap():
    from autoclose.review.reasoner import comparison_note
    inv, ck, ps, cs = prep("C22"); n = comparison_note(ck, ps)
    assert "ABOVE the 10% limit" in n and "E2" in n
    inv, ck, ps, cs = prep("C07"); assert "within the $12,000 contract cap" in comparison_note(ck, ps)
    inv, ck, ps, cs = prep("C15"); assert "ABOVE the $12,000 contract cap" in comparison_note(ck, ps)

def test_cross_check_blocks_model_approving_what_rules_escalate():
    out = json.dumps({"decision": "approve", "edge": None, "confidence": 0.95, "citations": ["P-2"], "rationale": "Overage is fine."})
    s = svc(llm=FakeListChatModel(responses=[out])); r = s.run("C12")
    assert r["status"] == "needs_review" and r["edge"] == "E1" and any("rules require a person" in n for n in r["guard_notes"])


def test_comparison_note_mentions_approved_case_memory_only_after_a_review():
    from autoclose.review.reasoner import comparison_note
    s = svc(); s.run("C22"); s.decide("C22", "benign")
    inv = {k: v for k, v in s.cases["C23"].items() if k != "expected"}; inv["id"] = "C23x"
    ck = run_checks(inv, s.vendors, s.prior); ps, cs = retrieve(s.store, inv, ck)
    assert "CASE-C22" in comparison_note(ck, ps, cs)
    s2 = svc(); inv2 = dict(inv); ck2 = run_checks(inv2, s2.vendors, s2.prior); ps2, cs2 = retrieve(s2.store, inv2, ck2)
    assert "earlier case" not in comparison_note(ck2, ps2, cs2)


def test_llm_gets_one_retry_when_first_reply_is_not_json():
    good = json.dumps({"decision": "approve", "edge": None, "confidence": 0.9, "citations": ["P-1"], "rationale": "Within tolerance."})
    s = svc(llm=FakeListChatModel(responses=["Sure, let me think.", good])); r = s.run("C01")
    assert r["status"] == "completed" and r["final"] == "approved" and "fallback" not in (r["provider"] or "")


def test_new_service_forgets_cases_learned_by_an_earlier_one():
    store = LocalStore(); a = ReviewService(store=store, llm=None); a.run("C22"); a.decide("C22", "benign")
    assert store.count(CASES) == 6
    b = ReviewService(store=store, llm=None)              # like a server restart against the same Pinecone index
    assert store.count(CASES) == 5 and b.run("C22")["status"] == "needs_review"


def test_production_mode_keeps_learned_cases_across_restarts(monkeypatch):
    monkeypatch.setenv("AUTOCLOSE_ENV", "production")
    store = LocalStore(); a = ReviewService(store=store, llm=None); a.run("C22"); a.decide("C22", "benign")
    b = ReviewService(store=store, llm=None)                  # restart
    assert store.count(CASES) == 6
    r = b.run("C23"); assert r["status"] == "completed" and "CASE-C22" in r["citations"]
    b.reset(); assert store.count(CASES) == 6                 # Reset must not erase real learned data in production


def test_approval_resolves_linked_waiting_case(monkeypatch):
    """Approving one case that is learned also resolves a waiting case that cites it after re-check."""
    from autoclose.review.service import ReviewService
    import autoclose.review.service as sv
    monkeypatch.setenv("AUTOCLOSE_ENV", "demo")
    s = ReviewService(llm=None)
    a, b = s.run("C22"), s.run("C23")
    assert a["status"] == "needs_review" and b["status"] == "needs_review"
    s.decide("C22", "benign", "ok")
    rows = {h["case_id"]: h for h in s.history}
    assert rows["C23"]["status"] == "completed" and rows["C23"]["linked_to"] == "C22" and "CASE-C22" in rows["C23"]["citations"]
    assert not s.pending
