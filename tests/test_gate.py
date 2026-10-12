import random
from dataclasses import asdict
import pytest
from langgraph.types import Command
from autoclose.generate_data import generate, _base_run, _inject
from autoclose.tools import Baseline
from autoclose.case_store import CaseStore
from autoclose.graph import build_graph

@pytest.fixture(scope="module")
def baseline():
    return Baseline(generate(60, anomaly_rate=0.0, seed=99))

def run_case(baseline, kind, seed=3):
    rng = random.Random(seed)
    r = _base_run(rng, 1)
    if kind:
        _inject(rng, r, kind)
    g = build_graph(baseline, CaseStore(), audit_rate=0.0)
    cfg = {"configurable": {"thread_id": f"t-{kind}-{seed}"}}
    g.invoke({"run": asdict(r)}, cfg)
    return g.get_state(cfg).values

def test_clean_run_auto_completes(baseline):
    v = run_case(baseline, None)
    assert v["route"] == "auto" and v["actions"] == []

def test_duplicate_is_fixed_by_agent_not_human(baseline):
    v = run_case(baseline, "duplicate")
    assert v["route"] == "auto"
    assert any(a["action"] == "void" for a in v["actions"])

def test_misclassified_is_fixed_by_agent_not_human(baseline):
    v = run_case(baseline, "misclassified")
    assert v["route"] == "auto"
    assert any(a["action"] == "recategorize" for a in v["actions"])

def test_E1_writeoff_escalates(baseline):
    v = run_case(baseline, "writeoff")
    assert v["route"] == "escalate" and v["reasons"][0].startswith("E1")

def test_E2_missing_bank_escalates(baseline):
    v = run_case(baseline, "missing_bank")
    assert v["route"] == "escalate" and v["reasons"][0].startswith("E2")

def test_E3_novel_vendor_escalates(baseline):
    v = run_case(baseline, "novel_vendor")
    assert v["route"] == "escalate" and v["reasons"][0].startswith("E3")

def test_source_data_never_mutated(baseline):
    rng = random.Random(3); r = _base_run(rng, 1); _inject(rng, r, "duplicate")
    before = len(r.ledger)
    g = build_graph(baseline, CaseStore(), audit_rate=0.0)
    g.invoke({"run": asdict(r)}, {"configurable": {"thread_id": "m"}})
    assert len(r.ledger) == before   # proposal only, original untouched
