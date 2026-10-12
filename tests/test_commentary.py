from autoclose.commentary import validate_commentary, make_commentator, template_commentary
from langchain_core.runnables import RunnableLambda

F = [dict(kind="duplicate", detail="L7dup duplicates L7"),
     dict(kind="missing_bank", detail="L3 AWS $4,200.00 has no bank match")]
RUN = dict(run_id=1, entity="ENT-01")

def test_template_is_valid():
    assert validate_commentary(template_commentary(F), F) == []

def test_flags_uncited_line():
    assert validate_commentary("- AWS looks high\nSummary: [F1][F2]", F)

def test_flags_invented_number():
    bad = "- L3 AWS $5,100.00 has no bank match [F2]\n- L7dup duplicates L7 [F1]\nSummary: [F1][F2]"
    assert any("5100" in p for p in validate_commentary(bad, F))

def test_flags_unknown_id_and_missing_finding():
    assert validate_commentary("- L7dup duplicates L7 [F9]\nSummary: [F9]", F)

def test_retry_then_fallback_marks_not_ok():
    bad = RunnableLambda(lambda _: "- looks fine")   # never complies
    text, ok, attempts, problems = make_commentator(bad)(RUN, F)
    assert not ok and attempts == 2 and problems

def test_retry_recovers():
    calls = {"n": 0}
    def flaky(_):
        calls["n"] += 1
        return "- looks fine" if calls["n"] == 1 else template_commentary(F)
    text, ok, attempts, _ = make_commentator(RunnableLambda(flaky))(RUN, F)
    assert ok and attempts == 2
