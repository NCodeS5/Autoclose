from autoclose.review.service import ReviewService
from autoclose.rag import get_store
import autoclose.integrations.salesforce as sfm


class FakeResp:
    def __init__(self, data): self._d = data; self.content = b"x"
    def raise_for_status(self): pass
    def json(self): return self._d


def test_case_created_and_closed(monkeypatch):
    calls = []
    def fake(method, url, **kw):
        calls.append((method, url.split("/data/")[-1], kw.get("json")))
        if "CaseNumber" in url: return FakeResp({"CaseNumber": "00001001"})
        return FakeResp({"id": "500XX0000001"})
    monkeypatch.setattr(sfm.requests, "request", fake)
    monkeypatch.setattr(sfm.requests, "post", lambda *a, **k: FakeResp({"access_token": "t"}))
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    s = ReviewService(llm=None)
    s.sf = sfm.SalesforceCases("https://x.my.salesforce.com", "id", "secret")
    row = s.run("C16")                      # credit memo -> escalates
    assert row["status"] == "needs_review" and row["salesforce"]["number"] == "00001001"
    done = s.decide("C16", "issue", "over limit")
    assert done["salesforce"]["status"] == "Closed"
    assert any(c[0] == "POST" and "CaseComment" in c[1] for c in calls)


def test_salesforce_failure_never_blocks_review(monkeypatch):
    def boom(*a, **k): raise RuntimeError("down")
    monkeypatch.setattr(sfm.requests, "post", boom)
    s = ReviewService(llm=None)
    s.sf = sfm.SalesforceCases("https://x.my.salesforce.com", "id", "secret")
    row = s.run("C16")
    assert row["status"] == "needs_review" and "error" in row["salesforce"]
    assert s.decide("C16", "benign")["status"] == "completed"


def test_disabled_without_settings(monkeypatch):
    for k in ("SF_DOMAIN", "SF_CLIENT_ID", "SF_CLIENT_SECRET"): monkeypatch.delenv(k, raising=False)
    assert not sfm.SalesforceCases().enabled
