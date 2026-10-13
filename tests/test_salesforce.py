"""Salesforce Cases link, tested against a fake HTTP layer (no real org needed)."""
import autoclose.integrations.salesforce as sfm
from autoclose.rag import LocalStore
from autoclose.review.service import ReviewService

class Fake:
    def __init__(self, fail=False): self.calls, self.fail = [], fail
    def post(self, url, **kw):
        return R({"access_token": "t"})
    def request(self, method, url, **kw):
        if self.fail: raise RuntimeError("boom")
        self.calls.append((method, url.split("/data/")[1], kw.get("json")))
        if method == "POST" and url.endswith("/Case"): return R({"id": "500X"})
        if method == "GET": return R({"CaseNumber": "00001"})
        return R({})

class R:
    def __init__(self, j): self._j, self.content = j, b"x" if j else b""
    def raise_for_status(self): pass
    def json(self): return self._j

def svc(monkeypatch, fake, enabled=True):
    monkeypatch.setattr(sfm, "requests", fake)
    if enabled:
        for k, v in (("SF_DOMAIN", "https://x.my.salesforce.com"), ("SF_CLIENT_ID", "a"), ("SF_CLIENT_SECRET", "b")): monkeypatch.setenv(k, v)
    else:
        for k in ("SF_DOMAIN", "SF_CLIENT_ID", "SF_CLIENT_SECRET"): monkeypatch.delenv(k, raising=False)
    return ReviewService(store=LocalStore(), llm=None)

def test_case_opened_then_closed(monkeypatch):
    f = Fake(); s = svc(monkeypatch, f)
    row = s.run("C16")
    assert row["status"] == "needs_review" and row["salesforce"]["number"] == "00001"
    done = s.decide("C16", "benign", "ok")
    assert done["salesforce"]["status"] == "Closed"
    assert any(c[0] == "PATCH" for c in f.calls) and any(c[1].endswith("/sobjects/CaseComment") for c in f.calls)

def test_salesforce_failure_never_blocks(monkeypatch):
    s = svc(monkeypatch, Fake(fail=True))
    row = s.run("C16")
    assert row["status"] == "needs_review" and "error" in row["salesforce"]
    assert s.decide("C16", "issue", "no")["status"] == "completed"

def test_disabled_without_settings(monkeypatch):
    s = svc(monkeypatch, Fake(), enabled=False)
    assert s.metrics()["salesforce"] is False
    assert "salesforce" not in s.run("C16") or s.run("C16").get("salesforce") is None
