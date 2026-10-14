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


class FakeCustom(Fake):
    """Org where custom fields exist (or not, when strict=True is flipped by the test)."""
    def __init__(self, reject_custom=False): super().__init__(); self.reject = reject_custom
    def request(self, method, url, **kw):
        body = kw.get("json") or {}
        if self.reject and any(k.endswith("__c") for k in body): raise RuntimeError("No such column")
        return super().request(method, url, **kw)

def test_custom_fields_filled_and_final_decision_set(monkeypatch):
    monkeypatch.setenv("SF_CUSTOM_FIELDS", "true"); f = FakeCustom(); s = svc(monkeypatch, f)
    row = s.run("C16"); assert row["salesforce"]["custom_fields"] is True
    posted = [c[2] for c in f.calls if c[0] == "POST" and c[1].endswith("/sobjects/Case")][0]
    assert posted["Autoclose_Review_Id__c"] == "C16" and posted["Agent_Edge__c"] and "Agent_Confidence__c" in posted
    s.decide("C16", "issue", "no")
    patch = [c[2] for c in f.calls if c[0] == "PATCH"][0]
    assert patch == {"Status": "Closed", "Final_Decision__c": "Rejected"}

def test_missing_custom_fields_never_lose_the_case(monkeypatch):
    monkeypatch.setenv("SF_CUSTOM_FIELDS", "true"); f = FakeCustom(reject_custom=True); s = svc(monkeypatch, f)
    row = s.run("C16")
    assert row["salesforce"]["number"] == "00001" and row["salesforce"]["custom_fields"] is False
    assert s.decide("C16", "benign", "ok")["salesforce"]["status"] == "Closed"

def test_custom_fields_off_by_default(monkeypatch):
    monkeypatch.delenv("SF_CUSTOM_FIELDS", raising=False); f = FakeCustom(); s = svc(monkeypatch, f)
    s.run("C16")
    posted = [c[2] for c in f.calls if c[0] == "POST" and c[1].endswith("/sobjects/Case")][0]
    assert not any(k.endswith("__c") for k in posted)
