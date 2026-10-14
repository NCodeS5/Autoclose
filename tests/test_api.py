from fastapi.testclient import TestClient
from autoclose.api import app

c = TestClient(app)

def test_health():
    assert c.get("/health").json() == {"status": "ok"}

def test_clean_close_completes_without_human():
    r = c.post("/closes", json={"scenario": "clean"}).json()
    assert r["status"] == "completed" and r["final"] == "published"

def test_duplicate_is_fixed_by_agent():
    r = c.post("/closes", json={"scenario": "duplicate"}).json()
    assert r["status"] == "completed" and any(a["action"] == "void" for a in r["actions"])

def test_writeoff_pauses_then_human_resolves():
    r = c.post("/closes", json={"scenario": "writeoff"}).json()
    assert r["status"] == "needs_review" and r["reasons"][0].startswith("E1")
    tid = r["thread_id"]
    assert tid in [q["thread_id"] for q in c.get("/queue").json()]
    d = c.post(f"/cases/{tid}/decision", json={"resolution": "issue", "note": "over limit"}).json()
    assert d["status"] == "completed" and d["final"] == "held"
    assert tid not in [q["thread_id"] for q in c.get("/queue").json()]

def test_bad_requests():
    assert c.post("/closes", json={}).status_code == 422
    assert c.post("/closes", json={"scenario": "nope"}).status_code == 422
    assert c.post("/cases/missing/decision", json={"resolution": "benign"}).status_code == 404

def test_metrics_counts():
    assert c.get("/metrics").json()["closes"] >= 1

def test_dashboard_and_history():
    r = c.get("/")
    assert r.status_code == 200 and "AutoClose" in r.text
    c.post("/closes", json={"scenario": "clean"})
    assert c.get("/history").json()[0]["scenario"] == "clean"


def test_demo_password_protects_everything_but_health(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "s3cret")
    assert c.get("/health").status_code == 200
    assert c.get("/").status_code == 401
    assert c.get("/review/metrics").status_code == 401
    assert c.get("/", auth=("anyone", "wrong")).status_code == 401
    assert c.get("/", auth=("anyone", "s3cret")).status_code == 200


def test_async_run_completes_and_audit_consistent():
    import time
    from fastapi.testclient import TestClient
    from autoclose.api import app
    c = TestClient(app)
    auth = {}
    c.post("/review/reset")
    ids = c.post("/review/run-all?wait=false").json()
    assert len(ids) == 31
    for _ in range(300):
        m = c.get("/review/metrics").json()
        if m["running"] == 0: break
        time.sleep(0.2)
    assert m["running"] == 0 and m["runs"] == 31
    assert all(h["status"] in ("completed", "needs_review") for h in c.get("/review/history").json())
    assert len(c.get("/review/audit/C01").json()) > 0
