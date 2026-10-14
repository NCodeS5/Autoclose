"""Evaluate the invoice review agent on the 31 hand-written cases (data/cases.json).

python run_rag_eval.py [--min-accuracy 0.95 --min-recall 1.0 --max-auto-error 0.0 --min-retrieval 0.95]
Reviewer is simulated from each case's 'reviewer' field. Time figures use stated ASSUMPTIONS, not measurements."""
import argparse, sys, time
try:
    from dotenv import load_dotenv; load_dotenv()      # same .env the server uses
except ImportError:
    pass
from autoclose.rag import LocalStore, get_store, CASES
from autoclose.review.service import ReviewService

MANUAL_MIN, REVIEW_MIN = 5.0, 3.0      # ASSUMPTIONS: minutes to check one invoice by hand / to review one escalation

def run(store, order=None):
    svc = ReviewService(store=store)
    rows = []
    for c in svc.cases.values():
        if order and c["id"] not in order: continue
        r = svc.run(c["id"]); e = c["expected"]
        if r["status"] == "needs_review":
            r = svc.decide(r["case_id"], "benign" if (e.get("reviewer") or "approve") == "approve" else "issue")
        pids = {p["id"] for p in r["passages"]} | {s["id"] for s in r["similar_cases"]}
        need = set(e["cites"])
        rows.append(dict(id=c["id"], exp_dec=e["decision"], exp_edge=e["edge"], dec=r["proposed"], edge=r["edge"],
                         guard=bool(r["guard_notes"]), reviewer=r.get("reviewer"), ms=r["ms"],
                         retr=need <= pids if need else None, learned=r.get("learned"), provider=r.get("provider"),
                         rationale=r.get("rationale"), notes=r.get("guard_notes"),
                         missing=sorted(need - pids), cases=[(x["id"], x["score"]) for x in r["similar_cases"]]))
    learned = [f'CASE-{x["id"]}' for x in rows if x["learned"]]
    if learned: store.delete_ids(CASES, learned)      # leave the shared store as we found it
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None); ap.add_argument("--min-accuracy", type=float); ap.add_argument("--min-recall", type=float)
    ap.add_argument("--max-auto-error", type=float); ap.add_argument("--min-retrieval", type=float)
    a = ap.parse_args(argv)
    rows = run(get_store(a.store) if a.store else LocalStore())
    n = len(rows)
    acc = sum(r["dec"] == r["exp_dec"] and r["edge"] == r["exp_edge"] for r in rows) / n
    esc_exp = [r for r in rows if r["exp_dec"] == "escalate"]; esc_got = [r for r in rows if r["dec"] == "escalate"]
    tp = [r for r in esc_got if r["exp_dec"] == "escalate"]
    prec, rec = len(tp) / max(len(esc_got), 1), len(tp) / max(len(esc_exp), 1)
    auto = [r for r in rows if r["dec"] != "escalate"]
    auto_err = sum(r["exp_dec"] != r["dec"] for r in auto) / max(len(auto), 1)
    retr = [r["retr"] for r in rows if r["retr"] is not None]
    hit = sum(retr) / len(retr)
    print(f"cases {n} | agent handled alone {len(auto)} ({len(auto)/n:.0%}) | escalated {len(esc_got)}")
    print(f"decision+edge accuracy        {acc:.1%}")
    print(f"escalation precision / recall {prec:.1%} / {rec:.1%}")
    for e in ("E1", "E2", "E3"):
        ex = [r for r in rows if r["exp_edge"] == e]; go = [r for r in rows if r["edge"] == e and r["dec"] == "escalate"]
        t = [r for r in go if r["exp_edge"] == e]
        print(f"  {e}: expected {len(ex)}, flagged {len(go)}, correct {len(t)}")
    print(f"auto-handled error rate       {auto_err:.1%}  (of {len(auto)})")
    print(f"retrieval hit rate            {hit:.1%}  (expected clause was in the retrieved set)")
    print(f"guard stepped in on           {sum(r['guard'] for r in rows)} cases")
    print(f"feedback loop: C23 handled alone after C22 review: {next(r for r in rows if r['id']=='C23')['dec']=='approve'}")
    solo = run(LocalStore(), order={"C23"})[0]
    print(f"  same invoice with no prior review: {solo['dec']} ({solo['edge']})")
    manual = n * MANUAL_MIN; withagent = len(esc_got) * REVIEW_MIN
    miss = [r for r in rows if r["dec"] != r["exp_dec"] or r["edge"] != r["exp_edge"]]
    print(f"\nDISAGREEMENTS WITH THE EXPECTED ANSWER ({len(miss)}):")
    for r in miss:
        print(f'  {r["id"]}: got {r["dec"]} {r["edge"]}, expected {r["exp_dec"]} {r["exp_edge"]}')
        print(f'      said: {(r["rationale"] or "")[:200]}')
        if r["notes"]: print(f'      guard: {r["notes"]}')
        if r["cases"]: print(f'      similar cases retrieved: {r["cases"]}')
    for r in rows:
        if r["retr"] is False: print(f'  retrieval miss {r["id"]}: expected clause(s) {r["missing"]} were not retrieved')
    print(f'  model used: {rows[0]["provider"]}\n')
    print(f"time (assumptions {MANUAL_MIN:g} min/invoice by hand, {REVIEW_MIN:g} min/review): {manual:.0f} min manual vs {withagent:.0f} min with agent")
    print(f"avg agent latency {sum(r['ms'] for r in rows)/n:.0f} ms/case (includes the model call when one is configured)")
    bad = []
    if a.min_accuracy is not None and acc < a.min_accuracy: bad.append("accuracy")
    if a.min_recall is not None and rec < a.min_recall: bad.append("escalation recall")
    if a.max_auto_error is not None and auto_err > a.max_auto_error: bad.append("auto error")
    if a.min_retrieval is not None and hit < a.min_retrieval: bad.append("retrieval")
    if bad: print("GATE FAILED:", ", ".join(bad)); return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
