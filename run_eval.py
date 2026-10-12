"""Measure AutoClose against labeled synthetic ground truth. Reviewer is simulated from labels.
Usage: python run_eval.py [N] [--min-recall X] [--max-auto-error Y]   (exit 1 if a gate is breached)"""
import json, sys, time, argparse
from dataclasses import asdict
from langgraph.types import Command
from autoclose.generate_data import generate
from autoclose.tools import Baseline
from autoclose.case_store import CaseStore
from autoclose.graph import build_graph

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=300)
ap.add_argument("--min-recall", type=float, default=None)
ap.add_argument("--max-auto-error", type=float, default=None)
args = ap.parse_args()
N = args.n
MANUAL_MIN_PER_RUN = 20      # ASSUMPTION: analyst minutes per manual close
REVIEW_MIN_PER_CASE = 6      # ASSUMPTION: reviewer minutes per escalated/audited case

warm = generate(60, anomaly_rate=0.0, seed=99)
base, store = Baseline(warm), CaseStore()
graph = build_graph(base, store)
runs = generate(N, seed=7)

rows = []
for r in runs:
    cfg = {"configurable": {"thread_id": f"run-{r.run_id}"}}
    t = time.perf_counter()
    out = graph.invoke({"run": asdict(r)}, cfg)
    if "__interrupt__" in out:     # reviewer = ground truth
        out = graph.invoke(Command(resume={"resolution": "issue" if r.needs_human else "benign"}), cfg)
    rows.append(dict(id=r.run_id, truth=sorted(r.edge_cases), route=out["route"], final=out["final"],
                     reason=out["reasons"][0], actions=out["actions"], injected=r.injected,
                     secs=time.perf_counter() - t))

needs = [x for x in rows if x["truth"]]
esc = [x for x in rows if x["route"] == "escalate"]
tp = [x for x in esc if x["truth"]]
fn = [x for x in needs if x["route"] != "escalate"]
done = [x for x in rows if x["route"] in ("auto", "audit")]          # completed without an escalation
auto_bad = [x for x in done if x["truth"]]
audit = [x for x in rows if x["route"] == "audit"]
touched = [x for x in rows if x["route"] != "auto"]

# agent actions vs injected truth
good = tot = 0
truth_actions = found = 0
for x in rows:
    dups = {i["entry"] for i in x["injected"] if i["type"] == "duplicate"}
    cats = {i["entry"]: i["true_category"] for i in x["injected"] if i["type"] == "misclassified"}
    truth_actions += len(dups) + len(cats)
    for a in x["actions"]:
        tot += 1
        ok = (a["action"] == "void" and a["entry_id"] in dups) or \
             (a["action"] == "recategorize" and cats.get(a["entry_id"]) == a["to_category"])
        good += ok; found += ok

print(f"Runs: {N}  needing a human: {len(needs)}  "
      f"(E1 {sum('E1' in x['truth'] for x in rows)}, E2 {sum('E2' in x['truth'] for x in rows)}, E3 {sum('E3' in x['truth'] for x in rows)})")
print(f"Automation rate (no human touch):    {sum(x['route']=='auto' for x in rows)/N:.1%}")
print(f"Escalation precision:                {len(tp)/max(1,len(esc)):.1%}  ({len(tp)}/{len(esc)})")
recall = len(needs) - len(fn)
print(f"Escalation recall:                   {recall/max(1,len(needs)):.1%}  (missed: {len(fn)})")
for e in ("E1", "E2", "E3"):
    grp = [x for x in rows if e in x["truth"]]
    print(f"  recall {e}: {sum(x['route']=='escalate' for x in grp)}/{len(grp)}")
agree = sum(1 for x in tp if x["reason"][:2] in x["truth"])
print(f"Escalation reason matches true case: {agree}/{len(tp)}")
print(f"Agent actions correct:               {good}/{tot}  (injected fixable issues found: {found}/{truth_actions})")
print(f"Error rate on auto-completed (truth): {len(auto_bad)/max(1,len(done)):.2%}  ({len(auto_bad)}/{len(done)})")
print(f"Audit sample errors:                 {sum(bool(x['truth']) for x in audit)}/{len(audit)}")
print(f"Human minutes /100 runs: {len(touched)*REVIEW_MIN_PER_CASE/N*100:.0f}  vs manual baseline {MANUAL_MIN_PER_RUN*100}")
print(f"Agent compute per run: {sum(x['secs'] for x in rows)/N*1000:.1f} ms (stub LLM)")
print("\nEscalation rate by window (feedback loop effect):")
w = max(1, N // 6)
for i in range(0, N, w):
    c = rows[i:i+w]
    print(f"  runs {i:>3}-{i+len(c)-1:<3}: {sum(x['route']=='escalate' for x in c)/len(c):.1%}")
json.dump(rows, open("results.json", "w"), indent=1)

fail = []
if args.min_recall is not None and recall / max(1, len(needs)) < args.min_recall:
    fail.append(f"recall below {args.min_recall}")
if args.max_auto_error is not None and len(auto_bad) / max(1, len(done)) > args.max_auto_error:
    fail.append(f"auto-completed error rate above {args.max_auto_error}")
if fail:
    print("\nGATE FAILED: " + "; ".join(fail)); sys.exit(1)
