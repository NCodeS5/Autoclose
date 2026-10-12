"""Rule-based tools. Each returns findings: {kind, confidence, detail, ...}.
In production the validation tool would also consult an LLM + RAG store; here the
deterministic version keeps results reproducible and measurable."""
import statistics as st

class Baseline:
    """Per-vendor norms learned from clean warm-up runs."""
    def __init__(self, warmup_runs):
        amts, cats = {}, {}
        for r in warmup_runs:
            for e in r.ledger:
                amts.setdefault(e["vendor"], []).append(e["amount"])
                cats.setdefault(e["vendor"], set()).add(e["category"])
        self.stats = {v: (st.mean(a), st.pstdev(a) or 1.0, next(iter(cats[v])))
                      for v, a in amts.items()}

def _baseline_from_csv(path):
    """Load a baseline produced by spark/build_baseline.py (columns: vendor, mean, sd, category)."""
    import csv
    b = Baseline.__new__(Baseline)
    b.stats = {r["vendor"]: (float(r["mean"]), float(r["sd"]) or 1.0, r["category"])
               for r in csv.DictReader(open(path))}
    return b
Baseline.from_csv = staticmethod(_baseline_from_csv)

def validate(run, base):
    f, L = [], run["ledger"]
    seen = {}
    for e in L:
        key = (e["vendor"], round(e["amount"], 2))
        if key in seen:
            f.append(dict(kind="duplicate", entry_id=e["id"], confidence=0.9, detail=f"{e['id']} duplicates {seen[key]}"))
        seen[key] = e["id"]
        if e["category"] == "writeoff":
            f.append(dict(kind="policy_writeoff", confidence=1.0, detail=f"{e['id']} write-off ${e['amount']:,.2f}"))
        if e["vendor"] not in base.stats:
            f.append(dict(kind="novel_vendor", confidence=0.95, detail=f"{e['id']} unknown vendor {e['vendor']}"))
            continue
        mu, sd, cat = base.stats[e["vendor"]]
        z = (e["amount"] - mu) / sd
        if z > 3:
            f.append(dict(kind="outlier", confidence=min(0.99, 0.6 + z / 20), detail=f"{e['id']} z={z:.1f} vs vendor norm"))
        if e["category"] != cat and e["category"] != "writeoff":
            f.append(dict(kind="misclassified", entry_id=e["id"], to_category=cat, confidence=0.85, detail=f"{e['id']} {e['vendor']} booked as {e['category']}, expected {cat}"))
    return f

def reconcile(run):
    bank, used, f = run["bank"], set(), []
    for e in run["ledger"]:
        hit = None
        for i, b in enumerate(bank):
            if i in used or b["vendor"] != e["vendor"]:
                continue
            if abs(b["amount"] - e["amount"]) <= 0.05 and 0 <= b["day"] - e["day"] <= 5:
                hit = i; break
        if hit is None:
            f.append(dict(kind="missing_bank", confidence=0.9, detail=f"{e['id']} {e['vendor']} ${e['amount']:,.2f} has no bank match"))
        else:
            used.add(hit)
    return f

def _cat_totals(run):
    t = {}
    for e in run["ledger"]:
        t[e["category"]] = t.get(e["category"], 0) + e["amount"]
    return t

def variance(run, pct_threshold=0.35):
    f = []
    for c, cur in _cat_totals(run).items():
        hist = run["history"].get(c)
        if not hist:
            continue
        mu = st.mean(hist); pct = (cur - mu) / mu
        if abs(pct) > pct_threshold:
            f.append(dict(kind="variance", category=c, confidence=0.6, detail=f"{c} {pct:+.0%} vs 6-mo mean"))
    return f

def forecast(run, k=2.5):
    cats = list(run["history"])
    n = len(next(iter(run["history"].values())))
    totals = [sum(run["history"][c][i] for c in cats if len(run["history"][c]) > i) for i in range(n)]
    mu, sd = st.mean(totals), (st.pstdev(totals) or 1.0)
    cur = sum(e["amount"] for e in run["ledger"])
    if abs(cur - mu) > k * sd:
        return [dict(kind="forecast_oor", confidence=0.6, detail=f"total ${cur:,.0f} outside [{mu-k*sd:,.0f}, {mu+k*sd:,.0f}]")]
    return []
