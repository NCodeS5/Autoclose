"""Synthetic month-end data with injected, LABELED anomalies (ground truth).

One "run" = one entity's month-end close: ledger entries + bank records.
"""
import random
from dataclasses import dataclass, field

VENDORS = {
    "AWS": ("cloud", 4200, 600), "Datadog": ("software", 1800, 200),
    "WeWork": ("rent", 6500, 150), "Staples": ("office", 340, 90),
    "Deloitte": ("professional", 9000, 1200), "Zoom": ("software", 520, 40),
    "Uber": ("travel", 210, 70), "Delta": ("travel", 780, 220),
    "Gusto": ("payroll", 15000, 900), "Stripe": ("fees", 1300, 180),
}
ANOMALY_TYPES = ["duplicate", "missing_bank", "misclassified", "outlier", "writeoff", "novel_vendor"]

@dataclass
class Run:
    run_id: int
    entity: str
    ledger: list = field(default_factory=list)
    bank: list = field(default_factory=list)
    history: dict = field(default_factory=dict)   # category -> trailing monthly totals
    anomalies: list = field(default_factory=list)  # ground truth labels
    benign_note: str = ""                          # e.g. seasonal swing that is NOT an error

    injected: list = field(default_factory=list)   # details, for scoring the agent's actions

    @property
    def edge_cases(self) -> set:
        """Which human edge cases this run SHOULD trigger. duplicate/misclassified are
        auto-fixable and expected to need no human."""
        m = {"writeoff": "E1", "novel_vendor": "E3", "missing_bank": "E2", "outlier": "E2"}
        return {m[a] for a in self.anomalies if a in m}

    @property
    def needs_human(self) -> bool:
        return bool(self.edge_cases)

def _base_run(rng, run_id):
    entity = f"ENT-{rng.randint(1, 12):02d}"
    r = Run(run_id, entity)
    tid = 0
    for v, (cat, mu, sd) in VENDORS.items():
        for _ in range(rng.randint(1, 3)):
            amt = round(max(20, rng.gauss(mu, sd)), 2)
            day = rng.randint(1, 28)
            tid += 1
            r.ledger.append(dict(id=f"L{tid}", vendor=v, category=cat, amount=amt, day=day,
                                 invoice=f"INV-{run_id}-{tid}"))
            # bank record matches; benign noise: +/- a few days, occasional cent rounding
            bamt = amt + (rng.choice([0, 0, 0, 0.01, -0.01]))
            r.bank.append(dict(vendor=v, amount=round(bamt, 2), day=min(31, day + rng.randint(0, 3))))
    totals = {}
    for e in r.ledger:
        totals[e["category"]] = totals.get(e["category"], 0) + e["amount"]
    # trailing history: 6 prior months near current level (non-anomalous baseline)
    r.history = {c: [round(t * rng.uniform(0.9, 1.1), 2) for _ in range(6)] for c, t in totals.items()}
    return r

def _inject(rng, r, kind):
    L = r.ledger
    if kind == "duplicate":
        e = dict(rng.choice(L)); e["id"] += "dup"; e["invoice"] = e["invoice"] + "-B"
        L.append(e); r.injected.append(dict(type="duplicate", entry=e["id"]))
    elif kind == "missing_bank":
        e = rng.choice(L); r.bank[:] = [b for b in r.bank if not (b["vendor"] == e["vendor"] and abs(b["amount"] - e["amount"]) < 0.02)]
    elif kind == "misclassified":
        e = rng.choice(L); orig = e["category"]
        e["category"] = rng.choice([c for c in {x["category"] for x in L} if c != e["category"]])
        r.injected.append(dict(type="misclassified", entry=e["id"], true_category=orig))
    elif kind == "outlier":
        e = rng.choice(L); factor = rng.uniform(1.6, 6); e["amount"] = round(e["amount"] * factor, 2)
        for b in r.bank:
            if b["vendor"] == e["vendor"] and abs(b["amount"] - e["amount"] / factor) < 0.02:
                b["amount"] = e["amount"]; break
    elif kind == "writeoff":
        L.append(dict(id="Lwo", vendor="Stripe", category="writeoff", amount=round(rng.uniform(800, 4000), 2),
                      day=rng.randint(25, 28), invoice=f"ADJ-{r.run_id}"))
        r.bank.append(dict(vendor="Stripe", amount=L[-1]["amount"], day=L[-1]["day"]))
    elif kind == "novel_vendor":
        amt = round(rng.uniform(2500, 9000), 2)
        L.append(dict(id="Lnv", vendor=f"NewCo-{rng.randint(100,999)}", category="other", amount=amt,
                      day=rng.randint(1, 28), invoice=f"INV-{r.run_id}-NV"))
        r.bank.append(dict(vendor=L[-1]["vendor"], amount=amt, day=L[-1]["day"]))
    r.anomalies.append(kind)

def generate(n_runs=300, anomaly_rate=0.30, seed=7):
    rng = random.Random(seed)
    runs = []
    for i in range(n_runs):
        r = _base_run(rng, i)
        if rng.random() < anomaly_rate:
            for kind in rng.sample(ANOMALY_TYPES, k=1 if rng.random() < 0.8 else 2):
                _inject(rng, r, kind)
        elif rng.random() < 0.20:
            # benign seasonal swing: variance tools will flag it, but it is NOT an error
            c = rng.choice(list(r.history))
            r.history[c] = [round(h * 0.6, 2) for h in r.history[c]]
            r.benign_note = f"seasonal_{c}"
        runs.append(r)
    return runs
