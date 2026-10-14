"""Deterministic facts about an invoice, computed in code (not by the LLM)."""
import json, os, pathlib, re
from datetime import date

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"

def load_vendors() -> dict:
    return {v["vendor"]: v for v in json.loads((DATA / "vendor_master.json").read_text())}

def human_only_vendors(vendors=None) -> set:
    """Vendors only a person may approve or reject. Set HUMAN_ONLY_VENDORS="A, B" or add "human_only": true to a vendor record."""
    names = {x.strip().lower() for x in os.getenv("HUMAN_ONLY_VENDORS", "").split(",") if x.strip()}
    names |= {n.lower() for n, v in (vendors or {}).items() if v.get("human_only")}
    return names

def load_prior() -> list:
    return json.loads((DATA / "prior_invoices.json").read_text())

def _d(s): return date.fromisoformat(s)

CASE_MATCH_MIN = float(os.getenv("CASE_MATCH_MIN", "0.35"))

def case_base(case_id) -> str:
    """'C29-r2' and 'C29~chk1' are re-runs of case C29."""
    m = re.match(r"[^-~]+", str(case_id or "")); return m.group(0) if m else ""

def is_recode_case(c) -> bool:
    """Memory of an accepted category change. It says nothing about price variances."""
    return bool(re.search(r"coded as .+? recoded to .+? approved by reviewer", c.get("text", "")))

def learned_recode(cases, inv, ck, cited=None):
    """An earlier case where a person approved recoding THIS vendor from THIS category to its default category."""
    for c in cases or []:
        if c.get("outcome") != "approved" or c.get("score", 0) < CASE_MATCH_MIN: continue
        if cited is not None and c["id"] not in cited: continue
        m = re.search(r"coded as (.+?) recoded to (.+?) approved by reviewer", c.get("text", ""))
        if m and m.group(1) == inv["category"] and m.group(2) == ck.get("vendor_default_category") \
                and c.get("vendor", inv["vendor"]) == inv["vendor"]:
            return c
    return None

def run_checks(inv: dict, vendors: dict, prior: list, seen: list = None) -> dict:
    """`seen` = invoices already received in this session (each has seq and case). An invoice is only compared with ones
    received BEFORE it, from a different case, so the first copy is accepted and the later copy is the duplicate."""
    v = vendors.get(inv["vendor"])
    amt, po = inv["amount"], inv.get("po_amount")
    dup = None
    seq, base = inv.get("seq"), case_base(inv.get("id"))
    earlier = [p for p in (seen or []) if seq is not None and p["seq"] < seq and p["case"] != base]
    for p in list(prior) + earlier:
        if p["vendor"] != inv["vendor"]: continue
        if p["invoice_no"] == inv["invoice_no"] or (
                p["amount"] == amt and abs((_d(p["date"]) - _d(inv["date"])).days) <= 7):
            dup = p["invoice_no"]; break
    ytd = None
    if v:
        year = inv["date"][:4]
        ytd = round(sum(p["amount"] for p in prior if p["vendor"] == inv["vendor"] and p["date"][:4] == year) + amt, 2)
    var = None if po in (None, 0) else (amt - po) / po * 100
    return {
        "known_vendor": v is not None,
        "usage_based": bool(v and v["usage_based"]),
        "vendor_default_category": v["category"] if v else None,
        "category_mismatch": bool(v and inv["category"] != v["category"]),
        "duplicate_of": dup,
        "variance_pct": None if var is None else round(var, 2),
        "ytd_billed": ytd,
        "late_fee": "late fee" in inv["description"].lower(),
        "doc_type": inv.get("doc_type", "invoice"),
        "total": abs(amt),
        "human_only": inv["vendor"].lower() in human_only_vendors(vendors),
    }
