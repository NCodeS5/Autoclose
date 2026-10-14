"""Deterministic facts about an invoice, computed in code (not by the LLM)."""
import json, os, pathlib
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

def run_checks(inv: dict, vendors: dict, prior: list) -> dict:
    v = vendors.get(inv["vendor"])
    amt, po = inv["amount"], inv.get("po_amount")
    dup = None
    for p in prior:
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
